"""HTTP MJPEG streams, and the ESP32 CameraWebServer which adds controls.

Frames are split on JPEG SOI/EOI markers rather than multipart headers, which
is robust to the slightly non-standard boundaries some firmwares emit (ported
from ``tools/esp32_camera_probe.py``).
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from typing import Any, Iterator, List, Optional, Tuple

import cv2
import numpy as np

from .base import BaseSource, Control, Frame, SourceEnded, SourceInfo


def iter_mjpeg(url: str, timeout: float) -> Iterator[Tuple[bytes, float]]:
    """Yield ``(jpeg_bytes, host_receive_monotonic)`` from an MJPEG stream."""
    with urllib.request.urlopen(url, timeout=timeout) as r:
        buf = b""
        while True:
            # read1, not read: read(n) waits for n bytes, which holds each frame's
            # tail back until the next frame starts and skews every timestamp.
            chunk = r.read1(65536)
            if not chunk:
                return
            buf += chunk
            while True:
                start = buf.find(b"\xff\xd8")
                end = buf.find(b"\xff\xd9", start + 2) if start >= 0 else -1
                if start < 0 or end < 0:
                    if start > 0:
                        buf = buf[start:]
                    break
                yield buf[start:end + 2], time.monotonic()
                buf = buf[end + 2:]


class MJPEGSource(BaseSource):
    """``mjpeg:http://host:81/stream``."""

    transport = "mjpeg"

    def __init__(self, url: str, timeout: float = 5.0, uri: str = "") -> None:
        self.url, self.timeout, self.uri = url, timeout, uri
        self._it: Optional[Iterator[Tuple[bytes, float]]] = None
        self._seq = 0
        self._first: Optional[Frame] = None

    def _info(self, w: int, h: int) -> SourceInfo:
        host = urllib.parse.urlsplit(self.url).netloc
        return SourceInfo(name=host, unique_id=host, transport=self.transport,
                          resolution=(w, h), uri=self.uri)

    def open(self) -> SourceInfo:
        self._it = iter_mjpeg(self.url, self.timeout)
        self._first = self.read()
        h, w = self._first.img.shape[:2]
        return self._info(w, h)

    def read(self) -> Frame:
        if self._first is not None:
            f, self._first = self._first, None
            return f
        if self._it is None:
            raise RuntimeError("source is not open")
        for jpeg, t in self._it:
            img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            self._seq += 1
            return Frame(img=img, t_host=t, seq=self._seq, jpeg=jpeg)
        raise SourceEnded(f"stream {self.url} closed")

    def close(self) -> None:
        if self._it is not None:
            self._it.close()
            self._it = None


# CameraWebServer /control variables worth a GUI widget: (kind, min, max).
_ESP32_CONTROLS = {
    "framesize": ("int", 0, 13),
    "quality": ("int", 4, 63),
    "brightness": ("int", -2, 2),
    "contrast": ("int", -2, 2),
    "saturation": ("int", -2, 2),
    "aec": ("bool", None, None),
    "aec_value": ("int", 0, 1200),
    "ae_level": ("int", -2, 2),
    "agc": ("bool", None, None),
    "agc_gain": ("int", 0, 30),
    "awb": ("bool", None, None),
    "hmirror": ("bool", None, None),
    "vflip": ("bool", None, None),
}


class ESP32Source(MJPEGSource):
    """``esp32:192.168.1.142`` — CameraWebServer: control port 80, stream on port + 1."""

    transport = "esp32"

    def __init__(self, host: str, timeout: float = 5.0, uri: str = "") -> None:
        name, _, port = host.partition(":")
        super().__init__(f"http://{name}:{int(port or 80) + 1}/stream", timeout, uri)
        self.host = host
        self._status: dict = {}

    def _get(self, path: str) -> bytes:
        with urllib.request.urlopen(f"http://{self.host}{path}", timeout=self.timeout) as r:
            return r.read()

    def open(self) -> SourceInfo:
        try:
            self._status = json.loads(self._get("/status"))
        except (OSError, ValueError):
            self._status = {}
        return super().open()

    def _info(self, w: int, h: int) -> SourceInfo:
        info = super()._info(w, h)
        info.name, info.unique_id = f"esp32@{self.host}", self.host
        if self._status:
            info.extra = {"status": self._status}
        return info

    def controls(self) -> List[Control]:
        out = []
        for name, (kind, lo, hi) in _ESP32_CONTROLS.items():
            if name not in self._status:
                continue
            v = self._status[name]
            out.append(Control(name=name, kind=kind, value=bool(v) if kind == "bool" else v,
                               min=lo, max=hi))
        return out

    def set_control(self, name: str, value: Any) -> None:
        if name not in _ESP32_CONTROLS:
            raise KeyError(f"esp32 has no control {name!r}")
        self._get(f"/control?var={name}&val={int(value)}")
        self._status[name] = int(value)
