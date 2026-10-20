"""HTTP MJPEG streams, and the ESP32 CameraWebServer which adds controls.

Frames are split on JPEG SOI/EOI markers rather than multipart headers, which
is robust to the slightly non-standard boundaries some firmwares emit (as the
ESP32 CameraWebServer does).
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


_FRAMESIZES = [[k, f"{k} \u00b7 {v}"] for k, v in {
    0: "96X96", 1: "QQVGA", 2: "128X128", 3: "QCIF", 4: "HQVGA", 5: "240X240",
    6: "QVGA", 7: "320X320", 8: "CIF", 9: "HVGA", 10: "VGA", 11: "SVGA",
    12: "XGA", 13: "HD", 14: "SXGA", 15: "UXGA"}.items()]
_GAINCEILING = [[k, f"{2 ** (k + 1)}x"] for k in range(7)]
_EFFECTS = [[k, n] for k, n in enumerate(
    ["None", "Negative", "Grayscale", "Red tint", "Green tint", "Blue tint", "Sepia"])]
_WB_MODES = [[k, n] for k, n in enumerate(["Auto", "Sunny", "Cloudy", "Office", "Home"])]

# CameraWebServer /control variables worth a GUI widget, in display order:
# name -> (group, label, kind, min, max, choices).
_ESP32_CONTROLS = {
    "framesize": ("Image", "Frame size", "choice", None, None, _FRAMESIZES),
    "quality": ("Image", "JPEG quality (lower = better)", "int", 4, 63, None),
    "brightness": ("Image", "Brightness", "int", -2, 2, None),
    "contrast": ("Image", "Contrast", "int", -2, 2, None),
    "saturation": ("Image", "Saturation", "int", -2, 2, None),
    "special_effect": ("Image", "Effect", "choice", None, None, _EFFECTS),
    "aec": ("Exposure & gain", "Auto exposure", "bool", None, None, None),
    "aec2": ("Exposure & gain", "AEC DSP", "bool", None, None, None),
    "ae_level": ("Exposure & gain", "AE level", "int", -2, 2, None),
    "aec_value": ("Exposure & gain", "Exposure", "int", 0, 1200, None),
    "agc": ("Exposure & gain", "Auto gain", "bool", None, None, None),
    "agc_gain": ("Exposure & gain", "Gain", "int", 0, 30, None),
    "gainceiling": ("Exposure & gain", "Gain ceiling", "choice", None, None, _GAINCEILING),
    "awb": ("White balance", "Auto white balance", "bool", None, None, None),
    "awb_gain": ("White balance", "AWB gain", "bool", None, None, None),
    "wb_mode": ("White balance", "WB mode", "choice", None, None, _WB_MODES),
    "lenc": ("Corrections & orientation", "Lens correction", "bool", None, None, None),
    "raw_gma": ("Corrections & orientation", "Raw gamma", "bool", None, None, None),
    "bpc": ("Corrections & orientation", "Black pixel correction", "bool", None, None, None),
    "wpc": ("Corrections & orientation", "White pixel correction", "bool", None, None, None),
    "dcw": ("Corrections & orientation", "Downsize (DCW)", "bool", None, None, None),
    "hmirror": ("Corrections & orientation", "Mirror", "bool", None, None, None),
    "vflip": ("Corrections & orientation", "Flip", "bool", None, None, None),
    "colorbar": ("Corrections & orientation", "Colour bar test", "bool", None, None, None),
    "led_intensity": ("Corrections & orientation", "LED", "int", 0, 255, None),
}


class ESP32Source(MJPEGSource):
    """``esp32:192.168.1.142`` — CameraWebServer: control port 80, stream on port + 1.

    Only controls the firmware reports in ``/status`` are offered; ``/status`` is
    re-read at most every 2 s, so changes made from elsewhere show up.
    """

    transport = "esp32"

    def __init__(self, host: str, timeout: float = 5.0, uri: str = "",
                 stream: Optional[str] = None) -> None:
        name, _, port = host.partition(":")
        super().__init__(stream or f"http://{name}:{int(port or 80) + 1}/stream", timeout, uri)
        self.host = host
        self._status: dict = {}
        self._status_t = 0.0

    def _get(self, path: str) -> bytes:
        with urllib.request.urlopen(f"http://{self.host}{path}", timeout=self.timeout) as r:
            return r.read()

    def status(self, max_age: float = 2.0) -> dict:
        """CameraWebServer ``/status``, cached for ``max_age`` seconds."""
        if time.monotonic() - self._status_t > max_age:
            try:
                self._status = json.loads(self._get("/status"))
            except (OSError, ValueError):
                pass  # keep the last good copy
            self._status_t = time.monotonic()
        return self._status

    def open(self) -> SourceInfo:
        self.status(max_age=0)
        return super().open()

    def _info(self, w: int, h: int) -> SourceInfo:
        info = super()._info(w, h)
        info.name, info.unique_id = f"esp32@{self.host}", self.host
        if self._status:
            info.extra = {"status": self._status}
        return info

    def controls(self) -> List[Control]:
        status = self.status()
        out = []
        for name, (group, label, kind, lo, hi, choices) in _ESP32_CONTROLS.items():
            if name not in status:
                continue
            v = status[name]
            out.append(Control(name=name, kind=kind, value=bool(v) if kind == "bool" else v,
                               min=lo, max=hi, choices=choices, label=label, group=group))
        return out

    def set_control(self, name: str, value: Any) -> None:
        if name not in _ESP32_CONTROLS:
            raise KeyError(f"esp32 has no control {name!r}")
        self._get(f"/control?var={name}&val={int(value)}")
        self._status[name] = int(value)
