"""USB webcams through ``cv2.VideoCapture``, chosen by index or by name.

Indices move whenever a camera is plugged in, so a calibration should name its
camera. :func:`list_cameras` asks the OS for names: sysfs on Linux, where
``/dev/videoN`` is index ``N`` exactly, and ``system_profiler`` on macOS, where
the name order is *assumed* to match AVFoundation's index order. That macOS
assumption is unverified with three or more cameras attached; the opened
camera's resolution is checked against the request either way.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2

from .base import BaseSource, Control, Frame, SourceEnded, SourceInfo

# Settings worth exposing. Backends report -1 (or 0 on AVFoundation) for what
# they do not support; those are filtered out at open time.
_CONTROLS: Dict[str, Tuple[int, str]] = {
    "exposure": (cv2.CAP_PROP_EXPOSURE, "int"),
    "auto_exposure": (cv2.CAP_PROP_AUTO_EXPOSURE, "int"),
    "gain": (cv2.CAP_PROP_GAIN, "int"),
    "brightness": (cv2.CAP_PROP_BRIGHTNESS, "int"),
    "contrast": (cv2.CAP_PROP_CONTRAST, "int"),
    "focus": (cv2.CAP_PROP_FOCUS, "int"),
    "autofocus": (cv2.CAP_PROP_AUTOFOCUS, "bool"),
    "white_balance": (cv2.CAP_PROP_WB_TEMPERATURE, "int"),
    "auto_white_balance": (cv2.CAP_PROP_AUTO_WB, "bool"),
}


def list_cameras() -> List[Dict[str, Any]]:
    """Return ``[{"index", "name", "unique_id"}]`` for the attached cameras, best effort."""
    if sys.platform.startswith("linux"):
        out = []
        for node in sorted(Path("/sys/class/video4linux").glob("video*")):
            try:
                name = (node / "name").read_text().strip()
            except OSError:
                continue
            out.append({"index": int(node.name[5:]), "name": name, "unique_id": str(node)})
        return out
    if sys.platform == "darwin":
        try:
            raw = subprocess.run(
                ["system_profiler", "SPCameraDataType", "-json"],
                capture_output=True, text=True, timeout=10, check=True,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return []
        cams = json.loads(raw).get("SPCameraDataType", [])
        return [
            {"index": i, "name": c.get("_name", ""),
             "unique_id": c.get("spcamera_unique-id", ""),
             "model_id": c.get("spcamera_model-id", "")}
            for i, c in enumerate(cams)
        ]
    return []


def find_camera(name: str) -> Dict[str, Any]:
    """The camera whose name contains *name* (case-insensitive); exact matches win."""
    cams = list_cameras()
    exact = [c for c in cams if c["name"].lower() == name.lower()]
    partial = [c for c in cams if name.lower() in c["name"].lower()]
    hits = exact or partial
    if not hits:
        known = ", ".join(repr(c["name"]) for c in cams) or "none found"
        raise LookupError(f"no camera named {name!r} (cameras: {known})")
    if len(hits) > 1:
        raise LookupError(f"camera name {name!r} is ambiguous: {[c['name'] for c in hits]}")
    return hits[0]


class UVCSource(BaseSource):
    """``uvc:0``, ``uvc:name=USB Camera``; options ``size=WxH``, ``fps=N``, ``fourcc=MJPG``."""

    def __init__(self, index: Optional[int] = None, name: Optional[str] = None,
                 size: Tuple[int, int] = (0, 0), fps: float = 0.0,
                 fourcc: str = "", uri: str = "") -> None:
        if (index is None) == (name is None):
            raise ValueError("give exactly one of index or name")
        self.index, self.name, self.size, self.fps = index, name, size, fps
        self.fourcc = fourcc
        self.uri = uri
        self._cap: Optional[cv2.VideoCapture] = None
        self._seq = 0
        self._controls: Dict[str, Control] = {}

    def open(self) -> SourceInfo:
        cam: Dict[str, Any] = {}
        if self.name is not None:
            cam = find_camera(self.name)
            self.index = cam["index"]
        else:
            cam = next((c for c in list_cameras() if c["index"] == self.index), {})
        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            raise OSError(f"cannot open camera index {self.index}")
        if self.fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        if self.size[0]:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
        if self.fps:
            cap.set(cv2.CAP_PROP_FPS, self.fps)
        # The negotiated size is only certain once a frame arrives.
        ok, img = cap.read()
        if not ok:
            cap.release()
            raise OSError(f"camera index {self.index} opened but delivered no frame")
        self._cap = cap
        h, w = img.shape[:2]
        if self.size[0] and (w, h) != tuple(self.size):
            print(f"[uvc] asked for {self.size[0]}x{self.size[1]}, camera gave {w}x{h}",
                  file=sys.stderr)
        self._controls = self._probe_controls()
        fps = cap.get(cv2.CAP_PROP_FPS) or None
        return SourceInfo(
            name=cam.get("name") or f"uvc{self.index}",
            unique_id=cam.get("unique_id", ""),
            transport="uvc",
            resolution=(w, h),
            fps=fps,
            uri=self.uri,
            extra={"index": self.index, "backend": cap.getBackendName(),
                   **({"model_id": cam["model_id"]} if cam.get("model_id") else {})},
        )

    def read(self) -> Frame:
        if self._cap is None:
            raise RuntimeError("source is not open")
        ok, img = self._cap.read()
        if not ok:
            raise SourceEnded(f"camera index {self.index} stopped delivering frames")
        self._seq += 1
        return Frame(img=img, t_host=time.monotonic(), seq=self._seq)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _probe_controls(self) -> Dict[str, Control]:
        out = {}
        for name, (prop, kind) in _CONTROLS.items():
            v = self._cap.get(prop)
            if v == -1 or (v == 0 and self._cap.getBackendName() == "AVFOUNDATION"):
                continue
            out[name] = Control(name=name, kind=kind,
                                value=bool(v) if kind == "bool" else v)
        return out

    def controls(self) -> List[Control]:
        return list(self._controls.values())

    def set_control(self, name: str, value: Any) -> None:
        if name not in self._controls or self._cap is None:
            raise KeyError(f"camera has no control {name!r}")
        prop, _ = _CONTROLS[name]
        if not self._cap.set(prop, float(value)):
            raise ValueError(f"backend refused {name}={value}")
        self._controls[name].value = self._cap.get(prop)
