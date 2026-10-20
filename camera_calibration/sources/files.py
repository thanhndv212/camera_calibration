"""Offline sources: a directory of images and a video file.

These let every procedure run without a camera — in CI, and when re-solving a
calibration from the images a session saved.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

import cv2

from .base import BaseSource, Frame, SourceEnded, SourceInfo

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class DirSource(BaseSource):
    """``dir:runs/calib01/`` — images in name order; ``loop=1`` restarts at the end.

    ``fps`` paces delivery (0 = as fast as they decode), so a directory can
    stand in for a live camera in the capture loop.
    """

    def __init__(self, path: str, loop: bool = False, fps: float = 0.0, uri: str = "") -> None:
        self.path, self.loop, self.fps, self.uri = Path(path), loop, fps, uri
        self._files: List[Path] = []
        self._i = 0
        self._seq = 0
        self._next_t = 0.0

    def open(self) -> SourceInfo:
        if not self.path.is_dir():
            raise FileNotFoundError(f"not a directory: {self.path}")
        self._files = sorted(p for p in self.path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if not self._files:
            raise FileNotFoundError(f"no images in {self.path}")
        first = cv2.imread(str(self._files[0]))
        if first is None:
            raise OSError(f"cannot decode {self._files[0]}")
        h, w = first.shape[:2]
        return SourceInfo(name=self.path.name, unique_id=str(self.path.resolve()),
                          transport="dir", resolution=(w, h), fps=self.fps or None,
                          uri=self.uri, extra={"count": len(self._files)})

    def read(self) -> Frame:
        if self._i >= len(self._files):
            if not self.loop or not self._files:
                raise SourceEnded(f"end of {self.path}")
            self._i = 0
        if self.fps:
            now = time.monotonic()
            if self._next_t > now:
                time.sleep(self._next_t - now)
            self._next_t = max(now, self._next_t) + 1.0 / self.fps
        p = self._files[self._i]
        self._i += 1
        img = cv2.imread(str(p))
        if img is None:
            raise OSError(f"cannot decode {p}")
        self._seq += 1
        return Frame(img=img, t_host=time.monotonic(), seq=self._seq, path=str(p))


class VideoSource(BaseSource):
    """``video:clip.mp4``; ``loop=1`` restarts at the end. Delivered as fast as it decodes."""

    def __init__(self, path: str, loop: bool = False, uri: str = "") -> None:
        self.path, self.loop, self.uri = Path(path), loop, uri
        self._cap: Optional[cv2.VideoCapture] = None
        self._seq = 0

    def open(self) -> SourceInfo:
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise OSError(f"cannot open video {self.path}")
        self._cap = cap
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return SourceInfo(name=self.path.name, unique_id=str(self.path.resolve()),
                          transport="video", resolution=(w, h),
                          fps=cap.get(cv2.CAP_PROP_FPS) or None, uri=self.uri)

    def read(self) -> Frame:
        if self._cap is None:
            raise RuntimeError("source is not open")
        ok, img = self._cap.read()
        if not ok and self.loop:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, img = self._cap.read()
        if not ok:
            raise SourceEnded(f"end of {self.path}")
        self._seq += 1
        return Frame(img=img, t_host=time.monotonic(), seq=self._seq)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
