"""Three-stage capture loop: capture never waits for analysis or the preview.

1. **Capture thread** — the only reader of the source. Reads as fast as the
   camera delivers, timestamps on receipt, and keeps just the latest frame.
2. **Analysis worker** — when free, takes the newest frame and runs the
   ``analyze`` callable. A slow detector lowers the analysis rate only; frames
   it never saw are counted as ``skipped``, not lost from the camera.
3. **Preview** — :meth:`CaptureLoop.preview_jpeg` encodes the latest frame on
   demand, downscaled to ``preview_max`` px wide, with an optional overlay.

Saved samples should come from :meth:`CaptureLoop.latest` (full resolution,
never re-encoded), not from the preview.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, Optional, Tuple

import cv2
import numpy as np

from .sources.base import BaseSource, Frame, SourceEnded, SourceInfo

Analyze = Callable[[Frame], Any]
Overlay = Callable[[np.ndarray, float, Optional[Frame], Any], None]


class _Rate:
    """Events per second over a sliding window."""

    def __init__(self, window_s: float = 2.0) -> None:
        self.window_s = window_s
        self._t: Deque[float] = deque()

    def tick(self, t: float) -> None:
        self._t.append(t)
        while self._t and t - self._t[0] > self.window_s:
            self._t.popleft()

    def rate(self, now: Optional[float] = None) -> float:
        if len(self._t) < 2:
            return 0.0
        now = time.monotonic() if now is None else now
        if now - self._t[-1] > self.window_s:  # stalled: report 0, not a stale rate
            return 0.0
        return (len(self._t) - 1) / max(self._t[-1] - self._t[0], 1e-9)


class CaptureLoop:
    """Own a source on a thread and serve its newest frame and analysis result.

    ``analyze(frame) -> result`` runs on the worker thread. ``overlay(img,
    scale, frame, result)`` draws on the downscaled preview in place; ``scale``
    maps full-resolution pixel coordinates to preview ones.
    """

    def __init__(self, source: BaseSource, analyze: Optional[Analyze] = None,
                 overlay: Optional[Overlay] = None, preview_max: int = 960,
                 preview_quality: int = 80) -> None:
        self.source = source
        self.analyze = analyze
        self.overlay = overlay
        self.preview_max = preview_max
        self.preview_quality = preview_quality
        self.info: Optional[SourceInfo] = None

        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._frame: Optional[Frame] = None
        self._result: Optional[Tuple[Frame, Any]] = None
        self._threads: list = []
        self._ended = False
        self._error: Optional[str] = None
        self._cam_rate, self._ana_rate = _Rate(), _Rate()
        self._frames = 0
        self._source_gaps = 0
        self._skipped = 0
        self._analysed = 0
        self._last_seq = 0
        self._analysis_ms: Deque[float] = deque(maxlen=30)
        self._preview: Tuple[int, int, Optional[bytes]] = (-1, -1, None)  # (seq, result id, jpeg)

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> SourceInfo:
        self.info = self.source.open()
        self._stop.clear()
        self._threads = [threading.Thread(target=self._capture, name="capture", daemon=True)]
        if self.analyze is not None:
            self._threads.append(threading.Thread(target=self._worker, name="analysis",
                                                  daemon=True))
        for t in self._threads:
            t.start()
        return self.info

    def stop(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        for t in self._threads:
            t.join(timeout=3)
        self._threads = []
        self.source.close()

    def __enter__(self) -> "CaptureLoop":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    @property
    def ended(self) -> bool:
        """True once a finite source (``dir:``, ``video:``, ``mock:?frames=N``) ran out."""
        return self._ended

    # -- stage 1: capture -----------------------------------------------------

    def _capture(self) -> None:
        backoff = 0.05
        while not self._stop.is_set():
            try:
                frame = self.source.read()
            except SourceEnded:
                self._ended = True
                break
            except Exception as exc:  # a flaky camera must not kill the loop
                self._error = f"{type(exc).__name__}: {exc}"
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 2.0)
                continue
            backoff = 0.05
            with self._cond:
                if self._last_seq and frame.seq > self._last_seq + 1:
                    self._source_gaps += frame.seq - self._last_seq - 1
                self._last_seq = frame.seq
                self._frame = frame
                self._frames += 1
                self._cam_rate.tick(frame.t_host)
                self._cond.notify_all()
        with self._cond:
            self._cond.notify_all()

    # -- stage 2: analysis ----------------------------------------------------

    def _worker(self) -> None:
        done_seq = 0
        while not self._stop.is_set():
            with self._cond:
                while not self._stop.is_set() and (
                        self._frame is None or self._frame.seq == done_seq):
                    if self._ended:
                        return
                    self._cond.wait(0.5)
                if self._stop.is_set():
                    return
                frame = self._frame
            if done_seq and frame.seq > done_seq + 1:
                self._skipped += frame.seq - done_seq - 1
            done_seq = frame.seq
            t0 = time.perf_counter()
            try:
                result = self.analyze(frame)
            except Exception as exc:
                self._error = f"analyze: {type(exc).__name__}: {exc}"
                result = None
            self._analysis_ms.append((time.perf_counter() - t0) * 1000)
            with self._cond:
                self._result = (frame, result)
                self._analysed += 1
                self._ana_rate.tick(time.monotonic())
                self._cond.notify_all()

    # -- readers --------------------------------------------------------------

    def latest(self) -> Optional[Frame]:
        """The newest full-resolution frame (shared; copy ``img`` before editing it)."""
        with self._cond:
            return self._frame

    def latest_result(self) -> Optional[Tuple[Frame, Any]]:
        """``(frame, result)`` from the most recent analysis."""
        with self._cond:
            return self._result

    def wait_frame(self, after_seq: int = 0, timeout: float = 5.0) -> Optional[Frame]:
        """Block until a frame newer than *after_seq* arrives (or the source ends)."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._frame is None or self._frame.seq <= after_seq:
                left = deadline - time.monotonic()
                if left <= 0 or self._ended or self._stop.is_set():
                    return None
                self._cond.wait(left)
            return self._frame

    def wait_result(self, after_seq: int = 0, timeout: float = 5.0) -> Optional[Tuple[Frame, Any]]:
        """Block until an analysis of a frame newer than *after_seq* is available."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._result is None or self._result[0].seq <= after_seq:
                left = deadline - time.monotonic()
                if left <= 0 or self._stop.is_set():
                    return None
                if self._ended and (self._frame is None or self._result is not None
                                    and self._result[0].seq == self._frame.seq):
                    return None
                self._cond.wait(min(left, 0.5))
            return self._result

    # -- stage 3: preview -----------------------------------------------------

    def preview_jpeg(self) -> Optional[bytes]:
        """JPEG of the latest frame, at most ``preview_max`` px wide, overlay drawn.

        Cached per (frame, analysis result), so many viewers cost one encode.
        """
        with self._cond:
            frame, res = self._frame, self._result
        if frame is None:
            return None
        res_id = id(res)
        if self._preview[0] == frame.seq and self._preview[1] == res_id:
            return self._preview[2]
        h, w = frame.img.shape[:2]
        scale = min(1.0, self.preview_max / w) if self.preview_max else 1.0
        if scale == 1.0 and frame.jpeg is not None and self.overlay is None:
            jpeg = frame.jpeg
        else:
            img = (cv2.resize(frame.img, (round(w * scale), round(h * scale)),
                              interpolation=cv2.INTER_AREA)
                   if scale < 1.0 else frame.img.copy())
            if self.overlay is not None:
                self.overlay(img, scale, *(res if res else (None, None)))
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.preview_quality])
            jpeg = buf.tobytes() if ok else None
        self._preview = (frame.seq, res_id, jpeg)
        return jpeg

    def stats(self) -> Dict[str, Any]:
        """Camera and analysis rates, reported separately, plus loss counters.

        ``source_gaps`` are frames the source itself skipped (its ``seq`` jumped);
        ``analysis_skipped`` are frames the camera delivered but the analysis
        worker never got to, which is expected whenever analysis is slower.
        """
        now = time.monotonic()
        with self._cond:
            frame = self._frame
            ms = list(self._analysis_ms)
            return {
                "camera_fps": round(self._cam_rate.rate(now), 2),
                "analysis_fps": round(self._ana_rate.rate(now), 2),
                "analysis_ms": round(float(np.median(ms)), 1) if ms else None,
                "frames": self._frames,
                "analysed": self._analysed,
                "analysis_skipped": self._skipped,
                "source_gaps": self._source_gaps,
                "frame_age_ms": round((now - frame.t_host) * 1000, 1) if frame else None,
                "resolution": list(frame.img.shape[1::-1]) if frame else None,
                "ended": self._ended,
                "error": self._error,
            }


def find_chessboard(gray: np.ndarray, pattern: Tuple[int, int],
                    search_width: int = 640) -> Optional[np.ndarray]:
    """Chessboard corners at full resolution, found on a downscaled copy.

    The search (the slow part) runs at ``search_width`` px; ``cornerSubPix``
    then refines at full resolution, so accuracy does not depend on the
    downscale. Returns ``(N, 1, 2)`` float32 corners or ``None``.
    """
    h, w = gray.shape[:2]
    scale = min(1.0, search_width / w)
    small = cv2.resize(gray, (round(w * scale), round(h * scale)),
                       interpolation=cv2.INTER_AREA) if scale < 1.0 else gray
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    ok, corners = cv2.findChessboardCorners(small, pattern, flags)
    if not ok:
        return None
    corners = (corners / scale).astype(np.float32)
    # Window sized to the board's square spacing at full resolution, capped.
    spacing = float(np.median(np.linalg.norm(np.diff(corners[:pattern[0], 0], axis=0), axis=1)))
    win = int(np.clip(spacing / 4, 3, 11))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    cv2.cornerSubPix(gray, corners, (win, win), (-1, -1), criteria)
    return corners
