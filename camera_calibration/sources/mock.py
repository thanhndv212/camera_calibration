"""A synthetic camera with known intrinsics, for tests and GUI work without hardware.

Each frame is a true pinhole view of a planar chessboard on a smooth pose
trajectory, so a calibration run against it can be checked against
:attr:`MockSource.camera_matrix` and :meth:`MockSource.pose`.
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import cv2
import numpy as np

from .base import BaseSource, Frame, SourceEnded, SourceInfo

_PX_PER_SQUARE = 40


class MockSource(BaseSource):
    """``mock:`` — options ``size=640x480``, ``fps=30`` (0 = unpaced), ``frames=N``
    (0 = endless), ``target=chessboard`` or ``charuco``, ``board=9x6`` (chessboard
    inner corners, or ChArUco squares, default 7x5), ``square=0.025`` metres,
    ``k1=0`` radial distortion, ``noise=2`` pixel noise sigma, ``seed=0``.
    """

    def __init__(self, size: Tuple[int, int] = (640, 480), fps: float = 30.0, frames: int = 0,
                 board: Optional[Tuple[int, int]] = None, square: float = 0.025,
                 k1: float = 0.0, noise: float = 2.0, seed: int = 0, target: str = "chessboard",
                 uri: str = "") -> None:
        from ..targets import CharucoTarget, ChessboardTarget

        self.size, self.fps, self.frames = size, fps, frames
        self.square, self.noise = square, noise
        if target == "charuco":
            self.target = CharucoTarget(board or (7, 5), square, square * 0.73)
        elif target == "chessboard":
            self.target = ChessboardTarget(board or (9, 6), square)
        else:
            raise ValueError(f"mock target must be chessboard or charuco, got {target!r}")
        self.board = board or (self.target.squares if target == "charuco" else self.target.inner)
        self.uri = uri
        w, h = size
        f = 0.9 * w
        self.camera_matrix = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float64)
        self.dist_coeffs = np.array([k1, 0, 0, 0, 0], np.float64)
        self._rng = np.random.default_rng(seed)
        self._seq = 0
        self._next_t = 0.0
        self._texture: Optional[np.ndarray] = None
        self._maps: Optional[Tuple[np.ndarray, np.ndarray]] = None

    # -- ground truth ---------------------------------------------------------

    def object_points(self) -> np.ndarray:
        """Every target corner in the board frame, metres, indexed by corner id."""
        return self.target.all_object_points()

    def pose(self, seq: int) -> Tuple[np.ndarray, np.ndarray]:
        """``(rvec, tvec)`` of T_camera_board for frame *seq*: board frame -> camera frame."""
        s = seq * 0.05
        rvec = np.array([0.35 * np.sin(s), 0.45 * np.sin(0.7 * s + 1.0), 0.2 * np.sin(0.3 * s)])
        R, _ = cv2.Rodrigues(rvec)
        obj = self.object_points()
        center_b = (obj.min(axis=0) + obj.max(axis=0)) / 2
        center_c = np.array([0.04 * np.sin(0.9 * s), 0.03 * np.cos(1.1 * s),
                             0.42 + 0.08 * np.sin(0.5 * s)])
        return rvec, center_c - R @ center_b

    # -- source protocol ------------------------------------------------------

    def open(self) -> SourceInfo:
        self._texture, self._origin_px = self.target.texture(_PX_PER_SQUARE)
        if self.dist_coeffs.any():
            # For each distorted output pixel, where it lands in the ideal image.
            w, h = self.size
            u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            pts = np.stack([u.ravel(), v.ravel()], 1).reshape(-1, 1, 2)
            ideal = cv2.undistortPoints(pts, self.camera_matrix, self.dist_coeffs,
                                        P=self.camera_matrix).reshape(h, w, 2)
            self._maps = (ideal[..., 0].copy(), ideal[..., 1].copy())
        return SourceInfo(name="mock", unique_id="mock", transport="mock",
                          resolution=self.size, fps=self.fps or None, uri=self.uri,
                          extra={"target": self.target.spec()})

    def render(self, seq: int) -> np.ndarray:
        """The BGR image for frame *seq* (deterministic apart from the noise)."""
        if self._texture is None:
            raise RuntimeError("source is not open")
        P, s = _PX_PER_SQUARE, self.square
        # Texture pixel index -> board metres (see Target.texture).
        off = (0.5 - self._origin_px) * s / P
        A = np.array([[s / P, 0, off], [0, s / P, off], [0, 0, 1]])
        rvec, tvec = self.pose(seq)
        R, _ = cv2.Rodrigues(rvec)
        H = self.camera_matrix @ np.column_stack([R[:, 0], R[:, 1], tvec]) @ A
        img = cv2.warpPerspective(self._texture, H, self.size, flags=cv2.INTER_AREA,
                                  borderValue=110)
        if self._maps is not None:
            img = cv2.remap(img, *self._maps, cv2.INTER_LINEAR, borderValue=110)
        if self.noise:
            img = np.clip(img + self._rng.normal(0, self.noise, img.shape), 0, 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    def read(self) -> Frame:
        if self.frames and self._seq >= self.frames:
            raise SourceEnded("mock frame budget used up")
        if self.fps:
            now = time.monotonic()
            if self._next_t > now:
                time.sleep(self._next_t - now)
            self._next_t = max(now, self._next_t) + 1.0 / self.fps
        self._seq += 1
        return Frame(img=self.render(self._seq), t_host=time.monotonic(), seq=self._seq)
