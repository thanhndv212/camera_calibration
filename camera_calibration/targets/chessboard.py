"""Plain chessboard, kept for the existing calibrations in ``data/``."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from .base import Detection, Target, downscale, refine


class ChessboardTarget(Target):
    """``inner`` = inner corners (cols, rows). The whole board must be in view."""

    kind = "chessboard"
    quiet_squares = 1.0  # findChessboardCorners needs a white border

    def __init__(self, inner: Tuple[int, int] = (9, 6), square_m: float = 0.025) -> None:
        self.inner, self.square_m = (int(inner[0]), int(inner[1])), float(square_m)

    def spec(self) -> Dict[str, Any]:
        return {"type": "chessboard", "inner_corners": list(self.inner), "square_m": self.square_m}

    def spec_string(self) -> str:
        return f"chessboard?inner={self.inner[0]}x{self.inner[1]}&square={self.square_m:g}"

    @property
    def min_points(self) -> int:
        return self.inner[0] * self.inner[1]

    def all_object_points(self) -> np.ndarray:
        cols, rows = self.inner
        grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * self.square_m
        return np.hstack([grid, np.zeros((len(grid), 1))]).astype(np.float32)

    def detect(self, gray: np.ndarray, search_width: int = 640) -> Optional[Detection]:
        small, s = downscale(gray, search_width)
        flags = (cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
                 | cv2.CALIB_CB_FAST_CHECK)
        ok, corners = cv2.findChessboardCorners(small, self.inner, flags)
        if not ok:
            return None
        pts = corners.reshape(-1, 2) / s
        spacing = float(np.median(np.linalg.norm(np.diff(pts[:self.inner[0]], axis=0), axis=1)))
        pts = refine(gray, pts, spacing / 3)
        ids = np.arange(len(pts))
        return Detection(pts, self.all_object_points(), ids)

    def texture(self, px_per_square: int) -> Tuple[np.ndarray, float]:
        cols, rows = self.inner
        P = px_per_square
        # One white square of margin; inner corner (0, 0) sits on the edge at 2P.
        tex = np.full(((rows + 3) * P, (cols + 3) * P), 255, np.uint8)
        for j in range(rows + 1):
            for i in range(cols + 1):
                if (i + j) % 2 == 0:
                    tex[(j + 1) * P:(j + 2) * P, (i + 1) * P:(i + 2) * P] = 0
        return tex, 2 * P

    def draw(self, img, det, scale=1.0, color=(0, 220, 0)) -> None:
        cv2.drawChessboardCorners(img, self.inner, (det.img_pts * scale).astype(np.float32)
                                  .reshape(-1, 1, 2), True)
