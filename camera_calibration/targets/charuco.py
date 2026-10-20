"""ChArUco: a chessboard with ArUco markers in the white squares.

The default for new calibrations: every corner is identified, so partial
views count, and the board can reach the image corners where distortion is
largest. Corners are numbered row-major over the inner grid, origin at the
board's top-left outer corner (OpenCV's ``getChessboardCorners``).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from .base import Detection, Target, downscale, refine


class CharucoTarget(Target):
    """``squares`` = whole squares (cols, rows); sizes in metres."""

    kind = "charuco"

    def __init__(self, squares: Tuple[int, int] = (7, 5), square_m: float = 0.03,
                 marker_m: float = 0.022, dictionary: str = "DICT_4X4_50") -> None:
        if marker_m >= square_m:
            raise ValueError("marker must be smaller than the square")
        self.squares = (int(squares[0]), int(squares[1]))
        self.square_m, self.marker_m, self.dictionary = float(square_m), float(marker_m), dictionary
        self._dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
        self.board = cv2.aruco.CharucoBoard(self.squares, self.square_m, self.marker_m, self._dict)
        self._detector = cv2.aruco.CharucoDetector(self.board)
        self._corners = self.board.getChessboardCorners().astype(np.float32)

    def spec(self) -> Dict[str, Any]:
        return {"type": "charuco", "squares": list(self.squares), "square_m": self.square_m,
                "marker_m": self.marker_m, "dictionary": self.dictionary,
                "legacy_pattern": bool(self.board.getLegacyPattern())}

    def spec_string(self) -> str:
        return (f"charuco?squares={self.squares[0]}x{self.squares[1]}&square={self.square_m:g}"
                f"&marker={self.marker_m:g}&dict={self.dictionary}")

    def all_object_points(self) -> np.ndarray:
        return self._corners

    @property
    def min_points(self) -> int:
        return 6

    def detect(self, gray: np.ndarray, search_width: int = 960) -> Optional[Detection]:
        # Markers are found and corners interpolated on a downscaled copy; the
        # corners are then refined against the full-resolution image.
        small, s = downscale(gray, search_width)
        cc, ci, mc, mi = self._detector.detectBoard(small)
        if ci is None or len(ci) < self.min_points:
            return None
        ids = ci.ravel().astype(np.int32)
        obj = self._corners[ids]
        # Interpolated corners on a planar board must not be all on one line.
        if np.linalg.matrix_rank(obj[:, :2] - obj[:1, :2], tol=1e-6) < 2:
            return None
        pts = cc.reshape(-1, 2) / s
        # Square size in pixels, from neighbouring ids, sizes the refinement window.
        d_obj = np.linalg.norm(np.diff(obj[:, :2], axis=0), axis=1)
        d_img = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        step = d_obj > 0
        spacing = float(np.median(d_img[step] / d_obj[step])) * self.square_m if step.any() else 20.0
        # Clear zone: the white border between a corner and the nearest marker.
        pts = refine(gray, pts, spacing * (1 - self.marker_m / self.square_m) / 2)
        markers = ([m / s for m in mc], mi) if mi is not None else None
        return Detection(pts, obj, ids, markers)

    def texture(self, px_per_square: int) -> Tuple[np.ndarray, float]:
        P = px_per_square
        cols, rows = self.squares
        img = self.board.generateImage((cols * P, rows * P), marginSize=0, borderBits=1)
        img = cv2.copyMakeBorder(img, P, P, P, P, cv2.BORDER_CONSTANT, value=255)
        return img, P

    def draw(self, img, det, scale=1.0, color=(0, 220, 0)) -> None:
        if det.markers is not None:
            corners, ids = det.markers
            cv2.aruco.drawDetectedMarkers(img, [c * scale for c in corners], ids, (255, 160, 0))
        super().draw(img, det, scale, color)
