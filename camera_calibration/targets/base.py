"""Shared target types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


@dataclass
class Detection:
    """Matched points: ``img_pts`` (N, 2) pixels, ``obj_pts`` (N, 3) board metres.

    ``ids`` index the board's corners, so two detections of a partial board
    can be compared point for point.
    """

    img_pts: np.ndarray
    obj_pts: np.ndarray
    ids: np.ndarray
    markers: Optional[Tuple[Any, Any]] = None  # (corners, ids) of ArUco markers, for drawing

    def __len__(self) -> int:
        return len(self.ids)


class Target:
    """A planar board. Subclasses fill in detection and rendering."""

    kind = ""
    square_m: float = 0.0
    quiet_squares: float = 0.0  # white border the detector needs, in squares

    def spec(self) -> Dict[str, Any]:
        raise NotImplementedError

    def spec_string(self) -> str:
        raise NotImplementedError

    def all_object_points(self) -> np.ndarray:
        """Every corner of the board, (N, 3) metres, indexed by corner id."""
        raise NotImplementedError

    @property
    def min_points(self) -> int:
        """Fewest corners a view needs to count as a sample."""
        return 8

    def detect(self, gray: np.ndarray, search_width: int = 960) -> Optional[Detection]:
        raise NotImplementedError

    def texture(self, px_per_square: int) -> Tuple[np.ndarray, float]:
        """Grey board image and the pixel offset of the board origin.

        Board metres = ``(pixel_index + 0.5 - offset) * square_m / px_per_square``.
        Used by the mock camera and by :meth:`render`.
        """
        raise NotImplementedError

    def render(self, dpi: int = 300, margin_mm: float = 10.0) -> np.ndarray:
        """Printable image at *dpi*, with a white margin and a size label.

        Textures carry a one-square white border for rendering; the print
        replaces it with *margin_mm*, or ``quiet_squares`` if the detector needs more.
        """
        P = max(8, round(self.square_m / 0.0254 * dpi))
        img, _ = self.texture(P)
        img = img[P:-P, P:-P]
        m = max(round(margin_mm / 25.4 * dpi), round(self.quiet_squares * P))
        img = cv2.copyMakeBorder(img, m, m + max(m, 40), m, m, cv2.BORDER_CONSTANT, value=255)
        label = (f"{self.spec_string()}   at {dpi} dpi: square = {self.square_m * 1000:.1f} mm. "
                 f"Measure after printing and pass the measured size.")
        scale = max(0.4, dpi / 300 * 0.6)
        cv2.putText(img, label, (m, img.shape[0] - max(m, 40) // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, 0, max(1, dpi // 300), cv2.LINE_AA)
        return img

    def draw(self, img: np.ndarray, det: Detection, scale: float = 1.0,
             color=(0, 220, 0)) -> None:
        """Draw detected corners onto *img* (already scaled by *scale*) in place."""
        pts = det.img_pts * scale
        r = max(2, round(img.shape[1] / 300))
        for (x, y) in pts:
            cv2.circle(img, (round(x), round(y)), r, color, -1, cv2.LINE_AA)


def refine(gray: np.ndarray, pts: np.ndarray, clear_px: float) -> np.ndarray:
    """cornerSubPix at full resolution.

    *clear_px* is how far from a corner the image holds only the two squares
    meeting there; the window stays inside it; reaching past it (into a ChArUco
    marker) biases the corner by most of a pixel.
    """
    pts = np.ascontiguousarray(pts, np.float32).reshape(-1, 1, 2)
    win = int(np.clip(clear_px * 0.75, 2, 11))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    cv2.cornerSubPix(gray, pts, (win, win), (-1, -1), criteria)
    return pts.reshape(-1, 2)


def downscale(gray: np.ndarray, width: int) -> Tuple[np.ndarray, float]:
    h, w = gray.shape[:2]
    s = min(1.0, width / w) if width else 1.0
    if s >= 1.0:
        return gray, 1.0
    return cv2.resize(gray, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA), s
