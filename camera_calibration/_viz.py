"""Rerun visualization helpers for camera calibration.

All public functions are no-ops when rerun-sdk is not installed, so that
the rest of the package can import this module unconditionally.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

try:
    import rerun as rr  # type: ignore

    _RERUN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _RERUN_AVAILABLE = False


def init_rerun(
    app_id: str = "camera_calibration",
    *,
    spawn: bool = True,
    save_path: Optional[Path] = None,
) -> None:
    """Initialise rerun, optionally spawning the viewer and/or saving to a file.

    Args:
        app_id: Application identifier shown in the rerun viewer.
        spawn: Open the rerun viewer automatically (default ``True``).
        save_path: If given, write the recording to this ``.rrd`` file.
    """
    if not _RERUN_AVAILABLE:
        return
    rr.init(app_id, spawn=spawn)
    if save_path is not None:
        rr.save(str(save_path))


def log_frame(
    entity: str,
    frame: np.ndarray,
    frame_idx: Optional[int] = None,
) -> None:
    """Log a camera frame (BGR or grayscale) to rerun.

    Args:
        entity: Rerun entity path, e.g. ``"camera/live"``.
        frame: BGR colour or grayscale ``uint8`` image from OpenCV.
        frame_idx: If given, advance the ``"frame"`` timeline.
    """
    if not _RERUN_AVAILABLE:
        return
    if frame_idx is not None:
        rr.set_time_sequence("frame", frame_idx)
    if frame.ndim == 2:
        rr.log(entity, rr.Image(frame))
    else:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rr.log(entity, rr.Image(rgb))


def log_status(entity: str, text: str) -> None:
    """Log a plain-text status message to rerun.

    Args:
        entity: Rerun entity path, e.g. ``"calibration/status"``.
        text: Status text to display.
    """
    if not _RERUN_AVAILABLE:
        return
    rr.log(entity, rr.TextDocument(text))


def log_calibration_result(
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    rpe: float,
    num_images: int,
) -> None:
    """Log a calibration summary as a Markdown document to rerun.

    Args:
        camera_matrix: 3×3 intrinsic matrix.
        dist_coeffs: Distortion coefficient vector (at least 4 elements).
        rpe: Reprojection error in pixels.
        num_images: Number of images used for calibration.
    """
    if not _RERUN_AVAILABLE:
        return
    K = camera_matrix
    d = dist_coeffs.flatten()
    quality = (
        "Excellent" if rpe < 0.5
        else "Good" if rpe < 1.0
        else "Acceptable" if rpe < 2.0
        else "Poor — consider recalibrating"
    )
    lines = [
        "# Camera Calibration Result",
        "",
        f"| Property | Value |",
        f"|---|---|",
        f"| Reprojection error | **{rpe:.4f} px** ({quality}) |",
        f"| Images used | {num_images} |",
        "",
        "## Intrinsics",
        "",
        f"| Parameter | Value |",
        f"|---|---|",
        f"| fx | {K[0, 0]:.2f} px |",
        f"| fy | {K[1, 1]:.2f} px |",
        f"| cx | {K[0, 2]:.2f} px |",
        f"| cy | {K[1, 2]:.2f} px |",
        "",
    ]
    if len(d) >= 5:
        lines += [
            "## Distortion",
            "",
            f"| Coefficient | Value |",
            f"|---|---|",
            f"| k1 | {d[0]:.6f} |",
            f"| k2 | {d[1]:.6f} |",
            f"| p1 | {d[2]:.6f} |",
            f"| p2 | {d[3]:.6f} |",
            f"| k3 | {d[4]:.6f} |",
        ]
    rr.log(
        "calibration/result",
        rr.TextDocument("\n".join(lines), media_type="text/markdown"),
    )


def log_undistorted(
    frame_bgr: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    frame_idx: Optional[int] = None,
) -> None:
    """Log both the original and undistorted frames to separate rerun entities.

    Args:
        frame_bgr: Input BGR frame from OpenCV.
        camera_matrix: 3×3 camera intrinsic matrix.
        dist_coeffs: Distortion coefficients.
        frame_idx: If given, advance the ``"frame"`` timeline.
    """
    if not _RERUN_AVAILABLE:
        return
    if frame_idx is not None:
        rr.set_time_sequence("frame", frame_idx)
    h, w = frame_bgr.shape[:2]
    new_K, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )
    undistorted = cv2.undistort(frame_bgr, camera_matrix, dist_coeffs, None, new_K)
    log_frame("camera/original", frame_bgr, frame_idx)
    log_frame("camera/undistorted", undistorted, frame_idx)


def log_aruco_corners(
    entity: str,
    corners: Iterable,
    frame_idx: Optional[int] = None,
) -> None:
    """Log ArUco corner positions as 2-D points in rerun.

    Args:
        entity: Rerun entity path, e.g. ``"detection/corners"``.
        corners: Sequence of corner arrays returned by ``cv2.aruco.detectMarkers``.
        frame_idx: If given, advance the ``"frame"`` timeline.
    """
    if not _RERUN_AVAILABLE:
        return
    if frame_idx is not None:
        rr.set_time_sequence("frame", frame_idx)
    corners_list = list(corners)
    if not corners_list:
        return
    pts = np.concatenate([c.reshape(-1, 2) for c in corners_list], axis=0)
    rr.log(entity, rr.Points2D(pts, radii=4.0))
