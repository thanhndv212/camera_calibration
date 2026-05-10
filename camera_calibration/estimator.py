"""Estimate camera intrinsics when a full calibration is not available."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from . import _viz

# Regex patterns for patching source files with derived intrinsics
_MATRIX_PATTERN = re.compile(
    r"camera_matrix = np\.array\([\s\S]*?dtype=np\.float32\)",
    re.MULTILINE,
)
_DIST_PATTERN = re.compile(
    r"dist_coeffs = np\.[\s\S]*?dtype=np\.float32\)",
    re.MULTILINE,
)
_DEFAULT_DIST_SNIPPET = "dist_coeffs = np.zeros((4, 1))"

__all__ = [
    "WebcamParameterEstimator",
    "auto_detect_camera_parameters",
    "print_parameters_summary",
    "update_camera_collection_file",
]


class WebcamParameterEstimator:
    """Helpers for inferring camera intrinsics for common webcams."""

    @staticmethod
    def get_typical_parameters(
        resolution: Tuple[int, int] = (1280, 720),
        fov_degrees: float = 60.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return a heuristic camera matrix and zero distortion coefficients.

        Args:
            resolution: ``(width, height)`` in pixels.
            fov_degrees: Horizontal field of view in degrees.

        Returns:
            ``(camera_matrix, dist_coeffs)`` as ``float32`` numpy arrays.
        """
        width, height = resolution
        fov_rad = np.radians(fov_degrees)
        fx = width / (2 * np.tan(fov_rad / 2))
        cx, cy = width / 2.0, height / 2.0
        camera_matrix = np.array(
            [[fx, 0.0, cx], [0.0, fx, cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        dist_coeffs = np.zeros((5,), dtype=np.float32)
        return camera_matrix, dist_coeffs

    @staticmethod
    def get_webcam_presets() -> Dict[str, Dict[str, object]]:
        """Return predefined presets for popular webcam configurations."""
        presets = {
            "basic_720p":  {"resolution": (1280, 720),  "fov": 60.0, "description": "Basic 720p webcam"},
            "wide_720p":   {"resolution": (1280, 720),  "fov": 78.0, "description": "Wide-angle 720p webcam"},
            "basic_1080p": {"resolution": (1920, 1080), "fov": 65.0, "description": "Basic 1080p webcam"},
            "wide_1080p":  {"resolution": (1920, 1080), "fov": 78.0, "description": "Wide-angle 1080p webcam"},
            "ultrawide":   {"resolution": (1280, 720),  "fov": 90.0, "description": "Ultra-wide webcam"},
        }
        results: Dict[str, Dict[str, object]] = {}
        for name, params in presets.items():
            K, d = WebcamParameterEstimator.get_typical_parameters(
                params["resolution"], params["fov"]
            )
            results[name] = {
                "camera_matrix": K,
                "dist_coeffs": d,
                "resolution": params["resolution"],
                "fov": params["fov"],
                "description": params["description"],
            }
        return results

    @staticmethod
    def detect_camera_resolution(camera_id: int = 0) -> Optional[Tuple[int, int]]:
        """Probe the camera for its active resolution.

        Args:
            camera_id: ``cv2.VideoCapture`` device ID.

        Returns:
            ``(width, height)`` or ``None`` if the camera cannot be opened.
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        h, w = frame.shape[:2]
        return (w, h)

    @staticmethod
    def estimate_fov_from_detection(
        camera_id: int = 0,
        known_object_size: float = 0.1,
        known_distance: float = 0.5,
    ) -> Optional[float]:
        """Estimate horizontal FOV by measuring a known object in the frame.

        Streams the camera feed to rerun so the user can visually align the
        object, then prompts for the measured pixel width on the terminal.

        Args:
            camera_id: Camera device ID.
            known_object_size: Real-world size of the object in metres.
            known_distance: Distance from camera to object in metres.

        Returns:
            Estimated FOV in degrees, or ``None`` on failure.
        """
        print(f"Place an object of {known_object_size * 100:.1f} cm "
              f"at {known_distance * 100:.1f} cm from the camera.")
        print("Watch the rerun viewer. Press Ctrl-C when ready to measure.")

        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            return None

        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Draw crosshair guides on the frame for alignment
                h, w = frame.shape[:2]
                display = frame.copy()
                cv2.line(display, (w // 2, 0), (w // 2, h), (0, 255, 0), 1)
                cv2.line(display, (0, h // 2), (w, h // 2), (0, 255, 0), 1)
                _viz.log_frame("camera/fov_estimation", display, frame_idx)
                frame_idx += 1
        except KeyboardInterrupt:
            pass
        finally:
            cap.release()

        try:
            pixel_width = float(input("Measured object width in pixels: "))
        except (ValueError, EOFError):
            return None

        angular_size = 2 * np.arctan(known_object_size / (2 * known_distance))
        pixels_per_rad = pixel_width / angular_size
        fov_rad = frame.shape[1] / pixels_per_rad
        return float(np.degrees(fov_rad))


def auto_detect_camera_parameters(
    camera_id: int = 0,
) -> Optional[Dict[str, object]]:
    """Detect camera resolution and return the best-matching preset.

    Args:
        camera_id: Camera device ID.

    Returns:
        Parameter dictionary with ``camera_matrix``, ``dist_coeffs``,
        ``resolution``, ``fov``, and ``description``, or ``None`` on failure.
    """
    print("Auto-detecting camera parameters…")
    resolution = WebcamParameterEstimator.detect_camera_resolution(camera_id)
    if resolution is None:
        print("Could not detect camera resolution")
        return None

    width, height = resolution
    print(f"Detected resolution: {width}x{height}")

    presets = WebcamParameterEstimator.get_webcam_presets()
    best_match: Optional[Tuple[str, Dict[str, object]]] = None
    for name, preset in presets.items():
        if preset["resolution"] == resolution:
            if best_match is None or preset["fov"] == 60.0:
                best_match = (name, preset)

    if best_match is None:
        K, d = WebcamParameterEstimator.get_typical_parameters(resolution)
        return {
            "camera_matrix": K,
            "dist_coeffs": d,
            "resolution": resolution,
            "fov": 60.0,
            "description": f"Generic parameters for {width}x{height}",
        }

    name, preset = best_match
    print(f"Best preset match: {name} — {preset['description']}")
    return preset


def print_parameters_summary(params: Dict[str, object]) -> None:
    """Print estimated camera parameters to the terminal.

    Args:
        params: Dictionary as returned by :func:`auto_detect_camera_parameters`
                or :meth:`WebcamParameterEstimator.get_webcam_presets`.
    """
    K = params["camera_matrix"]
    d = params["dist_coeffs"]
    resolution = params["resolution"]

    print("\n" + "=" * 60)
    print("ESTIMATED CAMERA PARAMETERS")
    print("=" * 60)
    print(f"Description : {params.get('description', 'N/A')}")
    print(f"Resolution  : {resolution[0]} x {resolution[1]}")
    print(f"Field of view: {params['fov']:.1f}°")
    print(f"\nCamera Matrix:")
    print(f"  fx = {K[0, 0]:.2f}  fy = {K[1, 1]:.2f}")
    print(f"  cx = {K[0, 2]:.2f}  cy = {K[1, 2]:.2f}")
    print(f"\nDistortion: {d.flatten()}")
    print("=" * 60)


def update_camera_collection_file(
    params: Dict[str, object],
    target_file: Optional[Path] = None,
) -> bool:
    """Patch a Python source file with the estimated intrinsics.

    Uses regex substitution to replace ``camera_matrix`` and ``dist_coeffs``
    assignments in the target file.

    Args:
        params: Parameter dict from :func:`auto_detect_camera_parameters`.
        target_file: File to patch; falls back to the
                     ``CAMERA_CALIBRATION_COLLECTION_FILE`` environment variable.

    Returns:
        ``True`` on success, ``False`` otherwise.
    """
    if target_file is None:
        override = os.environ.get("CAMERA_CALIBRATION_COLLECTION_FILE")
        if override is None:
            print("No target file specified and CAMERA_CALIBRATION_COLLECTION_FILE not set")
            return False
        target_file = Path(override)
    else:
        target_file = Path(target_file)

    if not target_file.exists():
        print(f"Target file not found: {target_file}")
        return False

    try:
        content = target_file.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"Error reading {target_file}: {exc}")
        return False

    K = params["camera_matrix"]
    d = params["dist_coeffs"]
    matrix_line = f"camera_matrix = np.array({K.tolist()}, dtype=np.float32)"
    dist_line = f"dist_coeffs = np.array({d.flatten().tolist()}, dtype=np.float32)"

    new_content, n = _MATRIX_PATTERN.subn(matrix_line, content, count=1)
    if n == 0:
        new_content = content.replace(
            "camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)",
            matrix_line,
            1,
        )

    final_content, n = _DIST_PATTERN.subn(dist_line, new_content, count=1)
    if n == 0:
        final_content = final_content.replace(_DEFAULT_DIST_SNIPPET, dist_line, 1)

    try:
        target_file.write_text(final_content, encoding="utf-8")
    except OSError as exc:
        print(f"Error writing {target_file}: {exc}")
        return False

    print(f"Updated camera parameters in {target_file}")
    return True


# ---------------------------------------------------------------------------
# Stand-alone CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Webcam parameter estimator")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--preset",
        choices=["basic_720p", "wide_720p", "basic_1080p", "wide_1080p", "ultrawide"],
        default=None,
    )
    parser.add_argument("--update-target", type=Path, default=None, dest="update_target")
    return parser.parse_args()


def main() -> None:  # pragma: no cover
    args = _parse_args()
    if args.preset:
        presets = WebcamParameterEstimator.get_webcam_presets()
        params = presets[args.preset]
    else:
        params = auto_detect_camera_parameters(args.camera)
        if params is None:
            raise SystemExit("Could not detect camera parameters")

    print_parameters_summary(params)

    if args.update_target:
        update_camera_collection_file(params, args.update_target)


if __name__ == "__main__":  # pragma: no cover
    main()
