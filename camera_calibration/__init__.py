"""camera_calibration — standalone camera calibration package.

Provides camera intrinsic calibration (chessboard), ArUco marker
generation/detection, and parameter estimation utilities.  All live
visualization is handled through rerun-sdk.

Quick-start::

    from camera_calibration import CameraCalibrator
    cal = CameraCalibrator()
    cal.capture_calibration_images(num_images=20)
    cal.calibrate_camera()
    cal.save_calibration()

CLI::

    camera-calibration capture --images 20
    camera-calibration detect --dictionary DICT_4X4_50
    camera-calibration generate --marker-id 0 --sheet 4
    camera-calibration view --calibration-file data/my_calib.json
    camera-calibration --help
"""

from __future__ import annotations

from pathlib import Path

__version__ = "0.1.0"

# Convenience path constants
PACKAGE_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PACKAGE_ROOT / "data"
PATTERNS_DIR: Path = PACKAGE_ROOT / "patterns"

from .calibrator import CameraCalibrator
from .detector import run_detection_monitor
from .estimator import (
    WebcamParameterEstimator,
    auto_detect_camera_parameters,
    print_parameters_summary,
)
from .markers import (
    generate_aruco_marker,
    generate_marker_set,
    generate_marker_sheet,
    get_aruco_dictionary,
)
from ._shared import (
    load_calibration_parameters,
    parse_resolution,
    resolve_camera_output_prefix,
)

__all__ = [
    # Core classes
    "CameraCalibrator",
    "WebcamParameterEstimator",
    # Functions
    "auto_detect_camera_parameters",
    "generate_aruco_marker",
    "generate_marker_set",
    "generate_marker_sheet",
    "get_aruco_dictionary",
    "load_calibration_parameters",
    "parse_resolution",
    "print_parameters_summary",
    "resolve_camera_output_prefix",
    "run_detection_monitor",
    # Paths
    "DATA_DIR",
    "PACKAGE_ROOT",
    "PATTERNS_DIR",
    # Metadata
    "__version__",
]
