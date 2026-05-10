"""Unified CLI for the camera_calibration package.

Usage::

    camera-calibration <subcommand> [options]
    python -m camera_calibration <subcommand> [options]

Subcommands
-----------
capture     Capture chessboard images from webcam and calibrate.
calibrate   Load an existing calibration file and optionally run a live test.
detect      Live ArUco detection monitor.
generate    Generate ArUco marker PNG files.
estimate    Estimate camera intrinsics from a preset or auto-detection.
record      Stream raw camera feed to rerun (and optionally save .rrd).
view        Load a calibration JSON and display its summary in rerun.

Global rerun flags (available on all subcommands)
-------------------------------------------------
--no-spawn      Do not open the rerun viewer window.
--rerun-save    Save the rerun recording to the given .rrd file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

from . import _viz


# ---------------------------------------------------------------------------
# Shared argument helpers
# ---------------------------------------------------------------------------

def _add_rerun_args(parser: argparse.ArgumentParser) -> None:
    """Attach rerun-specific arguments to *parser*."""
    g = parser.add_argument_group("rerun")
    g.add_argument(
        "--no-spawn",
        action="store_true",
        dest="no_spawn",
        help="Do not open the rerun viewer window.",
    )
    g.add_argument(
        "--rerun-save",
        type=Path,
        metavar="FILE",
        dest="rerun_save",
        help="Write the rerun recording to FILE (.rrd).",
    )


def _init_rerun(args: argparse.Namespace, app_id: str) -> None:
    _viz.init_rerun(
        app_id,
        spawn=not args.no_spawn,
        save_path=args.rerun_save,
    )


def _parse_size(value: str) -> Tuple[int, int]:
    try:
        cols, rows = value.split(",")
        return int(cols.strip()), int(rows.strip())
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            "Pattern size must be formatted as 'cols,rows' (e.g. '10,7')"
        ) from exc


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------

def cmd_capture(args: argparse.Namespace) -> None:
    """Capture chessboard images, calibrate, and save results."""
    from .calibrator import CameraCalibrator

    _init_rerun(args, "camera_calibration/capture")
    cal = CameraCalibrator(
        chessboard_size=args.pattern_size,
        square_size=args.square_size,
    )
    success = cal.capture_calibration_images(
        num_images=args.images,
        camera_id=args.camera,
        manual=args.manual,
    )
    if not success:
        raise SystemExit("Not enough calibration images collected")
    cal.calibrate_camera()
    cal.print_calibration_summary()
    path = cal.save_calibration(args.output)
    print(f"\nCalibration saved to: {path}")


def cmd_calibrate(args: argparse.Namespace) -> None:
    """Load an existing calibration file and optionally run a live test."""
    from .calibrator import CameraCalibrator

    _init_rerun(args, "camera_calibration/calibrate")
    cal = CameraCalibrator(
        chessboard_size=args.pattern_size,
        square_size=args.square_size,
    )
    if not cal.load_calibration(args.calibration_file):
        raise SystemExit(f"Failed to load calibration from {args.calibration_file}")
    cal.print_calibration_summary()
    _viz.log_calibration_result(
        cal.camera_matrix,
        cal.dist_coeffs,
        rpe=float("nan"),
        num_images=0,
    )
    if args.test:
        cal.test_calibration(args.camera)


def cmd_detect(args: argparse.Namespace) -> None:
    """Run live ArUco detection and stream results to rerun."""
    from .detector import run_detection_monitor

    _init_rerun(args, "camera_calibration/detect")
    run_detection_monitor(
        camera_id=args.camera,
        dictionary_name=args.dictionary,
        highlight_ids=args.highlight or [],
    )


def cmd_generate(args: argparse.Namespace) -> None:
    """Generate ArUco marker PNG(s) and log them to rerun."""
    from .markers import PATTERNS_DIR, generate_aruco_marker, generate_marker_sheet

    _init_rerun(args, "camera_calibration/generate")
    save_path = args.output or PATTERNS_DIR / f"aruco_marker_{args.marker_id:02d}.png"
    generate_aruco_marker(
        marker_id=args.marker_id,
        marker_size=args.size,
        save_path=save_path,
        display=not args.no_display,
        dict_name=args.dict,
    )
    if args.sheet > 0:
        ids = range(args.marker_id, args.marker_id + args.sheet)
        generate_marker_sheet(
            marker_ids=ids,
            marker_size=args.size,
            dict_name=args.dict,
        )


def cmd_estimate(args: argparse.Namespace) -> None:
    """Estimate camera parameters from a preset or auto-detection."""
    from .estimator import (
        WebcamParameterEstimator,
        auto_detect_camera_parameters,
        print_parameters_summary,
    )

    _init_rerun(args, "camera_calibration/estimate")
    if args.preset:
        presets = WebcamParameterEstimator.get_webcam_presets()
        if args.preset not in presets:
            raise SystemExit(f"Unknown preset '{args.preset}'")
        params = presets[args.preset]
    else:
        params = auto_detect_camera_parameters(args.camera)
        if params is None:
            raise SystemExit("Could not auto-detect camera parameters")

    print_parameters_summary(params)


def cmd_record(args: argparse.Namespace) -> None:
    """Stream raw camera frames to rerun (no calibration)."""
    import cv2

    _init_rerun(args, "camera_calibration/record")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")

    print(f"Recording camera {args.camera} to rerun. Ctrl-C to stop.")
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            _viz.log_frame("camera/raw", frame, frame_idx)
            frame_idx += 1
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()

    print(f"Recorded {frame_idx} frames")


def cmd_view(args: argparse.Namespace) -> None:
    """Load a calibration JSON and display its summary in rerun."""
    import numpy as np

    from ._shared import load_calibration_parameters

    _init_rerun(args, "camera_calibration/view")
    params = load_calibration_parameters(
        args.camera_name or "camera",
        calibration_file=args.calibration_file,
        allow_fallback=False,
    )
    K = np.asarray(params["camera_matrix"])
    d = np.asarray(params["dist_coeffs"])
    meta = params.get("metadata", {})
    rpe = float(meta.get("reprojection_error", float("nan")))
    num_images = int(meta.get("num_images", 0))
    _viz.log_calibration_result(K, d, rpe, num_images)
    print("Calibration summary sent to rerun.")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="camera-calibration",
        description=(
            "Camera calibration and ArUco detection with rerun visualization."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- capture -------------------------------------------------------------
    p = sub.add_parser("capture", help="Capture calibration images and calibrate")
    p.add_argument("--camera", type=int, default=0, metavar="ID",
                   help="Camera device ID (default: 0)")
    p.add_argument("--pattern-size", type=_parse_size, default=(10, 7),
                   dest="pattern_size", metavar="COLSxROWS",
                   help="Chessboard inner corners as cols,rows (default: 10,7)")
    p.add_argument("--square-size", type=float, default=0.025,
                   dest="square_size", metavar="M",
                   help="Square side length in metres (default: 0.025)")
    p.add_argument("--images", type=int, default=20, metavar="N",
                   help="Target number of captured images (default: 20)")
    p.add_argument("--output", type=str, default=None, metavar="STEM",
                   help="Output filename stem (no extension)")
    p.add_argument("--manual", action="store_true",
                   help="Manual capture via Enter key (default: auto)")
    _add_rerun_args(p)

    # -- calibrate -----------------------------------------------------------
    p = sub.add_parser("calibrate", help="Load and display an existing calibration")
    p.add_argument("--camera", type=int, default=0, metavar="ID")
    p.add_argument("--calibration-file", required=True, dest="calibration_file",
                   metavar="FILE", help="Path to a .json or .pkl calibration file")
    p.add_argument("--pattern-size", type=_parse_size, default=(10, 7),
                   dest="pattern_size", metavar="COLSxROWS")
    p.add_argument("--square-size", type=float, default=0.025,
                   dest="square_size", metavar="M")
    p.add_argument("--test", action="store_true",
                   help="Run a live undistortion test after loading")
    _add_rerun_args(p)

    # -- detect --------------------------------------------------------------
    p = sub.add_parser("detect", help="Live ArUco detection monitor")
    p.add_argument("--camera", type=int, default=0, metavar="ID")
    p.add_argument("--dictionary", default=DEFAULT_DICT_NAME,
                   help=f"ArUco dictionary name (default: {DEFAULT_DICT_NAME})")
    p.add_argument("--highlight", type=int, nargs="*", default=[0],
                   metavar="ID", help="Marker IDs to flag when detected")
    _add_rerun_args(p)

    # -- generate ------------------------------------------------------------
    p = sub.add_parser("generate", help="Generate ArUco marker PNG(s)")
    p.add_argument("--marker-id", type=int, default=0, dest="marker_id")
    p.add_argument("--size", type=int, default=400, metavar="PX",
                   help="Marker image side length in pixels (default: 400)")
    p.add_argument("--dict", default="DICT_6X6_250",
                   help="ArUco dictionary (default: DICT_6X6_250)")
    p.add_argument("--output", type=Path, default=None,
                   help="Explicit output path for the single marker PNG")
    p.add_argument("--sheet", type=int, default=0, metavar="N",
                   help="Also generate a sheet of N sequential markers")
    p.add_argument("--no-display", action="store_true", dest="no_display",
                   help="Skip logging marker to rerun (headless mode)")
    _add_rerun_args(p)

    # -- estimate ------------------------------------------------------------
    p = sub.add_parser("estimate", help="Estimate camera parameters")
    p.add_argument("--camera", type=int, default=0, metavar="ID")
    p.add_argument(
        "--preset",
        choices=["basic_720p", "wide_720p", "basic_1080p", "wide_1080p", "ultrawide"],
        default=None,
        help="Use a predefined preset instead of auto-detecting",
    )
    _add_rerun_args(p)

    # -- record --------------------------------------------------------------
    p = sub.add_parser("record", help="Stream raw camera feed to rerun")
    p.add_argument("--camera", type=int, default=0, metavar="ID")
    _add_rerun_args(p)

    # -- view ----------------------------------------------------------------
    p = sub.add_parser("view", help="Display a calibration file in rerun")
    p.add_argument("--calibration-file", type=Path, default=None,
                   dest="calibration_file", metavar="FILE",
                   help="Path to a .json calibration file")
    p.add_argument("--camera-name", type=str, default=None,
                   dest="camera_name", metavar="NAME",
                   help="Camera name to search in the data directory")
    _add_rerun_args(p)

    return parser


# Map subcommand names to handler functions
_COMMANDS = {
    "capture":  cmd_capture,
    "calibrate": cmd_calibrate,
    "detect":   cmd_detect,
    "generate": cmd_generate,
    "estimate": cmd_estimate,
    "record":   cmd_record,
    "view":     cmd_view,
}

# Import DEFAULT_DICT_NAME for use in the parser definition above
from .detector import DEFAULT_DICT_NAME  # noqa: E402


def main() -> None:
    """Entry point for the ``camera-calibration`` console script."""
    parser = _build_parser()
    args = parser.parse_args()
    _COMMANDS[args.command](args)


if __name__ == "__main__":
    main()
