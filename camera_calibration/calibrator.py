"""Camera calibration using chessboard patterns and rerun visualization."""

from __future__ import annotations

import json
import pickle
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from . import _viz
from ._shared import DATA_DIR

# Minimum seconds between successive auto-captures
_AUTO_CAPTURE_INTERVAL: float = 1.5

__all__ = ["CameraCalibrator"]


class CameraCalibrator:
    """Intrinsic camera calibration using chessboard patterns.

    Workflow::

        cal = CameraCalibrator()
        cal.capture_calibration_images(num_images=20, camera_id=0)
        cal.calibrate_camera()
        cal.print_calibration_summary()
        cal.save_calibration()

    All live frames are streamed to the active rerun recording so that the
    session can be inspected and replayed.
    """

    def __init__(
        self,
        chessboard_size: Tuple[int, int] = (10, 7),
        square_size: float = 0.025,
    ) -> None:
        """Initialise the calibrator.

        Args:
            chessboard_size: ``(cols, rows)`` of inner chessboard corners.
            square_size: Physical side length of each square in metres.
        """
        self.chessboard_size = chessboard_size
        self.square_size = square_size

        # Pre-compute 3-D object points for one chessboard pose
        num_corners = chessboard_size[0] * chessboard_size[1]
        self.objp = np.zeros((num_corners, 3), np.float32)
        self.objp[:, :2] = (
            np.mgrid[0 : chessboard_size[0], 0 : chessboard_size[1]]
            .T.reshape(-1, 2)
        )
        self.objp *= square_size

        # Collected calibration data
        self.objpoints: List[np.ndarray] = []
        self.imgpoints: List[np.ndarray] = []
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.image_size: Optional[Tuple[int, int]] = None

        print(
            f"Calibrator ready: {chessboard_size[0]}x{chessboard_size[1]} "
            f"chessboard, square_size={square_size * 1000:.1f} mm"
        )

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def capture_calibration_images(
        self,
        num_images: int = 20,
        camera_id: int = 0,
        *,
        manual: bool = False,
    ) -> bool:
        """Capture calibration images from the webcam.

        Frames are streamed live to rerun.  Two capture modes are supported:

        * **Auto** (default): a frame is captured every
          ``_AUTO_CAPTURE_INTERVAL`` seconds whenever chessboard corners are
          detected.  Press ``Ctrl-C`` to stop early.
        * **Manual** (``manual=True``): press ``Enter`` in the terminal to
          capture the current frame; ``Ctrl-C`` to finish.

        Args:
            num_images: Target number of accepted captures.
            camera_id: ``cv2.VideoCapture`` device ID.
            manual: Switch to manual capture mode.

        Returns:
            ``True`` when at least 10 images were collected (minimum for a
            reasonable calibration).
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return False

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print(f"\nCapture started — target: {num_images} images")
        if manual:
            print("Mode: MANUAL — press Enter to capture, Ctrl-C to stop")
            _manual_event = threading.Event()

            def _stdin_reader() -> None:
                try:
                    while True:
                        input()
                        _manual_event.set()
                except (EOFError, OSError):
                    pass

            threading.Thread(target=_stdin_reader, daemon=True).start()
        else:
            print(
                f"Mode: AUTO — captures every {_AUTO_CAPTURE_INTERVAL}s "
                "when corners are found. Ctrl-C to stop."
            )
            _manual_event = None  # type: ignore[assignment]

        detect_flags = (
            cv2.CALIB_CB_ADAPTIVE_THRESH
            | cv2.CALIB_CB_FAST_CHECK
            | cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        refine_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        captured_count = 0
        last_capture_time = 0.0
        frame_idx = 0

        try:
            while captured_count < num_images:
                ret, frame = cap.read()
                if not ret:
                    print("Error: failed to read from camera")
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                found, corners = cv2.findChessboardCorners(
                    gray, self.chessboard_size, detect_flags
                )

                display = frame.copy()
                corners_refined: Optional[np.ndarray] = None

                if found:
                    corners_refined = cv2.cornerSubPix(
                        gray, corners, (11, 11), (-1, -1), refine_criteria
                    )
                    cv2.drawChessboardCorners(
                        display, self.chessboard_size, corners_refined, True
                    )
                    status = f"DETECTED — {captured_count}/{num_images} captured"
                else:
                    status = f"No corners — {captured_count}/{num_images} captured"

                _viz.log_frame("camera/live", display, frame_idx)
                _viz.log_status("calibration/status", status)

                # Determine whether to capture this frame
                should_capture = False
                now = time.monotonic()

                if manual:
                    if _manual_event.is_set():
                        _manual_event.clear()
                        if found:
                            should_capture = True
                        else:
                            print("No corners detected — reposition the chessboard")
                else:
                    if found and (now - last_capture_time) >= _AUTO_CAPTURE_INTERVAL:
                        should_capture = True

                if should_capture and corners_refined is not None:
                    self.objpoints.append(self.objp.copy())
                    self.imgpoints.append(corners_refined)
                    captured_count += 1
                    last_capture_time = now
                    self.image_size = gray.shape[::-1]  # (width, height)
                    print(f"Captured {captured_count}/{num_images}")
                    _viz.log_status(
                        "calibration/status",
                        f"Captured {captured_count}/{num_images}",
                    )

                frame_idx += 1

        except KeyboardInterrupt:
            print(f"\nStopped — {captured_count} images captured")
        finally:
            cap.release()

        if captured_count >= 9:
            print(f"Collected {captured_count} calibration images")
            return True

        print(
            f"Insufficient images ({captured_count}); "
            "at least 10 are needed for a reliable calibration."
        )
        return False

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate_camera(self) -> Tuple[float, np.ndarray, np.ndarray]:
        """Run OpenCV calibration on the captured data.

        Returns:
            ``(reprojection_error, camera_matrix, dist_coeffs)``

        Raises:
            ValueError: Fewer than 10 captured images are available.
        """
        if len(self.objpoints) < 10:
            raise ValueError(
                f"Need at least 10 captured images; have {len(self.objpoints)}"
            )

        print(f"\nCalibrating with {len(self.objpoints)} images…")
        rpe, self.camera_matrix, self.dist_coeffs, _, _ = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, self.image_size, None, None
        )

        quality = (
            "Excellent" if rpe < 0.5
            else "Good" if rpe < 1.0
            else "Acceptable" if rpe < 2.0
            else "Poor — consider recalibrating"
        )
        print(f"Reprojection error: {rpe:.4f} px ({quality})")

        _viz.log_calibration_result(
            self.camera_matrix,
            self.dist_coeffs,
            rpe,
            len(self.objpoints),
        )
        return rpe, self.camera_matrix, self.dist_coeffs

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_calibration(self, filename: Optional[str] = None) -> str:
        """Save calibration to JSON, pickle, and FIGAROH Python format.

        Args:
            filename: Base output stem (no extension).  Defaults to a
                      timestamped name inside ``DATA_DIR``.

        Returns:
            Path to the saved JSON file.
        """
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            stem = DATA_DIR / f"camera_calibration_{ts}"
        else:
            stem = Path(filename)
            if not stem.is_absolute():
                stem = DATA_DIR / stem

        stem.parent.mkdir(parents=True, exist_ok=True)

        # Re-run calibration with the current intrinsic guess to get final RPE
        rpe = cv2.calibrateCamera(
            self.objpoints,
            self.imgpoints,
            self.image_size,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.CALIB_USE_INTRINSIC_GUESS,
        )[0]

        data = {
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist(),
            "image_size": list(self.image_size),
            "chessboard_size": list(self.chessboard_size),
            "square_size": self.square_size,
            "num_images": len(self.objpoints),
            "timestamp": datetime.now().isoformat(),
            "reprojection_error": float(rpe),
        }

        json_path = stem.with_suffix(".json")
        json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        pkl_path = stem.with_suffix(".pkl")
        pkl_path.write_bytes(pickle.dumps(data))

        ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        figaroh_path = Path(f"{stem}_figaroh.py")
        figaroh_path.write_text(
            "# Camera calibration parameters\n"
            f"# Generated: {ts_str}\n\n"
            "import numpy as np\n\n"
            "# Camera intrinsic matrix\n"
            f"camera_matrix = np.array({self.camera_matrix.tolist()}, dtype=np.float32)\n\n"
            "# Distortion coefficients\n"
            f"dist_coeffs = np.array({self.dist_coeffs.flatten().tolist()}, dtype=np.float32)\n\n"
            "# Image size (width, height)\n"
            f"image_size = {tuple(self.image_size)}\n\n"
            "# Calibration info\n"
            f"reprojection_error = {rpe:.4f}  # pixels\n"
            f"num_calibration_images = {len(self.objpoints)}\n",
            encoding="utf-8",
        )

        print(f"Saved JSON   : {json_path}")
        print(f"       Pickle: {pkl_path}")
        print(f"       FIGAROH: {figaroh_path}")
        return str(json_path)

    def load_calibration(self, filename: str) -> bool:
        """Load calibration parameters from a JSON or pickle file.

        Args:
            filename: Path to a ``.json`` or ``.pkl`` calibration file.

        Returns:
            ``True`` on success.
        """
        try:
            path = Path(filename)
            if path.suffix == ".json":
                data = json.loads(path.read_text(encoding="utf-8"))
            elif path.suffix == ".pkl":
                data = pickle.loads(path.read_bytes())
            else:
                raise ValueError("File must be .json or .pkl")

            self.camera_matrix = np.array(data["camera_matrix"], dtype=np.float32)
            self.dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float32)
            self.image_size = tuple(data["image_size"])  # type: ignore[arg-type]
            print(f"Loaded calibration from {filename}")
            return True
        except Exception as exc:
            print(f"Error loading calibration: {exc}")
            return False

    # ------------------------------------------------------------------
    # Testing & summary
    # ------------------------------------------------------------------

    def test_calibration(self, camera_id: int = 0) -> None:
        """Stream original and undistorted frames to rerun side-by-side.

        Press ``Ctrl-C`` to stop.

        Args:
            camera_id: Camera device ID.
        """
        if self.camera_matrix is None:
            print("No calibration data loaded")
            return

        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return

        print("Streaming original vs undistorted to rerun. Ctrl-C to stop.")
        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                _viz.log_undistorted(
                    frame, self.camera_matrix, self.dist_coeffs, frame_idx
                )
                frame_idx += 1
        except KeyboardInterrupt:
            pass
        finally:
            cap.release()

    def print_calibration_summary(self) -> None:
        """Print the calibration parameters to the terminal."""
        if self.camera_matrix is None:
            print("No calibration data available")
            return

        K = self.camera_matrix
        d = self.dist_coeffs.flatten()
        print("\n" + "=" * 60)
        print("CAMERA CALIBRATION SUMMARY")
        print("=" * 60)
        print(f"  fx = {K[0, 0]:.2f}  fy = {K[1, 1]:.2f}")
        print(f"  cx = {K[0, 2]:.2f}  cy = {K[1, 2]:.2f}")
        if len(d) >= 5:
            print(
                f"  k1={d[0]:.6f}  k2={d[1]:.6f}  "
                f"p1={d[2]:.6f}  p2={d[3]:.6f}  k3={d[4]:.6f}"
            )
        if self.image_size:
            w, h = self.image_size
            fov_x = np.degrees(2 * np.arctan(w / (2 * K[0, 0])))
            fov_y = np.degrees(2 * np.arctan(h / (2 * K[1, 1])))
            print(f"  Resolution: {w}x{h}  FOV: H={fov_x:.1f}°  V={fov_y:.1f}°")
        print("=" * 60)


# ---------------------------------------------------------------------------
# Stand-alone CLI
# ---------------------------------------------------------------------------

def _parse_size(value: str) -> Tuple[int, int]:
    try:
        cols, rows = value.split(",")
        return int(cols), int(rows)
    except Exception as exc:
        raise argparse.ArgumentTypeError("Pattern size must be 'cols,rows'") from exc


def main() -> None:  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Camera calibration tool")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--pattern-size", type=_parse_size, default="10,7", dest="pattern_size")
    parser.add_argument("--square-size", type=float, default=0.025, dest="square_size")
    parser.add_argument("--images", type=int, default=20)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--manual", action="store_true",
                        help="Manual capture via Enter (default: auto)")
    parser.add_argument("--test-only", action="store_true", dest="test_only")
    parser.add_argument("--calibration-file", type=str, default=None, dest="calibration_file")
    args = parser.parse_args()

    cal = CameraCalibrator(
        chessboard_size=args.pattern_size, square_size=args.square_size
    )

    if args.test_only:
        if not args.calibration_file:
            raise SystemExit("--calibration-file is required with --test-only")
        if cal.load_calibration(args.calibration_file):
            cal.print_calibration_summary()
            cal.test_calibration(args.camera)
        return

    if not cal.capture_calibration_images(
        num_images=args.images, camera_id=args.camera, manual=args.manual
    ):
        raise SystemExit("Not enough calibration images collected")

    cal.calibrate_camera()
    cal.print_calibration_summary()
    cal.save_calibration(args.output)


if __name__ == "__main__":  # pragma: no cover
    main()
