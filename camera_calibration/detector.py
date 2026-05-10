"""Real-time ArUco marker detection with rerun visualization."""

from __future__ import annotations

import argparse
import time
from typing import Iterable, Optional, Set, Tuple

import cv2

from . import _viz

DEFAULT_DICT_NAME = "DICT_4X4_50"

__all__ = ["get_detector", "run_detection_monitor"]


def get_detector(
    dictionary_name: str = DEFAULT_DICT_NAME,
) -> Tuple[object, object, Optional[object]]:
    """Build an ArUco detector compatible with the installed OpenCV version.

    Args:
        dictionary_name: OpenCV aruco dictionary name.

    Returns:
        ``(dictionary, parameters, detector)`` — *detector* is ``None`` when
        using the legacy OpenCV < 4.7 API.
    """
    dict_attr = getattr(cv2.aruco, dictionary_name)
    try:
        dictionary = cv2.aruco.getPredefinedDictionary(dict_attr)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        return dictionary, parameters, detector
    except AttributeError:  # OpenCV < 4.7
        dictionary = cv2.aruco.Dictionary_get(dict_attr)  # type: ignore[attr-defined]
        parameters = cv2.aruco.DetectorParameters_create()  # type: ignore[attr-defined]
        return dictionary, parameters, None


def run_detection_monitor(
    camera_id: int = 0,
    dictionary_name: str = DEFAULT_DICT_NAME,
    highlight_ids: Iterable[int] = (0,),
) -> None:
    """Stream live ArUco detection results to rerun.

    Detected marker corners are drawn on each frame and logged under
    ``camera/detection``.  Corner positions are also logged as 2-D point
    clouds under ``detection/corners``.  Press ``Ctrl-C`` to quit.

    Args:
        camera_id: ``cv2.VideoCapture`` device ID.
        dictionary_name: OpenCV aruco dictionary name.
        highlight_ids: Marker IDs to flag when detected.
    """
    highlight_set: Set[int] = set(highlight_ids)
    dictionary, parameters, detector = get_detector(dictionary_name)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return

    print("ArUco detection monitor started. Ctrl-C to stop.")
    frame_idx = 0
    last_detection_time: Optional[float] = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: failed to read frame from camera")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if detector is not None:
                corners, ids, _ = detector.detectMarkers(gray)
            else:
                corners, ids, _ = cv2.aruco.detectMarkers(  # type: ignore[attr-defined]
                    gray, dictionary, parameters=parameters
                )

            display = frame.copy()
            status_lines = [f"Frame: {frame_idx}"]

            if ids is not None:
                cv2.aruco.drawDetectedMarkers(display, corners, ids)
                detected_ids = [int(i[0]) for i in ids]
                last_detection_time = time.time()
                status_lines.append(f"Detected IDs: {detected_ids}")
                if highlight_set.intersection(detected_ids):
                    status_lines.append("TARGET MARKER DETECTED")
                _viz.log_aruco_corners("detection/corners", corners, frame_idx)
            else:
                status_lines.append("No markers detected")

            if last_detection_time is not None:
                elapsed = time.time() - last_detection_time
                status_lines.append(f"Last seen: {elapsed:.1f}s ago")

            _viz.log_frame("camera/detection", display, frame_idx)
            _viz.log_status("detection/status", "\n".join(status_lines))

            frame_idx += 1

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()

    print(f"Detection monitor closed after {frame_idx} frames")


# ---------------------------------------------------------------------------
# Stand-alone CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ArUco detection monitor")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--dictionary", default=DEFAULT_DICT_NAME)
    parser.add_argument("--highlight", type=int, nargs="*", default=[0])
    return parser.parse_args()


def main() -> None:  # pragma: no cover
    args = _parse_args()
    run_detection_monitor(
        camera_id=args.camera,
        dictionary_name=args.dictionary,
        highlight_ids=args.highlight or [],
    )


if __name__ == "__main__":  # pragma: no cover
    main()
