"""Smoke test: verify the webcam can be opened and delivers frames."""

from __future__ import annotations

import time
import unittest

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore[assignment]


@unittest.skipIf(cv2 is None, "OpenCV is not installed")
class CameraWindowSmokeTest(unittest.TestCase):
    """Ensure OpenCV can open the default camera and read at least one frame."""

    def test_camera_opens_and_delivers_frames(self) -> None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.skipTest("No accessible camera available in this environment")

        success = False
        try:
            deadline = time.time() + 5.0
            while time.time() < deadline:
                ret, frame = cap.read()
                if ret and frame is not None:
                    success = True
                    break
        finally:
            cap.release()

        self.assertTrue(success, "Camera opened but failed to supply frames")


if __name__ == "__main__":
    unittest.main()
