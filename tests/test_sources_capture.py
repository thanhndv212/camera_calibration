"""Sources, the capture loop and the calibration file — no camera needed."""

from __future__ import annotations

import json
import time

import cv2
import numpy as np
import pytest

from camera_calibration import artifact
from camera_calibration._shared import load_calibration_parameters
from camera_calibration.capture import CaptureLoop, find_chessboard
from camera_calibration.sources import (
    DirSource, ESP32Source, MJPEGSource, MockSource, SourceEnded, UVCSource,
    open_source, parse_uri,
)


# -- URI parsing ----------------------------------------------------------------

@pytest.mark.parametrize("uri,cls", [
    ("0", UVCSource),
    ("uvc:1?size=1280x720&fps=30", UVCSource),
    ("uvc:name=USB%20Camera", UVCSource),
    ("esp32:192.168.1.142", ESP32Source),
    ("mjpeg:http://h:81/stream?x=1", MJPEGSource),
    ("dir:runs/x?loop=1", DirSource),
    ("mock:?size=320x240", MockSource),
])
def test_open_source_picks_class(uri, cls):
    assert isinstance(open_source(uri), cls)


def test_uri_details():
    s = open_source("uvc:name=USB Camera?size=1280x720&fps=30")
    assert (s.name, s.index, s.size, s.fps) == ("USB Camera", None, (1280, 720), 30.0)
    assert open_source("esp32:10.0.0.5").url == "http://10.0.0.5:81/stream"
    assert open_source("esp32:10.0.0.5:8080").url == "http://10.0.0.5:8081/stream"
    # mjpeg keeps its own query string
    assert parse_uri("mjpeg:http://h/s?a=1&b=2") == ("mjpeg", "http://h/s?a=1&b=2", {})
    with pytest.raises(ValueError):
        open_source("webcam")
    with pytest.raises(ValueError):
        open_source("ftp:x")


# -- mock and dir sources -------------------------------------------------------

def test_mock_is_a_true_pinhole_camera():
    """Detected corners land where the known K and pose project them."""
    src = open_source("mock:?fps=0&frames=5&noise=0")
    src.open()
    obj = src.object_points()
    for _ in range(5):
        f = src.read()
        corners = find_chessboard(cv2.cvtColor(f.img, cv2.COLOR_BGR2GRAY), src.board)
        assert corners is not None
        rvec, tvec = src.pose(f.seq)
        proj, _ = cv2.projectPoints(obj, rvec, tvec, src.camera_matrix, src.dist_coeffs)
        err = np.linalg.norm(proj.reshape(-1, 2) - corners.reshape(-1, 2), axis=1)
        assert err.max() < 0.5, err.max()
    with pytest.raises(SourceEnded):
        src.read()


def test_calibrating_the_mock_recovers_its_intrinsics():
    src = open_source("mock:?fps=0&frames=40&k1=-0.15")
    src.open()
    obj_pts, img_pts = [], []
    for _ in range(40):
        f = src.read()
        c = find_chessboard(cv2.cvtColor(f.img, cv2.COLOR_BGR2GRAY), src.board)
        if c is not None:
            obj_pts.append(src.object_points())
            img_pts.append(c)
    assert len(img_pts) >= 30
    # The mock only has k1, so solve only k1; k1 and k2 trade off otherwise.
    flags = cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST
    rms, K, D, _, _ = cv2.calibrateCamera(obj_pts, img_pts, src.size, None, None, flags=flags)
    assert rms < 0.5
    np.testing.assert_allclose(K, src.camera_matrix, rtol=0.02, atol=3.0)
    assert abs(D.ravel()[0] - (-0.15)) < 0.01


def test_dir_source_reads_in_order_and_ends(tmp_path):
    for i in range(3):
        cv2.imwrite(str(tmp_path / f"img_{i:02d}.png"), np.full((48, 64, 3), i * 50, np.uint8))
    (tmp_path / "notes.txt").write_text("not an image")
    src = open_source(f"dir:{tmp_path}")
    info = src.open()
    assert info.resolution == (64, 48) and info.extra["count"] == 3
    seen = [int(src.read().img[0, 0, 0]) for _ in range(3)]
    assert seen == [0, 50, 100]
    with pytest.raises(SourceEnded):
        src.read()


def test_dir_source_rejects_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        open_source(f"dir:{tmp_path}").open()


# -- capture loop ---------------------------------------------------------------

def test_slow_analysis_does_not_slow_the_camera():
    def slow(frame):
        time.sleep(0.1)
        return frame.seq

    with CaptureLoop(open_source("mock:?fps=30&noise=0"), analyze=slow) as loop:
        time.sleep(2.0)
        st = loop.stats()
    assert st["camera_fps"] > 25, st
    assert 5 < st["analysis_fps"] < 12, st
    assert st["analysis_skipped"] > 0 and st["source_gaps"] == 0


def test_loop_runs_a_finite_source_to_the_end_and_previews():
    loop = CaptureLoop(open_source("mock:?fps=0&frames=10&size=1280x720"),
                       analyze=lambda f: f.seq, preview_max=640)
    with loop:
        frame = loop.wait_frame(after_seq=9, timeout=5)
        assert frame is not None and frame.seq == 10
        deadline = time.monotonic() + 5
        while not loop.ended and time.monotonic() < deadline:
            time.sleep(0.01)
        assert loop.ended
        jpeg = loop.preview_jpeg()
        assert loop.preview_jpeg() is jpeg  # cached
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[1] == 640
    assert loop.latest().img.shape[1] == 1280  # samples stay full resolution


def test_loop_survives_a_flaky_source():
    class Flaky(MockSource):
        def read(self):
            if self._seq in (3, 4) and not getattr(self, "_failed", 0) >= 2:
                self._failed = getattr(self, "_failed", 0) + 1
                raise OSError("usb hiccup")
            return super().read()

    with CaptureLoop(Flaky(fps=0, frames=20)) as loop:
        assert loop.wait_frame(after_seq=19, timeout=5) is not None
        assert "usb hiccup" in loop.stats()["error"]


# -- calibration file -----------------------------------------------------------

def _cal(**kw):
    base = dict(camera_matrix=np.diag([500.0, 500.0, 1.0]) + [[0, 0, 320], [0, 0, 240], [0, 0, 0]],
                dist_coeffs=[0.1, -0.05, 0, 0, 0], image_size=(640, 480),
                camera={"name": "USB Camera", "unique_id": "0x1100032e69221"},
                target={"type": "charuco"}, reprojection_error=0.3)
    base.update(kw)
    return artifact.Calibration(**base)


def test_artifact_round_trip_and_legacy_loader(tmp_path):
    T = np.eye(4)
    T[:3, 3] = [0.1, 0.2, 0.3]
    p = artifact.save(_cal(transforms={"T_base_camera": T}), tmp_path / "cal.json")
    c = artifact.load(p)
    assert c.schema_version == 1 and c.image_size == (640, 480)
    np.testing.assert_allclose(c.transform("T_camera_base"), np.linalg.inv(T))
    legacy = load_calibration_parameters("x", calibration_file=p)
    assert legacy["resolution"] == (640, 480)
    np.testing.assert_allclose(legacy["camera_matrix"], c.camera_matrix)


def test_artifact_hash_catches_hand_edits(tmp_path):
    p = artifact.save(_cal(), tmp_path / "cal.json")
    d = json.loads(p.read_text())
    d["camera_matrix"][0][0] = 600
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="hash"):
        artifact.load(p)
    assert artifact.load(p, verify_hash=False).camera_matrix[0, 0] == 600


def test_artifact_reads_pre_schema_files(tmp_path):
    p = tmp_path / "old.json"
    p.write_text(json.dumps({
        "camera_matrix": [[1000, 0, 640], [0, 1000, 360], [0, 0, 1]],
        "dist_coeffs": [[0.1, 0, 0, 0, 0]], "image_size": [1280, 720],
        "chessboard_size": [9, 6], "square_size": 0.025, "reprojection_error": 0.39}))
    c = artifact.load(p)
    assert c.schema_version == 0
    assert c.target == {"type": "chessboard", "inner_corners": [9, 6], "square_m": 0.025}
    assert artifact.Calibration.from_dict(c.to_dict()).target == c.target


def test_artifact_check_refuses_the_wrong_camera():
    cal = _cal()
    cal.check({"name": "USB Camera", "unique_id": "0x1100032e69221", "resolution": [640, 480]})
    with pytest.raises(artifact.CalibrationMismatch, match="1280x720"):
        cal.check({"name": "USB Camera", "resolution": [1280, 720]})
    with pytest.raises(artifact.CalibrationMismatch, match="camera id"):
        cal.check({"name": "USB Camera", "unique_id": "other", "resolution": [640, 480]})
    with pytest.raises(artifact.CalibrationMismatch):
        cal.check({"name": "FaceTime HD Camera", "resolution": [640, 480]})


def test_artifact_rejects_unnamed_transforms():
    with pytest.raises(ValueError):
        _cal(transforms={"extrinsics": np.eye(4)})
