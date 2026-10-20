"""Targets, the intrinsics procedure, ``solve`` and the workbench server — on the mock camera."""

from __future__ import annotations

import json
import threading
import urllib.request

import cv2
import numpy as np
import pytest

from camera_calibration import artifact
from camera_calibration.__main__ import main as cli_main
from camera_calibration.procedures import IntrinsicsProcedure, solve_from_source
from camera_calibration.sources import Frame, open_source
from camera_calibration.targets import (
    DEFAULT_TARGET, CharucoTarget, parse_target, target_from_spec,
)


def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# -- targets --------------------------------------------------------------------

def test_default_target_is_charuco_and_specs_round_trip():
    t = parse_target(DEFAULT_TARGET)
    assert isinstance(t, CharucoTarget)
    for spec in (DEFAULT_TARGET, "chessboard?inner=9x6&square=0.025"):
        t = parse_target(spec)
        assert target_from_spec(t.spec()).spec() == t.spec()
        assert target_from_spec(t.spec_string()).spec() == t.spec()
    with pytest.raises(ValueError):
        parse_target("circles?grid=4x11")


@pytest.mark.parametrize("kind", ["chessboard", "charuco"])
def test_target_corners_match_ground_truth(kind):
    """Detected corners land within a fraction of a pixel of the mock's projection."""
    src = open_source(f"mock:?fps=0&frames=5&target={kind}&size=1280x720&noise=2")
    src.open()
    obj = src.target.all_object_points()
    for _ in range(5):
        f = src.read()
        det = src.target.detect(_gray(f.img))
        assert det is not None and len(det) >= src.target.min_points
        rvec, tvec = src.pose(f.seq)
        truth, _ = cv2.projectPoints(obj[det.ids], rvec, tvec, src.camera_matrix, src.dist_coeffs)
        truth = truth.reshape(-1, 2)
        err = np.linalg.norm(det.img_pts - truth, axis=1)
        assert err.max() < 0.3, err.max()
        assert np.allclose(det.obj_pts, obj[det.ids])


@pytest.mark.parametrize("spec", [DEFAULT_TARGET, "chessboard?inner=9x6&square=0.025"])
def test_printed_board_is_detected(spec):
    t = target_from_spec(spec)
    det = t.detect(t.render(dpi=150))
    assert det is not None and len(det) == len(t.all_object_points())


# -- intrinsics procedure -------------------------------------------------------

@pytest.mark.parametrize("kind", ["charuco", "chessboard"])
def test_procedure_recovers_intrinsics_with_held_out_check(kind):
    src = open_source(f"mock:?fps=0&frames=40&target={kind}&size=1280x720&k1=-0.2")
    K_true = src.camera_matrix
    proc, info = solve_from_source(src, src.target)
    r = proc.result
    assert len(proc.samples) == 40 and sum(s.held_out for s in proc.samples) == 8
    assert len(r.per_view_errors) == 32 and len(r.held_out_per_view) == 8
    assert r.rms < 0.2 and r.held_out_error < 0.2
    assert np.allclose(r.camera_matrix[[0, 1, 0, 1], [0, 1, 2, 2]],
                       K_true[[0, 1, 0, 1], [0, 1, 2, 2]], rtol=0.005)
    assert abs(r.dist_coeffs[0] + 0.2) < 0.05

    cal = proc.artifact(info)
    assert cal.held_out_error == r.held_out_error and cal.target == src.target.spec()
    cal.check(info)


def test_auto_accept_needs_steady_and_new_view():
    src = open_source("mock:?fps=0&frames=3&target=charuco")
    info = src.open()
    proc = IntrinsicsProcedure(src.target, info.resolution, auto=True, steady_s=0.4)
    f0 = src.read()
    det = src.target.detect(_gray(f0.img))
    for k, t in enumerate([0.0, 0.2, 0.39]):  # not steady long enough yet
        fb = proc.observe(Frame(f0.img, t, k + 1), det)
        assert not fb["accepted"]
    assert proc.observe(Frame(f0.img, 0.45, 4), det)["accepted"]
    # The same view again, however steady, is not new.
    for k, t in enumerate([1.0, 1.5, 2.0]):
        fb = proc.observe(Frame(f0.img, t, 10 + k), det)
        assert not fb["accepted"] and not fb["new"]
    assert "seen this view" in fb["hint"]
    assert len(proc.samples) == 1 and proc.coverage.sum() == len(det)
    proc.drop(0)
    assert proc.coverage.sum() == 0


def test_solve_refuses_too_few_views():
    src = open_source("mock:?fps=0&frames=4&target=charuco")
    with pytest.raises(RuntimeError, match="at least 6"):
        solve_from_source(src, src.target)


# -- solve CLI from a directory -------------------------------------------------

def test_solve_cli_from_dir(tmp_path, monkeypatch, capsys):
    src = open_source("mock:?fps=0&frames=15&target=charuco&size=960x540")
    src.open()
    for i in range(15):
        cv2.imwrite(str(tmp_path / f"{i:03d}.png"), src.read().img)
    out = tmp_path / "cal.json"
    monkeypatch.setattr("sys.argv", ["camera-calibration", "solve", "--from", f"dir:{tmp_path}",
                                     "--out", str(out)])
    cli_main()
    assert "held-out RMS" in capsys.readouterr().out
    cal = artifact.load(out)
    assert cal.held_out_error is not None and cal.image_size == (960, 540)
    assert abs(cal.camera_matrix[0, 0] - src.camera_matrix[0, 0]) / src.camera_matrix[0, 0] < 0.01
    assert len(cal.inputs["samples"]) == 15


# -- workbench server -----------------------------------------------------------

def test_workbench_server_smoke(tmp_path):
    from camera_calibration.workbench import Workbench, make_server

    wb = Workbench("mock:?fps=60&target=charuco&size=640x480", save_dir=tmp_path)
    srv = make_server(wb, "127.0.0.1:0")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    def get(path):
        return urllib.request.urlopen(base + path, timeout=5)

    def post(path, body=None):
        req = urllib.request.Request(base + path, json.dumps(body or {}).encode(),
                                     {"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=10))

    try:
        assert b"Camera Workbench" in get("/").read()
        assert get("/target.png").read()[:4] == b"\x89PNG"
        assert get("/video").read(9) == b"--frame\r\n"
        wb.loop.wait_result(timeout=5)
        for _ in range(14):
            wb.loop.wait_result(after_seq=wb.loop.latest().seq, timeout=5)
            post("/api/calib/capture")
        res = post("/api/calib/solve")
        assert res["held_out_error"] is not None and res["rms"] < 0.5
        saved = post("/api/calib/save")
        assert artifact.load(saved["path"]).camera["transport"] == "mock"
        state = json.load(get("/api/state"))
        assert state["calib"]["samples"] >= 14 and state["stats"]["camera_fps"] > 0
        post("/api/target", {"spec": "chessboard?inner=9x6&square=0.025"})
        assert json.load(get("/api/state"))["calib"]["samples"] == 0
    finally:
        srv.shutdown()
        wb.close()
