"""Benchmarks and jobs, on the mock camera and the mock ESP32 CameraWebServer."""

from __future__ import annotations

import json
import threading
import urllib.request

import pytest

from camera_calibration import bench
from camera_calibration.esp32_mock import MockCameraWebServer
from camera_calibration.workbench import Workbench, make_server


def test_summarize_and_clock_lag():
    assert bench.summarize([]) is None
    s = bench.summarize([1.0, 2.0, 3.0, 4.0])
    assert s["n"] == 4 and s["max"] == 4.0 and s["mean"] == pytest.approx(2.5)
    # Slot 1000 of a 100 ms clock with 50 IDs shows ID 0 from t=100.0 to 100.1. A frame
    # showing it at t=100.08 is 30 ms past the slot's midpoint; at 100.18, 130 ms. ID 49
    # was the slot before.
    period, n = 0.1, 50
    assert bench.clock_lag_ms(100.08, 0, period, n) == pytest.approx(30, abs=1e-6)
    assert bench.clock_lag_ms(100.18, 0, period, n) == pytest.approx(130, abs=1e-6)
    assert bench.clock_lag_ms(100.08, 49, period, n) == pytest.approx(130, abs=1e-6)
    assert "DICT_4X4_50" in bench.clock_page_html()


def test_stream_bench_on_mock(tmp_path):
    wb = Workbench("mock:?fps=30&target=charuco&size=640x480", save_dir=tmp_path)
    try:
        jobs = {j["kind"]: j for j in wb.jobs.available()}
        assert jobs["stream"]["available"]
        assert not jobs["snapshots"]["available"] and not jobs["stream-sweep"]["available"]
        wb.jobs.start("stream", {"duration": 1.5, "n": 99})
        st = wb.jobs.wait(timeout=20)
        assert st["error"] is None, st["error"]
        r = st["result"]
        assert r["mode"] == "stream" and 20 < r["fps"] < 40 and r["frames"] >= 30
        assert r["stalls_over_250ms"] == 0 and r["frames_analysed"] > 0
        assert r["target_detect_rate"] > 0.5
        report = json.loads(open(st["report"]).read())
        assert report["meta"]["kind"] == "stream" and report["meta"]["params"] == {"duration": 1.5}
    finally:
        wb.close()


@pytest.fixture
def esp32():
    with MockCameraWebServer(port=0, jitter=False) as cam:
        yield cam


def test_esp32_controls_and_jobs(esp32, tmp_path):
    wb = Workbench(f"esp32:{esp32.host}", save_dir=tmp_path)
    try:
        src = wb.source_object()
        ctrls = {c["name"]: c for c in wb.cached_controls()}
        assert ctrls["framesize"]["kind"] == "choice" and ctrls["framesize"]["group"]
        assert {"quality", "aec", "awb", "hmirror"} <= set(ctrls)
        src.set_control("quality", 20)
        assert esp32.snapshot_state()["quality"] == 20
        assert all(j["available"] for j in wb.jobs.available())

        wb.jobs.start("snapshots", {"n": 4})
        st = wb.jobs.wait(timeout=30)
        assert st["error"] is None, st["error"]
        assert st["result"]["mode"] == "snapshot" and st["result"]["ok"] == 4
        assert wb.loop is not None and wb.loop.source is not None  # stream resumed

        before = esp32.snapshot_state()["framesize"]
        wb.jobs.start("stream-sweep", {"duration": 1, "framesizes": "6,8"})
        st = wb.jobs.wait(timeout=60)
        assert st["error"] is None, st["error"]
        rows = st["result"]
        assert [r["framesize"] for r in rows] == [6, 8]
        assert rows[0]["resolutions"] == ["320x240"] and rows[0]["fps"] > 0
        assert esp32.snapshot_state()["framesize"] == before
    finally:
        wb.close()


def test_job_endpoint_and_clock_page(tmp_path):
    wb = Workbench("mock:?fps=30&size=320x240", save_dir=tmp_path)
    srv = make_server(wb, "127.0.0.1:0")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    def post(path, body):
        req = urllib.request.Request(base + path, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=10))

    try:
        assert b"DICT_4X4_50" in urllib.request.urlopen(base + "/clock", timeout=5).read()
        post("/api/overlay", {"aruco": True, "grid": True})
        post("/api/job", {"kind": "stream", "params": {"duration": 1}})
        wb.jobs.wait(timeout=20)
        state = json.load(urllib.request.urlopen(base + "/api/state", timeout=5))
        assert state["overlay"]["aruco"] and state["job"]["result"]["mode"] == "stream"
        assert {j["kind"] for j in state["jobs"]} == {"stream", "stream-sweep", "snapshots",
                                                      "snapshot-sweep"}
    finally:
        srv.shutdown()
        wb.close()
