"""Benchmark jobs the workbench runs in the background, one at a time.

Each job is in :data:`JOBS` with the parameters it takes and the sources it
works with: the stream benchmark works on any source; the sweeps need a
``framesize`` control; snapshot jobs need an ESP32 CameraWebServer and pause the
live stream while they run, since the camera serves stills and the stream from
one sensor. Every finished job writes ``<save_dir>/bench_<time>/report.json``
(plus a few sample frames) and keeps the result for the page.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import cv2

from .. import bench
from ..sources import ESP32Source

if TYPE_CHECKING:
    from .server import Workbench

DEFAULT_FRAMESIZES = "6,8,10,11,12,13"

# kind -> label, parameters, requirement ("" | "framesize" | "esp32").
JOBS: Dict[str, Dict[str, Any]] = {
    "stream": {"label": "Stream benchmark", "params": ["duration", "clock"], "needs": ""},
    "stream-sweep": {"label": "Stream sweep", "params": ["duration", "framesizes", "clock"],
                     "needs": "framesize"},
    "snapshots": {"label": "Snapshot bench", "params": ["n"], "needs": "esp32"},
    "snapshot-sweep": {"label": "Snapshot sweep", "params": ["n", "framesizes"], "needs": "esp32"},
}


def _framesizes(text: Any) -> List[int]:
    vals = [int(v) for v in str(text or DEFAULT_FRAMESIZES).replace(" ", "").split(",") if v]
    if not vals:
        raise ValueError("no framesizes")
    return vals


class JobRunner:
    def __init__(self, wb: "Workbench") -> None:
        self.wb = wb
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self.status: Dict[str, Any] = {"running": False}

    @property
    def running(self) -> bool:
        return bool(self.status.get("running"))

    def available(self) -> List[Dict[str, Any]]:
        """The job list for the page, each with ``available`` and the reason if not."""
        loop = self.wb.loop
        src = loop.source if loop else None
        names = {c["name"] for c in self.wb.cached_controls()}
        out = []
        for kind, j in JOBS.items():
            why = ""
            if src is None:
                why = "open a source first"
            elif j["needs"] == "framesize" and "framesize" not in names:
                why = "source has no framesize control"
            elif j["needs"] == "esp32" and not isinstance(src, ESP32Source):
                why = "ESP32 CameraWebServer only"
            out.append({"kind": kind, "label": j["label"], "params": j["params"],
                        "available": not why, "why": why})
        return out

    def start(self, kind: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if kind not in JOBS:
            raise KeyError(kind)
        with self._lock:
            if self.running:
                raise RuntimeError(f"{self.status['kind']} is already running")
            why = next(j["why"] for j in self.available() if j["kind"] == kind)
            if why:
                raise RuntimeError(why)
            params = {k: params[k] for k in JOBS[kind]["params"] if params.get(k) not in (None, "")}
            self.status = {"running": True, "kind": kind, "label": JOBS[kind]["label"],
                           "progress": 0.0, "params": params, "started": time.time()}
        self._thread = threading.Thread(target=self._run, args=(kind, params),
                                        name=f"job-{kind}", daemon=True)
        self._thread.start()
        return self.status

    def wait(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        if self._thread is not None:
            self._thread.join(timeout)
        return self.status

    def _progress(self, f: float) -> None:
        self.status["progress"] = round(max(0.0, min(1.0, f)), 3)

    def _run(self, kind: str, params: Dict[str, Any]) -> None:
        t0 = time.monotonic()
        out_dir = self.wb.save_dir / f"bench_{time.strftime('%Y%m%d_%H%M%S')}"
        try:
            result = self._execute(kind, params, out_dir)
            meta = self._meta(kind, params, time.monotonic() - t0)
            report = out_dir / "report.json"
            out_dir.mkdir(parents=True, exist_ok=True)
            meta["report"] = str(report)
            report.write_text(json.dumps({"meta": meta, "result": result}, indent=2, default=str))
            self.status.update(result=result, meta=meta, report=str(report), error=None)
        except Exception as exc:
            self.status.update(result=None, error=f"{type(exc).__name__}: {exc}")
        finally:
            self.status.update(running=False, progress=1.0,
                               finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    def _meta(self, kind: str, params: Dict[str, Any], duration: float) -> Dict[str, Any]:
        wb = self.wb
        src = wb.source_object()
        meta = {
            "kind": kind, "params": params,
            "source": src.uri if src else None,
            "host": getattr(src, "host", None),
            "aruco_dict": wb.overlay["aruco_dict"] if wb.overlay["aruco"] else None,
            "target": wb.target.spec_string(),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "opencv": cv2.__version__,
            "duration_s": round(duration, 2),
        }
        if params.get("clock"):
            meta["clock"] = {"dict": "DICT_4X4_50", "period_ms": bench.CLOCK_PERIOD_MS,
                             "n_ids": bench.clock_ids("DICT_4X4_50")}
        return meta

    def _execute(self, kind: str, p: Dict[str, Any], out_dir: Path) -> Any:
        wb = self.wb
        aruco = bench.aruco_detector(wb.overlay["aruco_dict"]) if wb.overlay["aruco"] else None
        clock = bench.make_clock() if p.get("clock") else None
        duration = float(p.get("duration") or 10)
        n = int(p.get("n") or 30)

        def stream(tag: str, prog) -> dict:
            loop = wb.loop
            if loop is None:
                raise RuntimeError("source closed")
            return bench.stream_bench(loop, duration, aruco=aruco, target=wb.target, clock=clock,
                                      save=out_dir, tag=tag, progress=prog)

        with wb.benchmarking():
            if kind == "stream":
                return stream("stream", self._progress)
            if kind == "stream-sweep":
                return bench.control_sweep(
                    wb.loop.source, "framesize", _framesizes(p.get("framesizes")),
                    lambda v, prog: stream(f"fs{v}", prog), progress=self._progress)
            with wb.suspended() as src:
                if kind == "snapshots":
                    return bench.snapshot_bench(src.host, n, timeout=src.timeout, aruco=aruco,
                                                target=wb.target, save=out_dir,
                                                progress=self._progress)
                per = max(3, n // 3)
                return bench.control_sweep(
                    src, "framesize", _framesizes(p.get("framesizes")),
                    lambda v, prog: bench.snapshot_bench(
                        src.host, per, timeout=src.timeout, aruco=aruco, target=wb.target,
                        save=out_dir, tag=f"fs{v}", progress=prog),
                    progress=self._progress)
