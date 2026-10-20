"""Workbench state and its HTTP server.

``Workbench`` owns one :class:`~camera_calibration.capture.CaptureLoop` and one
:class:`~camera_calibration.procedures.IntrinsicsProcedure`; the server only
translates HTTP to its methods. Swapping the source or the target rebuilds the
procedure, since samples from another camera or board cannot be mixed.

It also keeps live image-quality numbers, a two-minute rate history, optional
ArUco and thirds-grid overlays, and runs benchmark jobs (:mod:`.jobs`).

Endpoints: ``GET /`` (the page), ``/video`` (multipart MJPEG of the preview),
``/frame.jpg``, ``/api/state``, ``/api/cameras``, ``/target.png?dpi=300``,
``/clock`` (the lag clock page); ``POST /api/source``, ``/api/control``,
``/api/target``, ``/api/snapshot``, ``/api/overlay``, ``/api/job`` and
``/api/calib/<auto|capture|solve|drop|save|reset>``.
"""

from __future__ import annotations

import contextlib
import json
import re
import threading
import time
import urllib.parse
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Deque, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

from .. import artifact, bench
from ..capture import CaptureLoop
from ..procedures import IntrinsicsProcedure
from ..sources import Frame, list_cameras, open_source
from ..targets import DEFAULT_TARGET, Detection, target_from_spec
from .jobs import JobRunner

PAGE = Path(__file__).with_name("workbench.html")
HISTORY_S = 120
NOISE_FRAMES, NOISE_WIDTH = 10, 160
PERMISSION_HINT = ("macOS gives camera access per app: run the workbench from Terminal.app "
                   "and allow it under System Settings > Privacy & Security > Camera.")


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower() or "camera"


class Workbench:
    def __init__(self, source: Optional[str] = None, target: str = DEFAULT_TARGET, *,
                 model: str = "pinhole", save_dir: Path = Path("runs/workbench"),
                 preview_max: int = 960) -> None:
        self.save_dir = Path(save_dir)
        self.preview_max = preview_max
        self.target = target_from_spec(target)
        self.model = model
        self.auto = True
        self.collecting = False
        self.loop: Optional[CaptureLoop] = None
        self.proc: Optional[IntrinsicsProcedure] = None
        self.session: Optional[Path] = None
        self.error: Optional[str] = None
        self.saved: Optional[str] = None
        self._flash = 0.0
        self._lock = threading.RLock()
        self.overlay: Dict[str, Any] = {"aruco": False, "aruco_dict": "DICT_4X4_50", "grid": False}
        self._aruco = None
        self.live: Dict[str, Any] = {}
        self._live_t = 0.0
        self._noise: Deque[np.ndarray] = deque(maxlen=NOISE_FRAMES)
        self._aruco_hits: Deque[bool] = deque(maxlen=30)
        self.history: Deque[Dict[str, Any]] = deque(maxlen=HISTORY_S)
        self._history_t = 0.0
        self._bench = False
        self.paused = False
        self._paused_source = None
        self._controls: List[Dict[str, Any]] = []
        self.jobs = JobRunner(self)
        if source:
            self.set_source(source)

    # -- source / target ------------------------------------------------------

    def set_source(self, uri: str) -> None:
        if self.jobs.running:
            raise RuntimeError("a benchmark is running")
        with self._lock:
            self.close()
            self.error = None
            loop = CaptureLoop(open_source(uri), analyze=self._analyze, overlay=self._overlay,
                               preview_max=self.preview_max)
            try:
                info = loop.start()
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                if uri.startswith(("uvc", "0", "1", "2")):
                    self.error += " — " + PERMISSION_HINT
                raise
            self.loop = loop
            self._reset_live()
            self._new_session(info.resolution)

    def set_target(self, spec: str, model: Optional[str] = None) -> None:
        with self._lock:
            self.target = target_from_spec(spec)
            if model:
                self.model = model
            if self.loop is not None and self.loop.info is not None:
                frame = self.loop.latest()
                size = frame.img.shape[1::-1] if frame is not None else self.loop.info.resolution
                self._new_session(size)

    def _new_session(self, size: Tuple[int, int]) -> None:
        info = self.loop.info
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.session = self.save_dir / f"{_slug(info.name)}_{size[0]}x{size[1]}_{stamp}"
        self.proc = IntrinsicsProcedure(self.target, size, model=self.model, auto=False,
                                        sample_dir=self.session / "samples")
        self.collecting, self.saved = False, None

    def close(self) -> None:
        with self._lock:
            self.paused = False
            if self.loop is not None:
                self.loop.stop()
            self.loop, self.proc = None, None

    def set_control(self, name: str, value: Any) -> None:
        if self.loop is None:
            raise RuntimeError("no source")
        self.loop.source.set_control(name, value)
        self._noise.clear()  # the old frames no longer describe this setting

    def source_object(self):
        """The current source, also while a snapshot job has the stream paused."""
        return self._paused_source if self.paused else (self.loop.source if self.loop else None)

    def cached_controls(self) -> List[Dict[str, Any]]:
        """Source controls, re-read except while a benchmark job owns the camera."""
        if self.loop is not None and not self.jobs.running:
            try:
                self._controls = [c.to_dict() for c in self.loop.source.controls()]
            except Exception:
                pass
        return self._controls

    def set_overlay(self, body: Dict[str, Any]) -> Dict[str, Any]:
        for k in ("aruco", "grid"):
            if k in body:
                self.overlay[k] = bool(body[k])
        if body.get("aruco_dict"):
            if body["aruco_dict"] not in bench.ARUCO_DICTS:
                raise ValueError(f"unknown dictionary {body['aruco_dict']}")
            self.overlay["aruco_dict"] = body["aruco_dict"]
        self._aruco = (bench.aruco_detector(self.overlay["aruco_dict"])
                       if self.overlay["aruco"] else None)
        self._aruco_hits.clear()
        return self.overlay

    def _reset_live(self) -> None:
        self.live, self._live_t = {}, 0.0
        self._noise.clear()
        self._aruco_hits.clear()

    # -- benchmark hooks (used by jobs.JobRunner) -----------------------------

    @contextlib.contextmanager
    def benchmarking(self) -> Iterator[None]:
        """Skip calibration analysis while a job measures, so it does not compete."""
        self._bench = True
        try:
            yield
        finally:
            self._bench = False

    @contextlib.contextmanager
    def suspended(self) -> Iterator[Any]:
        """Close the live stream, yield its source, then reopen the same URI.

        The calibration session survives if the resolution comes back unchanged.
        """
        with self._lock:
            if self.loop is None:
                raise RuntimeError("no source")
            loop, proc, session = self.loop, self.proc, self.session
            self.loop.stop()
            self._paused_source, self.paused, self.loop = loop.source, True, None
        try:
            yield loop.source
        finally:
            with self._lock:
                self.paused = False
                new = CaptureLoop(open_source(loop.source.uri), analyze=self._analyze,
                                  overlay=self._overlay, preview_max=self.preview_max)
                try:
                    info = new.start()
                except Exception as exc:
                    self.error = f"reopening after the benchmark: {type(exc).__name__}: {exc}"
                    raise
                self.loop = new
                self._reset_live()
                if proc is not None and tuple(proc.image_size) == tuple(info.resolution):
                    self.proc, self.session = proc, session
                else:
                    self._new_session(info.resolution)

    # -- analysis (worker thread) ---------------------------------------------

    def _analyze(self, frame: Frame) -> Dict[str, Any]:
        gray = cv2.cvtColor(frame.img, cv2.COLOR_BGR2GRAY) if frame.img.ndim == 3 else frame.img
        if self._bench:
            return {}
        aruco = self._aruco
        markers = None
        if aruco is not None:
            corners, ids, _ = aruco.detectMarkers(gray)
            markers = (corners, ids)
            self._aruco_hits.append(ids is not None and len(ids) > 0)
        self._live_metrics(gray, markers)
        target, proc = self.target, self.proc
        det = target.detect(gray)
        fb = None
        if proc is not None:
            size = frame.img.shape[1::-1]
            if size != proc.image_size and not proc.samples:
                with self._lock:  # resolution changed before any sample: follow it
                    self._new_session(size)
                    proc = self.proc
            proc.auto = self.collecting and self.auto
            fb = proc.observe(frame, det)
            if fb.get("accepted"):
                self._flash = time.monotonic()
        return {"det": det, "fb": fb, "target": target, "markers": markers}

    def _live_metrics(self, gray: np.ndarray, markers) -> None:
        """Quality numbers for the page, at most twice a second."""
        now = time.monotonic()
        if now - self._live_t < 0.5:
            return
        self._live_t = now
        m = bench.frame_metrics(gray)
        h, w = gray.shape
        small = cv2.resize(gray, (NOISE_WIDTH, max(1, round(h * NOISE_WIDTH / w))),
                           interpolation=cv2.INTER_AREA).astype(np.float32)
        if self._noise and self._noise[-1].shape != small.shape:
            self._noise.clear()
        self._noise.append(small)
        m["temporal_noise"] = (float(np.stack(self._noise).std(axis=0).mean())
                               if len(self._noise) >= 5 else None)
        if markers is not None:
            corners, ids = markers
            m["aruco_ids"] = [] if ids is None else sorted(int(i) for i in ids.ravel())
            m["aruco_min_edge_px"] = (float(min(np.linalg.norm(c[0][0] - c[0][1]) for c in corners))
                                      if corners else None)
            m["aruco_detect_rate"] = (sum(self._aruco_hits) / len(self._aruco_hits)
                                      if self._aruco_hits else None)
        self.live = m

    def _overlay(self, img: np.ndarray, scale: float, frame: Optional[Frame], res: Any) -> None:
        proc = self.proc
        if proc is not None and proc.samples:
            h, w = img.shape[:2]
            gy, gx = proc.coverage.shape
            tint = np.zeros_like(img)
            for r in range(gy):
                for c in range(gx):
                    if proc.coverage[r, c]:
                        tint[r * h // gy:(r + 1) * h // gy, c * w // gx:(c + 1) * w // gx] = (60, 160, 40)
            cv2.addWeighted(img, 1.0, tint, 0.25, 0, dst=img)
        if res and isinstance(res.get("det"), Detection):
            res["target"].draw(img, res["det"], scale)
        if res and res.get("markers") is not None and res["markers"][1] is not None:
            corners, ids = res["markers"]
            cv2.aruco.drawDetectedMarkers(img, [c * scale for c in corners], ids)
        if self.overlay["grid"]:
            h, w = img.shape[:2]
            for k in (1, 2):
                cv2.line(img, (w * k // 3, 0), (w * k // 3, h), (200, 200, 200), 1)
                cv2.line(img, (0, h * k // 3), (w, h * k // 3), (200, 200, 200), 1)
        if time.monotonic() - self._flash < 0.25:
            cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (255, 255, 255), 8)

    # -- calibration ----------------------------------------------------------

    def _need_proc(self) -> IntrinsicsProcedure:
        if self.proc is None:
            raise RuntimeError("open a source first")
        return self.proc

    def calib(self, action: str, body: Dict[str, Any]) -> Dict[str, Any]:
        proc = self._need_proc()
        if action == "auto":
            self.collecting = bool(body.get("collecting", not self.collecting))
            if "auto" in body:
                self.auto = bool(body["auto"])
            return {"collecting": self.collecting, "auto": self.auto}
        if action == "capture":
            s = proc.capture()
            self._flash = time.monotonic()
            return {"samples": len(proc.samples), "held_out": s.held_out}
        if action == "solve":
            return proc.solve().to_dict()
        if action == "drop":
            proc.drop(int(body["index"]))
            return proc.solve().to_dict() if len(proc.samples) - proc.state()["held_out"] >= 6 else {}
        if action == "reset":
            proc.reset()
            self.saved = None
            return {}
        if action == "save":
            if proc.result is None:
                proc.solve()
            cal = proc.artifact(self.loop.info, capture=self._capture_settings())
            path = Path(body.get("path") or self.session / "calibration.json")
            artifact.save(cal, path)
            self.saved = str(path)
            return {"path": self.saved, "hash": cal.to_dict()["hash"]}
        raise KeyError(action)

    def _capture_settings(self) -> Dict[str, Any]:
        try:
            return {c.name: c.value for c in self.loop.source.controls()}
        except Exception:
            return {}

    def snapshot(self) -> str:
        if self.loop is None or self.loop.latest() is None:
            raise RuntimeError("no frame yet")
        frame = self.loop.latest()
        d = (self.session or self.save_dir) / "snapshots"
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{time.strftime('%H%M%S')}_{frame.seq}.png"
        cv2.imwrite(str(p), frame.img)
        return str(p)

    def _sample_history(self, stats: Optional[Dict[str, Any]]) -> None:
        """One point per second; a gap (paused, no source) is a null that breaks the line."""
        now = time.time()
        if now - self._history_t < 1.0:
            return
        self._history_t = now
        ok = stats is not None and not stats.get("ended")
        p95 = (stats.get("interframe_ms") or {}).get("p95") if ok else None
        self.history.append({"t": round(now, 1), "fps": stats["camera_fps"] if ok else None,
                             "p95": p95})

    def state(self) -> Dict[str, Any]:
        loop = self.loop
        stats = loop.stats() if loop else None
        self._sample_history(stats)
        out: Dict[str, Any] = {
            "source": loop.info.to_dict() if loop and loop.info else None,
            "stats": stats, "paused": self.paused,
            "controls": self.cached_controls() if loop or self.paused else [],
            "error": self.error, "collecting": self.collecting, "auto": self.auto,
            "target_spec": self.target.spec_string(), "model": self.model,
            "session": str(self.session) if self.session else None, "saved": self.saved,
            "calib": self.proc.state() if self.proc else None,
            "overlay": self.overlay, "aruco_dicts": bench.ARUCO_DICTS,
            "live": self.live if loop else {}, "history": list(self.history),
            "job": self.jobs.status, "jobs": self.jobs.available(),
        }
        return out


def _jsonable(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def make_handler(wb: Workbench):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *a):
            pass

        def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(json.dumps(obj, default=_jsonable).encode(), "application/json", code)

        def _body(self) -> Dict[str, Any]:
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}") if n else {}

        def do_GET(self):
            url = urllib.parse.urlparse(self.path)
            q = {k: v[-1] for k, v in urllib.parse.parse_qs(url.query).items()}
            try:
                if url.path in ("/", "/index.html"):
                    self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
                elif url.path == "/api/state":
                    self._json(wb.state())
                elif url.path == "/api/cameras":
                    self._json({"cameras": list_cameras()})
                elif url.path == "/frame.jpg":
                    jpeg = wb.loop.preview_jpeg() if wb.loop else None
                    if jpeg is None:
                        self._json({"error": "no frame"}, 404)
                    else:
                        self._send(jpeg, "image/jpeg")
                elif url.path == "/target.png":
                    t = target_from_spec(q["target"]) if "target" in q else wb.target
                    ok, buf = cv2.imencode(".png", t.render(dpi=int(q.get("dpi", 300))))
                    self._send(buf.tobytes(), "image/png")
                elif url.path == "/video":
                    self._video()
                elif url.path == "/clock":
                    self._send(bench.clock_page_html().encode(), "text/html; charset=utf-8")
                else:
                    self._json({"error": "not found"}, 404)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:
                self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

        def do_POST(self):
            path = urllib.parse.urlparse(self.path).path
            try:
                b = self._body()
                if path == "/api/source":
                    wb.set_source(b["uri"])
                    self._json(wb.state())
                elif path == "/api/control":
                    wb.set_control(b["name"], b["value"])
                    self._json({"ok": True})
                elif path == "/api/target":
                    wb.set_target(b.get("spec") or DEFAULT_TARGET, b.get("model"))
                    self._json({"target": wb.target.spec()})
                elif path == "/api/snapshot":
                    self._json({"path": wb.snapshot()})
                elif path == "/api/overlay":
                    self._json(wb.set_overlay(b))
                elif path == "/api/job":
                    self._json(wb.jobs.start(b["kind"], b.get("params") or {}))
                elif path.startswith("/api/calib/"):
                    self._json(wb.calib(path.rsplit("/", 1)[1], b))
                else:
                    self._json({"error": "not found"}, 404)
            except KeyError as exc:
                self._json({"error": f"missing or unknown: {exc}"}, 400)
            except Exception as exc:
                self._json({"error": f"{type(exc).__name__}: {exc}"}, 400)

        def _video(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            loop_id, seq = None, 0
            while True:
                loop = wb.loop
                if loop is None:
                    time.sleep(0.2)
                    continue
                if id(loop) != loop_id:
                    loop_id, seq = id(loop), 0
                frame = loop.wait_frame(after_seq=seq, timeout=1.0)
                if frame is None:
                    # A stopped or ended loop returns at once: don't spin on it.
                    if loop.ended or wb.loop is not loop:
                        time.sleep(0.2)
                    continue
                seq = frame.seq
                jpeg = loop.preview_jpeg()
                if jpeg:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n"
                                     % len(jpeg) + jpeg + b"\r\n")

    return Handler


def make_server(wb: Workbench, listen: str = "127.0.0.1:8000") -> ThreadingHTTPServer:
    host, _, port = listen.rpartition(":")
    srv = ThreadingHTTPServer((host or "127.0.0.1", int(port)), make_handler(wb))
    srv.daemon_threads = True
    return srv


def serve(wb: Workbench, listen: str = "127.0.0.1:8000", open_browser: bool = False) -> None:
    srv = make_server(wb, listen)
    url = f"http://{srv.server_address[0]}:{srv.server_address[1]}/"
    print(f"workbench on {url}", flush=True)
    if open_browser:
        import webbrowser
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        wb.close()
