"""Stand-in for Espressif's ``CameraWebServer`` firmware, for testing without a board.

Serves the same endpoints on the same port layout as the real example:
``/status``, ``/control?var=&val=`` and ``/capture`` on PORT, the MJPEG
``/stream`` on PORT+1. The scene is synthetic: a table, a printed chessboard and
a moving cube carrying a ``DICT_4X4_50`` marker (id 7). Frame rate falls with
resolution, and with ``jitter`` on, requests get Wi-Fi-like latency and
occasional stalls, so benchmarks have something realistic to measure. These are
not measurements of any real camera.

    python -m camera_calibration esp32-mock --port 8080
    python -m camera_calibration workbench --source esp32:127.0.0.1:8080
"""

from __future__ import annotations

import json
import math
import random
import socket
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import List

import cv2
import numpy as np

# framesize -> (width, height, sensor fps). Rough OV2640-over-Wi-Fi figures.
SIZES = {
    5: (240, 240, 25), 6: (320, 240, 25), 8: (400, 296, 22), 9: (480, 320, 20),
    10: (640, 480, 15), 11: (800, 600, 12), 12: (1024, 768, 8), 13: (1280, 720, 7),
    14: (1280, 1024, 5), 15: (1600, 1200, 4),
}

DEFAULT_STATE = {
    "framesize": 10, "quality": 12, "brightness": 0, "contrast": 0, "saturation": 0,
    "special_effect": 0, "wb_mode": 0, "awb": 1, "awb_gain": 1, "aec": 1, "aec2": 0,
    "ae_level": 0, "aec_value": 168, "agc": 1, "agc_gain": 0, "gainceiling": 0,
    "bpc": 0, "wpc": 1, "raw_gma": 1, "lenc": 1, "hmirror": 0, "vflip": 0, "dcw": 1,
    "colorbar": 0, "led_intensity": 0,
}
_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
_MARKER = cv2.aruco.generateImageMarker(_DICT, 7, 200)


def _chessboard(cols=10, rows=7, sq=24) -> np.ndarray:
    board = np.indices((rows, cols)).sum(axis=0) % 2 * 255
    board = np.kron(board, np.ones((sq, sq))).astype(np.uint8)
    return cv2.copyMakeBorder(board, sq, sq, sq, sq, cv2.BORDER_CONSTANT, value=255)


_BOARD = _chessboard()


def _paste_warped(img, tile, quad):
    h, w = tile.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(src, np.float32(quad))
    size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(tile, M, size)
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, size)
    img[mask > 0] = warped[mask > 0]


def render(t: float, st: dict) -> np.ndarray:
    """One frame of the synthetic scene under sensor settings ``st``."""
    w, h, _ = SIZES.get(st["framesize"], SIZES[10])
    if st["colorbar"]:
        bars = np.array([[255, 255, 255], [0, 255, 255], [255, 255, 0], [0, 255, 0],
                         [255, 0, 255], [0, 0, 255], [255, 0, 0], [0, 0, 0]], np.uint8)
        return np.repeat(bars[None, np.arange(w) * 8 // w], h, axis=0)

    # Wooden-ish table with a vignette (lens shading, unless lenc is on).
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.zeros((h, w, 3), np.float32)
    img[..., 0], img[..., 1], img[..., 2] = 95, 130, 165
    img += (np.sin(xx / w * 40 + np.sin(yy / h * 6) * 2) * 6)[..., None]
    if not st["lenc"]:
        r = np.hypot((xx - w / 2) / w, (yy - h / 2) / h)
        img *= (1 - 0.6 * r ** 2)[..., None]
    img = img.astype(np.uint8)

    sx, sy = w / 640, h / 480
    s = min(sx, sy)
    board = cv2.cvtColor(_BOARD, cv2.COLOR_GRAY2BGR)
    _paste_warped(img, board, [(40 * sx, 250 * sy), (250 * sx, 240 * sy),
                               (262 * sx, 400 * sy), (30 * sx, 412 * sy)])

    # Cube moving on an ellipse, its top face carrying the marker.
    cx = (440 + 90 * math.cos(t * 0.6)) * sx
    cy = (220 + 60 * math.sin(t * 0.6)) * sy
    half, ang = 52 * s, t * 0.4
    face = [(cx + half * math.cos(ang + k * math.pi / 2 + math.pi / 4) * 1.41,
             cy + half * math.sin(ang + k * math.pi / 2 + math.pi / 4) * 1.41 * 0.85)
            for k in range(4)]
    shadow = np.int32([(x + 10 * s, y + 14 * s) for x, y in face])
    cv2.fillConvexPoly(img, shadow, (55, 70, 85), cv2.LINE_AA)
    tag = cv2.copyMakeBorder(_MARKER, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
    _paste_warped(img, cv2.cvtColor(tag, cv2.COLOR_GRAY2BGR), face)

    # Sensor controls.
    f = img.astype(np.float32)
    f = (f - 128) * (1 + 0.25 * st["contrast"]) + 128 + 20 * st["brightness"]
    f += 18 * st["ae_level"] if st["aec"] else (st["aec_value"] - 300) / 12
    gray = f.mean(axis=2, keepdims=True)
    f = gray + (f - gray) * (1 + 0.35 * st["saturation"])
    gain = 1 + (st["agc_gain"] / 10 if not st["agc"] else 0.3)
    f += np.random.normal(0, 2.5 * gain, f.shape)
    img = np.clip(f, 0, 255).astype(np.uint8)
    effect = st["special_effect"]
    if effect == 1:
        img = 255 - img
    elif effect == 2:
        img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    elif effect in (3, 4, 5):
        tint = np.zeros_like(img)
        tint[..., {3: 2, 4: 1, 5: 0}[effect]] = 255
        img = cv2.addWeighted(img, 0.6, tint, 0.4, 0)
    elif effect == 6:
        img = cv2.transform(img, np.array([[.131, .534, .272], [.168, .686, .349],
                                           [.189, .769, .393]])).clip(0, 255).astype(np.uint8)
    if st["hmirror"]:
        img = img[:, ::-1]
    if st["vflip"]:
        img = img[::-1]
    return np.ascontiguousarray(img)


def encode(img, q: int) -> bytes:
    # q  # esp32-camera: 0-63, lower is better
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(max(5, 100 - q * 1.5))])[1].tobytes()


def wifi_delay(base: float = 0.015) -> None:
    """Wi-Fi-like latency: exponential jitter and a 1% chance of a 250-600 ms stall."""
    d = base + random.expovariate(1 / 0.012)
    if random.random() < 0.01:
        d += random.uniform(0.25, 0.6)  # occasional Wi-Fi stall
    time.sleep(d)



class MockCameraWebServer:
    """Control server on ``port`` and stream server on ``port + 1``, on threads.

    ``port=0`` picks a free pair. ``jitter=False`` drops the simulated Wi-Fi
    latency and stalls, for deterministic tests.
    """

    BOUNDARY = b"123456789000000000000987654321"

    def __init__(self, bind: str = "127.0.0.1", port: int = 8080, jitter: bool = True) -> None:
        self.bind, self.port, self.jitter = bind, port, jitter
        self.state = dict(DEFAULT_STATE)
        self.lock = threading.Lock()
        self._servers: List[ThreadingHTTPServer] = []

    @property
    def host(self) -> str:
        """``host:port`` as an ``esp32:`` URI takes it."""
        return f"{self.bind}:{self.port}"

    def snapshot_state(self) -> dict:
        with self.lock:
            return dict(self.state)

    def _delay(self, base: float) -> None:
        if self.jitter:
            wifi_delay(base)

    def start(self) -> "MockCameraWebServer":
        for _ in range(50):
            port = self.port or _free_port(self.bind)
            try:
                servers = [ThreadingHTTPServer((self.bind, port), _control_handler(self))]
                try:
                    servers.append(ThreadingHTTPServer((self.bind, port + 1), _stream_handler(self)))
                except OSError:
                    servers[0].server_close()
                    raise
            except OSError:
                if self.port:
                    raise
                continue
            self.port, self._servers = port, servers
            for srv in servers:
                srv.daemon_threads = True
                threading.Thread(target=srv.serve_forever, daemon=True).start()
            return self
        raise OSError("no free port pair")

    def stop(self) -> None:
        for srv in self._servers:
            srv.shutdown()
            srv.server_close()
        self._servers = []

    def __enter__(self) -> "MockCameraWebServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def _free_port(bind: str) -> int:
    with socket.socket() as s:
        s.bind((bind, 0))
        return s.getsockname()[1]


class _Quiet(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass


def _control_handler(cam: MockCameraWebServer):
    class Control(_Quiet):
        def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            q = dict(urllib.parse.parse_qsl(u.query))
            if u.path == "/status":
                self._send(json.dumps(cam.snapshot_state()).encode(), "application/json")
            elif u.path == "/control":
                var, val = q.get("var"), q.get("val")
                if (var not in cam.state or val is None
                        or (var == "framesize" and int(val) not in SIZES)):
                    self._send(b"", "text/plain", 500)
                    return
                with cam.lock:
                    cam.state[var] = int(val)
                self._send(b"", "text/plain")
            elif u.path == "/capture":
                st = cam.snapshot_state()
                cam._delay(0.02 + SIZES[st["framesize"]][0] * 1e-4)
                self._send(encode(render(time.monotonic(), st), st["quality"]), "image/jpeg")
            else:
                self._send(b"not found", "text/plain", 404)

    return Control


def _stream_handler(cam: MockCameraWebServer):
    class Stream(_Quiet):
        def do_GET(self):
            if not self.path.startswith("/stream"):
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace;boundary=" + cam.BOUNDARY.decode())
            self.end_headers()
            try:
                while cam._servers:
                    t0 = time.monotonic()
                    st = cam.snapshot_state()
                    jpeg = encode(render(t0, st), st["quality"])
                    self.wfile.write(b"\r\n--" + cam.BOUNDARY + b"\r\n"
                                     b"Content-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n"
                                     % len(jpeg) + jpeg)
                    cam._delay(0)
                    fps = SIZES.get(st["framesize"], SIZES[10])[2]
                    time.sleep(max(0, 1 / fps - (time.monotonic() - t0)))
            except OSError:
                return

    return Stream


def serve(bind: str = "127.0.0.1", port: int = 8080, jitter: bool = True) -> None:
    """Run until Ctrl-C (the ``esp32-mock`` subcommand)."""
    cam = MockCameraWebServer(bind, port, jitter).start()
    print(f"mock CameraWebServer: http://{cam.host}  stream http://{bind}:{cam.port + 1}/stream"
          f"  (use --source esp32:{cam.host})", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
