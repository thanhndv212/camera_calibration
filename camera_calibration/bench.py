"""Camera benchmarks: frame rate, jitter, stalls, image quality and glass-to-host lag.

Ported from the ESP32 evaluation probe so any source can be measured the same way:

- :func:`stream_bench` measures a running :class:`~camera_calibration.capture.CaptureLoop`.
  Timing comes from the loop's per-frame timeline, which the capture thread records,
  so analysing frames here never lowers the measured rate.
- :func:`snapshot_bench` times ESP32 CameraWebServer ``/capture`` stills.
- :func:`control_sweep` repeats either at each value of a source control
  (``framesize`` on the ESP32), then restores the original value.
- :func:`clock_page_html` is a screen clock: with it in view, :func:`stream_bench`
  also reports glass-to-host lag.

Every function returns plain JSON-able dicts; :func:`summarize` gives
n/mean/p50/p95/max/stdev for a list of numbers.
"""

from __future__ import annotations

import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np

from .capture import STALL_MS

CLOCK_PERIOD_MS = 100
ARUCO_DICTS = ["DICT_4X4_50", "DICT_5X5_100", "DICT_6X6_250", "DICT_7X7_1000", "DICT_ARUCO_ORIGINAL"]
FRAMESIZE_NAMES = {
    0: "96X96", 1: "QQVGA", 2: "128X128", 3: "QCIF", 4: "HQVGA", 5: "240X240",
    6: "QVGA", 7: "320X320", 8: "CIF", 9: "HVGA", 10: "VGA", 11: "SVGA",
    12: "XGA", 13: "HD", 14: "SXGA", 15: "UXGA",
}

Progress = Optional[Callable[[float], None]]


# ------------------------------------------------------------------------ metrics


def aruco_detector(dict_name: str = "DICT_4X4_50"):
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    return cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())


def summarize(values: List[float]) -> Optional[Dict[str, float]]:
    if not values:
        return None
    v = sorted(values)
    return {
        "n": len(v),
        "mean": statistics.fmean(v),
        "p50": v[len(v) // 2],
        "p95": v[min(len(v) - 1, int(round(0.95 * (len(v) - 1))))],
        "max": v[-1],
        "stdev": statistics.pstdev(v),
    }


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def frame_metrics(img: np.ndarray, aruco=None, target=None) -> Dict[str, Any]:
    """Quality of one frame, plus ArUco markers and target corners when asked.

    ``target`` is a :class:`~camera_calibration.targets.Target`; ``target_points``
    is how many of its corners were found (0 when none).
    """
    gray = _gray(img)
    m: Dict[str, Any] = {
        "width": int(img.shape[1]),
        "height": int(img.shape[0]),
        # Variance of the Laplacian: higher = sharper. Compare between settings
        # on the same scene, not as an absolute number.
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "brightness": float(gray.mean()),
        "clipped_pct": float(((gray <= 2) | (gray >= 253)).mean() * 100),
    }
    if aruco is not None:
        corners, ids, _ = aruco.detectMarkers(gray)
        m["aruco_ids"] = [] if ids is None else sorted(int(i) for i in ids.ravel())
        # Smallest marker edge in pixels — the number that limits PnP accuracy.
        m["aruco_min_edge_px"] = (
            float(min(np.linalg.norm(c[0][0] - c[0][1]) for c in corners)) if corners else None
        )
    if target is not None:
        det = target.detect(gray)
        m["target_points"] = 0 if det is None else len(det.ids)
    return m


def temporal_noise(frames: List[np.ndarray]) -> Optional[float]:
    """Mean per-pixel std over a static scene (grey levels). Needs >=5 frames."""
    if len(frames) < 5 or len({f.shape[:2] for f in frames}) != 1:
        return None
    stack = np.stack([_gray(f).astype(np.float32) for f in frames])
    return float(stack.std(axis=0).mean())


def aggregate(per_frame: List[dict], frames: List[np.ndarray]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "resolutions": sorted({f"{m['width']}x{m['height']}" for m in per_frame}),
        "sharpness": summarize([m["sharpness"] for m in per_frame]),
        "brightness": summarize([m["brightness"] for m in per_frame]),
        "clipped_pct": summarize([m["clipped_pct"] for m in per_frame]),
        "temporal_noise": temporal_noise(frames),
    }
    if per_frame and "aruco_ids" in per_frame[0]:
        hits = [m for m in per_frame if m["aruco_ids"]]
        out["aruco_detect_rate"] = len(hits) / len(per_frame)
        out["aruco_ids_seen"] = sorted({i for m in hits for i in m["aruco_ids"]})
        out["aruco_min_edge_px"] = summarize(
            [m["aruco_min_edge_px"] for m in hits if m["aruco_min_edge_px"]])
    if per_frame and "target_points" in per_frame[0]:
        out["target_detect_rate"] = sum(m["target_points"] > 0 for m in per_frame) / len(per_frame)
        out["target_points"] = summarize([m["target_points"] for m in per_frame])
    return out


# -------------------------------------------------------------------- lag clock


def clock_ids(dict_name: str) -> int:
    """How many IDs the lag clock cycles through: the dictionary size, capped at 50."""
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    return min(50, d.bytesList.shape[0])


def clock_page_html(dict_name: str = "DICT_4X4_50", period_ms: int = CLOCK_PERIOD_MS,
                    n_ids: Optional[int] = None) -> str:
    """A page that shows marker ``floor(now / period) % n_ids``, from the host's clock.

    Open it on a screen driven by the machine that runs the benchmark and point
    the camera at it: the marker ID in each received frame then says when that
    frame's light left the screen, so receipt time minus that is glass-to-host lag.
    """
    n_ids = clock_ids(dict_name) if n_ids is None else n_ids
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    side = d.markerSize + 2  # one black border cell each side
    bits = [(cv2.aruco.generateImageMarker(d, i, side) > 127).astype(int).ravel().tolist()
            for i in range(n_ids)]
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Camera lag clock</title>
<style>html,body{{margin:0;height:100%;background:#fff;color:#000;font:14px system-ui}}
body{{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px}}
canvas{{width:min(70vmin,560px);height:min(70vmin,560px);image-rendering:pixelated}}</style></head>
<body><canvas id="c" width="{side + 2}" height="{side + 2}"></canvas>
<div id="t">{dict_name}, one ID per {period_ms} ms, IDs 0-{n_ids - 1}. Keep only this marker in view.</div>
<script>
const BITS={json.dumps(bits)}, SIDE={side}, PERIOD={period_ms}, N={n_ids};
const ctx=document.getElementById('c').getContext('2d'); let shown=-1;
function draw(){{
  const id=Math.floor(Date.now()/PERIOD)%N;
  if(id!==shown){{
    ctx.fillStyle='#fff'; ctx.fillRect(0,0,SIDE+2,SIDE+2); ctx.fillStyle='#000';
    BITS[id].forEach((b,k)=>{{ if(!b) ctx.fillRect(1+k%SIDE,1+Math.floor(k/SIDE),1,1); }});
    shown=id;
  }}
  requestAnimationFrame(draw);
}}
draw();
</script></body></html>
"""


def clock_lag_ms(t_wall: float, marker_id: int, period_s: float, n_ids: int) -> float:
    """Lag of a frame that shows clock marker ``marker_id``, received at ``t_wall``.

    The screen showed that ID during the most recent slot ``s <= now`` with
    ``s % n_ids == marker_id``. The camera sampled somewhere inside the slot, so
    the slot midpoint is the estimate; each sample is quantised to +-period/2,
    the median over many frames is not. Includes the screen's own display delay.
    """
    now_slot = int(t_wall // period_s)
    slot = now_slot - (now_slot - marker_id) % n_ids
    return (t_wall - (slot + 0.5) * period_s) * 1000


def make_clock(dict_name: str = "DICT_4X4_50", period_ms: int = CLOCK_PERIOD_MS) -> dict:
    """The ``clock`` argument of :func:`stream_bench`."""
    return {"detector": aruco_detector(dict_name), "dict": dict_name,
            "period_ms": period_ms, "n_ids": clock_ids(dict_name)}


# ------------------------------------------------------------------- benchmarks


def _save(save: Optional[Path], name: str, img: np.ndarray) -> None:
    if save is not None:
        save.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save / name), img)


def stream_bench(loop, duration: float, *, warmup: float = 1.0, aruco=None, target=None,
                 clock: Optional[dict] = None, save: Optional[Path] = None, tag: str = "stream",
                 progress: Progress = None, sample_every_s: float = 0.2) -> Dict[str, Any]:
    """Measure a running ``CaptureLoop`` for ``duration`` seconds after ``warmup``.

    Rate, inter-frame gaps, stalls and JPEG size cover *every* frame the source
    delivered (from the capture thread's timeline). Image quality, ArUco and
    target detection are sampled every ``sample_every_s``; lag (``clock`` from
    :func:`make_clock`) is measured on every frame this function gets to see.
    """
    wall_offset = time.time() - time.monotonic()
    t_begin = time.monotonic()
    t_start, t_end = t_begin + warmup, t_begin + warmup + duration
    first_seq: Optional[int] = None
    seq, next_sample = 0, 0.0
    per_frame: List[dict] = []
    frames: List[np.ndarray] = []
    lags: List[float] = []
    ids_seen: List[int] = []
    errors: List[str] = []
    seen = 0
    while time.monotonic() < t_end:
        fr = loop.wait_frame(after_seq=seq, timeout=min(2.0, max(0.1, t_end - time.monotonic())))
        if fr is None:
            if loop.ended:
                errors.append("source ended")
                break
            continue
        seq = fr.seq
        if fr.t_host < t_start:
            continue
        if first_seq is None:
            first_seq = fr.seq - 1  # timeline entries *after* this belong to the run
        seen += 1
        if clock is not None:
            _, ids, _ = clock["detector"].detectMarkers(_gray(fr.img))
            if ids is not None and len(ids) == 1:
                ids_seen.append(int(ids[0][0]))
                lags.append(clock_lag_ms(fr.t_host + wall_offset, int(ids[0][0]),
                                         clock["period_ms"] / 1000, clock["n_ids"]))
        if fr.t_host >= next_sample:
            next_sample = fr.t_host + sample_every_s
            per_frame.append(frame_metrics(fr.img, aruco, target))
            if len(frames) < 15:
                frames.append(_gray(fr.img).copy())
            if len(per_frame) <= 5:
                _save(save, f"{tag}_{len(per_frame) - 1:02d}.jpg", fr.img)
        if progress is not None:
            progress(min(1.0, (fr.t_host - t_start) / duration))

    timeline = [e for e in loop.timeline(first_seq or 0) if e[1] <= t_end] if first_seq is not None else []
    stamps = [t for _, t, _ in timeline]
    seqs = [s for s, _, _ in timeline]
    # Gaps only between consecutive frames: a seq jump means the source itself skipped.
    gaps = [(b - a) * 1000 for (sa, a), (sb, b) in zip(zip(seqs, stamps), zip(seqs[1:], stamps[1:]))
            if sb == sa + 1]
    span = stamps[-1] - stamps[0] if len(stamps) > 1 else 0.0
    sizes = [n / 1024 for _, _, n in timeline if n]
    out: Dict[str, Any] = {
        "mode": "stream",
        "duration_s": span,
        "frames": len(timeline),
        "fps": (len(timeline) - 1) / span if span else None,
        "interframe_ms": summarize(gaps),
        "stalls_over_250ms": sum(g > STALL_MS for g in gaps),
        "source_gaps": (seqs[-1] - seqs[0] + 1 - len(seqs)) if seqs else 0,
        "jpeg_kb": summarize(sizes),
        "frames_analysed": seen,
        "errors": errors[:10],
        "error_count": len(errors),
        **aggregate(per_frame, frames),
    }
    if clock is not None:
        out["clock_detect_rate"] = len(ids_seen) / seen if seen else 0.0
        # A printed or stuck marker also decodes as one ID; only a running clock
        # cycles through many. Without that, the "lag" would be meaningless.
        expected = min(clock["n_ids"], span * 1000 / clock["period_ms"], len(ids_seen))
        out["clock_ids_distinct"] = len(set(ids_seen))
        out["clock_ok"] = bool(ids_seen) and len(set(ids_seen)) >= max(3, 0.3 * expected)
        out["lag_ms"] = summarize(lags) if out["clock_ok"] else None
    return out


def http_capture(host: str, timeout: float) -> tuple:
    """One ``/capture`` still JPEG from CameraWebServer plus its round-trip seconds."""
    t0 = time.perf_counter()
    url = f"http://{host}/capture?_={time.monotonic_ns()}"
    with urllib.request.urlopen(url, timeout=timeout) as r:
        data = r.read()
    return data, time.perf_counter() - t0


def snapshot_bench(host: str, n: int, *, timeout: float = 5.0, aruco=None, target=None,
                   save: Optional[Path] = None, tag: str = "capture",
                   progress: Progress = None) -> Dict[str, Any]:
    """``n`` CameraWebServer ``/capture`` stills: latency, size and quality.

    Run it with the stream closed; the camera serves both from one sensor.
    """
    latencies, sizes, per_frame, frames, errors = [], [], [], [], []
    for i in range(n):
        if progress is not None:
            progress(i / n)
        try:
            jpeg, dt = http_capture(host, timeout)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            errors.append(repr(e))
            continue
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            errors.append("jpeg decode failed")
            continue
        latencies.append(dt * 1000)
        sizes.append(len(jpeg) / 1024)
        per_frame.append(frame_metrics(img, aruco, target))
        if len(frames) < 30:
            frames.append(_gray(img))
        if i < 5:
            _save(save, f"{tag}_{i:02d}.jpg", img)
    return {
        "mode": "snapshot",
        "requested": n,
        "ok": len(latencies),
        "errors": errors[:10],
        "error_count": len(errors),
        "latency_ms": summarize(latencies),
        "jpeg_kb": summarize(sizes),
        "effective_fps": (1000 / statistics.fmean(latencies)) if latencies else None,
        **aggregate(per_frame, frames),
    }


def control_sweep(source, name: str, values: List[Any], run: Callable[[Any, Progress], dict],
                  *, settle: float = 1.0, progress: Progress = None) -> List[Dict[str, Any]]:
    """``run(value, progress)`` at each value of control ``name``, then restore it.

    One failing value is recorded as ``{"value", "error"}`` and the sweep goes on.
    """
    current = {c.name: c.value for c in source.controls()}
    original = current.get(name)
    results = []
    try:
        for k, v in enumerate(values):
            def sub(f: float, k=k) -> None:
                if progress is not None:
                    progress((k + f) / len(values))
            sub(0.0)
            try:
                source.set_control(name, v)
                time.sleep(settle)
                r = run(v, sub)
            except Exception as e:
                results.append({name: v, "error": f"{type(e).__name__}: {e}"})
                continue
            row = {name: v, **r}
            if name == "framesize":
                row["name"] = FRAMESIZE_NAMES.get(int(v), "?")
            results.append(row)
    finally:
        if original is not None:
            source.set_control(name, original)
    return results
