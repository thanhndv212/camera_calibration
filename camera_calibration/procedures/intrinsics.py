"""Intrinsic calibration (pinhole or fisheye) with coverage feedback and a held-out check.

Every fifth accepted view is held out of the solve and used afterwards: the
board pose is fitted with the solved intrinsics (``solvePnP``) and the RMS
reprojection error over those views is reported. A model that only fits its
own views shows up there, where the training RMS hides it.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..artifact import Calibration
from ..sources.base import Frame, SourceEnded, SourceInfo
from ..targets.base import Detection, Target

GRID = (8, 6)  # coverage cells (cols, rows)


@dataclass
class Sample:
    img_pts: np.ndarray
    obj_pts: np.ndarray
    ids: np.ndarray
    descriptor: np.ndarray
    held_out: bool = False
    path: Optional[str] = None
    t: float = 0.0


@dataclass
class IntrinsicsResult:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: Tuple[int, int]
    model: str
    rms: float
    per_view_errors: List[float]           # px, one per training sample, in sample order
    held_out_error: Optional[float]        # px RMS over held-out views, None if none
    held_out_per_view: List[float] = field(default_factory=list)
    train_index: List[int] = field(default_factory=list)  # sample index of each per_view error

    def to_dict(self) -> Dict[str, Any]:
        K = self.camera_matrix
        return {"model": self.model, "rms": self.rms, "held_out_error": self.held_out_error,
                "fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2],
                "dist_coeffs": self.dist_coeffs.ravel().tolist(),
                "per_view_errors": self.per_view_errors, "train_index": self.train_index,
                "held_out_per_view": self.held_out_per_view,
                "image_size": list(self.image_size)}


def descriptor(pts: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Where the board is, how big, and how foreshortened, in image-relative units.

    Two views close in this space add little to a calibration. Anisotropy and
    its orientation stand in for tilt, which cannot be measured before the
    intrinsics are known.
    """
    w, h = size
    c = pts.mean(axis=0)
    cov = np.cov((pts - c).T) if len(pts) > 2 else np.eye(2)
    evals, evecs = np.linalg.eigh(cov)
    evals = np.maximum(evals, 1e-9)
    extent = float(np.sqrt(evals.sum())) / w
    aniso = float(1 - np.sqrt(evals[0] / evals[1]))
    ang = float(np.arctan2(evecs[1, 1], evecs[0, 1]))
    return np.array([c[0] / w, c[1] / h, 2 * extent,
                     aniso * np.cos(2 * ang), aniso * np.sin(2 * ang)])


class IntrinsicsProcedure:
    """Collect views of *target* and solve intrinsics for one camera at one resolution.

    Call :meth:`observe` on every analysed frame. With ``auto`` on it accepts a
    view once the board has been steady for ``steady_s`` and differs from every
    sample so far by ``novelty``; :meth:`capture` takes the current view anyway.
    """

    def __init__(self, target: Target, image_size: Tuple[int, int], *, model: str = "pinhole",
                 auto: bool = True, steady_s: float = 0.4, steady_px: float = 1.5,
                 novelty: float = 0.08, hold_out_every: int = 5,
                 sample_dir: Optional[Path] = None) -> None:
        if model not in ("pinhole", "fisheye"):
            raise ValueError("model must be pinhole or fisheye")
        self.target, self.image_size, self.model = target, tuple(image_size), model
        self.auto, self.steady_s, self.novelty = auto, steady_s, novelty
        # Steadiness threshold scales with resolution: 1.5 px means the same at VGA and 1080p.
        self.steady_px = steady_px * self.image_size[0] / 640
        self.hold_out_every = hold_out_every
        self.sample_dir = Path(sample_dir) if sample_dir else None
        self.samples: List[Sample] = []
        self.result: Optional[IntrinsicsResult] = None
        self.coverage = np.zeros(GRID[::-1], np.int32)  # rows, cols
        self._lock = threading.RLock()
        self._prev: Optional[Detection] = None
        self._steady_since: Optional[float] = None
        self._last: Optional[Tuple[Frame, Optional[Detection]]] = None
        self._feedback: Dict[str, Any] = {}

    # -- live -----------------------------------------------------------------

    def observe(self, frame: Frame, det: Optional[Detection]) -> Dict[str, Any]:
        """Update feedback for one analysed frame; auto-accept if it qualifies."""
        with self._lock:
            self._last = (frame, det)
            fb: Dict[str, Any] = {"points": 0, "steady": False, "new": False, "accepted": False}
            if frame.img.shape[1::-1] != self.image_size:
                fb["hint"] = (f"camera is at {frame.img.shape[1]}x{frame.img.shape[0]}, "
                              f"calibration at {self.image_size[0]}x{self.image_size[1]}")
                self._feedback = fb
                return fb
            if det is None or len(det) < self.target.min_points:
                self._prev, self._steady_since = None, None
                fb["hint"] = "show the board"
                self._feedback = fb
                return fb
            fb["points"] = len(det)
            moved = self._motion(det)
            now = frame.t_host
            if moved is None or moved > self.steady_px:
                self._steady_since = now
            self._prev = det
            fb["motion_px"] = None if moved is None else round(moved, 2)
            fb["steady"] = self._steady_since is not None and now - self._steady_since >= self.steady_s
            d = descriptor(det.img_pts, self.image_size)
            dist = self._nearest(d)
            fb["novelty"] = None if dist is None else round(dist, 3)
            fb["new"] = dist is None or dist >= self.novelty
            fb["hint"] = self._hint(fb)
            if self.auto and fb["steady"] and fb["new"]:
                self._accept(frame, det, d)
                fb["accepted"] = True
                self._steady_since = now  # require a fresh steady period for the next one
            self._feedback = fb
            return fb

    def capture(self) -> Sample:
        """Accept the last observed view now (manual capture)."""
        with self._lock:
            if self._last is None or self._last[1] is None:
                raise RuntimeError("no board in view")
            frame, det = self._last
            if len(det) < self.target.min_points:
                raise RuntimeError(f"need at least {self.target.min_points} corners")
            return self._accept(frame, det, descriptor(det.img_pts, self.image_size))

    def _motion(self, det: Detection) -> Optional[float]:
        if self._prev is None:
            return None
        common, a, b = np.intersect1d(self._prev.ids, det.ids, return_indices=True)
        if len(common) < 4:
            return None
        return float(np.linalg.norm(self._prev.img_pts[a] - det.img_pts[b], axis=1).mean())

    def _nearest(self, d: np.ndarray) -> Optional[float]:
        if not self.samples:
            return None
        return float(min(np.linalg.norm(s.descriptor - d) for s in self.samples))

    def _hint(self, fb: Dict[str, Any]) -> str:
        if not fb["new"]:
            empty = np.argwhere(self.coverage == 0)
            if len(empty):
                r, c = empty[len(empty) // 2]
                side = "left" if c < GRID[0] / 3 else "right" if c >= 2 * GRID[0] / 3 else "centre"
                vert = "top" if r < GRID[1] / 3 else "bottom" if r >= 2 * GRID[1] / 3 else "middle"
                return f"seen this view; move the board to the {vert} {side}, or tilt it"
            return "seen this view; tilt the board or change distance"
        if not fb["steady"]:
            return "hold still"
        return "capturing"

    def _accept(self, frame: Frame, det: Detection, d: np.ndarray) -> Sample:
        n = len(self.samples)
        held = self.hold_out_every > 0 and n % self.hold_out_every == self.hold_out_every - 1
        path = None
        if self.sample_dir is not None:
            self.sample_dir.mkdir(parents=True, exist_ok=True)
            p = self.sample_dir / f"{n:04d}.png"  # lossless, full resolution
            cv2.imwrite(str(p), frame.img)
            path = str(p)
        s = Sample(det.img_pts.astype(np.float32), det.obj_pts.astype(np.float32), det.ids,
                   d, held, path, time.time())
        self.samples.append(s)
        self._add_coverage(s.img_pts)
        self.result = None
        return s

    def _add_coverage(self, pts: np.ndarray) -> None:
        w, h = self.image_size
        cols = np.clip((pts[:, 0] / w * GRID[0]).astype(int), 0, GRID[0] - 1)
        rows = np.clip((pts[:, 1] / h * GRID[1]).astype(int), 0, GRID[1] - 1)
        np.add.at(self.coverage, (rows, cols), 1)

    def drop(self, index: int) -> None:
        """Remove a sample (an outlier from the per-view errors) and rebuild coverage."""
        with self._lock:
            del self.samples[index]
            self.coverage[:] = 0
            for s in self.samples:
                self._add_coverage(s.img_pts)
            self.result = None

    def reset(self) -> None:
        with self._lock:
            self.samples.clear()
            self.coverage[:] = 0
            self.result = None

    def state(self) -> Dict[str, Any]:
        with self._lock:
            covered = float((self.coverage > 0).mean())
            return {
                "target": self.target.spec(), "model": self.model, "auto": self.auto,
                "image_size": list(self.image_size),
                "samples": len(self.samples),
                "train": sum(not s.held_out for s in self.samples),
                "held_out": sum(s.held_out for s in self.samples),
                "coverage": self.coverage.tolist(), "coverage_frac": round(covered, 3),
                "feedback": self._feedback,
                "result": self.result.to_dict() if self.result else None,
                "ready": sum(not s.held_out for s in self.samples) >= 6,
            }

    # -- solve ----------------------------------------------------------------

    def solve(self) -> IntrinsicsResult:
        with self._lock:
            samples = list(self.samples)
        train = [(i, s) for i, s in enumerate(samples) if not s.held_out]
        test = [s for s in samples if s.held_out]
        if len(train) < 6:
            raise RuntimeError(f"need at least 6 training views, have {len(train)}")
        obj = [s.obj_pts for _, s in train]
        img = [s.img_pts for _, s in train]
        if self.model == "pinhole":
            rms, K, D, rvecs, tvecs, _, _, per_view = cv2.calibrateCameraExtended(
                obj, img, self.image_size, None, None)
            per_view = per_view.ravel().tolist()
        else:
            K, D, rms, per_view = _fisheye(obj, img, self.image_size)
        held_per_view = [_pnp_rms(s, K, D, self.model) for s in test]
        held = (float(np.sqrt(np.mean(np.square(held_per_view)))) if held_per_view else None)
        res = IntrinsicsResult(K, D.ravel(), self.image_size, self.model, float(rms),
                               [float(e) for e in per_view], held,
                               [float(e) for e in held_per_view], [i for i, _ in train])
        with self._lock:
            self.result = res
        return res

    def artifact(self, info: Optional[SourceInfo] = None, capture: Optional[dict] = None) -> Calibration:
        """The calibration file for the last solve."""
        if self.result is None:
            raise RuntimeError("solve first")
        r = self.result
        return Calibration(
            camera_matrix=r.camera_matrix, dist_coeffs=r.dist_coeffs, image_size=r.image_size,
            model=r.model, camera=info.to_dict() if info else {}, capture=capture or {},
            target=self.target.spec(), reprojection_error=r.rms,
            per_view_errors=r.per_view_errors, held_out_error=r.held_out_error,
            inputs={"samples": [s.path for s in self.samples if s.path],
                    "held_out": [i for i, s in enumerate(self.samples) if s.held_out],
                    "views": len(self.samples)},
        )


def _fisheye(obj, img, size):
    o = [p.reshape(-1, 1, 3).astype(np.float64) for p in obj]
    i = [p.reshape(-1, 1, 2).astype(np.float64) for p in img]
    flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
    rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        o, i, size, np.eye(3), np.zeros((4, 1)), flags=flags,
        criteria=(cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 100, 1e-7))
    per_view = []
    for oo, ii, r, t in zip(o, i, rvecs, tvecs):
        proj, _ = cv2.fisheye.projectPoints(oo, r, t, K, D)
        per_view.append(float(np.sqrt(np.mean(np.sum((proj - ii) ** 2, axis=2)))))
    return K, D, rms, per_view


def _pnp_rms(s: Sample, K: np.ndarray, D: np.ndarray, model: str) -> float:
    """Reprojection RMS of one view with the board pose fitted using K, D."""
    if model == "fisheye":
        und = cv2.fisheye.undistortPoints(s.img_pts.reshape(-1, 1, 2).astype(np.float64), K, D)
        ok, r, t = cv2.solvePnP(s.obj_pts, und, np.eye(3), None)
        proj, _ = cv2.fisheye.projectPoints(s.obj_pts.reshape(-1, 1, 3).astype(np.float64), r, t, K, D)
    else:
        ok, r, t = cv2.solvePnP(s.obj_pts, s.img_pts, K, D)
        proj, _ = cv2.projectPoints(s.obj_pts, r, t, K, D)
    return float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - s.img_pts) ** 2, axis=1))))


def solve_from_source(source, target: Target, *, model: str = "pinhole",
                      hold_out_every: int = 5, max_frames: int = 0) -> Tuple[IntrinsicsProcedure, SourceInfo]:
    """Headless solve: take every frame of a finite source (``dir:``) as a sample."""
    info = source.open()
    proc = IntrinsicsProcedure(target, info.resolution, model=model, auto=False,
                               hold_out_every=hold_out_every)
    try:
        n = 0
        while not max_frames or n < max_frames:
            try:
                frame = source.read()
            except SourceEnded:
                break
            n += 1
            det = target.detect(cv2.cvtColor(frame.img, cv2.COLOR_BGR2GRAY))
            proc.observe(frame, det)
            if det is not None and len(det) >= target.min_points:
                s = proc.capture()
                s.path = frame.path
    finally:
        source.close()
    proc.solve()
    return proc, info
