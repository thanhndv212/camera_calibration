"""The versioned calibration file (``schema_version`` 1).

One JSON document per calibration, holding everything needed to trust it later:
which camera (name + unique id), at what capture settings, against which
target, the model and its parameters, any extrinsics, the errors, the inputs,
and a content hash.

The top-level ``camera_matrix`` / ``dist_coeffs`` / ``image_size`` /
``reprojection_error`` keys are kept in the same shape as the pre-schema files
in ``data/``, so :func:`camera_calibration._shared.load_calibration_parameters`
reads new files unchanged, and :func:`load` reads old files as version 0.

Transforms are named ``T_<to>_<from>``: ``T_base_camera`` maps points in the
camera frame into the robot base frame. Stored as 4x4 row-major, metres.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .sources.base import SourceInfo

SCHEMA_VERSION = 1
MODELS = ("pinhole", "fisheye")

try:
    from importlib.metadata import version as _pkg_version

    TOOL_VERSION = _pkg_version("camera_calibration")
except Exception:  # not installed (running from a checkout)
    TOOL_VERSION = "unknown"


class CalibrationMismatch(ValueError):
    """A calibration file does not belong to the camera/settings it is used with."""


@dataclass
class Calibration:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: Tuple[int, int]
    model: str = "pinhole"
    camera: Dict[str, Any] = field(default_factory=dict)      # SourceInfo.to_dict()
    capture: Dict[str, Any] = field(default_factory=dict)     # fps, codec, focus, exposure...
    target: Dict[str, Any] = field(default_factory=dict)      # target spec
    transforms: Dict[str, np.ndarray] = field(default_factory=dict)  # "T_base_camera": 4x4
    reprojection_error: Optional[float] = None                # RMS px over all views
    per_view_errors: List[float] = field(default_factory=list)
    held_out_error: Optional[float] = None
    inputs: Dict[str, Any] = field(default_factory=dict)      # image paths, pose source...
    created: str = ""
    schema_version: int = SCHEMA_VERSION                      # as read; 0 = pre-schema file
    extra: Dict[str, Any] = field(default_factory=dict)       # unknown keys, kept on round-trip

    def __post_init__(self) -> None:
        self.camera_matrix = np.asarray(self.camera_matrix, np.float64).reshape(3, 3)
        self.dist_coeffs = np.asarray(self.dist_coeffs, np.float64).reshape(-1)
        self.image_size = (int(self.image_size[0]), int(self.image_size[1]))
        if self.model not in MODELS:
            raise ValueError(f"model must be one of {MODELS}, got {self.model!r}")
        for name, T in list(self.transforms.items()):
            T = np.asarray(T, np.float64)
            if T.shape != (4, 4) or not name.startswith("T_"):
                raise ValueError(f"transform {name!r} must be a 4x4 named T_<to>_<from>")
            self.transforms[name] = T

    # -- serialisation --------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,  # writing upgrades a legacy file
            "created": self.created or datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tool_version": TOOL_VERSION,
            "camera": self.camera,
            "capture": self.capture,
            "target": self.target,
            "model": self.model,
            # Legacy-compatible keys, same shapes as data/camera_calibration_*.json.
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": [self.dist_coeffs.tolist()],
            "image_size": list(self.image_size),
            "reprojection_error": self.reprojection_error,
            "per_view_errors": [float(e) for e in self.per_view_errors],
            "held_out_error": self.held_out_error,
            "transforms": {k: v.tolist() for k, v in self.transforms.items()},
            "inputs": self.inputs,
            **self.extra,
        }
        body["hash"] = content_hash(body)
        return body

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Calibration":
        """Read schema 1, or a pre-schema file (reported as ``schema_version`` 0)."""
        if "camera_matrix" not in d or "dist_coeffs" not in d:
            raise ValueError("not a calibration: missing camera_matrix / dist_coeffs")
        version = int(d.get("schema_version", 0))
        if version > SCHEMA_VERSION:
            raise ValueError(f"schema_version {version} is newer than this tool ({SCHEMA_VERSION})")
        size = d.get("image_size") or d.get("resolution")
        if size is None:
            raise ValueError("calibration has no image_size")
        known = {"schema_version", "created", "tool_version", "camera", "capture", "target",
                 "model", "camera_matrix", "dist_coeffs", "image_size", "resolution",
                 "reprojection_error", "per_view_errors", "held_out_error", "transforms",
                 "inputs", "hash", "timestamp"}
        extra = {k: v for k, v in d.items() if k not in known}
        target = dict(d.get("target") or {})
        if version == 0 and "chessboard_size" in d:  # legacy chessboard fields -> target spec
            target = {"type": "chessboard", "inner_corners": d["chessboard_size"],
                      "square_m": d.get("square_size")}
            extra.pop("chessboard_size", None)
            extra.pop("square_size", None)
        return cls(
            camera_matrix=d["camera_matrix"], dist_coeffs=d["dist_coeffs"], image_size=size,
            model=d.get("model", "pinhole"), camera=d.get("camera") or {},
            capture=d.get("capture") or {}, target=target,
            transforms=d.get("transforms") or {},
            reprojection_error=d.get("reprojection_error"),
            per_view_errors=d.get("per_view_errors") or [],
            held_out_error=d.get("held_out_error"),
            inputs=d.get("inputs") or {}, created=d.get("created") or d.get("timestamp", ""),
            schema_version=version, extra=extra,
        )

    # -- use ------------------------------------------------------------------

    def check(self, info: Union[SourceInfo, Dict[str, Any]], *, strict_id: bool = True) -> None:
        """Refuse to be used with a camera or resolution it was not made for.

        Resolution must match exactly (intrinsics scale with it). The camera's
        unique id must match when both sides have one; ``strict_id=False``
        downgrades that to the name check, for cameras whose id changes.
        """
        d = info.to_dict() if isinstance(info, SourceInfo) else info
        res = tuple(d.get("resolution") or ())
        if res and res != self.image_size:
            raise CalibrationMismatch(
                f"calibrated at {self.image_size[0]}x{self.image_size[1]}, "
                f"camera is running at {res[0]}x{res[1]}")
        mine, theirs = self.camera.get("unique_id"), d.get("unique_id")
        if strict_id and mine and theirs and mine != theirs:
            raise CalibrationMismatch(
                f"calibration is for camera id {mine!r} ({self.camera.get('name')}), "
                f"this is {theirs!r} ({d.get('name')})")
        if not (strict_id and mine and theirs):
            name, other = self.camera.get("name"), d.get("name")
            if name and other and name != other:
                raise CalibrationMismatch(f"calibration is for {name!r}, this is {other!r}")

    def transform(self, name: str) -> np.ndarray:
        """A named transform, or its inverse if only that is stored (``T_a_b`` <- ``T_b_a``)."""
        if name in self.transforms:
            return self.transforms[name].copy()
        parts = name.split("_")
        if len(parts) == 3:
            inv = f"T_{parts[2]}_{parts[1]}"
            if inv in self.transforms:
                return np.linalg.inv(self.transforms[inv])
        raise KeyError(f"no transform {name!r} (have {sorted(self.transforms)})")


def content_hash(body: Dict[str, Any]) -> str:
    """sha256 of the document without its ``hash``/``created`` keys, canonical JSON."""
    core = {k: v for k, v in body.items() if k not in ("hash", "created", "tool_version")}
    blob = json.dumps(core, sort_keys=True, separators=(",", ":"), default=float)
    return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()


def save(cal: Calibration, path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cal.to_dict(), indent=2), encoding="utf-8")
    return path


def load(path: Union[str, Path], *, verify_hash: bool = True) -> Calibration:
    """Load a calibration file; a schema-1 file whose hash no longer matches is refused."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if verify_hash and "hash" in d and content_hash(d) != d["hash"]:
        raise ValueError(f"{path}: content does not match its hash (edited by hand?)")
    return Calibration.from_dict(d)
