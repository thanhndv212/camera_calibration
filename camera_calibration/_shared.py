"""Shared helpers for camera calibration scripts."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

# Data directory: <project-root>/data/  (one level above the package dir)
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

DEFAULT_RESOLUTION: Tuple[int, int] = (1280, 720)
DEFAULT_FOV_DEGREES: float = 60.0

_SLUG_PATTERN = re.compile(r"[^0-9a-zA-Z]+")

__all__ = [
    "DATA_DIR",
    "DEFAULT_RESOLUTION",
    "DEFAULT_FOV_DEGREES",
    "parse_resolution",
    "load_calibration_parameters",
    "resolve_camera_output_prefix",
]


def parse_resolution(resolution: Optional[str]) -> Tuple[int, int]:
    """Parse a resolution string such as ``"1280x720"`` or ``"1280,720"``.

    Returns:
        ``(width, height)`` tuple of positive integers.

    Raises:
        ValueError: If the string cannot be parsed or contains non-positive values.
    """
    if not resolution:
        return DEFAULT_RESOLUTION
    numbers = re.findall(r"\d+", resolution)
    if len(numbers) < 2:
        raise ValueError(f"Could not parse resolution from '{resolution}'")
    width, height = int(numbers[0]), int(numbers[1])
    if width <= 0 or height <= 0:
        raise ValueError("Resolution dimensions must be positive")
    return width, height


def load_calibration_parameters(
    camera_name: str,
    *,
    calibration_file: Optional[Path] = None,
    allow_fallback: bool = True,
    fallback_resolution: Optional[Tuple[int, int]] = None,
    fallback_fov: Optional[float] = None,
) -> Dict[str, object]:
    """Load calibration parameters for *camera_name*.

    Searches ``DATA_DIR`` for matching JSON files.  Falls back to heuristic
    webcam parameters when no file is found and *allow_fallback* is ``True``.

    Returns:
        Dictionary with keys ``camera_matrix``, ``dist_coeffs``, ``resolution``,
        ``fov``, and ``metadata``.
    """
    path = _resolve_calibration_path(camera_name, calibration_file)
    if path is not None:
        return _load_calibration_json(path)

    if not allow_fallback:
        slug = _slugify(camera_name)
        raise FileNotFoundError(
            f"No calibration data found for '{slug}' and fallback disabled"
        )

    resolution = fallback_resolution or DEFAULT_RESOLUTION
    fov = fallback_fov or DEFAULT_FOV_DEGREES
    camera_matrix, dist_coeffs = _get_typical_parameters(resolution, fov)
    return {
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "resolution": resolution,
        "fov": fov,
        "metadata": {
            "source": "fallback",
            "camera_name": camera_name,
            "note": "Returned heuristic webcam parameters",
        },
    }


def resolve_camera_output_prefix(
    camera_name: str,
    *,
    suffix: Optional[str] = None,
    timestamp: bool = False,
    create_parent: bool = True,
) -> Path:
    """Return a ``Path`` prefix inside ``DATA_DIR`` for camera artefacts.

    Args:
        camera_name: Human-readable camera name (slugified automatically).
        suffix: Optional extra label appended to the slug.
        timestamp: Append a ``YYYYMMDD_HHMMSS`` timestamp when ``True``.
        create_parent: Create the parent directory if it does not exist.

    Returns:
        ``Path`` object pointing to the output prefix (no extension).
    """
    slug = _slugify(camera_name)
    base_dir = DATA_DIR / slug
    if create_parent:
        base_dir.mkdir(parents=True, exist_ok=True)

    segments = [slug]
    if suffix:
        segments.append(_slugify(suffix))
    if timestamp:
        segments.append(datetime.now().strftime("%Y%m%d_%H%M%S"))

    return base_dir / "_".join(segments)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _slugify(value: str) -> str:
    slug = _SLUG_PATTERN.sub("_", value.lower()).strip("_")
    return slug or "camera"


def _resolve_calibration_path(
    camera_name: str,
    calibration_file: Optional[Path],
) -> Optional[Path]:
    if calibration_file is not None:
        path = Path(calibration_file)
        if path.is_dir():
            return _latest_json_file(list(path.glob("*.json")))
        if path.exists():
            return path
        return None

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    slug = _slugify(camera_name)
    candidate_paths: List[Path] = []

    camera_dir = DATA_DIR / slug
    if camera_dir.is_dir():
        candidate_paths.extend(camera_dir.glob("*.json"))

    candidate_paths.extend(DATA_DIR.glob(f"{slug}*.json"))

    unique_paths: List[Path] = []
    seen: Set[Path] = set()
    for candidate in candidate_paths:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_paths.append(candidate)

    return _latest_json_file(unique_paths)


def _latest_json_file(paths: Iterable[Path]) -> Optional[Path]:
    latest: Optional[Path] = None
    latest_mtime = float("-inf")
    for path in paths:
        if not path.exists():
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if mtime > latest_mtime:
            latest = path
            latest_mtime = mtime
    return latest


def _load_calibration_json(path: Path) -> Dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid calibration JSON in {path}") from exc
    except OSError as exc:
        raise OSError(f"Failed to read calibration file {path}") from exc

    if "camera_matrix" not in payload or "dist_coeffs" not in payload:
        raise ValueError(
            f"Calibration file {path} is missing camera intrinsics"
        )

    camera_matrix = np.asarray(payload["camera_matrix"], dtype=np.float32)
    if camera_matrix.size != 9:
        raise ValueError(
            f"Calibration file {path} has malformed camera matrix"
        )
    camera_matrix = camera_matrix.reshape(3, 3)

    dist_coeffs = np.asarray(payload["dist_coeffs"], dtype=np.float32).reshape(-1)

    resolution_data = payload.get("image_size") or payload.get("resolution")
    if resolution_data is None:
        resolution = DEFAULT_RESOLUTION
    else:
        if len(resolution_data) != 2:
            raise ValueError(
                f"Calibration file {path} has invalid resolution data"
            )
        resolution = (int(resolution_data[0]), int(resolution_data[1]))

    fov = payload.get("fov")
    if fov is None:
        fov = _infer_horizontal_fov(camera_matrix, resolution)

    metadata = {
        key: value
        for key, value in payload.items()
        if key not in {"camera_matrix", "dist_coeffs"}
    }
    metadata.update({"source": "file", "path": str(path)})

    return {
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "resolution": resolution,
        "fov": float(fov),
        "metadata": metadata,
    }


def _infer_horizontal_fov(
    camera_matrix: np.ndarray,
    resolution: Tuple[int, int],
) -> float:
    width = float(resolution[0])
    fx = float(camera_matrix[0, 0])
    if fx <= 0 or width <= 0:
        return DEFAULT_FOV_DEGREES
    fov_rad = 2 * np.arctan(width / (2 * fx))
    return float(np.degrees(fov_rad))


def _get_typical_parameters(
    resolution: Tuple[int, int],
    fov_degrees: float,
) -> Tuple[np.ndarray, np.ndarray]:
    width, height = resolution
    fov_rad = np.radians(fov_degrees)
    fx = width / (2 * np.tan(fov_rad / 2))
    cx = width / 2.0
    cy = height / 2.0
    camera_matrix = np.array(
        [[fx, 0.0, cx], [0.0, fx, cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    dist_coeffs = np.zeros((5,), dtype=np.float32)
    return camera_matrix, dist_coeffs
