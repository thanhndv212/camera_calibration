"""Generate ArUco markers and pattern sheets for calibration workflows."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np

from . import _viz

# Patterns directory lives next to the package's data/ folder
PATTERNS_DIR = Path(__file__).resolve().parents[1] / "patterns"
PATTERNS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_DICT_NAME = "DICT_6X6_250"

__all__ = [
    "DEFAULT_DICT_NAME",
    "PATTERNS_DIR",
    "get_aruco_dictionary",
    "generate_aruco_marker",
    "generate_marker_sheet",
    "generate_marker_set",
]


def get_aruco_dictionary(
    dict_name: str = DEFAULT_DICT_NAME,
) -> Tuple[object, bool]:
    """Return the requested ArUco dictionary and an OpenCV-4.7+ API flag.

    Args:
        dict_name: OpenCV aruco dictionary name (e.g. ``"DICT_6X6_250"``).

    Returns:
        ``(dictionary, modern_api)`` where *modern_api* is ``True`` when the
        OpenCV 4.7+ API is available.
    """
    dict_attr = getattr(cv2.aruco, dict_name)
    try:
        dictionary = cv2.aruco.getPredefinedDictionary(dict_attr)
        return dictionary, True
    except AttributeError:  # OpenCV < 4.7
        dictionary = cv2.aruco.Dictionary_get(dict_attr)  # type: ignore[attr-defined]
        return dictionary, False


def _generate_single_marker(
    dictionary: object,
    marker_id: int,
    marker_size: int,
    modern_api: bool,
) -> np.ndarray:
    if modern_api:
        return cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size)
    return cv2.aruco.drawMarker(dictionary, marker_id, marker_size)  # type: ignore[attr-defined]


def generate_aruco_marker(
    marker_id: int = 0,
    marker_size: int = 200,
    save_path: Path | None = None,
    display: bool = True,
    dict_name: str = DEFAULT_DICT_NAME,
) -> np.ndarray:
    """Generate a single ArUco marker, optionally saving and displaying it.

    When *display* is ``True`` the marker is logged to the active rerun
    recording under ``markers/id_<marker_id>``.

    Args:
        marker_id: Marker identifier.
        marker_size: Side length of the generated image in pixels.
        save_path: If given, write the marker as a PNG to this path.
        display: Log the marker image to rerun when ``True``.
        dict_name: OpenCV ArUco dictionary name.

    Returns:
        Generated marker as a grayscale ``uint8`` array.
    """
    dictionary, modern_api = get_aruco_dictionary(dict_name)
    marker = _generate_single_marker(dictionary, marker_id, marker_size, modern_api)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), marker)
        print(f"Marker saved to: {save_path}")

    if display:
        _viz.log_frame(f"markers/id_{marker_id}", marker)

    return marker


def generate_marker_sheet(
    marker_ids: Iterable[int],
    marker_size: int = 200,
    padding: int = 25,
    dict_name: str = DEFAULT_DICT_NAME,
    save_dir: Path | None = None,
) -> Path:
    """Generate a grid sheet of ArUco markers and return its saved path.

    The sheet is also logged to rerun under ``markers/sheet``.

    Args:
        marker_ids: Ordered sequence of marker IDs to include.
        marker_size: Side length of each marker in pixels.
        padding: White-space border around each marker in pixels.
        dict_name: OpenCV ArUco dictionary name.
        save_dir: Directory for the output PNG (defaults to ``PATTERNS_DIR``).

    Returns:
        Path to the saved PNG sheet.
    """
    marker_ids = list(marker_ids)
    if not marker_ids:
        raise ValueError("marker_ids must contain at least one ID")

    save_dir = Path(save_dir) if save_dir else PATTERNS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    dictionary, modern_api = get_aruco_dictionary(dict_name)
    grid_dim = int(np.ceil(np.sqrt(len(marker_ids))))
    cell_size = marker_size + 2 * padding
    sheet_size = cell_size * grid_dim
    sheet = np.ones((sheet_size, sheet_size), dtype=np.uint8) * 255

    for idx, marker_id in enumerate(marker_ids):
        row, col = divmod(idx, grid_dim)
        marker = _generate_single_marker(dictionary, marker_id, marker_size, modern_api)
        y0 = row * cell_size + padding
        x0 = col * cell_size + padding
        sheet[y0 : y0 + marker_size, x0 : x0 + marker_size] = marker
        cv2.putText(
            sheet,
            f"ID {marker_id}",
            (x0, y0 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            0,
            2,
        )

    sheet_path = save_dir / "aruco_marker_sheet.png"
    cv2.imwrite(str(sheet_path), sheet)
    print(f"Marker sheet saved to: {sheet_path}")

    _viz.log_frame("markers/sheet", sheet)
    return sheet_path


def generate_marker_set(
    num_markers: int = 4,
    marker_size: int = 200,
    dict_name: str = DEFAULT_DICT_NAME,
    save_dir: Path | None = None,
) -> Path:
    """Backward-compatible wrapper: generate sequential IDs 0 … num_markers-1."""
    return generate_marker_sheet(
        marker_ids=range(num_markers),
        marker_size=marker_size,
        dict_name=dict_name,
        save_dir=save_dir,
    )


# ---------------------------------------------------------------------------
# Stand-alone CLI (also reachable via ``camera-calibration generate``)
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ArUco marker generation tool")
    parser.add_argument("--marker-id", type=int, default=0, dest="marker_id")
    parser.add_argument("--size", type=int, default=400)
    parser.add_argument("--dict", default=DEFAULT_DICT_NAME)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--sheet", type=int, default=0,
                        help="Number of sequential markers for a sheet")
    parser.add_argument("--no-display", action="store_true", dest="no_display")
    return parser.parse_args()


def main() -> None:  # pragma: no cover
    args = _parse_args()
    display = not args.no_display

    save_path = args.output or PATTERNS_DIR / f"aruco_marker_{args.marker_id:02d}.png"
    generate_aruco_marker(
        marker_id=args.marker_id,
        marker_size=args.size,
        save_path=save_path,
        display=display,
        dict_name=args.dict,
    )

    if args.sheet > 0:
        ids = range(args.marker_id, args.marker_id + args.sheet)
        generate_marker_sheet(marker_ids=ids, marker_size=args.size, dict_name=args.dict)


if __name__ == "__main__":  # pragma: no cover
    main()
