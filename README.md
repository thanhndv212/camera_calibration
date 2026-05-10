# camera_calibration

Standalone Python package for camera calibration and ArUco marker detection. All visualization is done through [Rerun](https://rerun.io/) — no `cv2.imshow`, no matplotlib windows.

## Features

- Chessboard intrinsic calibration with **auto** or **manual** capture modes
- Live **ArUco detection** monitor
- **ArUco marker / sheet generation** (no display dependency)
- Webcam **parameter estimation** from presets or live detection
- Unified CLI with 7 subcommands
- Rerun as the sole GUI backend — works headless with `--no-spawn`

## Installation

```bash
pip install -e .
```

Requires Python ≥ 3.10. Key dependencies: `opencv-python>=4.7`, `numpy>=1.24`, `rerun-sdk>=0.22`.

## CLI usage

```
camera-calibration <subcommand> [options]
python -m camera_calibration <subcommand> [options]
```

### Subcommands

| Subcommand  | Description |
|-------------|-------------|
| `capture`   | Capture chessboard images from webcam and calibrate |
| `calibrate` | Load an existing calibration file; optionally run a live undistortion test |
| `detect`    | Live ArUco detection monitor |
| `generate`  | Generate ArUco marker PNG files |
| `estimate`  | Estimate camera intrinsics from a preset or auto-detection |
| `record`    | Stream raw camera feed to rerun |
| `view`      | Load a calibration JSON and display its summary in rerun |

### Global rerun flags

Available on every subcommand:

| Flag | Description |
|------|-------------|
| `--no-spawn` | Do not open the rerun viewer window |
| `--rerun-save FILE` | Save the rerun recording to a `.rrd` file |

### Examples

```bash
# Capture 20 chessboard images (auto mode) and calibrate
camera-calibration capture --images 20

# Manual capture (press Enter in terminal to capture each frame)
camera-calibration capture --images 20 --manual

# View an existing calibration
camera-calibration view --calibration-file data/my_calibration.json

# Live ArUco detection (DICT_6X6_250, markers 0–3 highlighted)
camera-calibration detect --highlight-ids 0 1 2 3

# Generate marker PNGs for IDs 0–4
camera-calibration generate --marker-ids 0 1 2 3 4

# Save rerun recording to file (rerun viewer still opens)
camera-calibration capture --images 15 --rerun-save capture.rrd

# Headless (no viewer window)
camera-calibration capture --images 15 --no-spawn --rerun-save capture.rrd
```

## Package layout

```
camera_calibration/
├── camera_calibration/
│   ├── __init__.py       # Public API re-exports
│   ├── __main__.py       # Unified CLI (argparse)
│   ├── _viz.py           # Rerun visualization helpers
│   ├── _shared.py        # Shared paths and calibration JSON loader
│   ├── calibrator.py     # Chessboard calibration engine
│   ├── detector.py       # ArUco detection monitor
│   ├── markers.py        # ArUco marker generation
│   └── estimator.py      # Webcam parameter estimation
├── data/                 # Saved calibration JSON / CSV files
├── patterns/             # Generated ArUco PNG files
├── tests/
│   ├── test_shared_helpers.py
│   └── test_camera_window.py
└── pyproject.toml
```

## Data files

Calibration results are saved as JSON (and optionally pickle / FIGAROH Python format) under `data/`. The JSON format:

```json
{
  "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "dist_coeffs": [k1, k2, p1, p2, k3],
  "resolution": [width, height],
  "rpe": 0.42,
  "num_images": 20,
  "metadata": { "timestamp": "..." }
}
```

## Running tests

```bash
pip install pytest
pytest tests/ -v
```

## License

Apache-2.0
