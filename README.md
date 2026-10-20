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
| `workbench` | Browser workbench for any camera (`--source` URI): live view, calibration, sensor controls, Benchmark tab |
| `solve`     | Solve intrinsics headlessly from saved views (`--from dir:PATH`) |
| `target`    | Write a printable calibration board |
| `bench`     | Stream, snapshot and sweep benchmarks without the GUI; writes `report.json` |
| `clock-page`| Write the lag-clock page that `bench --clock` reads |
| `esp32-mock`| Emulate an ESP32 CameraWebServer (synthetic scene) for testing without a board |

Sources are URIs: `uvc:0`, `uvc:name=USB Camera`, `esp32:<host>`, `mjpeg:<url>`,
`dir:<path>`, `video:<file>`, `mock:`.

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

# Workbench on an ESP32 CameraWebServer, and the same benchmarks headless
camera-calibration workbench --source esp32:192.168.1.42 --open
camera-calibration bench --source esp32:192.168.1.42 --sweep --duration 10
camera-calibration bench --source uvc:0 --duration 30
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
│   ├── estimator.py      # Webcam parameter estimation
│   ├── sources/          # Camera sources by URI (uvc, esp32, mjpeg, dir, video, mock)
│   ├── capture.py        # CaptureLoop: capture, analysis and preview threads
│   ├── targets/          # ChArUco and chessboard targets
│   ├── procedures/       # Intrinsics procedure and headless solve
│   ├── artifact.py       # Versioned calibration file
│   ├── bench.py          # Stream/snapshot benchmarks, image metrics, lag clock
│   ├── esp32_mock.py     # Mock ESP32 CameraWebServer
│   └── workbench/        # HTTP server, benchmark jobs and the page
├── firmware/esp32_uvc/   # sdkconfig for the ESP32-S3 USB webcam build
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

MIT
