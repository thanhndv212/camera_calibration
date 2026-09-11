# Changelog

All notable changes to `camera-calibration` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `LICENSE` (MIT), matching the license declared in `pyproject.toml`.
- This file.

## [0.1.0] — 2026-05-10

Reconstructed from git history; this package had no changelog before now.

### Added

- Standalone camera intrinsics and ArUco tooling, independent of the rest
  of the workspace — nothing here imports the teleop or SDK packages.
- `calibrator.py` — chessboard intrinsic calibration, auto or manual
  capture. Results are written to `data/` as JSON: camera matrix,
  distortion coefficients, resolution, reprojection error, metadata.
- `detector.py` — live ArUco detection monitor.
- `markers.py` — ArUco marker and sheet PNG generation, with no display
  dependency.
- `estimator.py` — webcam parameter estimation from presets or live
  detection.
- `__main__.py` — unified CLI: `capture`, `calibrate`, `detect`,
  `generate`, `estimate`, `record`, `view`.
- All visualization goes through **Rerun** (`_viz.py`). There is no
  `cv2.imshow` or matplotlib path anywhere in the package, and headless
  operation is supported via `--no-spawn`.

### Fixed

- Updated to the Rerun v0.18+ API, where `set_time_sequence` was removed.
