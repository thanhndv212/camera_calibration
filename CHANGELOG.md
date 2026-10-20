# Changelog

All notable changes to `camera-calibration` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `LICENSE` (MIT), matching the license declared in `pyproject.toml`.
- This file.
- Ruff lint selection pinned in `pyproject.toml` (`E4,E7,E9,F`, as in `soarm_sdk`).
- **`camera_calibration.sources`** — one `CameraSource` protocol for any camera, picked
  by URI: `uvc:0`, `uvc:name=USB Camera` (resolved to an index when opened, since indices
  move on re-plug), `esp32:<host>` (with its `/control` settings), `mjpeg:<url>`,
  `dir:<path>`, `video:<file>`, and `mock:` (a synthetic pinhole camera with known
  intrinsics and poses, for tests).
- **`camera_calibration.capture.CaptureLoop`** — capture thread, analysis worker and
  preview encoder that never block one another, so slow detection lowers the analysis
  rate instead of the camera rate. Reports camera fps and analysis fps separately.
  `find_chessboard` searches a downscaled copy and refines with `cornerSubPix` at full
  resolution.
- **`camera_calibration.artifact`** — versioned calibration file (`schema_version` 1):
  camera identity, capture settings, target spec, model, intrinsics, named transforms
  (`T_<to>_<from>`), per-view and held-out errors, inputs, and a content hash.
  `Calibration.check(source_info)` refuses a file made at another resolution or with
  another camera. Pre-schema files load as version 0, and new files keep the legacy keys,
  so `load_calibration_parameters` reads both.
- **`camera_calibration.targets`**: one `Target` interface for detection, printing and
  synthetic rendering. **ChArUco is the default** (`charuco?squares=7x5&square=0.03&marker=0.022&dict=DICT_4X4_50`):
  partial views still count, and corners keep their identity. Chessboard is also available
  (`chessboard?inner=9x6&square=0.025`). Detection searches a downscaled copy and refines at full
  resolution. The ChArUco refine window stays inside the white area between a corner
  and its markers; a larger window biased the corners by about 0.9 px.
- **`camera_calibration.procedures.IntrinsicsProcedure`**: live feedback (corner coverage
  grid, steadiness, novelty against earlier views, a hint), auto- or manual capture,
  pinhole or fisheye solve with per-view errors, drop-and-re-solve. Every fifth accepted
  view is **held out** of the solve and its reprojection error is reported separately.
- **`camera_calibration.workbench`**, a browser workbench on `CaptureLoop`:
  - Any source URI, with a camera list.
  - Live view with detections and a coverage tint.
  - Camera and analysis rates.
  - Source controls.
  - Collect, solve, drop, save.
  - A printable board at `/target.png`.
- Subcommands:
  - `workbench`: `--source`, `--target`, `--listen`, `--open`.
  - `solve --from dir:PATH --out cal.json`: headless, with held-out error.
  - `target --out board.png`: a printable board labelled with its spec.
- `mock:` renders either target (`mock:?target=charuco`).

### Changed

- **One default ArUco dictionary, `DICT_4X4_50`,** for both `generate` and `detect`
  (generation used `DICT_6X6_250`, detection `DICT_4X4_50`, so a marker generated with the
  defaults was never detected with the defaults). Markers already printed from the old
  default need `--dictionary DICT_6X6_250` to be detected.

### Fixed

- `save_calibration` saved the reprojection error of a second solve whose intrinsics it
  then discarded; it now saves the error of the solve that produced the saved intrinsics,
  and refuses to save before `calibrate_camera()`.
- `calibrator.py` raised `NameError` on a malformed `--pattern-size` (missing `argparse`
  import).
- README license now says MIT, matching `LICENSE` and `pyproject.toml`.

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
