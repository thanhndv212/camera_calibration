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
- **Benchmarks, ported from the ESP32 camera GUI that lived in `soarm-ws/tools`**
  (now removed):
  - `camera_calibration.bench`:
    - image metrics: sharpness, brightness, clipping, temporal noise, ArUco
      and target detection
    - the lag clock: a marker that cycles with the host clock, for glass-to-host lag
    - stream bench, snapshot bench, and a sweep over a control that restores the
      control afterwards
  - Stream timing uses every frame's receive time from `CaptureLoop`, so the
    analysis doesn't throttle the measurement.
  - Workbench **Benchmark** tab:
    - live stream health and image-quality tiles
    - two-minute charts of frame rate and interval p95
    - stream benchmark, stream sweep, snapshot bench and snapshot sweep, run in the
      background with a results dialog
    - each run writes `bench_<time>/report.json` and sample frames
  - Snapshot jobs pause the stream while they run.
  - `/clock` serves the lag clock.
  - Workbench **Source** tab:
    - ArUco and thirds-grid overlay
    - ESP32 sensor controls, grouped
  - CLI: `bench` (the same jobs headless; `--info` lists the controls), `clock-page`,
    and `esp32-mock`.
- `camera_calibration.esp32_mock.MockCameraWebServer`: a CameraWebServer with
  `/status`, `/control`, `/capture` and `/stream`, a synthetic scene and optional Wi-Fi-like
  jitter; `port=0` for tests.
- `CaptureLoop.timeline()` (per-frame receive times and JPEG sizes), and stats for interval
  p50/p95/max, stalls over 250 ms and JPEG size.
- `esp32:` sources expose all 25 CameraWebServer controls in four groups (resolution with
  names, exposure and gain, white balance, corrections). Only the controls that `/status`
  reports are listed, and `/status` is cached for 2 s. `esp32:HOST?stream=URL` overrides
  the stream URL.
- `firmware/esp32_uvc/sdkconfig.freenove`: the ESP-IDF config for running the Freenove
  ESP32-S3 CAM as a USB webcam.

### Changed

- **One default ArUco dictionary, `DICT_4X4_50`,** for both `generate` and `detect`
  (generation used `DICT_6X6_250`, detection `DICT_4X4_50`, so a marker generated with the
  defaults was never detected with the defaults). Markers already printed from the old
  default need `--dictionary DICT_6X6_250` to be detected.

### Fixed

- The workbench's `/video` stream busy-looped after the source was replaced; it now waits for
  the new loop.

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
