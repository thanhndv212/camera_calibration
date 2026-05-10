# Calibration Data Samples

This directory mirrors the reference calibration captures that originally
shipped with the FIGAROH SO-ARM example.  The files provide quick fixtures for
testing pipelines without requiring a live camera session.

| File | Description |
| --- | --- |
| `camera_calibration_20250607_174327.json` | Final calibration bundle captured from a 720p webcam (intrinsics, distortion, metadata). |
| `camera_calibration_20250607_174327.pkl` | Binary pickle with the same payload for rapid Python loading. |
| `camera_calibration_20250607_174327_figaroh.py` | Python snippet formatted for legacy FIGAROH scripts. |
| `camera_calibration_data_20250607_*.json` | Raw frame detections recorded during calibration runs. |
| `camera_calibration_data_20250607_*.csv` | CSV companions for spreadsheet inspections of the raw detections. |

These artifacts are intended for local experimentation only.  When deploying to
hardware, regenerate calibration data that matches the target camera and
mounting configuration.
