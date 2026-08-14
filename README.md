# Project AUTO

Project AUTO is a local computer-vision system that will remember where physical objects
were last placed on a table.

## Implemented so far

- Webcam capture through OpenCV.
- YOLO11s detection exported to and run through OpenVINO.
- Structured detections and a live debug display.
- SQLAlchemy/SQLite models for permanent items and meaningful item events.
- Database operations for item creation, history, movement, and status changes.

Tracking, identity association, the state machine, automatic event detection, and the query
interface are not yet connected to the live pipeline.

## Setup

Create and activate a Python 3.10-3.13 virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

Run the live detector:

```powershell
project-auto
```

Press `q` while the camera window is focused to stop it.

## Runtime configuration

- `configs/camera.yaml` controls the camera device, resolution, and FPS.
- `configs/perception.yaml` controls model paths and inference thresholds.
- On first run, Ultralytics downloads `models/yolo11s.pt` and exports
  `models/yolo11s_openvino_model/`. Later runs reuse the exported model.

## Current development task

The next feature is the item state machine. See `src/project_auto/TASKS.md` for its exact
scope and `src/project_auto/PROJECT_CONTEXT.md` for architectural decisions.

## Tests

The persistent-memory database suite currently contains 14 passing tests:

```powershell
python -m pytest -q tests\test_memory_database.py
```

The current virtual environment may require pytest to be installed separately before that
command is available.
