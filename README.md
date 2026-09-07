# Project AUTO

Project AUTO is a local computer-vision system that will remember where physical objects
were last placed on a table.

## Implemented so far

- Webcam capture through OpenCV.
- YOLO11s detection exported to and run through OpenVINO.
- Structured detections and a live debug display.
- SQLAlchemy/SQLite models for permanent items and meaningful item events.
- Database operations for item creation, history, movement, and status changes.

Tracking, state decisions, and ADD/MOVED/REMOVE persistence are connected to the live pipeline.
Standalone ReID, SAM2 masking, reference models, and prototype calculation exist but are not
connected to live identity decisions. Reliable recognition and the query interface remain pending.

## Setup

Create and activate a Python 3.10-3.13 virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

Install the optional ReID/SAM dependencies when working on identity processing:

```powershell
python -m pip install -e ".[reid]"
```

Existing databases need migration for the embedding/prototype schema before live use;
`create_all()` does not add `items.item_prototype` to an existing table.

Run the live application after preparing the database schema:

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

The next architecture uses an app coordinator to request scene processing after tracker
confirmation, before creating a permanent item. Scene processing prepares crops and returns
identity decisions; the event layer chooses ADD, RETURNED, or association only. The tracker
does not call image models or persistence. Implementation of this integration is deferred.

See `src/project_auto/TASKS.md` for scope and `src/project_auto/PROJECT_CONTEXT.md` for
pending retries, gallery ownership, and collection of six spaced reference crops per new item.

## Tests

Run the persistent-memory database suite:

```powershell
python -m pytest -q tests\test_memory_database.py
```

The current virtual environment may require pytest to be installed separately before that
command is available.
