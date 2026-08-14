# Project AUTO Tasks

## Current task

Build the item state machine in `memory/state_machine.py`:

- Define valid transitions between `PRESENT`, `OCCLUDED`, and `REMOVED`.
- Return the correct event type for each meaningful transition.
- Suppress repeated observations that do not change state.
- Keep transition decisions separate from `DatabaseStore` persistence.
- Add focused tests for valid, repeated, and invalid transitions.

## Next

- Add tracking with ByteTrack or BoT-SORT.
- Introduce stable in-memory track state.
- Associate temporary tracker IDs with permanent item IDs.
- Detect placement and relocation events.
- Save high-quality object crops and context evidence.
- Add object-location queries.

## Completed

- Project structure and dependencies established.
- Basic webcam capture implemented and verified.
- Basic vision pipeline implemented and verified on the target laptop:
  - YOLO11s export and inference through OpenVINO;
  - structured detections;
  - debug bounding boxes and labels.
- Initial persistent-memory database foundation:
  - SQLAlchemy 2.x typed `Item` and `ItemEvent` models;
  - constrained lowercase string enums;
  - SQLite schema creation and foreign-key enforcement;
  - item creation, retrieval, counting, and present-item listing;
  - event recording and chronological item history;
  - atomic removed, occluded, present, and movement operations;
  - cascade deletion of an item's event history.
- Persistent-memory database tests: 14 passed.

## Not implemented yet

- `perception/tracker.py` is empty; tracking has not been added.
- `memory/state_machine.py` is empty; this is the current task.
- `events/event_engine.py` and `memory/regions.py` are empty.
- The live detector does not yet write to the database.
