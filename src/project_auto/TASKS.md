# Project AUTO Tasks

## Current task

Connect the verified initial `ADD` workflow to the live pipeline, one code chunk at a time:

- add a YAML-configured SQLite database path;
- construct `DatabaseStore`, create its schema, and construct `DetectionTracker` and
  `EventEngine` once before the frame loop;
- pass each frame's complete detection list through `DetectionTracker.update()`;
- pass each meaningful signal through `EventEngine.process_signal()`;
- preserve the existing debug display and avoid database writes for ordinary frames;
- add focused integration tests where practical before relying on live camera verification.

## Next

- Replace provisional track bindings with reliable permanent identity association.
- Implement the remaining `PRESENT`, `OCCLUDED`, and `REMOVED` state transitions.
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
- Initial BoT-SORT detector integration:
  - inference now uses `model.track()` with `persist=True` and bundled `botsort.yaml`;
  - structured detections expose an optional temporary `track_id`;
  - tracked and missing-ID conversion paths passed focused manual checks.
- Initial tracker and state-decision scaffolding:
  - candidate and confirmed tracking statuses defined;
  - active-track counters and tracker configuration structures defined;
  - newly added items have a `PRESENT`/`ADDED` state decision.
- Initial tracker add-confirmation behavior:
  - consumes temporary IDs assigned by BoT-SORT rather than generating competing IDs;
  - retains candidate history across frames and ignores detections without an ID;
  - confirms a candidate after 30 actual sightings;
  - tolerates up to 10 consecutive candidate misses;
  - emits one immutable `ADD` signal when confirmation occurs;
  - does not write directly to the state machine or database.
- Tracker tests cover confirmation timing, one-time signaling, missing IDs, the 10/11-frame
  dropout boundary, and invalid configuration.
- Initial state-decision and event workflow:
  - `decide_track_signal()` maps tracker `ADD` to `PRESENT` plus `ADDED`;
  - `add_item_with_event()` atomically creates the permanent item and linked initial event;
  - `EventEngine.process_signal()` dispatches the decision and records a provisional
    track-to-item association;
  - event-engine tests cover persistence, metadata, duplicate rejection, and mismatched IDs.
- Detector unit test updated for `model.track()` and temporary track IDs; the full suite
  currently passes with 24 tests.
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

- BoT-SORT detections, tracker signals, and event persistence are not yet connected in
  `main.py`.
- Missing, reappeared, and removed tracker signals are not implemented.
- `memory/state_machine.py` and `events/event_engine.py` handle only the initial `ADD` path;
  remaining transitions are not implemented.
- `memory/regions.py` is empty.
- The live detector does not yet write to the database.
