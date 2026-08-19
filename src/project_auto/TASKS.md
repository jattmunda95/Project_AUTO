# Project AUTO Tasks

## Current task

Add focused unit tests for the initial add-confirmation lifecycle in
`tests/test_state_machine.py` or a dedicated tracker test file, one code chunk at a time:

- no signal before the 30th sighting;
- one `ADD` signal on the 30th sighting;
- no repeated `ADD` signal after confirmation;
- detections without a BoT-SORT ID are ignored;
- candidates tolerate 10 consecutive missed frames and expire on the 11th;
- invalid confirmation and missed-frame configuration is rejected.

## Next

- Associate temporary tracker IDs with permanent item IDs.
- Connect a newly confirmed track to the state-machine `ADDED` decision and atomic database
  persistence.
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
- Detector unit test updated for `model.track()` and temporary track IDs; full suite currently
  passes with 15 tests.
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

- Formal unit tests for tracker confirmation and dropout behavior have not been added.
- BoT-SORT IDs and tracker signals are not yet connected in the live pipeline or shown by the
  live display.
- Missing, reappeared, and removed tracker signals are not implemented.
- `memory/state_machine.py` only handles the initial `ADDED` decision; remaining transitions
  are not implemented.
- `events/event_engine.py` and `memory/regions.py` are empty.
- The live detector does not yet write to the database.
