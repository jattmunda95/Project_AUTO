# Project AUTO Tasks

## Current task

Implement associative memory/ReID one explicitly approved feature at a time:

1. Add a standalone associative-memory/ReID interface:
   - accept a detection and its object crop;
   - search eligible permanently identified items without mutating tracker or database state;
   - return a permanent `item_id`, similarity score, and acceptance decision;
   - use a configurable confidence threshold;
   - keep the interface disconnected from `RETURNED`, tracker emission, and the live app until
     its behavior is implemented and verified.

## Next

- Implement crop and embedding storage for permanent identities.
- Implement and verify ReID candidate matching.
- Connect the `RETURNED` boilerplate to ReID output.
- Implement remaining occlusion/status-change behavior.
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
- Time-based tracker add-confirmation behavior:
  - consumes temporary IDs assigned by BoT-SORT rather than generating competing IDs;
  - retains candidate history across frames and ignores detections without an ID;
  - confirms a visible candidate after a YAML-configured two seconds using monotonic time;
  - tolerates up to 15 cumulative candidate misses per attempt and resets on the 16th;
  - emits one immutable `ADD` signal when confirmation occurs;
  - does not write directly to the state machine or database.
- Tracker tests cover deterministic confirmation timing, one-time signaling, missing IDs,
  cumulative dropout/reset behavior, and invalid configuration.
- Initial state-decision and event workflow:
  - `decide_track_signal()` maps tracker `ADD` to `PRESENT` plus `ADDED`;
  - tracker `REMOVE` maps to `REMOVED` plus `REMOVED`;
  - `add_item_with_event()` atomically creates the permanent item and linked initial event;
  - `EventEngine.process_signal()` dispatches the decision and records a provisional
    track-to-item association;
  - `EventEngine.process_remove()` updates the permanent item and creates its removal event
    without deleting database data or the provisional session association;
  - event-engine tests cover persistence, metadata, duplicate rejection, and mismatched IDs.
- Confirmed-track removal behavior:
  - only confirmed tracks enter the missing/removal lifecycle;
  - absence uses a configurable two-second timeout;
  - the same BoT-SORT ID reappearing inside the window cancels removal;
  - expiry emits one `REMOVE` signal and retires the active tracker record.
- Stable-placement movement behavior:
  - `ADD` establishes a stable bbox and centered 1.2x placement buffer;
  - exiting the buffer assigns temporary `MOVING` tracker status;
  - fixed-reference bbox-centre checks prevent slow drift from looking stopped;
  - remaining within 5 pixels for one visible second emits one timed `MOVED` signal;
  - signals include source/destination boxes plus movement start/finish timestamps;
  - a brief missing period preserves movement but restarts visible stop confirmation;
  - a two-second missing timeout emits `REMOVE` rather than `MOVED`;
  - state decisions map movement to unchanged `PRESENT` plus `MOVED`;
  - the event engine resolves the permanent item and persists movement atomically.
- The YAML-configured SQLite path is present, and `app.py` constructs the long-lived store,
  tracker, and event engine outside the frame loop.
- Each frame's complete detection list flows through the tracker; only emitted lifecycle
  signals reach persistence.
- Detector unit tests cover `model.track()` and temporary track IDs.
- Initial persistent-memory database foundation:
  - SQLAlchemy 2.x typed `Item` and `ItemEvent` models;
  - constrained lowercase string enums;
  - SQLite schema creation and foreign-key enforcement;
  - item creation, retrieval, counting, and present-item listing;
  - event recording and chronological item history;
  - atomic removed, occluded, present, and movement operations;
  - nullable event `started_at` and `finished_at` columns for duration-based events;
  - shared store validation rejects partial, timezone-naive, and reversed event intervals;
  - general and movement-specific store operations persist valid event intervals;
  - nullable JSON source/destination bbox columns preserve movement coordinates separately
    from optional semantic regions;
  - cascade deletion of an item's event history.
- Persistent-memory database tests: 17 passed.
- Tracker lifecycle tests: 19 passed, including stable, moving, stopped, missing, reappeared,
  removed, and invalid movement-configuration paths.
- The local SQLite database was successfully recreated with the updated schema.
- A live `ADD` then `MOVED` demonstration completed successfully on 22 August 2026.
- Modular `MOVED` behavior is implemented end to end; only its remaining permanent
  event-engine and persistence validation tests are outstanding.
- Disconnected `RETURNED` boilerplate:
  - `RETURNED` model and tracker signal types are defined;
  - return signals carry an optional permanent `item_id` that future ReID must resolve;
  - the tracker contains an inactive ReID return-check stub and never infers identity from a
    temporary BoT-SORT ID;
  - standalone state-decision, atomic store, and event-engine return interfaces are defined;
  - successful returns require an existing `REMOVED` permanent item, update it to `PRESENT`,
    and create a linked `RETURNED` event;
  - return interfaces remain disconnected from normal tracker dispatch, provisional track
    association, and `app.py`;
  - focused tests cover decisions, permanent identity requirements, persistence, metadata,
    invalid item states, unknown items, and mismatched tracker IDs.
- The full suite passes with 46 tests.

## Not implemented yet

- Reliable identity association/ReID is not implemented; temporary BoT-SORT IDs remain
  session/debug metadata and must not become permanent identity.
- Permanent event-engine `MOVED` tests and permanent store interval/bbox validation tests have
  not been added yet; the behavior itself is implemented, and direct integration, persistence,
  and live-demo checks pass.
- Future existing databases will require migration when schemas change because `create_all()`
  does not alter existing tables; the current local database has been recreated successfully.
- `RETURNED` activation is not implemented and must remain inactive until ReID reliably
  resolves a detection to a permanent `item_id`.
- Occlusion and general status-change tracker behavior are not implemented.
- `memory/regions.py` is empty.
- The debug display does not yet show tracker lifecycle states or emitted events.
