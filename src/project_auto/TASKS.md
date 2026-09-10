# Project AUTO Tasks

## Current task

Prepare the next identity architecture one explicitly approved feature at a time.
Implementation of scene processing and live integration is deferred for now.

- The app/coordinator reacts to tracker confirmation and calls a stage-blind scene processor.
  The tracker must never call SAM, ReID, the scene processor, or the database.
- Compare after confirmation, before the existing ADD path creates a permanent item.
  Short-lived candidates create neither items nor references.
- Scene processing prepares a detection crop, applies SAM's inverted keep-mask, embeds it,
  and returns a proposed NEW, EXISTING, or PENDING identity decision with source_track_id,
  optional permanent item_id, and similarity. These result types are not implemented yet.
- The event layer combines confirmation, identity, and current database status: NEW -> ADD;
  EXISTING + REMOVED -> RETURNED; EXISTING + PRESENT -> association only. Define OCCLUDED
  restoration separately; an existing match alone must never imply RETURNED.
- The coordinator retains pending decisions and retries on later visible frames because
  confirmation is a one-time signal. Cancel on retirement and guard against reused track IDs.
  Define handling of movement/removal signals while identity remains unresolved.
- Prevent two visible tracks from claiming the same permanent item.

## Next

- Migrate existing SQLite data for item_embeddings and items.item_prototype before live use;
  create_all() does not add columns to existing tables. Preserve existing data.
- Track model/preprocessing compatibility, including prototype provenance; never mix models
  or masked and unmasked references. Gallery eligibility is metadata filtering, not vector search.
- Verify real SAM polarity/quality and DINOv2 matching against the local models before trusting
  live decisions. Preserve the requested inverted mask convention until explicitly changed.
  Current checks are synthetic/mocked, not accuracy validation, and no live demo has run yet.
- Add permanent tests for prototype calculation and segmentation.
- Implement remaining occlusion/status-change behavior; OCCLUDED restoration is still undefined.
- Save high-quality object crops and context evidence (object_image_path is still unset by
  the coordinator's reference captures).
- Add object-location queries.
- Handle movement/removal signals for tracks whose identity is still PENDING beyond the
  retirement guard already in place (see Completed); no timeout/backoff on repeated PENDING.

## Completed

- Two-stage ReID matching in memory/reid.py: match_candidate first shortlists the
  prototype_shortlist_size (default 3, YAML-configured) permanent items whose stored
  prototype is closest to the query embedding, then runs the existing per-reference
  top-k mean comparison only against that shortlist's references; an item with no
  stored prototype is always kept in the shortlist rather than silently dropped, since
  it cannot be ranked. A gallery at or below the shortlist size skips ranking entirely.
  GalleryEntry now carries each item's prototype, and store.load_reid_gallery loads it
  alongside the grouped reference arrays. Focused tests cover small-gallery bypass,
  prototype-based exclusion of an otherwise-winning reference, and missing-prototype
  fallback; real-image accuracy of the shortlist stage is unverified.
- Live identity wiring in app.py through a new IdentityCoordinator
  (src/project_auto/events/coordinator.py):
  - the tracker's ADD confirmation now routes through scene_processor before any permanent
    item is created; a NEW decision dispatches EventEngine.process_add, a match against a
    REMOVED item dispatches EventEngine.process_return, and a match against a PRESENT/OCCLUDED
    item calls the new EventEngine.associate_existing_item (binds the track, no event);
  - a PENDING (unusable crop/mask) decision is retried on later visible frames for the same
    track ID; a track that retires while still PENDING is dropped without ever reaching the
    event layer, since no permanent item was created for it;
  - EventEngine.is_item_claimed / associate_existing_item / item_id_for_track prevent two
    visible tracks from claiming the same permanent item; process_return now binds its track
    after successful persistence and rejects a reused track or a doubly claimed item, replacing
    the earlier disconnected behavior that left no binding;
  - store.count_item_embeddings and store.add_reference_if_needed are implemented: the latter
    checks the current count and saves in one transaction against a caller-supplied target,
    keeping the prototype update atomic with the accepted write, and reporting whether it saved;
  - the coordinator captures the identity-resolution embedding as an item's first reference,
    then schedules further spaced captures (YAML-configured interval) for resolved, visible
    items below configs/scene_processor.yaml's reference_target_count (six by default); capture
    naturally stops while a track is not visible in the current frame and resumes without
    re-resolving identity once it reappears under the same track ID;
  - new configs/segmenter.yaml (SAM2 model/device) and additions to configs/scene_processor.yaml
    (reference_target_count, capture_interval_seconds) back the live construction in app.py;
  - focused tests cover NEW/EXISTING/PENDING dispatch, retirement while pending, double-claim
    guarding, and capture stopping at the target count; no live-camera/model demo has run yet,
    and the SAM/DINOv2 accuracy caveats above still apply.
- Validated reference saving and gallery loading in store.py:
  - save_item_embedding validates/normalizes the vector, rejects dimension mismatches against
    existing references for the same model_name, appends the ItemEmbedding row, and updates
    the item's prototype atomically in one transaction.
  - load_reid_gallery returns an independent in-memory snapshot (float32 arrays grouped by
    item_id), filterable by model_name and item status; not a live view, reload explicitly
    after writes.
- Standalone DINOv2 ReID matcher and configs/reid.yaml; synthetic matching tests implemented.
  The optional reid dependency group declares torch, transformers, and Pillow.
- ItemEmbedding model stores permanent-item references, model name, crop path, and timestamp;
  Item.item_prototype stores a nullable JSON vector. Database model tests cover references.
- DatabaseStore.update_item_prototype() averages references and normalizes the mean, clears
  missing references, and rejects invalid/mixed-model inputs. Focused manual checks passed;
  it is explicitly called, not an automatic reference-save hook.
- Standalone SAM2 segmenter uses Transformers, box prompts, BGR-to-RGB conversion, and
  frame-sized masks. Selected raw masks are inverted by user request; True means retain.
  Mocked checks passed; real weights, latency, and segmentation accuracy remain unverified.
- Last recorded full test run: 62 passed before later prototype/segmenter changes. Subsequent
  focused checks passed; this is historical verification, not a new full-suite run.

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
- The pre-ReID lifecycle suite passed with 46 tests.

## Not implemented yet

- Real-world identity accuracy is still unverified; SAM2/DINOv2 wiring in app.py has not been
  exercised against a live camera or real weights, only against mocked collaborators in tests.
- Permanent event-engine `MOVED` tests and permanent store interval/bbox validation tests have
  not been added yet; the behavior itself is implemented, and direct integration, persistence,
  and live-demo checks pass.
- The new embedding/prototype schema has not been migrated into the local database in this
  work; the earlier database recreation covered lifecycle fields only.
- Occlusion and general status-change tracker behavior are not implemented.
- `memory/regions.py` is empty.
- The debug display does not yet show tracker lifecycle states or emitted events.
