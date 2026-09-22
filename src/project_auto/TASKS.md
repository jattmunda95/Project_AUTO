# Project AUTO Tasks

## Current task

Updated 14 September 2026 after region-memory implementation and documentation review.

1. Confirm the intended live database and reset/recovery decision. A missing-column startup failure was diagnosed; a manual schema update passed checks on a copy but was not applied to the live file by this work. The file later disappeared and is present again; current schema and retained history remain unverified. Inspect and back up before any deliberate migration.
2. Validate polygon calibration and ADD/MOVED/REMOVED/RETURNED region state on physical hardware, including overlap selection, current contents, historical names, and w/r highlights. Automated UI tests use simulated input.
3. Calibrate weighted ReID acceptance_threshold=0.55 and margin_threshold=0.15 using logged same/different-item scores. Current weights are 0.65 DINO + 0.20 color + 0.15 aspect; old 0.4 DINO-only thresholds are historical.
4. Measure frame dropout/latency with current YOLO confidence=0.18 and image_size=960. agnostic_nms inherits the inspected Ultralytics False default. No throughput/accuracy improvement is established yet.

## Implemented and verified in the region work

- Pure simple-polygon validation, shoelace area, inclusive ray casting, centroid resolution, smallest-area overlap selection and stable ID tie-breaks.
- Region table; live Item region/box; event region FKs, historical names and separate normalized source/destination areas. SET NULL region deletion preserves history.
- Lifecycle transactions update meaningful spatial snapshots; removal clears live location. Tracker and geometry remain database-independent.
- Region CRUD and read-only current contents/location, last/recent activity, historical association, and item-history APIs.
- Separate calibration entry point, shared polygon drawing, terminal w/r queries and temporary highlights. Calibration is not automatic normal startup.
- Full implementation suite: 160 tests passed on 14 September; no fresh hardware validation implied. Schema assertion updated from three to four tables with approval. Ruff unavailable.
- Weighted descriptor preparation, storage, gallery fields, margin check and CSV diagnostics are present in current source. Focused accuracy and weighted/margin regression coverage still need review.
- Modular Markdown/HTML docs, README and PROJECT_CONTEXT updated to represent current async and spatial contracts.

## Next

- Improve worker startup/schema failure reporting; initial gallery failure currently bypasses per-job error handling.
- Replace remaining diagnostic prints with structured logging; keep useful match CSV evidence for calibration.
- Avoid duplicate DINO inference in the nonempty-gallery resolve path; measure before optimizing further.
- Stop capture scheduling when the reference target is met; extend capture-path quality/cooldown handling.
- Define ambiguity handling for margin rejection, currently interpreted as NEW and potentially producing duplicate identities.
- Review generation-aware stale-result correlation and whether unresolved MOVED signals should be buffered/replayed.
- Track model/preprocessing/descriptor-layout compatibility and prototype provenance.
- Add/verify targeted tests for weighted descriptors, margin behavior, missing-descriptor fallback, prototype calculation and a real SAM mask-polarity contract.
- Define remaining occlusion/status-change semantics and save high-quality object/context evidence files.
- Keep natural-language/full GUI queries, 3D/homography, automatic migrations, polygon versions and region statistics outside this implementation.

## Historical implementation milestones

These entries record earlier stages, not current architecture/configuration. Statements about disconnected RETURNED paths, retained bindings, old thresholds or old pass counts are superseded by the current status above.

- Real-hardware ReID debugging session: three real (not synthetic) bugs found and fixed,
  in the order that made each next one visible:
  1. **Camera device swap**: `configs/camera.yaml` had `usb_device`/`webcam_device` reversed
     for the target machine; corrected to `usb_device: 0`, `webcam_device: 1`.
  2. **Inverted SAM mask polarity** (`perception/segmenter.py`): the segmenter inverted
     SAM2's mask (`keep_mask = ~candidates[best]`) to "preserve the user's Colab convention."
     Transformers' `Sam2Processor.post_process_masks` returns True for the segmented
     foreground object; the inversion made the *background* get kept and the *object itself*
     get painted over with the fill colour before DINOv2 ever saw it. This produced
     near-zero, unstable same-item similarity scores (observed ~0.0685 for a genuine
     same-item match) since DINOv2 was embedding leftover scene context, not the object.
     Fixed to `keep_mask = candidates[best]`; docstrings in `segmenter.py`/`scene_processor.py`
     updated to drop "inverted"/"Colab convention" language.
  2b. **Acceptance threshold miscalibration** (`configs/reid.yaml`): `acceptance_threshold`
     was 0.75, but raw DINOv2 CLS-token cosine similarity runs much lower than that even for
     correct matches. Lowered to 0.4 as a starting point (needs empirical retuning — see
     Current task); this alone did not fix identity matching (see bug 2), but was masking
     the real signal even after the mask fix and needed correcting alongside it.
  3. **Stale track-to-item binding** (`events/event_engine.py::process_remove`): removing a
     track's item (`mark_removed`) never cleared `_item_ids_by_track_id[track_id]`; the
     docstring said this was deliberate ("retaining its track binding"). Since a retired
     track_id is never reused for the same physical object, that binding lived forever,
     making `is_item_claimed()` report the item as permanently claimed by a track that no
     longer existed — silently blocking every future RETURNED/associate resolution for that
     item, forever, not just for a cooldown window. Fixed by deleting the binding in
     `process_remove`. Regression tests added in `test_event_engine.py` reproducing the
     exact sequence (track removed -> new track resolves as existing -> RETURNED persists).
  - Diagnostic `print()` statements added across `reid.py`, `scene_processor.py`, and
    `coordinator.py` (gallery scores, mask pixel counts/SAM score, NEW/EXISTING/RETURNED
    decisions) made all three of the above traceable from console output; see Current task
    for removing/replacing them with real logging.
- SAM2 checkpoint changed from `facebook/sam2-hiera-small` to `facebook/sam2-hiera-tiny`
  (frame-rate experiment) and then to `facebook/sam2-hiera-base-plus` (accuracy over the
  smallest-model tradeoff, once the async worker removed the frame-rate pressure to stay
  small); see `configs/segmenter.yaml`.
- Asynchronous identification/reference pipeline (replaces the old synchronous
  `pending_retry_interval_seconds` throttle entirely):
  - `IdentityCoordinator` (`events/coordinator.py`) no longer calls SAM/DINO/matching
    synchronously in the video loop. It submits a lightweight `IdentificationJob` on ADD and
    drains completed `IdentificationResult`s from a non-blocking queue every frame; the main
    loop never blocks on segmentation, embedding, matching, or persistence.
  - `IdentificationWorker` (`events/identification_worker.py`) owns one background thread,
    a bounded job queue (`queue.Queue`, `job_queue_max_size`, default 8) and an unbounded
    result queue. It owns the in-memory ReID gallery snapshot, refreshed only in its own
    thread after an accepted save, so no cross-thread gallery mutation is possible.
  - `ReferenceManager` (`events/identification.py`) enforces "at most one outstanding job
    per track" via an explicit `IdentificationState` (UNIDENTIFIED/QUEUED/PROCESSING/
    IDENTIFIED/WAITING_FOR_BETTER_VIEW/DEFERRED), fixing the old bad-mask retry-loop failure
    mode: a bad mask now defers the track with a cooldown (`bad_mask_cooldown_seconds`,
    default 5s) instead of retrying every frame. A full queue defers with its own shorter
    cooldown (`queue_full_retry_seconds`, default 0.5s) rather than blocking `put()`.
  - Cheap pre-SAM quality filtering (box area, detector confidence, frame-edge clipping) in
    the coordinator rejects obviously poor candidates before ever invoking SAM/DINO.
  - Fixed a crash this introduced: the tracker's `MOVED` signal assumed a track already had
    a permanent item bound, which is no longer guaranteed the instant a track is confirmed
    (identity resolution is now backgrounded and can still be QUEUED/PROCESSING/DEFERRED
    when placement/movement logic independently confirms movement). `handle_frame` now drops
    a non-ADD/REMOVE signal for a still-unidentified track instead of crashing; see Next for
    the tradeoff this implies.
  - New tests: `test_identification.py` (ReferenceManager dedup/cooldown), test_identification_worker.py
    (job-to-result mapping, bounded-queue behavior, clean thread shutdown), and a rewritten
    `test_coordinator.py` (async submission/dedup/result-application/stale-result handling,
    using a deterministic `FakeWorker` double instead of a real thread).
- Camera device fallback and a live performance fix:
  - `capture/camera.py` now tries an external USB camera index before the built-in webcam
    index (`configs/camera.yaml`: `usb_device`, `webcam_device`), confirming each candidate
    with a real frame read (not just `isOpened()`) before accepting it, since a stale OS
    handle for a disconnected camera can report open while producing nothing;
  - fixed a crash (`cv::Mat` assertion, preceded by repeated OpenCV MSMF
    "Failed to select stream 0" warnings) caused by setting resolution/FPS *after* the
    confirmation read; changing capture properties mid-stream corrupted the frame buffer on
    Windows MSMF. Properties are now set before the first read. Regression test added
    asserting the set-before-read call order;
  - fixed: the assumed `usb_device: 1` / `webcam_device: 0` indices were reversed for the
    target machine; corrected to `usb_device: 0`, `webcam_device: 1` (see the real-hardware
    debugging entry above);
  - fixed unbounded per-frame retry cost: `IdentityCoordinator` used to re-run full SAM2+
    DINOv2 inference on every still-PENDING track on every single frame with no throttling,
    which could stall the camera loop indefinitely on one hard-to-segment object. This was
    superseded entirely by the asynchronous identification pipeline (see above), which moves
    that inference off the main thread and replaces frame-based retry throttling with the
    `ReferenceManager`'s cooldown/state machine.
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
  frame-sized masks. True means retain (kept as `Sam2Processor.post_process_masks` returns
  it; a prior inversion here was a real bug, see the real-hardware debugging entry above).
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
