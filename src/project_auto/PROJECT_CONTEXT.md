# Project AUTO Context

## Removal pipeline update — 2 October 2026

Permanent item removal is now owned by `events/removal_policy.py` and the coordinator.
Tracker REMOVE only retires a temporary ID. The coordinator observes bound items each
frame in memory, starts absence on the first missing frame, and probes plausible replacement
IDs before their normal ADD confirmation. Geometry nominates candidates; the existing
gallery ReID acceptance/margin decision verifies identity. A verified handoff transfers
the binding without REMOVED/RETURNED and reconciles a settled relocation as MOVED.

`scene_processor.yaml -> removal` configures 2s absence plus at most 3s extra while
a plausible check is pending, overlap/motion geometry, four candidate IDs per missing
episode and two outstanding handoff jobs. The worker has two reserved priority queue
slots in addition to the regular eight, with regular work served after two urgent jobs.
Jobs carry runtime IDs so stale completions cannot affect a newer job on a reused track ID.
Identity results also require a currently visible detection before creating or returning items.

Verified with 291 offline tests on 2 October; live camera validation and threshold calibration
remain required. No schema change. See [the full design and validation checklist](../../docs/14-removal-handoff.md).
This update supersedes the older tracker-timeout-to-removal descriptions below.

Source-reviewed architecture handoff: 26 September 2026 (ReID diagnostic entry point; builds on 23 September Reference Capture V2 and action-classified logging). Detailed contracts live in the [modular docs](../../docs/README.md), especially [architecture](../../docs/02-architecture.md), [data](../../docs/05-data.md), and [regions](../../docs/11-regions.md). TASKS holds current priorities and historical verification.

## MVP definition

Recorded 27 September 2026 as the working MVP. It is synthesised from the recorded scope, not a formally signed-off acceptance test, and its accuracy targets are not yet quantified.

**Goal:** a local, single-camera system that remembers individual physical objects and can answer where an object was last placed, including after it has left the camera's view.

**Required capabilities:**

1. **Perceive:** detect and track objects in a fixed tabletop scene.
2. **Identify:** assign each object a permanent individual identity, not only a class label (this cup, not "a cup").
3. **Re-identify:** recognise a returning object as the same item, not create a new one.
4. **Record change:** persist meaningful lifecycle events (ADDED, MOVED, REMOVED, RETURNED), never per-frame observations.
5. **Persist:** keep item identities, events and last locations across restarts in a local SQLite database.
6. **Answer:** respond to basic location queries (where an item is or was last seen, and what a named region contains). A console interface is sufficient.

**Acceptance demonstration** (one continuous run, with named regions already calibrated):

1. Place an object → it is ADDED as a new item in the correct region.
2. Move it to another named region → a MOVED event records both regions.
3. Remove it from view → it is marked REMOVED.
4. Query its location → the console reports its last-seen region.
5. Return it → it is re-identified as the original item (RETURNED), not added as a new one.
6. Restart the application and repeat step 4 → the answer survives the restart.

**Out of scope:** voice or natural-language interaction, a polished GUI, multiple cameras, 3D or depth positioning, cloud services, and guaranteed recognition of visually identical objects.

**Dependable-MVP gate:** the capabilities above are implemented end to end, but the MVP cannot be called dependable until (a) ReID `acceptance_threshold`/`margin_threshold` and score weights are calibrated from labelled diagnostic runs and hold on a held-out session, (b) Reference Capture V2 has run against a live camera, and (c) region calibration and console queries are validated on physical hardware. Quantified pass criteria (for example re-identification and false-match rates) are still to be defined; see `TASKS.md`.

## Purpose and architecture

Project AUTO remembers permanent physical objects and meaningful placement/presence changes in a fixed camera scene. SQLite history persists across sessions; detector track IDs and EventEngine bindings are temporary. The camera frame is a stable pixel coordinate system: no depth, homography or 3D calibration.

```text
MAIN: Camera -> YOLO11s/OpenVINO + BoT-SORT -> DetectionTracker
      -> IdentityCoordinator -> EventEngine -> DatabaseStore -> SQLite
      -> detection drawing / terminal queries / temporary region highlights

WORKER: bounded copied-frame jobs -> SceneProcessor -> SAM2 + DINOv2
        -> prototype shortlist + reference/descriptor matching -> result queue
        -> reference/prototype saves + private gallery refresh

SPATIAL TRANSACTION: event boxes -> store loads regions -> pure centroid resolver
        -> event region IDs/name snapshots/boxes/area fractions + current Item location

SEPARATE CALIBRATION: Camera -> clicked polygon -> validated geometry -> store Region

SEPARATE DIAGNOSTICS: reid_diagnostics_app -> run_app(overrides) -> same pipeline
        + ReidDiagnostics recorder (resolve jobs only) -> logs/reid_runs/<run_id>/
```

SAM/DINO matching and reference saves run on one background worker. The coordinator applies identity results and lifecycle writes on the main thread. Nonblocking queue calls do not make capture, detection, terminal input, or lifecycle SQL nonblocking. Initial gallery loading is outside per-job exception handling, so schema failures can stop the worker.

## Tracking and identity

Candidate confirmation takes two seconds of stillness (the clock restarts when the candidate's box centre moves more than candidate_stillness_tolerance_pixels, so a hand or carried object is not added until put down) with at most 15 cumulative missing frames per attempt; the 16th miss retires that attempt. Stable placement uses a 1.2x centered buffer. Exiting it emits a runtime-only `MOVE_START` signal (arms reference capture, see below; never persisted). A centroid must then remain within five pixels for one visible second before `MOVE_END` (also runtime-only) fires together with the persisted `MOVED` event in the same frame. Stable or moving tracks absent for two seconds emit REMOVE. Only detections with temporary track IDs enter this lifecycle.

ADD is temporal confirmation, not proof of new identity. The coordinator submits a quality-prefiltered resolve job. SceneProcessor retains True foreground SAM pixels, computes descriptors before background replacement, and prepares the RGB crop and normalized DINO embedding. NEW creates an Item/ADDED; a matched removed item produces RETURNED; a matched PRESENT/OCCLUDED item associates without an event. Removal releases the track binding after persistence. Unusable views defer; unresolved MOVED signals are currently dropped rather than replayed.

ReferenceManager limits outstanding work per track and applies resolve cooldowns: five seconds for unusable/conflicting views and 0.5 seconds for full queues. The worker owns the gallery snapshot and refreshes it after accepted reference saves.

Reference capture is sparse and event-driven (`ReferencePolicy`, `events/reference_policy.py`), not continuous/interval-based. A new item captures `initial_reference_count` (2) baseline references — the identity-resolve embedding is baseline reference 1 at no extra cost — then captures nothing further while it stays static; a still object earning no new references is intentional, not a bug. `DetectionTracker` emits runtime-only `MOVE_START`/`MOVE_END` signals (never persisted as ItemEvents, never forwarded to EventEngine) on each STABLE<->MOVING transition; `MOVE_START` arms up to `max_movement_reference_attempts` bounded candidate nominations spaced `candidate_retry_frames` apart, and `MOVE_END` nominates exactly one final settled-pose candidate exempt from that budget. Candidate evaluation is two-staged: a cheap Stage A gate (crop validity, box area, confidence, frame visibility, sharpness) runs on the main thread before a job is ever queued; an expensive Stage B gate (SAM mask score, mask occupancy, novelty against the existing gallery) runs in the worker after SAM/DINO, and a rejected candidate never reaches `add_reference_if_needed` or updates the prototype. `max_references_per_item` (8) is a hard ceiling with no replacement policy yet (`TODO(gallery)`). `mask_occupancy` is a mask-sanity measure, explicitly not occlusion detection (`TODO(occlusion)`). All of this is unit/integration tested (200 tests) but unverified against a live camera.

Diagnostic output is action-classified (`utils/logging.py::log_action()`: `EVENT`, `NEW`, `MATCH`, `ASSOC`, `AMBIG`, `REJECT`, `DEFER`, `CAPTURE`, `QUEUE`, `ERROR`, `REGION`, `SYSTEM`), each logged exactly once by the layer that owns its data, replacing the earlier per-file `[Coordinator]`/`[SceneProcessor]`/`[ReID]` prints that echoed one action through multiple lines.

## Matching contract

DINO prototypes shortlist three ranked items; missing prototypes remain eligible. Top-k reference similarities use k=3. When both supplementary scores are available, final score is 0.65 DINO + 0.20 color + 0.15 aspect, otherwise DINO alone. Aspect ratio is max(w/h, h/w); masked 32x32 Hue/Saturation histograms exclude background. The current score threshold is 0.55 and runner-up margin 0.15, both uncalibrated. Margin rejection currently flows to NEW rather than an explicit ambiguity state. The matcher returns every scored candidate and a decision reason (accepted/low_margin/below_threshold/gallery_empty) as diagnostic evidence only; the normal app writes no match CSV (the old logs/reid_match_log.csv writer was removed).

## ReID diagnostics (separate entry point)

`python -m project_auto.reid_diagnostics_app` runs the normal `run_app()` with overrides from `configs/reid_diagnostics.yaml`: a `utils/reid_diagnostics.py` recorder, `prototype_shortlist_size: 0` (whole gallery scored) and a separate database `data/diagnostics/project_auto_diagnostics.db`. Each run writes `logs/reid_runs/<run_id>/` with queries.csv (one row per resolve job, keyed by query_id), candidates.csv (every scored item's component scores), outcomes.csv (the applied ADDED/RETURNED/ASSOC/DEFER/... outcome) and raw/masked crops. Labelling is done offline by hand (ground_truth.csv + objects.csv, joined on query_id). The normal `project-auto` entry point never reads that config and passes no recorder, so its behaviour is unchanged; the recording hooks in coordinator/worker are inert without a recorder.

## Persistent and spatial memory

Four tables are defined: Item, ItemEmbedding, ItemEvent and Region. References store normalized vectors plus optional aspect/color descriptors; accepted saves update the item's normalized-mean prototype atomically. Model/preprocessing compatibility must be maintained; complete provenance/versioning is still absent.

Region stores a unique name, integer polygon, shoelace area and creation time. The geometry module validates simple nonzero-area polygons, uses inclusive-edge ray casting for the box centroid, chooses the smallest containing area and breaks ties by ID. It has no database imports or writes.

DatabaseStore resolves spatial evidence within existing lifecycle transactions. ADD/MOVED/RETURNED set current_box/current_region_id and last_seen_at. MOVED stores both ends; REMOVED stores source evidence and clears both live location fields. Source/destination area fractions use actual frame dimensions passed in memory by the coordinator. No boxes or region state are persisted per frame. The generic record_event method remains a historical append, not a live-state transition.

Event region IDs are canonical relationships; strings are historical name snapshots. Region deletion SET NULLs references without removing events or names. Current contents use Item state and PRESENT/OCCLUDED status; activity and distinct historical associations use event FKs. Legacy current-location fallback considers only the latest location event's destination FK and never revives a removed/unassigned location.

## Operator workflow and schema

Run python -m project_auto.region_calibration separately: click vertices, u undo, Enter/c finish, terminal name, q quit. Normal project-auto supports w item location and r region inspection, with in-memory highlights for 150 frames. There is no automatic calibration startup or in-window naming form. A changed camera position or actual capture resolution requires recalibration.

create_all creates tables but never adds missing columns. The observed startup failure was missing item_embeddings.aspect_ratio; eight descriptor/spatial columns were absent. Manual SQL was prepared and tested on a copy, not applied to the live file by this work. The database later disappeared and a file is present again; reset/recovery intent, retained history and current schema remain to confirm. No automatic migration/recreation has been added.

## Verification and open limits

The region implementation run passed 160 tests, including geometry, transactional location transitions, removal/deletion preservation, query fallback, and simulated calibration input. Earlier notes report successful live ReID scenarios. Physical-camera region calibration, current weighted-score accuracy, and performance at conf=0.18/imgsz=960 are not yet validated here. Installed Ultralytics defaults agnostic_nms to False; the project does not explicitly override it.

Main follow-ups: confirm database provenance/schema, live region lifecycle checks, a first labelled diagnostic run plus the join/analysis script for threshold calibration, structured logging, measured frame dropout/latency, worker startup failure visibility, capture-cap scheduling, generation-aware stale-result protection, complete occlusion semantics, and saved evidence files. LLM UI, world coordinates, polygon versions and materialized region statistics remain deferred.

## Class-agnostic identification and tracker configuration (2 October 2026)

Detector class labels are unreliable: one phone was labelled cell phone, mouse and apple depending on which face showed. No class denylist, class gate or class vote may decide whether a track is identified, added or removed; `Item.class_name` is descriptive text captured at creation only. Scenery and hands are handled by geometry, behaviour (stillness before ADD) and, later, appearance negatives from the ask-and-answer path. BoT-SORT settings are the repo-owned `configs/botsort.yaml` (Ultralytics defaults except `track_buffer: 45`). The ReID worker needs about 3.7 s per resolve job (4 threads, before the raise to 6), which bounds how many handoff probes can complete inside the removal deadlines.
