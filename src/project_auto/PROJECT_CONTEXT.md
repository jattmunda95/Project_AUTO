# Project AUTO Context

Source-reviewed architecture handoff: 23 September 2026 (Reference Capture V2 and action-classified logging). Detailed contracts live in the [modular docs](../../docs/README.md), especially [architecture](../../docs/02-architecture.md), [data](../../docs/05-data.md), and [regions](../../docs/11-regions.md). TASKS holds current priorities and historical verification.

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
```

SAM/DINO matching and reference saves run on one background worker. The coordinator applies identity results and lifecycle writes on the main thread. Nonblocking queue calls do not make capture, detection, terminal input, or lifecycle SQL nonblocking. Initial gallery loading is outside per-job exception handling, so schema failures can stop the worker.

## Tracking and identity

Candidate confirmation takes two seconds with at most 15 cumulative missing frames per attempt; the 16th miss retires that attempt. Stable placement uses a 1.2x centered buffer. Exiting it emits a runtime-only `MOVE_START` signal (arms reference capture, see below; never persisted). A centroid must then remain within five pixels for one visible second before `MOVE_END` (also runtime-only) fires together with the persisted `MOVED` event in the same frame. Stable or moving tracks absent for two seconds emit REMOVE. Only detections with temporary track IDs enter this lifecycle.

ADD is temporal confirmation, not proof of new identity. The coordinator submits a quality-prefiltered resolve job. SceneProcessor retains True foreground SAM pixels, computes descriptors before background replacement, and prepares the RGB crop and normalized DINO embedding. NEW creates an Item/ADDED; a matched removed item produces RETURNED; a matched PRESENT/OCCLUDED item associates without an event. Removal releases the track binding after persistence. Unusable views defer; unresolved MOVED signals are currently dropped rather than replayed.

ReferenceManager limits outstanding work per track and applies resolve cooldowns: five seconds for unusable/conflicting views and 0.5 seconds for full queues. The worker owns the gallery snapshot and refreshes it after accepted reference saves.

Reference capture is sparse and event-driven (`ReferencePolicy`, `events/reference_policy.py`), not continuous/interval-based. A new item captures `initial_reference_count` (2) baseline references — the identity-resolve embedding is baseline reference 1 at no extra cost — then captures nothing further while it stays static; a still object earning no new references is intentional, not a bug. `DetectionTracker` emits runtime-only `MOVE_START`/`MOVE_END` signals (never persisted as ItemEvents, never forwarded to EventEngine) on each STABLE<->MOVING transition; `MOVE_START` arms up to `max_movement_reference_attempts` bounded candidate nominations spaced `candidate_retry_frames` apart, and `MOVE_END` nominates exactly one final settled-pose candidate exempt from that budget. Candidate evaluation is two-staged: a cheap Stage A gate (crop validity, box area, confidence, frame visibility, sharpness) runs on the main thread before a job is ever queued; an expensive Stage B gate (SAM mask score, mask occupancy, novelty against the existing gallery) runs in the worker after SAM/DINO, and a rejected candidate never reaches `add_reference_if_needed` or updates the prototype. `max_references_per_item` (8) is a hard ceiling with no replacement policy yet (`TODO(gallery)`). `mask_occupancy` is a mask-sanity measure, explicitly not occlusion detection (`TODO(occlusion)`). All of this is unit/integration tested (200 tests) but unverified against a live camera.

Diagnostic output is action-classified (`utils/logging.py::log_action()`: `EVENT`, `NEW`, `MATCH`, `ASSOC`, `AMBIG`, `REJECT`, `DEFER`, `CAPTURE`, `QUEUE`, `ERROR`, `REGION`, `SYSTEM`), each logged exactly once by the layer that owns its data, replacing the earlier per-file `[Coordinator]`/`[SceneProcessor]`/`[ReID]` prints that echoed one action through multiple lines.

## Matching contract

DINO prototypes shortlist three ranked items; missing prototypes remain eligible. Top-k reference similarities use k=3. When both supplementary scores are available, final score is 0.65 DINO + 0.20 color + 0.15 aspect, otherwise DINO alone. Aspect ratio is max(w/h, h/w); masked 32x32 Hue/Saturation histograms exclude background. The current score threshold is 0.55 and runner-up margin 0.15, both uncalibrated. Diagnostic CSV output is configured at logs/reid_match_log.csv. Margin rejection currently flows to NEW rather than an explicit ambiguity state.

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

Main follow-ups: confirm database provenance/schema, live region lifecycle checks, threshold calibration, structured logging, measured frame dropout/latency, worker startup failure visibility, capture-cap scheduling, generation-aware stale-result protection, complete occlusion semantics, and saved evidence files. LLM UI, world coordinates, polygon versions and materialized region statistics remain deferred.
