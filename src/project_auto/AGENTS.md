# Project AUTO - Codex Instructions

## Project goal

Build a local computer-vision system that remembers where physical objects were last placed
on a table.

The target architecture is:

```text
Camera
-> YOLO11s/OpenVINO detection
-> ByteTrack or BoT-SORT tracking
-> permanent object identity association
-> state and placement-event decisions
-> SQLite persistence
-> query interface
```

The detector invokes Ultralytics BoT-SORT and exposes optional temporary track IDs. The live
application passes detections through the tracker, state machine, event engine, and SQLite
store. The verified lifecycle supports time-confirmed `ADD`, stable-placement `MOVED`, and
`REMOVE` after a stable or moving track has been absent for two seconds. Track-to-item
bindings are provisional session mappings owned by `EventEngine`, not permanent identity
recognition by themselves — permanent identity comes from ReID (SAM2 + DINOv2 + gallery
matching), which now runs and has been verified against real hardware: items are correctly
added, re-identified on reappearance, and returned. Match quality at the threshold margins is
still uncalibrated (see `TASKS.md`); do not describe it as tuned or production-accurate.

## Target hardware

- Lenovo Yoga Pro 7i
- Intel integrated graphics (CPU inference for SAM2/DINOv2; no GPU acceleration configured yet)
- OpenVINO-optimized detector inference
- Camera: prefers an external USB camera, falls back to the built-in/inbuilt webcam if it is
  not connected (`capture/camera.py`, `configs/camera.yaml`: `usb_device: 0`,
  `webcam_device: 1`, confirmed correct for this machine).
- Entirely local operation for the MVP

## Engineering conventions

- Python 3.10 or newer.
- Use type hints for public functions.
- Keep camera capture, inference, tracking, state decisions, and persistence separate.
- Configuration belongs in YAML rather than hard-coded constants.
- Use `pathlib.Path` for filesystem paths.
- Do not introduce a dependency without explaining why.
- Prefer small, testable components over one large application file.
- Never treat a tracker ID alone as permanent object identity.
- Never create database observations for individual video frames.
- Store meaningful state changes and events instead.
- Store evidence file paths in SQLite, not image/video binary data.

## Working agreement

- By default, change only one file per user prompt.
- Explain the intended file change before editing it.
- When building function by function, add only one explicitly approved code chunk at a time
  and explain it before moving on.
- After each main function or feature is verified, update `TASKS.md` in a separate approved
  step.
- Ask for approval before moving to the next function when working function by function.

## Before changing functional code

1. Read `PROJECT_CONTEXT.md`.
2. Read `TASKS.md`.
3. Inspect the relevant existing files.
4. Explain any architectural change before implementing it.
5. Ask before changing additional functional files.

## Verification

After modifying code:

- run focused tests for the changed behavior;
- run formatting and lint checks when available;
- report what changed and what remains unverified;
- do not mark work complete until verification passes.

## Current priority

Follow the `Current task` section in `TASKS.md`. ReID is now verified working end to end on
real hardware: items are added, correctly re-identified when they reappear (including under
a brand-new track ID), and returned items are persisted as RETURNED. Getting here required
fixing three real bugs, not just calibration — see `TASKS.md`'s "real-hardware ReID debugging
session" entry for the full account:
1. `configs/camera.yaml` had `usb_device`/`webcam_device` swapped.
2. `perception/segmenter.py` inverted SAM2's mask, discarding the object and keeping the
   background before DINOv2 ever saw it. **The SAM keep-mask is not inverted anymore** —
   `keep_mask = candidates[best]` matches `Sam2Processor`'s own True-means-object convention.
   Do not reintroduce an inversion here without strong evidence; it silently destroys ReID
   signal while still looking like it runs successfully.
3. `EventEngine.process_remove` used to leave a track's `_item_ids_by_track_id` binding in
   place forever after removal, permanently (not just temporarily) blocking
   `is_item_claimed()` from ever releasing that item for a future RETURNED/associate
   resolution on a new track. It now deletes the binding on removal.

Identification is asynchronous: `IdentityCoordinator` (`events/coordinator.py`) submits a
lightweight `IdentificationJob` and returns immediately; `IdentificationWorker`
(`events/identification_worker.py`) runs SAM2 + DINOv2 + gallery matching + persistence on
one background thread and reports back through a non-blocking result queue.
`ReferenceManager` (`events/identification.py`) guarantees at most one outstanding job per
track and replaces per-frame retrying with an explicit cooldown/state machine
(`bad_mask_cooldown_seconds`, `queue_full_retry_seconds`) — never reintroduce a synchronous
SAM/DINO call inside the main capture loop, or a resubmit-every-frame retry on a bad mask;
both were real production bugs here, not hypothetical ones. Never call scene processing, SAM,
ReID, or the database from the tracker itself. The event layer distinguishes new identities
(ADD), matched removed items (RETURNED), and already-present associations (no event). See
`TASKS.md` for current schema recovery, hardware region verification, weighted-score calibration,
performance measurement and logging priorities. `PROJECT_CONTEXT.md` now describes the current
async and spatial architecture; the detailed reference is `docs/11-regions.md`.

## Region-memory boundaries

Region geometry is pure pixel-space logic in `memory/regions.py`; keep SQL/session access in
DatabaseStore. EventEngine supplies meaningful event boxes; the store resolves polygons and
commits event IDs/name snapshots plus Item.current_box/current_region_id atomically. Removed
items have neither live field. Do not introduce per-frame location writes, tracker database
access, region event cascades, or name-based canonical associations. Region FKs use SET NULL;
event names remain historical snapshots.

Calibration runs separately through `project_auto.region_calibration` and reuses Camera.
Normal tracking uses w/r console queries and temporary in-memory highlights. The 160-test
implementation run includes simulated region UI and persistence checks, not physical-camera
verification. Current ReID 0.55 acceptance/0.15 margin and YOLO 0.18 confidence/960 image size
are configured values, not calibrated performance guarantees. Existing databases require
explicit inspection/migration; never silently delete, recreate or auto-migrate them.
