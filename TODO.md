# Project AUTO - To Do

Last updated 15 October 2026. Detailed evidence and historical sessions are in
`src/project_auto/TASKS.md`. This is the current work order.

**Current goal:** fix the camera placement, confirm the detector at that placement, then
demonstrate the MVP end to end. ReID calibration is dropped: performance is satisfactory.

## Completed

- [x] Clip recorder and detector comparison harness; seven clips recorded.
- [x] Compared YOLO11s, agnostic NMS, YOLOv8-s, YOLO26s, YOLOE and background control.
  YOLO11s agnostic had the lowest churn (23 handovers versus 65 baseline).
- [x] Implemented YAML `agnostic_nms: true` in the live detector (32 tests passed).
- [x] Git hygiene: weights, exports, caches, venv and runtime databases ignored.
- [x] ReID weights set by hand to 0.70 DINO / 0.25 colour / 0.05 aspect; acceptance 0.68,
  margin 0.06 (`memory/reid.py`, `configs/reid.yaml`). Evaluated as satisfactory.

## 1. Fix the camera placement (do first)

Observed: objects too close to the camera detect poorly; a far camera with full scene
context detects much more. The detector ranking was measured on existing clips, not at a
far placement.

- [ ] Mount the camera far with the whole table in view, then keep it fixed.
- [ ] Record 2-3 clips there with `scripts/record_clip.py`: still objects, pick/move/place,
  hands, empty table.
- [ ] Run `compare_detectors.py` on them (yolo11s, yolo11s_agnostic, yolov8s). Confirm
  agnostic NMS still wins and recall is better.
- [ ] Check crop size of objects at that distance; confirm identity still resolves.
- [ ] Review clip `215947` (still tracks 3 to 2) on the new clips.

## 2. Demonstrate the MVP (also the live validation)

- [ ] Recalibrate at least two regions at the final camera position.
- [ ] Normal app, two distinct objects: ADDED, MOVED between regions, REMOVED, last-location
  query, RETURNED as the original item, location query after restart. Wait for identity
  before moving.
- [ ] While there, watch: handoff causes no REMOVED/RETURNED; a settled move gives one
  MOVED; a genuine exit is removed in time; two baseline references then none while still.
- [ ] Record pass/fail and any unexpected NEW/AMBIG/DEFER or duplicate identities.
  Preserve existing memory; no database reset or automatic migration.

## 3. Save verified work in Git

- [ ] Review staged artifact removals and unstaged changes; commit in logical groups:
  YOLOv8 comparison, agnostic integration, ReID weights, artifact cleanup, task updates.
- [ ] Keep a local database backup and a model acquisition plan for a fresh clone.

## 4. Fix what the MVP run shows

- [ ] Measure quiet-CPU FPS and worker latency (old timings unreliable).
- [ ] Skip queued handoff probes whose candidate vanished, with lifecycle tests.
- [ ] Quiet not_still logging until visible time reaches the confirmation window.
- [ ] Tune BoT-SORT settings one at a time only if failures justify it.

## 5. Ask-and-answer identity flow (only if the MVP shows the need)

- [ ] Known gap: one face of a phone scores about 0.49 against another, so it becomes a
  duplicate item. Build pure identity policy, question registry, coordinator routing and UI.
- [ ] Apply confirmed identities/names, learn alternate views, non-item examples for hands.
  Review any schema change explicitly.

## 6. Later maintenance

- [ ] Sync docs (`06-configuration.md`, architecture, PROJECT_CONTEXT, TASKS) with the new
  weights/thresholds and agnostic NMS; rebuild/check the offline HTML portal.
- [ ] Writable pytest basetemp by default; recheck the lint backlog before fixing it.
- [ ] Worker startup reporting, model/reference provenance, unresolved movement replay.
- [ ] Tidy experimental detector defaults/flags (YOLO26s, YOLOE rows are historical).
- [ ] Deferred: ReID calibration and S0-S12 labelled sessions, full GUI, natural-language
  queries, 3D mapping, automatic migrations.
