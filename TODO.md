# Project AUTO - To Do

Last updated 17 October 2026. Detailed evidence and historical sessions are in
`src/project_auto/TASKS.md`. This is the current work order.

**Current goal:** fix the camera placement, confirm the detector at that placement, then
demonstrate the MVP end to end. ReID calibration is dropped: performance is satisfactory.

## Completed

- [x] Clip recorder and detector comparison harness; seven clips recorded.
- [x] Compared YOLO11s, agnostic NMS, YOLOv8-s, YOLO26s, YOLOE and background control.
  YOLO11s agnostic had the lowest churn (23 handovers versus 65 baseline).
- [x] Implemented YAML `agnostic_nms: true` in the live detector (32 tests passed).
- [x] Git hygiene: weights, exports, caches, venv, runtime databases, all video formats,
  `egg-info`, `.env` and editor files ignored; `egg-info` untracked.
- [x] Squashed three unpushed commits into one (`a79b416`) so the large files stay out of
  pushed history; `git gc` run. Push still pending (section 3).
- [x] ReID weights set by hand to 0.70 DINO / 0.25 colour / 0.05 aspect; acceptance 0.68,
  margin 0.06 (`memory/reid.py`, `configs/reid.yaml`). Evaluated as satisfactory.
- [x] Ask-and-answer flow built and wired (policy, question registry, overlay, thumbnails,
  confirmed views, schema, startup guard); live database migrated. 624 tests pass; not yet
  proven over a full live session.
- [x] 17 October: near-acceptance floor 0.55 -> 0.50 and plausible weight 0.25 -> 0.35, so a
  returning phone scoring about 0.54 now asks.

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
- [ ] Restart the app first (it must load the new ask settings), then check the console: a
  `?> ASK ... action=asked` line means the overlay should be drawing a question;
  `action=waiting code=...` means it was held back and says why. Note how many questions
  appear per session; if there are too many, tune `ask_threshold` or
  `item_pair_cooldown_seconds` in `configs/identity_policy.yaml`.

## 3. Save verified work in Git

- [ ] Run `git push origin main` (one squashed commit `a79b416`, a few hundred KB). The first
  push failed because `mobileclip2_b.ts` (242 MB) exceeded GitHub's 100 MB limit.
- [ ] After a successful push, free space: `git branch -D backup-before-squash`, then
  `git reflog expire --expire=now --all` and `git gc --prune=now`.
- [ ] Decide whether to untrack `notebooks/ReID_Tuning/` (old tuning images, CSVs and a
  `.pt`); it is still tracked and the `.pt` matches the new ignore rule.
- [ ] Keep a local database backup and a model acquisition plan for a fresh clone.

## 4. Fix what the MVP run shows

- [ ] Measure quiet-CPU FPS and worker latency (old timings unreliable).
- [ ] Skip queued handoff probes whose candidate vanished, with lifecycle tests.
- [ ] Quiet not_still logging until visible time reaches the confirmation window.
- [ ] Tune BoT-SORT settings one at a time only if failures justify it.

## 5. Persistent question queue (next build, designed 17 October, not started)

Questions should survive the object leaving and the app restarting, and be answered later in
the existing overlay. Design: `docs/15-ask-and-answer-plan.md`, "Next phase".

- [ ] Answer the open points: does "yes" last night cover the `WAITING` log label and the
  overlay-as-MVP-channel; do repeated "unconvincing" waits still give up and add NEW after 3;
  is 20 pending questions the right cap.
- [ ] Step 1 (needs your go): `pending_questions` table, manual SQL migration, startup guard.
  Before running it: stop the app, back up `data/project_auto.db` with its `-wal` and `-shm`.
- [ ] Step 2: pure question service (queue, merge at similarity 0.9, never merge tracks seen
  together, states, apply answers).
- [ ] Step 3: coordinator and overlay integration; poor views wait and add nothing if never
  good; `WAITING` log label; update TASKS.md.
- [ ] After that, not before: REST API and web page (separate shippable unit, internet access,
  auth), items and locations endpoints, feature-matching or vision-language checker.

## 6. Later maintenance

- [ ] Sync docs (`06-configuration.md`, architecture, PROJECT_CONTEXT, TASKS) with the new
  weights/thresholds and agnostic NMS; rebuild/check the offline HTML portal. They also do
  not yet describe the ask system (policy, registry, overlay, thumbnails, confirmed views),
  and `01`, `06`, `12` and README still quote the old ReID 0.55/0.15 values.
- [ ] Look into track 185 (matched item 2 repeatedly but deferred `item_already_claimed`),
  the many `stale_job` lines and track ids reaching ~438.
- [ ] Writable pytest basetemp by default; recheck the lint backlog before fixing it.
- [ ] Worker startup reporting, model/reference provenance, unresolved movement replay.
- [ ] Tidy experimental detector defaults/flags (YOLO26s, YOLOE rows are historical).
- [ ] Deferred: ReID calibration and S0-S12 labelled sessions, full GUI, natural-language
  queries, 3D mapping, automatic migrations.
