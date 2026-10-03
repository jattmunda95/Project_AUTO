# Project AUTO Tasks

## Current task

**2 October 2026 priority update:** validate the implemented item-level removal/handoff
pipeline on camera before continuing the older sequence below. Follow
`docs/14-removal-handoff.md`: unchanged-position ID switch, moved ID switch, true exit,
nearby different object, hand occlusion, and a loaded worker. Confirm ASSOC action=handoff
without REMOVED/RETURNED, one settled MOVED when applicable, and bounded true removal.
The old spatial-handover/deferred-REMOVED proposal in item 2 is now implemented in the
coordinator/pure removal policy; the detector/BoT-SORT tuning and true occlusion work remain.

**Live-verified 2 October 2026 (one camera session, after the flicker fix below):** ID-change
handoffs succeed (ASSOC action=handoff, item re-detected without REMOVED/RETURNED); every REMOVED
in the session was a genuine removal; a moved item produced exactly one MOVED; behaviour
otherwise matched expectations. Still unchecked live: a different object placed near the old
box, hand occlusion, and a loaded worker. An object removed before its ADD could apply logs
NEW then nothing, by design (no item exists yet); that branch now logs
`DEFER reason=not_visible`.

**Next steps (end of 2 October 2026), details in the section "2 October 2026 (later)" below:** (1) live-check the stillness gate, `track_buffer: 45` and `worker_torch_threads: 6`; (2) skip queued handoff probes whose track vanished; (3) quiet the `not_still` log; (4) remaining BoT-SORT settings one at a time; (5) Phase B ask-and-answer path starting with the pure identity policy; (6) commit the large uncommitted tree in logical chunks.

## Implemented and verified 2 October 2026: item-level removal and ReID handoff

- The coordinator now owns permanent-item absence. Tracker REMOVE only retires its ID;
  it no longer directly persists REMOVED. Default absence is 2s, with at most 4s additional
  grace for plausible handoff work (hard 6s deadline; `reid_grace_seconds: 4.0`). Configuration is under `scene_processor.yaml: removal`.
- Candidate nomination starts during the missing window using recent-box overlap, plus
  bounded proximity/size agreement for a moving item. Candidate IDs remain eligible if
  they subsequently move away. No class or region filtering. Up to four candidates per episode.
- Existing queued/running resolves are reused. New handoff jobs can bypass normal ADD
  confirmation, but cannot create an item until that normal confirmation occurs. Two
  reserved priority slots and two outstanding handoff jobs bound load; regular jobs receive
  service after two priority jobs. Active inference is never interrupted.
- A confident gallery match to the missing item transfers its exclusive binding; no
  REMOVED/RETURNED is emitted. Settling after handoff records one MOVED from the retained
  placement box. A failed/ambiguous/disappearing candidate cannot maintain presence indefinitely.
- Flicker fix (2 October 2026): a first live session showed 9 of 9 handoff probes discarded as
  `stale_job` and no handoff, because one frame of candidate dropout wiped its in-flight job and
  ended the grace. Candidates are now forgotten only after `absence_seconds` unseen, grace follows
  an outstanding check rather than current-frame visibility, and a match arriving during a
  dropout is held and applied when the candidate reappears. 269 offline tests pass (19 need a writable `--basetemp` on this machine).
- Runtime job IDs prevent stale completion on a newer job for the same numeric track ID.
  Same-frame retirement rejects results first, and an absent detection cannot create/return
  an item using a historical crop. The same still-confirmed ID can request ReID after an
  item timeout without waiting for another tracker ADD.
- Verification: full suite **291 passed**; lint passed on changed Python files. Tests include
  actual lifecycle tracker ID changes, in-memory SQLite event/location checks, threaded worker
  priority service, queue fairness/capacity, timeouts, failure paths, and stale job correlation.
  Models and camera were not exercised. Existing database history/schema were not modified.

## 2 October 2026 (later): stillness-gated ADD, class-agnostic cleanup, tracker config, live findings

Implemented (offline-verified, 281 tests pass with `--basetemp`; live behaviour of the first three not yet confirmed):
- **Stillness-gated ADD** (`perception/tracker.py`): ADD needs `candidate_confirmation_seconds` of stillness; a box centre moving more than `candidate_stillness_tolerance_pixels` (12.0, uncalibrated) from the still anchor restarts the clock. Logs `DEFER reason=not_still resets visible_s moved_px tolerance_px`, at most once per window. 10 new tests. Aim: stop hands and carried objects becoming items without any class or region rule. Does not stop a hand resting still for 2 s.
- **Class voter removed** (`perception/class_filter.py` and its tests deleted). Decision: identification is class-agnostic because labels are unreliable (one phone labelled cell phone, mouse, apple). Phase A of the earlier plan (denylist, settled-class gate) is superseded; `max_box_area_fraction` was set to 1.0 (disabled) after an uncalibrated 0.35.
- **`worker_torch_threads` 4 to 6** after a measured ~3.7 s per resolve job; reported to help a lot, not yet measured. Check YOLO fps.
- **`configs/botsort.yaml`** with `track_buffer: 45` (default 30); detector reads `perception.yaml: tracker_config`; Ultralytics confirmed to load it (max_frames_lost 45 versus 30).
- Tooling: ruff installed via the `dev` extra (also declares pytest); 29 pre-existing ruff errors elsewhere. A `--basetemp` entry in `pyproject.toml` (plus `.gitignore`) is not done.

Live findings (diagnostic runs 20261002T082907Z and 20261002T085810Z, crops inspected):
- Worker throughput versus ID churn removed a visible phone: see `docs/14-removal-handoff.md`.
- One phone appears as screen-off, screen-on and rear-camera views (DINO about 0.38 between faces) and became items 1, 2 ("apple") and 3 ("mouse"). The consistency gate (`min_reference_consistency: 0.50`) rejects the other faces (0.17 to 0.40), so the gallery cannot learn them. Needs the ask-and-answer path (user-confirmed references).
- A hand ("person") was added as an item and later re-recognised at 0.647. Hand-in-crop also lowers match scores.
- Stillness log: first-frame box jumps (49 px, 108 px under 0.2 s) are noise; track 48 reset 6 times in 3.2 s at 12.37 px against 12, the perpetual-reset case to confirm.

Next, in order: live checks above; skip queued probes for vanished tracks; log `not_still` only after `visible_s` passes the confirmation window; then Phase B Step 3 (pure identity policy; real numbers: 0.9339 same-face return, about 0.50 other face of the same phone), question registry, coordinator routing, UI, answer binding, with negatives for hands. Still unwritten: S0 to S12 operator checklist and the diagnostics analysis script. The generated HTML under `docs/` was not rebuilt from these Markdown changes.

## Earlier priority sequence (28 September 2026)

Updated 28 September 2026: order is MVP run-through -> settle detector -> Reference Capture V2 live -> ReID data collection and tuning -> occlusion handling.

1. Run the full MVP acceptance demonstration (see `PROJECT_CONTEXT.md`, "MVP definition") end to end on the real camera with 2 visually distinct objects, using the normal `project-auto` entry point and current settings: calibrate at least two named regions; place each object (ADDED, correct region); move one between regions (MOVED with both regions; wait for ADDED first, since a move before identity resolves is dropped); remove it (REMOVED); query its last location with `w`/`r`; return it (RETURNED as the original item, not a new one); restart the application and query again. Record which steps pass or fail and any unexpected NEW/AMBIG/DEFER lines. This is a smoke test before ReID tuning: it checks for structural failures (never-live region calibration and queries, weighted-score matching) and, in passing, gives a first live look at Reference Capture V2 (item 3).
2. Settle the detector before any ReID tuning, so thresholds are fitted once against the final crops: (a) implement YOLO class-agnostic NMS by passing `agnostic_nms` explicitly from `configs/perception.yaml` to `model.track()` instead of inheriting the installed Ultralytics default (False, class-aware), and decide its value; (b) measure frame dropout/latency with current YOLO confidence=0.18 and image_size=960. agnostic_nms inherits the inspected Ultralytics False default. No throughput/accuracy improvement is established yet. Freeze confidence, image_size and agnostic_nms once decided: any later change to which boxes survive changes the crops ReID sees and requires a labelled re-validation session.
   - 28 September evidence for (a): a live run showed five different BoT-SORT track IDs (48, 52, 81, 82, 83) resolving to the same physical item within seconds of each other, each rejected only by `item_already_claimed`/`stale_result` — consistent with class-aware NMS failing to suppress a duplicate box when YOLO's class label flickers between frames for one real object (observed real-object labels included `person`, `knife`, `dining table`, `cell phone`, none of which were the actual test objects). Sets `agnostic_nms=True` as the leading candidate, to confirm once implemented.
   - Partly done 2 October 2026: `configs/botsort.yaml` now exists and `perception.yaml: tracker_config` selects it; it equals the bundled defaults except `track_buffer: 45` (one variable at a time). `match_thresh`, `gmc_method` and `new_track_thresh` below are still the bundled values.
   - Original sub-task (file part done): add a repo-owned `configs/botsort.yaml` (originally `detector.py` passed the bare string `"botsort.yaml"`, which Ultralytics resolves to its own bundled copy inside `.venv` — untracked, and silently reverts on every reinstall). Tune for a fixed tabletop camera: `gmc_method: none` (global motion compensation is for a moving camera and can inject spurious motion here), `match_thresh: 0.9` (IoU floor 0.2 -> 0.1, doubling the tolerable inter-frame displacement before an ID switch), `track_buffer: 60` (keep a lost track's ID alive longer for re-acquisition), `new_track_thresh: 0.35`. Targets the recurring REMOVED-then-RETURNED-as-same-item pattern seen on fast pick-ups (still observed 28 September, e.g. `track=6`), caused by BoT-SORT losing IoU association on fast motion and the tracker treating a lost ID as a genuine departure. `with_reid` left off for now to keep one variable at a time. A pure motion-budget calculation showed this can reduce but not eliminate ID loss on fast moves — see item 2c.
   - New sub-task (deferred, needs task 6 first): a spatial-handover check in `perception/tracker.py`'s missing-track logic, split by the `status_before_missing` a track had when it vanished — STABLE (plain IoU/overlap re-check against the last known box is reliable) vs. MOVING (IoU alone cannot tell "picked up and carried away" from "tracker lost it mid-slide"; needs ReID evidence the tracker is architecturally forbidden to use, so this case routes to task 6's deferred-REMOVED design instead).
3. Run Reference Capture V2 against a real camera. Every V1-era behavior above was verified only with fakes/synthetic images/deterministic test doubles (200 tests passing); nothing has yet run live. Confirm: a static item captures exactly `initial_reference_count` (2) baseline references then stops; `MOVE_START` arms bounded `MOVEMENT` candidate nomination; `MOVE_END` yields one final settled-pose candidate; then back to idle. Watch the new action-classified `CAPTURE`/`REJECT` log lines for real rejection reasons.
4. Calibrate the new reference-quality thresholds against real data: `min_reference_sharpness` (30.0), `min_mask_score` (0.60), `min_mask_occupancy` (0.10), `reference_novelty_threshold` (0.93) are all uncalibrated starting estimates in `configs/scene_processor.yaml`. Watch `reason=redundant`/`low_mask_score`/`low_mask_occupancy`/`blurred` rejection rates specifically.
5. Requires tasks 2-4. Calibrate weighted ReID acceptance_threshold=0.55 and margin_threshold=0.15 using labelled same/different-item scores. The recording side now exists (see the 26 September entry below): run `python -m project_auto.reid_diagnostics_app` with 6-10 staged objects following the S0-S12 scenario protocol (enrolment, same-pose/rotated/relocated/re-lit/occluded/in-hand returns, look-alike swaps, unseen and late novel objects, track-loss motion, simultaneous returns, held-out session; restore `scripts/reid_snapshot.py restore post_enrolment` before each isolated block S1-S9, run continuously afterwards), add a `scenario` column to ground_truth.csv, then label each query offline in Excel (`ground_truth.csv`: query_id, true_object, expected_outcome NEW/SAME, label_quality, notes; `objects.csv`: true_object -> item_id). Still to do: run the first live diagnostic session, label it, and write the join/analysis script (genuine vs impostor score distributions, then offline weight fitting and threshold replay). Current weights are 0.65 DINO + 0.20 color + 0.15 aspect (hardcoded constants in `memory/reid.py`, not YAML); old 0.4 DINO-only thresholds are historical.
6. Design partial-occlusion and full-occlusion handling from the task 5 data (S5 occluded returns, S10 track-loss, `mask_occupancy`/`sam_score` in queries.csv), after ReID is tuned: full-occlusion reasoning depends on a trustworthy same-item decision, and partial-occlusion rules need measured score degradation rather than guesses. Scope: define OCCLUDED/PRESENT transitions (`store.mark_occluded()` exists but is never called; see TODO(occlusion) in event_engine.py) so an item hidden in place for more than `removal_timeout_seconds` is not recorded as REMOVED/RETURNED; and true partial-occlusion evidence (see TODO(occlusion) on PreparedReference.mask_occupancy, which cannot distinguish a small object from a hidden one). Build any query gate on signals already logged by the diagnostics recorder so it can be validated by offline replay on the labelled data, then confirm with one live session (S5 + S12); a gate changes which views reach ReID, so thresholds must be re-checked.
7. Confirm the intended live database and reset/recovery decision. Root cause of a prior "database is locked" investigation turned out to be manual record deletion from the live file outside the app, not concurrency — but current schema and retained history from that episode remain unverified. Inspect and back up before any deliberate migration.
8. Validate polygon calibration and ADD/MOVED/REMOVED/RETURNED region state on physical hardware, including overlap selection, current contents, historical names, and w/r highlights. Automated UI tests use simulated input.
9. To confirm: a 28 September live log showed `AMBIG ... margin_threshold=0.1500` (the pre-edit value) in the same apparently-continuous process as a `MATCH ... margin=0.1116` that is only possible under the new `margin_threshold=0.05` (`configs/reid.yaml`) — the two are mutually exclusive within one process, since `ReidConfig` loads once at startup. Re-run after a confirmed clean restart and check the `AMBIG` lines' printed `margin_threshold` value directly before trusting either the AMBIG-defer fix or the 0.05 relaxation are both in effect together.

## Implemented and verified 28 September 2026: fast-motion diagnosis, AMBIG-defer, worker thread cap

- Diagnosed the reported "REMOVED then RETURNED despite never leaving the frame" bug: BoT-SORT loses IoU-based track association on fast motion and switches track ID; `perception/tracker.py`'s missing-track timeout only checks "is this track ID present", so it cannot distinguish that from a genuine departure, and unconditionally emits REMOVE after `removal_timeout_seconds`. `removal_timeout_seconds` and `candidate_confirmation_seconds` were both 2.0s, which guaranteed REMOVE fired before a new track's identity could even resolve. Raised `removal_timeout_seconds` to 3.0 (`configs/perception.yaml`) to remove that race as a confound before further tracker changes; does not fix ID switching itself. See Current task item 2 for the full three-layer fix plan (detector/tracker settings, spatial handover check, deferred-REMOVED).
- Added lightweight `frame_perf` instrumentation to the main loop (`app.py`): logs fps/detect_ms/track_ms/coordinator_ms/frame_ms every 60 frames. Measured steady-state ~20-22 fps on the target hardware (22 logical cores), detect_ms (YOLO+BoT-SORT) dominating; periodic dips to ~15-16 fps correlate exactly with `CAPTURE`/`QUEUE` log lines. Diagnosed as CPU oversubscription: OpenVINO (main thread) and PyTorch (background identification worker) each independently default to sizing their own thread pool to the full logical core count, so concurrent bursts thrash the scheduler instead of running either at full speed. Marked `TODO(perf)` for removal once task 2(b) is done.
- Fixed a duplicate-identity bug confirmed live (same physical iPhone re-added produced three separate permanent items): `scene_processor.py` flattened both "genuine new item" and "ambiguous match" (score clears `acceptance_threshold` but fails `margin_threshold`) into the same `"new"` decision — a gap the code already documented via a `TODO(UI)` comment. Changed a `low_margin` decision to `"pending"` instead, reusing the existing PENDING/DEFER retry-with-cooldown path rather than committing to a wrong duplicate; a later, more decisive frame gets a real chance to resolve correctly first. Also fixed `identification_worker.py` hardcoding `failure_reason="unusable_reference"` for every PENDING result, which would have hidden `low_margin` as the actual defer reason in logs. Known accepted trade-off: a track that is genuinely, permanently ambiguous will now retry forever rather than resolve wrongly — same behavior class the pipeline already accepts for a chronically bad mask. Tests: `test_scene_processor.py` split into an AMBIG-defers-as-pending case and a genuine-non-match-still-becomes-new case. Full suite: 229 passed.
- Lowered `margin_threshold` 0.15 -> 0.05 (`configs/reid.yaml`) as an interim relaxation, not a calibrated value: 0.15 was rejecting confident, correct same-item matches (observed margin=0.0671 on a real repeat match) as AMBIG; now that AMBIG defers instead of creating a duplicate, an overly strict margin mainly cost correct matches an extra retry rather than a safety violation, so relaxing it is lower-risk pending real calibration (task 5). Live-run confirmation still pending — see Current task item 9.
- Added `worker_torch_threads: 4` (`configs/scene_processor.yaml`), threaded through `IdentityCoordinator` -> `IdentificationWorker`, which calls `torch.set_num_threads()` once at worker-thread startup before any SAM2/DINOv2 call, to address the oversubscription dips above. Untested live: `torch.set_num_threads` can raise if called after the process's torch threadpool has already done real parallel work, which the test suite (mocked scene_processor/store) cannot exercise — first live run after this change should be watched for a startup exception. Full suite: 229 passed (no test exercises the real worker thread with real torch models).

## Implemented and verified 26 September 2026: ReID diagnostic entry point

- New separate entry point `project_auto/reid_diagnostics_app.py` (`python -m project_auto.reid_diagnostics_app`; `project-auto-diagnostics` after reinstalling the package). It calls the normal `app.run_app()` with three overrides from `configs/reid_diagnostics.yaml`: a `ReidDiagnostics` recorder, `prototype_shortlist_size: 0` (whole-gallery scoring, so the true item's score is always recorded), and a separate database `data/diagnostics/project_auto_diagnostics.db`. All other settings come from the normal YAML files, so the same pipeline is measured.
- The normal `project-auto` entry point is behaviourally unchanged: `run_app()` with no arguments uses `configs/` as-is (shortlist 3), the normal database, and no recorder. The only difference is that the old `logs/reid_match_log.csv` writer (`ReidConfig.match_log_path`) was removed; it recorded only winner/runner-up scores with no join key, so it could not support ground truth.
- New standalone module `utils/reid_diagnostics.py` (imports nothing from the pipeline). Per run it writes `logs/reid_runs/<run_id>/` containing `queries.csv` (one row per resolve job: query_id, frame/track/box, detector confidence, SAM score, mask occupancy, gallery size, decision MATCH/AMBIG/NEW/DEFER, decision_reason, matched item, thresholds, weights, config hash, git commit), `candidates.csv` (one row per query x scored item: final/DINO/color/aspect scores, rank, reference count, DINO-only fallback flag), `outcomes.csv` (what the main thread applied: ADDED/RETURNED/ASSOC/DEFER/DEFER_CLAIMED/ERROR/DROPPED/STALE) and `crops/` (raw and masked PNG per query). Resolve jobs only; capture jobs never reach the matcher. Write failures are logged and swallowed, never failing a job.
- Plumbing (inert when no recorder is passed): `ReidMatch` now carries every scored `CandidateScore` and a `decision_reason` (accepted/low_margin/below_threshold/gallery_empty), both excluded from equality; `IdentityDecision` passes them through and gives PENDING a reason (no_mask/mask_too_small); `IdentificationJob` gains query_id/frame_index and resolve jobs now carry the detection; `IdentificationResult` echoes query_id. Scoring and accept/margin logic are unchanged.
- `logs/` and `data/diagnostics/` added to `.gitignore`; the old tracked `logs/reid_match_log.csv` was removed with `git rm`.
- New tests: `test_reid_diagnostics.py` (8), `test_reid_diagnostics_app.py` (2, including a guard that `reid.yaml` keeps shortlist 3 and no diagnostic keys), plus reid/scene-processor/worker/coordinator additions. Full suite: 228 passed. Neither entry point has been run against a live camera since this change.
- Not yet done: `docs/06-configuration.md` still lists `match_log_path` and does not describe `reid_diagnostics.yaml`; the labelling join/analysis script does not exist yet.

## Implemented and verified 23 September 2026: Reference Capture V2 and action-classified logging

- Replaced continuous/time-based reference capture (`capture_interval_seconds`, `_last_capture_at`, flat `reference_target_count`) with a sparse, event-driven `ReferencePolicy` state machine (`events/reference_policy.py`): `NEEDS_INITIAL -> IDLE -> ARMED -> EXHAUSTED`. A newly added item captures 2 baseline references (the identity-resolve embedding is baseline reference 1 at no extra cost, plus one more) then captures nothing further while static.
- Added explicit runtime `MOVE_START`/`MOVE_END` signals to `TrackSignalType` (tracker.py), emitted exactly once per STABLE<->MOVING transition. These are deliberately never forwarded to EventEngine/persisted as ItemEvents; `MOVED` (completed, meaningful placement change) remains the only persisted location semantic and can co-occur with `MOVE_END` in the same frame's signal list without being the same thing. `DetectionTracker.update()`'s internal methods changed from `TrackSignal | None` to `list[TrackSignal]` to support two signals firing on one frame (`MOVE_END` + `MOVED` together).
- `MOVE_START` arms a bounded number (`max_movement_reference_attempts`, default 3) of candidate nominations, spaced `candidate_retry_frames` apart, each first passing a cheap Stage A gate (crop validity, box area, detector confidence, frame visibility, sharpness — computed on the main thread, no SAM/DINO) before ever being queued to the worker. `MOVE_END` nominates exactly one final "settled pose" candidate, exempt from the attempt budget since it's a one-shot opportunity.
- Added Stage B (expensive, worker-side) gate in `identification_worker.py`: mask score -> mask occupancy -> novelty-vs-existing-gallery -> save. A rejected candidate returns before `add_reference_if_needed`, so it never touches the item's prototype. Baseline (`INITIAL`) candidates are exempt from novelty gating (two similar baseline views are expected).
- Added `perception/descriptors.py::sharpness()` (Laplacian variance, measures only, no built-in threshold) and `frame_visibility_ratio()`/`clip_box_to_frame()`, which replace the old "reject any box touching the frame edge" rule with a visible-fraction-of-predicted-box ratio (`min_reference_frame_visibility`, 0.80 starting value). The full predicted box is kept separate from its image-clipped box; clipping happens only when indexing pixels, never before measuring visibility.
- `PreparedReference` now carries `sam_score` and `mask_occupancy` (masked foreground pixels / segmented box area) as evidence for the capture path only; `prepare_reference` itself stays permissive so identity *resolution* keeps working on imperfect views. `mask_occupancy` is explicitly a mask-sanity measure, not occlusion detection (`TODO(occlusion)` — cannot distinguish a small object from a partially hidden one).
- Added a hard per-item reference ceiling, `max_references_per_item` (renamed from `reference_target_count`); at the cap, new references are simply refused (`TODO(gallery)` for future diversity-aware replacement, not implemented).
- Replaced file-tagged diagnostic `print()`s (`[Coordinator]`, `[SceneProcessor]`, `[ReID]`) with a single action-classified logging module, `utils/logging.py::log_action()`. Fixed taxonomy: `EVENT`, `NEW`, `MATCH`, `ASSOC`, `AMBIG`, `REJECT`, `DEFER`, `CAPTURE`, `QUEUE`, `ERROR`, `REGION`, `SYSTEM` — each with a distinct ASCII sigil (no Unicode, safe on any Windows codepage) and ANSI color. Each action now logs exactly once, at the layer that owns its data, instead of echoing through 3-4 files per decision.
- New tests: `test_reference_policy.py` (14), `test_descriptors.py` (20, covering frame-visibility geometry and sharpness), plus 4 new coordinator integration tests and updated tracker/coordinator/worker tests for the new signatures. Full suite: 200 passed. None of this has run against a live camera yet — see Current task #3.
- `pytest` was not installed in `.venv` (a standing gap); installed it, and confirmed the wider test-run `PermissionError`s on `tmp_path` are a Windows/OneDrive temp-directory locking artifact (same pattern as the earlier database-locking investigation), not logic failures — resolved by pointing `--basetemp` at a writable directory.

## Implemented and verified in the region work

- Pure simple-polygon validation, shoelace area, inclusive ray casting, centroid resolution, smallest-area overlap selection and stable ID tie-breaks.
- Region table; live Item region/box; event region FKs, historical names and separate normalized source/destination areas. SET NULL region deletion preserves history.
- Lifecycle transactions update meaningful spatial snapshots; removal clears live location. Tracker and geometry remain database-independent.
- Region CRUD and read-only current contents/location, last/recent activity, historical association, and item-history APIs.
- Separate calibration entry point, shared polygon drawing, terminal w/r queries and temporary highlights. Calibration is not automatic normal startup.
- Full implementation suite: 160 tests passed on 14 September; no fresh hardware validation implied. Schema assertion updated from three to four tables with approval. Ruff unavailable.
- Weighted descriptor preparation, storage, gallery fields and margin check are present in current source (the CSV diagnostics of that time were later replaced by the 26 September diagnostic entry point). Focused accuracy and weighted/margin regression coverage still need review.
- Modular Markdown/HTML docs, README and PROJECT_CONTEXT updated to represent current async and spatial contracts.

## Next

- Write the offline join/analysis script for labelled diagnostic runs (queries + candidates + outcomes + ground_truth + objects): genuine vs impostor score distributions per component, offline weight fitting, acceptance/margin threshold replay, and a held-out second session. Consider moving the 0.65/0.20/0.15 weights into YAML once fitted.
- Update `docs/06-configuration.md` for the removed `match_log_path` and the new `configs/reid_diagnostics.yaml`.
- Write the S0-S12 ReID scenario protocol (designed 27 September; see Current task 5) into docs as an operator checklist with an action-log template, before the first labelled session.
- Add diversity-aware gallery replacement once max_references_per_item is regularly hit in practice (see TODO(gallery) in identification_worker.py); V1 just refuses new references at the cap.
- Improve worker startup/schema failure reporting; initial gallery failure currently bypasses per-job error handling.
- Avoid duplicate DINO inference in the nonempty-gallery resolve path; measure before optimizing further.
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
