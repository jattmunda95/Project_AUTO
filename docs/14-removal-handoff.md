# Item removal and ReID handoff

Implemented 2 October 2026. A temporary tracker ID can disappear while the physical object
remains visible under another ID. Permanent removal is therefore decided by the coordinator's
item-presence policy, rather than directly from the tracker's REMOVE signal.

## Runtime sequence

1. An identified item's owner ID is absent from the current detections. Start an in-memory
   absence window. Continue remembering its last detection, recent boxes, last meaningful
   placement box, and whether it was moving.
2. Nominate unbound IDs overlapping recent boxes. For a moving item, also allow bounded
   center displacement with comparable box area. Class names and calibrated regions do not
   determine eligibility. Remember nominated IDs across the missing window, even if they
   move out of that original area. A candidate ID is forgotten only after being unseen for
   `absence_seconds`; within that window its in-flight job and handoff target are kept, so a
   brief flicker neither discards the job nor submits a duplicate.
3. Reuse an outstanding resolve for that candidate, promoting queued work when possible.
   Otherwise submit a priority resolve without waiting for normal tracker ADD confirmation.
   SAM/DINO still run only on the worker. The existing gallery matcher retains its score and
   runner-up margin checks; overlap alone never establishes identity.
4. If the candidate is still visible and ReID accepts the missing item, transfer its exclusive
   binding. The old ID can retire later without removing the transferred item. No REMOVED or
   RETURNED is emitted. A stable new position produces one MOVED with the original placement
   as source; a handoff at essentially the same position produces no movement event.
5. Without a viable check, persist REMOVED once the item has been absent for two seconds.
   An outstanding check may extend this to a hard six-second deadline (2s plus 4s grace). The
   extension follows the outstanding check, not the candidate's visibility on the current frame,
   so a one-frame detector dropout cannot cancel it. Rejected, ambiguous, missing, or delayed
   candidates cannot extend that deadline repeatedly.
6. A match that arrives while its candidate is briefly invisible is held and applied when the
   candidate reappears, instead of being discarded as stale. If the candidate stays unseen past
   the window, the held match is dropped.
7. Results must match the current outstanding job ID. Retirement in the same frame invalidates
   results before they are applied. Resolve results also require current visibility. A match
   arriving after removal can produce a normal RETURNED if the new track has been confirmed.

The disappearance timer starts on the first missing frame. Timers are monotonic; event dates
are UTC. Policy state is session-local, while existing item/event persistence is unchanged.

## Ownership and bounded work

| Responsibility | Owner |
| --- | --- |
| Temporary detection IDs and movement signals | DetectionTracker / BoT-SORT |
| Item absence, candidate geometry, deadlines, handoff settling | RemovalPolicy |
| ReID scheduling, current visibility, stale results, handoff application | IdentityCoordinator |
| Reserved queue capacity and fair priority service | IdentificationQueue |
| SAM/DINO and gallery comparison | IdentificationWorker / SceneProcessor |
| Exclusive binding transfer and event persistence | EventEngine / DatabaseStore |

The default regular queue remains eight jobs, with two additional reserved priority slots.
The coordinator allows at most two outstanding handoff jobs, including reused jobs. After two
priority jobs the worker serves a regular job if one is waiting. Active inference is not
interrupted. There are at most four candidate IDs per missing episode, each submitted/reused
once for that episode. Ordinary confirmed-track retries retain their existing cooldowns.

Failed early probes do not create new items or learn reference crops. Successful handoffs
also do not immediately save the matching crop: subsequent movement/settled capture still
passes the established quality, consistency, and novelty checks.

No new database schema or per-frame database writes are needed. Detector confidence, NMS,
and the selected tracking backend are unchanged by this migration.

## Configuration

All new settings are in `configs/scene_processor.yaml` under `removal`:

| Setting | Initial value | Purpose |
| --- | --- | --- |
| absence_seconds | 2.0 | Minimum item absence before removal; also how long a candidate ID may go unseen before it is forgotten |
| reid_grace_seconds | 4.0 | Maximum additional wait for a plausible check |
| min_iou | 0.5 | Overlap nomination threshold |
| moving_distance_scale | 1.5 | Maximum moving-candidate distance in units of the old box's longer side |
| min_area_ratio | 0.4 | Smaller/larger box-area ratio for proximity candidates |
| max_candidates | 4 | Candidate IDs per missing episode |
| max_inflight | 2 | Outstanding handoff cap and reserved priority queue capacity |
| settled_seconds | 1.0 | Stable visibility needed after handoff |
| movement_tolerance_pixels | 5.0 | Center displacement tolerance for settling and relocation |

`configs/perception.yaml: removal_timeout_seconds` remains three seconds and only controls
temporary track retirement. It no longer controls permanent REMOVED events.

## Live finding: worker throughput versus ID churn (2 October 2026)

After the flicker fix, a second diagnostic session (run 20261002T085810Z) removed a phone that was still on the table. The tracker produced IDs 8, 15, 18, 19, 26 and 36 within seconds, each a handoff probe; each resolve job takes about 3.7 s on the worker (4 torch threads), so answers arrived roughly 4 to 11 s after submission. Item 1 was REMOVED at the 6 s deadline, then correct matches (0.57, 0.81, 0.80 against item 1) were discarded as `stale_job` (the candidate IDs had already vanished) or `not_visible`. The earlier live success had one candidate and an empty queue.

This is a throughput problem, separate from the flicker bug. Mitigations applied: `worker_torch_threads` 4 to 6 (reported to help a lot, not measured) and `track_buffer` 30 to 45 (fewer new IDs; untested). Not done: skip queued probes whose track has vanished before they reach the head of the queue; the remaining BoT-SORT settings in TASKS.md item 2.

The same session showed the phone being treated as a different object when its screen lit up (DINO about 0.38 against its screen-off reference), producing a new item. That is an appearance problem for the planned ask-and-answer path, not a removal bug.

## Verification and limits

The full offline suite passed 269 tests after the flicker fix, and 281 after the later stillness and tracker-config changes (19 of them, the `tmp_path` tests, only run when pytest has a writable temp folder; on this machine the default one raises a Windows `PermissionError`, so use `--basetemp`). New tests cover genuine exits,
same-ID recovery, different-ID handoffs before ADD, settled movement and stored source/destination
boxes, stationary handoffs, moving candidates without overlap, unrelated/ambiguous matches,
queue saturation and hard deadlines, vanished candidates, ownership protection, recycled IDs,
same-frame retirement, actual threaded worker priority order, and single-frame candidate dropout (no duplicate job, a match arriving during a dropout, and a dropout at the absence boundary). ReID responses use deterministic
test doubles; physical-camera performance and matching accuracy are not claimed.

Live checks to run:

1. Keep an object present while inducing an ID change near the same box. Expect ASSOC
   `action=handoff`, with no REMOVED/RETURNED or spurious MOVED.
2. Move it to a new position while its ID changes. Expect one settled MOVED for the original
   permanent item and updated region/location.
3. Remove it with no replacement detection. Expect one REMOVED after the absence window.
4. Put a different object near the old box. A rejected/ambiguous match must not transfer identity.
5. Repeat while reference work occupies the worker. Watch `reason=checking_handoff`; waiting
   must remain bounded and regular work must continue.
6. Cover an object with a hand. This design still cannot reliably distinguish true exit from
   full occlusion when no usable detection remains; dedicated occlusion reasoning is separate.

A live session before the flicker fix showed 9 of 9 handoff probes discarded as `stale_job` and
no `ASSOC action=handoff`, including a correct 0.646 match that was thrown away while the item was
removed. After the fix, one live camera session confirmed that ID-change handoffs succeed, every
REMOVED was a genuine removal, and a moved item produced a single MOVED. Still unchecked live: a
different object placed near the old box, hand occlusion, and a loaded worker. An object removed
before its ADD could apply logs NEW and then `DEFER reason=not_visible`, by design, since no item
exists yet.

Geometry can miss large jumps, especially when the ID disappears before any motion is observed.
ReID thresholds and the new timing/geometry thresholds remain uncalibrated. A false confident
ReID match can still produce a false handoff. Runtime job IDs protect asynchronous correlation;
they cannot detect an arbitrary identity swap inside an otherwise continuous tracker ID.
