# Scope, assumptions, and limitations

## Current scope

The implemented product is a local, single-camera tabletop perception and event-memory prototype. It captures images, detects and tracks objects, proposes permanent identities from appearance, records addition/movement/removal/return events, and supports polygon calibration, console spatial queries, and temporary region highlights. It retains history across process restarts through SQLite; temporary tracker bindings are session-local.

## Assumption register

| ID | Assumption | Consequence if false |
| --- | --- | --- |
| A01 | Camera is fixed and actual capture resolution remains stable | Stored polygons and pixel placement buffers no longer describe equivalent locations; recalibrate |
| A02 | Objects receive usable YOLO detections and track IDs | Untracked or undetected objects do not enter the lifecycle |
| A03 | Appearance and lighting yield useful masks/embeddings | Identity may defer, split into duplicate items, or falsely match |
| A04 | Model and preprocessing are compatible across saved references | Similarity and prototypes may be invalid after model/preprocessing changes |
| A05 | Local devices, model files/cache, desktop, and disk are available | Startup or capture can fail; offline startup may lack weights |
| A06 | Workload fits one CPU worker and local SQLite | Queue deferral, latency, and frame competition can increase |
| A07 | Existing database schema matches the current models | create_all cannot repair old columns; gallery load or event writes may fail |
| A08 | Camera indexes are configured for the actual machine | Wrong device may open, or fallback may fail |
| A09 | A local trusted-user workflow is sufficient | Files and camera data have no application-level access-control layer |

These are operating assumptions, not validated guarantees for every deployment.

## Known limitations and follow-up work

| Limitation | Impact | Follow-up direction |
| --- | --- | --- |
| ReID threshold is provisional | Recognition quality at decision margins is unknown | Empirical same/different-item calibration and held-out evaluation; blocked on adding a ground-truth label to reid_match_log.csv |
| Reference-capture quality/novelty thresholds are provisional | min_reference_sharpness, min_mask_score, min_mask_occupancy, reference_novelty_threshold are uncalibrated starting estimates | Run against a live camera and tune from the CAPTURE/REJECT action-classified logs |
| Reference Capture V2 is untested on live hardware | Sparse/event-driven capture behavior (baseline stop, MOVE_START/MOVE_END nomination) is verified only by fakes/synthetic images | Run project-auto against a real camera and confirm the documented lifecycle |
| Movement before identity is dropped | Event history can omit an early movement | Define buffering/reconciliation policy |
| Delayed results have no track generation token | Numeric ID reuse can defeat active-ID checks | Add generation-aware correlation and tests |
| Initial identity and first reference are separate transactions/jobs | Newly created items may briefly lack matchable references | Review consistency and failure recovery |
| Reference gallery has no replacement policy at its cap | max_references_per_item simply refuses new references once reached | Add diversity-aware replacement (evict most redundant member) |
| mask_occupancy is a sanity check, not occlusion detection | Cannot distinguish a physically small object from a partially hidden one | Add true partial-occlusion reasoning with better evidence |
| Occlusion lifecycle is incomplete | Long occlusion can look like removal | Define visibility/occlusion policy |
| Migration system is absent | Older databases require deliberate migration | Version schema and preserve existing history |
| No complete measured benchmark | Capacity and latency cannot be promised | Implement the performance protocol |

## Out of scope for the current implementation

- Natural-language questions, voice interaction, a full graphical object-search UI, or a public API; temporary w/r console queries are implemented.
- Calibrated 3D positions, depth sensing, homography, robotic manipulation, region hierarchies or geometry versioning; 2D semantic polygons are implemented.
- Multi-camera fusion, cross-device synchronization, cloud-hosted inference, or multi-user service operation.
- Training or fine-tuning YOLO/SAM2/DINOv2 within the live application.
- Guaranteed identification of visually identical objects or unrestricted object classes.
- Production SLAs, automated schema upgrades, deployment packaging, and comprehensive monitoring.
- Automatic evidence video recording or a complete evidence retention/export workflow.

These boundaries describe current code, not permanent exclusions from the roadmap. The HTML documentation portal provides navigation to project information; it does not implement authentication, remote hosting, or the application's console object-query interface.

## Priority order suggested by current evidence

Stabilize schema compatibility and worker failure visibility; measure post-async performance; calibrate recognition; tighten capture scheduling and stale-result handling; validate live polygon calibration and spatial event/query behavior against the implemented memory layer. The task history remains the owner's working backlog, rather than a commitment to release dates.

Sources: [task history](../src/project_auto/TASKS.md), [coordinator](../src/project_auto/events/coordinator.py), [worker](../src/project_auto/events/identification_worker.py), [region geometry](../src/project_auto/memory/regions.py).

Additional spatial limits: current_box is the latest meaningful-event snapshot, not a live trajectory. An already-present ReID association makes no event or location write; early MOVED signals can be dropped while identity is unresolved. Calibration is not automatically offered on startup. Polygon editing/versioning, materialized region statistics, and automatic schema upgrades are deliberately deferred.
