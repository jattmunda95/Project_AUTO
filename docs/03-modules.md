# Module reference

All package paths below are relative to `src/project_auto`. Modules separate capture, perception, lifecycle decisions, orchestration, and persistence. The production entry point is `project-auto`, not the research notebooks.

For the visual view, open the [complete project map](12-project-map.md). The [Python and file index](13-source-index.md) lists every class, function, method and supporting file.

## Application and capture

| Module | Public entry points | Responsibility and boundary |
| --- | --- | --- |
| [main.py](../src/project_auto/main.py) | main | Console entry point; delegates to run_app |
| [app.py](../src/project_auto/app.py) | run_app | Loads services, capture/display, w/r console queries and temporary region highlights; owns no region SQL. Optional reid_config/database_path/diagnostics arguments are used only by the diagnostic entry point |
| [reid_diagnostics_app.py](../src/project_auto/reid_diagnostics_app.py) | run_diagnostics, main | Separate diagnostic launch; overrides shortlist and database, constructs recorder, delegates to run_app |
| [region_calibration.py](../src/project_auto/region_calibration.py) | run_calibration | Separate live-camera polygon editor; reuses Camera and store; terminal naming |
| [reid_diagnostics_app.py](../src/project_auto/reid_diagnostics_app.py) | run_diagnostics, main | Separate ReID ground-truth entry point; runs run_app with a recorder, shortlist override and separate database from reid_diagnostics.yaml |
| [region_queries.py](../src/project_auto/region_queries.py) | query_regions | Prints item location or region contents/activity/history; returns highlight IDs |
| [capture/camera.py](../src/project_auto/capture/camera.py) | CameraConfig, Camera.open/read/close | Configures properties before the first read; verifies devices in fallback order; returns BGR frames; supports a context manager |

## Perception

| Module | Public types / operations | Input and output |
| --- | --- | --- |
| [perception/detector.py](../src/project_auto/perception/detector.py) | Detection, YoloDetector.detect | BGR frame to class/confidence/xyxy box/optional temporary ID; loads or exports OpenVINO model; runs persistent BoT-SORT |
| [perception/tracker.py](../src/project_auto/perception/tracker.py) | DetectionTracker.update, TrackSignal, TrackStatus | Detection list to immutable ADD/MOVED/REMOVE/MOVE_START/MOVE_END signals (0-2 per track per frame); ignores detections without IDs; maintains candidate, stable, moving, and missing lifecycle. MOVE_START/MOVE_END are runtime-only, never persisted |
| [perception/segmenter.py](../src/project_auto/perception/segmenter.py) | SegmenterConfig, SamSegmenter.segment, Segmentation | uint8 BGR frame plus integer boxes to frame-sized boolean masks and scores; True retains foreground; selects best finite nonempty mask |
| [perception/scene_processor.py](../src/project_auto/perception/scene_processor.py) | SceneProcessor.prepare_reference/process, PreparedReference, IdentityDecision | Prepares masked RGB crop, normalized embedding, aspect ratio and foreground color histogram; proposes new/existing/pending; owns no scheduling or persistence |

Tracker confirmation lasts two seconds with at most 15 cumulative candidate misses per attempt. A 16th miss resets the attempt. Stable placement uses a centered buffer, and movement must stop visibly for the configured interval before MOVED. Absence of a stable or moving track for two seconds emits REMOVE. These are heuristics in image coordinates, not physical motion measurement.

## Orchestration and events

| Module | Main contracts | Responsibility |
| --- | --- | --- |
| [events/coordinator.py](../src/project_auto/events/coordinator.py) | IdentityCoordinator.handle_frame/shutdown | Applies results, submits resolve/capture jobs, schedules retries, routes lifecycle events, guards active claims, drives ReferencePolicy from MOVE_START/MOVE_END |
| [events/reference_policy.py](../src/project_auto/events/reference_policy.py) | ReferencePolicy, ReferencePolicyState, CandidateKind, CandidateDecision | Sparse/event-driven reference-capture scheduling: baseline-then-idle for a static item, movement-triggered bounded candidate nomination. Pure bookkeeping over caller-supplied frame numbers; no SAM/DINO/clock/persistence |
| [events/identification.py](../src/project_auto/events/identification.py) | IdentificationJob, IdentificationResult, IdentificationState, ReferenceManager | Immutable cross-thread message containers (now carrying candidate_kind) and per-track bookkeeping; queue eligibility and cooldowns |
| [events/identification_worker.py](../src/project_auto/events/identification_worker.py) | IdentificationWorkerProtocol, IdentificationWorker.start/stop/submit/poll_results | One worker, bounded jobs, unbounded results, gallery snapshot, model work, the expensive reference-quality/novelty gate, and reference persistence |
| [events/event_engine.py](../src/project_auto/events/event_engine.py) | EventEngine.process_signal/process_add/process_move/process_remove/process_return | Resolves session track bindings and performs atomic lifecycle store operations; rejects conflicting claims; MOVE_START/MOVE_END are never routed here |

Worker exceptions during a job become error results. Initial gallery loading is synchronized with start(), and failures propagate to its caller. Shutdown uses a stop event and nonblocking sentinel attempt.

## Memory and presentation

| Module | Main contracts | Responsibility |
| --- | --- | --- |
| [memory/reid.py](../src/project_auto/memory/reid.py) | ReidConfig, GalleryEntry, CandidateScore, ReidMatch, ReidMatcher | Loads DINOv2 once, normalizes embeddings, ranks prototypes and reference similarities; returns every scored candidate and a decision reason as evidence; DB-agnostic and writes no files |
| [memory/models.py](../src/project_auto/memory/models.py) | Region, Item, ItemEmbedding, ItemEvent, ItemStatus, ItemEventType | Typed SQLAlchemy schema, constraints, indexes, relationships |
| [memory/store.py](../src/project_auto/memory/store.py) | DatabaseStore | Short-lived sessions, lifecycle/spatial transactions, region CRUD and queries, vector/descriptor persistence, prototypes and gallery snapshots |
| [memory/state_machine.py](../src/project_auto/memory/state_machine.py) | StateDecision, decide_track_signal, decide_return_signal | Maps supported signals to status and event type; does not infer identity |
| [utils/drawing.py](../src/project_auto/utils/drawing.py) | draw_detections, draw_regions | Returns annotated copies; polygons use translucent fills, outlines, names and optional BGR colors |

## Descriptors and supporting components

[perception/descriptors.py](../src/project_auto/perception/descriptors.py) implements pure supplementary descriptors. Aspect ratio is max(width/height, height/width), so a 90-degree box rotation has the same ratio; it is not absolute size. Foreground RGB pixels produce a normalized 32 x 32 Hue/Saturation histogram before background replacement. Color similarity is 1 minus Bhattacharyya distance; aspect similarity uses Gaussian falloff with sigma 0.25. These descriptors are now wired through preparation, worker saves, gallery loading, and matching. Calibration/held-out accuracy remains unverified.

Also in `descriptors.py`: `sharpness()` (Laplacian variance, measures only, no built-in threshold) and `frame_visibility_ratio()`/`clip_box_to_frame()` (visible fraction of a *predicted*, unclipped box against the image bounds — the predicted box is never overwritten by its clipped version before measuring). Both are cheap, main-thread-safe geometry/pixel operations used by the reference-capture Stage A gate; neither invokes SAM or DINO.

The matcher first shortlists by DINO prototypes. It averages top-k scores separately for DINO, valid color references, and valid aspect references. With both supplementary scores available, final score = 0.65 * DINO + 0.20 * color + 0.15 * aspect; otherwise it falls back to DINO alone. Acceptance requires the configured score threshold and best-minus-runner-up margin; a single candidate has no margin competitor. ReidMatch also carries every scored candidate (final/DINO/color/aspect scores and reference count) and a decision reason (accepted, low_margin, below_threshold, gallery_empty); these are excluded from equality and only recorded by the diagnostic entry point. Margin rejection currently yields no accepted identity, which SceneProcessor interprets as NEW; an ambiguity/confirmation flow is still deferred.

| Path | Status | Usage |
| --- | --- | --- |
| [utils/reid_diagnostics.py](../src/project_auto/utils/reid_diagnostics.py) | Implemented; optional | Per-run query/candidate/outcome CSVs and raw/masked crops; called by worker/coordinator only when a recorder is supplied |
| [memory/regions.py](../src/project_auto/memory/regions.py) | Implemented, pure geometry | RegionGeometry, validate_polygon, polygon_area, point_in_polygon, resolve_region, box_area_fraction; no SQL/session imports |
| [scripts/define_regions.py](../scripts/define_regions.py) | Thin wrapper | Delegates to region_calibration.run_calibration |
| [display/viewer.py](../src/project_auto/display/viewer.py) | Empty | Actual display currently lives in app.py |
| [utils/logging.py](../src/project_auto/utils/logging.py) | Implemented | `log_action(category, **fields)`: fixed action taxonomy (EVENT/NEW/MATCH/ASSOC/AMBIG/REJECT/DEFER/CAPTURE/QUEUE/ERROR/REGION/SYSTEM), one ASCII-sigil/ANSI-color line per action, emitted once by the layer that owns the data. Still print()-based, not the stdlib logging module |
| [utils/reid_diagnostics.py](../src/project_auto/utils/reid_diagnostics.py) | Implemented, diagnostic only | ReidDiagnostics, QueryRecord, CandidateRecord: per-run queries/candidates/outcomes CSVs and raw/masked crops keyed by query_id, with config hash and git commit. Imports nothing from the pipeline; write failures are logged, never raised |
| [scripts/run_camera.py](../scripts/run_camera.py) | Implemented wrapper | Calls the main application |
| [scripts/run_video.py](../scripts/run_video.py) | Empty | No video-file execution path provided here |
| [scripts/benchmark_local.py](../scripts/benchmark_local.py) | Empty | No benchmark implementation |
| [cmaera_no_identifier.py](../cmaera_no_identifier.py) | Standalone diagnostic | Cycles camera indexes 0 through 4; opens camera windows |
| notebooks/ReID_Tuning | Research assets | Reference/query examples, notebooks, embeddings, and result images; not an automated production evaluation |
| tests | Automated checks | Synthetic/unit/persistence coverage; see verification chapter |

Do not infer implemented behavior from a placeholder filename or a historical task entry. Package egg-info directories are generated metadata; `pyproject.toml` is the maintained dependency source.

The region store API and detailed event semantics are documented in [chapter 11](11-regions.md). Geometry, current contents, historical association, and activity remain distinct responsibilities.
