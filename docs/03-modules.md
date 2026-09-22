# Module reference

All package paths below are relative to `src/project_auto`. Modules separate capture, perception, lifecycle decisions, orchestration, and persistence. The production entry point is `project-auto`, not the research notebooks.

## Application and capture

| Module | Public entry points | Responsibility and boundary |
| --- | --- | --- |
| [main.py](../src/project_auto/main.py) | main | Console entry point; delegates to run_app |
| [app.py](../src/project_auto/app.py) | run_app | Loads services, capture/display, w/r console queries and temporary region highlights; owns no region SQL |
| [region_calibration.py](../src/project_auto/region_calibration.py) | run_calibration | Separate live-camera polygon editor; reuses Camera and store; terminal naming |
| [region_queries.py](../src/project_auto/region_queries.py) | query_regions | Prints item location or region contents/activity/history; returns highlight IDs |
| [capture/camera.py](../src/project_auto/capture/camera.py) | CameraConfig, Camera.open/read/close | Configures properties before the first read; verifies devices in fallback order; returns BGR frames; supports a context manager |

## Perception

| Module | Public types / operations | Input and output |
| --- | --- | --- |
| [perception/detector.py](../src/project_auto/perception/detector.py) | Detection, YoloDetector.detect | BGR frame to class/confidence/xyxy box/optional temporary ID; loads or exports OpenVINO model; runs persistent BoT-SORT |
| [perception/tracker.py](../src/project_auto/perception/tracker.py) | DetectionTracker.update, TrackSignal, TrackStatus | Detection list to immutable ADD/MOVED/REMOVE signals; ignores detections without IDs; maintains candidate, stable, moving, and missing lifecycle |
| [perception/segmenter.py](../src/project_auto/perception/segmenter.py) | SegmenterConfig, SamSegmenter.segment, Segmentation | uint8 BGR frame plus integer boxes to frame-sized boolean masks and scores; True retains foreground; selects best finite nonempty mask |
| [perception/scene_processor.py](../src/project_auto/perception/scene_processor.py) | SceneProcessor.prepare_reference/process, PreparedReference, IdentityDecision | Prepares masked RGB crop, normalized embedding, aspect ratio and foreground color histogram; proposes new/existing/pending; owns no scheduling or persistence |

Tracker confirmation lasts two seconds with at most 15 cumulative candidate misses per attempt. A 16th miss resets the attempt. Stable placement uses a centered buffer, and movement must stop visibly for the configured interval before MOVED. Absence of a stable or moving track for two seconds emits REMOVE. These are heuristics in image coordinates, not physical motion measurement.

## Orchestration and events

| Module | Main contracts | Responsibility |
| --- | --- | --- |
| [events/coordinator.py](../src/project_auto/events/coordinator.py) | IdentityCoordinator.handle_frame/shutdown | Applies results, submits resolve/capture jobs, schedules retries, routes lifecycle events, guards active claims |
| [events/identification.py](../src/project_auto/events/identification.py) | IdentificationJob, IdentificationResult, IdentificationState, ReferenceManager | Immutable cross-thread message containers and per-track bookkeeping; queue eligibility and cooldowns |
| [events/identification_worker.py](../src/project_auto/events/identification_worker.py) | IdentificationWorkerProtocol, IdentificationWorker.start/stop/submit/poll_results | One worker, bounded jobs, unbounded results, gallery snapshot, model work and reference persistence |
| [events/event_engine.py](../src/project_auto/events/event_engine.py) | EventEngine.process_signal/process_add/process_move/process_remove/process_return | Resolves session track bindings and performs atomic lifecycle store operations; rejects conflicting claims |

Worker exceptions during a job become error results with logged diagnostics. Initial gallery loading occurs before the per-job exception wrapper; startup/schema failures can therefore stop the worker rather than becoming a normal per-track retry.

## Memory and presentation

| Module | Main contracts | Responsibility |
| --- | --- | --- |
| [memory/reid.py](../src/project_auto/memory/reid.py) | ReidConfig, GalleryEntry, ReidMatch, ReidMatcher | Loads DINOv2 once, normalizes embeddings, ranks prototypes and reference similarities; DB-agnostic |
| [memory/models.py](../src/project_auto/memory/models.py) | Region, Item, ItemEmbedding, ItemEvent, ItemStatus, ItemEventType | Typed SQLAlchemy schema, constraints, indexes, relationships |
| [memory/store.py](../src/project_auto/memory/store.py) | DatabaseStore | Short-lived sessions, lifecycle/spatial transactions, region CRUD and queries, vector/descriptor persistence, prototypes and gallery snapshots |
| [memory/state_machine.py](../src/project_auto/memory/state_machine.py) | StateDecision, decide_track_signal, decide_return_signal | Maps supported signals to status and event type; does not infer identity |
| [utils/drawing.py](../src/project_auto/utils/drawing.py) | draw_detections, draw_regions | Returns annotated copies; polygons use translucent fills, outlines, names and optional BGR colors |

## Descriptors and supporting components

[perception/descriptors.py](../src/project_auto/perception/descriptors.py) implements pure supplementary descriptors. Aspect ratio is max(width/height, height/width), so a 90-degree box rotation has the same ratio; it is not absolute size. Foreground RGB pixels produce a normalized 32 x 32 Hue/Saturation histogram before background replacement. Color similarity is 1 minus Bhattacharyya distance; aspect similarity uses Gaussian falloff with sigma 0.25. These descriptors are now wired through preparation, worker saves, gallery loading, and matching. Calibration/held-out accuracy remains unverified.

The matcher first shortlists by DINO prototypes. It averages top-k scores separately for DINO, valid color references, and valid aspect references. With both supplementary scores available, final score = 0.65 * DINO + 0.20 * color + 0.15 * aspect; otherwise it falls back to DINO alone. Acceptance requires the configured score threshold and best-minus-runner-up margin; a single candidate has no margin competitor. Optional CSV output records diagnostic match scores. Margin rejection currently yields no accepted identity, which SceneProcessor interprets as NEW; an ambiguity/confirmation flow is still deferred.

| Path | Status | Usage |
| --- | --- | --- |
| [memory/regions.py](../src/project_auto/memory/regions.py) | Implemented, pure geometry | RegionGeometry, validate_polygon, polygon_area, point_in_polygon, resolve_region, box_area_fraction; no SQL/session imports |
| [scripts/define_regions.py](../scripts/define_regions.py) | Thin wrapper | Delegates to region_calibration.run_calibration |
| [display/viewer.py](../src/project_auto/display/viewer.py) | Empty | Actual display currently lives in app.py |
| [utils/logging.py](../src/project_auto/utils/logging.py) | Empty | No centralized logging setup; runtime mixes print and logging |
| [scripts/run_camera.py](../scripts/run_camera.py) | Implemented wrapper | Calls the main application |
| [scripts/run_video.py](../scripts/run_video.py) | Empty | No video-file execution path provided here |
| [scripts/benchmark_local.py](../scripts/benchmark_local.py) | Empty | No benchmark implementation |
| [cmaera_no_identifier.py](../cmaera_no_identifier.py) | Standalone diagnostic | Cycles camera indexes 0 through 4; opens camera windows |
| notebooks/ReID_Tuning | Research assets | Reference/query examples, notebooks, embeddings, and result images; not an automated production evaluation |
| tests | Automated checks | Synthetic/unit/persistence coverage; see verification chapter |

Do not infer implemented behavior from a placeholder filename or a historical task entry. Package egg-info directories are generated metadata; `pyproject.toml` is the maintained dependency source.

The region store API and detailed event semantics are documented in [chapter 11](11-regions.md). Geometry, current contents, historical association, and activity remain distinct responsibilities.
