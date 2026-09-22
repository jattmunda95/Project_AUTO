# Performance specifications and evidence

## What is specified

The project has timing and resource-control settings, but no published measured throughput, accuracy SLA, maximum gallery size, or minimum hardware specification. The benchmark script is empty. Do not interpret a 30 FPS camera setting as 30 FPS end-to-end processing.

| Property | Configured or derived value | Evidence category |
| --- | --- | --- |
| Capture request | 1280 x 720 at 30 FPS | Configuration; actual negotiation unverified |
| Detector image-size argument | 960 | Configuration |
| Candidate confirmation | 2 seconds | Configured temporal gate, not total identification latency |
| Removal threshold | 2 seconds absent | Evaluated when the loop runs, not a hard deadline |
| Movement stop confirmation | 1 second within 5 pixels | Configured image-space heuristic |
| Background concurrency | One worker | Source implementation |
| Waiting jobs | Up to 8 with current setting | Source/config; executing job is additional |
| Result queue | Unbounded | Source implementation |
| Accepted reference target | 6 per item/model | Guarded database-save cap |
| Display FPS, p95 latency, memory peak, recognition accuracy | Not measured here | Requires benchmark/evaluation |

## Useful estimates, not benchmarks

A 1280 x 720 three-channel uint8 frame contains 2,764,800 bytes, about 2.64 MiB. Eight queued full-frame copies account for about 21.1 MiB of raw frame storage. This excludes the active frame/job, object overhead, result crops, masks, model tensors, and inference workspaces; it is not a process-memory bound.

An identity outcome requires at least the confirmation gate plus queue wait, inference, and result application. Failed-view retries add cooldowns. Six successful references require an initial view and five additional spaced saves; five two-second gaps contribute roughly ten seconds before processing/queue delays. Neither is a guaranteed completion time.

Prototype shortlisting still compares against gallery prototypes and sorts ranked items. It reduces detailed reference comparisons for ranked items, but is not a vector index or constant-time lookup. Entries without prototypes remain eligible beyond the nominal shortlist size.

## Current implementation caveats

- SAM2 and DINOv2 are moved off the frame loop, but main-thread detection and lifecycle database writes can still stall display.
- With a nonempty gallery, SceneProcessor prepares an embedding, then match_candidate embeds the same prepared crop again. Identity resolution can therefore perform two DINOv2 passes.
- The reference target caps saves, not inference. The worker prepares a crop before checking the store cap, and the coordinator does not stop scheduling on a skipped result.
- Bad additional-reference captures and full capture queues do not receive all resolve-path cooldown protections. Repeated capture work remains a possible load source.
- The initial gallery load is outside per-job error handling. A schema failure can leave the identification worker unavailable.
- Full-frame copies and CPU model workloads share memory/compute with capture and detection.
- Diagnostic prints remain in hot paths; centralized logging is unfinished.
- Task notes identify high frame dropout as unresolved pending a post-async hardware check.

These observations are source-based limitations, not measurements or fixes made by this documentation task.

## Proposed benchmark protocol

Use an isolated database and record the code revision, exact package versions, CPU/RAM, power mode, camera/backend, negotiated resolution/rate, model assets, and all YAML settings. Separate cold startup/download/export from a warmed steady-state run.

Measure frame read, detector/tracker, coordinator, and display duration; worker queue wait and processing duration; end-to-end identity latency; delivered FPS; failed reads; CPU usage; peak resident memory; and queue occupancy. Report median and p95 values plus sample counts and run duration, not only averages.

Exercise an empty scene, several stable objects, repeated additions/removals, movement before identity completes, clipped/bad masks, similar-looking objects, and growing galleries. Repeat fixed scenarios under consistent lighting. For identity quality, label same-item versus different-item trials and report false matches, missed matches, and duplicate permanent identities. Tune the threshold on calibration examples and evaluate separately held-out examples.

Acceptance thresholds for throughput and accuracy remain to be agreed for the intended workload. This protocol is proposed; no benchmark execution is claimed.

Sources: [benchmark placeholder](../scripts/benchmark_local.py), [worker](../src/project_auto/events/identification_worker.py), [coordinator](../src/project_auto/events/coordinator.py), [scene processor](../src/project_auto/perception/scene_processor.py), [ReID matcher](../src/project_auto/memory/reid.py).

## Spatial and inference changes

Region resolution scans configured polygons when committing event evidence or making explicit queries. Polygon areas are stored once at creation. Ordinary frames do not write region/current-box state. Translucent drawing runs during calibration or temporary query highlights, and terminal prompts block capture until input completes; this stopgap UI is unsuitable for uninterrupted acquisition guarantees.

YOLO confidence is now 0.18 and imgsz is 960; these are configured changes, not measured quality or performance improvements. ReID now computes supplementary histogram/aspect scores and can append CSV diagnostics. Include these settings, region count/vertex count, and query pauses when benchmarking.
