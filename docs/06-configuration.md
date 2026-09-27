# Configuration reference

Values below are the working-tree YAML settings inspected for this documentation. They are configured defaults, not calibrated requirements or measured output. Restart the application after changes; no live reload is implemented.

## Camera and database

| File / key | Current value | Interpretation |
| --- | --- | --- |
| camera.yaml: usb_device | 0 | First camera attempted |
| camera.yaml: webcam_device | 1 | Fallback camera |
| camera.yaml: width / height | 1280 / 720 | Requested capture resolution; device negotiation can differ |
| camera.yaml: fps | 30 | Requested capture rate, not application throughput |
| table.yaml: database.path | data/project_auto.db | SQLite path resolved from project root |
| table.yaml: regions.highlight_frames | 150 | Frames to show query-selected polygons; in-memory state only |

## Detector and lifecycle: perception.yaml

| Key | Current value | Interpretation |
| --- | --- | --- |
| source_model | models/yolo11s.pt | YOLO source weights |
| openvino_model | models/yolo11s_openvino_model | Export directory reused when present |
| confidence | 0.18 | Detector confidence filter |
| iou | 0.45 | Inference overlap threshold passed to Ultralytics |
| image_size | 960 | Inference image-size argument; not camera resolution |
| device | cpu | Inference device argument |
| candidate_confirmation_seconds | 2.0 | Confirmation duration before ADD signal |
| max_candidate_missing_frames | 15 | Cumulative misses tolerated per candidate attempt |
| removal_timeout_seconds | 2.0 | Confirmed-track absence before REMOVE |
| movement_buffer_scale | 1.2 | Centered stable placement buffer scale |
| movement_stop_tolerance_pixels | 5.0 | Allowed center displacement during stop confirmation |
| movement_stopped_confirmation_seconds | 1.0 | Visible stopped duration before MOVED (and the runtime-only MOVE_END, fired the same frame) |

BoT-SORT's `botsort.yaml` name and `persist=True` are selected in detector code rather than these project YAML files. agnostic_nms is not passed explicitly; the installed Ultralytics default inspected in .venv/Lib/site-packages/ultralytics/cfg/default.yaml is False (class-aware NMS). This is an installed-library default, not a pinned project setting.

## Identity models

| File / key | Current value | Interpretation |
| --- | --- | --- |
| segmenter.yaml: model_name | facebook/sam2-hiera-base-plus | SAM2 checkpoint |
| segmenter.yaml: device | cpu | PyTorch segmentation device |
| reid.yaml: model_name | facebook/dinov2-base | Embedding checkpoint |
| reid.yaml: device | cpu | PyTorch embedding device |
| reid.yaml: acceptance_threshold | 0.55 | Final weighted score threshold, or DINO-only fallback; provisional, not a probability |
| reid.yaml: margin_threshold | 0.15 | Minimum best-minus-runner-up final score; provisional |
| reid_diagnostics.yaml: output_dir | logs/reid_runs | Separate diagnostic launcher only; resolved from project root |
| reid_diagnostics.yaml: prototype_shortlist_size | 0 | Override: score every gallery item in diagnostic mode |
| reid_diagnostics.yaml: database_path | data/diagnostics/project_auto_diagnostics.db | Separate diagnostic memory; normal tracking still uses table.yaml |
| reid.yaml: top_k | 3 | Number of strongest references averaged, capped by available references |
| reid.yaml: prototype_shortlist_size | 3 | Ranked prototype candidates; missing prototypes are retained additionally |

Small galleries bypass prototype ranking; a nonpositive shortlist setting also bypasses it. Keep top_k positive. Library configuration parsing does not provide comprehensive validation of every setting.

## Preparation and scheduling: scene_processor.yaml

| Key | Current value | Interpretation |
| --- | --- | --- |
| background_color | [0, 0, 0] | RGB background replacement |
| min_mask_pixels | 100 | Minimum retained mask pixels inside the crop |
| job_queue_max_size | 8 | Waiting-job capacity, excluding the executing job; keep positive |
| bad_mask_cooldown_seconds | 5.0 | Resolve retry delay after bad views, errors, or claimed-item conflicts |
| queue_full_retry_seconds | 0.5 | Resolve retry delay after full queue |
| min_box_area | 400 | Resolve prefilter minimum box area in square pixels |
| min_detector_confidence | 0.0 | Additional resolve filter; detector still uses 0.18 |

### Reference capture (event-driven; `events/reference_policy.py`)

Replaces the earlier continuous `reference_target_count`/`capture_interval_seconds`/`edge_margin_pixels` interval scheme, which kept re-capturing a static item and filled the gallery with near-identical views.

| Key | Current value | Interpretation |
| --- | --- | --- |
| initial_reference_count | 2 | Baseline references per new item; the resolve embedding is reference 1 at no extra cost |
| initial_capture_spacing_frames | 15 | Frames to wait before the remaining baseline capture(s) |
| move_candidate_delay_frames | 5 | Frames to wait after MOVE_START before the first movement candidate |
| candidate_retry_frames | 10 | Spacing between candidate nominations, including after a rejected one |
| max_movement_reference_attempts | 3 | Attempt budget per movement event; consumed at nomination, not on result |
| max_references_per_item | 8 | Hard ceiling; new references are refused once reached, no replacement policy yet |
| min_reference_sharpness | 30.0 | Stage A cheap gate: Laplacian-variance threshold; uncalibrated |
| min_reference_frame_visibility | 0.80 | Stage A cheap gate: visible fraction of the *predicted* (unclipped) box; replaces the old edge-touch rejection |
| min_mask_score | 0.60 | Stage B expensive gate (worker, after SAM): uncalibrated |
| min_mask_occupancy | 0.10 | Stage B expensive gate: masked pixels / segmented box area; a mask-sanity check, not occlusion detection |
| reference_novelty_threshold | 0.93 | Stage B: candidate similarity to an existing reference at or above this is rejected as redundant; baseline candidates are exempt |

These filters and cooldowns are not universal capture guarantees. Stage A runs on the main thread for nominated candidates only (never every frame); Stage B runs in the worker after SAM/DINO, and a rejection there never updates the item's stored prototype. None of these five thresholds have been calibrated against real data yet — see [performance caveats](07-performance.md) and TASKS.md.

Sources: [camera settings](../configs/camera.yaml), [perception settings](../configs/perception.yaml), [database settings](../configs/table.yaml), [SAM2 settings](../configs/segmenter.yaml), [ReID settings](../configs/reid.yaml), [scene settings](../configs/scene_processor.yaml).

ReID weights (0.65 DINO, 0.20 color, 0.15 aspect) are constants in memory/reid.py, not YAML options. The older 0.4 threshold applied to DINO-only scoring and is historical; the current 0.55/0.15 settings need empirical calibration for the weighted score and descriptor-missing fallback. The 960 detector input requests more inference pixels than 640; no measured throughput effect is claimed.

## Normal versus diagnostic configuration

Normal tracking uses the six original YAML files and shortlist 3. Only reid_diagnostics_app.py reads the seventh file, reid_diagnostics.yaml, and supplies its overrides to app.run_app. Its diagnostic database starts without the normal database’s regions; calibration does not copy them automatically.
