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
| movement_stopped_confirmation_seconds | 1.0 | Visible stopped duration before MOVED |

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
| reid.yaml: match_log_path | logs/reid_match_log.csv | Optional diagnostic CSV; omit to disable; relative paths follow working directory |
| reid.yaml: top_k | 3 | Number of strongest references averaged, capped by available references |
| reid.yaml: prototype_shortlist_size | 3 | Ranked prototype candidates; missing prototypes are retained additionally |

Small galleries bypass prototype ranking; a nonpositive shortlist setting also bypasses it. Keep top_k positive. Library configuration parsing does not provide comprehensive validation of every setting.

## Preparation and scheduling: scene_processor.yaml

| Key | Current value | Interpretation |
| --- | --- | --- |
| background_color | [0, 0, 0] | RGB background replacement |
| min_mask_pixels | 100 | Minimum retained mask pixels inside the crop |
| reference_target_count | 6 | Maximum accepted references per item/model via guarded worker saves |
| capture_interval_seconds | 2.0 | Nominal spacing after successful capture result application |
| job_queue_max_size | 8 | Waiting-job capacity, excluding the executing job; keep positive |
| bad_mask_cooldown_seconds | 5.0 | Resolve retry delay after bad views, errors, or claimed-item conflicts |
| queue_full_retry_seconds | 0.5 | Resolve retry delay after full queue |
| min_box_area | 400 | Resolve prefilter minimum box area in square pixels |
| min_detector_confidence | 0.0 | Additional resolve filter; detector still uses 0.18 |
| edge_margin_pixels | 2 | Resolve candidates touching this frame-edge margin are rejected |

These filters and cooldowns are not universal capture guarantees. The additional-reference path does not apply the same prefilter or all the same retry cooldowns. It can continue performing preparation after the database reference cap has been reached; the store rejects extra saves. See [performance caveats](07-performance.md).

Sources: [camera settings](../configs/camera.yaml), [perception settings](../configs/perception.yaml), [database settings](../configs/table.yaml), [SAM2 settings](../configs/segmenter.yaml), [ReID settings](../configs/reid.yaml), [scene settings](../configs/scene_processor.yaml).

ReID weights (0.65 DINO, 0.20 color, 0.15 aspect) are constants in memory/reid.py, not YAML options. The older 0.4 threshold applied to DINO-only scoring and is historical; the current 0.55/0.15 settings need empirical calibration for the weighted score and descriptor-missing fallback. The 960 detector input requests more inference pixels than 640; no measured throughput effect is claimed.
