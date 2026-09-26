"""Compose and run Project AUTO; this file is the application coordinator.

Implemented responsibilities:
- Load configuration and construct camera, detector, tracker, store, and event engine.
- Construct the segmenter, ReID matcher, scene processor, and identity coordinator.
- Read frames, run detection/tracking, resolve identity, and draw the display.

The identity coordinator (project_auto.events.coordinator) requests scene processing
after tracker confirmation, retries PENDING identity requests on later visible frames,
routes NEW/EXISTING results through the event layer, and schedules spaced reference
captures toward the configured target count. This file only wires the collaborators
together; it does not decide identity itself.
"""

from pathlib import Path

import cv2
import yaml

from project_auto.capture.camera import Camera, CameraConfig
from project_auto.events.coordinator import IdentityCoordinator
from project_auto.events.event_engine import EventEngine
from project_auto.memory.reid import (
    ASPECT_WEIGHT,
    COLOR_WEIGHT,
    DINO_WEIGHT,
    ReidConfig,
    ReidMatcher,
)
from project_auto.memory.store import DatabaseStore
from project_auto.perception.detector import Detection, YoloDetector
from project_auto.perception.scene_processor import SceneProcessor, SceneProcessorConfig
from project_auto.perception.segmenter import SamSegmenter, SegmenterConfig
from project_auto.perception.tracker import DetectionTracker
from project_auto.utils.drawing import draw_detections, draw_regions
from project_auto.utils.logging import Category, log_action
from project_auto.utils.reid_diagnostics import ReidDiagnostics
from project_auto.region_queries import query_regions


def run_app() -> None:
    """Run capture, perception, lifecycle decisions, persistence, and display."""
    project_root = Path(__file__).resolve().parents[2]
    camera_config_path = project_root / "configs" / "camera.yaml"
    perception_config_path = project_root / "configs" / "perception.yaml"
    table_config_path = project_root / "configs" / "table.yaml"
    segmenter_config_path = project_root / "configs" / "segmenter.yaml"
    reid_config_path = project_root / "configs" / "reid.yaml"
    scene_processor_config_path = project_root / "configs" / "scene_processor.yaml"

    with camera_config_path.open(encoding="utf-8") as config_file:
        camera_settings = yaml.safe_load(config_file)
    with perception_config_path.open(encoding="utf-8") as config_file:
        perception_settings = yaml.safe_load(config_file)
    with table_config_path.open(encoding="utf-8") as config_file:
        table_settings = yaml.safe_load(config_file)
    with scene_processor_config_path.open(encoding="utf-8") as config_file:
        scene_processor_settings = yaml.safe_load(config_file)
    with reid_config_path.open(encoding="utf-8") as config_file:
        reid_settings = yaml.safe_load(config_file)

    database_path = project_root / table_settings["database"]["path"]
    store = DatabaseStore(database_path)

    try:
        store.create_schema()
        camera = Camera(CameraConfig(**camera_settings))
        detector = YoloDetector(perception_config_path)
        tracker = DetectionTracker(
            candidate_confirmation_seconds=float(
                perception_settings["candidate_confirmation_seconds"]
            ),
            max_candidate_missing_frames=int(
                perception_settings["max_candidate_missing_frames"]
            ),
            removal_timeout_seconds=float(
                perception_settings["removal_timeout_seconds"]
            ),
            movement_buffer_scale=float(
                perception_settings["movement_buffer_scale"]
            ),
            movement_stop_tolerance_pixels=float(
                perception_settings["movement_stop_tolerance_pixels"]
            ),
            movement_stopped_confirmation_seconds=float(
                perception_settings["movement_stopped_confirmation_seconds"]
            ),
        )
        event_engine = EventEngine(store)

        reid_config = ReidConfig.from_yaml(reid_config_path)
        segmenter = SamSegmenter(SegmenterConfig.from_yaml(segmenter_config_path))
        matcher = ReidMatcher(reid_config)
        scene_processor = SceneProcessor(
            SceneProcessorConfig.from_yaml(scene_processor_config_path),
            segmenter,
            matcher,
        )
        diagnostics_dir = reid_settings.get("diagnostics_dir")
        diagnostics = (
            ReidDiagnostics(
                root_dir=project_root / diagnostics_dir,
                run_settings={
                    "acceptance_threshold": reid_config.acceptance_threshold,
                    "margin_threshold": reid_config.margin_threshold,
                    "weights": f"{DINO_WEIGHT}/{COLOR_WEIGHT}/{ASPECT_WEIGHT}",
                    "top_k": reid_config.top_k,
                    "prototype_shortlist_size": reid_config.prototype_shortlist_size,
                },
                config_paths=(
                    reid_config_path,
                    scene_processor_config_path,
                    segmenter_config_path,
                ),
                project_root=project_root,
            )
            if diagnostics_dir
            else None
        )
        coordinator = IdentityCoordinator(
            scene_processor=scene_processor,
            event_engine=event_engine,
            store=store,
            reid_model_name=reid_config.model_name,
            max_references_per_item=int(scene_processor_settings["max_references_per_item"]),
            bad_mask_cooldown_seconds=float(scene_processor_settings["bad_mask_cooldown_seconds"]),
            queue_full_retry_seconds=float(scene_processor_settings["queue_full_retry_seconds"]),
            min_box_area=int(scene_processor_settings["min_box_area"]),
            min_detector_confidence=float(scene_processor_settings["min_detector_confidence"]),
            initial_reference_count=int(scene_processor_settings["initial_reference_count"]),
            initial_capture_spacing_frames=int(
                scene_processor_settings["initial_capture_spacing_frames"]
            ),
            move_candidate_delay_frames=int(
                scene_processor_settings["move_candidate_delay_frames"]
            ),
            candidate_retry_frames=int(scene_processor_settings["candidate_retry_frames"]),
            max_movement_reference_attempts=int(
                scene_processor_settings["max_movement_reference_attempts"]
            ),
            min_reference_sharpness=float(scene_processor_settings["min_reference_sharpness"]),
            min_reference_frame_visibility=float(
                scene_processor_settings["min_reference_frame_visibility"]
            ),
            min_mask_score=float(scene_processor_settings["min_mask_score"]),
            min_mask_occupancy=float(scene_processor_settings["min_mask_occupancy"]),
            reference_novelty_threshold=float(
                scene_processor_settings["reference_novelty_threshold"]
            ),
            job_queue_max_size=int(scene_processor_settings["job_queue_max_size"]),
            diagnostics=diagnostics,
        )

        with camera:
            log_action(Category.SYSTEM, event="camera_opened", device=camera.device)
            highlighted_regions = []
            highlight_frames_left = 0
            highlight_duration = int(table_settings.get("regions", {}).get("highlight_frames", 150))
            while True:
                frame = camera.read()
                detections = detector.detect(frame)
                signals = tracker.update(detections)
                detections_by_track_id: dict[int, Detection] = {
                    detection.track_id: detection
                    for detection in detections
                    if detection.track_id is not None
                }

                coordinator.handle_frame(frame, signals, detections_by_track_id)

                debug_frame = draw_detections(frame, detections)
                if highlight_frames_left > 0:
                    debug_frame = draw_regions(debug_frame, highlighted_regions)
                    highlight_frames_left -= 1
                cv2.imshow("Project AUTO - q quit, w where item, r inspect region", debug_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key in (ord("w"), ord("r")):
                    region_ids = query_regions(store, key)
                    highlighted_regions = [
                        region for region in store.list_regions() if region.id in region_ids
                    ]
                    highlight_frames_left = highlight_duration
    finally:
        cv2.destroyAllWindows()
        if "coordinator" in locals():
            coordinator.shutdown()
        store.engine.dispose()
