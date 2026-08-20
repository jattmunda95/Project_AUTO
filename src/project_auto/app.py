"""Compose and run the complete Project AUTO application."""

from pathlib import Path

import cv2
import yaml

from project_auto.capture.camera import Camera, CameraConfig
from project_auto.events.event_engine import EventEngine
from project_auto.memory.store import DatabaseStore
from project_auto.perception.detector import YoloDetector
from project_auto.perception.tracker import DetectionTracker
from project_auto.utils.drawing import draw_detections


def run_app() -> None:
    """Run capture, perception, lifecycle decisions, persistence, and display."""
    project_root = Path(__file__).resolve().parents[2]
    camera_config_path = project_root / "configs" / "camera.yaml"
    perception_config_path = project_root / "configs" / "perception.yaml"
    table_config_path = project_root / "configs" / "table.yaml"

    with camera_config_path.open(encoding="utf-8") as config_file:
        camera_settings = yaml.safe_load(config_file)
    with table_config_path.open(encoding="utf-8") as config_file:
        table_settings = yaml.safe_load(config_file)

    database_path = project_root / table_settings["database"]["path"]
    store = DatabaseStore(database_path)

    try:
        store.create_schema()
        camera = Camera(CameraConfig(**camera_settings))
        detector = YoloDetector(perception_config_path)
        tracker = DetectionTracker()
        event_engine = EventEngine(store)

        with camera:
            while True:
                frame = camera.read()
                detections = detector.detect(frame)
                signals = tracker.update(detections)

                for signal in signals:
                    event_engine.process_signal(signal)

                debug_frame = draw_detections(frame, detections)
                cv2.imshow("Project AUTO - press q to quit", debug_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cv2.destroyAllWindows()
        store.engine.dispose()
