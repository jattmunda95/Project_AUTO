"""Run the Project AUTO live detection pipeline."""

from pathlib import Path

import cv2
import yaml

from project_auto.capture.camera import Camera, CameraConfig
from project_auto.perception.detector import YoloDetector
from project_auto.utils.drawing import draw_detections


def main() -> None:
    """Pass camera frames through detection and display the predictions."""
    project_root = Path(__file__).resolve().parents[2]
    camera_config_path = project_root / "configs" / "camera.yaml"
    perception_config_path = project_root / "configs" / "perception.yaml"

    with camera_config_path.open(encoding="utf-8") as config_file:
        camera_settings = yaml.safe_load(config_file)

    camera = Camera(CameraConfig(**camera_settings))
    detector = YoloDetector(perception_config_path)

    try:
        with camera:
            while True:
                frame = camera.read()
                detections = detector.detect(frame)
                debug_frame = draw_detections(frame, detections)
                cv2.imshow("Project AUTO - press q to quit", debug_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
