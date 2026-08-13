"""Project AUTO live detection entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import yaml

from project_auto.capture.camera import Camera, CameraConfig
from project_auto.perception.detector import YoloDetector
from project_auto.utils.drawing import draw_detections


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def main() -> None:
    camera_settings = _read_yaml(PROJECT_ROOT / "configs" / "camera.yaml")
    detector_settings = _read_yaml(PROJECT_ROOT / "configs" / "perception.yaml")
    camera = Camera(CameraConfig(**camera_settings))
    detector = YoloDetector(
        source_model=PROJECT_ROOT / detector_settings.pop("source_model"),
        openvino_model=PROJECT_ROOT / detector_settings.pop("openvino_model"),
        **detector_settings,
    )

    try:
        with camera:
            while True:
                frame = camera.read()
                detections = detector.detect(frame)
                cv2.imshow("Project AUTO - press q to quit", draw_detections(frame, detections))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
