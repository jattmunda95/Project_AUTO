"""Object detection for frames supplied by the camera."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from numpy.typing import NDArray
from ultralytics import YOLO


@dataclass(frozen=True, slots=True)
class Detection:
    """One structured object prediction."""

    track_id: int | None
    class_id: int
    class_name: str
    confidence: float
    box: tuple[int, int, int, int]


class YoloDetector:
    """Run a YOLO model on frames without owning the camera."""

    def __init__(
        self,
        config_path: Path,
    ) -> None:
        with config_path.open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)

        project_root = config_path.resolve().parent.parent
        source_model = project_root / config["source_model"]
        openvino_model = project_root / config["openvino_model"]

        self.model = self._load_model(source_model, openvino_model)
        self.confidence = float(config["confidence"])
        self.iou = float(config["iou"])
        self.image_size = int(config["image_size"])
        self.device = str(config["device"])

    @staticmethod
    def _load_model(source_model: Path, openvino_model: Path) -> YOLO:
        """Load an existing OpenVINO model, or export it on the first run."""
        if not openvino_model.exists():
            source_model.parent.mkdir(parents=True, exist_ok=True)
            YOLO(str(source_model)).export(format="openvino", dynamic=True)

        return YOLO(str(openvino_model), task="detect")

    def detect(self, frame: NDArray) -> list[Detection]:
        """Run inference on one frame and return structured detections."""
        results = self.model.track(
            source=frame,
            persist=True,
            tracker="botsort.yaml",
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )

        if not results:
            return []

        result = results[0]
        if result.boxes is None:
            return []

        detections: list[Detection] = []
        for box in result.boxes:
            track_id = int(box.id.item()) if box.id is not None else None
            class_id = int(box.cls.item())
            coordinates = tuple(round(value) for value in box.xyxy[0].tolist())
            detections.append(
                Detection(
                    track_id=track_id,
                    class_id=class_id,
                    class_name=str(result.names[class_id]),
                    confidence=float(box.conf.item()),
                    box=coordinates,
                )
            )

        return detections
