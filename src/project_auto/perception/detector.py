"""YOLO object detection with an OpenVINO inference model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from numpy.typing import NDArray
from ultralytics import YOLO


@dataclass(frozen=True, slots=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    box: tuple[int, int, int, int]


class YoloDetector:
    """Export YOLO11s once, then run its OpenVINO representation."""

    def __init__(
        self,
        source_model: Path,
        openvino_model: Path,
        confidence: float = 0.35,
        iou: float = 0.45,
        image_size: int = 640,
        device: str = "AUTO",
    ) -> None:
        self.confidence = confidence
        self.iou = iou
        self.image_size = image_size
        self.device = device
        self.model = self._load_model(source_model, openvino_model)

    @staticmethod
    def _load_model(source_model: Path, openvino_model: Path) -> YOLO:
        if not openvino_model.exists():
            source_model.parent.mkdir(parents=True, exist_ok=True)
            exported = YOLO(str(source_model)).export(format="openvino", dynamic=True)
            exported_path = Path(str(exported))
            if exported_path.resolve() != openvino_model.resolve():
                raise RuntimeError(
                    f"Ultralytics exported to {exported_path}, but configuration expects "
                    f"{openvino_model}"
                )
        return YOLO(str(openvino_model), task="detect")

    def detect(self, frame: NDArray) -> list[Detection]:
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []

        result: Any = results[0]
        names = result.names
        detections: list[Detection] = []
        for box in result.boxes:
            class_id = int(box.cls.item())
            coordinates = tuple(round(value) for value in box.xyxy[0].tolist())
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=str(names[class_id]),
                    confidence=float(box.conf.item()),
                    box=coordinates,
                )
            )
        return detections
