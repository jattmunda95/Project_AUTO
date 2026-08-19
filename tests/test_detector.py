from types import SimpleNamespace

import numpy as np

from project_auto.perception.detector import YoloDetector


class Scalar:
    def __init__(self, value: float) -> None:
        self.value = value

    def item(self) -> float:
        return self.value


class Coordinates:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def __getitem__(self, index: int) -> "Coordinates":
        return self

    def tolist(self) -> list[float]:
        return self.values


def test_detect_returns_structured_detections() -> None:
    detector = YoloDetector.__new__(YoloDetector)
    detector.confidence = 0.35
    detector.iou = 0.45
    detector.image_size = 640
    detector.device = "AUTO"
    box = SimpleNamespace(
        id=Scalar(7),
        cls=Scalar(2),
        conf=Scalar(0.875),
        xyxy=Coordinates([1, 2, 30, 40]),
    )
    result = SimpleNamespace(names={2: "cup"}, boxes=[box])
    detector.model = SimpleNamespace(track=lambda **_: [result])

    detections = detector.detect(np.zeros((50, 50, 3), dtype=np.uint8))

    assert len(detections) == 1
    assert detections[0].track_id == 7
    assert detections[0].class_name == "cup"
    assert detections[0].confidence == 0.875
    assert detections[0].box == (1, 2, 30, 40)
