from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

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
    detector.tracker_config = Path("configs/botsort.yaml")
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


def test_detect_passes_the_repo_owned_tracker_config_to_ultralytics() -> None:
    detector = YoloDetector.__new__(YoloDetector)
    detector.confidence = 0.35
    detector.iou = 0.45
    detector.image_size = 640
    detector.device = "AUTO"
    detector.tracker_config = Path("configs/botsort.yaml")
    received: dict = {}

    def fake_track(**kwargs):
        received.update(kwargs)
        return []

    detector.model = SimpleNamespace(track=fake_track)

    detector.detect(np.zeros((50, 50, 3), dtype=np.uint8))

    assert received["tracker"] == str(Path("configs/botsort.yaml"))
    assert received["persist"] is True


def test_the_configured_tracker_file_exists_and_changes_only_the_track_buffer() -> None:
    project_root = Path(__file__).resolve().parents[1]
    perception = yaml.safe_load((project_root / "configs" / "perception.yaml").read_text())
    tracker_path = project_root / perception["tracker_config"]
    assert tracker_path.is_file()

    tracker = yaml.safe_load(tracker_path.read_text(encoding="utf-8"))

    assert tracker["tracker_type"] == "botsort"
    assert tracker["track_buffer"] == 45
    # Everything else must still equal Ultralytics' bundled defaults, so a result can be
    # attributed to track_buffer alone.
    assert tracker["match_thresh"] == 0.8
    assert tracker["new_track_thresh"] == 0.25
    assert tracker["gmc_method"] == "sparseOptFlow"
    assert tracker["with_reid"] is False
