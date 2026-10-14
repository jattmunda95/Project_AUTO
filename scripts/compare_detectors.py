"""Compare candidate detectors on recorded clips, offline.

Every detector's boxes go through the same Ultralytics BoT-SORT, configured from
``configs/botsort.yaml``, so differences in ID churn come from the detector rather than from
a different tracker. Detector settings (confidence, IoU, image size) come from
``configs/perception.yaml``. Nothing here touches the application, the database or ReID.

Output goes to ``data/comparisons/<timestamp>/``:
  videos/<clip>/<detector>.mp4   annotated clips (boxes coloured by track ID, HUD, raw boxes grey)
  frames/<clip>__<detector>.csv  per-frame detections, tracked boxes, unique IDs, timings
  results.json                   every metric plus a per-track table
  report.html                    tables and charts

Models that are not on disk are skipped unless ``--download`` is given.
Run ``python scripts/compare_detectors.py --list`` to see the detectors.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
import yaml
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
RECORDINGS_DIR = ROOT / "data" / "recordings"
COMPARISONS_DIR = ROOT / "data" / "comparisons"
TIMING_WARMUP_FRAMES = 3
CHART_POINTS = 600


# --------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class InferenceSettings:
    confidence: float = 0.18
    iou: float = 0.45
    image_size: int = 960
    device: str = "cpu"
    tracker_config: Path = ROOT / "configs" / "botsort.yaml"


@dataclass(frozen=True, slots=True)
class MetricSettings:
    """Values mirror the app's stillness gate so "addable" means what it means there."""

    still_tolerance_px: float = 12.0
    still_seconds: float = 2.0
    max_missing_frames: int = 15
    short_track_seconds: float = 1.0
    handover_window_seconds: float = 2.0
    handover_min_iou: float = 0.3


def load_settings() -> tuple[InferenceSettings, MetricSettings]:
    with (ROOT / "configs" / "perception.yaml").open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    inference = InferenceSettings(
        confidence=float(config["confidence"]),
        iou=float(config["iou"]),
        image_size=int(config["image_size"]),
        device=str(config["device"]),
        tracker_config=ROOT / config["tracker_config"],
    )
    metric = MetricSettings(
        still_tolerance_px=float(config.get("candidate_stillness_tolerance_pixels", 12.0)),
        still_seconds=float(config.get("candidate_confirmation_seconds", 2.0)),
        max_missing_frames=int(config.get("max_candidate_missing_frames", 15)),
    )
    return inference, metric


# --------------------------------------------------------------------------------------
# Detection records
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RawDetections:
    """Detector output for one frame, before tracking."""

    xyxy: NDArray
    conf: NDArray
    cls: NDArray

    @staticmethod
    def empty() -> RawDetections:
        return RawDetections(
            np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.float32)
        )


@dataclass(frozen=True, slots=True)
class TrackObs:
    track_id: int
    box: tuple[float, float, float, float]
    confidence: float
    label: str


@dataclass(slots=True)
class FrameRecord:
    index: int
    raw_boxes: NDArray
    tracks: list[TrackObs]
    detect_ms: float
    track_ms: float


class Detector(Protocol):
    name: str
    backend: str

    def detect(self, frame: NDArray) -> RawDetections: ...

    def label(self, class_id: int) -> str: ...


# --------------------------------------------------------------------------------------
# Metrics (pure, no models or video needed)
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class TrackHistory:
    track_id: int
    frames: list[int] = field(default_factory=list)
    boxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)


def _centre(box: Sequence[float]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def _area(box: Sequence[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def collect_tracks(frames: Sequence[FrameRecord]) -> dict[int, TrackHistory]:
    tracks: dict[int, TrackHistory] = {}
    for record in frames:
        for obs in record.tracks:
            history = tracks.setdefault(obs.track_id, TrackHistory(obs.track_id))
            history.frames.append(record.index)
            history.boxes.append(obs.box)
            history.labels.append(obs.label)
    return tracks


def longest_still_seconds(history: TrackHistory, fps: float, settings: MetricSettings) -> float:
    """Longest stretch the track's box centre stayed within the app's stillness tolerance.

    Same rule as the app's ADD gate: the clock restarts when the centre leaves its anchor by
    more than the tolerance, or when the track vanished for more than the allowed missing
    frames.
    """
    anchor: tuple[float, float] | None = None
    start = 0
    previous: int | None = None
    best = 0.0
    for frame, box in zip(history.frames, history.boxes, strict=True):
        centre = _centre(box)
        if previous is not None and frame - previous > settings.max_missing_frames:
            anchor = None
        if anchor is None or np.hypot(centre[0] - anchor[0], centre[1] - anchor[1]) > (
            settings.still_tolerance_px
        ):
            anchor = centre
            start = frame
        best = max(best, (frame - start) / fps)
        previous = frame
    return best


def count_handovers(
    tracks: dict[int, TrackHistory], fps: float, settings: MetricSettings
) -> int:
    """Count new IDs that appear where another ID just vanished (ID churn on one object).

    A handover is a track whose first box overlaps (IoU) the last box of a track that ended
    1..window frames earlier. Each ended track is matched at most once. Overlapping tracks
    are two objects, not a handover.
    """
    window = round(settings.handover_window_seconds * fps)
    births = sorted(tracks.values(), key=lambda t: t.frames[0])
    used: set[int] = set()
    count = 0
    for new in births:
        best_id: int | None = None
        best_iou = settings.handover_min_iou
        for old in tracks.values():
            if old.track_id == new.track_id or old.track_id in used:
                continue
            gap = new.frames[0] - old.frames[-1]
            if not 1 <= gap <= window:
                continue
            iou = box_iou(old.boxes[-1], new.boxes[0])
            if iou >= best_iou:
                best_iou, best_id = iou, old.track_id
        if best_id is not None:
            used.add(best_id)
            count += 1
    return count


def _stats(values: NDArray) -> dict[str, float]:
    if len(values) == 0:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def compute_metrics(
    frames: Sequence[FrameRecord], fps: float, settings: MetricSettings
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Return (metrics, per-track table) for one detector on one clip."""
    count = len(frames)
    det_counts = np.array([len(f.raw_boxes) for f in frames], dtype=float)
    track_counts = np.array([len(f.tracks) for f in frames], dtype=float)
    timed = frames[TIMING_WARMUP_FRAMES:] if count > TIMING_WARMUP_FRAMES else frames
    detect_ms = np.array([f.detect_ms for f in timed], dtype=float)
    track_ms = np.array([f.track_ms for f in timed], dtype=float)
    infer_seconds = float(np.sum(detect_ms + track_ms)) / 1000.0

    tracks = collect_tracks(frames)
    short_frames = settings.short_track_seconds * fps
    table: list[dict[str, Any]] = []
    missing_total = span_total = gap_events = 0
    short_tracks = addable = never_still = flicker = 0
    jitters: list[float] = []
    area_cvs: list[float] = []
    for history in tracks.values():
        span = history.frames[-1] - history.frames[0] + 1
        missing = span - len(history.frames)
        gaps = sum(1 for a, b in pairwise(history.frames) if b - a > 1)
        missing_total += missing
        span_total += span
        gap_events += gaps
        still = longest_still_seconds(history, fps, settings)
        is_short = span < short_frames
        is_addable = still >= settings.still_seconds
        short_tracks += is_short
        addable += is_addable
        never_still += (not is_short) and (not is_addable)
        distinct = sorted(set(history.labels))
        flicker += len(distinct) > 1
        if not is_short:
            centres = [_centre(b) for b in history.boxes]
            steps = [
                float(np.hypot(c2[0] - c1[0], c2[1] - c1[1]))
                for (f1, c1), (f2, c2) in zip(
                    zip(history.frames, centres), zip(history.frames[1:], centres[1:])
                )
                if f2 - f1 == 1
            ]
            if steps:
                jitters.append(float(np.median(steps)))
            areas = np.array([_area(b) for b in history.boxes])
            if areas.mean() > 0:
                area_cvs.append(float(areas.std() / areas.mean()))
        table.append(
            {
                "track_id": history.track_id,
                "first_s": round(history.frames[0] / fps, 2),
                "last_s": round(history.frames[-1] / fps, 2),
                "frames": len(history.frames),
                "missing_frames": missing,
                "longest_still_s": round(still, 2),
                "labels": distinct,
            }
        )
    table.sort(key=lambda row: row["first_s"])

    detect = _stats(detect_ms)
    metrics = {
        "frames": count,
        "duration_s": count / fps if fps else 0.0,
        "det_per_frame_mean": float(det_counts.mean()) if count else 0.0,
        "det_per_frame_std": float(det_counts.std()) if count else 0.0,
        "det_per_frame_max": float(det_counts.max()) if count else 0.0,
        "zero_det_frames_pct": float((det_counts == 0).mean() * 100) if count else 0.0,
        "tracked_per_frame_mean": float(track_counts.mean()) if count else 0.0,
        "unique_ids": len(tracks),
        "id_handovers": count_handovers(tracks, fps, settings),
        "short_tracks": int(short_tracks),
        "dropout_pct": 100.0 * missing_total / span_total if span_total else 0.0,
        "gap_events": gap_events,
        "addable_tracks": int(addable),
        "never_still_tracks": int(never_still),
        "label_flicker_tracks": int(flicker),
        "jitter_px": float(np.median(jitters)) if jitters else 0.0,
        "area_cv": float(np.median(area_cvs)) if area_cvs else 0.0,
        "detect_ms_mean": detect["mean"],
        "detect_ms_p50": detect["p50"],
        "detect_ms_p95": detect["p95"],
        "detect_ms_max": detect["max"],
        "track_ms_mean": float(track_ms.mean()) if len(track_ms) else 0.0,
        "infer_seconds": infer_seconds,
        "timed_frames": len(timed),
        "effective_fps": len(timed) / infer_seconds if infer_seconds > 0 else 0.0,
    }
    return {key: round(float(value), 3) for key, value in metrics.items()}, table


SUM_KEYS = (
    "frames",
    "duration_s",
    "unique_ids",
    "id_handovers",
    "short_tracks",
    "gap_events",
    "addable_tracks",
    "never_still_tracks",
    "label_flicker_tracks",
)
FRAME_WEIGHTED_KEYS = (
    "det_per_frame_mean",
    "zero_det_frames_pct",
    "tracked_per_frame_mean",
    "dropout_pct",
    "jitter_px",
    "area_cv",
    "detect_ms_mean",
    "detect_ms_p50",
    "detect_ms_p95",
    "track_ms_mean",
)


def aggregate_metrics(runs: Sequence[dict[str, float]]) -> dict[str, float]:
    """Combine one detector's per-clip metrics: counts add, rates are frame-weighted."""
    if not runs:
        return {}
    total_frames = sum(r["frames"] for r in runs) or 1.0
    combined: dict[str, float] = {key: sum(r[key] for r in runs) for key in SUM_KEYS}
    for key in FRAME_WEIGHTED_KEYS:
        combined[key] = sum(r[key] * r["frames"] for r in runs) / total_frames
    combined["detect_ms_max"] = max(r["detect_ms_max"] for r in runs)
    seconds = sum(r["infer_seconds"] for r in runs)
    timed = sum(r["timed_frames"] for r in runs)
    combined["effective_fps"] = timed / seconds if seconds > 0 else 0.0
    return {key: round(float(value), 3) for key, value in combined.items()}


# --------------------------------------------------------------------------------------
# Detectors
# --------------------------------------------------------------------------------------


class SkipDetector(Exception):
    """Raised when a detector cannot run here for an expected, explainable reason."""


class UltralyticsDetector:
    """Any Ultralytics model run through ``predict`` with the app's inference settings."""

    def __init__(
        self,
        name: str,
        model: Any,
        settings: InferenceSettings,
        backend: str,
        agnostic_nms: bool = False,
    ) -> None:
        self.name = name
        self.backend = backend
        self._model = model
        self._settings = settings
        self._agnostic = agnostic_nms
        self._names: dict[int, str] = {}

    def detect(self, frame: NDArray) -> RawDetections:
        settings = self._settings
        result = self._model.predict(
            source=frame,
            conf=settings.confidence,
            iou=settings.iou,
            imgsz=settings.image_size,
            device=settings.device,
            agnostic_nms=self._agnostic,
            verbose=False,
        )[0]
        self._names = result.names
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return RawDetections.empty()
        return RawDetections(
            boxes.xyxy.cpu().numpy().astype(np.float32),
            boxes.conf.cpu().numpy().astype(np.float32),
            boxes.cls.cpu().numpy().astype(np.float32),
        )

    def label(self, class_id: int) -> str:
        return str(self._names.get(class_id, class_id))


class BackgroundChangeDetector:
    """Boxes around whatever differs from a fixed reference (median of the first frames).

    A control, not a contender: an object present from the start is "background" and is
    never detected, and lighting changes, shadows and hands all show up as changes. The
    first ``reference_frames`` frames only build the reference and yield no detections.
    """

    backend = "opencv"

    def __init__(
        self,
        name: str = "background",
        reference_frames: int = 15,
        threshold: int = 30,
        min_area_fraction: float = 0.0005,
    ) -> None:
        self.name = name
        self._needed = reference_frames
        self._threshold = threshold
        self._min_area_fraction = min_area_fraction
        self._buffer: list[NDArray] = []
        self._reference: NDArray | None = None

    @staticmethod
    def _prepare(frame: NDArray) -> NDArray:
        return cv2.GaussianBlur(frame, (7, 7), 0)

    def detect(self, frame: NDArray) -> RawDetections:
        prepared = self._prepare(frame)
        if self._reference is None:
            self._buffer.append(prepared)
            if len(self._buffer) >= self._needed:
                self._reference = np.median(np.stack(self._buffer), axis=0).astype(np.uint8)
                self._buffer.clear()
            return RawDetections.empty()
        difference = cv2.absdiff(prepared, self._reference).max(axis=2)
        mask = (difference > self._threshold).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        minimum = self._min_area_fraction * frame.shape[0] * frame.shape[1]
        boxes: list[list[float]] = []
        for contour in contours:
            if cv2.contourArea(contour) < minimum:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append([x, y, x + w, y + h])
        if not boxes:
            return RawDetections.empty()
        xyxy = np.array(boxes, dtype=np.float32)
        return RawDetections(
            xyxy, np.full(len(boxes), 0.9, np.float32), np.zeros(len(boxes), np.float32)
        )

    def label(self, class_id: int) -> str:
        return "change"


@dataclass(frozen=True, slots=True)
class DetectorSpec:
    name: str
    description: str
    kind: str  # yolo | yoloe_pf | yoloe_text | background
    weights: str = ""
    agnostic_nms: bool = False
    prompts: tuple[str, ...] = ()
    openvino_default: bool = False


SPECS: dict[str, DetectorSpec] = {
    spec.name: spec
    for spec in (
        DetectorSpec("yolov8s", "YOLOv8-s (OpenVINO)", "yolo",
                     "yolov8s.pt", openvino_default=True),
        DetectorSpec("yolo11s", "YOLO11s, the app's current settings (baseline)", "yolo",
                     "yolo11s.pt", openvino_default=True),
        DetectorSpec("yolo11s_agnostic", "YOLO11s with class-agnostic NMS", "yolo",
                     "yolo11s.pt", agnostic_nms=True, openvino_default=True),
        DetectorSpec("yoloe26s_pf", "YOLOE-26s prompt-free", "yoloe_pf", "yoloe-26s-seg-pf.pt"),
        DetectorSpec("yoloe26s_text", "YOLOE-26s with the text prompt 'object'", "yoloe_text",
                     "yoloe-26s-seg.pt", prompts=("object",)),
        DetectorSpec("yoloe11s_pf", "YOLOE-11s prompt-free", "yoloe_pf", "yoloe-11s-seg-pf.pt"),
        DetectorSpec("yoloe11s_text", "YOLOE-11s with the text prompt 'object'", "yoloe_text",
                     "yoloe-11s-seg.pt", prompts=("object",)),
        DetectorSpec("background", "Change from a fixed reference frame (control)", "background"),
    )
}
DEFAULT_DETECTORS = (
    "yolo11s",
    "yolo11s_agnostic",
    "yoloe26s_text",
    "yoloe11s_text",
    "background",
)
# Prompt-free YOLOE scores a vocabulary of thousands of names per box, so it is slow. It
# only runs when asked for with --pf, as its own run.
PF_DETECTORS = ("yoloe11s_pf", "yoloe26s_pf")


def select_detectors(requested: Sequence[str] | None, pf: bool) -> tuple[str, ...]:
    """Pick what to run: ``--detectors`` as given, the prompt-free models for ``--pf``
    (plus anything also named in ``--detectors``), otherwise the default set."""
    if pf:
        extra = [name for name in requested or () if name not in PF_DETECTORS]
        return (*PF_DETECTORS, *extra)
    return tuple(requested) if requested else DEFAULT_DETECTORS


def _require_weights(path: Path, allow_download: bool) -> None:
    if not path.exists() and not allow_download:
        raise SkipDetector(f"{path.name} is not in models/; rerun with --download to fetch it")


def build_detector(
    spec: DetectorSpec,
    settings: InferenceSettings,
    allow_download: bool,
    use_openvino: bool,
) -> Detector:
    """Construct one detector, or raise SkipDetector when its files are unavailable."""
    if spec.kind == "background":
        return BackgroundChangeDetector(spec.name)

    from ultralytics import YOLO, YOLOE

    weights = MODELS_DIR / spec.weights
    if spec.kind == "yolo":
        openvino_dir = MODELS_DIR / f"{weights.stem}_openvino_model"
        if (spec.openvino_default or use_openvino) and not openvino_dir.exists():
            _require_weights(weights, allow_download)
            YOLO(str(weights)).export(format="openvino", dynamic=True)
        if openvino_dir.exists() and (spec.openvino_default or use_openvino):
            model = YOLO(str(openvino_dir), task="detect")
            return UltralyticsDetector(spec.name, model, settings, "openvino", spec.agnostic_nms)
        _require_weights(weights, allow_download)
        return UltralyticsDetector(
            spec.name, YOLO(str(weights)), settings, "pytorch", spec.agnostic_nms
        )

    if spec.kind == "yoloe_text" and not allow_download:
        raise SkipDetector(
            "text prompts need a text-encoder model that may be downloaded; "
            "rerun with --download"
        )
    _require_weights(weights, allow_download)
    model = YOLOE(str(weights))
    if spec.kind == "yoloe_text":
        names = list(spec.prompts)
        model.set_classes(names, model.get_text_pe(names))
    return UltralyticsDetector(spec.name, model, settings, "pytorch")


# --------------------------------------------------------------------------------------
# Tracking
# --------------------------------------------------------------------------------------


class BotsortTracker:
    """The app's BoT-SORT, built from the repo's botsort.yaml, fed any detector's boxes."""

    def __init__(self, config_path: Path) -> None:
        from ultralytics.trackers.bot_sort import BOTSORT
        from ultralytics.utils import YAML, IterableSimpleNamespace

        self._tracker = BOTSORT(args=IterableSimpleNamespace(**YAML.load(config_path)))

    def update(
        self, raw: RawDetections, frame: NDArray, label: Callable[[int], str]
    ) -> list[TrackObs]:
        from ultralytics.engine.results import Boxes

        data = np.concatenate([raw.xyxy, raw.conf[:, None], raw.cls[:, None]], axis=1)
        rows = self._tracker.update(Boxes(data.astype(np.float32), frame.shape[:2]), frame)
        return [
            TrackObs(
                track_id=int(row[4]),
                box=(float(row[0]), float(row[1]), float(row[2]), float(row[3])),
                confidence=float(row[5]),
                label=label(int(row[6])),
            )
            for row in rows
        ]


# --------------------------------------------------------------------------------------
# Running a detector over a clip
# --------------------------------------------------------------------------------------


def track_colour(track_id: int) -> tuple[int, int, int]:
    hue = np.uint8([[[(track_id * 47) % 180, 220, 255]]])
    b, g, r = cv2.cvtColor(hue, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def annotate(
    frame: NDArray,
    record: FrameRecord,
    detector_name: str,
    unique_ids: int,
    fps: float,
) -> NDArray:
    output = frame.copy()
    for x1, y1, x2, y2 in record.raw_boxes:
        cv2.rectangle(output, (int(x1), int(y1)), (int(x2), int(y2)), (150, 150, 150), 1)
    for obs in record.tracks:
        x1, y1, x2, y2 = (int(v) for v in obs.box)
        colour = track_colour(obs.track_id)
        cv2.rectangle(output, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(output, f"#{obs.track_id} {obs.label} {obs.confidence:.2f}",
                    (x1, max(72, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA)
    lines = (
        f"{detector_name}   t={record.index / fps:5.1f}s   frame {record.index}",
        (
            f"detections {len(record.raw_boxes)}   tracked {len(record.tracks)}   "
            f"IDs so far {unique_ids}   {record.detect_ms:.0f} ms"
        ),
    )
    cv2.rectangle(output, (0, 0), (output.shape[1], 54), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(output, line, (10, 22 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return output


def run_clip(
    detector: Detector,
    clip: Path,
    settings: InferenceSettings,
    video_path: Path | None,
    max_frames: int | None,
) -> tuple[list[FrameRecord], float]:
    capture = cv2.VideoCapture(str(clip))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {clip}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    tracker = BotsortTracker(settings.tracker_config)
    writer: cv2.VideoWriter | None = None
    records: list[FrameRecord] = []
    seen_ids: set[int] = set()
    try:
        while max_frames is None or len(records) < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            started = time.perf_counter()
            raw = detector.detect(frame)
            detected = time.perf_counter()
            tracks = tracker.update(raw, frame, detector.label)
            finished = time.perf_counter()
            record = FrameRecord(
                index=len(records),
                raw_boxes=raw.xyxy,
                tracks=tracks,
                detect_ms=(detected - started) * 1000.0,
                track_ms=(finished - detected) * 1000.0,
            )
            records.append(record)
            seen_ids.update(t.track_id for t in tracks)
            if video_path is not None:
                if writer is None:
                    video_path.parent.mkdir(parents=True, exist_ok=True)
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                    )
                writer.write(annotate(frame, record, detector.name, len(seen_ids), fps))
            if len(records) % 100 == 0:
                print(f"    {len(records)} frames", flush=True)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    return records, fps


def write_frame_csv(path: Path, records: Sequence[FrameRecord], fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[int] = set()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "time_s", "detections", "tracked", "ids_so_far",
                         "detect_ms", "track_ms"])
        for record in records:
            seen.update(t.track_id for t in record.tracks)
            writer.writerow([record.index, round(record.index / fps, 3), len(record.raw_boxes),
                             len(record.tracks), len(seen), round(record.detect_ms, 2),
                             round(record.track_ms, 2)])


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------

TABLE_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("det_per_frame_mean", "Detections / frame", "{:.2f}"),
    ("zero_det_frames_pct", "Empty frames %", "{:.1f}"),
    ("tracked_per_frame_mean", "Tracked / frame", "{:.2f}"),
    ("unique_ids", "Unique IDs", "{:.0f}"),
    ("id_handovers", "ID handovers", "{:.0f}"),
    ("short_tracks", "Short tracks (<1 s)", "{:.0f}"),
    ("dropout_pct", "Dropout %", "{:.1f}"),
    ("addable_tracks", "Still >= 2 s tracks", "{:.0f}"),
    ("never_still_tracks", "Never-still tracks", "{:.0f}"),
    ("label_flicker_tracks", "Label-flicker tracks", "{:.0f}"),
    ("jitter_px", "Jitter px/frame", "{:.2f}"),
    ("detect_ms_p50", "Detect ms p50", "{:.0f}"),
    ("detect_ms_p95", "Detect ms p95", "{:.0f}"),
    ("effective_fps", "Effective fps", "{:.1f}"),
)

GLOSSARY = (
    ("Detections / frame", "Boxes the detector returned before tracking, averaged over frames."),
    ("Empty frames %", "Frames where the detector returned nothing."),
    ("Tracked / frame", "Boxes the tracker reported, which is what the app would see."),
    ("Unique IDs",
     (
        "Distinct track IDs. For a clip with N real items, anything well above N is "
        "churn, hands and false boxes."
     )),
    ("ID handovers",
     (
        "A new ID appearing where another ID vanished within 2 s: the same "
        "object getting a new ID. Each one costs a ReID job in the app."
     )),
    ("Short tracks", "IDs that lived under 1 s: flicker and one-off boxes."),
    ("Dropout %", "Frames missing inside tracks' lifespans: how steadily an item is held."),
    ("Still >= 2 s tracks",
     (
        "Tracks with a 2 s stretch inside the app's 12 px stillness "
        "tolerance, so the ones that could pass the ADD gate. A clip with "
        "N still items should show about N."
     )),
    ("Never-still tracks",
     (
        "IDs lasting 1 s or more that never held still for 2 s. A proxy "
        "for hands and carried objects, not a hand detector."
     )),
    ("Label-flicker tracks",
     (
        "Tracks whose class label changed. Informational only: the app "
        "never names or gates identity by class."
     )),
    ("Jitter px/frame",
     (
        "Median frame-to-frame box-centre step on tracks of 1 s or more, "
        "lower is steadier. Includes real movement."
     )),
    ("Effective fps",
     (
        "Frames per second of detector plus tracker only (no drawing or "
        "video). PyTorch rows are slower than OpenVINO rows by backend, not "
        "necessarily by model."
     )),
)

PALETTE = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#be185d",
           "#4b5563")


def downsample(values: Sequence[float], points: int, how: str) -> list[float]:
    if len(values) <= points:
        return list(values)
    edges = np.linspace(0, len(values), points + 1).astype(int)
    out: list[float] = []
    for start, end in pairwise(edges):
        chunk = values[start:max(end, start + 1)]
        out.append(float(chunk[-1]) if how == "last" else float(np.mean(chunk)))
    return out


def svg_chart(series: dict[str, list[float]], title: str, width: int = 760,
              height: int = 200) -> str:
    left, top, right, bottom = 44, 24, 12, 22
    longest = max((len(v) for v in series.values()), default=0)
    top_value = max((max(v) for v in series.values() if v), default=0.0) or 1.0
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
             f'<text x="{left}" y="14" class="ct">{html.escape(title)}</text>']
    for fraction in (0, 0.5, 1):
        y = top + (height - top - bottom) * (1 - fraction)
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}" '
                     'class="grid"/>')
        parts.append(f'<text x="{left - 4}" y="{y + 4:.1f}" text-anchor="end" class="ax">'
                     f'{top_value * fraction:.3g}</text>')
    for colour, (name, values) in zip(PALETTE * 2, series.items()):
        if len(values) < 2 or longest < 2:
            continue
        points = " ".join(
            f"{left + (width - left - right) * i / (longest - 1):.1f},"
            f"{top + (height - top - bottom) * (1 - v / top_value):.1f}"
            for i, v in enumerate(values)
        )
        parts.append(f'<polyline points="{points}" fill="none" stroke="{colour}" '
                     f'stroke-width="1.6"><title>{html.escape(name)}</title></polyline>')
    parts.append("</svg>")
    return "".join(parts)


def _metric_table(rows: dict[str, dict[str, float]], backends: dict[str, str]) -> str:
    head = "".join(f"<th>{html.escape(label)}</th>" for _, label, _ in TABLE_COLUMNS)
    body = []
    for name, metrics in rows.items():
        cells = "".join(
            f"<td>{fmt.format(metrics[key])}</td>" if key in metrics else "<td>-</td>"
            for key, _, fmt in TABLE_COLUMNS
        )
        body.append(f"<tr><th>{html.escape(name)}<small>{html.escape(backends.get(name, ''))}"
                    f"</small></th>{cells}</tr>")
    return f"<div class='scroll'><table><tr><th>Detector</th>{head}</tr>{''.join(body)}</table></div>"


def build_report(results: dict[str, Any], series: dict[str, dict[str, dict[str, list[float]]]]) -> str:
    """Render the HTML report. ``series[clip][detector]`` holds the chart series."""
    backends = {r["detector"]: r["backend"] for r in results["runs"]}
    by_clip: dict[str, dict[str, dict[str, float]]] = {}
    by_detector: dict[str, list[dict[str, float]]] = {}
    for run in results["runs"]:
        by_clip.setdefault(run["clip"], {})[run["detector"]] = run["metrics"]
        by_detector.setdefault(run["detector"], []).append(run["metrics"])
    overall = {name: aggregate_metrics(runs) for name, runs in by_detector.items()}

    sections = [f"<h2>All clips combined</h2>{_metric_table(overall, backends)}"]
    for clip, rows in by_clip.items():
        legend = "".join(
            f"<span><i style='background:{PALETTE[i % len(PALETTE)]}'></i>{html.escape(n)}</span>"
            for i, n in enumerate(rows)
        )
        charts = ""
        if clip in series:
            charts = (
                svg_chart({n: series[clip][n]["detections"] for n in rows if n in series[clip]},
                          "Detections per frame (raw detector boxes)")
                + svg_chart({n: series[clip][n]["ids"] for n in rows if n in series[clip]},
                            "Unique IDs so far (steep = churn)")
            )
        videos = "".join(
            f"<li><a href='{html.escape(run['video'])}'>{html.escape(run['detector'])}</a></li>"
            for run in results["runs"] if run["clip"] == clip and run.get("video")
        )
        sections.append(
            f"<h2>{html.escape(clip)}</h2>{_metric_table(rows, backends)}"
            f"<div class='legend'>{legend}</div>{charts}<ul class='videos'>{videos}</ul>"
        )
    skipped = "".join(
        f"<li><b>{html.escape(s['detector'])}</b>: {html.escape(s['reason'])}</li>"
        for s in results["skipped"]
    )
    glossary = "".join(f"<dt>{html.escape(k)}</dt><dd>{html.escape(v)}</dd>" for k, v in GLOSSARY)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Detector comparison</title><style>
body{{font:14px/1.45 system-ui,sans-serif;margin:24px;max-width:1200px;color:#1f2937}}
h1,h2{{margin:.8em 0 .4em}}table{{border-collapse:collapse;font-size:12.5px}}
th,td{{border:1px solid #d1d5db;padding:4px 8px;text-align:right;white-space:nowrap}}
th:first-child,tr>th{{text-align:left;background:#f3f4f6}}th small{{display:block;color:#6b7280;font-weight:400}}
.scroll{{overflow-x:auto}}svg{{width:100%;max-width:760px;display:block;margin:6px 0}}
.grid{{stroke:#e5e7eb}}.ax{{font-size:10px;fill:#6b7280}}.ct{{font-size:12px;fill:#374151}}
.legend span{{margin-right:14px}}.legend i{{display:inline-block;width:10px;height:10px;margin-right:4px}}
dt{{font-weight:600;margin-top:.5em}}dd{{margin:0 0 0 1em;color:#4b5563}}
@media(prefers-color-scheme:dark){{body{{background:#111827;color:#e5e7eb}}
th,td{{border-color:#374151}}th:first-child,tr>th{{background:#1f2937}}
.grid{{stroke:#374151}}.ct{{fill:#d1d5db}}dd{{color:#9ca3af}}a{{color:#93c5fd}}}}
</style></head><body><h1>Detector comparison</h1>
<p>{results['settings']['clips']} clip(s), tracker {html.escape(results['settings']['tracker'])},
confidence {results['settings']['confidence']}, image size {results['settings']['image_size']}.
There is no ground truth: judge counts against what you know is on the table, and check the
videos.</p>
{'<h2>Skipped</h2><ul>' + skipped + '</ul>' if skipped else ''}
{''.join(sections)}<h2>How to read the columns</h2><dl>{glossary}</dl></body></html>"""


# --------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------


def settings_snapshot(
    inference: InferenceSettings, metric: MetricSettings, max_frames: int | None
) -> dict[str, Any]:
    """Everything that changes a result. A resumed run must match it to be comparable."""
    return {
        "tracker": inference.tracker_config.name,
        "confidence": inference.confidence,
        "iou": inference.iou,
        "image_size": inference.image_size,
        "max_frames": max_frames,
        "still_tolerance_px": metric.still_tolerance_px,
        "still_seconds": metric.still_seconds,
        "max_missing_frames": metric.max_missing_frames,
    }


def series_from_csv(path: Path) -> dict[str, list[float]]:
    """Rebuild a run's chart series from its saved per-frame CSV."""
    detections: list[float] = []
    ids: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            detections.append(float(row["detections"]))
            ids.append(float(row["ids_so_far"]))
    return {
        "detections": downsample(detections, CHART_POINTS, "mean"),
        "ids": downsample(ids, CHART_POINTS, "last"),
    }


def load_previous(
    out_dir: Path, current: dict[str, Any], want_video: bool
) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, list[float]]]]]:
    """Load an earlier run's results for ``--resume``.

    Keeps only runs whose per-frame CSV (and MP4, when videos are wanted) are still on disk,
    so a pair that was interrupted mid-write is redone. Refuses to continue when a stored
    setting differs from the current one, since mixing them would make the report invalid.
    Settings the old run never recorded are not compared.
    """
    path = out_dir / "results.json"
    if not path.exists():
        raise SystemExit(f"Cannot resume: {path} does not exist")
    results = json.loads(path.read_text(encoding="utf-8"))
    stored = results.get("settings", {})
    mismatched = [
        f"{key}: was {stored[key]!r}, now {value!r}"
        for key, value in current.items()
        if key in stored and stored[key] != value
    ]
    if mismatched:
        raise SystemExit("Cannot resume, settings changed since that run:\n  "
                         + "\n  ".join(mismatched))
    kept: list[dict[str, Any]] = []
    series: dict[str, dict[str, dict[str, list[float]]]] = {}
    for run in results.get("runs", []):
        csv_path = out_dir / run["csv"]
        video_ok = not want_video or bool(run.get("video") and (out_dir / run["video"]).exists())
        if csv_path.exists() and video_ok:
            kept.append(run)
            series.setdefault(run["clip"], {})[run["detector"]] = series_from_csv(csv_path)
    results["runs"] = kept
    results.setdefault("skipped", [])
    return results, series


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--detectors", nargs="+", choices=sorted(SPECS), default=None,
                        help=f"default: {' '.join(DEFAULT_DETECTORS)}")
    parser.add_argument("--pf", action="store_true",
                        help=f"run the prompt-free models ({' '.join(PF_DETECTORS)}) instead of "
                             "the default set; names given with --detectors are added")
    parser.add_argument("--clips", nargs="+", type=Path, default=None,
                        help="clip files (default: every mp4 in data/recordings)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="stop each clip after this many frames (quick trial runs)")
    parser.add_argument("--download", action="store_true",
                        help="allow downloading missing model weights")
    parser.add_argument("--openvino", action="store_true",
                        help="export and use OpenVINO for YOLO models that are not yet exported")
    parser.add_argument("--no-video", action="store_true", help="skip annotated MP4 output")
    parser.add_argument("--out", type=Path, default=None, help="output folder")
    parser.add_argument("--resume", type=Path, default=None, metavar="FOLDER",
                        help="continue an earlier run in FOLDER: finished detector/clip pairs "
                             "are kept, the rest are run; settings must match that run")
    parser.add_argument("--list", action="store_true", help="list detectors and exit")
    args = parser.parse_args(argv)
    if args.resume and args.out:
        parser.error("--resume and --out are alternatives: resume writes into its own folder")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.list:
        for spec in SPECS.values():
            mark = "*" if spec.name in DEFAULT_DETECTORS else "p" if spec.name in PF_DETECTORS else " "
            print(f"{mark} {spec.name:18} {spec.description}")
        print("(* = in the default run, p = runs with --pf)")
        return

    clips = args.clips or sorted(RECORDINGS_DIR.glob("*.mp4"))
    if not clips:
        raise SystemExit(f"No clips found in {RECORDINGS_DIR}")
    inference, metric_settings = load_settings()
    snapshot = settings_snapshot(inference, metric_settings, args.max_frames)
    series: dict[str, dict[str, dict[str, list[float]]]] = {}
    if args.resume:
        out_dir = args.resume
        results, series = load_previous(out_dir, snapshot, want_video=not args.no_video)
        results["settings"].update(snapshot)
        print(f"Resuming {out_dir.name}: {len(results['runs'])} finished run(s) kept")
    else:
        out_dir = args.out or COMPARISONS_DIR / f"{datetime.now().astimezone():%Y%m%d_%H%M%S}"
        results = {"settings": dict(snapshot), "runs": [], "skipped": []}
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = select_detectors(args.detectors, args.pf)
    # A detector chosen again gets a fresh chance, so drop its stale skip/failure notes.
    results["skipped"] = [s for s in results["skipped"] if s["detector"] not in selected]
    done = {(run["detector"], run["clip"]) for run in results["runs"]}

    def save() -> None:
        results["settings"]["clips"] = len({run["clip"] for run in results["runs"]} | {
            clip.stem for clip in clips})
        (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        (out_dir / "report.html").write_text(build_report(results, series), encoding="utf-8")

    for name in selected:
        spec = SPECS[name]
        pending = [clip for clip in clips if (name, clip.stem) not in done]
        if not pending:
            print(f"[{name}] already complete, nothing to run")
            continue
        print(f"[{name}] loading...", flush=True)
        try:
            detector = build_detector(spec, inference, args.download, args.openvino)
        except SkipDetector as reason:
            print(f"[{name}] skipped: {reason}")
            results["skipped"].append({"detector": name, "reason": str(reason)})
            save()
            continue
        except Exception as error:  # noqa: BLE001 - one bad model must not stop the rest
            print(f"[{name}] failed to load: {error}")
            results["skipped"].append({"detector": name, "reason": f"failed to load: {error}"})
            save()
            continue

        for clip in pending:
            print(f"[{name}] {clip.name}", flush=True)
            if isinstance(detector, BackgroundChangeDetector):
                detector = BackgroundChangeDetector(name)  # fresh reference per clip
            video = None if args.no_video else out_dir / "videos" / clip.stem / f"{name}.mp4"
            try:
                records, fps = run_clip(detector, clip, inference, video, args.max_frames)
            except Exception as error:  # noqa: BLE001 - keep going, record the failure
                print(f"[{name}] {clip.name} failed: {error}")
                results["skipped"].append(
                    {"detector": name, "reason": f"{clip.name} failed: {error}"})
                save()
                continue
            metrics, tracks = compute_metrics(records, fps, metric_settings)
            csv_path = out_dir / "frames" / f"{clip.stem}__{name}.csv"
            write_frame_csv(csv_path, records, fps)
            seen: set[int] = set()
            cumulative: list[float] = []
            for record in records:
                seen.update(t.track_id for t in record.tracks)
                cumulative.append(float(len(seen)))
            series.setdefault(clip.stem, {})[name] = {
                "detections": downsample([float(len(r.raw_boxes)) for r in records],
                                         CHART_POINTS, "mean"),
                "ids": downsample(cumulative, CHART_POINTS, "last"),
            }
            results["runs"].append({
                "detector": name, "clip": clip.stem, "backend": detector.backend,
                "metrics": metrics, "tracks": tracks,
                "video": str(video.relative_to(out_dir)).replace("\\", "/") if video else None,
                "csv": str(csv_path.relative_to(out_dir)).replace("\\", "/"),
            })
            print(f"    {metrics['frames']:.0f} frames, {metrics['det_per_frame_mean']:.2f} "
                  f"det/frame, {metrics['unique_ids']:.0f} IDs, "
                  f"{metrics['id_handovers']:.0f} handovers, "
                  f"{metrics['effective_fps']:.1f} fps", flush=True)
            save()

    save()
    print(f"\nReport: {out_dir / 'report.html'}")


if __name__ == "__main__":
    main()
