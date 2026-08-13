"""Debug visualisation helpers."""

from __future__ import annotations

import cv2
from numpy.typing import NDArray

from project_auto.perception.detector import Detection


def draw_detections(frame: NDArray, detections: list[Detection]) -> NDArray:
    """Return a copy of the frame annotated with boxes and labels."""
    output = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection.box
        cv2.rectangle(output, (x1, y1), (x2, y2), (60, 220, 60), 2)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        cv2.putText(
            output,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (60, 220, 60),
            2,
            cv2.LINE_AA,
        )
    return output
