"""Debug visualisation helpers."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray
from collections.abc import Sequence
from typing import Protocol

from project_auto.perception.detector import Detection


class DrawableRegion(Protocol):
    id: int
    name: str
    polygon: list[list[int]]


def draw_regions(
    frame: NDArray,
    regions_to_highlight: Sequence[DrawableRegion],
    color_map: dict[int, tuple[int, int, int]] | None = None,
) -> NDArray:
    """Return a copy with translucent fills, outlines and labels, using BGR colors."""
    output = frame.copy()
    for region in regions_to_highlight:
        color = (color_map or {}).get(region.id, (60, 220, 60))
        points = np.array(region.polygon, dtype=np.int32)
        overlay = output.copy()
        cv2.fillPoly(overlay, [points], color)
        cv2.addWeighted(overlay, 0.2, output, 0.8, 0, dst=output)
        cv2.polylines(output, [points], True, color, 2)
        x, y = map(int, points[0])
        cv2.putText(output, region.name, (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return output


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
