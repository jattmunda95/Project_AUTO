"""Standalone size and color descriptors used to supplement embedding similarity.

Subfunctions:
- aspect_ratio reads width/height directly from a detection box.
- hsv_histogram builds a lighting-invariant Hue+Saturation distribution over the
  masked (foreground-only) pixels of a crop.
- color_similarity compares two histograms with Bhattacharyya distance.
- size_similarity compares two aspect ratios with a Gaussian falloff on their
  difference.

Pure functions only: no tracker, gallery, database, or model state. Callers own
when descriptors are computed (before background replacement, so the fill color
never enters the histogram) and how the resulting similarities are combined.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from numpy.typing import NDArray

# Fixed bin layout for hsv_histogram's default call. Callers that flatten a
# histogram for storage (e.g. as a JSON column) reshape it back to this shape
# before comparing, since the shape itself is not persisted alongside it.
DEFAULT_H_BINS = 32
DEFAULT_S_BINS = 32


def aspect_ratio(box: tuple[int, int, int, int]) -> float:
    """Return the orientation-agnostic aspect ratio max(w/h, h/w) for a box.

    Objects here are not constrained to one orientation (not fixed flat on a
    table), so a box rotated 90 degrees must score as the same size/shape
    rather than as a mismatch. Always >= 1.0; 0.0 for a degenerate box.
    """
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        return 0.0
    ratio = width / height
    return max(ratio, 1.0 / ratio)


def hsv_histogram(
    crop_rgb: NDArray[np.uint8],
    mask: NDArray[np.bool_],
    h_bins: int = DEFAULT_H_BINS,
    s_bins: int = DEFAULT_S_BINS,
) -> NDArray:
    """Return a normalized Hue+Saturation histogram over masked pixels only.

    Value (brightness) is dropped so the descriptor is less sensitive to shadows
    and exposure changes. The mask must already be cropped to the same region as
    crop_rgb (as SamSegmenter's masks are).
    """
    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    mask_u8 = mask.astype(np.uint8)
    histogram = cv2.calcHist(
        [hsv], [0, 1], mask_u8, [h_bins, s_bins], [0, 180, 0, 256]
    )
    cv2.normalize(histogram, histogram, alpha=1.0, norm_type=cv2.NORM_L1)
    return histogram.astype(np.float32)


def color_similarity(hist_a: NDArray, hist_b: NDArray) -> float:
    """Return 1 - Bhattacharyya distance between two histograms; 1.0 means identical."""
    distance = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_BHATTACHARYYA)
    return 1.0 - float(distance)


def size_similarity(ratio_a: float, ratio_b: float, sigma: float = 0.25) -> float:
    """Return a Gaussian-falloff similarity in (0, 1] for two aspect ratios.

    sigma controls how quickly similarity decays with |ratio_a - ratio_b|; smaller
    sigma penalizes aspect-ratio mismatches more harshly.
    """
    difference = ratio_a - ratio_b
    return math.exp(-(difference * difference) / (2.0 * sigma * sigma))
