"""Standalone descriptor tests: frame-visibility geometry and focus measurement."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from project_auto.perception.descriptors import (
    clip_box_to_frame,
    frame_visibility_ratio,
    sharpness,
)

FRAME_SIZE = (640, 480)  # (width, height)


def test_box_fully_inside_frame_is_fully_visible() -> None:
    assert frame_visibility_ratio((100, 100, 200, 200), FRAME_SIZE) == pytest.approx(1.0)


def test_box_touching_an_edge_is_still_fully_visible() -> None:
    """The old rule rejected these outright; only the visible fraction matters now."""
    assert frame_visibility_ratio((0, 0, 100, 100), FRAME_SIZE) == pytest.approx(1.0)
    assert frame_visibility_ratio((540, 380, 640, 480), FRAME_SIZE) == pytest.approx(1.0)


def test_box_partially_outside_left_edge() -> None:
    # 100 wide, 50 of it left of x=0.
    assert frame_visibility_ratio((-50, 100, 50, 200), FRAME_SIZE) == pytest.approx(0.5)


def test_box_partially_outside_right_edge() -> None:
    # 100 wide starting at x=600; 40 visible before x=640.
    assert frame_visibility_ratio((600, 100, 700, 200), FRAME_SIZE) == pytest.approx(0.4)


def test_box_partially_outside_top_and_bottom_edges() -> None:
    assert frame_visibility_ratio((100, -25, 200, 75), FRAME_SIZE) == pytest.approx(0.75)
    assert frame_visibility_ratio((100, 460, 200, 560), FRAME_SIZE) == pytest.approx(0.2)


def test_approximately_eighty_percent_visible_box_clears_a_080_threshold() -> None:
    # 100 wide starting at x=-20: 80 of 100 columns are inside the frame.
    ratio = frame_visibility_ratio((-20, 100, 80, 200), FRAME_SIZE)

    assert ratio == pytest.approx(0.8)
    assert ratio >= 0.80


def test_mostly_outside_box_is_rejected_by_a_080_threshold() -> None:
    ratio = frame_visibility_ratio((-60, 100, 40, 200), FRAME_SIZE)

    assert ratio == pytest.approx(0.4)
    assert ratio < 0.80


def test_box_entirely_outside_the_frame_has_zero_visibility() -> None:
    assert frame_visibility_ratio((700, 500, 800, 600), FRAME_SIZE) == 0.0
    assert frame_visibility_ratio((-200, -200, -100, -100), FRAME_SIZE) == 0.0


@pytest.mark.parametrize(
    "box",
    [(100, 100, 100, 200), (100, 100, 200, 100), (200, 200, 100, 100)],
)
def test_zero_area_and_reversed_boxes_are_handled_safely(box) -> None:
    assert frame_visibility_ratio(box, FRAME_SIZE) == 0.0


def test_clip_box_keeps_the_predicted_box_separate_from_the_visible_one() -> None:
    predicted_box = (-20, 100, 80, 200)

    visible_box = clip_box_to_frame(predicted_box, FRAME_SIZE)

    assert visible_box == (0, 100, 80, 200)
    # The caller's predicted box is untouched, so visibility stays measurable.
    assert predicted_box == (-20, 100, 80, 200)


def test_clip_box_returns_none_when_no_overlap() -> None:
    assert clip_box_to_frame((700, 500, 800, 600), FRAME_SIZE) is None


def _noisy_image(seed: int = 0) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return generator.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)


def test_sharp_image_scores_higher_than_its_blurred_version() -> None:
    sharp = _noisy_image()
    blurred = cv2.GaussianBlur(sharp, (9, 9), 0)

    assert sharpness(sharp) > sharpness(blurred)


def test_heavily_blurred_image_falls_below_a_threshold_a_sharp_one_clears() -> None:
    sharp = _noisy_image()
    blurred = cv2.GaussianBlur(sharp, (21, 21), 0)
    threshold = 30.0

    assert sharpness(sharp) >= threshold
    assert sharpness(blurred) < threshold


def test_flat_image_has_effectively_no_detail() -> None:
    assert sharpness(np.full((32, 32, 3), 128, dtype=np.uint8)) == pytest.approx(0.0)


def test_grayscale_input_is_accepted() -> None:
    gray = cv2.cvtColor(_noisy_image(), cv2.COLOR_BGR2GRAY)

    assert sharpness(gray) > 0.0


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((0, 0, 3), dtype=np.uint8),
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.zeros((1, 40, 3), dtype=np.uint8),
    ],
)
def test_empty_or_tiny_crops_are_handled_safely(image: np.ndarray) -> None:
    assert sharpness(image) == 0.0
