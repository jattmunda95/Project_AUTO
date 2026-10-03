"""Standalone scene-processor tests using mocked segmenter/matcher collaborators."""

# TODO(tests): add/verify targeted coverage for prototype calculation and a real (not
# mocked) SAM mask-polarity contract, since a prior mask-polarity inversion bug went
# undetected without one (see TASKS.md's real-hardware debugging entry).

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from project_auto.memory.reid import CandidateScore, GalleryEntry, ReidMatch
from project_auto.perception.scene_processor import (
    IdentityDecision,
    SceneProcessor,
    SceneProcessorConfig,
)
from project_auto.perception.segmenter import Segmentation


@pytest.fixture
def config() -> SceneProcessorConfig:
    return SceneProcessorConfig(background_color=(0, 0, 0), min_mask_pixels=1)


@pytest.fixture
def frame() -> np.ndarray:
    # BGR frame; distinct channel values make RGB conversion errors visible.
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    frame[:, :] = (10, 20, 30)  # B, G, R
    return frame


def full_mask(box: tuple[int, int, int, int], frame_shape: tuple[int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    mask = np.zeros(frame_shape, dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


@pytest.fixture
def segmenter() -> Mock:
    return Mock()


@pytest.fixture
def matcher() -> Mock:
    matcher = Mock()
    matcher.create_embedding.return_value = np.array([1.0, 0.0])
    return matcher


@pytest.fixture
def processor(config: SceneProcessorConfig, segmenter: Mock, matcher: Mock) -> SceneProcessor:
    return SceneProcessor(config, segmenter, matcher)


def test_prepare_reference_returns_none_when_no_usable_mask(
    processor: SceneProcessor, segmenter: Mock, frame: np.ndarray
) -> None:
    segmenter.segment.return_value = [None]

    assert processor.prepare_reference(frame, (0, 0, 4, 4)) is None


def test_prepare_reference_returns_none_below_min_mask_pixels(
    processor: SceneProcessor, segmenter: Mock, matcher: Mock, frame: np.ndarray
) -> None:
    box = (0, 0, 4, 4)
    mask = np.zeros(frame.shape[:2], dtype=bool)
    mask[0, 0] = True  # one pixel, below the minimum set on the processor below
    segmenter.segment.return_value = [Segmentation(box, mask, 0.9)]
    processor.config = SceneProcessorConfig(background_color=(0, 0, 0), min_mask_pixels=2)

    assert processor.prepare_reference(frame, box) is None
    matcher.create_embedding.assert_not_called()


def test_prepare_reference_replaces_background_and_converts_to_rgb(
    processor: SceneProcessor, segmenter: Mock, matcher: Mock, frame: np.ndarray
) -> None:
    box = (2, 2, 6, 6)
    mask = np.zeros(frame.shape[:2], dtype=bool)
    mask[2:6, 2:4] = True  # keep left half of the crop, replace the right half
    segmenter.segment.return_value = [Segmentation(box, mask, 0.9)]
    processor.config = SceneProcessorConfig(background_color=(9, 8, 7), min_mask_pixels=1)

    reference = processor.prepare_reference(frame, box)

    assert reference is not None
    pixels = np.array(reference.crop)
    assert pixels.shape == (4, 4, 3)
    # Kept pixels are the frame's BGR(10, 20, 30) converted to RGB.
    np.testing.assert_array_equal(pixels[:, :2], np.full((4, 2, 3), (30, 20, 10)))
    # Replaced pixels use the configured RGB background colour directly.
    np.testing.assert_array_equal(pixels[:, 2:], np.full((4, 2, 3), (9, 8, 7)))
    matcher.create_embedding.assert_called_once_with(reference.crop)
    np.testing.assert_array_equal(reference.embedding, matcher.create_embedding.return_value)


def test_process_returns_pending_without_matching_when_unusable(
    processor: SceneProcessor, segmenter: Mock, matcher: Mock, frame: np.ndarray
) -> None:
    segmenter.segment.return_value = [None]

    result = processor.process(frame, (0, 0, 4, 4), source_track_id=7, gallery=[])

    assert result == IdentityDecision("pending", 7, None, 0.0, None)
    assert result.reason == "no_mask"
    matcher.match_candidate.assert_not_called()


def test_process_defers_as_pending_on_ambiguous_low_margin_match(
    processor: SceneProcessor, segmenter: Mock, matcher: Mock, frame: np.ndarray
) -> None:
    """A low-margin tie between two-or-more plausible items must not silently
    become a wrong duplicate; retry later with the existing PENDING cooldown
    instead of falling through to NEW (see scene_processor.py's TODO(UI))."""
    box = (0, 0, 4, 4)
    segmenter.segment.return_value = [Segmentation(box, full_mask(box, frame.shape[:2]), 0.9)]
    candidates = (CandidateScore(42, 0.6, 0.6, None, None, 1),)
    matcher.match_candidate.return_value = ReidMatch(
        None, 0.6, False, candidates=candidates, decision_reason="low_margin"
    )

    result = processor.process(frame, box, source_track_id=3, gallery=[])

    assert result.decision == "pending"
    assert result.item_id is None
    assert result.candidates == candidates
    assert result.reason == "low_margin"


def test_process_returns_new_when_nothing_clears_acceptance_threshold(
    processor: SceneProcessor, segmenter: Mock, matcher: Mock, frame: np.ndarray
) -> None:
    """A genuine non-match (no ambiguity, nothing close) still becomes NEW."""
    box = (0, 0, 4, 4)
    segmenter.segment.return_value = [Segmentation(box, full_mask(box, frame.shape[:2]), 0.9)]
    candidates = (CandidateScore(42, 0.1, 0.1, None, None, 1),)
    matcher.match_candidate.return_value = ReidMatch(
        None, 0.1, False, candidates=candidates, decision_reason="below_threshold"
    )

    result = processor.process(frame, box, source_track_id=3, gallery=[])

    assert result.decision == "new"
    assert result.candidates == candidates
    assert result.reason == "below_threshold"


def test_process_returns_existing_on_accepted_match(
    processor: SceneProcessor, segmenter: Mock, matcher: Mock, frame: np.ndarray
) -> None:
    box = (0, 0, 4, 4)
    mask = full_mask(box, frame.shape[:2])
    segmenter.segment.return_value = [Segmentation(box, mask, 0.9)]
    matcher.match_candidate.return_value = ReidMatch(item_id=42, similarity=0.9, accepted=True)
    gallery = [GalleryEntry(42, np.array([[1.0, 0.0]]))]

    result = processor.process(frame, box, source_track_id=3, gallery=gallery)

    assert result.decision == "existing"
    assert result.item_id == 42
    assert result.similarity == pytest.approx(0.9)
    assert result.source_track_id == 3
    assert result.reference is not None
    matcher.match_candidate.assert_called_once_with(
        result.reference.crop,
        gallery,
        aspect_ratio=result.reference.aspect_ratio,
        color_histogram=result.reference.color_histogram,
        source_track_id=3,
    )


def test_process_returns_new_on_rejected_match(
    processor: SceneProcessor, segmenter: Mock, matcher: Mock, frame: np.ndarray
) -> None:
    box = (0, 0, 4, 4)
    mask = full_mask(box, frame.shape[:2])
    segmenter.segment.return_value = [Segmentation(box, mask, 0.9)]
    matcher.match_candidate.return_value = ReidMatch(item_id=None, similarity=0.2, accepted=False)

    result = processor.process(frame, box, source_track_id=3, gallery=[])

    assert result.decision == "new"
    assert result.item_id is None
    assert result.similarity == pytest.approx(0.2)
    assert result.reference is not None


def test_config_loads_custom_values(tmp_path: Path) -> None:
    config_path = tmp_path / "scene_processor.yaml"
    config_path.write_text(
        "background_color: [1, 2, 3]\nmin_mask_pixels: 50\n", encoding="utf-8"
    )

    assert SceneProcessorConfig.from_yaml(config_path) == SceneProcessorConfig((1, 2, 3), 50)
