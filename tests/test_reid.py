"""Offline matching tests using synthetic, unit-normalized embeddings."""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from PIL import Image

from project_auto.memory.reid import GalleryEntry, ReidConfig, ReidMatch, ReidMatcher


@pytest.fixture
def matcher() -> ReidMatcher:
    # Exercise the real constructor but prevent pretrained model downloads.
    with (
        patch("project_auto.memory.reid.AutoImageProcessor.from_pretrained"),
        patch("project_auto.memory.reid.AutoModel.from_pretrained"),
    ):
        result = ReidMatcher(ReidConfig("unused-offline-model", "cpu", 0.75, 3))
    result.create_embedding = Mock(return_value=np.array([1.0, 0.0]))
    return result


@pytest.fixture
def crop() -> Image.Image:
    return Image.new("RGB", (8, 8), "red")


def references(*scores: float) -> np.ndarray:
    """Build unit vectors whose cosine similarity to [1, 0] is each score."""
    return np.array([[score, np.sqrt(1.0 - score**2)] for score in scores])


def test_empty_gallery_skips_embedding(matcher: ReidMatcher, crop: Image.Image) -> None:
    assert matcher.match_candidate(crop, []) == ReidMatch(None, 0.0, False)
    matcher.create_embedding.assert_not_called()


def test_best_permanent_item_wins_and_crop_is_embedded_once(
    matcher: ReidMatcher, crop: Image.Image
) -> None:
    gallery = [GalleryEntry(101, references(0.8)), GalleryEntry(902, references(1.0))]

    assert matcher.match_candidate(crop, gallery) == ReidMatch(902, 1.0, True)
    matcher.create_embedding.assert_called_once_with(crop)


@pytest.mark.parametrize(
    ("score", "accepted"), [(0.749, False), (0.75, True), (0.751, True), (-1.0, False)]
)
def test_threshold_is_inclusive(
    matcher: ReidMatcher, crop: Image.Image, score: float, accepted: bool
) -> None:
    result = matcher.match_candidate(crop, [GalleryEntry(101, references(score))])

    assert result.item_id == (101 if accepted else None)
    assert result.similarity == pytest.approx(score)
    assert result.accepted is accepted


def test_top_k_mean_can_beat_an_item_with_one_perfect_reference(
    matcher: ReidMatcher, crop: Image.Image
) -> None:
    gallery = [
        GalleryEntry(101, references(1.0, 0.1, 0.1, -1.0)),
        GalleryEntry(902, references(0.8, 0.8, 0.8, -1.0)),
    ]

    result = matcher.match_candidate(crop, gallery)

    assert result.item_id == 902
    assert result.similarity == pytest.approx(0.8)
    assert result.accepted


@pytest.mark.parametrize("top_k", [1, 2, 3, 10])
def test_top_k_uses_only_available_references(
    matcher: ReidMatcher, crop: Image.Image, top_k: int
) -> None:
    matcher.config = ReidConfig("unused-offline-model", "cpu", 0.75, top_k)

    result = matcher.match_candidate(crop, [GalleryEntry(101, references(0.6, 1.0))])

    assert result.similarity == pytest.approx(1.0 if top_k == 1 else 0.8)
    assert result.item_id == 101
    assert result.accepted


def test_equal_scores_keep_first_gallery_item(matcher: ReidMatcher, crop: Image.Image) -> None:
    gallery = [GalleryEntry(902, references(1.0)), GalleryEntry(101, references(1.0))]

    assert matcher.match_candidate(crop, gallery) == ReidMatch(902, 1.0, True)


def test_matching_preserves_gallery_query_and_crop(
    matcher: ReidMatcher, crop: Image.Image
) -> None:
    embedding = references(0.9, 0.8)
    original_embedding = embedding.copy()
    query = matcher.create_embedding.return_value
    original_query = query.copy()
    original_pixels = crop.tobytes()
    entry = GalleryEntry(101, embedding)
    gallery = [entry]

    matcher.match_candidate(crop, gallery)

    assert gallery == [entry]
    assert gallery[0] is entry
    np.testing.assert_array_equal(embedding, original_embedding)
    np.testing.assert_array_equal(query, original_query)
    assert crop.tobytes() == original_pixels


def test_config_loads_custom_values(tmp_path: Path) -> None:
    config_path = tmp_path / "reid.yaml"
    config_path.write_text(
        "model_name: local-model\ndevice: cpu\nacceptance_threshold: 0.85\ntop_k: 2\n",
        encoding="utf-8",
    )

    assert ReidConfig.from_yaml(config_path) == ReidConfig("local-model", "cpu", 0.85, 2)
