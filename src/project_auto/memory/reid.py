"""Standalone embedding extraction and permanent-item similarity matching.

Subfunctions:
- ReidConfig loads model/device, acceptance threshold, reference top-k, and the
  prototype-shortlist size.
- GalleryEntry holds one permanent item's references and its stored prototype;
  ReidMatch reports the decision.
- ReidMatcher loads DINOv2 and creates a normalized embedding from a prepared RGB crop.
- match_candidate first shortlists the closest permanent items by prototype similarity,
  then compares only their references, averages each shortlisted item's top-k scores,
  and applies the acceptance threshold to the best of those.

The caller supplies eligible gallery entries; store.py loads them, prototype included.
No database writes, tracker updates, ADD/RETURNED decisions, or capture scheduling
occur here. Real-image identity reliability remains unverified.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from numpy.typing import NDArray
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from project_auto.perception.descriptors import color_similarity, size_similarity
from project_auto.utils.logging import Category, log_action

# Final score = DINO_WEIGHT * dino_similarity + COLOR_WEIGHT * color_similarity
#             + ASPECT_WEIGHT * aspect_similarity
DINO_WEIGHT = 0.65
COLOR_WEIGHT = 0.20
ASPECT_WEIGHT = 0.15


@dataclass(frozen=True, slots=True)
class ReidConfig:
    """Configuration for embedding extraction and match acceptance."""

    # TODO(calibrate): acceptance_threshold=0.55 and margin_threshold=0.15 (configs/reid.yaml)
    # are uncalibrated for the current weighted score (0.65 DINO + 0.20 color + 0.15 aspect).
    # Tune both against logged same/different-item scores in logs/reid_match_log.csv.
    model_name: str
    device: str
    acceptance_threshold: float
    top_k: int
    prototype_shortlist_size: int = 3
    # The best-scoring item must beat the runner-up by at least this much to be
    # accepted; 0.0 (the default) disables the check. Guards against accepting a
    # weak best-of-a-bad-lot when two items score nearly the same.
    margin_threshold: float = 0.0
    # Optional CSV path; when set, match_candidate appends one diagnostic row per
    # call (scores, margin, threshold, accept/reject) for offline threshold tuning.
    match_log_path: Path | None = None

    @classmethod
    def from_yaml(cls, config_path: Path) -> ReidConfig:
        with config_path.open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)

        match_log_path = config.get("match_log_path")
        return cls(
            model_name=str(config["model_name"]),
            device=str(config["device"]),
            acceptance_threshold=float(config["acceptance_threshold"]),
            top_k=int(config["top_k"]),
            prototype_shortlist_size=int(config.get("prototype_shortlist_size", 3)),
            margin_threshold=float(config.get("margin_threshold", 0.0)),
            match_log_path=Path(match_log_path) if match_log_path else None,
        )


@dataclass(frozen=True, slots=True)
class GalleryEntry:
    """Known reference embeddings for one permanently identified item."""

    item_id: int
    embeddings: NDArray  # shape [num_references, embedding_size]
    prototype: NDArray | None = None  # unit-normalized mean of embeddings, if known
    aspect_ratios: NDArray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    # shape [num_references, h_bins * s_bins]; NaN rows mark references predating
    # this descriptor and are excluded from the color similarity's top-k pool.
    color_histograms: NDArray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float32))


@dataclass(frozen=True, slots=True)
class ReidMatch:
    """Result of comparing one detection crop against the gallery."""

    item_id: int | None
    similarity: float
    accepted: bool


class ReidMatcher:
    """Extract embeddings and match detection crops against known items."""

    def __init__(self, config: ReidConfig) -> None:
        self.config = config
        self._processor = AutoImageProcessor.from_pretrained(config.model_name)
        self._model = AutoModel.from_pretrained(config.model_name)
        self._model.to(config.device)
        self._model.eval()

    @torch.inference_mode()
    def create_embedding(self, image: Image.Image) -> NDArray:
        """Return a normalized embedding vector for one object crop."""
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {name: tensor.to(self.config.device) for name, tensor in inputs.items()}

        outputs = self._model(**inputs)
        # CLS token represents the overall image.
        embedding = outputs.last_hidden_state[:, 0, :]
        # Normalize so dot product can be used as cosine similarity.
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding.squeeze(0).cpu().numpy()

    def _shortlist_by_prototype(
        self,
        query_embedding: NDArray,
        gallery: list[GalleryEntry],
    ) -> list[GalleryEntry]:
        """Return the closest entries by prototype similarity, capped at the config size.

        An entry without a stored prototype cannot be ranked this way and is always
        kept, since excluding it would silently drop an otherwise-eligible item.
        """
        size = self.config.prototype_shortlist_size
        if size <= 0 or len(gallery) <= size:
            return gallery

        ranked: list[tuple[float, GalleryEntry]] = []
        unranked: list[GalleryEntry] = []
        for entry in gallery:
            if entry.prototype is None:
                unranked.append(entry)
            else:
                ranked.append((float(entry.prototype @ query_embedding), entry))
        ranked.sort(key=lambda scored: scored[0], reverse=True)

        return [entry for _, entry in ranked[:size]] + unranked

    def _top_k_mean(self, similarities: NDArray) -> float:
        """Return the mean of the top-k values, k capped at how many are available."""
        k = min(self.config.top_k, len(similarities))
        return float(np.sort(similarities)[-k:].mean())

    def _color_score(
        self,
        query_histogram: NDArray | None,
        entry: GalleryEntry,
    ) -> float | None:
        """Top-k mean color similarity across an entry's references, or None.

        None means either the query or every one of the entry's references lacks
        a histogram (e.g. captured before this descriptor existed) — the caller
        falls back to embedding-only scoring rather than guessing a value.
        """
        if query_histogram is None or entry.color_histograms.shape[0] == 0:
            return None
        valid_rows = ~np.isnan(entry.color_histograms).any(axis=1)
        if not valid_rows.any():
            return None
        flat_query = np.asarray(query_histogram, dtype=np.float32).reshape(-1)
        similarities = np.array(
            [
                color_similarity(flat_query, row)
                for row in entry.color_histograms[valid_rows]
            ]
        )
        return self._top_k_mean(similarities)

    def _aspect_score(self, query_ratio: float | None, entry: GalleryEntry) -> float | None:
        """Top-k mean size similarity across an entry's references, or None.

        None means either the query or every one of the entry's references lacks
        an aspect ratio (e.g. captured before this descriptor existed).
        """
        if query_ratio is None or entry.aspect_ratios.size == 0:
            return None
        valid_ratios = entry.aspect_ratios[~np.isnan(entry.aspect_ratios)]
        if valid_ratios.size == 0:
            return None
        similarities = np.array([size_similarity(query_ratio, ratio) for ratio in valid_ratios])
        return self._top_k_mean(similarities)

    _MATCH_LOG_HEADER = (
        "timestamp",
        "source_track_id",
        "gallery_size",
        "shortlisted",
        "best_item_id",
        "best_score",
        "second_best_score",
        "margin",
        "dino_score",
        "color_score",
        "aspect_score",
        "acceptance_threshold",
        "margin_threshold",
        "accepted",
    )

    def _log_match(self, row: dict[str, object]) -> None:
        """Append one diagnostic row to config.match_log_path; writes the header once."""
        log_path = self.config.match_log_path
        if log_path is None:
            return
        log_path.parent.mkdir(parents=True, exist_ok=True)
        is_new_file = not log_path.exists()
        with log_path.open("a", newline="", encoding="utf-8") as log_file:
            writer = csv.DictWriter(log_file, fieldnames=self._MATCH_LOG_HEADER)
            if is_new_file:
                writer.writeheader()
            writer.writerow(row)

    def match_candidate(
        self,
        crop: Image.Image,
        gallery: list[GalleryEntry],
        aspect_ratio: float | None = None,
        color_histogram: NDArray | None = None,
        source_track_id: int | None = None,
    ) -> ReidMatch:
        """Compare a detection crop against eligible items and decide a match.

        First shortlists the items whose prototype is closest to the query, then
        scores only those items by their per-reference top-k mean similarity.
        When aspect_ratio and color_histogram are both supplied and an item has
        at least one reference with that descriptor, its score is the weighted
        sum (DINO_WEIGHT, COLOR_WEIGHT, ASPECT_WEIGHT); otherwise the item's
        score falls back to embedding similarity alone.

        Acceptance requires both the acceptance_threshold and margin_threshold:
        the best score must clear the threshold, and must beat the runner-up
        score by at least margin_threshold (a single shortlisted item has no
        runner-up, so the margin check is automatically satisfied). source_track_id
        is optional and used only to correlate the diagnostic CSV log, when
        config.match_log_path is set, with the caller's track.

        Does not mutate tracker or database state; callers own persistence.
        """
        if not gallery:
            log_action(Category.NEW, track=source_track_id, reason="gallery_empty")
            return ReidMatch(item_id=None, similarity=0.0, accepted=False)

        query_embedding = self.create_embedding(crop)
        shortlist = self._shortlist_by_prototype(query_embedding, gallery)

        # (item_id, final_score, dino_score, color_score, aspect_score)
        scored: list[tuple[int, float, float, float | None, float | None]] = []

        for entry in shortlist:
            similarities = entry.embeddings @ query_embedding
            dino_score = self._top_k_mean(similarities)

            color_score = self._color_score(color_histogram, entry)
            aspect_score = self._aspect_score(aspect_ratio, entry)

            if color_score is not None and aspect_score is not None:
                final_score = (
                    DINO_WEIGHT * dino_score
                    + COLOR_WEIGHT * color_score
                    + ASPECT_WEIGHT * aspect_score
                )
            else:
                final_score = dino_score

            scored.append((entry.item_id, final_score, dino_score, color_score, aspect_score))

        scored.sort(key=lambda item: item[1], reverse=True)
        best_item_id, best_score, best_dino, best_color, best_aspect = scored[0]
        second_best_score = scored[1][1] if len(scored) > 1 else None
        margin = float("inf") if second_best_score is None else best_score - second_best_score

        # TODO(UI): a margin failure here (best clears acceptance_threshold but
        # not margin_threshold) means two-or-more items are plausible, not that
        # none are. Once the app has a UI, surface this as an UNKNOWN status
        # with the tied candidate item_ids so a user can pick the right one,
        # instead of silently falling through to a brand-new item identity as
        # scene_processor.process currently does with any rejected match.
        accepted = (
            best_score >= self.config.acceptance_threshold
            and margin >= self.config.margin_threshold
        )
        ambiguous = not accepted and best_score >= self.config.acceptance_threshold

        if accepted:
            log_action(
                Category.MATCH,
                track=source_track_id,
                item=best_item_id,
                score=best_score,
                margin=margin,
                dino=best_dino,
                color=best_color,
                aspect=best_aspect,
            )
        elif ambiguous:
            log_action(
                Category.AMBIGUOUS,
                track=source_track_id,
                best_item=best_item_id,
                score=best_score,
                margin=margin,
                margin_threshold=self.config.margin_threshold,
            )
        else:
            log_action(
                Category.NEW,
                track=source_track_id,
                reason="below_threshold",
                best_item=best_item_id,
                score=best_score,
                threshold=self.config.acceptance_threshold,
            )

        self._log_match(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source_track_id": source_track_id,
                "gallery_size": len(gallery),
                "shortlisted": len(shortlist),
                "best_item_id": best_item_id,
                "best_score": best_score,
                "second_best_score": second_best_score,
                "margin": margin,
                "dino_score": best_dino,
                "color_score": best_color,
                "aspect_score": best_aspect,
                "acceptance_threshold": self.config.acceptance_threshold,
                "margin_threshold": self.config.margin_threshold,
                "accepted": accepted,
            }
        )

        return ReidMatch(
            item_id=best_item_id if accepted else None,
            similarity=best_score,
            accepted=accepted,
        )
