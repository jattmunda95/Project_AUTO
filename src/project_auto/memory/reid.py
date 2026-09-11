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

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from numpy.typing import NDArray
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


@dataclass(frozen=True, slots=True)
class ReidConfig:
    """Configuration for embedding extraction and match acceptance."""

    model_name: str
    device: str
    acceptance_threshold: float
    top_k: int
    prototype_shortlist_size: int = 3

    @classmethod
    def from_yaml(cls, config_path: Path) -> ReidConfig:
        with config_path.open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)

        return cls(
            model_name=str(config["model_name"]),
            device=str(config["device"]),
            acceptance_threshold=float(config["acceptance_threshold"]),
            top_k=int(config["top_k"]),
            prototype_shortlist_size=int(config.get("prototype_shortlist_size", 3)),
        )


@dataclass(frozen=True, slots=True)
class GalleryEntry:
    """Known reference embeddings for one permanently identified item."""

    item_id: int
    embeddings: NDArray  # shape [num_references, embedding_size]
    prototype: NDArray | None = None  # unit-normalized mean of embeddings, if known


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

    def match_candidate(
        self,
        crop: Image.Image,
        gallery: list[GalleryEntry],
    ) -> ReidMatch:
        """Compare a detection crop against eligible items and decide a match.

        First shortlists the items whose prototype is closest to the query, then
        scores only those items by their per-reference top-k mean similarity.
        Does not mutate tracker or database state; callers own persistence.
        """
        if not gallery:
            print("[ReID] match_candidate: gallery is empty, nothing to match against -> NEW")
            return ReidMatch(item_id=None, similarity=0.0, accepted=False)

        query_embedding = self.create_embedding(crop)
        shortlist = self._shortlist_by_prototype(query_embedding, gallery)

        best_item_id: int | None = None
        best_score = float("-inf")
        per_item_scores: list[tuple[int, float]] = []

        for entry in shortlist:
            similarities = entry.embeddings @ query_embedding
            k = min(self.config.top_k, len(similarities))
            top_k_mean = float(np.sort(similarities)[-k:].mean())
            per_item_scores.append((entry.item_id, top_k_mean))

            if top_k_mean > best_score:
                best_score = top_k_mean
                best_item_id = entry.item_id

        accepted = best_score >= self.config.acceptance_threshold

        print(
            f"[ReID] match_candidate: gallery_size={len(gallery)} shortlisted={len(shortlist)} "
            f"scores={per_item_scores} best_item_id={best_item_id} best_score={best_score:.4f} "
            f"threshold={self.config.acceptance_threshold:.4f} accepted={accepted}"
        )

        return ReidMatch(
            item_id=best_item_id if accepted else None,
            similarity=best_score,
            accepted=accepted,
        )
