"""Standalone associative-memory/ReID matching for permanently identified items.

This module never mutates tracker or database state. It searches embeddings of
eligible permanent items for one detection crop and returns a candidate match
decision. It stays disconnected from RETURNED emission, tracker dispatch, and
the live app until its behavior is implemented and verified (see TASKS.md).
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

    @classmethod
    def from_yaml(cls, config_path: Path) -> ReidConfig:
        with config_path.open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)

        return cls(
            model_name=str(config["model_name"]),
            device=str(config["device"]),
            acceptance_threshold=float(config["acceptance_threshold"]),
            top_k=int(config["top_k"]),
        )


@dataclass(frozen=True, slots=True)
class GalleryEntry:
    """Known reference embeddings for one permanently identified item."""

    item_id: int
    embeddings: NDArray  # shape [num_references, embedding_size]


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

    def match_candidate(
        self,
        crop: Image.Image,
        gallery: list[GalleryEntry],
    ) -> ReidMatch:
        """Compare a detection crop against eligible items and decide a match.

        Does not mutate tracker or database state; callers own persistence.
        """
        if not gallery:
            return ReidMatch(item_id=None, similarity=0.0, accepted=False)

        query_embedding = self.create_embedding(crop)

        best_item_id: int | None = None
        best_score = float("-inf")

        for entry in gallery:
            similarities = entry.embeddings @ query_embedding
            k = min(self.config.top_k, len(similarities))
            top_k_mean = float(np.sort(similarities)[-k:].mean())

            if top_k_mean > best_score:
                best_score = top_k_mean
                best_item_id = entry.item_id

        accepted = best_score >= self.config.acceptance_threshold

        return ReidMatch(
            item_id=best_item_id if accepted else None,
            similarity=best_score,
            accepted=accepted,
        )
