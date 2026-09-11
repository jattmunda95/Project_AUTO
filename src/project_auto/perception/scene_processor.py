"""Standalone image preparation and identity comparison.

Subfunctions:
- SceneProcessorConfig loads the background fill colour and minimum usable mask size.
- prepare_reference crops a detection, segments it, applies SAM's keep-mask,
  replaces background pixels, and embeds the result with ReID's preprocessing.
- process compares a prepared reference against the caller's eligible gallery and
  returns NEW, EXISTING, or PENDING plus evidence.

Inputs: a BGR frame, a detection box, a source track ID, and an eligible in-memory
gallery already filtered by the caller for model/status compatibility.
Outputs: an IdentityDecision carrying source_track_id, an optional permanent item_id,
similarity, and the reusable PreparedReference (crop and embedding) when usable.
An unusable crop/mask always yields PENDING, never NEW.

No tracker-stage inspection, capture scheduling, database writes, or event emission.
app.py chooses when to call; store.py persists; the event layer chooses transitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from numpy.typing import NDArray
from PIL import Image

from project_auto.memory.reid import GalleryEntry, ReidMatcher
from project_auto.perception.segmenter import SamSegmenter

IdentityDecisionType = Literal["new", "existing", "pending"]


@dataclass(frozen=True, slots=True)
class SceneProcessorConfig:
    """Background fill colour and minimum usable mask size, supplied by the caller."""

    background_color: tuple[int, int, int]  # RGB
    min_mask_pixels: int

    @classmethod
    def from_yaml(cls, config_path: Path) -> SceneProcessorConfig:
        with config_path.open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
        red, green, blue = config["background_color"]
        return cls(
            background_color=(int(red), int(green), int(blue)),
            min_mask_pixels=int(config["min_mask_pixels"]),
        )


@dataclass(frozen=True, slots=True)
class PreparedReference:
    """A usable, background-replaced RGB crop plus its normalized embedding."""

    crop: Image.Image
    embedding: NDArray


@dataclass(frozen=True, slots=True)
class IdentityDecision:
    """Proposed identity outcome for one detection; the caller owns persistence."""

    decision: IdentityDecisionType
    source_track_id: int
    item_id: int | None
    similarity: float
    reference: PreparedReference | None


class SceneProcessor:
    """Prepare masked detection crops and propose identity decisions.

    Disconnected from the tracker and database; the coordinator supplies the frame,
    box, track ID, and eligible gallery, and owns every decision this module returns.
    """

    def __init__(
        self,
        config: SceneProcessorConfig,
        segmenter: SamSegmenter,
        matcher: ReidMatcher,
    ) -> None:
        self.config = config
        self._segmenter = segmenter
        self._matcher = matcher

    def prepare_reference(
        self,
        frame: NDArray[np.uint8],
        box: tuple[int, int, int, int],
    ) -> PreparedReference | None:
        """Crop, mask, and embed one detection box; None when it is unusable.

        The segmenter clips and validates the box. Its keep-mask (True retains) is
        applied within the crop, background pixels are replaced with the configured
        colour, and the result is embedded with ReID's own preprocessing.
        """
        segmentation = self._segmenter.segment(frame, [box])[0]
        if segmentation is None:
            print(f"[SceneProcessor] prepare_reference: box={box} -> segmenter returned no usable mask")
            return None

        x1, y1, x2, y2 = segmentation.box
        mask_pixels = int(segmentation.mask[y1:y2, x1:x2].sum())
        if mask_pixels < self.config.min_mask_pixels:
            print(
                f"[SceneProcessor] prepare_reference: box={box} -> mask_pixels={mask_pixels} "
                f"below min_mask_pixels={self.config.min_mask_pixels}, sam_score={segmentation.score:.4f}"
            )
            return None
        print(
            f"[SceneProcessor] prepare_reference: box={box} -> usable mask, mask_pixels={mask_pixels}, "
            f"sam_score={segmentation.score:.4f}"
        )

        crop = frame[y1:y2, x1:x2]
        mask = segmentation.mask[y1:y2, x1:x2]
        background_rgb = np.array(self.config.background_color, dtype=np.uint8)
        crop_rgb = crop[:, :, ::-1]
        masked_rgb = np.where(mask[:, :, None], crop_rgb, background_rgb)

        image = Image.fromarray(masked_rgb)
        embedding = self._matcher.create_embedding(image)
        return PreparedReference(crop=image, embedding=embedding)

    def process(
        self,
        frame: NDArray[np.uint8],
        box: tuple[int, int, int, int],
        source_track_id: int,
        gallery: list[GalleryEntry],
    ) -> IdentityDecision:
        """Prepare one detection and propose NEW, EXISTING, or PENDING.

        An unusable crop/mask always returns PENDING with no reference; the caller
        decides whether and when to retry. A usable crop is always embedded and
        returned, whether or not it matched an existing item.
        """
        reference = self.prepare_reference(frame, box)
        if reference is None:
            print(f"[SceneProcessor] process: track_id={source_track_id} -> PENDING (no usable reference)")
            return IdentityDecision(
                decision="pending",
                source_track_id=source_track_id,
                item_id=None,
                similarity=0.0,
                reference=None,
            )

        match = self._matcher.match_candidate(reference.crop, gallery)
        decision: IdentityDecisionType = "existing" if match.accepted else "new"
        print(
            f"[SceneProcessor] process: track_id={source_track_id} -> decision={decision} "
            f"item_id={match.item_id} similarity={match.similarity:.4f}"
        )
        return IdentityDecision(
            decision=decision,
            source_track_id=source_track_id,
            item_id=match.item_id if match.accepted else None,
            similarity=match.similarity,
            reference=reference,
        )
