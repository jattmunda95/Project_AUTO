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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from numpy.typing import NDArray
from PIL import Image

from project_auto.memory.reid import GalleryEntry, ReidMatcher
from project_auto.perception.descriptors import aspect_ratio as compute_aspect_ratio
from project_auto.perception.descriptors import hsv_histogram
from project_auto.perception.segmenter import SamSegmenter
from project_auto.utils.logging import Category, log_action

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
    aspect_ratio: float = 0.0
    # Flattened H+S histogram, computed from the masked crop before background
    # replacement so the fill colour never enters the distribution.
    color_histogram: NDArray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    # Evidence carried for the reference-quality gate applied by the capture path.
    # prepare_reference stays permissive (identity resolution must still work on an
    # imperfect view); only the caller saving a reference thresholds these.
    sam_score: float = 0.0
    # Masked foreground pixels divided by the segmented box area. Low values mean
    # the box mostly contains something other than the segmented object.
    # TODO(occlusion): this is a mask-sanity measure, NOT occlusion detection. It
    # cannot distinguish a physically small object from a partially hidden one;
    # true partial-occlusion reasoning needs evidence this pipeline does not have.
    mask_occupancy: float = 0.0


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
        source_track_id: int | None = None,
    ) -> PreparedReference | None:
        """Crop, mask, and embed one detection box; None when it is unusable.

        The segmenter clips and validates the box. Its keep-mask (True retains) is
        applied within the crop, background pixels are replaced with the configured
        colour, and the result is embedded with ReID's own preprocessing.
        source_track_id is optional and used only for diagnostic output.
        """
        segmentation = self._segmenter.segment(frame, [box])[0]
        if segmentation is None:
            log_action(Category.REJECT, track=source_track_id, box=box, reason="no_mask")
            return None

        x1, y1, x2, y2 = segmentation.box
        mask_pixels = int(segmentation.mask[y1:y2, x1:x2].sum())
        if mask_pixels < self.config.min_mask_pixels:
            log_action(
                Category.REJECT,
                track=source_track_id,
                box=box,
                reason="mask_too_small",
                mask_pixels=mask_pixels,
                min_mask_pixels=self.config.min_mask_pixels,
                sam_score=segmentation.score,
            )
            return None

        crop = frame[y1:y2, x1:x2]
        mask = segmentation.mask[y1:y2, x1:x2]
        crop_rgb = crop[:, :, ::-1]

        # Computed on the foreground pixels before the fill colour is applied,
        # so it never contaminates the color descriptor.
        color_histogram = hsv_histogram(crop_rgb, mask).reshape(-1)

        background_rgb = np.array(self.config.background_color, dtype=np.uint8)
        masked_rgb = np.where(mask[:, :, None], crop_rgb, background_rgb)

        image = Image.fromarray(masked_rgb)
        embedding = self._matcher.create_embedding(image)
        box_area = (x2 - x1) * (y2 - y1)
        return PreparedReference(
            crop=image,
            embedding=embedding,
            aspect_ratio=compute_aspect_ratio(segmentation.box),
            color_histogram=color_histogram,
            sam_score=float(segmentation.score),
            mask_occupancy=(mask_pixels / box_area) if box_area > 0 else 0.0,
        )

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
        # TODO(perf): prepare_reference always runs full SAM+DINO inference even when the
        # gallery is nonempty; measure whether the resolve path is redoing embedding work
        # that a cheaper prefilter could skip before optimizing.
        reference = self.prepare_reference(frame, box, source_track_id)
        if reference is None:
            # prepare_reference already logged the REJECT reason (no_mask/mask_too_small).
            return IdentityDecision(
                decision="pending",
                source_track_id=source_track_id,
                item_id=None,
                similarity=0.0,
                reference=None,
            )

        match = self._matcher.match_candidate(
            reference.crop,
            gallery,
            aspect_ratio=reference.aspect_ratio,
            color_histogram=reference.color_histogram,
            source_track_id=source_track_id,
        )
        # TODO(UI): match.accepted is False both when nothing looked close (a
        # genuine NEW item) and when a margin_threshold tie left two-or-more
        # plausible items (see reid.py's TODO). Once the app has a UI, the
        # margin-failure case should become its own "unknown" IdentityDecision
        # carrying the tied item_ids, so a user can disambiguate, instead of
        # both cases being flattened into "new" here.
        # match_candidate already logged the MATCH/AMBIG/NEW decision with full scores.
        decision: IdentityDecisionType = "existing" if match.accepted else "new"
        return IdentityDecision(
            decision=decision,
            source_track_id=source_track_id,
            item_id=match.item_id if match.accepted else None,
            similarity=match.similarity,
            reference=reference,
        )
