"""Standalone box-prompted SAM2 masks; no tracking, cropping, or persistence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
import yaml
from numpy.typing import NDArray
from PIL import Image
from transformers import Sam2Model, Sam2Processor


@dataclass(frozen=True, slots=True)
class SegmenterConfig:
    """SAM2 checkpoint and runtime settings, supplied by the caller."""

    model_name: str
    device: str

    @classmethod
    def from_yaml(cls, config_path: Path) -> SegmenterConfig:
        with config_path.open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
        return cls(model_name=str(config["model_name"]), device=str(config["device"]))


@dataclass(frozen=True, slots=True)
class Segmentation:
    """Inverted keep-mask in frame coordinates; score belongs to the raw SAM mask."""

    box: tuple[int, int, int, int]
    mask: NDArray[np.bool_]
    score: float


class SamSegmenter:
    """Load SAM2 once and segment existing detection boxes together per frame.

    Model loading may download uncached weights. This uses PyTorch, not OpenVINO.
    """

    def __init__(self, config: SegmenterConfig) -> None:
        self.config = config
        self._device = torch.device(config.device)
        if self._device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"Configured device is unavailable: {config.device}")
        self._processor = Sam2Processor.from_pretrained(config.model_name)
        self._model = Sam2Model.from_pretrained(config.model_name)
        # Transformers' decorated .to() can expose an unbound signature to type checkers.
        cast(torch.nn.Module, self._model).to(device=self._device)
        self._model.eval()

    @torch.inference_mode()
    def segment(
        self, frame: NDArray[np.uint8], boxes: list[tuple[int, int, int, int]]
    ) -> list[Segmentation | None]:
        """Return one result per box, preserving order; None means no usable mask.

        Input is an OpenCV BGR uint8 frame and integer xyxy boxes in its coordinates.
        Boxes are clipped to frame bounds; empty or reversed boxes raise ValueError.
        The selected SAM mask is inverted to match the Colab convention: True
        pixels are to be retained, False pixels replaced with background colour.
        Frames are not modified and masks are not crops.
        """
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("frame must be a BGR uint8 array with shape [height, width, 3]")
        height, width = frame.shape[:2]
        if height == 0 or width == 0:
            raise ValueError("frame must not be empty")
        if not boxes:
            return []

        clipped_boxes = []
        for box in boxes:
            if len(box) != 4 or any(
                isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                for value in box
            ):
                raise ValueError("boxes must contain four integer xyxy coordinates")
            x1, y1, x2, y2 = (int(value) for value in box)
            clipped = (max(0, x1), max(0, y1), min(width, x2), min(height, y2))
            if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
                raise ValueError("boxes must have positive area inside the frame")
            clipped_boxes.append(clipped)

        image = Image.fromarray(frame[:, :, ::-1].copy())
        inputs = self._processor(
            images=image, input_boxes=[[list(box) for box in clipped_boxes]], return_tensors="pt"
        ).to(self._device)
        outputs = self._model(**inputs, multimask_output=True)
        masks = self._processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), binarize=True
        )[0].cpu().numpy().astype(bool)
        scores = outputs.iou_scores[0].cpu().numpy()
        if (
            masks.ndim != 4
            or masks.shape[0] != len(boxes)
            or masks.shape[2:] != (height, width)
            or scores.shape != masks.shape[:2]
        ):
            raise RuntimeError("SAM2 returned unexpected mask or score dimensions")

        results: list[Segmentation | None] = []
        for box, candidates, candidate_scores in zip(clipped_boxes, masks, scores):
            valid = candidates.any(axis=(1, 2)) & np.isfinite(candidate_scores)
            if not valid.any():
                results.append(None)
                continue
            best = int(np.argmax(np.where(valid, candidate_scores, -np.inf)))
            # Preserve Colab's convention: raw SAM True pixels are replaced.
            keep_mask = ~candidates[best]
            results.append(Segmentation(box, keep_mask, float(candidate_scores[best])))
        return results
