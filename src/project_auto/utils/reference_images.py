"""Save a thumbnail of each stored reference view, so items can be shown, not just named.

Subfunctions:
- ReferenceImageStore.save writes one prepared crop as a downscaled image and returns the
  path to record in ItemEmbedding.object_image_path.
- ReferenceImageStore.resolve turns a recorded path back into an absolute one for display.
- ReferenceImageStore.remove_item_images deletes one item's thumbnails.

Why this exists: the views table has carried an object_image_path column all along, but
nothing ever wrote it, so a stored view kept an embedding, a histogram and an aspect ratio
and no picture. A question asking "is this the same object?" is close to useless without
one, so the ask-and-answer flow needs thumbnails to exist before it can show them.

Only the path goes in the database, never the image bytes. Paths are recorded relative to
the configured root, so moving or re-syncing the checkout does not invalidate them.

Saving a thumbnail must never fail an identity job: save() reports failure by returning
None and logging one ERROR line, and the caller persists the reference either way. It is
deliberately the caller's job to call this on a background thread -- the capture path
already runs in the identification worker. Never call it from the camera loop.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from project_auto.utils.logging import Category, log_action

# Long edge of a saved thumbnail, in pixels. Large enough to recognise an object beside
# its candidates in the overlay, small enough that a full gallery stays a few megabytes:
# at the 8-references-per-item cap, 256px PNGs run tens of kilobytes each.
DEFAULT_MAX_EDGE = 256


def _default_name() -> str:
    """Return a sortable, collision-resistant file stem."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


@dataclass(slots=True)
class ReferenceImageStore:
    """Write and locate reference thumbnails under one root directory.

    root is created on first save, not at construction, so configuring a path costs
    nothing until a reference is actually stored. name_factory is injectable so tests get
    deterministic filenames; production uses a UTC timestamp plus a short random suffix.
    """

    root: Path
    max_edge: int = DEFAULT_MAX_EDGE
    image_format: str = "PNG"
    name_factory: Callable[[], str] = field(default=_default_name, repr=False)

    def save(self, item_id: int, crop: Image.Image) -> Path | None:
        """Save one crop for one item; return the path to record, or None on failure.

        The returned path is relative to root, which is what belongs in the database.
        Never raises: a thumbnail is evidence, and losing one must not lose a reference.
        """
        try:
            relative = Path(str(item_id)) / f"{self.name_factory()}.{self._suffix()}"
            destination = self.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            self._thumbnail(crop).save(destination, format=self.image_format)
            return relative
        # Deliberately broad: a reference must still be saved when its picture cannot be.
        # Pillow raises a bare KeyError for an unknown format, OSError for an unwritable
        # path, and anything a plugin chooses in between, so narrowing this by type has
        # already let a failure through once.
        except Exception as error:  # noqa: BLE001
            log_action(
                Category.ERROR,
                item=item_id,
                action="reference_image_save",
                error=type(error).__name__,
            )
            return None

    def resolve(self, recorded_path: str | Path) -> Path:
        """Return the absolute path of a recorded thumbnail, for display.

        An absolute recorded path is returned unchanged, so rows written before this
        module existed, or by hand, still load.
        """
        path = Path(recorded_path)
        return path if path.is_absolute() else self.root / path

    def discard(self, recorded_path: str | Path) -> None:
        """Delete one thumbnail whose reference was never stored; never raises.

        The capture path writes the picture before the database decides whether the
        reference fits, so a refused reference (gallery full) must not leave a file behind.
        """
        try:
            self.resolve(recorded_path).unlink(missing_ok=True)
        except OSError:
            log_action(Category.ERROR, action="reference_image_discard", path=str(recorded_path))

    def remove_item_images(self, item_id: int) -> int:
        """Delete one item's thumbnails and return how many files were removed.

        Nothing calls this during normal tracking: a removed item keeps its row and its
        history, so it keeps its pictures. It exists for explicit cleanup only.
        """
        directory = self.root / str(item_id)
        removed = 0
        if not directory.is_dir():
            return removed
        for path in sorted(directory.iterdir()):
            if path.is_file():
                path.unlink()
                removed += 1
        if not any(directory.iterdir()):
            directory.rmdir()
        return removed

    def _thumbnail(self, crop: Image.Image) -> Image.Image:
        """Return an RGB copy scaled so its long edge fits max_edge; never upscale."""
        image = crop.convert("RGB")
        if self.max_edge > 0 and max(image.size) > self.max_edge:
            image.thumbnail((self.max_edge, self.max_edge), Image.Resampling.LANCZOS)
        return image

    def _suffix(self) -> str:
        """Return the filename extension for the configured image format."""
        return "jpg" if self.image_format.upper() in {"JPEG", "JPG"} else "png"
