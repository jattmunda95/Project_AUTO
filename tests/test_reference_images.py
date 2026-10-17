"""Reference-thumbnail tests: what gets written, where, and what happens when it fails.

The store is pure filesystem work, so these use tmp_path and a fixed name factory; no
database, worker, camera or model is involved.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest
from PIL import Image

from project_auto.utils.reference_images import DEFAULT_MAX_EDGE, ReferenceImageStore


@pytest.fixture
def names() -> object:
    """Deterministic file stems, so assertions can name the file they expect."""
    counter = itertools.count(1)
    return lambda: f"view{next(counter)}"


@pytest.fixture
def store(tmp_path: Path, names: object) -> ReferenceImageStore:
    return ReferenceImageStore(root=tmp_path / "reference_images", name_factory=names)


def crop(width: int = 400, height: int = 300, mode: str = "RGB") -> Image.Image:
    """Build a plain test crop of a given size; single-channel modes take a scalar."""
    colour = (30, 90, 200) if mode in {"RGB", "RGBA"} else 120
    return Image.new(mode, (width, height), color=colour)


class TestSaving:
    def test_a_crop_is_written_under_its_item(self, store: ReferenceImageStore) -> None:
        recorded = store.save(7, crop())

        assert recorded == Path("7") / "view1.png"
        assert (store.root / recorded).is_file()

    def test_the_recorded_path_is_relative_so_the_checkout_can_move(
        self, store: ReferenceImageStore
    ) -> None:
        """Only a relative path belongs in the database; the root is configuration."""
        recorded = store.save(7, crop())

        assert recorded is not None
        assert not recorded.is_absolute()
        assert store.root.as_posix() not in recorded.as_posix()

    def test_the_root_is_created_on_first_save_not_at_construction(
        self, tmp_path: Path, names: object
    ) -> None:
        store = ReferenceImageStore(root=tmp_path / "later", name_factory=names)
        assert not store.root.exists()

        store.save(1, crop())

        assert store.root.is_dir()

    def test_several_references_for_one_item_do_not_collide(
        self, store: ReferenceImageStore
    ) -> None:
        first = store.save(7, crop())
        second = store.save(7, crop())

        assert first != second
        assert len(list((store.root / "7").iterdir())) == 2

    def test_different_items_are_kept_apart(self, store: ReferenceImageStore) -> None:
        store.save(7, crop())
        store.save(8, crop())

        assert sorted(path.name for path in store.root.iterdir()) == ["7", "8"]


class TestThumbnailing:
    def test_a_large_crop_is_scaled_to_the_long_edge(self, store: ReferenceImageStore) -> None:
        recorded = store.save(7, crop(1200, 600))

        assert recorded is not None
        with Image.open(store.root / recorded) as saved:
            assert max(saved.size) == DEFAULT_MAX_EDGE
            assert saved.size == (DEFAULT_MAX_EDGE, DEFAULT_MAX_EDGE // 2)

    def test_a_small_crop_is_never_upscaled(self, store: ReferenceImageStore) -> None:
        """Enlarging a crop invents detail and makes a thumbnail look better than its view."""
        recorded = store.save(7, crop(64, 48))

        assert recorded is not None
        with Image.open(store.root / recorded) as saved:
            assert saved.size == (64, 48)

    def test_scaling_can_be_switched_off(self, tmp_path: Path, names: object) -> None:
        store = ReferenceImageStore(root=tmp_path / "full", max_edge=0, name_factory=names)

        recorded = store.save(7, crop(1200, 600))

        assert recorded is not None
        with Image.open(store.root / recorded) as saved:
            assert saved.size == (1200, 600)

    def test_a_non_rgb_crop_is_converted(self, store: ReferenceImageStore) -> None:
        recorded = store.save(7, crop(mode="L"))

        assert recorded is not None
        with Image.open(store.root / recorded) as saved:
            assert saved.mode == "RGB"

    def test_the_original_crop_is_left_untouched(self, store: ReferenceImageStore) -> None:
        """PIL's thumbnail() resizes in place, so the store must work on a copy.

        The same PreparedReference.crop is also what the matcher and the overlay use.
        """
        original = crop(1200, 600)

        store.save(7, original)

        assert original.size == (1200, 600)

    def test_jpeg_output_gets_a_jpg_suffix(self, tmp_path: Path, names: object) -> None:
        store = ReferenceImageStore(root=tmp_path / "jpeg", image_format="JPEG", name_factory=names)

        recorded = store.save(7, crop())

        assert recorded == Path("7") / "view1.jpg"
        assert (store.root / recorded).is_file()


class TestFailureNeverLosesAReference:
    def test_an_unwritable_destination_returns_none_instead_of_raising(
        self, tmp_path: Path, names: object
    ) -> None:
        """A thumbnail is evidence; losing one must never lose the reference itself."""
        blocked = tmp_path / "blocked"
        blocked.write_text("not a directory", encoding="utf-8")
        store = ReferenceImageStore(root=blocked, name_factory=names)

        assert store.save(7, crop()) is None

    def test_an_unsupported_format_returns_none(self, tmp_path: Path, names: object) -> None:
        store = ReferenceImageStore(
            root=tmp_path / "bad", image_format="NOT_A_FORMAT", name_factory=names
        )

        assert store.save(7, crop()) is None

    def test_a_failed_save_logs_one_error_line(
        self, tmp_path: Path, names: object, capsys: pytest.CaptureFixture[str]
    ) -> None:
        store = ReferenceImageStore(
            root=tmp_path / "bad", image_format="NOT_A_FORMAT", name_factory=names
        )

        store.save(7, crop())

        output = capsys.readouterr().out
        assert output.count("ERROR") == 1
        assert "reference_image_save" in output


class TestResolving:
    def test_a_recorded_path_resolves_under_the_root(self, store: ReferenceImageStore) -> None:
        recorded = store.save(7, crop())

        assert recorded is not None
        assert store.resolve(recorded) == store.root / recorded
        assert store.resolve(str(recorded)).is_file()

    def test_an_absolute_recorded_path_is_returned_unchanged(
        self, store: ReferenceImageStore, tmp_path: Path
    ) -> None:
        """Rows written by hand, or before this module existed, must still load."""
        elsewhere = tmp_path / "elsewhere" / "old.png"

        assert store.resolve(elsewhere) == elsewhere


class TestRemoval:
    def test_one_items_thumbnails_are_removed(self, store: ReferenceImageStore) -> None:
        store.save(7, crop())
        store.save(7, crop())
        store.save(8, crop())

        assert store.remove_item_images(7) == 2
        assert not (store.root / "7").exists()
        assert (store.root / "8").is_dir()

    def test_removing_an_item_with_no_thumbnails_is_not_an_error(
        self, store: ReferenceImageStore
    ) -> None:
        assert store.remove_item_images(99) == 0


class TestAppWiring:
    """run_app builds the store from scene_processor.yaml and keeps diagnostics out of it."""

    def test_the_shipped_config_enables_thumbnails_under_a_git_ignored_folder(self) -> None:
        import subprocess

        import yaml

        from project_auto.app import _build_reference_images

        root = Path(__file__).resolve().parents[1]
        settings = yaml.safe_load((root / "configs" / "scene_processor.yaml").read_text("utf-8"))

        built = _build_reference_images(root, settings)

        assert built is not None
        assert built.root == root / "data" / "reference_images"
        ignored = subprocess.run(
            ["git", "check-ignore", "-q", str(built.root / "1" / "x.png")],
            cwd=root,
            check=False,
        )
        assert ignored.returncode == 0, "saved references must never be committed to Git"

    def test_an_empty_directory_setting_turns_thumbnails_off(self, tmp_path: Path) -> None:
        from project_auto.app import _build_reference_images

        assert _build_reference_images(tmp_path, {"reference_image_dir": ""}) is None
        assert _build_reference_images(tmp_path, {}) is None

    def test_the_edge_and_folder_come_from_the_settings(self, tmp_path: Path) -> None:
        from project_auto.app import _build_reference_images

        built = _build_reference_images(
            tmp_path, {"reference_image_dir": "pics", "reference_image_max_edge": 128}
        )

        assert built is not None
        assert built.root == tmp_path / "pics"
        assert built.max_edge == 128
