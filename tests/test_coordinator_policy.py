"""Coordinator + identity-policy tests: when the policy is consulted and what it may change.

The score cases come from real queries in logs/reid_runs/20261016T030339Z. The three modes
are the contract: "off" and "observe" must leave behaviour exactly as it was, and only
"enforce" may defer a track instead of adding it.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest
import test_coordinator as shared
from test_coordinator import FakeWorker, add_signal, frame, make_detection, reference

from project_auto.events.coordinator import IdentityCoordinator
from project_auto.events.event_engine import EventEngine
from project_auto.events.identification import IdentificationResult, IdentificationState
from project_auto.events.identity_policy import IdentityPolicy, IdentityPolicyConfig
from project_auto.memory.models import Item
from project_auto.memory.reid import CandidateScore
from project_auto.memory.store import DatabaseStore
from project_auto.perception.scene_processor import PreparedReference

# Register the shared fixtures explicitly; no camera, models, or live database.
store = shared.store
worker = shared.worker

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "identity_policy.yaml"
TRACK = 7


def policy(mode: str) -> IdentityPolicy:
    """The shipped policy configuration, switched to one mode."""
    config = replace(IdentityPolicyConfig.from_yaml(CONFIG_PATH), mode=mode)
    return IdentityPolicy(config=config, acceptance_threshold=0.68, margin_threshold=0.06)


def build(store: DatabaseStore, worker: FakeWorker, mode: str | None) -> IdentityCoordinator:
    return IdentityCoordinator(
        scene_processor=Mock(),
        event_engine=EventEngine(store),
        store=store,
        reid_model_name="test-model",
        max_references_per_item=2,
        initial_capture_spacing_frames=0,
        move_candidate_delay_frames=0,
        candidate_retry_frames=0,
        bad_mask_cooldown_seconds=5.0,
        queue_full_retry_seconds=0.5,
        clock=Mock(return_value=0.0),
        worker=worker,
        identity_policy=policy(mode) if mode is not None else None,
    )


def score(item_id: int, final: float, dino: float | None = None, colour: float | None = None):
    return CandidateScore(
        item_id=item_id,
        final_score=final,
        dino_score=final if dino is None else dino,
        color_score=final if colour is None else colour,
        aspect_score=0.5,
        n_references=5,
    )


def good_reference() -> PreparedReference:
    return replace(reference(), mask_occupancy=0.70, sam_score=0.95)


def near_miss_result() -> IdentificationResult:
    """q00029 of the run: item 7 at 0.654, rejected by the 0.68 threshold, nothing close."""
    return IdentificationResult(
        kind="resolve",
        track_id=TRACK,
        status="new",
        reference=good_reference(),
        candidates=(score(7, 0.654), score(6, 0.322)),
        decision_reason="below_threshold",
    )


def clearly_new_result() -> IdentificationResult:
    """q00017 of the run: the best candidate is 0.34, nothing like it is saved."""
    return IdentificationResult(
        kind="resolve",
        track_id=TRACK,
        status="new",
        reference=good_reference(),
        candidates=(score(8, 0.34), score(5, 0.21)),
        decision_reason="below_threshold",
    )


def apply(coordinator: IdentityCoordinator, worker: FakeWorker, result: IdentificationResult):
    coordinator.handle_frame(frame(), [add_signal(TRACK)], {})
    worker.push_result(result)
    coordinator.handle_frame(frame(), [], {TRACK: make_detection(TRACK)})


@pytest.fixture
def seeded(store: DatabaseStore) -> DatabaseStore:
    """A store holding items 1..8 so candidate ids refer to real rows."""
    for _ in range(8):
        store.create_item("thing")
    return store


class TestPolicyIsInertUnlessEnforcing:
    @pytest.mark.parametrize("mode", [None, "off", "observe"])
    def test_a_near_miss_is_still_added_exactly_as_before(
        self, seeded: DatabaseStore, worker: FakeWorker, mode: str | None
    ) -> None:
        """Observing must not change the outcome: that is the whole point of the mode."""
        before = seeded.count_items()
        coordinator = build(seeded, worker, mode)

        apply(coordinator, worker, near_miss_result())

        assert seeded.count_items() == before + 1

    def test_observe_mode_reports_what_it_would_have_asked(
        self, seeded: DatabaseStore, worker: FakeWorker, capsys: pytest.CaptureFixture[str]
    ) -> None:
        coordinator = build(seeded, worker, "observe")

        apply(coordinator, worker, near_miss_result())

        line = next(row for row in capsys.readouterr().out.splitlines() if "ASK" in row)
        assert "mode=observe" in line
        assert "outcome=ask" in line
        assert "applied=False" in line
        assert "items=7,6" in line

    def test_off_mode_logs_nothing(
        self, seeded: DatabaseStore, worker: FakeWorker, capsys: pytest.CaptureFixture[str]
    ) -> None:
        apply(build(seeded, worker, "off"), worker, near_miss_result())

        assert "ASK" not in capsys.readouterr().out


class TestEnforcing:
    def test_a_near_miss_is_deferred_not_added_as_a_duplicate(
        self, seeded: DatabaseStore, worker: FakeWorker
    ) -> None:
        before = seeded.count_items()
        coordinator = build(seeded, worker, "enforce")

        apply(coordinator, worker, near_miss_result())

        assert seeded.count_items() == before
        assert coordinator._reference_manager.state(TRACK) is IdentificationState.DEFERRED
        assert coordinator._reference_manager.reason(TRACK) == "policy_ask"

    def test_the_deferral_uses_the_policy_cooldown_not_the_bad_mask_one(
        self, seeded: DatabaseStore, worker: FakeWorker
    ) -> None:
        coordinator = build(seeded, worker, "enforce")

        apply(coordinator, worker, near_miss_result())

        eligible_at = coordinator._reference_manager._eligible_at[TRACK]
        assert eligible_at == pytest.approx(
            coordinator.identity_policy.config.track_cooldown_seconds
        )

    def test_a_deferred_track_does_not_retry_until_its_cooldown(
        self, seeded: DatabaseStore, worker: FakeWorker
    ) -> None:
        coordinator = build(seeded, worker, "enforce")
        apply(coordinator, worker, near_miss_result())
        submitted = len(worker.submitted)

        coordinator.handle_frame(frame(), [], {TRACK: make_detection(TRACK)})

        assert len(worker.submitted) == submitted

    def test_a_clearly_new_object_is_still_added(
        self, seeded: DatabaseStore, worker: FakeWorker
    ) -> None:
        """Nothing resembles it, so there is nothing to ask: it is added as it always was."""
        before = seeded.count_items()
        coordinator = build(seeded, worker, "enforce")

        apply(coordinator, worker, clearly_new_result())

        assert seeded.count_items() == before + 1

    def test_a_clearly_new_object_is_added_even_on_a_poor_view(
        self, seeded: DatabaseStore, worker: FakeWorker
    ) -> None:
        """A poor view only ever suppresses questions; it must not stop an add."""
        before = seeded.count_items()
        poor = replace(clearly_new_result(), reference=replace(reference(), mask_occupancy=0.2))
        coordinator = build(seeded, worker, "enforce")

        apply(coordinator, worker, poor)

        assert seeded.count_items() == before + 1

    def test_a_near_miss_on_a_poor_view_is_still_asked_about(
        self, seeded: DatabaseStore, worker: FakeWorker, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A live run held a phone at 0.60 back as a bad view and never asked at all."""
        before = seeded.count_items()
        poor = replace(near_miss_result(), reference=replace(reference(), mask_occupancy=0.2))
        coordinator = build(seeded, worker, "enforce")

        apply(coordinator, worker, poor)

        # No registry here, so the ASK is held as a wait; what matters is that it is an ASK.
        assert coordinator._reference_manager.reason(TRACK) == "policy_ask"
        assert "can't see this clearly" in capsys.readouterr().out
        assert seeded.count_items() == before

    def test_a_low_margin_tie_is_weighed_by_the_policy(
        self, seeded: DatabaseStore, worker: FakeWorker
    ) -> None:
        """Today this arrives as a bare 'pending'; its evidence must now survive."""
        tie = IdentificationResult(
            kind="resolve",
            track_id=TRACK,
            status="pending",
            failure_reason="low_margin",
            reference=good_reference(),
            candidates=(score(3, 0.80), score(4, 0.78)),
            decision_reason="low_margin",
        )
        coordinator = build(seeded, worker, "enforce")

        apply(coordinator, worker, tie)

        assert coordinator._reference_manager.reason(TRACK) == "policy_ask"

    def test_a_candidate_held_by_another_visible_track_is_not_offered(
        self, seeded: DatabaseStore, worker: FakeWorker, capsys: pytest.CaptureFixture[str]
    ) -> None:
        coordinator = build(seeded, worker, "enforce")
        coordinator.event_engine.associate_existing_item(99, 7)
        near = near_miss_result()
        near = replace(near, candidates=(score(7, 0.654), score(6, 0.60)))

        apply(coordinator, worker, near)

        out = capsys.readouterr().out
        assert "items=6" in out
        assert "items=7" not in out


class TestWhatIsNeverConsulted:
    def test_a_confident_match_never_reaches_the_policy(
        self, seeded: DatabaseStore, worker: FakeWorker, capsys: pytest.CaptureFixture[str]
    ) -> None:
        coordinator = build(seeded, worker, "enforce")
        match = IdentificationResult(
            kind="resolve",
            track_id=TRACK,
            status="existing",
            item_id=7,
            reference=good_reference(),
            candidates=(score(7, 0.86), score(1, 0.28)),
            decision_reason="accepted",
        )

        apply(coordinator, worker, match)

        assert coordinator.event_engine.item_id_for_track(TRACK) == 7
        assert "ASK" not in capsys.readouterr().out

    def test_an_unusable_crop_is_not_a_policy_matter(
        self, seeded: DatabaseStore, worker: FakeWorker, capsys: pytest.CaptureFixture[str]
    ) -> None:
        coordinator = build(seeded, worker, "enforce")
        unusable = IdentificationResult(
            kind="resolve", track_id=TRACK, status="pending", failure_reason="no_mask"
        )

        apply(coordinator, worker, unusable)

        assert coordinator._reference_manager.reason(TRACK) == "no_mask"
        assert "ASK" not in capsys.readouterr().out

    def test_an_empty_gallery_adds_the_first_item_as_before(
        self, store: DatabaseStore, worker: FakeWorker
    ) -> None:
        coordinator = build(store, worker, "enforce")
        first = IdentificationResult(
            kind="resolve", track_id=TRACK, status="new", reference=good_reference()
        )

        apply(coordinator, worker, first)

        assert store.count_items() == 1


class TestRegionContext:
    def test_the_region_an_item_was_last_seen_in_raises_confidence(
        self, seeded: DatabaseStore, worker: FakeWorker, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A look-alike appearing where that item sat is a genuine point of confusion."""
        region = seeded.create_region("desk", [[0, 0], [640, 0], [640, 480], [0, 480]])
        coordinator = build(seeded, worker, "observe")
        with seeded.session_factory() as session:
            session.get(Item, 7).current_region_id = region.id
            session.commit()

        apply(coordinator, worker, near_miss_result())

        assert "region_match" in capsys.readouterr().out

    def test_a_different_region_adds_nothing(
        self, seeded: DatabaseStore, worker: FakeWorker, capsys: pytest.CaptureFixture[str]
    ) -> None:
        desk = seeded.create_region("desk", [[0, 0], [640, 0], [640, 480], [0, 480]])
        shelf = seeded.create_region("shelf", [[700, 0], [800, 0], [800, 100], [700, 100]])
        coordinator = build(seeded, worker, "observe")
        with seeded.session_factory() as session:
            session.get(Item, 7).current_region_id = shelf.id
            session.commit()
        assert desk.id != shelf.id

        apply(coordinator, worker, near_miss_result())

        assert "region_match" not in capsys.readouterr().out


class TestStoreThumbnailLookup:
    """get_reference_image_path: which picture a question shows for a candidate."""

    def test_the_newest_stored_picture_is_returned(self, seeded: DatabaseStore) -> None:
        seeded.save_item_embedding(3, [1.0, 0.0], "m", object_image_path=Path("3/old.png"))
        seeded.save_item_embedding(3, [0.0, 1.0], "m", object_image_path=Path("3/new.png"))

        assert seeded.get_reference_image_path(3) == str(Path("3/new.png"))

    def test_views_stored_without_a_picture_are_skipped(self, seeded: DatabaseStore) -> None:
        """Everything saved before thumbnails existed has no picture; an older one still shows."""
        seeded.save_item_embedding(3, [1.0, 0.0], "m", object_image_path=Path("3/a.png"))
        seeded.save_item_embedding(3, [0.0, 1.0], "m")

        assert seeded.get_reference_image_path(3) == str(Path("3/a.png"))

    def test_an_item_with_no_pictures_gives_none(self, seeded: DatabaseStore) -> None:
        seeded.save_item_embedding(3, [1.0, 0.0], "m")

        assert seeded.get_reference_image_path(3) is None
        assert seeded.get_reference_image_path(404) is None

    def test_another_items_picture_is_never_returned(self, seeded: DatabaseStore) -> None:
        seeded.save_item_embedding(4, [1.0, 0.0], "m", object_image_path=Path("4/a.png"))

        assert seeded.get_reference_image_path(3) is None


class TestPolicyModeForDiagnostics:
    """The diagnostic entry point records ground truth and must not be changed by questions."""

    def test_the_shipped_default_is_to_ask(self) -> None:
        assert IdentityPolicyConfig.from_yaml(CONFIG_PATH).mode == "enforce"

    def test_a_mode_can_be_forced_for_a_run(self) -> None:
        from project_auto.app import _build_identity_policy
        from project_auto.memory.reid import ReidConfig

        root = Path(__file__).resolve().parents[1]
        reid = ReidConfig.from_yaml(root / "configs" / "reid.yaml")

        assert _build_identity_policy(root, reid).config.mode == "enforce"
        assert _build_identity_policy(root, reid, mode="off").config.mode == "off"

    def test_a_diagnostic_run_never_asks_questions(self) -> None:
        """The recorder is what marks a run as diagnostic; a normal run has none."""
        from project_auto.app import _policy_mode_override

        assert _policy_mode_override(None) is None
        assert _policy_mode_override(object()) == "off"  # any recorder

    def test_the_diagnostic_entry_point_passes_a_recorder_to_run_app(self) -> None:
        """If it ever stopped doing so, diagnostic runs would start asking questions."""
        import inspect

        from project_auto import reid_diagnostics_app

        source = inspect.getsource(reid_diagnostics_app)

        assert "diagnostics=" in source and "run_app(" in source

    def test_the_thresholds_still_come_from_the_matchers_config(self) -> None:
        from project_auto.app import _build_identity_policy
        from project_auto.memory.reid import ReidConfig

        root = Path(__file__).resolve().parents[1]
        reid = ReidConfig.from_yaml(root / "configs" / "reid.yaml")

        policy = _build_identity_policy(root, reid, mode="off")

        assert policy.acceptance_threshold == reid.acceptance_threshold
        assert policy.margin_threshold == reid.margin_threshold

    def test_an_unknown_mode_is_refused_loudly(self, tmp_path: Path) -> None:
        bad = tmp_path / "identity_policy.yaml"
        bad.write_text("mode: sometimes\n", encoding="utf-8")

        with pytest.raises(ValueError, match="mode must be one of"):
            IdentityPolicyConfig.from_yaml(bad)
