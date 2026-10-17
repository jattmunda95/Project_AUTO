"""Question-overlay tests: what is drawn, where, and which key gives which answer.

Rendering is checked on real pixel arrays rather than mocks: the top of the frame must be
left alone, the band must actually change the bottom, and nothing may crash on a missing
picture, a tiny window or a hostile string.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from project_auto.events.identity_policy import PolicyCandidate
from project_auto.events.question_registry import Answer, AnswerKind, Question
from project_auto.utils.question_overlay import (
    QuestionOverlay,
    ThumbnailCache,
    answer_for_key,
    fit_label,
    to_bgr,
    wrap_text,
)

WIDTH, HEIGHT = 1280, 720


def candidate(item_id: int, label: str | None = None) -> PolicyCandidate:
    return PolicyCandidate(
        item_id=item_id,
        final_score=0.6,
        dino_score=0.6,
        color_score=0.6,
        aspect_score=0.5,
        n_references=3,
        item_label=label,
    )


def make_question(
    candidates: int = 2,
    reasons: tuple[str, ...] = (
        "This looks like item 7, but not clearly enough to be sure.",
        "Only a few views of item 7 are saved, so a close match is harder to confirm.",
    ),
    crop: object | None = None,
    images: dict[int, Path | None] | None = None,
    created_at: float = 0.0,
    expires_at: float = 60.0,
) -> Question:
    return Question(
        id=1,
        track_id=3,
        created_at=created_at,
        expires_at=expires_at,
        candidates=tuple(candidate(7 + index) for index in range(candidates)),
        reasons=reasons,
        crop=crop,
        candidate_images=images or {},
    )


def blank(width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    """A mid-grey 'camera frame' so that darkening and drawing are both visible."""
    return np.full((height, width, 3), 140, dtype=np.uint8)


@pytest.fixture
def overlay() -> QuestionOverlay:
    return QuestionOverlay()


class TestKeys:
    @pytest.mark.parametrize(
        ("key", "expected"),
        [
            (ord("1"), Answer.pick(7)),
            (ord("2"), Answer.pick(8)),
            (ord("n"), Answer.new_item()),
            (ord("N"), Answer.new_item()),
            (ord("x"), Answer.not_an_item()),
            (ord("s"), Answer.skip()),
            (ord("S"), Answer.skip()),
        ],
    )
    def test_each_key_gives_its_answer(self, key: int, expected: Answer) -> None:
        assert answer_for_key(key, make_question(candidates=2)) == expected

    def test_a_number_with_no_candidate_behind_it_is_not_an_answer(self) -> None:
        question = make_question(candidates=2)

        assert answer_for_key(ord("3"), question) is None

    def test_a_question_with_one_candidate_only_picks_it_with_one(self) -> None:
        question = make_question(candidates=1)

        assert answer_for_key(ord("1"), question) == Answer.pick(7)
        assert answer_for_key(ord("2"), question) is None

    @pytest.mark.parametrize("key", [ord("q"), ord("w"), ord("r"), ord("4"), ord(" "), 255, -1])
    def test_every_other_key_is_not_an_answer(self, key: int) -> None:
        """q, w and r belong to the app (quit and the region queries); they must pass through."""
        assert answer_for_key(key, make_question()) is None

    def test_a_raw_waitkey_value_with_high_bits_is_still_understood(self) -> None:
        """cv2.waitKey can return more than eight bits on some platforms."""
        assert answer_for_key(0x100000 | ord("s"), make_question()) == Answer.skip()

    def test_a_pick_is_always_for_an_item_that_was_offered(self) -> None:
        question = make_question(candidates=3)

        for key in (ord("1"), ord("2"), ord("3")):
            answer = answer_for_key(key, question)
            assert answer is not None and answer.kind is AnswerKind.PICK
            assert answer.item_id in question.offered_item_ids


class TestDrawing:
    def test_nothing_open_returns_the_same_frame_untouched(self, overlay: QuestionOverlay) -> None:
        frame = blank()

        assert overlay.draw(frame, [], now=0.0) is frame

    def test_drawing_never_alters_the_input_frame(self, overlay: QuestionOverlay) -> None:
        frame = blank()
        before = frame.copy()

        overlay.draw(frame, [make_question()], now=1.0)

        assert np.array_equal(frame, before)

    def test_the_output_keeps_the_frame_size_and_type(self, overlay: QuestionOverlay) -> None:
        out = overlay.draw(blank(), [make_question()], now=1.0)

        assert out.shape == (HEIGHT, WIDTH, 3)
        assert out.dtype == np.uint8

    def test_the_top_of_the_picture_is_left_alone(self, overlay: QuestionOverlay) -> None:
        """The objects being asked about must stay visible above the band."""
        out = overlay.draw(blank(), [make_question()], now=1.0)

        assert np.array_equal(out[: HEIGHT // 2], blank()[: HEIGHT // 2])

    def test_the_bottom_band_is_darkened_and_drawn_on(self, overlay: QuestionOverlay) -> None:
        out = overlay.draw(blank(), [make_question()], now=1.0)
        band = out[-150:]

        assert band.mean() < 140 * 0.5, "the band must dim the picture behind it"
        assert len(np.unique(band.reshape(-1, 3), axis=0)) > 10, "something must be drawn on it"

    def test_the_new_objects_picture_is_shown(self, overlay: QuestionOverlay) -> None:
        crop = Image.new("RGB", (80, 80), (255, 0, 0))  # red in RGB
        with_crop = overlay.draw(blank(), [make_question(crop=crop)], now=1.0)
        without = QuestionOverlay().draw(blank(), [make_question(crop=None)], now=1.0)

        red_in_bgr = np.all(with_crop[-200:] == (0, 0, 255), axis=-1).sum()
        assert red_in_bgr > 1000, "the crop should appear, with RGB converted to BGR"
        assert np.all(without[-200:] == (0, 0, 255), axis=-1).sum() == 0

    def test_candidate_pictures_are_shown_beside_their_keys(self, tmp_path: Path) -> None:
        green = np.zeros((60, 60, 3), dtype=np.uint8)
        green[:] = (0, 255, 0)
        path = tmp_path / "7.png"
        cv2.imwrite(str(path), green)
        overlay = QuestionOverlay(thumbnails=ThumbnailCache())

        out = overlay.draw(blank(), [make_question(images={7: path})], now=1.0)

        assert np.all(out[-200:] == (0, 255, 0), axis=-1).sum() > 500

    def test_a_missing_picture_is_a_placeholder_not_a_crash(self, tmp_path: Path) -> None:
        overlay = QuestionOverlay()
        question = make_question(images={7: tmp_path / "gone.png", 8: None})

        out = overlay.draw(blank(), [question], now=1.0)

        assert out.shape == (HEIGHT, WIDTH, 3)

    def test_only_the_oldest_question_is_drawn(self, overlay: QuestionOverlay) -> None:
        first, second = make_question(), make_question()
        second.id = 2
        second.reasons = ("ZZZZZZZZZZ",)

        alone = overlay.draw(blank(), [first], now=1.0)
        both = QuestionOverlay().draw(blank(), [first, second], now=1.0)

        # The only difference is the "+1 more waiting" note; the first question is intact.
        assert not np.array_equal(alone, both)
        assert np.array_equal(alone[: HEIGHT - 60], both[: HEIGHT - 60])

    def test_a_waiting_count_is_shown_when_more_are_queued(self) -> None:
        one = QuestionOverlay().draw(blank(), [make_question()], now=1.0)
        extra = make_question()
        extra.id = 2
        three = QuestionOverlay().draw(blank(), [make_question(), extra, extra], now=1.0)

        assert not np.array_equal(one[-30:], three[-30:])

    def test_the_countdown_bar_shrinks_as_time_runs_out(self) -> None:
        question = make_question(created_at=0.0, expires_at=60.0)

        early = QuestionOverlay().draw(blank(), [question], now=6.0)
        late = QuestionOverlay().draw(blank(), [question], now=54.0)
        bar_row = HEIGHT - max(200, int(HEIGHT * 0.36))  # the panel's first row

        def bar_width(image: np.ndarray) -> int:
            row = image[bar_row + 1]
            return int((row.sum(axis=-1) > 300).sum())

        assert bar_width(early) > bar_width(late) > 0

    def test_the_countdown_turns_red_when_almost_out_of_time(self) -> None:
        question = make_question(created_at=0.0, expires_at=60.0)
        bar_row = HEIGHT - max(200, int(HEIGHT * 0.36)) + 1

        early = QuestionOverlay().draw(blank(), [question], now=5.0)[bar_row, 5]
        late = QuestionOverlay().draw(blank(), [question], now=55.0)[bar_row, 5]

        assert not np.array_equal(early, late)

    def test_a_question_past_its_time_does_not_crash(self, overlay: QuestionOverlay) -> None:
        question = make_question(expires_at=10.0)

        out = overlay.draw(blank(), [question], now=500.0)

        assert out.shape == (HEIGHT, WIDTH, 3)


class TestRobustness:
    @pytest.mark.parametrize("size", [(640, 480), (320, 240), (1920, 1080), (200, 150)])
    def test_it_draws_at_other_window_sizes_without_error(self, size: tuple[int, int]) -> None:
        out = QuestionOverlay().draw(blank(*size), [make_question(candidates=3)], now=1.0)

        assert out.shape == (size[1], size[0], 3)

    def test_a_question_with_no_candidates_still_draws(self, overlay: QuestionOverlay) -> None:
        out = overlay.draw(blank(), [make_question(candidates=0)], now=1.0)

        assert out.shape == (HEIGHT, WIDTH, 3)

    def test_a_very_long_reason_is_clipped_not_overflowed(self, overlay: QuestionOverlay) -> None:
        question = make_question(reasons=("word " * 400, "A" * 500))

        out = overlay.draw(blank(), [question], now=1.0)

        assert out.shape == (HEIGHT, WIDTH, 3)

    def test_non_ascii_text_does_not_break_the_font(self, overlay: QuestionOverlay) -> None:
        question = Question(
            id=1,
            track_id=1,
            created_at=0.0,
            expires_at=60.0,
            candidates=(candidate(7, label="the bläck phone ☃"),),
            reasons=("It looks like the bläck phone ☃.",),
        )

        out = overlay.draw(blank(), [question], now=1.0)

        assert out.shape == (HEIGHT, WIDTH, 3)

    def test_a_grayscale_or_malformed_crop_is_ignored_not_fatal(
        self, overlay: QuestionOverlay
    ) -> None:
        for bad in (np.zeros((10, 10), dtype=np.uint8), "not an image", 42, object()):
            out = overlay.draw(blank(), [make_question(crop=bad)], now=1.0)
            assert out.shape == (HEIGHT, WIDTH, 3)


class TestCrops:
    def test_a_pil_crop_is_converted_from_rgb_to_bgr(self) -> None:
        converted = to_bgr(Image.new("RGB", (4, 4), (10, 20, 30)))

        assert converted is not None
        assert tuple(converted[0, 0]) == (30, 20, 10)

    def test_a_bgr_array_is_used_as_it_is(self) -> None:
        array = np.zeros((4, 4, 3), dtype=np.uint8)

        assert to_bgr(array) is array

    def test_nothing_to_show_is_none(self) -> None:
        assert to_bgr(None) is None
        assert to_bgr(np.zeros((4, 4), dtype=np.uint8)) is None

    def test_a_crop_is_converted_once_per_question_not_every_frame(self) -> None:
        overlay = QuestionOverlay()
        question = make_question(crop=Image.new("RGB", (30, 30), (1, 2, 3)))

        overlay.draw(blank(), [question], now=1.0)
        first = overlay._crops[question.id]
        overlay.draw(blank(), [question], now=2.0)

        assert overlay._crops[question.id] is first

    def test_a_closed_questions_crop_is_dropped(self) -> None:
        overlay = QuestionOverlay()
        question = make_question(crop=Image.new("RGB", (30, 30)))
        overlay.draw(blank(), [question], now=1.0)

        overlay.draw(blank(), [], now=2.0)

        assert not overlay._crops


class TestThumbnailCache:
    def test_a_picture_is_read_once_however_often_it_is_drawn(self) -> None:
        reads: list[Path] = []

        def load(path: Path):
            reads.append(path)
            return np.zeros((8, 8, 3), dtype=np.uint8)

        cache = ThumbnailCache(load=load)
        for _ in range(100):
            cache.get(Path("7/a.png"))

        assert reads == [Path("7/a.png")]

    def test_a_missing_picture_is_not_retried_every_frame(self) -> None:
        reads: list[Path] = []

        def load(path: Path):
            reads.append(path)

        cache = ThumbnailCache(load=load)
        for _ in range(100):
            assert cache.get(Path("gone.png")) is None

        assert len(reads) == 1

    def test_no_path_means_no_read(self) -> None:
        cache = ThumbnailCache(load=lambda path: pytest.fail("must not read"))

        assert cache.get(None) is None

    def test_the_recorded_path_is_resolved_before_it_is_opened(self) -> None:
        opened: list[Path] = []
        cache = ThumbnailCache(
            resolve=lambda path: Path("/root") / path,
            load=lambda path: opened.append(path),
        )

        cache.get(Path("7/a.png"))

        assert opened == [Path("/root/7/a.png")]

    def test_pictures_no_longer_needed_are_released(self) -> None:
        cache = ThumbnailCache(load=lambda path: np.zeros((2, 2, 3), dtype=np.uint8))
        cache.get(Path("a.png"))
        cache.get(Path("b.png"))

        cache.retain({Path("b.png")})

        assert set(cache._images) == {Path("b.png")}

    def test_the_overlay_releases_pictures_of_closed_questions(self, tmp_path: Path) -> None:
        path = tmp_path / "a.png"
        cv2.imwrite(str(path), np.zeros((8, 8, 3), dtype=np.uint8))
        overlay = QuestionOverlay()
        overlay.draw(blank(), [make_question(images={7: path})], now=1.0)
        assert overlay.thumbnails._images

        overlay.draw(blank(), [make_question(images={})], now=2.0)

        assert not overlay.thumbnails._images

    def test_a_real_file_round_trips_through_the_default_loader(self, tmp_path: Path) -> None:
        path = tmp_path / "unicode é.png"
        ok, encoded = cv2.imencode(".png", np.full((6, 6, 3), 200, dtype=np.uint8))
        assert ok
        encoded.tofile(str(path))

        image = ThumbnailCache().get(path)

        assert image is not None and image.shape == (6, 6, 3)

    def test_an_empty_or_corrupt_file_is_treated_as_missing(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.png"
        empty.write_bytes(b"")
        corrupt = tmp_path / "corrupt.png"
        corrupt.write_bytes(b"this is not a png")
        cache = ThumbnailCache()

        assert cache.get(empty) is None
        assert cache.get(corrupt) is None


class TestWrapping:
    def test_text_is_wrapped_to_the_width(self) -> None:
        lines = wrap_text("one two three four five six seven eight nine ten", 120, 0.45)

        assert len(lines) > 1
        for line in lines:
            assert cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0] <= 120

    def test_wrapping_loses_no_words(self) -> None:
        text = "This looks like item 7 but not clearly enough to be sure"

        assert " ".join(wrap_text(text, 140, 0.45)).split() == text.split()

    def test_one_word_wider_than_the_space_is_shortened_to_fit(self) -> None:
        (line,) = wrap_text("A" * 200, 100, 0.45)

        assert cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0] <= 100

    def test_empty_text_gives_no_lines(self) -> None:
        assert wrap_text("", 100, 0.45) == []


class TestLabelsFitTheirCards:
    """Found by looking at a 640-wide render: three labels ran into one another."""

    def test_a_label_wider_than_its_card_is_shortened_with_an_ellipsis(self) -> None:
        clipped = fit_label("[1] the black phone with the cracked case", 120, 0.4)

        assert clipped.endswith("...")
        assert cv2.getTextSize(clipped, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0] <= 120

    def test_a_label_that_fits_is_left_alone(self) -> None:
        assert fit_label("[2] item 7", 300, 0.5) == "[2] item 7"

    def test_the_key_stays_visible_when_the_name_is_cut(self) -> None:
        """The number is what the person presses, so it must survive any clipping."""
        assert fit_label("[3] an extremely long item name indeed", 90, 0.5).startswith("[3]")


class TestReasonsAreWholeOrOmitted:
    """Found by looking at a 640-wide render: a reason was cut off mid-sentence."""

    reasons = (
        "This looks like item 7, but not clearly enough to be sure.",
        "Item 7 and item 6 look about equally alike.",
        "Its overall appearance points to item 7, but its colour points to item 6.",
        "Only a few views of item 7 are saved, so a close match is harder to confirm.",
    )

    def lines(self, max_lines: int, width: int = 180) -> list[str]:
        return QuestionOverlay._reason_lines(self.reasons, width, 0.45, max_lines)

    def test_everything_is_shown_when_there_is_room(self) -> None:
        lines = self.lines(max_lines=40)

        assert not any("..." in line for line in lines)
        for reason in self.reasons:
            assert " ".join(reason.split()[:3]) in " ".join(lines)

    def test_a_reason_that_does_not_fit_is_dropped_whole_not_cut(self) -> None:
        """For every room size, a reason is either fully present or fully absent."""
        for room in range(3, 14):
            text = " ".join(line.strip().removeprefix("- ") for line in self.lines(room))
            for reason in self.reasons:
                words = reason.split()
                started = " ".join(words[:3]) in text
                finished = " ".join(words[-3:]) in text
                assert started == finished, f"room={room}: cut mid-sentence: {reason!r}"

    def test_an_ellipsis_says_when_reasons_were_left_out(self) -> None:
        lines = self.lines(max_lines=5)

        assert lines[-1].strip() == "..."

    def test_the_result_never_exceeds_the_room(self) -> None:
        for room in range(1, 12):
            assert len(self.lines(max_lines=room)) <= room

    def test_the_first_reason_is_always_kept(self) -> None:
        """A question with no reason at all explains nothing."""
        for room in range(2, 12):
            assert self.lines(max_lines=room)[0].startswith("- This looks like")

    def test_no_reasons_gives_no_lines(self) -> None:
        assert QuestionOverlay._reason_lines((), 180, 0.45, 8) == []


class TestFromAskToAnswer:
    """The whole chain without a camera: a question is asked, drawn, answered by a key."""

    @pytest.fixture
    def setup(self):
        from project_auto.app import _answer_open_question
        from project_auto.events.identity_policy import (
            IdentityOutcome,
            IdentityPolicy,
            IdentityPolicyConfig,
            ViewQuality,
        )
        from project_auto.events.question_registry import QuestionRegistry

        config = IdentityPolicyConfig.from_yaml(
            Path(__file__).resolve().parents[1] / "configs" / "identity_policy.yaml"
        )
        policy = IdentityPolicy(config=config, acceptance_threshold=0.68, margin_threshold=0.06)
        registry = QuestionRegistry(config)
        decision = policy.decide(
            (candidate(7, "the black phone"), candidate(6)), ViewQuality(0.7, 0.95)
        )
        # candidate() above scores 0.6 each, a close race at a real score, so this is an ASK.
        assert decision.outcome is IdentityOutcome.ASK
        question = registry.ask(3, decision, now=0.0, crop=Image.new("RGB", (40, 40)))
        assert question is not None
        return registry, QuestionOverlay(), question, _answer_open_question

    def test_the_question_is_drawn_while_it_is_open(self, setup) -> None:
        registry, overlay, _, _ = setup

        drawn = overlay.draw(blank(), registry.open_questions(), now=1.0)

        assert not np.array_equal(drawn, blank())

    def test_a_key_press_answers_and_closes_it(self, setup) -> None:
        registry, overlay, question, answer_open = setup

        answer = answer_open(registry.answer, overlay, registry.open_questions(), ord("1"), 5.0)

        assert answer == Answer.pick(question.candidates[0].item_id)
        assert registry.open_count() == 0
        assert question.answer == answer

    def test_once_answered_the_overlay_goes_away(self, setup) -> None:
        registry, overlay, _, answer_open = setup
        answer_open(registry.answer, overlay, registry.open_questions(), ord("n"), 5.0)

        frame = blank()

        assert overlay.draw(frame, registry.open_questions(), now=6.0) is frame

    def test_a_key_that_is_not_an_answer_leaves_the_question_open(self, setup) -> None:
        registry, overlay, _, answer_open = setup

        for key in (ord("q"), ord("w"), ord("r"), ord("9"), 255):
            assert (
                answer_open(registry.answer, overlay, registry.open_questions(), key, 5.0) is None
            )

        assert registry.open_count() == 1

    def test_keys_do_nothing_when_nothing_is_asked(self, setup) -> None:
        registry, overlay, _, answer_open = setup
        registry.forget_track(3)

        assert (
            answer_open(registry.answer, overlay, registry.open_questions(), ord("s"), 5.0) is None
        )

    def test_a_late_key_press_is_ignored_and_closes_it_as_expired(self, setup) -> None:
        from project_auto.events.question_registry import QuestionState

        registry, overlay, question, answer_open = setup
        late = question.expires_at + 1

        assert (
            answer_open(registry.answer, overlay, registry.open_questions(), ord("1"), late) is None
        )
        assert question.state is QuestionState.EXPIRED

    def test_only_the_oldest_question_is_answered_by_a_key(self, setup) -> None:
        from project_auto.events.identity_policy import IdentityOutcome

        registry, overlay, first, answer_open = setup
        from project_auto.events.identity_policy import (
            IdentityPolicy,
            ViewQuality,
        )

        policy = IdentityPolicy(
            config=registry.config, acceptance_threshold=0.68, margin_threshold=0.06
        )
        decision = policy.decide((candidate(3), candidate(4)), ViewQuality(0.7, 0.95))
        assert decision.outcome is IdentityOutcome.ASK
        second = registry.ask(9, decision, now=1.0)
        assert second is not None

        answer_open(registry.answer, overlay, registry.open_questions(), ord("s"), 5.0)

        assert not first.is_open
        assert second.is_open
