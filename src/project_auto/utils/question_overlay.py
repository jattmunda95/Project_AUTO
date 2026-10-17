"""Show a question on the live window, and turn a key press into an answer.

Subfunctions:
- answer_for_key maps one key press to an Answer for a question: 1/2/3 pick that candidate,
  N is a new item, X is not an item, S skips. Any other key, or a number with no candidate
  behind it, is not an answer.
- ThumbnailCache loads each candidate's stored picture once and remembers it, including the
  fact that a picture is missing, so the frame loop never reads the disk twice for one file.
- QuestionOverlay draws the oldest open question as a band along the bottom of the frame:
  the new object's picture, the plain-English reasons, up to three candidate pictures with
  their key, a countdown to when the question closes, and the keys that answer it.

Why a band and not a dialog: the OpenCV window has no widgets or text box, and the camera
keeps running while someone decides, so the question sits over the bottom of the picture and
leaves the top, where the objects are, untouched.

Drawing is plain numpy and OpenCV work on the main thread: no model, no database, no network.
The only disk access is each stored picture's first load, once, small, and cached. Nothing in
here decides anything about identity; it shows a Question and reports an Answer, and applying
that answer belongs to the coordinator.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from project_auto.events.question_registry import Answer, Question

# Keys, as cv2.waitKey reports them after masking with 0xFF.
KEY_PICK = {ord("1"): 0, ord("2"): 1, ord("3"): 2}
KEY_NEW = ord("n")
KEY_NOT_AN_ITEM = ord("x")
KEY_SKIP = ord("s")

MAX_CANDIDATES = 3

# BGR.
_PANEL_DIM = 0.20
_TEXT = (235, 235, 235)
_MUTED = (165, 165, 165)
_ACCENT = (80, 190, 255)
_WARN = (70, 90, 240)
_TILE = (48, 48, 48)
_EDGE = (110, 110, 110)

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def answer_for_key(key: int, question: Question) -> Answer | None:
    """Return the answer a key press gives to this question, or None if it is not one.

    Letters are matched case-insensitively. A number only answers if that candidate exists.
    """
    key &= 0xFF
    if ord("A") <= key <= ord("Z"):
        key += ord("a") - ord("A")
    if key in KEY_PICK:
        index = KEY_PICK[key]
        if index < min(len(question.candidates), MAX_CANDIDATES):
            return Answer.pick(question.candidates[index].item_id)
        return None
    if key == KEY_NEW:
        return Answer.new_item()
    if key == KEY_NOT_AN_ITEM:
        return Answer.not_an_item()
    if key == KEY_SKIP:
        return Answer.skip()
    return None


def _load_image(path: Path) -> NDArray | None:
    """Read an image file as BGR, or None if it cannot be read.

    Decoded from bytes rather than read by path, because OpenCV's own path handling fails
    on non-ASCII Windows paths without saying so.
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except OSError:
        return None


@dataclass(slots=True)
class ThumbnailCache:
    """Load each stored picture at most once, and remember when one is missing.

    resolve turns a recorded path into one that can be opened (ReferenceImageStore.resolve).
    A picture that failed to load is remembered as missing so a bad file costs one read, not
    one per frame.
    """

    resolve: Callable[[Path], Path] = field(default=lambda path: path)
    load: Callable[[Path], NDArray | None] = field(default=_load_image)
    _images: dict[Path, NDArray | None] = field(default_factory=dict, init=False, repr=False)

    def get(self, path: Path | None) -> NDArray | None:
        """Return the picture at a recorded path, loading it the first time."""
        if path is None:
            return None
        if path not in self._images:
            self._images[path] = self.load(self.resolve(path))
        return self._images[path]

    def retain(self, paths: set[Path]) -> None:
        """Forget every picture not in paths, so a long session does not keep them all."""
        self._images = {path: image for path, image in self._images.items() if path in paths}


def to_bgr(image: object | None) -> NDArray | None:
    """Return an object's picture as a BGR array, or None if there is nothing to show.

    Accepts a PIL image (as PreparedReference.crop is, in RGB) or a BGR array.
    """
    if image is None:
        return None
    if isinstance(image, np.ndarray):
        return image if image.ndim == 3 and image.shape[2] == 3 else None
    try:
        array = np.asarray(image.convert("RGB"))  # type: ignore[attr-defined]
    except (AttributeError, ValueError, OSError):
        return None
    return np.ascontiguousarray(array[:, :, ::-1])


def _ascii(text: str) -> str:
    """Make text safe for OpenCV's font, which cannot draw anything outside ASCII."""
    return text.encode("ascii", "replace").decode("ascii")


def _text_width(text: str, scale: float, thickness: int = 1) -> int:
    return int(cv2.getTextSize(text, _FONT, scale, thickness)[0][0])


def _clip(word: str, max_width: int, scale: float) -> str:
    """Shorten one word that is wider than the space it has to fit in."""
    while len(word) > 1 and _text_width(word, scale) > max_width:
        word = word[:-1]
    return word


def fit_label(text: str, max_width: int, scale: float) -> str:
    """Shorten a phrase with an ellipsis so it is no wider than max_width pixels."""
    text = _ascii(text)
    if _text_width(text, scale) <= max_width:
        return text
    while len(text) > 1 and _text_width(text + "...", scale) > max_width:
        text = text[:-1]
    return text.rstrip() + "..."


def wrap_text(text: str, max_width: int, scale: float) -> list[str]:
    """Break text into lines no wider than max_width pixels at this font scale."""
    lines: list[str] = []
    current = ""
    for word in _ascii(text).split():
        word = _clip(word, max_width, scale)
        trial = f"{current} {word}".strip()
        if current and _text_width(trial, scale) > max_width:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    return lines


def _fit(image: NDArray | None, width: int, height: int) -> NDArray:
    """Return a width x height tile with the image centred in it, aspect preserved."""
    tile = np.full((height, width, 3), _TILE, dtype=np.uint8)
    if image is None or image.size == 0:
        cv2.putText(tile, "no picture", (8, height // 2), _FONT, 0.4, _MUTED, 1, cv2.LINE_AA)
        return tile
    source_h, source_w = image.shape[:2]
    scale = min(width / source_w, height / source_h)
    new_w, new_h = max(1, int(source_w * scale)), max(1, int(source_h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x, y = (width - new_w) // 2, (height - new_h) // 2
    tile[y : y + new_h, x : x + new_w] = resized
    return tile


def _put(frame: NDArray, image: NDArray, x: int, y: int) -> None:
    """Copy an image onto the frame at x, y, clipped to the frame's edges."""
    frame_h, frame_w = frame.shape[:2]
    h, w = image.shape[:2]
    x2, y2 = min(frame_w, x + w), min(frame_h, y + h)
    if x < 0 or y < 0 or x2 <= x or y2 <= y:
        return
    frame[y:y2, x:x2] = image[: y2 - y, : x2 - x]


def _text(frame: NDArray, text: str, x: int, y: int, scale: float, color, thickness: int = 1):
    cv2.putText(frame, _ascii(text), (x, y), _FONT, scale, color, thickness, cv2.LINE_AA)


@dataclass(slots=True)
class QuestionOverlay:
    """Draw the oldest open question over the bottom of the live frame.

    thumbnails supplies stored candidate pictures; crops are converted once per question
    and dropped when the question is gone. draw returns the frame unchanged (the same
    object, not a copy) when nothing is being asked, so an idle overlay costs nothing.
    """

    thumbnails: ThumbnailCache = field(default_factory=ThumbnailCache)
    _crops: dict[int, NDArray | None] = field(default_factory=dict, init=False, repr=False)

    def answer_for_key(self, key: int, question: Question) -> Answer | None:
        """Return the answer a key press gives to this question; see answer_for_key."""
        return answer_for_key(key, question)

    def draw(self, frame: NDArray, questions: Sequence[Question], now: float) -> NDArray:
        """Return the frame with the oldest open question drawn over its bottom edge."""
        if not questions:
            self._crops.clear()
            return frame

        question = questions[0]
        self._forget_stale(questions)
        output = frame.copy()
        frame_h, frame_w = output.shape[:2]
        panel_h = min(frame_h, max(200, int(frame_h * 0.36)))
        top = frame_h - panel_h
        margin = 14
        # Text shrinks on a small window so labels and reasons still fit; it never grows.
        k = max(0.75, min(1.0, frame_w / 1280))

        band = output[top:frame_h]
        output[top:frame_h] = cv2.convertScaleAbs(band, alpha=_PANEL_DIM)
        self._draw_countdown(output, question, now, top, frame_w)

        content_top = top + 30
        content_bottom = frame_h - 38
        box = max(24, content_bottom - content_top)

        # The new object.
        _text(output, "THIS OBJECT", margin, top + 22, 0.45 * k, _MUTED)
        crop = self._crops.setdefault(question.id, to_bgr(question.crop))
        _put(output, _fit(crop, box, box), margin, content_top)
        cv2.rectangle(output, (margin, content_top), (margin + box, content_top + box), _EDGE, 1)

        # Why it is being asked.
        text_x = margin * 2 + box
        text_w = max(160, int(frame_w * 0.30))
        _text(output, "Is this a known item?", text_x, top + 22, 0.6 * k, _ACCENT, 2)
        line_h = max(14, int(18 * k))
        y = content_top + 12
        for line in self._reason_lines(question.reasons, text_w - 10, 0.45 * k, box // line_h):
            _text(output, line, text_x, y, 0.45 * k, _TEXT)
            y += line_h

        # The candidates.
        self._draw_cards(output, question, text_x + text_w + margin, content_top, box, margin, k)

        self._draw_help(output, question, len(questions) - 1, now, frame_h, frame_w, margin, k)
        return output

    @staticmethod
    def _reason_lines(
        reasons: Sequence[str], max_width: int, scale: float, max_lines: int
    ) -> list[str]:
        """Lay the reasons out as bullets, whole or not at all.

        A reason that does not fit in the room left is dropped rather than cut mid-sentence,
        and a closing "..." line says that some were left out. The first reason is always
        kept, clipped if need be, because a question with no reason at all explains nothing.
        """
        blocks = []
        for reason in reasons:
            wrapped = wrap_text(reason, max_width, scale)
            blocks.append([("- " if i == 0 else "  ") + line for i, line in enumerate(wrapped)])

        everything = [line for block in blocks for line in block]
        if len(everything) <= max_lines:
            return everything

        # Not all of it fits: keep whole reasons, leaving one line for the ellipsis.
        lines: list[str] = []
        for block in blocks:
            if len(lines) + len(block) <= max_lines - 1:
                lines.extend(block)
            elif not lines:
                lines.extend(block[: max(1, max_lines - 1)])
                break
            else:
                break
        lines.append("  ...")
        return lines[:max_lines]

    def _forget_stale(self, questions: Sequence[Question]) -> None:
        """Drop converted crops and pictures for questions no longer open."""
        live = {question.id for question in questions}
        self._crops = {key: crop for key, crop in self._crops.items() if key in live}
        paths = {
            path
            for question in questions
            for path in question.candidate_images.values()
            if path is not None
        }
        self.thumbnails.retain(paths)

    def _draw_countdown(
        self, output: NDArray, question: Question, now: float, top: int, frame_w: int
    ) -> None:
        """Draw how much of the question's time is left as a bar along the panel's top."""
        span = max(1e-6, question.expires_at - question.created_at)
        left = min(1.0, max(0.0, (question.expires_at - now) / span))
        color = _WARN if left < 0.2 else _ACCENT
        cv2.rectangle(output, (0, top), (int(frame_w * left), top + 3), color, -1)

    def _draw_cards(
        self, output: NDArray, question: Question, x: int, top: int, box: int, margin: int, k: float
    ) -> None:
        """Draw each candidate's picture with the key that picks it."""
        candidates = question.candidates[:MAX_CANDIDATES]
        available = output.shape[1] - margin - x
        if not candidates or available < 60:
            return
        card_w = max(48, min(220, (available - margin * (len(candidates) - 1)) // len(candidates)))
        picture_h = max(16, box - 24)
        for index, candidate in enumerate(candidates):
            left = x + index * (card_w + margin)
            image = self.thumbnails.get(question.candidate_images.get(candidate.item_id))
            _put(output, _fit(image, card_w, picture_h), left, top)
            cv2.rectangle(output, (left, top), (left + card_w, top + picture_h), _EDGE, 1)
            label = fit_label(f"[{index + 1}] {candidate.name}", card_w, 0.5 * k)
            _text(output, label, left, top + picture_h + 18, 0.5 * k, _TEXT, 1)

    def _draw_help(
        self,
        output: NDArray,
        question: Question,
        waiting: int,
        now: float,
        frame_h: int,
        frame_w: int,
        margin: int,
        k: float,
    ) -> None:
        """Draw the keys along the bottom, and how long is left."""
        keys = "N new item   X not an item   S skip"
        if question.candidates:
            keys = f"1-{min(len(question.candidates), MAX_CANDIDATES)} pick   " + keys
        _text(output, keys, margin, frame_h - 14, 0.5 * k, _ACCENT)
        seconds = max(0, round(question.expires_at - now))
        status = f"closes in {seconds}s"
        if waiting > 0:
            status = f"+{waiting} more waiting   " + status
        scale = 0.45 * k
        x = max(margin, frame_w - margin - _text_width(status, scale))
        _text(output, status, x, frame_h - 14, scale, _MUTED)
