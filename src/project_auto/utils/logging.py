"""Unified, action-classified diagnostic output for the identification pipeline.

The old approach printed once per function, tagged by which file happened to run
(``[Coordinator]``, ``[SceneProcessor]``, ``[ReID]``), so one real action (e.g. one
track becoming a NEW item) produced three or four separate, differently-formatted
lines scattered across the console. That made the output noisy without making it
any easier to diagnose identity decisions.

log_action() instead groups every line by what action it represents, not which
module emitted it. Each category is logged exactly once, at the point that action
is finalized, with a fixed set of fields so lines are easy to scan and grep:

    EVENT     a lifecycle event was persisted (ADDED/MOVED/REMOVED/RETURNED)
    NEW       a track was resolved to a brand-new permanent item
    MATCH     a track was accepted as an existing item (ReID match)
    ASSOC     a track was bound to an already-present item (no event)
    AMBIG     a candidate cleared acceptance_threshold but failed margin_threshold
    REJECT    a candidate never reached matching (unusable mask/box/quality gate)
    DEFER     work was deferred/retried due to cooldown or backpressure
    CAPTURE   an additional reference was saved (or skipped) for a resolved item
    QUEUE     a job was submitted to the background worker
    ERROR     a background job raised an exception
    REGION    an event box resolved (or failed to resolve) to a named region
    SYSTEM    one-time startup/lifecycle noise (camera, model loading)

Every category has a fixed-width tag and a distinct ANSI colour plus an ASCII
sigil (not Unicode, so it never breaks on a legacy Windows console codepage) so
the category is visible even without colour support.
"""

from __future__ import annotations

import os
import sys
from typing import Any

if sys.platform == "win32":
    # Enables ANSI escape processing in legacy Windows consoles (cmd.exe); a
    # documented no-op call that has this side effect. Modern Windows Terminal
    # and VS Code's integrated terminal already support ANSI without it.
    os.system("")  # noqa: S605 - fixed, non-shell-interpreted argument


class Category:
    """Fixed set of action classifications; use these, not ad-hoc strings."""

    EVENT = "EVENT"
    NEW = "NEW"
    MATCH = "MATCH"
    ASSOCIATE = "ASSOC"
    AMBIGUOUS = "AMBIG"
    REJECT = "REJECT"
    DEFER = "DEFER"
    CAPTURE = "CAPTURE"
    QUEUE = "QUEUE"
    ERROR = "ERROR"
    REGION = "REGION"
    SYSTEM = "SYSTEM"


_RESET = "\033[0m"
_BOLD = "\033[1m"

# category -> (ASCII sigil, ANSI colour code)
_STYLE: dict[str, tuple[str, str]] = {
    Category.EVENT: ("##", "\033[95m"),  # magenta - persisted lifecycle truth
    Category.NEW: ("++", "\033[92m"),  # green - a new identity
    Category.MATCH: ("==", "\033[96m"),  # cyan - a confident match
    Category.ASSOCIATE: ("->", "\033[94m"),  # blue - silent claim, no event
    Category.AMBIGUOUS: ("??", "\033[93m"),  # yellow - could not disambiguate
    Category.REJECT: ("xx", "\033[91m"),  # red - never reached matching
    Category.DEFER: ("..", "\033[90m"),  # grey - cooldown/backpressure
    Category.CAPTURE: ("**", "\033[36m"),  # dim cyan - reference capture
    Category.QUEUE: ("::", "\033[90m"),  # grey - worker submission
    Category.ERROR: ("!!", "\033[97;41m"),  # white on red - worker exception
    Category.REGION: ("@@", "\033[35m"),  # dim magenta - spatial resolution
    Category.SYSTEM: ("--", "\033[90m"),  # grey - startup/one-time noise
}

_TAG_WIDTH = max(len(tag) for tag in _STYLE)
_use_color = sys.stdout.isatty()


def log_action(category: str, **fields: Any) -> None:
    """Print exactly one formatted diagnostic line for one classified action.

    fields are rendered as ``key=value`` in insertion order; floats are fixed to
    4 decimal places and None renders as ``-`` so columns stay easy to scan.
    """
    sigil, color = _STYLE.get(category, ("??", ""))
    tag = category.ljust(_TAG_WIDTH)
    body = " ".join(f"{key}={_format_value(value)}" for key, value in fields.items())

    if _use_color:
        line = f"{color}{_BOLD}{sigil} {tag}{_RESET}{color} {body}{_RESET}"
    else:
        line = f"{sigil} {tag} {body}"
    print(line)


def _format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
