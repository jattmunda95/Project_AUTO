"""Per-run ReID diagnostic tables for building a labelled ground-truth dataset.

Subfunctions:
- ReidDiagnostics creates one folder per application run (logs/reid_runs/<run_id>/)
  and records run provenance: git commit (with a dirty-tree marker) and a hash of
  the configuration files that affect matching.
- new_query_id() issues one stable ID per identity-resolve job; it is the join key
  for every table below and for the hand-written ground_truth.csv.
- record_query() writes one queries.csv row per resolve job, one candidates.csv row
  per scored gallery item, and saves the raw and masked crops used for labelling.
- record_outcome() writes one outcomes.csv row once the main thread has applied the
  result (ADDED/RETURNED/ASSOC/DEFER/...).

Only identity-resolve jobs are recorded; reference-capture jobs never reach the
matcher and stay in the CAPTURE/REJECT console log. This module imports nothing
from the pipeline: callers pass plain values, so it can be removed or disabled
(pass None instead of an instance) without changing any decision. A failure to
write diagnostics is reported and swallowed; it never fails an identity job.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import logging
import subprocess
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from project_auto.utils.logging import Category, log_action

logger = logging.getLogger(__name__)

QUERY_COLUMNS = (
    "query_id",
    "run_id",
    "timestamp",
    "frame_index",
    "track_id",
    "box_x1",
    "box_y1",
    "box_x2",
    "box_y2",
    "det_confidence",
    "box_area",
    "sam_score",
    "mask_occupancy",
    "gallery_size",
    "decision",
    "decision_reason",
    "matched_item_id",
    "acceptance_threshold",
    "margin_threshold",
    "weights",
    "top_k",
    "prototype_shortlist_size",
    "config_hash",
    "git_commit",
    "masked_crop_path",
    "raw_crop_path",
)

CANDIDATE_COLUMNS = (
    "query_id",
    "item_id",
    "rank",
    "final_score",
    "dino_score",
    "color_score",
    "aspect_score",
    "n_references",
    "used_fallback",
)

OUTCOME_COLUMNS = ("query_id", "timestamp", "item_id", "event")


@dataclass(frozen=True, slots=True)
class QueryRecord:
    """Per-job evidence for one identity-resolve attempt."""

    query_id: str
    frame_index: int | None
    track_id: int
    box: tuple[int, int, int, int]
    det_confidence: float | None
    sam_score: float | None
    mask_occupancy: float | None
    gallery_size: int
    decision: str  # MATCH, AMBIG, NEW or DEFER
    decision_reason: str
    matched_item_id: int | None


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    """One gallery item's component scores for one query."""

    item_id: int
    final_score: float
    dino_score: float
    color_score: float | None
    aspect_score: float | None
    n_references: int


class ReidDiagnostics:
    """Append ReID query/candidate/outcome rows and crops for one application run."""

    def __init__(
        self,
        root_dir: Path,
        run_settings: Mapping[str, object],
        config_paths: Iterable[Path] = (),
        project_root: Path | None = None,
    ) -> None:
        """run_settings supplies the constant per-run columns: acceptance_threshold,
        margin_threshold, weights, top_k and prototype_shortlist_size."""
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_dir = Path(root_dir) / self.run_id
        self.crops_dir = self.run_dir / "crops"
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        self.queries_path = self.run_dir / "queries.csv"
        self.candidates_path = self.run_dir / "candidates.csv"
        self.outcomes_path = self.run_dir / "outcomes.csv"

        self._run_columns = {
            "run_id": self.run_id,
            "acceptance_threshold": run_settings.get("acceptance_threshold"),
            "margin_threshold": run_settings.get("margin_threshold"),
            "weights": run_settings.get("weights"),
            "top_k": run_settings.get("top_k"),
            "prototype_shortlist_size": run_settings.get("prototype_shortlist_size"),
            "config_hash": _hash_files(config_paths),
            "git_commit": _git_commit(project_root),
        }
        self._counter = itertools.count(1)
        self._lock = threading.Lock()
        log_action(Category.SYSTEM, event="reid_diagnostics", run_dir=self.run_dir)

    def new_query_id(self) -> str:
        """Return a run-unique ID for one resolve job (safe from any thread)."""
        with self._lock:
            return f"{self.run_id}-q{next(self._counter):05d}"

    def record_query(
        self,
        query: QueryRecord,
        candidates: Iterable[CandidateRecord],
        raw_crop_bgr: NDArray[np.uint8] | None = None,
        masked_crop: Image.Image | None = None,
    ) -> None:
        """Save the crops and append the query row plus its ranked candidate rows."""
        try:
            raw_path = self._save_raw_crop(query.query_id, raw_crop_bgr)
            masked_path = self._save_masked_crop(query.query_id, masked_crop)
            ranked = sorted(candidates, key=lambda candidate: candidate.final_score, reverse=True)
            x1, y1, x2, y2 = query.box
            query_row = {
                **self._run_columns,
                "query_id": query.query_id,
                "timestamp": _utc_now(),
                "frame_index": query.frame_index,
                "track_id": query.track_id,
                "box_x1": x1,
                "box_y1": y1,
                "box_x2": x2,
                "box_y2": y2,
                "det_confidence": query.det_confidence,
                "box_area": max(0, x2 - x1) * max(0, y2 - y1),
                "sam_score": query.sam_score,
                "mask_occupancy": query.mask_occupancy,
                "gallery_size": query.gallery_size,
                "decision": query.decision,
                "decision_reason": query.decision_reason,
                "matched_item_id": query.matched_item_id,
                "masked_crop_path": masked_path,
                "raw_crop_path": raw_path,
            }
            candidate_rows = [
                {
                    "query_id": query.query_id,
                    "item_id": candidate.item_id,
                    "rank": rank,
                    "final_score": candidate.final_score,
                    "dino_score": candidate.dino_score,
                    "color_score": candidate.color_score,
                    "aspect_score": candidate.aspect_score,
                    "n_references": candidate.n_references,
                    # The matcher falls back to DINO-only when either descriptor is missing.
                    "used_fallback": candidate.color_score is None
                    or candidate.aspect_score is None,
                }
                for rank, candidate in enumerate(ranked, start=1)
            ]
            with self._lock:
                _append_rows(self.queries_path, QUERY_COLUMNS, [query_row])
                _append_rows(self.candidates_path, CANDIDATE_COLUMNS, candidate_rows)
        except Exception as exc:  # noqa: BLE001 - diagnostics must never fail a job
            self._report_failure("record_query", query.query_id, exc)

    def record_outcome(self, query_id: str, item_id: int | None, event: str) -> None:
        """Append what the main thread actually did with one resolve result."""
        try:
            row = {
                "query_id": query_id,
                "timestamp": _utc_now(),
                "item_id": item_id,
                "event": event,
            }
            with self._lock:
                _append_rows(self.outcomes_path, OUTCOME_COLUMNS, [row])
        except Exception as exc:  # noqa: BLE001 - diagnostics must never fail a job
            self._report_failure("record_outcome", query_id, exc)

    def _save_raw_crop(self, query_id: str, crop_bgr: NDArray[np.uint8] | None) -> str | None:
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        path = self.crops_dir / f"{query_id}_raw.png"
        Image.fromarray(np.ascontiguousarray(crop_bgr[:, :, ::-1])).save(path)
        return path.relative_to(self.run_dir).as_posix()

    def _save_masked_crop(self, query_id: str, crop: Image.Image | None) -> str | None:
        if crop is None:
            return None
        path = self.crops_dir / f"{query_id}_masked.png"
        crop.save(path)
        return path.relative_to(self.run_dir).as_posix()

    def _report_failure(self, operation: str, query_id: str, exc: Exception) -> None:
        logger.exception("reid_diagnostics.%s_failed query_id=%s", operation, query_id)
        log_action(Category.ERROR, diagnostics=operation, query=query_id, error=repr(exc))


def _append_rows(path: Path, columns: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    """Append rows to a CSV, writing the header only when the file is new."""
    is_new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        if is_new_file:
            writer.writeheader()
        writer.writerows(rows)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_files(paths: Iterable[Path]) -> str | None:
    """Short content hash of the given config files, so runs can be grouped by config."""
    digest = hashlib.sha256()
    hashed_any = False
    for path in paths:
        try:
            digest.update(Path(path).read_bytes())
            hashed_any = True
        except OSError:
            continue
    return digest.hexdigest()[:12] if hashed_any else None


def _git_commit(project_root: Path | None) -> str:
    """Current commit, suffixed -dirty for uncommitted changes; 'unknown' without git."""
    try:
        completed = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"
