"""ReidDiagnostics tests: run folder, CSV tables, crops, outcome linking, fail-soft."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from project_auto.utils.reid_diagnostics import (
    CANDIDATE_COLUMNS,
    OUTCOME_COLUMNS,
    QUERY_COLUMNS,
    CandidateRecord,
    QueryRecord,
    ReidDiagnostics,
)

RUN_SETTINGS = {
    "acceptance_threshold": 0.55,
    "margin_threshold": 0.15,
    "weights": "0.65/0.2/0.15",
    "top_k": 3,
    "prototype_shortlist_size": 0,
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def read_header(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as csv_file:
        return next(csv.reader(csv_file))


def query(query_id: str, **overrides: object) -> QueryRecord:
    values: dict[str, object] = {
        "query_id": query_id,
        "frame_index": 12,
        "track_id": 7,
        "box": (10, 20, 30, 60),
        "det_confidence": 0.8,
        "sam_score": 0.9,
        "mask_occupancy": 0.5,
        "gallery_size": 2,
        "decision": "MATCH",
        "decision_reason": "accepted",
        "matched_item_id": 42,
    }
    values.update(overrides)
    return QueryRecord(**values)  # type: ignore[arg-type]


@pytest.fixture
def diagnostics(tmp_path: Path) -> ReidDiagnostics:
    config = tmp_path / "reid.yaml"
    config.write_text("acceptance_threshold: 0.55\n", encoding="utf-8")
    return ReidDiagnostics(
        tmp_path / "runs", RUN_SETTINGS, config_paths=[config], project_root=tmp_path
    )


def test_creates_a_run_folder_with_a_crops_directory(diagnostics: ReidDiagnostics) -> None:
    assert diagnostics.run_dir.name == diagnostics.run_id
    assert diagnostics.crops_dir.is_dir()


def test_query_ids_are_unique_and_prefixed_by_the_run(diagnostics: ReidDiagnostics) -> None:
    first, second = diagnostics.new_query_id(), diagnostics.new_query_id()

    assert first != second
    assert first.startswith(f"{diagnostics.run_id}-q")


def test_record_query_writes_one_query_row_and_ranked_candidate_rows(
    diagnostics: ReidDiagnostics,
) -> None:
    candidates = [
        CandidateRecord(101, 0.4, 0.3, 0.6, 0.9, 2),
        CandidateRecord(42, 0.8, 0.7, None, 0.9, 5),
    ]

    diagnostics.record_query(query("q1"), candidates)

    assert read_header(diagnostics.queries_path) == list(QUERY_COLUMNS)
    assert read_header(diagnostics.candidates_path) == list(CANDIDATE_COLUMNS)
    [query_row] = read_rows(diagnostics.queries_path)
    assert query_row["query_id"] == "q1"
    assert query_row["run_id"] == diagnostics.run_id
    assert query_row["box_area"] == "800"
    assert query_row["decision"] == "MATCH"
    assert query_row["acceptance_threshold"] == "0.55"
    assert query_row["prototype_shortlist_size"] == "0"
    assert query_row["config_hash"] != ""
    assert query_row["git_commit"] != ""

    candidate_rows = read_rows(diagnostics.candidates_path)
    assert [row["item_id"] for row in candidate_rows] == ["42", "101"]
    assert [row["rank"] for row in candidate_rows] == ["1", "2"]
    assert candidate_rows[0]["used_fallback"] == "True"  # color missing -> DINO-only
    assert candidate_rows[1]["used_fallback"] == "False"
    assert candidate_rows[0]["n_references"] == "5"


def test_header_is_written_once_across_appends(diagnostics: ReidDiagnostics) -> None:
    diagnostics.record_query(query("q1"), [CandidateRecord(1, 0.5, 0.5, 0.5, 0.5, 1)])
    diagnostics.record_query(query("q2"), [CandidateRecord(1, 0.5, 0.5, 0.5, 0.5, 1)])

    assert [row["query_id"] for row in read_rows(diagnostics.queries_path)] == ["q1", "q2"]
    assert len(read_rows(diagnostics.candidates_path)) == 2


def test_crops_are_saved_and_referenced_relative_to_the_run(
    diagnostics: ReidDiagnostics,
) -> None:
    raw_bgr = np.zeros((4, 6, 3), dtype=np.uint8)
    raw_bgr[:, :, 0] = 255  # pure blue in BGR
    masked = Image.new("RGB", (4, 4), "red")

    diagnostics.record_query(query("q1"), [], raw_crop_bgr=raw_bgr, masked_crop=masked)

    [row] = read_rows(diagnostics.queries_path)
    raw_path = diagnostics.run_dir / row["raw_crop_path"]
    masked_path = diagnostics.run_dir / row["masked_crop_path"]
    assert raw_path.is_file() and masked_path.is_file()
    # Saved as RGB, so the BGR blue channel lands in the third RGB channel.
    assert Image.open(raw_path).getpixel((0, 0)) == (0, 0, 255)
    assert Image.open(raw_path).size == (6, 4)


def test_pending_query_without_crops_or_candidates_still_records_a_row(
    diagnostics: ReidDiagnostics,
) -> None:
    diagnostics.record_query(
        query("q1", decision="DEFER", decision_reason="no_mask", sam_score=None), []
    )

    [row] = read_rows(diagnostics.queries_path)
    assert row["decision_reason"] == "no_mask"
    assert row["masked_crop_path"] == ""
    assert row["raw_crop_path"] == ""
    assert read_rows(diagnostics.candidates_path) == []


def test_record_outcome_links_by_query_id(diagnostics: ReidDiagnostics) -> None:
    diagnostics.record_outcome("q1", 42, "RETURNED")

    assert read_header(diagnostics.outcomes_path) == list(OUTCOME_COLUMNS)
    [row] = read_rows(diagnostics.outcomes_path)
    assert (row["query_id"], row["item_id"], row["event"]) == ("q1", "42", "RETURNED")


def test_a_write_failure_is_reported_not_raised(
    diagnostics: ReidDiagnostics, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("project_auto.utils.reid_diagnostics._append_rows", fail)

    diagnostics.record_query(query("q1"), [])  # must not raise
    diagnostics.record_outcome("q1", None, "DEFER")  # must not raise
