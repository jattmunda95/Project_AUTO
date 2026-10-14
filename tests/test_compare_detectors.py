"""Tests for the measurement code in scripts/compare_detectors.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_detectors.py"
spec = importlib.util.spec_from_file_location("compare_detectors", SCRIPT)
cd = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["compare_detectors"] = cd  # dataclasses resolves annotations through sys.modules
spec.loader.exec_module(cd)

FPS = 30.0
SETTINGS = cd.MetricSettings()
BOX = (100.0, 100.0, 200.0, 200.0)


def frames_from(layout: dict[int, list[int]], box_for=lambda i: BOX, label="phone", total=None):
    """layout maps track_id -> frame indices on which that track is reported."""
    total = total if total is not None else max(max(v) for v in layout.values()) + 1
    records = []
    for index in range(total):
        tracks = [
            cd.TrackObs(tid, box_for(index), 0.9, label)
            for tid, on in layout.items()
            if index in on
        ]
        raw = np.array([t.box for t in tracks], dtype=np.float32).reshape(-1, 4)
        records.append(cd.FrameRecord(index, raw, tracks, detect_ms=10.0, track_ms=1.0))
    return records


def test_one_stationary_item_is_one_addable_track_with_no_churn() -> None:
    metrics, table = cd.compute_metrics(frames_from({1: list(range(90))}), FPS, SETTINGS)
    assert metrics["unique_ids"] == 1
    assert metrics["id_handovers"] == 0
    assert metrics["addable_tracks"] == 1
    assert metrics["never_still_tracks"] == 0
    assert metrics["dropout_pct"] == 0
    assert metrics["jitter_px"] == 0
    assert table[0]["longest_still_s"] == pytest.approx(89 / FPS, abs=0.01)


def test_new_id_where_another_just_vanished_is_a_handover() -> None:
    layout = {1: list(range(60)), 2: list(range(62, 120))}
    metrics, _ = cd.compute_metrics(frames_from(layout), FPS, SETTINGS)
    assert metrics["unique_ids"] == 2
    assert metrics["id_handovers"] == 1


def test_overlapping_tracks_are_two_objects_not_a_handover() -> None:
    layout = {1: list(range(90)), 2: list(range(30, 120))}
    metrics, _ = cd.compute_metrics(frames_from(layout), FPS, SETTINGS)
    assert metrics["id_handovers"] == 0


def test_new_id_far_away_or_long_after_is_not_a_handover() -> None:
    far = {1: list(range(30)), 2: list(range(32, 60))}
    boxes = lambda i: BOX if i < 31 else (800.0, 400.0, 900.0, 500.0)
    metrics, _ = cd.compute_metrics(frames_from(far, boxes), FPS, SETTINGS)
    assert metrics["id_handovers"] == 0
    late = {1: list(range(30)), 2: list(range(200, 260))}
    metrics, _ = cd.compute_metrics(frames_from(late), FPS, SETTINGS)
    assert metrics["id_handovers"] == 0


def test_gap_inside_a_track_counts_as_dropout() -> None:
    layout = {1: list(range(10)) + list(range(15, 30))}
    metrics, _ = cd.compute_metrics(frames_from(layout), FPS, SETTINGS)
    assert metrics["gap_events"] == 1
    assert metrics["dropout_pct"] == pytest.approx(100 * 5 / 30, abs=0.01)


def test_a_moving_track_is_never_still_and_a_brief_one_is_short() -> None:
    moving = lambda i: (10.0 + 10 * i, 100.0, 110.0 + 10 * i, 200.0)
    metrics, _ = cd.compute_metrics(frames_from({1: list(range(90))}, moving), FPS, SETTINGS)
    assert metrics["addable_tracks"] == 0
    assert metrics["never_still_tracks"] == 1
    assert metrics["jitter_px"] == pytest.approx(10.0)
    metrics, _ = cd.compute_metrics(frames_from({1: list(range(10))}), FPS, SETTINGS)
    assert metrics["short_tracks"] == 1
    assert metrics["never_still_tracks"] == 0


def test_stillness_clock_restarts_when_the_box_moves_past_tolerance() -> None:
    jump = lambda i: BOX if i < 30 else (300.0, 100.0, 400.0, 200.0)
    history = cd.collect_tracks(frames_from({1: list(range(60))}, jump))[1]
    assert cd.longest_still_seconds(history, FPS, SETTINGS) == pytest.approx(29 / FPS, abs=0.01)


def test_detections_per_frame_and_empty_frames() -> None:
    records = frames_from({1: list(range(30))}, total=60)
    metrics, _ = cd.compute_metrics(records, FPS, SETTINGS)
    assert metrics["det_per_frame_mean"] == pytest.approx(0.5)
    assert metrics["zero_det_frames_pct"] == pytest.approx(50.0)
    assert metrics["det_per_frame_max"] == 1


def test_label_flicker_is_counted_per_track() -> None:
    records = frames_from({1: list(range(40))})
    for record in records[::2]:
        record.tracks[0] = cd.TrackObs(1, BOX, 0.9, "mouse")
    metrics, table = cd.compute_metrics(records, FPS, SETTINGS)
    assert metrics["label_flicker_tracks"] == 1
    assert table[0]["labels"] == ["mouse", "phone"]


def test_timing_ignores_warmup_frames() -> None:
    records = frames_from({1: list(range(30))})
    records[0].detect_ms = 5000.0
    metrics, _ = cd.compute_metrics(records, FPS, SETTINGS)
    assert metrics["detect_ms_max"] == 10.0
    assert metrics["effective_fps"] == pytest.approx(1000 / 11.0, rel=0.01)


def test_no_tracks_gives_zeroes_not_errors() -> None:
    records = frames_from({1: []}, total=40)
    metrics, table = cd.compute_metrics(records, FPS, SETTINGS)
    assert metrics["unique_ids"] == 0
    assert metrics["dropout_pct"] == 0
    assert table == []


def test_aggregate_adds_counts_and_frame_weights_rates() -> None:
    a, _ = cd.compute_metrics(frames_from({1: list(range(30))}), FPS, SETTINGS)
    b, _ = cd.compute_metrics(frames_from({1: list(range(90)), 2: list(range(90))}), FPS, SETTINGS)
    combined = cd.aggregate_metrics([a, b])
    assert combined["unique_ids"] == 3
    assert combined["frames"] == 120
    assert combined["det_per_frame_mean"] == pytest.approx((1 * 30 + 2 * 90) / 120)
    assert cd.aggregate_metrics([]) == {}


def test_background_detector_boxes_only_what_changes_after_the_reference() -> None:
    detector = cd.BackgroundChangeDetector(reference_frames=5)
    blank = np.full((240, 320, 3), 80, np.uint8)
    for _ in range(5):
        assert len(detector.detect(blank).xyxy) == 0
    assert len(detector.detect(blank).xyxy) == 0
    changed = blank.copy()
    changed[60:140, 100:200] = 230
    found = detector.detect(changed)
    assert len(found.xyxy) == 1
    x1, y1, x2, y2 = found.xyxy[0]
    assert (x1, y1, x2, y2) == pytest.approx((100, 60, 200, 140), abs=2)
    assert detector.label(0) == "change"


def test_botsort_wrapper_keeps_one_id_for_a_steadily_detected_box() -> None:
    inference, _ = cd.load_settings()
    tracker = cd.BotsortTracker(inference.tracker_config)
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    ids = set()
    for step in range(12):
        x = 100 + 2 * step
        raw = cd.RawDetections(
            np.array([[x, 100, x + 80, 180]], np.float32),
            np.array([0.9], np.float32),
            np.array([3.0], np.float32),
        )
        ids.update(t.track_id for t in tracker.update(raw, frame, lambda c: f"class{c}"))
    assert len(ids) == 1


def test_registry_and_default_set_are_consistent() -> None:
    assert set(cd.DEFAULT_DETECTORS) <= set(cd.SPECS)
    assert {"yolo11s", "yolo11s_agnostic", "background"} <= set(cd.DEFAULT_DETECTORS)
    assert "yolo26s" not in cd.SPECS


def test_prompt_free_models_are_not_in_the_default_run() -> None:
    assert set(cd.PF_DETECTORS) <= set(cd.SPECS)
    assert not set(cd.PF_DETECTORS) & set(cd.DEFAULT_DETECTORS)
    assert all(cd.SPECS[name].kind == "yoloe_pf" for name in cd.PF_DETECTORS)
    assert all(cd.SPECS[name].kind != "yoloe_pf" for name in cd.DEFAULT_DETECTORS)


def test_select_detectors_default_explicit_and_pf() -> None:
    assert cd.select_detectors(None, pf=False) == cd.DEFAULT_DETECTORS
    assert cd.select_detectors(["yolo11s"], pf=False) == ("yolo11s",)
    assert cd.select_detectors(None, pf=True) == cd.PF_DETECTORS
    both = cd.select_detectors(["yolo11s", "yoloe11s_pf"], pf=True)
    assert both == (*cd.PF_DETECTORS, "yolo11s")


def test_pf_flag_parses() -> None:
    assert cd.parse_args(["--pf"]).pf is True
    assert cd.parse_args([]).pf is False


def test_missing_weights_are_skipped_not_downloaded(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cd, "MODELS_DIR", tmp_path)
    inference, _ = cd.load_settings()
    with pytest.raises(cd.SkipDetector, match="--download"):
        cd.build_detector(cd.SPECS["yoloe11s_pf"], inference, allow_download=False,
                          use_openvino=False)
    with pytest.raises(cd.SkipDetector, match="--download"):
        cd.build_detector(
            cd.SPECS["yoloe26s_text"], inference, allow_download=False, use_openvino=False
        )


def test_report_renders_runs_skips_and_charts() -> None:
    metrics, table = cd.compute_metrics(frames_from({1: list(range(60))}), FPS, SETTINGS)
    results = {
        "settings": {"clips": 1, "tracker": "botsort.yaml", "confidence": 0.18,
                     "image_size": 960},
        "runs": [{"detector": "yolo11s", "clip": "clip_a", "backend": "openvino",
                  "metrics": metrics, "tracks": table, "video": "videos/clip_a/yolo11s.mp4",
                  "csv": "frames/x.csv"}],
        "skipped": [{"detector": "yoloe11s_pf", "reason": "yoloe-11s-seg-pf.pt is not in models/"}],
    }
    series = {"clip_a": {"yolo11s": {"detections": [1.0] * 60, "ids": [1.0] * 60}}}
    page = cd.build_report(results, series)
    assert "yolo11s" in page and "yoloe-11s-seg-pf.pt is not in models/" in page
    assert "<polyline" in page and "videos/clip_a/yolo11s.mp4" in page


def test_downsample_keeps_short_series_and_shrinks_long_ones() -> None:
    assert cd.downsample([1, 2, 3], 10, "mean") == [1, 2, 3]
    long = list(range(1000))
    assert len(cd.downsample(long, 100, "last")) == 100
    assert cd.downsample(long, 100, "last")[-1] == 999


# ---- --resume ---------------------------------------------------------------------------


def make_clip(path: Path, frames: int = 40) -> Path:
    """A static textured scene where a bright square appears after the reference frames."""
    import cv2

    rng = np.random.default_rng(1)
    scene = rng.integers(40, 120, (240, 320, 3), dtype=np.uint8)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 240))
    for index in range(frames):
        frame = scene.copy()
        if index >= 20:
            frame[80:160, 120:220] = 240
        writer.write(frame)
    writer.release()
    return path


def test_resume_skips_finished_pairs_and_runs_only_the_new_clip(tmp_path, monkeypatch) -> None:
    clip_a = make_clip(tmp_path / "clip_a.mp4")
    clip_b = make_clip(tmp_path / "clip_b.mp4")
    out = tmp_path / "run"
    cd.main(["--detectors", "background", "--clips", str(clip_a), "--out", str(out)])
    assert [r["clip"] for r in cd_runs(out)] == ["clip_a"]

    calls: list[str] = []
    real_run_clip = cd.run_clip

    def counting(detector, clip, *args, **kwargs):
        calls.append(clip.stem)
        return real_run_clip(detector, clip, *args, **kwargs)

    monkeypatch.setattr(cd, "run_clip", counting)
    cd.main(["--detectors", "background", "--clips", str(clip_a), str(clip_b),
             "--resume", str(out)])
    assert calls == ["clip_b"]
    results = cd_results(out)
    assert sorted(r["clip"] for r in results["runs"]) == ["clip_a", "clip_b"]
    assert results["settings"]["clips"] == 2
    page = (out / "report.html").read_text(encoding="utf-8")
    assert "clip_a" in page and "clip_b" in page and page.count("<polyline") >= 4


def test_resume_redoes_a_pair_whose_files_were_lost(tmp_path, monkeypatch) -> None:
    clip = make_clip(tmp_path / "clip_a.mp4")
    out = tmp_path / "run"
    cd.main(["--detectors", "background", "--clips", str(clip), "--out", str(out)])
    (out / "videos" / "clip_a" / "background.mp4").unlink()  # interrupted mid-write

    calls: list[str] = []
    real_run_clip = cd.run_clip
    monkeypatch.setattr(cd, "run_clip", lambda d, c, *a, **k: (calls.append(c.stem),
                                                               real_run_clip(d, c, *a, **k))[1])
    cd.main(["--detectors", "background", "--clips", str(clip), "--resume", str(out)])
    assert calls == ["clip_a"]
    assert len(cd_runs(out)) == 1
    assert (out / "videos" / "clip_a" / "background.mp4").exists()


def test_resume_with_everything_finished_runs_nothing(tmp_path, monkeypatch) -> None:
    clip = make_clip(tmp_path / "clip_a.mp4")
    out = tmp_path / "run"
    cd.main(["--detectors", "background", "--clips", str(clip), "--out", str(out)])
    monkeypatch.setattr(cd, "run_clip", lambda *a, **k: pytest.fail("should not run"))
    cd.main(["--detectors", "background", "--clips", str(clip), "--resume", str(out)])
    assert len(cd_runs(out)) == 1


def test_resume_refuses_when_settings_changed(tmp_path) -> None:
    clip = make_clip(tmp_path / "clip_a.mp4")
    out = tmp_path / "run"
    cd.main(["--detectors", "background", "--clips", str(clip), "--out", str(out)])
    with pytest.raises(SystemExit, match="max_frames"):
        cd.main(["--detectors", "background", "--clips", str(clip), "--max-frames", "10",
                 "--resume", str(out)])


def test_resume_needs_an_existing_run_and_excludes_out(tmp_path) -> None:
    with pytest.raises(SystemExit, match="does not exist"):
        cd.main(["--detectors", "background", "--clips", str(make_clip(tmp_path / "c.mp4")),
                 "--resume", str(tmp_path / "nope")])
    with pytest.raises(SystemExit):
        cd.parse_args(["--resume", str(tmp_path), "--out", str(tmp_path)])


def test_load_previous_ignores_settings_the_old_run_never_recorded(tmp_path) -> None:
    import json

    (tmp_path / "results.json").write_text(json.dumps({
        "settings": {"confidence": 0.18, "max_frames": None},
        "runs": [], "skipped": [{"detector": "x", "reason": "r"}],
    }), encoding="utf-8")
    current = {"confidence": 0.18, "max_frames": None, "still_seconds": 2.0}
    results, series = cd.load_previous(tmp_path, current, want_video=True)
    assert results["runs"] == [] and series == {}


def test_a_stale_skip_is_dropped_when_that_detector_runs_again(tmp_path) -> None:
    import json

    clip = make_clip(tmp_path / "clip_a.mp4")
    out = tmp_path / "run"
    cd.main(["--detectors", "background", "--clips", str(clip), "--out", str(out)])
    data = cd_results(out)
    data["skipped"] = [{"detector": "background", "reason": "old failure"},
                       {"detector": "yoloe11s_pf", "reason": "kept"}]
    (out / "results.json").write_text(json.dumps(data), encoding="utf-8")
    cd.main(["--detectors", "background", "--clips", str(clip), "--resume", str(out)])
    assert [s["detector"] for s in cd_results(out)["skipped"]] == ["yoloe11s_pf"]


def cd_results(out: Path) -> dict:
    import json

    return json.loads((out / "results.json").read_text(encoding="utf-8"))


def cd_runs(out: Path) -> list[dict]:
    return cd_results(out)["runs"]
