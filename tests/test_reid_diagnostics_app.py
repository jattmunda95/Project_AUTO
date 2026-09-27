"""Diagnostic entry point tests: overrides reach run_app; the normal config is untouched."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import yaml

from project_auto import reid_diagnostics_app

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_diagnostic_entry_point_passes_its_overrides_to_run_app() -> None:
    with (
        patch.object(reid_diagnostics_app, "run_app") as run_app,
        patch.object(reid_diagnostics_app, "ReidDiagnostics") as recorder_class,
    ):
        reid_diagnostics_app.run_diagnostics()

    kwargs = run_app.call_args.kwargs
    assert kwargs["reid_config"].prototype_shortlist_size == 0
    assert kwargs["database_path"] == (
        PROJECT_ROOT / "data" / "diagnostics" / "project_auto_diagnostics.db"
    )
    assert kwargs["diagnostics"] is recorder_class.return_value
    assert recorder_class.call_args.kwargs["root_dir"] == PROJECT_ROOT / "logs" / "reid_runs"


def test_normal_reid_config_keeps_the_shortlist_and_has_no_diagnostic_keys() -> None:
    settings = yaml.safe_load((PROJECT_ROOT / "configs" / "reid.yaml").read_text(encoding="utf-8"))

    assert settings["prototype_shortlist_size"] == 3
    assert "diagnostics_dir" not in settings
    assert "match_log_path" not in settings
