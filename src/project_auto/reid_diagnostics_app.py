"""Diagnostic entry point: run Project AUTO while recording ReID ground-truth data.

Run with: python -m project_auto.reid_diagnostics_app

Runs the normal application pipeline (app.run_app) with three differences, all
read from configs/reid_diagnostics.yaml:
- a ReidDiagnostics recorder, so every resolve job writes query/candidate/outcome
  rows and crops to a new run folder for offline labelling;
- prototype_shortlist_size overridden (0 = score the whole gallery);
- a separate database, so diagnostic items never enter the normal memory.

The normal project-auto entry point (main.py) never imports this module or reads
its config, so its behaviour is unaffected.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import yaml

from project_auto.app import run_app
from project_auto.memory.reid import ASPECT_WEIGHT, COLOR_WEIGHT, DINO_WEIGHT, ReidConfig
from project_auto.utils.reid_diagnostics import ReidDiagnostics


def run_diagnostics() -> None:
    """Build the diagnostic overrides and run the normal application with them."""
    project_root = Path(__file__).resolve().parents[2]
    configs_dir = project_root / "configs"
    diagnostics_config_path = configs_dir / "reid_diagnostics.yaml"
    reid_config_path = configs_dir / "reid.yaml"

    with diagnostics_config_path.open(encoding="utf-8") as config_file:
        settings = yaml.safe_load(config_file)

    reid_config = dataclasses.replace(
        ReidConfig.from_yaml(reid_config_path),
        prototype_shortlist_size=int(settings["prototype_shortlist_size"]),
    )
    diagnostics = ReidDiagnostics(
        root_dir=project_root / settings["output_dir"],
        run_settings={
            "acceptance_threshold": reid_config.acceptance_threshold,
            "margin_threshold": reid_config.margin_threshold,
            "weights": f"{DINO_WEIGHT}/{COLOR_WEIGHT}/{ASPECT_WEIGHT}",
            "top_k": reid_config.top_k,
            "prototype_shortlist_size": reid_config.prototype_shortlist_size,
        },
        config_paths=(
            reid_config_path,
            diagnostics_config_path,
            configs_dir / "scene_processor.yaml",
            configs_dir / "segmenter.yaml",
        ),
        project_root=project_root,
    )
    run_app(
        reid_config=reid_config,
        database_path=project_root / settings["database_path"],
        diagnostics=diagnostics,
    )


def main() -> None:
    """Start Project AUTO in ReID diagnostic mode."""
    run_diagnostics()


if __name__ == "__main__":
    main()
