# Verification and documentation maintenance

## Verification boundaries

This documentation reflects source/YAML review through 26 September 2026 (ReID diagnostic entry point; earlier Reference Capture V2 and action-classified logging). The 26 September run passed 228 tests, adding coverage for the diagnostic recorder, the diagnostic entry point and the candidate/decision-reason evidence; neither entry point has been run against a live camera since. The 23 September implementation run passed 200 tests (34 new: reference-policy state machine and descriptor/geometry coverage, plus updated coordinator/tracker/worker tests). That is recorded implementation evidence, not a guarantee of live hardware behavior — Reference Capture V2 has not yet run against a real camera. The earlier region implementation run passed 160 tests; both counts are historical run evidence, not a claim about the current total.

`pytest` was not previously installed in `.venv`; it has since been installed. On Windows with the project checked out under a OneDrive-synced path, `pytest`'s default `tmp_path` fixture can hit `PermissionError` against the OneDrive temp-sync lock (the same class of issue as the SQLite `database is locked` investigation); pass `--basetemp` pointed at a plain local writable directory to avoid it.

The working tree contains uncommitted application/configuration and documentation changes. Weighted descriptor scoring and region integration are present in the inspected source; no held-out recognition calibration or throughput measurement is claimed. Geometry and GUI interaction tests use isolated databases and simulated camera/input. Ruff was unavailable during implementation. The earlier manual schema validation passed data and query checks on a copy, then encountered temporary-file cleanup trouble; no live database migration is claimed. Recovery/reset provenance is still to confirm.

The documentation build/link checks validate navigation, anchors, and generated chapter coverage without importing application models or opening its database. The existing portal visual style is retained; no new browser-based visual QA is claimed by this refresh.

## Existing automated coverage

| Test file | Focus |
| --- | --- |
| [test_camera.py](../tests/test_camera.py) | Camera fallback and capture setup behavior |
| [test_detector.py](../tests/test_detector.py) | Ultralytics tracking calls and detection conversion |
| [test_tracker.py](../tests/test_tracker.py) | Temporal confirmation, missing tracks, movement, invalid settings |
| [test_state_machine.py](../tests/test_state_machine.py) | Lifecycle decisions and permanent identity requirements |
| [test_event_engine.py](../tests/test_event_engine.py) | Event dispatch, bindings, return/removal behavior |
| [test_memory_database.py](../tests/test_memory_database.py) | Schema and persistent-memory behavior |
| [test_reid.py](../tests/test_reid.py) | Synthetic embedding matching, prototype shortlist behavior, whole-gallery scoring, returned candidates and decision reasons |
| [test_reid_diagnostics.py](../tests/test_reid_diagnostics.py) | Run folder, CSV headers and rows, candidate ranking, crop saving, outcome linking, fail-soft writes |
| [test_reid_diagnostics_app.py](../tests/test_reid_diagnostics_app.py) | Diagnostic overrides reach run_app; normal reid.yaml keeps shortlist 3 and has no diagnostic keys |
| [test_scene_processor.py](../tests/test_scene_processor.py) | Crop/mask preparation and identity proposals |
| [test_descriptors.py](../tests/test_descriptors.py) | Frame-visibility geometry (predicted vs. clipped box) and Laplacian sharpness |
| [test_reference_policy.py](../tests/test_reference_policy.py) | ReferencePolicy state machine: baseline scheduling, MOVE_START/MOVE_END nomination, attempt budgets, retry spacing |
| [test_identification.py](../tests/test_identification.py) | ReferenceManager queue/cooldown state |
| [test_identification_worker.py](../tests/test_identification_worker.py) | Worker jobs, results, queue behavior, shutdown, the reference-quality/novelty gate |
| [test_coordinator.py](../tests/test_coordinator.py) | Async submission, result application, retirement and binding scenarios, event-driven reference-capture nomination |
| [test_regions.py](../tests/test_regions.py) | Shoelace area, concave/edge inclusion, centroid overlap resolution, validation and normalized areas |
| [test_region_memory.py](../tests/test_region_memory.py) | ADD/MOVED/REMOVED/RETURNED spatial persistence, rollback, deletion/snapshots, fallback, query semantics and no query writes |
| [test_region_ui.py](../tests/test_region_ui.py) | Drawing copies/blending, terminal queries, simulated polygon clicks/undo/save |

Tests using synthetic embeddings and fake workers verify logic, not real-world recognition accuracy, camera throughput, or all thread interleavings. Presence of a test file does not certify complete coverage of its subsystem.

Run the suite in a prepared environment:

```powershell
python -m pytest -q
```

For persistence-focused work:

```powershell
python -m pytest -q tests/test_memory_database.py tests/test_event_engine.py
```

## Hardware validation still needed

Use an isolated database and record the hardware/configuration. Calibrate overlapping polygons, verify smallest-region selection and w/r highlights, and check removal clears current contents while history survives. Demonstrate addition, movement, removal, and return of the same physical item under a new track ID; ensure a different item does not falsely claim it. Exercise occlusion, bad views, queue pressure, retirement during inference, and application shutdown. Measure performance separately from correctness and retain labeled evidence for calibration.

## Documentation structure and ownership

The numbered Markdown files are the maintained content source. `build_docs.mjs` generates the HTML home page, chapter pages, shared CSS, and home-page search behavior using Node.js, with Python syntax parsing for the source index. Generated files are committed for immediate offline access. `check_docs.py` checks local HTML links, fragments, titles, and generated-page coverage without importing project models or opening its database.

| Change type | Chapters to review |
| --- | --- |
| Product capability or scope | Overview, scope, operations |
| Threading, event routing, module responsibilities | Architecture, modules, data, performance |
| Models or package versions | Stack, configuration, operations, performance |
| YAML defaults | Configuration and affected performance statements |
| Database schema or event semantics | Data, operations, verification |
| New benchmark or hardware run | Performance and verification, with date and reproducible conditions |

Update the review date after a substantive source review. Cite the relevant source file near behavioral claims. Mark each result as configured, derived, historically reported, newly measured, or pending. Never convert a camera FPS request, a similarity threshold, or a test count into an accuracy or performance guarantee.

## Maintenance boundaries

The root README and PROJECT_CONTEXT now summarize the integrated async and spatial architecture. TASKS separates current priorities from historical milestones; historical thresholds/pass counts are not current configuration. AGENTS retains the working agreement and points to the updated architecture and remaining verification work.

Keep source-level limits explicit: lifecycle writes still run on the main thread, reference save limits do not stop every inference request, resolve and capture retry behavior differ, and margin rejection currently becomes NEW through SceneProcessor. The region layer does not add per-frame DB observations or move SQL into geometry/tracking. Review [chapter 11](11-regions.md) with any change to these contracts.

## Diagram and source-index verification

The 26 September documentation integration adds local SVG diagrams, a complete project map and a generated Python/file index. The builder parses Python syntax without importing the application. The documentation checker validates local links and diagram assets. Rebuild after source changes so line numbers and catalog entries remain current. These documentation checks do not imply a live-camera or recognition-accuracy test.
