# Verification and documentation maintenance

## Verification boundaries

This documentation reflects source/YAML review on 14 September 2026. The earlier region implementation run in this chat passed 160 tests (130 existing tests plus 30 geometry, spatial memory and UI checks). That is recorded implementation evidence, not a fresh test execution during this documentation-only refresh or a guarantee of live hardware behavior.

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
| [test_reid.py](../tests/test_reid.py) | Synthetic embedding matching and prototype shortlist behavior |
| [test_scene_processor.py](../tests/test_scene_processor.py) | Crop/mask preparation and identity proposals |
| [test_identification.py](../tests/test_identification.py) | ReferenceManager queue/cooldown state |
| [test_identification_worker.py](../tests/test_identification_worker.py) | Worker jobs, results, queue behavior, shutdown |
| [test_coordinator.py](../tests/test_coordinator.py) | Async submission, result application, retirement and binding scenarios |
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

The numbered Markdown files are the maintained content source. `build_docs.py` generates the HTML home page, chapter pages, shared CSS, and home-page search behavior using Python's standard library. Generated files are committed for immediate offline access. `check_docs.py` checks local HTML links, fragments, titles, and generated-page coverage without importing project models or opening its database.

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
