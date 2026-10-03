# Setup and operations

## Prepare an editable checkout

Use a Python interpreter within the declared >=3.10,<3.14 range. From the project root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[reid]"
python -m pip install pytest
```

The reid extra is needed for the current full live app. Installation commands are instructions; dependencies were not installed or changed during documentation generation. If environment activation is restricted, invoke the virtual environment's python executable directly rather than changing system execution policy.

## Preflight

1. Check camera indexes and desired capture properties in camera.yaml.
2. Check CPU/model paths in perception.yaml, segmenter.yaml, and reid.yaml.
3. Ensure initial model acquisition/export can complete, or that the required assets are already cached for offline use.
4. Confirm the configured database schema is current. Back up an existing database with the app stopped before any schema changes. There is no automatic migration command. The inspected-schema SQL described below must be reviewed before deliberate use.
5. For an isolated demonstration, choose a new unused database path in table.yaml; this preserves the existing history and allows create_schema to create fresh tables. Restore the intended configuration after the demonstration.
6. Use an interactive desktop and keep the monitored scene/camera stable.

Do not replace or delete the user's existing database as a setup shortcut. Fresh database creation and upgrading an existing schema are different operations.

## Run and stop

```powershell
project-auto
```

Equivalent module invocation after installation:

```powershell
python -m project_auto.main
```

The camera window displays detection annotations. Press q with that window focused to quit. Press w for item whereabouts or r for region inspection, then answer the prompt in the terminal; matching polygons are highlighted temporarily. First startup can take substantially longer while models load or export; no startup time guarantee is established. The window does not yet expose complete identity/lifecycle state.

## Troubleshooting

| Symptom | Likely area | First check |
| --- | --- | --- |
| Missing torch, transformers, PIL, or SAM2 import | Environment/dependency compatibility | Use the active virtual environment and install the reid extra; verify the installed Transformers exposes the required SAM2 classes |
| Camera will not open / wrong view | Device mapping or other camera user | Check indexes, OS camera availability, and whether another app holds the device |
| Startup fails on download/model load | Model cache/network/path | Verify configured files and cached model access; export directory existence alone does not establish integrity |
| Missing-column error or identification stops resolving | Old schema / worker startup failure | Inspect schema against models.py and worker logs before migration |
| Repeated pending decisions | Bad crop/mask, edge clipping | Inspect scene quality and diagnostic rejection reasons; retain True foreground mask polarity |
| Same item repeatedly becomes NEW | Reference availability or similarity quality | Check successful reference saves, matching model/preprocessing, and same/different-item score distributions |
| Slow display / many deferred jobs | Main-thread inference or worker pressure | Measure stage timings and queue behavior using the performance chapter |
| MOVED absent for a newly seen object | Identity still unresolved | Current coordinator drops movement before binding an item |
| Shutdown takes longer than five seconds | Queued/in-progress worker work | Shutdown has a blocking sentinel enqueue; timeout is not a total bound |

## Data and model maintenance

Back up the SQLite database while the application is stopped, together with any externally managed evidence files and a copy of configuration/model provenance. The live code does not provide automated backups or retention. Model artifacts are generated/downloaded files; do not edit their binary contents. Changing a model or preprocessing requires checking reference compatibility and recalibrating recognition.

The repository's research examples are separate from the live database gallery. Notebook reference images and saved `.pt` files are not automatically imported into production memory by app.py.

## Documentation portal

Open [index.html](index.html) directly in a browser. Chapter pages render fully offline without a server, third-party scripts, or external fonts. Search on the home page filters chapters by their full text; navigation and content remain usable with JavaScript disabled. Each page offers its Markdown source and browser print support.

Edit the numbered Markdown chapters, then rebuild generated pages from the repository root:

```powershell
node docs/build_docs.mjs
python docs/check_docs.py
```

Sources: [application](../src/project_auto/app.py), [camera](../src/project_auto/capture/camera.py), [models guide](../models/README.md), [configuration](06-configuration.md).

## Define regions before tracking

Run python -m project_auto.region_calibration from the activated editable checkout. Click polygon vertices, use u to undo, Enter/c to close, and enter a unique name in the terminal. q exits calibration. The application reuses Camera/CameraConfig; calibration is a separate launch and does not run detector/identity inference. Complete controls, source-path invocation, and query semantics are in [regions and spatial memory](11-regions.md).

## Record ReID ground-truth data

Use the separate diagnostic entry point, not the normal app:

```powershell
python -m project_auto.reid_diagnostics_app
```

After reinstalling the editable package, `project-auto-diagnostics` is equivalent. Controls are the same as the normal app. It uses its own database, `data/diagnostics/project_auto_diagnostics.db`, so test objects never enter normal memory. Delete that file to start a session with an empty gallery, which keeps NEW/SAME labels unambiguous. It contains no calibrated regions.

Each launch creates `logs/reid_runs/<run_id>/` with queries.csv, candidates.csv, outcomes.csv and crops/. Label offline: open queries.csv beside the crops folder and fill in a ground_truth.csv (query_id, true_object, expected_outcome NEW/SAME, label_quality clear/partial/unsure, notes) plus an objects.csv mapping each object name to the item_id created when it was first enrolled. The first sighting of an object in a fresh diagnostic database is NEW; later sightings are SAME. Both `logs/` and `data/diagnostics/` are git-ignored.

### Enrolment, snapshots and restoring between blocks

Run these with the diagnostics app stopped unless stated. `scripts/reid_snapshot.py` reads the database path from reid_diagnostics.yaml and keeps snapshots in a `snapshots/` folder beside it.

1. Delete `data/diagnostics/project_auto_diagnostics.db` for an empty gallery. Use 6 to 10 test objects.
2. Start `python -m project_auto.reid_diagnostics_app`. Enrol objects one at a time: place one, wait for ADDED and its two baseline CAPTURE lines, remove it and wait for REMOVED. An object is only added after it has been still for the confirmation time, so hold nothing in the frame.
3. Stop the app, then `python scripts/reid_snapshot.py backup post_enrolment`. It uses SQLite's backup API because the store runs in WAL mode and a plain file copy loses recent writes.
4. Run scenario blocks (S0 to S12 in TASKS.md, item 5). Before each isolated block S1 to S9, stop the app and run `python scripts/reid_snapshot.py restore post_enrolment`; restore first saves the overwritten database as `_before_restore`. `list` shows snapshots.
5. Label offline in Excel, then run the (not yet written) join and analysis script.

outcomes.csv now also records handoff outcomes (HANDOFF, DEFER_HANDOFF, STALE, DROPPED). Early handoff probes are recorded as queries too, so label only the ones you intend to analyse. The raw and masked crops show exactly what ReID saw; on 2 October they showed one phone appearing as screen-off, screen-on and rear-camera views, and a hand being added as an item. The S0 to S12 operator checklist and the analysis script are still not written.

## Missing-column startup incident and recovery

On 14 September, the identification worker failed during initial gallery loading with sqlite3.OperationalError: no such column: item_embeddings.aspect_ratio. Inspection found eight absent nullable columns: embedding aspect_ratio/color_histogram, item current_region_id/current_box, and event source_region_id/destination_region_id/source_box_area_fraction/destination_box_area_fraction. The regions table already existed. The earlier model-download warnings were not the exception shown in this traceback.

[manual_schema_update.sql](manual_schema_update.sql) adds exactly these columns and three region-FK indexes in one transaction, retaining SET NULL deletion behavior. It is a one-time SQL artifact for that inspected schema, not a versioned migration framework or an idempotent startup routine. Do not run it on a fresh/already-updated database. Confirm the current schema and take a SQLite backup with the application stopped before explicitly applying it.

A test on a disposable copy preserved all original values across 9 items, 49 embeddings and 36 events, passed SQLite integrity/FK checks, and loaded the gallery and item/event/region queries. The first test subsequently failed temporary-directory cleanup because a SQLite handle remained open. Revalidation found the live file absent; no migration was applied to the live database by this work. A database file is present again at the documentation review, but its origin, retained history, and schema have not been reverified. Confirm reset/recovery intent before any schema action; do not infer continuity from file existence.

If a database was deliberately reset, normal startup creates fresh tables when the configured file is absent, but previous item history is not restored. If history is required, recovery from the intended backup takes priority. Initial worker gallery loading remains outside per-job exception handling, so schema failures still require startup diagnostics rather than ordinary per-track retries.
