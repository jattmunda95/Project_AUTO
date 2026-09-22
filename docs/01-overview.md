# Project overview

Project AUTO is a local computer-vision prototype that remembers physical objects observed on a table and records meaningful changes in their presence and placement. It combines live camera detection, temporary tracking, appearance-based identity matching, and a durable SQLite event history.

## Purpose and audience

The intended outcome is to answer where an object was last placed, even after it disappears from view. The current application includes polygon calibration and temporary console queries for item location, region contents, activity, and historical association; a richer query UI remains deferred.

This documentation describes the source snapshot inspected on 14 September 2026, package version 0.1.0. It is organized for three levels of use: stakeholders start here and with scope; engineers read architecture, modules, and data; operators use setup, configuration, and verification.

## Capability status

| Capability | Current status | Practical meaning |
| --- | --- | --- |
| Live camera input | Implemented | Tries configured USB camera, then webcam; verifies an actual frame |
| Detection and temporary tracking | Implemented | YOLO11s/OpenVINO with Ultralytics BoT-SORT |
| Object lifecycle | Implemented | Time-confirmed addition, stable movement, and absence-based removal |
| Permanent identity | Integrated, calibration pending | SAM2 masking and DINOv2 gallery matching run on a background worker |
| Return recognition | Integrated | A match to a removed item records RETURNED against its existing permanent ID |
| Persistent memory | Implemented | Items, events, references, prototypes, and polygon regions in SQLite |
| User display | Debug interface | OpenCV detections, keyboard-triggered console queries, temporary region highlights |
| Location questions and semantic regions | Implemented; hardware validation pending | Separate polygon calibration; event-based current location and region-history queries |
| Production accuracy or performance SLA | Not established | No measured benchmark report or calibrated recognition target |

Project task notes report successful real-hardware addition, re-identification, and return scenarios. These are historical reports, not a fresh hardware validation performed for this documentation.

## A typical interaction

An object becomes visible and receives a temporary BoT-SORT track ID. After temporal confirmation, the application schedules identity processing. A usable masked view is compared with saved references. An unmatched view creates a permanent item and ADDED event; a match to a removed item creates RETURNED; a match to an already present item associates the track without another addition event. Subsequent stable movement or sustained absence records MOVED or REMOVED.

A bad view defers identity rather than proving a new object exists. Detection class names such as a cup label are not unique physical identities. Likewise, a tracker ID is not a permanent item ID.

## Reading routes

- Stakeholders: [overview](01-overview.md), [scope and assumptions](08-scope.md), [performance](07-performance.md).
- Developers: [architecture](02-architecture.md), [module reference](03-modules.md), [data contracts](05-data.md), [verification](10-verification.md).
- Operators: [setup and operations](09-operations.md), [configuration](06-configuration.md), [dependencies](04-stack.md).

## Evidence and precedence

Current executable code and YAML settings are the primary evidence. Package declarations establish dependency ranges, not the exact versions installed on a machine. Historical README/context/task statements may describe earlier stages; where they conflict, this documentation follows the inspected implementation and labels unresolved gaps.

Sources: [application composition](../src/project_auto/app.py), [package metadata](../pyproject.toml), [project task history](../src/project_auto/TASKS.md).

Spatial memory uses the permanently fixed camera frame, not world coordinates. See [region workflow and contracts](11-regions.md). The latest implementation test run on 14 September passed 160 tests; live region calibration is not yet tested. Current YOLO settings are confidence 0.18 and image size 960. Weighted ReID acceptance 0.55 and margin 0.15 remain provisional.
