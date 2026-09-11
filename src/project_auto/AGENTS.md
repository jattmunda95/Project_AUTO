# Project AUTO - Codex Instructions

## Project goal

Build a local computer-vision system that remembers where physical objects were last placed
on a table.

The target architecture is:

```text
Camera
-> YOLO11s/OpenVINO detection
-> ByteTrack or BoT-SORT tracking
-> permanent object identity association
-> state and placement-event decisions
-> SQLite persistence
-> query interface
```

The detector invokes Ultralytics BoT-SORT and exposes optional temporary track IDs. The live
application passes detections through the tracker, state machine, event engine, and SQLite
store. The verified lifecycle supports time-confirmed `ADD`, stable-placement `MOVED`, and
`REMOVE` after a stable or moving track has been absent for two seconds. Track-to-item
bindings are provisional session mappings, not permanent identity recognition. Do not
describe reliable re-identification as implemented.

## Target hardware

- Lenovo Yoga Pro 7i
- Intel integrated graphics (CPU inference for SAM2/DINOv2; no GPU acceleration configured yet)
- OpenVINO-optimized detector inference
- Camera: prefers an external USB camera, falls back to the built-in/inbuilt webcam if it is
  not connected (`capture/camera.py`, `configs/camera.yaml`: `usb_device`, `webcam_device`).
  Known issue: currently falls back to the built-in camera even when the USB webcam is
  connected; the assumed USB device index has not been confirmed on this machine.
- Entirely local operation for the MVP

## Engineering conventions

- Python 3.10 or newer.
- Use type hints for public functions.
- Keep camera capture, inference, tracking, state decisions, and persistence separate.
- Configuration belongs in YAML rather than hard-coded constants.
- Use `pathlib.Path` for filesystem paths.
- Do not introduce a dependency without explaining why.
- Prefer small, testable components over one large application file.
- Never treat a tracker ID alone as permanent object identity.
- Never create database observations for individual video frames.
- Store meaningful state changes and events instead.
- Store evidence file paths in SQLite, not image/video binary data.

## Working agreement

- By default, change only one file per user prompt.
- Explain the intended file change before editing it.
- When building function by function, add only one explicitly approved code chunk at a time
  and explain it before moving on.
- After each main function or feature is verified, update `TASKS.md` in a separate approved
  step.
- Ask for approval before moving to the next function when working function by function.

## Before changing functional code

1. Read `PROJECT_CONTEXT.md`.
2. Read `TASKS.md`.
3. Inspect the relevant existing files.
4. Explain any architectural change before implementing it.
5. Ask before changing additional functional files.

## Verification

After modifying code:

- run focused tests for the changed behavior;
- run formatting and lint checks when available;
- report what changed and what remains unverified;
- do not mark work complete until verification passes.

## Current priority

Follow the `Current task` section in `TASKS.md`. Scene processing and live identity
integration are now wired end to end through `IdentityCoordinator`
(`events/coordinator.py`) and `app.py`, but unverified against real hardware: no live
camera/model run has succeeded yet. Standalone ReID, SAM2 segmentation, two-stage
prototype-then-reference matching, and prototype calculation exist; reliable identity
recognition is not verified against real images.

Immediate blockers before a usable live run:
- Camera device selection defaults to the built-in webcam even when a USB camera is
  connected; the assumed `usb_device` index in `configs/camera.yaml` needs confirming.
- Live frame rate after the new `pending_retry_interval_seconds` throttle is unconfirmed;
  SAM2/DINOv2 run synchronously on CPU inside the capture loop.

The app/coordinator requests stage-blind scene processing after tracker confirmation and
before permanent-item creation. Never call scene processing, SAM, ReID, or the database from
the tracker. Pending retries for unusable observations are preserved but now throttled by
time, not retried every frame. The event layer distinguishes new identities (ADD), matched
removed items (RETURNED), and already-present associations (no event). Preserve the user's
inverted SAM keep-mask. See PROJECT_CONTEXT.md for reference collection and gallery
ownership. Migrate the embedding and prototype schema before live use; create_all() does not
alter existing tables.
