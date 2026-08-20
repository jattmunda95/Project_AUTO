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

The detector invokes Ultralytics BoT-SORT and exposes optional temporary track IDs. The
tracker, state machine, event engine, and database store now implement and verify the initial
`ADD` workflow in isolation, but it is not connected to the live loop. Track-to-item bindings
are provisional session mappings, not permanent identity recognition. Do not describe live
persistence or reliable re-identification as implemented.

## Target hardware

- Lenovo Yoga Pro 7i
- Intel integrated graphics
- OpenVINO-optimized inference
- Fixed overhead camera
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

Follow the `Current task` section in `TASKS.md`. Connect the verified initial `ADD` workflow
to the live loop one approved code chunk at a time. Add database-path configuration before
editing `main.py`; then construct the store, tracker, and event engine once outside the frame
loop and process only meaningful tracker signals. Do not add per-frame database writes.
