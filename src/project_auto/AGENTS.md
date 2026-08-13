# Project AUTO — Codex Instructions

## Project goal

Build a local computer-vision system that remembers where physical objects
were last placed on a table.

The system must:

- capture a live overhead camera feed;
- detect objects using YOLO11s;
- track object instances across frames;
- preserve object identity during temporary occlusion;
- recognise meaningful relocation or placement events;
- store each unique object and its location history;
- save the clearest image associated with each relocation;
- answer queries such as "Where is my mug?"

## Current architecture

Pipeline:

Camera
→ YOLO11s detection
→ ByteTrack or BoT-SORT tracking
→ object identity association
→ placement-event detection
→ SQLite persistence
→ query interface

Target hardware:

- Lenovo Yoga Pro 7i
- Intel integrated graphics
- OpenVINO-optimised inference
- Fixed overhead camera
- Entirely local operation for the MVP

## Engineering conventions

- Python 3.10 or newer.
- Use type hints for public functions.
- Keep camera capture, inference, tracking and persistence separate.
- Configuration belongs in YAML rather than hard-coded constants.
- Use `pathlib.Path` for filesystem paths.
- Do not introduce a new dependency without explaining why.
- Prefer small, testable components over one large application file.
- Never treat a tracker ID alone as permanent object identity.
- Do not write a database observation for every video frame.
- Store state changes and meaningful events instead.

## Before changing code

1. Read `PROJECT_CONTEXT.md`.
2. Read `TASKS.md`.
3. Inspect the relevant existing files.
4. Explain any architectural change before implementing it.

## Verification

After modifying code:

- run the relevant tests;
- run formatting and lint checks when configured;
- report what was changed and what remains unverified.

## Current priority

Follow the `Current task` section in `TASKS.md`.