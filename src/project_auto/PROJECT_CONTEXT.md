# Project AUTO Context

## Problem

Objects on a table may disappear because they are occluded or removed.
The system must remember their last meaningful location rather than merely
reporting what is visible in the current frame.

## Data model

### Object instance

One record for each distinct physical object, such as one particular mug.

Suggested information:

- internal object ID;
- object class;
- user-assigned name;
- appearance embedding;
- representative image;
- first-seen and last-seen timestamps;
- current state.

### Location event

A new record is created when an object is meaningfully placed, relocated,
removed or rediscovered.

Suggested information:

- object ID;
- event type;
- timestamp;
- table region or coordinates;
- tracking confidence;
- identity confidence;
- clearest event image.

## Important decisions

- Do not create a permanent object for every detection.
- Tracker IDs are temporary and session-specific.
- Database writes represent events, not individual frames.
- Keep multiple event images, with one current representative image per object.
- Start with SQLite for the local MVP.
- Optimise YOLO11s using OpenVINO on the target laptop.