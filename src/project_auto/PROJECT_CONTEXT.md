# Project AUTO Context

## Problem

Objects on a table may disappear because they are occluded or removed. Project AUTO must
remember their last meaningful location rather than merely report what is visible in the
current frame.

## Implemented pipeline

```text
Camera -> YOLO11s/OpenVINO detection -> structured detections -> debug display
```

The camera and detector are working on the target laptop. Tracking and memory are not yet
connected to this live pipeline.

## Persistent-memory foundation

`memory/models.py` defines two SQLAlchemy tables:

### `items`

One row represents one permanent physical object. `Item.id` is Project AUTO's durable
identity and must not be replaced by a tracker ID.

Item states are:

- `present`: currently visible;
- `occluded`: temporarily not visible but still believed to be present;
- `removed`: believed to have left the monitored area.

### `item_events`

One row represents a meaningful change such as added, removed, returned, moved, placed, or
status changed. Events belong permanently to `item_id`.

`source_track_id` is optional diagnostic metadata copied from ByteTrack or BoT-SORT. It is
temporary and session-specific. One permanent item can have different tracker IDs over its
lifetime.

Evidence is stored as image/video file paths on events, not as binary database data. The
database records meaningful events, never individual frames.

## Database behavior

- SQLite is the MVP database.
- Foreign-key enforcement is enabled for every store connection.
- Deleting an item cascades to its events.
- Enum values are constrained lowercase strings in SQLite.
- Item history is returned chronologically.
- Present-item queries include `present` and `occluded`, but exclude `removed`.
- Store status operations update the item and insert the associated event atomically.
- Fourteen focused in-memory SQLite tests currently pass.

## Next architecture step

Implement `memory/state_machine.py` as decision logic separate from persistence:

```text
observation/state input
        -> state machine decides whether a meaningful transition occurred
        -> database store persists the resulting state and event
```

The state machine must suppress repeated observations that do not represent a state change.
It should not write directly for every video frame.

Expected transitions include:

- `present -> occluded` produces `status_changed`;
- `occluded -> present` produces `status_changed`;
- `present/occluded -> removed` produces `removed`;
- `removed -> present` produces `returned`;
- movement while visible produces `moved` without changing status.

## Deferred work

- Stable in-memory tracking state and tracker integration.
- Permanent identity association across tracker-ID changes.
- Placement and relocation-event detection.
- High-quality evidence crop selection.
- Object-location queries.
- Embeddings or recognition improvements.
- Schema migrations beyond the initial local MVP.
