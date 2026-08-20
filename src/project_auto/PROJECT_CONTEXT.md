# Project AUTO Context

## Problem

Objects on a table may disappear because they are occluded or removed. Project AUTO must
remember their last meaningful location rather than merely report what is visible in the
current frame.

## Implemented pipeline

```text
Camera -> YOLO11s/OpenVINO + BoT-SORT -> structured detections -> debug display
```

The live loop receives optional temporary BoT-SORT IDs, but it does not yet call the lifecycle
tracker or persistence workflow.

The initial `ADD` workflow is implemented and tested separately:

```text
list[Detection]
-> candidate confirmation after 30 sightings
-> TrackSignal.ADD
-> StateDecision(PRESENT, ADDED)
-> EventEngine
-> atomic Item + ItemEvent persistence
```

Candidates tolerate up to 10 consecutive missed frames and expire on the 11th. Detections
without a BoT-SORT ID are ignored by lifecycle processing.

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

One row represents a meaningful change: added, returned, removed, moved, or status changed.
Events belong permanently to `item_id`.

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
- Initial item creation and its `ADDED` event can be committed atomically.
- Twenty-four tests currently pass across detection, tracking, event coordination, and
  persistence.

## Next architecture step

Connect the isolated components to the live loop without collapsing their responsibilities:

```text
frame -> detections -> tracker signals -> state decisions -> event engine -> store
```

The store, tracker, and event engine must be created once before frame processing. Ordinary
frames should remain in memory and produce no database write. Database configuration belongs
in YAML.

Expected transitions include:

- `present -> occluded` produces `status_changed`;
- `occluded -> present` produces `status_changed`;
- `present/occluded -> removed` produces `removed`;
- `removed -> present` produces `returned`;
- movement while visible produces `moved` without changing status.

## Deferred work

- Reliable permanent identity association across tracker-ID changes; the event engine's
  current track-to-item dictionary is only a provisional session binding.
- Recognition, appearance embeddings, and trajectory checks for ID-switch recovery.
- Missing, occluded, reappeared, removed, returned, and movement signal processing.
- Placement and relocation-event detection.
- High-quality evidence crop selection.
- Object-location queries.
- Schema migrations beyond the initial local MVP.
