# Project AUTO Context

## Problem

Objects on a table may disappear because they are occluded or removed. Project AUTO must
remember their last meaningful location rather than merely report what is visible in the
current frame.

## Implemented pipeline

```text
Camera
-> YOLO11s/OpenVINO + BoT-SORT
-> structured detections
-> lifecycle tracker
-> state decisions
-> event engine
-> SQLite store
-> debug display
```

The live loop now calls the lifecycle and persistence workflow. The implemented meaningful
signals are:

```text
list[Detection]
-> candidate confirmation after 2 seconds with at most 15 cumulative missing frames
-> TrackSignal.ADD
-> StateDecision(PRESENT, ADDED)
-> EventEngine
-> atomic Item + ItemEvent persistence

stable bbox exits centered 1.2x placement buffer
-> TrackStatus.MOVING
-> bbox centre remains within 5 pixels for 1 second
-> timed TrackSignal.MOVED with source and destination boxes
-> StateDecision(PRESENT, MOVED)
-> permanent item_id resolution
-> atomic movement event persistence

stable or moving track absent for 2 seconds
-> TrackSignal.REMOVE
-> StateDecision(REMOVED, REMOVED)
-> EventEngine
-> atomic item-status update + removal event
```

Candidates tolerate up to 15 cumulative missing frames within each two-second confirmation
attempt. The 16th miss retires the attempt; the next observation starts a fresh attempt.
Detections without a BoT-SORT ID are ignored. A stable or moving track reappearing with the
same ID inside the two-second window cancels removal. A returning moving track must complete a
fresh visible stop-confirmation window. Removal does not delete item history or the event
engine's provisional association.

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
- Events now expose nullable `started_at` and `finished_at` timestamps for duration-based
  events. Store event-writing paths accept complete timezone-aware intervals and reject
  partial, naive, or reversed intervals.
- `record_movement()` can persist a valid movement interval while atomically updating the
  item's `last_seen_at`.
- Movement events expose nullable JSON `source_box` and `destination_box` coordinates while
  retaining optional source/destination region fields for future region resolution.
- Thirty-eight tests currently pass across detection, tracking, event coordination, and
  persistence.

## Next architecture step

Finish permanent event-engine and persistence tests for timed movement metadata, bbox storage,
invalid movement signals, and permanent item association. The tracker movement suite and
state-machine movement test are already permanent. The local database was recreated and the
complete `ADD` then `MOVED` live demo succeeded on 22 August 2026. `MOVING` remains a temporary
tracker state; the permanent item remains `PRESENT`.

After MOVED, add `RETURNED` interfaces as boilerplate for future ReID. A return means that
associative memory has resolved a new observation to an existing removed `item_id`; it must
never be inferred from a temporary BoT-SORT ID alone. Until ReID exists, the RETURNED path
should be defined but not activated by tracker behavior.

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
- Connecting RETURNED boilerplate to associative-memory/ReID output.
- Occlusion and general status-change signal processing.
- High-quality evidence crop selection.
- Object-location queries.
- Schema migrations beyond the initial local MVP.
