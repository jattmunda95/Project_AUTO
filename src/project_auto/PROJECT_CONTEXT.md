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

`memory/models.py` defines three SQLAlchemy tables (existing databases need migration):

### `items`

One row represents one permanent physical object. `Item.id` is Project AUTO's durable
identity and must not be replaced by a tracker ID.
The nullable JSON `item_prototype` is a normalized mean of compatible reference vectors.
Its calculation exists in store.py but reference changes do not trigger it automatically.

### `item_embeddings`

Each row stores one reference vector, permanent item_id, model_name, optional crop path,
and creation timestamp. Deleting the item cascades to its references. Reference-saving and
gallery-loading methods are still pending, as is explicit prototype model/preprocessing
provenance. The prototype must not combine incompatible embedding spaces.

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
- See TASKS.md for recorded verification and the distinction between model tests, synthetic
  checks, and unverified real-image recognition.

## Next architecture step

The following architecture is agreed but deferred; the live loop still dispatches tracker
ADD directly to item creation. RETURNED interfaces exist but remain disconnected.

```text
Tracker confirmation -> app/coordinator identity request
-> scene_processor: detection crop -> SAM keep-mask -> DINOv2 -> gallery comparison
-> NEW / EXISTING / PENDING decision
-> coordinator + event layer: confirmation and current item status
-> ADD / RETURNED / association only / retry
```

The app chooses when to call scene processing. The scene processor does not inspect tracker
stages, and the tracker never calls scene processing, models, or persistence. Start comparison
after confirmation to avoid inference on transient candidates, but before permanent-item
creation. The current ADD signal denotes temporal confirmation, not proof of a new identity.

NEW creates an item with ADD. EXISTING associates its permanent ID: only REMOVED produces
RETURNED; PRESENT produces no add/return event. Define OCCLUDED restoration separately.
Unusable crops/masks produce PENDING, not NEW. Retain retry state after the one-time
confirmation, retry on visible frames, and cancel on retirement. Handle unresolved movement
and removal without writing against unknown identities. Reject conflicting simultaneous
claims on an item and prevent stale decisions from binding reused track IDs.

Scene processing will crop each detection and prompt SAM with crop-local coordinates.
The segmenter already accepts BGR images and boxes, returns frame-sized masks, and inverts
the selected raw mask to preserve the user's Colab convention. True means retain; replace
False pixels with the configured background colour. Real mask quality remains unverified.
Use identical preprocessing for reference and query embeddings.

The coordinator holds a gallery loaded by store.py and passes it to DB-agnostic ReID without
copying/reloading on each call. Refresh after reference/eligibility changes. The matcher
currently scores individual references using top-k means; prototype search is not implemented.

For a new permanent item, retain the initial usable embedding and collect five additional
quality crops at configured intervals. Save references through the store and update the
prototype after each save, ideally atomically. Six is a YAML target, not a requirement for
identity creation: keep partial collections and resume only after re-establishing identity.
The coordinator owns capture counts/timing; scene processing owns image preparation and
matching. Models load once. No per-frame persistence. Live activation requires schema
migration, standalone tests, and real-image evaluation first.

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
