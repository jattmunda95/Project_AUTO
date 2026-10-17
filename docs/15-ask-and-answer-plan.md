# Ask-and-answer identity flow: build plan

Status: built and wired on 16 October 2026 (steps 1 to 8); not yet run on a live camera, and the
live database is not yet migrated. See the 16 October entry in `src/project_auto/TASKS.md` for
what exists, what was verified and what is left. Plan written 15 October 2026.

Changed later on 16 October 2026: check 1 below ("never ask on a bad view") was reversed after
the first live run, where a phone at 0.60 was held back as a bad view and never asked about. A
poor view now gets its question, marked as poor, and the answer decides identity without the
view ever being kept: a pick links without saving a picture, a new item is added only from the
track's first good view, and "not an item" stores no negative example. The diagram and step 1
below show the original plan.

Changed on 17 October 2026: the near-acceptance floor is now 0.50 (was 0.55) and the plausible
weight 0.35 (was 0.25), after a live run where a returning phone scored 0.5482 and 0.5374 and
never asked. The stored-question design in "Next phase" at the end of this file is agreed but
not built.

## The problem in one line

When the system is unsure whether an object is a known item, it guesses (makes a duplicate) or
quietly waits. It should ask the user, learn from the answer, and not ask again.

Real example: one phone, seen from its screen side and its back, became three items because
the two faces score only about 0.49 against each other (acceptance is 0.68).

## The flow, end to end

```
Object sits still, tracker confirms it            (exists)
        |
Background worker cuts out the object and         (exists)
scores it against every known item
        |
Worker returns ALL candidate scores + the crop    (CHANGE: today low-margin is
        |                                           flattened to "wait", weak is "new")
Identity policy looks at the scores and context   (NEW, step 1)
        |
   +----+------------+-------------+-------------+
   |                 |             |             |
 MATCH             NEW            WAIT          ASK
 (confident)    (clearly nothing  (bad view,    (unsure, good view)
   |             like it)         retry later)       |
   |                 |             |           Question registry   (NEW)
 existing path   existing path  existing path   stores the question
                                                     |
                                              Overlay shown in the window (NEW)
                                              user presses a key
                                                     |
                                              Answer applied by coordinator (NEW)
                                              - link to chosen item, or make new item
                                              - save the picture as a confirmed view
                                              - or remember "not an item"
```

The policy sits between "worker scored it" and "coordinator applies it". It runs on the main
thread, is pure arithmetic, and never touches the camera loop's speed.

## Building blocks, in build order

### Step 0. Replay old runs (no code risk)
Read the saved candidate scores in `logs/reid_runs/*/candidates.csv` and count how often each
trigger below would have fired and on which objects. Use this to pick starting thresholds.
Output: a small throwaway script in the scratchpad, not in the repo.

### Step 1. Identity policy (pure module, new file `events/identity_policy.py`)
Input: candidate scores (final, appearance, colour, shape, number of stored views), view
quality, which items are already claimed by other visible tracks, and the recent results for
this track.
Output: one of MATCH, NEW, WAIT, ASK, plus the reason and the candidate list to show.

Order of checks:
1. Bad view (blurry, poor mask, hand in the way) -> WAIT. Never ask on a bad view.
2. Remove impossible candidates (items already being tracked elsewhere in view).
3. Best score at or above acceptance and clear of rivals -> MATCH.
4. Best score below a "clearly new" floor and no near rivals -> NEW.
5. Otherwise ASK if any of: best score in the middle band; two or more candidates near the top;
   the score components disagree on the winner; the winner flipped across recent attempts.
6. Do not ask if a question about this track was already answered or skipped recently.

Limits: at most N open questions at once, a cooldown per track, a cooldown per item pair.
Settings live in a new YAML (`configs/identity_policy.yaml`), not in code.

### Step 2. Carry the evidence through (small edits to existing files)
- The scene processor stops turning low-margin into "wait" and weak into "new" by itself; it
  returns the raw outcome and the candidates (they are already attached).
- The worker result carries the candidates and the crop for the coordinator to use.
- The coordinator calls the policy on each resolve result.

### Step 3. Question registry (new file `events/question_registry.py`)
A question holds: id, track, crop, candidate items (with a thumbnail of each), time created,
expiry, state (open, answered, skipped, expired, cancelled). Rules: cancelled if the track
disappears; expires after a set time; one open question per track.
The track is held in a new "awaiting answer" state so it does not retry or create a duplicate.

### Step 4. Answer UI (edit to `app.py` and `utils/drawing.py`)
Overlay in the existing OpenCV window: the new object's picture beside up to 3 candidate
thumbnails. Keys: 1, 2, 3 pick a candidate; N it is a new item; X not an item; S skip.
Names (typing a label) are a later add-on, since the window has no text box.

### Step 5. Apply the answer (coordinator)
- Picked item: reuse the existing "matched item" path (return or associate), then save the
  picture as a confirmed view that skips the "looks too different" and "too similar" checks
  but still requires a good mask.
- New item: reuse the existing add path.
- Not an item: store the picture as a negative example, ignore that track.
- Skip: close the question, apply the cooldown, keep waiting. Never guess.

### Step 6. Storage (needs your approval before any schema change)
- Mark each stored view as automatic or confirmed (new column on the views table).
- New small table for "not an item" examples (picture embedding, time).
- Matching checks a new object against the negatives first, so a known hand does not trigger a
  question again.
- No automatic migration. A manual SQL script, like `docs/manual_schema_update.sql`, and a
  database backup first.

### Step 7. Logging and diagnostics
- One new log category for questions (asked, answered, expired), logged by the layer that owns
  it, matching the existing one-line-per-action rule.
- Record questions and answers in the diagnostics output so the policy can be replayed.

### Step 8. Tests (written with each step, not at the end)
- Policy: table of score cases to expected outcome, including the phone case, a two-way tie,
  a bad view, and "impossible candidate removed".
- Registry: expiry, cancel on track loss, one question per track, limits.
- Coordinator: answer links, new item, not-an-item, skip; no duplicate items created while
  waiting; stale answers ignored.
- Regression: a flickering detector label never changes whether a question is asked.

## Rules to keep (from the project guidelines)
- Never decide by detector label; hands are handled by appearance, behaviour or the
  "not an item" answer.
- Never run the cutting or scoring models on the camera thread.
- Never let a rejected or unconfirmed picture change an item's stored profile.
- Never silently change or recreate the database.
- One file changed per prompt, each explained and tested before the next.

## Decisions needed from you
1. What happens to a question nobody answers: keep waiting silently, or drop it after N seconds?
2. Is key-press enough for answers, or do you want clickable buttons?
3. Are item thumbnails available? Check whether stored views keep an image (the views table has
   an image-path column, but I have not confirmed crops are actually saved). If not, saving them
   becomes part of step 2.
4. Is "not an item" worth building in the first version, or defer it until hands cause trouble?

## Next phase: persistent, track-independent questions (agreed 17 October 2026, not built)

Why: questions currently die with their track (60 s timer; track ids reach ~400 in a session).
A good stored picture is enough for a person to judge the object, so the answer should not need
the object to still be in frame. The MVP channel stays the OpenCV overlay; a REST API and web page
come after the demo and are out of scope here.

- **Entry:** only a refused match with enough ask evidence and a GOOD view creates a question.
  A poor view keeps waiting for a good frame; if none ever comes, nothing is added.
- **Stored per question** (new table `pending_questions`, manual migration with backup): crop
  image, embedding, descriptors, model name, top-3 candidates, plain-English reasons, box,
  region, mask quality, bound track ids, status and times. Crops are written on the worker thread.
- **States:** queued, asked (the overlay has drawn it), answered, applied; or cancelled
  (candidate item gone) or superseded (a later confident match resolved it). No expiry; survives
  restarts.
- **Dedupe:** a query with embedding similarity of at least 0.9 to a pending question joins it by
  adding its track id. Two tracks visible in the same frame never merge. The first good crop is
  kept unless a later one is clearly better. At most 20 distinct questions are open; a join does
  not use cap space.
- **Overlay:** oldest first, no countdown, skip sends the question to the back.
- **Answers apply to the stored evidence** on the coordinator thread, once:

  | Answer | Always | Object still in view | Object gone |
  |---|---|---|---|
  | Pick item | add the view as a confirmed view | also bind the live track | no placement event |
  | New item | create the item with its reference | ADDED | ADDED then REMOVED |
  | Not an item | store a negative example | ignore the track | nothing more |

- **Build order** (one step per approval): 1) table, migration, startup guard, tests; 2) pure
  question service, tests, mutation check; 3) coordinator and overlay integration, `WAITING` log
  label, TASKS.md update.
- **Deferred:** web API and page, internet access, auth and pairing, items and locations
  endpoints, packaging, feature matching, vision-language checking, eviction at the 8-view cap,
  comparing new objects against negative examples.

## Tomorrow's order (original plan, 15 October)
1. Step 0 replay and pick starting thresholds.
2. Step 1: policy module, its YAML and its tests (the first file, nothing else touched).
3. Stop and review before touching the coordinator.
