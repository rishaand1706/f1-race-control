# 05. Streams, event time, and state as a fold

## Batch and stream are different problems

Stage 1 was batch: a fixed corpus, processed once, in whatever order was convenient, with
the whole thing in memory. Everything from here is a stream: events arrive one at a time,
you cannot see the future, and the work has to happen as they land.

Three things change, and each one is a class of bug:

- **You cannot sort the input.** A batch job sorts and then processes. A stream processes
  and then discovers the thing it needed arrived late.
- **State has to be maintained incrementally.** No `GROUP BY` over the whole race. You
  hold the current picture and update it per event.
- **Time is now two different things**, and conflating them is the classic mistake.

## Event time vs processing time

**Event time** is when the thing happened. **Processing time** is when your code saw it.

They are never equal, and the gap is not constant: network delay, buffering, a recorder
that starts two minutes late, an upstream system batching its writes. Systems that reason
in processing time produce results that change depending on how fast the machine was,
which means they cannot be tested and cannot be reproduced.

Everything here is placed on **event time**: `Event.t`, seconds since the session started.
Processing time appears in exactly one place, the wall-clock pacing in `replay()`, and
nothing downstream can see it.

## The trap: two feeds from one source, disagreeing by three and a half minutes

FastF1 hands back the same session on two incompatible clocks:

| feed | clock |
|---|---|
| `race_control_messages` | absolute UTC timestamps (`2026-08-23 12:12:06`) |
| `laps`, `track_status`, `session_status` | timedeltas on the session's own timeline |

To merge them you need the session timeline's origin. The obvious derivation is
`session.date - session.session_start_time`, and it is **wrong by 208 seconds**, because
`session.date` is the *scheduled* start and the race did not start then.

Nothing crashes. Messages land three and a half minutes from the lap data they describe,
which at racing speed is about two laps. A safety car explanation would cite the state
from two laps earlier and look entirely plausible.

**The fix is not to trust either clock.** It is to find events that appear in *both* feeds
and measure the offset between them. On the 2026 Dutch GP:

| anchor | implied skew |
|---|---|
| chequered flag ↔ session status `Finished` | 3:27.9 |
| red flag ↔ session status `Aborted` | 3:28.6 |
| red flag ↔ track status `5` | 3:28.4 |

Three independent anchors agreeing within 0.7 seconds. Take the median, so one bad anchor
cannot drag the alignment. `ingest/replay.py::clock_skew`.

And it is **not a constant to hardcode**: 208.4s at the Dutch GP, 197.2s at the Austrian.
It is simply how late each session actually started.

## Verifying a clock you had to guess at

Deriving the offset is one thing; knowing it is right is another. The witness turned out
to be inside the data. Race control prints its own timestamp into some messages:

```
WAVED BLUE FLAG FOR CAR 55 (SAI) TIMED AT 16:16:47
```

That is circuit local time, published by the FIA, and it touches neither clock we
reconciled. So it is genuine independent ground truth:

| session | messages | error |
|---|---|---|
| Dutch GP | 111 | +0.0 to +1.0s |
| Austrian GP | 116 | +0.0 to +1.0s |

The error is uniform across [0, 1) rather than scattered around zero, which is the
signature of the FIA truncating its stamp to whole seconds. We read half a second late on
average and never drift. `python -m ingest.replay --round 12 --verify-clock`.

**The transferable habit:** when you have had to *infer* something structural, go looking
for a field that lets you check it. Data that describes itself is common, and an inferred
alignment nobody verified is the kind of bug that survives to production.

## State as a fold

There is no "current state" anywhere upstream. There is a sequence of things that
happened, and the state is what you get by applying all of them in order:

```python
state = reduce(apply, events, RaceState())
```

That is **event sourcing**, and it buys three things this project needs:

**Reproducibility.** The same events always produce the same state. A bug at lap 54 is
reachable in seconds instead of by waiting for a race.

**Time travel.** Stop the fold early and you have the state as it was at that moment -
which is exactly what "what was the situation when this incident happened" requires. Stage
3 builds its retrieval query from it.

**One code path.** Live and replay cannot drift apart, because both feed the same reducer
the same `Event` type.

All three depend on one rule: **`apply` must be a pure function of the event and the
current state.** No clock reads, no network calls, no randomness. The moment it reads
`datetime.now()`, replay stops reproducing and every guarantee above is gone.

`engine/state.py`. Tests in `tests/test_replay.py`.

## Two details that quietly destroy determinism

**1. An unstable sort.** Events share timestamps constantly - twenty cars cross the line
inside the same second. `sorted()` is stable in Python, but the *key* must break ties
deterministically or the order depends on the order the source happened to yield rows in.
The key here is `(t, kind, car)`.

**2. Sleeping for the gap instead of scheduling against an origin.** The tempting loop:

```python
for e in events:                      # WRONG
    time.sleep((e.t - prev.t) / speed)
```

Every `sleep` overshoots slightly, and every overshoot is permanent. Over a three-hour
replay the stream drifts steadily late. Scheduling each event against one fixed origin
makes a late event self-correcting:

```python
delay = (e.t - t_start) / speed - (time.monotonic() - wall_start)
```

Same idea as a game loop or an audio callback: absolute schedule, not relative sleeps.

## A smaller one worth remembering

Pre-race messages legitimately have negative `t` - the pit exit opens before the race
starts. Rendering them went through `s // 3600`, and Python floors on negative division,
so -3518s printed as `-1:01:22` instead of `-0:58:38`. Take the sign out before doing
arithmetic on time.

## Interview drill

**Q: Event time versus processing time - why does the distinction matter?**
Event time is when it happened; processing time is when you saw it. Windowed aggregations,
ordering and any replay have to use event time, or results depend on machine speed and
network conditions and cannot be reproduced. Processing time is only for operational
concerns: how long to wait for stragglers, when to time out.

**Q: What is a watermark?**
A stream processor's assertion that no event older than time T is still expected, so
windows up to T can be closed. It is the mechanism for deciding when to stop waiting for
late data. We do not need one: replay has the whole file, and live is append-only with no
out-of-order delivery within a feed.

**Q: Why event sourcing rather than just keeping the current state?**
Because the event log is the source of truth and the state is a derived view. You can
rebuild state after a bug fix, reconstruct the state at any past moment, and test the
reducer as a pure function. The cost is that every read of history is a fold, so you add
snapshots when that gets expensive. At twenty cars it never does.

**Q: How do you test a streaming system?**
Record a real stream and replay it deterministically. That means one normalised event type
both sources emit, a reducer that is pure, and a tie-broken sort. Then a failure is a
fixture, not a race weekend.

**Q: Two feeds from the same system disagree on time. What now?**
Do not trust either. Find events that appear in both, measure the offset across several,
take the median so one mismatch cannot skew it, and then verify the result against
something that touches neither clock. Hardcoding the offset you measured once is the
mistake: here it varied by 11 seconds between two races.
