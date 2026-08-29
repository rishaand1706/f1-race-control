"""Replay a finished session as a live event stream.

Sessions happen on race weekends. Development happens every day. So the primary event
source is a recorded one, replayed against a wall clock at a chosen speed, emitting the
exact `Event` type that live mode will emit in Stage 6. Everything downstream is written
against replay and gets live for free.

This is not a testing convenience bolted on afterwards. A deterministic, re-runnable
stream is the only way to debug a streaming system at all: without it, every bug needs a
race to reproduce, and no two runs are the same.

Usage:
    python -m ingest.replay --round 12 --speed 300
    python -m ingest.replay --round 12 --speed 0 --only race_control   # no sleeping
    python -m ingest.replay --round 12 --from-lap 30 --speed 120
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
import time
from typing import Iterator

import fastf1
import pandas as pd

import config
from ingest.events import (
    Event, RACE_CONTROL, LAP_COMPLETED, TRACK_STATUS, SESSION_STATUS,
    TRACK_STATUS_LABELS,
)

# Events that appear in BOTH feeds, used to measure the skew between their clocks.
# Each entry: (regex matched against race control messages, series name, matching value).
CLOCK_ANCHORS = [
    (r"CHEQUERED FLAG", "session_status", "Finished"),
    (r"RED FLAG", "session_status", "Aborted"),
    (r"RED FLAG", "track_status", "5"),
    (r"SAFETY CAR DEPLOYED", "track_status", "4"),
]


def load_session(year: int, rnd: int, name: str = "R"):
    fastf1.Cache.enable_cache(str(config.CACHE))
    s = fastf1.get_session(year, rnd, name)
    # Telemetry is deliberately not loaded: it is gigabytes per season and nothing here
    # needs it. Lap and message data is a few MB.
    s.load(laps=True, telemetry=False, weather=False, messages=True)
    return s


def clock_skew(session) -> float:
    """Seconds by which the race control clock leads the timing clock.

    The two feeds do not agree, and not by a rounding error. Race control messages are
    absolute UTC timestamps; laps, track status and session status are timedeltas on the
    session's own timeline whose origin is only implied. Deriving that origin the obvious
    way - `session.date - session.session_start_time` - lands **three and a half minutes**
    early, because `session.date` is the *scheduled* start and the race did not start then.

    Rather than trust either clock, measure the offset from events that appear in both.
    On the 2026 Dutch GP three independent anchors agree to within 0.7 seconds:

        chequered flag  <-> session status "Finished"    3:27.9
        red flag        <-> session status "Aborted"     3:28.6
        red flag        <-> track status 5               3:28.4

    The median is used, so one mismatched anchor cannot drag the alignment. If no anchor
    is found - a clean session with no red flag or safety car may have only the chequered
    flag - the caller is warned rather than silently given a misaligned stream.

    The skew is not a constant to hardcode: it is 208.4s at the Dutch GP and 197.2s at
    the Austrian, because it is just how late that particular session actually started.
    It has to be measured per session.
    """
    naive_t0 = session.date - session.session_start_time
    rcm = session.race_control_messages

    offsets: list[float] = []
    for pattern, series_name, value in CLOCK_ANCHORS:
        hits = rcm[rcm["Message"].str.contains(pattern, case=False, na=False, regex=True)]
        if hits.empty:
            continue
        series = getattr(session, series_name, None)
        if series is None or series.empty:
            continue
        rows = series[series["Status"].astype(str) == value]
        if rows.empty:
            continue
        msg_t = (hits["Time"].iloc[0] - naive_t0).total_seconds()
        offsets.append(msg_t - rows["Time"].iloc[0].total_seconds())

    if not offsets:
        print("  ! no shared anchor events found; message timing may be misaligned",
              file=sys.stderr)
        return 0.0
    return statistics.median(offsets)


def local_utc_offset(session) -> float:
    """Seconds the circuit's local time leads UTC."""
    ev = session.event
    loc = ev.get_session_date(session.name, utc=False)
    utc = ev.get_session_date(session.name, utc=True)
    loc = loc.tz_localize(None) if loc.tzinfo else loc
    utc = utc.tz_localize(None) if utc.tzinfo else utc
    return (loc - utc).total_seconds()


def verify_clock(session, events: list[Event]) -> None:
    """Check the reconstructed race clock against ground truth inside the messages.

    Race control prints its own timestamp into some messages ("WAVED BLUE FLAG FOR CAR 55
    (SAI) TIMED AT 16:16:47", circuit local time). That is an independent witness to when
    the event happened, published by the same body, and it never touches either of the
    clocks we reconciled. If our race clock reproduces it, the alignment is right.

    Dutch GP: 111 messages, error +0.0 to +1.0s. Austrian GP: 116 messages, same range.
    The error is uniform across [0, 1) rather than scattered, which is the signature of
    the FIA truncating its stamp to whole seconds - so the alignment is exact and we read
    half a second late on average, not drifting.
    """
    start_local = (session.date
                   + pd.Timedelta(seconds=clock_skew(session))
                   + pd.Timedelta(seconds=local_utc_offset(session)))
    base = (start_local.hour * 3600 + start_local.minute * 60
            + start_local.second + start_local.microsecond / 1e6)

    stamped_re = re.compile(r"TIMED AT (\d{2}):(\d{2}):(\d{2})")
    errors: list[float] = []
    for e in events:
        if e.kind != RACE_CONTROL:
            continue
        m = stamped_re.search(e["message"])
        if not m:
            continue
        stamped = int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3])
        err = (base + e.t - stamped) % 86400
        errors.append(err - 86400 if err > 43200 else err)

    if not errors:
        print("  no self-timestamped messages in this session; clock unverified")
        return
    errors.sort()
    print(f"clock check against {len(errors)} self-timestamped messages:")
    print(f"  median {statistics.median(errors):+.2f}s   "
          f"mean {statistics.mean(errors):+.2f}s   "
          f"stdev {statistics.pstdev(errors):.2f}s   "
          f"range {errors[0]:+.1f}s .. {errors[-1]:+.1f}s")


def build_events(session) -> list[Event]:
    """Merge every feed onto one clock, rebased so t=0 is the session start."""
    naive_t0 = session.date - session.session_start_time
    skew = clock_skew(session)
    # Rebase so the race clock reads from zero at the green light. Pre-race messages
    # (pit exit open, formation lap) legitimately land at negative t.
    origin = session.session_start_time.total_seconds()

    events: list[Event] = []

    for _, m in session.race_control_messages.iterrows():
        t = (m["Time"] - naive_t0).total_seconds() - skew - origin
        events.append(Event(
            t=t, kind=RACE_CONTROL,
            lap=int(m["Lap"]) if pd.notna(m["Lap"]) else None,
            payload={
                "category": m["Category"],
                "message": str(m["Message"]).strip(),
                "flag": m["Flag"] if pd.notna(m["Flag"]) else None,
                "scope": m["Scope"] if pd.notna(m["Scope"]) else None,
                "sector": int(m["Sector"]) if pd.notna(m["Sector"]) else None,
                "car": int(m["RacingNumber"]) if pd.notna(m["RacingNumber"]) else None,
            },
        ))

    for _, lap in session.laps.iterrows():
        if pd.isna(lap["Time"]) or pd.isna(lap["LapNumber"]):
            continue
        events.append(Event(
            t=lap["Time"].total_seconds() - origin, kind=LAP_COMPLETED,
            lap=int(lap["LapNumber"]),
            payload={
                "car": int(lap["DriverNumber"]),
                "driver": lap["Driver"],
                "team": lap["Team"],
                "position": int(lap["Position"]) if pd.notna(lap["Position"]) else None,
                "lap_time": (lap["LapTime"].total_seconds()
                             if pd.notna(lap["LapTime"]) else None),
                "compound": lap["Compound"] if pd.notna(lap["Compound"]) else None,
                "tyre_life": (int(lap["TyreLife"]) if pd.notna(lap["TyreLife"]) else None),
                "stint": int(lap["Stint"]) if pd.notna(lap["Stint"]) else None,
                "pit_in": pd.notna(lap["PitInTime"]),
                "pit_out": pd.notna(lap["PitOutTime"]),
            },
        ))

    for _, r in session.track_status.iterrows():
        events.append(Event(
            t=r["Time"].total_seconds() - origin, kind=TRACK_STATUS,
            payload={"code": str(r["Status"]),
                     "label": TRACK_STATUS_LABELS.get(str(r["Status"]), str(r["Status"]))},
        ))

    for _, r in session.session_status.iterrows():
        events.append(Event(
            t=r["Time"].total_seconds() - origin, kind=SESSION_STATUS,
            payload={"status": r["Status"]},
        ))

    # Sort by event time. Two events can share a timestamp, so the sort must be stable
    # and the tiebreak explicit, or replay order changes between runs and determinism -
    # the entire reason replay exists - quietly disappears.
    events.sort(key=lambda e: (e.t, e.kind, str(e.payload.get("car") or "")))
    return events


def replay(events: list[Event], speed: float = 20.0) -> Iterator[Event]:
    """Emit events paced against a wall clock, `speed` times faster than real time.

    speed=0 emits everything immediately, which is what tests and batch runs want.

    Each event is scheduled against a single fixed origin rather than by sleeping for the
    gap since the previous one. Sleeping per-gap accumulates the overshoot of every sleep
    and every bit of processing time, so a long replay drifts steadily late. Anchoring to
    the origin makes any individual late event self-correcting.
    """
    if not events:
        return
    t_start = events[0].t
    wall_start = time.monotonic()
    for e in events:
        if speed > 0:
            delay = (e.t - t_start) / speed - (time.monotonic() - wall_start)
            if delay > 0:
                time.sleep(delay)
        yield e


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--year", type=int, default=2026)
    p.add_argument("--round", type=int, default=12)
    p.add_argument("--session", default="R")
    p.add_argument("--speed", type=float, default=300.0,
                   help="wall-clock multiplier; 0 = no sleeping")
    p.add_argument("--from-lap", type=int, default=None)
    p.add_argument("--only", default=None,
                   help="comma-separated event kinds to print")
    p.add_argument("--max", type=int, default=None, help="stop after N printed events")
    p.add_argument("--verify-clock", action="store_true",
                   help="check the race clock against message-embedded timestamps, then exit")
    args = p.parse_args()

    session = load_session(args.year, args.round, args.session)
    events = build_events(session)
    skew = clock_skew(session)

    if args.verify_clock:
        print(f"\n{session.event['EventName']} {args.year} - {args.session}")
        print(f"clock skew between feeds: {skew:.1f}s\n")
        verify_clock(session, events)
        return

    from engine.state import RaceState
    state = RaceState()

    kinds = set(args.only.split(",")) if args.only else None
    print(f"\n{session.event['EventName']} {args.year} - {args.session}")
    print(f"{len(events)} events   clock skew corrected by {skew:.1f}s   speed {args.speed}x\n")

    printed = 0
    for e in replay(events, speed=args.speed):
        # The reducer sees every event, including the ones we are not printing: state is
        # a fold over the whole stream, and filtering the fold would corrupt it.
        state.apply(e)

        if kinds and e.kind not in kinds:
            continue
        if args.from_lap and (state.lap or 0) < args.from_lap:
            continue

        marker = {RACE_CONTROL: "***", SESSION_STATUS: " ##", TRACK_STATUS: "  ~"}.get(e.kind, "   ")
        print(f"{marker} [{e.clock}] L{state.lap or 0:<2} {state.track_status:<11} {e.describe()}")

        printed += 1
        if args.max and printed >= args.max:
            break

    print(f"\n{state.summary()}")


if __name__ == "__main__":
    main()
