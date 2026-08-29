"""Stage 2's central claim is determinism. These lock it in.

Runs against the FastF1 cache in data/cache, so it is offline after the first fetch.

    python -m pytest tests/test_replay.py -q
"""

from __future__ import annotations

import re

import pytest

from engine.state import RaceState
from ingest.events import RACE_CONTROL, LAP_COMPLETED
from ingest.replay import (
    build_events, clock_skew, load_session, local_utc_offset, replay,
)
import pandas as pd

YEAR, ROUND = 2026, 12


@pytest.fixture(scope="module")
def session():
    return load_session(YEAR, ROUND, "R")


@pytest.fixture(scope="module")
def events(session):
    return build_events(session)


def fold(events) -> RaceState:
    state = RaceState()
    for e in events:
        state.apply(e)
    return state


def test_stream_is_ordered_by_event_time(events):
    times = [e.t for e in events]
    assert times == sorted(times)


def test_build_is_deterministic(session):
    assert build_events(session) == build_events(session)


def test_fold_is_deterministic(events):
    a, b = fold(events), fold(events)
    assert a.lap == b.lap
    assert [c.car for c in a.order()] == [c.car for c in b.order()]


def test_partial_fold_gives_the_state_at_that_moment(events):
    """Stopping the fold early must reconstruct a real earlier state, not a broken one.

    This is what "the state when the incident happened" depends on in Stage 3.
    """
    cut = len(events) // 2
    early, final = fold(events[:cut]), fold(events)
    assert 0 < early.lap < final.lap
    assert early.order()[0].position == 1


def test_apply_does_not_depend_on_anything_outside_the_stream(events):
    """A reducer that reads a clock or a network is not replayable. Same input, same
    output, whatever order the two runs happen in."""
    forward = fold(events)
    again = fold(events)
    assert forward.summary() == again.summary()


def test_race_clock_matches_timestamps_the_messages_carry(session, events):
    """Independent ground truth: race control prints "TIMED AT hh:mm:ss" into some
    messages. Our reconstructed clock must reproduce it."""
    start_local = (session.date
                   + pd.Timedelta(seconds=clock_skew(session))
                   + pd.Timedelta(seconds=local_utc_offset(session)))
    base = (start_local.hour * 3600 + start_local.minute * 60
            + start_local.second + start_local.microsecond / 1e6)

    stamped = re.compile(r"TIMED AT (\d{2}):(\d{2}):(\d{2})")
    errors = []
    for e in events:
        if e.kind != RACE_CONTROL:
            continue
        if m := stamped.search(e["message"]):
            want = int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3])
            err = (base + e.t - want) % 86400
            errors.append(err - 86400 if err > 43200 else err)

    assert len(errors) > 50, "expected plenty of self-timestamped messages"
    # The FIA truncates its stamp to whole seconds, so we read up to 1s late, never early.
    assert all(-0.5 <= x <= 1.5 for x in errors), (min(errors), max(errors))


def test_speed_zero_emits_everything_without_sleeping(events):
    assert list(replay(events[:200], speed=0)) == events[:200]


def test_lap_numbers_advance_monotonically_for_the_leader(events):
    state, seen = RaceState(), []
    for e in events:
        state.apply(e)
        seen.append(state.lap)
    assert seen == sorted(seen), "leader lap counter went backwards"
    assert state.lap >= 50


def test_gaps_are_measured_between_cars_on_the_same_lap(events):
    state = fold(events)
    leader = state.order()[0]
    assert state.gap_to_leader(leader.car) == pytest.approx(0.0, abs=1e-6)
    seconds = [state.gap_to_leader(c.car) for c in state.order()[1:6]]
    assert all(g is not None and g > 0 for g in seconds), seconds


def test_every_lap_event_carries_a_lap_number(events):
    laps = [e for e in events if e.kind == LAP_COMPLETED]
    assert laps and all(e.lap is not None for e in laps)
