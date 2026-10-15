"""Stage 4's claims, locked in.

Two pieces of Stage 4 are pure logic and fail silently, so they get tests rather than a
demo:

  - the streaming citation binder, because a split marker or a mis-closed gap produces
    text that reads fine and is subtly wrong, and nobody watching a stream would notice
  - the prose queue's shedding rules, because "which cards never get an explanation" is
    a policy, and a policy that quietly changes is a behaviour change nobody asked for

No Ollama, no index, no FastF1.

    python -m pytest tests/test_server.py -q
"""

from __future__ import annotations

import random

import pytest

from engine.explain import StreamBinder, bind_citations
from server.scheduler import STALE, SUPERSEDED, CardQueue, Job, simulate


# --- streaming citation binding ---------------------------------------------

SOURCES = [object()] * 4     # the binder only needs to know how many there are

TEXTS = [
    "Car 23 ignored an instruction [1]. The guideline sets a reprimand [2].",
    "Track limits at turn 4 [3][1], see also [7].",
    "Unsafe release [1] [9] . The team was fined [2].",
    "  Leading spaces [4]  and   doubled gaps [12] ; kept honest [2]  ",
    "A bracket that is not a marker [see 2] and one left open [3",
    "Line one [1].\nLine two [8].\n\nLine three [2]",
    "[1] starts with a marker and ends with one [2]",
    "",
]


def _stream(text: str, cuts: list[int]) -> str:
    binder = StreamBinder(len(SOURCES))
    pieces, prev = [], 0
    for c in cuts + [len(text)]:
        pieces.append(binder.feed(text[prev:c]))
        prev = c
    pieces.append(binder.flush())
    return "".join(pieces)


@pytest.mark.parametrize("text", TEXTS)
def test_every_single_split_matches_the_batch_binder(text):
    """A marker split across two chunks is the case this class exists for."""
    want = bind_citations(text, SOURCES, renumber=False)[0]
    for cut in range(len(text) + 1):
        assert _stream(text, [cut]) == want, f"split at {cut}: {text[:cut]!r} | {text[cut:]!r}"


@pytest.mark.parametrize("text", TEXTS)
def test_token_sized_chunks_match_the_batch_binder(text):
    want = bind_citations(text, SOURCES, renumber=False)[0]
    rng = random.Random(0)
    for _ in range(200):
        cuts = sorted(rng.sample(range(len(text) + 1), k=min(len(text), rng.randint(1, 12))))
        assert _stream(text, cuts) == want


def test_one_character_at_a_time():
    for text in TEXTS:
        assert _stream(text, list(range(1, len(text)))) == \
            bind_citations(text, SOURCES, renumber=False)[0]


def test_stream_records_what_was_used_and_what_was_invented():
    b = StreamBinder(4)
    for piece in ["Claim [", "2]. Other [", "7] and [1", "] again [2]."]:
        b.feed(piece)
    b.flush()
    assert b.used == [2, 1]
    assert b.dropped == [7]


def test_an_invented_marker_never_reaches_the_screen_even_mid_stream():
    """Held back until it closes, then deleted: no chunk ever shows '[7'."""
    b = StreamBinder(4)
    shown = [b.feed(p) for p in ["Fact one [", "7", "]", " fact two [1]."]] + [b.flush()]
    assert not any("7" in s for s in shown)
    assert "".join(shown) == "Fact one fact two [1]."


def test_renumber_false_keeps_the_numbers_already_on_screen():
    text = "B [3] then A [1]."
    assert bind_citations(text, SOURCES)[0] == "B [1] then A [2]."
    assert bind_citations(text, SOURCES, renumber=False)[0] == "B [3] then A [1]."


# --- the prose queue --------------------------------------------------------

def job(i, t, stage="noted", offence="causing a collision", cars=(1, 44)):
    return Job(id=i, t=t, stage=stage, offence=offence, cars=frozenset(cars))


def test_a_later_message_about_the_same_incident_supersedes_the_queued_one():
    q = CardQueue(max_age=None)
    assert q.put(job(1, 10)) == []
    shed = q.put(job(2, 50, stage="under_investigation", cars=(1,)))
    assert [(j.id, why) for j, why in shed] == [(1, SUPERSEDED)]
    assert [j.id for j in q.jobs] == [2]


def test_an_older_message_arriving_late_never_supersedes_a_newer_one():
    q = CardQueue(max_age=None)
    q.put(job(2, 50, stage="under_investigation"))
    shed = q.put(job(1, 10))
    assert [(j.id, why) for j, why in shed] == [(1, SUPERSEDED)]
    assert [j.id for j in q.jobs] == [2]


def test_different_offence_or_disjoint_cars_is_a_different_incident():
    q = CardQueue(max_age=None)
    q.put(job(1, 10))
    assert q.put(job(2, 20, offence="track limits")) == []
    assert q.put(job(3, 30, cars=(16, 55))) == []
    assert q.put(job(4, 40, offence=None)) == []
    assert len(q) == 4


def test_a_message_without_an_offence_never_supersedes_anything():
    """None == None would otherwise make every unparsed message one giant incident."""
    q = CardQueue(max_age=None)
    q.put(job(1, 10, offence=None))
    assert q.put(job(2, 20, offence=None)) == []


def test_supersession_can_be_switched_off():
    q = CardQueue(supersede=False, max_age=None)
    q.put(job(1, 10))
    assert q.put(job(2, 20)) == []
    assert len(q) == 2


def test_stale_jobs_are_shed_when_the_model_would_start_them():
    q = CardQueue(max_age=60)
    q.put(job(1, 0, offence="a"))
    q.put(job(2, 50, offence="b"))
    nxt, shed = q.get(now=100)
    assert [(j.id, why) for j, why in shed] == [(1, STALE)]
    assert nxt.id == 2


def test_queue_serves_oldest_first():
    q = CardQueue(max_age=None)
    for i, off in enumerate(["a", "b", "c"]):
        q.put(job(i, i * 10, offence=off))
    assert [q.get(now=30)[0].id for _ in range(3)] == [0, 1, 2]
    assert q.get(now=30) == (None, [])


# --- the simulation that justifies the policy -------------------------------

def test_fifo_under_a_burst_makes_the_last_card_wait_for_all_the_others():
    burst = [job(i, 0, offence=f"o{i}") for i in range(5)]
    r = simulate(burst, lambda _j: 10, CardQueue(supersede=False, max_age=None))
    assert [round(d) for _, d in r.served] == [10, 20, 30, 40, 50]
    assert r.shed == []


def test_replay_speed_multiplies_the_cost_of_a_card_in_race_time():
    one = [job(0, 0)]
    assert simulate(one, lambda _j: 8, CardQueue(), speed=1).served[0][1] == 8
    assert simulate(one, lambda _j: 8, CardQueue(), speed=20).served[0][1] == 160


def test_every_job_is_either_served_or_shed_never_lost():
    rng = random.Random(1)
    jobs = [job(i, rng.uniform(0, 600), offence=rng.choice("abc"),
                cars=(rng.randint(1, 4),)) for i in range(60)]
    r = simulate(jobs, lambda _j: 20, CardQueue(max_age=90))
    ids = sorted([j.id for j, _ in r.served] + [j.id for j, _ in r.shed])
    assert ids == list(range(60))
    assert all(d - 20 <= 90 for _, d in r.served)   # nothing started past its deadline
