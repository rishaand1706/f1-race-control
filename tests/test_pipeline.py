"""The serving pipeline, end to end, with no Ollama, no index and no FastF1.

A scripted replay feeds synthetic events through the real pipeline; the model is a fake
stream with controllable timing. What is locked in:

  - the order a card reaches the browser in: message, then sources, then words
  - that prose citations are bound on the way out, not after
  - supersession, as the browser sees it
  - the stage's central claim: inline generation makes the replay late, the pipeline
    does not
  - the bus: snapshot first, no gap before the first delta, a slow client is cut off
    without touching anyone else

    python -m pytest tests/test_pipeline.py -q
"""

from __future__ import annotations

import asyncio

import pytest

import server.pipeline as pl
from engine.retrieve import GUIDELINE, PRECEDENT, RULE, Hit
from ingest.events import LAP_COMPLETED, RACE_CONTROL, Event
from server.bus import Bus


class FakeRetriever:
    def search(self, q, k=2):
        def hit(i, kind):
            return Hit(pid=i, kind=kind, ref=f"source {i}", article=None, title="t",
                       text="passage text", score=1.0)
        return {RULE: [hit(1, "sporting")], GUIDELINE: [hit(2, "penalty_guidelines")],
                PRECEDENT: [hit(3, "decision")]}


def fake_model(chunks, delay=0.0):
    async def astream(prompt, model, think, client):
        for kind, x in chunks:
            await asyncio.sleep(delay)
            yield kind, x
        yield "done", {"done_reason": "stop", "eval_count": 9, "prompt_eval_count": 99}
    return astream


class Scripted(pl.Replay):
    def __init__(self, events, **kw):
        self.published: list[tuple[str, dict]] = []
        super().__init__(FakeRetriever(), lambda ev, d: self.published.append((ev, d)), **kw)
        self._events = events

    async def _load(self):
        return self._events


def rc(t, message, lap=1):
    return Event(t=t, kind=RACE_CONTROL, lap=lap,
                 payload={"message": message, "category": "Other", "flag": None, "car": None})


def lap(t, n=1, car=1):
    return Event(t=t, kind=LAP_COMPLETED, lap=n, payload={
        "car": car, "driver": "NOR", "team": "McLaren", "position": 1, "lap_time": 80.0,
        "compound": "MEDIUM", "tyre_life": n, "stint": 1, "pit_in": False, "pit_out": False})


@pytest.fixture(autouse=True)
def no_registry(monkeypatch):
    # The rulebook registry lives in data/, which is not in git.
    monkeypatch.setattr("engine.query.in_force", lambda *a: None)


async def settle(r: Scripted, timeout: float = 5.0) -> None:
    """Run the replay, then wait until no card is still in flight."""
    main = r.start()
    await asyncio.wait_for(main, timeout)
    busy = {"retrieving", "queued", "reasoning", "writing"}
    async def idle():
        while any(c.status in busy for c in r.cards.values()):
            await asyncio.sleep(0.01)
    await asyncio.wait_for(idle(), timeout)
    await r.stop()


def card_events(r, cid):
    return [(ev, d) for ev, d in r.published
            if ev in ("card", "token") and d.get("id") == cid]


def test_a_card_arrives_as_message_then_sources_then_words(monkeypatch):
    monkeypatch.setattr(pl, "astream", fake_model(
        [("answer", "Car 1 caused a collision ["), ("answer", "1]. Invented [9"),
         ("answer", "]. Guideline [2].")]))
    r = Scripted([rc(0, "INCIDENT INVOLVING CAR 1 (NOR) NOTED - CAUSING A COLLISION")], speed=0)
    asyncio.run(settle(r))

    names = [ev for ev, _ in r.published]
    assert names.index("rc") < names.index("token")
    evs = card_events(r, 1)
    first_sources = next(i for i, (ev, d) in enumerate(evs) if ev == "card" and d["sources"])
    first_token = next(i for i, (ev, _) in enumerate(evs) if ev == "token")
    assert first_sources < first_token, "sources must be on screen before any prose"

    streamed = "".join(d["text"] for ev, d in evs if ev == "token")
    card = r.cards[1]
    assert streamed == card.text == "Car 1 caused a collision [1]. Invented. Guideline [2]."
    assert "[9" not in streamed
    assert (card.status, card.cited, card.dropped) == ("done", [1, 2], [9])
    assert {"sources_ms", "first_word_ms", "done_ms"} <= card.timing.keys()


def test_reasoning_never_reaches_the_card_text(monkeypatch):
    monkeypatch.setattr(pl, "astream", fake_model(
        [("thinking", "Okay, the user wants")] * 40 + [("answer", "Explained [1].")]))
    r = Scripted([rc(0, "INCIDENT INVOLVING CAR 1 (NOR) NOTED - CAUSING A COLLISION")],
                 speed=0, model="m", think=True)
    asyncio.run(settle(r))
    assert r.cards[1].text == "Explained [1]."
    assert r.cards[1].thinking_tokens == 40
    assert any(ev == "progress" for ev, _ in r.published)


def test_an_empty_answer_is_a_failure_with_a_reason(monkeypatch):
    async def exhausted(prompt, model, think, client):
        yield "thinking", "..."
        yield "done", {"done_reason": "length"}
    monkeypatch.setattr(pl, "astream", exhausted)
    r = Scripted([rc(0, "INCIDENT INVOLVING CAR 1 (NOR) NOTED - CAUSING A COLLISION")], speed=0)
    asyncio.run(settle(r))
    assert r.cards[1].status == "failed"
    assert "budget" in r.cards[1].note
    assert r.cards[1].sources, "a failed explanation still keeps its sources"


def test_a_newer_message_supersedes_a_queued_card_about_the_same_incident(monkeypatch):
    monkeypatch.setattr(pl, "astream", fake_model([("answer", "Text [1].")] * 5, delay=0.05))
    r = Scripted([
        rc(0, "INCIDENT INVOLVING CAR 5 (VET) NOTED - UNSAFE RELEASE"),
        rc(1, "TURN 4 INCIDENT INVOLVING CARS 1 (NOR) AND 44 (HAM) NOTED - CAUSING A COLLISION"),
        rc(2, "TURN 4 INCIDENT INVOLVING CARS 1 (NOR) AND 44 (HAM) UNDER INVESTIGATION - CAUSING A COLLISION"),
    ], speed=0, max_age=None)
    asyncio.run(settle(r))
    assert [r.cards[i].status for i in (1, 2, 3)] == ["done", "superseded", "done"]
    assert "card 3" in r.cards[2].note
    assert r.cards[2].sources, "a superseded card keeps its sources"


def test_inline_generation_makes_the_replay_late_and_the_pipeline_does_not(monkeypatch):
    """The whole argument for the queue, as a test.

    One incident, then five laps due over the next half second. Inline, the loop waits
    for the model and every lap after the incident is delivered late. Queued, it does not.
    """
    monkeypatch.setattr(pl, "astream", fake_model([("answer", "Text [1].")] * 6, delay=0.1))
    events = [rc(0, "INCIDENT INVOLVING CAR 1 (NOR) NOTED - CAUSING A COLLISION")]
    events += [lap(t=i, n=i) for i in range(1, 6)]

    inline = Scripted(events, speed=10, mode="inline")
    asyncio.run(settle(inline))
    queued = Scripted(events, speed=10, mode="pipeline")
    asyncio.run(settle(queued))

    assert inline.max_late_s > 0.3
    assert queued.max_late_s < 0.1
    assert inline.cards[1].status == queued.cards[1].status == "done"


# --- the bus ------------------------------------------------------------------

def test_a_client_gets_a_snapshot_then_every_delta_after_it():
    async def go():
        bus = Bus(snapshot=lambda: {"cards": []})
        sub = bus.subscribe()
        first = await sub.__anext__()
        bus.publish("rc", {"message": "A"})     # published immediately after: not lost
        bus.publish("rc", {"message": "B"})
        nxt = lambda: asyncio.wait_for(sub.__anext__(), 1.0)
        return first, await nxt(), await nxt()
    first, a, b = asyncio.run(go())
    assert "event: snapshot" in first and "retry:" in first
    assert '"A"' in a and '"B"' in b


def test_a_slow_client_is_cut_off_and_the_others_are_not():
    async def go():
        bus = Bus(snapshot=lambda: {})
        slow, fast = bus.subscribe(), bus.subscribe()
        await slow.__anext__()
        await fast.__anext__()
        got = 0
        for i in range(2500):
            bus.publish("token", {"i": i})
            await asyncio.wait_for(fast.__anext__(), 1.0)   # fast keeps up
            got += 1
        # Checked inside the loop: asyncio.run closes both generators on the way out.
        return got, bus.dropped_total, len(bus.clients)
    got, dropped, still_connected = asyncio.run(go())
    assert got == 2500
    assert dropped == 1
    assert still_connected == 1
