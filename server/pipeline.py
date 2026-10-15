"""The live pipeline: replay -> state -> detect -> retrieve -> prose -> browser.

Three stages, each with its own worker, joined by queues:

    replay loop      paces events, folds state, publishes the feed      never waits
        |  card-worthy message: query built *now*, from state *now*
        v
    retrieval        ~1s per card, in arrival order                     guaranteed tier
        |  sources are on screen the moment this finishes
        v
    prose            one model, CardQueue decides what it writes next   best-effort tier

The replay loop must never wait on the model, because everything the viewer sees moves
through it: the clock, the timing tower, the race control feed. `mode="inline"` keeps the
naive version - retrieve and write inside the loop - so its failure stays reproducible:
the whole screen freezes for every card, and `late_s` climbs by the length of each one.

The query is built in the replay loop, not in the retrieval worker, and that is a
correctness point, not a style one. The situation that explains a message is the state at
the moment it was issued. A worker that picks the message up two seconds later and reads
the state then would describe positions and gaps from the future.

See docs/concepts/08-streaming-and-latency.md.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Callable

import httpx
import pandas as pd

import config
from engine import query as qmod
from engine.detect import TRACK_CONDITION, classify
from engine.explain import (
    PROMPT, StreamBinder, _situation, _sources_block, astream,
)
from engine.retrieve import GUIDELINE, PRECEDENT, RULE, Retriever
from engine.state import RaceState
from ingest.events import (
    LAP_COMPLETED, RACE_CONTROL, SESSION_STATUS, TRACK_STATUS, Event,
)
from ingest.replay import areplay, build_events, load_session, local_start
from server.scheduler import CardQueue, Job

STATE_HZ = 4            # timing-tower refreshes per second; it is a view, so coalesce it
SNIPPET = 320           # characters of each source shown on the card
PRE_RACE_S = 120.0      # replay starts this long before lights out; earlier is skipped


@dataclass
class CardView:
    """Everything the browser shows for one card. Sent whole on every status change."""
    id: int
    t: float
    clock: str
    lap: int | None
    message: str
    stage: str
    offence: str | None
    cars: list[int]
    turn: int | None
    penalty: str | None
    status: str = "retrieving"   # queued | reasoning | writing | done | superseded | stale | failed
    sources: list[dict] = field(default_factory=list)
    text: str = ""
    cited: list[int] = field(default_factory=list)
    dropped: list[int] = field(default_factory=list)
    note: str = ""
    thinking_tokens: int = 0
    # Wall milliseconds, all measured from the moment the message reached the screen.
    # These are the numbers a viewer experiences; model-only time is `generate_ms`.
    timing: dict = field(default_factory=dict)


class Replay:
    """One replay of one race, and the whole view of it that browsers are sent."""

    def __init__(self, retriever: Retriever, publish: Callable[[str, object], None], *,
                 year: int = 2026, rnd: int = 12, speed: float = 20.0,
                 from_lap: int | None = None, mode: str = "pipeline",
                 model: str | None = None, think: bool | None = None,
                 max_age: float | None = config.PROSE_MAX_AGE) -> None:
        self.retriever = retriever
        self.publish = publish
        self.year, self.rnd, self.speed, self.from_lap = year, rnd, speed, from_lap
        self.mode = mode
        self.model = model or config.CARD_MODEL
        self.think = config.CARD_THINK if model is None else think

        self.status = "loading"
        self.name = f"Round {rnd}"
        self.total_laps: int | None = None
        self.date_iso = f"{year}-01-01"
        self.t0_local = None            # circuit-local time at t=0, for precedent cut-off
        self.drivers: dict[int, dict] = {}

        self.state = RaceState()
        self.feed: deque[dict] = deque(maxlen=400)
        self.cards: dict[int, CardView] = {}
        self.late_s = 0.0
        self.max_late_s = 0.0
        self._next_id = 0
        self._last_state_pub = 0.0

        self.retrieval: asyncio.Queue = asyncio.Queue()
        self.prose = CardQueue(supersede=True, max_age=max_age)
        self._wake = asyncio.Event()
        self._tasks: list[asyncio.Task] = []
        self._http: httpx.AsyncClient | None = None

    # --- the view ------------------------------------------------------------

    def info(self) -> dict:
        return {"round": self.rnd, "year": self.year, "name": self.name,
                "speed": self.speed, "mode": self.mode, "model": self.model,
                "think": self.think, "status": self.status, "total_laps": self.total_laps,
                "drivers": self.drivers}

    def race_state(self) -> dict:
        s = self.state
        tower = []
        for c in s.order():
            if c.position is None:
                continue
            tower.append({
                "pos": c.position, "car": c.car, "drv": c.driver, "team": c.team,
                "color": self.drivers.get(c.car, {}).get("color"),
                "lap": c.lap, "gap": s.gap_to_leader(c.car), "int": s.gap_ahead(c.car),
                "tyre": c.compound, "age": c.tyre_life, "pit": c.in_pit,
            })
        by_status: dict[str, int] = {}
        for card in self.cards.values():
            by_status[card.status] = by_status.get(card.status, 0) + 1
        return {"t": s.t, "clock": Event(t=s.t, kind="").clock, "lap": s.lap,
                "track": s.track_status, "session": s.session_status,
                "late_s": round(self.late_s, 1), "max_late_s": round(self.max_late_s, 1),
                "queue": len(self.prose), "retrieving": self.retrieval.qsize(),
                "cards": by_status, "tower": tower}

    def snapshot(self) -> dict:
        return {"run": self.info(), "state": self.race_state(),
                "feed": list(self.feed), "cards": [asdict(c) for c in self.cards.values()]}

    def _card(self, card: CardView) -> None:
        self.publish("card", asdict(card))

    # --- lifecycle -----------------------------------------------------------

    def start(self) -> asyncio.Task:
        main = asyncio.create_task(self._run())
        self._tasks.append(main)
        return main

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        if self._http is not None:
            await self._http.aclose()

    async def _load(self) -> list[Event]:
        session = await asyncio.to_thread(load_session, self.year, self.rnd, "R", True)
        events = await asyncio.to_thread(build_events, session)
        self.name = str(session.event["EventName"])
        self.total_laps = int(session.total_laps) if session.total_laps else None
        self.date_iso = str(session.date.date())
        self.t0_local = await asyncio.to_thread(local_start, session)
        for _, r in session.results.iterrows():
            self.drivers[int(r["DriverNumber"])] = {
                "drv": r["Abbreviation"], "team": r["TeamName"],
                "color": f"#{r['TeamColor']}" if r["TeamColor"] else None,
            }
        return events

    async def _run(self) -> None:
        self.publish("snapshot", self.snapshot())
        events = await self._load()

        # Fast-forward to the start point: fold state and fill the feed, but make no
        # cards. Explaining a message the viewer never saw arrive helps nobody.
        start = 0
        for i, e in enumerate(events):
            ahead = (self.state.lap >= self.from_lap) if self.from_lap else e.t >= -PRE_RACE_S
            if ahead:
                start = i
                break
            self.state.apply(e)
            if e.kind == RACE_CONTROL:
                self.feed.append(self._feed_item(e, card=None))

        self._http = httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=300.0))
        if self.mode == "pipeline":
            self._tasks += [asyncio.create_task(self._retrieval_worker()),
                            asyncio.create_task(self._prose_worker())]

        self.status = "running"
        self.publish("snapshot", self.snapshot())

        async for e, late in areplay(events[start:], self.speed):
            self.late_s = late
            self.max_late_s = max(self.max_late_s, late)
            self.state.apply(e)
            if e.kind == RACE_CONTROL:
                await self._on_message(e)
            elif e.kind in (TRACK_STATUS, SESSION_STATUS):
                self._publish_state(force=True)
            elif e.kind == LAP_COMPLETED:
                self._publish_state()

        self.status = "finished"
        self._publish_state(force=True)
        self.publish("run", self.info())

    # --- stage 1: the replay loop ----------------------------------------------

    def _feed_item(self, e: Event, card: int | None) -> dict:
        return {"clock": e.clock, "t": e.t, "lap": e.lap, "flag": e["flag"],
                "category": e["category"], "message": e["message"], "card": card}

    def _asof(self, e: Event) -> str:
        """The moment the message was issued, circuit-local, for the precedent cut-off."""
        if self.t0_local is None:
            return ""
        return (self.t0_local + pd.Timedelta(seconds=e.t)).strftime("%Y-%m-%dT%H:%M")

    def _publish_state(self, force: bool = False) -> None:
        now = time.monotonic()
        if force or now - self._last_state_pub >= 1 / STATE_HZ:
            self._last_state_pub = now
            self.publish("state", self.race_state())

    async def _on_message(self, e: Event) -> None:
        f = classify(e["message"], lap=e.lap, car=e["car"])
        card = None
        if f.explainable and f.stage != TRACK_CONDITION:
            self._next_id += 1
            card = CardView(id=self._next_id, t=e.t, clock=e.clock, lap=e.lap,
                            message=f.raw, stage=f.stage, offence=f.offence,
                            cars=list(f.cars), turn=f.turn, penalty=f.penalty)
            self.cards[card.id] = card
            # Built here, from the state as it is at this event. See module docstring.
            q = qmod.build(f, state=self.state, season=self.year, date_iso=self.date_iso,
                           asof=self._asof(e))
            emitted = time.monotonic()

        item = self._feed_item(e, card.id if card else None)
        self.feed.append(item)
        self.publish("rc", item)
        if card is None:
            return
        self._card(card)

        if self.mode == "inline":
            # The naive version, kept on purpose. Everything below blocks the replay loop.
            job = await self._retrieve(card, q, f, emitted)
            if job is not None:
                await self._write(job)
        else:
            self.retrieval.put_nowait((card, q, f, emitted))

    # --- stage 2: retrieval ------------------------------------------------------

    async def _retrieval_worker(self) -> None:
        # One at a time, in arrival order. Parallel retrieval buys little - the embedding
        # call and BM25 both serialise anyway - and it would let a later message about an
        # incident reach the prose queue before an earlier one.
        while True:
            card, q, f, emitted = await self.retrieval.get()
            job = await self._retrieve(card, q, f, emitted)
            if job is None:
                continue
            for old, why in self.prose.put(job):
                self._shed(old, why, by=job.id)
            self._wake.set()

    async def _retrieve(self, card: CardView, q, f, emitted: float) -> Job | None:
        t0 = time.monotonic()
        try:
            results = await asyncio.to_thread(self.retriever.search, q, 2)
        except Exception as exc:              # a card that says why beats a dead replay
            card.status, card.note = "failed", f"retrieval failed: {type(exc).__name__}"
            self._card(card)
            return None
        hits = results[RULE] + results[GUIDELINE] + results[PRECEDENT]
        slot = {id(h): name for name in (RULE, GUIDELINE, PRECEDENT) for h in results[name]}
        card.sources = [{
            "n": i, "slot": slot[id(h)], "kind": h.kind, "ref": h.ref, "title": h.title,
            "article": h.article, "url": h.url,
            "snippet": " ".join(h.text.split())[:SNIPPET],
        } for i, h in enumerate(hits, 1)]
        done = time.monotonic()
        card.timing["retrieve_ms"] = round((done - t0) * 1000)
        card.timing["sources_ms"] = round((done - emitted) * 1000)
        if not hits:
            card.status, card.note = "failed", "no sources retrieved"
            self._card(card)
            return None
        card.status = "queued"
        self._card(card)
        prompt = PROMPT.format(message=f.raw, situation=_situation(q),
                               sources=_sources_block(hits))
        return Job(id=card.id, t=card.t, stage=f.stage, offence=f.offence,
                   cars=frozenset(f.cars), payload=(prompt, emitted, done))

    # --- stage 3: prose ----------------------------------------------------------

    def _shed(self, job: Job, why: str, by: int | None = None) -> None:
        card = self.cards[job.id]
        card.status = why
        card.note = (f"superseded by card {by}: a newer message about this incident"
                     if by is not None else
                     "not explained: the model was busy until this incident was too old")
        self._card(card)

    async def _prose_worker(self) -> None:
        while True:
            job, shed = self.prose.get(now=self.state.t)
            for old, why in shed:
                self._shed(old, why)
            if job is None:
                # No await between get() and clear(), so no put can slip in unseen.
                self._wake.clear()
                await self._wake.wait()
                continue
            await self._write(job)

    async def _write(self, job: Job) -> None:
        card = self.cards[job.id]
        prompt, emitted, retrieved = job.payload
        binder = StreamBinder(len(card.sources))
        t0 = time.monotonic()
        card.timing["wait_ms"] = round((t0 - retrieved) * 1000)
        card.status = "reasoning" if self.think else "writing"
        self._card(card)

        stats: dict = {}
        first = None
        try:
            async for kind, x in astream(prompt, self.model, self.think, self._http):
                if kind == "thinking":
                    card.thinking_tokens += 1
                    if card.thinking_tokens % 20 == 0:
                        self.publish("progress", {"id": card.id,
                                                  "thinking_tokens": card.thinking_tokens})
                elif kind == "answer":
                    out = binder.feed(x)
                    if out:
                        if first is None:
                            first = time.monotonic()
                            card.status = "writing"
                            card.timing["first_word_ms"] = round((first - emitted) * 1000)
                            self._card(card)
                        card.text += out
                        self.publish("token", {"id": card.id, "text": out})
                else:
                    stats = x
            tail = binder.flush()
            if tail:
                card.text += tail
                self.publish("token", {"id": card.id, "text": tail})
            if card.text:
                card.status = "done"
            else:
                card.status = "failed"
                card.note = ("the model reasoned until its token budget ran out"
                             if stats.get("done_reason") == "length" else "empty answer")
        except httpx.HTTPError as exc:
            card.status, card.note = "failed", f"model call failed: {type(exc).__name__}"

        end = time.monotonic()
        card.cited, card.dropped = binder.used, binder.dropped
        card.timing["generate_ms"] = round((end - t0) * 1000)
        card.timing["done_ms"] = round((end - emitted) * 1000)
        if stats:
            card.timing["prompt_tokens"] = stats.get("prompt_eval_count")
            card.timing["gen_tokens"] = stats.get("eval_count")
        self._card(card)
