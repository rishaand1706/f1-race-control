"""Which card gets the model next, and which cards do not get it at all.

A card has two tiers. Retrieval takes about a second and produces the part that carries
the value: the rule, the guideline and the precedents, each with a link. Prose takes the
model several seconds at best and a minute at worst, and there is one model on one GPU.
So the first tier is guaranteed for every card and the second is best-effort, and this
module is the policy that decides what "best effort" means when messages arrive faster
than the model can write about them.

The naive policy is a FIFO queue, and it fails the way every FIFO fails when arrivals
come in bursts: the queue grows during the burst, every card behind it waits for all the
cards in front, and prose arrives for an incident the stewards have already moved past.
Two changes fix it, and both are about deciding *not* to do work:

**Supersession.** One incident produces a chain of messages minutes apart - noted, under
investigation, penalty. If the model has not started on "noted" by the time "under
investigation" arrives, explaining "noted" is waste: the newer message says more. The
older job is dropped from the prose queue, and its card keeps its sources.

**A staleness deadline.** Prose for an incident that happened more than `max_age` race
seconds ago is not worth the model's time, because the viewer has moved on. Stale jobs
are shed at the moment the model would pick them up, not when they arrive, since that is
when their age is known.

`python -m server.scheduler --sim` replays real message arrival times from every cached
race through each policy and reports what the viewer would actually see.
See docs/concepts/08-streaming-and-latency.md.
"""

from __future__ import annotations

import argparse
import random
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable

SUPERSEDED = "superseded"
STALE = "stale"


@dataclass
class Job:
    id: int
    t: float                              # race time of the message, seconds
    stage: str
    offence: str | None = None
    cars: frozenset[int] = frozenset()
    payload: Any = field(default=None, repr=False, compare=False)

    def same_incident(self, other: "Job") -> bool:
        """Two messages about one incident: same offence, at least one car in common.

        Overlap rather than equality, because the chain narrows as it goes: the incident
        is noted for cars 1 and 44, and the penalty names only car 1. The stage is left
        out on purpose - differing stages is exactly what a chain is.
        """
        return (self.offence is not None and self.offence == other.offence
                and bool(self.cars & other.cars))


class CardQueue:
    """The prose queue. `put` when retrieval finishes, `get` when the model is free.

    Both return the jobs they shed, with a reason, so the caller can tell the viewer
    that a card will not get prose instead of leaving it waiting forever.
    """

    def __init__(self, supersede: bool = True, max_age: float | None = 120.0) -> None:
        self.supersede = supersede
        self.max_age = max_age
        self.jobs: list[Job] = []

    def __len__(self) -> int:
        return len(self.jobs)

    def put(self, job: Job) -> list[tuple[Job, str]]:
        shed: list[tuple[Job, str]] = []
        if self.supersede:
            same = [j for j in self.jobs if job.same_incident(j)]
            # Supersession runs forwards in race time only. If a *later* message about
            # this incident is already queued, the arrival is the out-of-date one.
            if any(j.t > job.t for j in same):
                return [(job, SUPERSEDED)]
            for old in same:
                self.jobs.remove(old)
                shed.append((old, SUPERSEDED))
        self.jobs.append(job)
        return shed

    def get(self, now: float) -> tuple[Job | None, list[tuple[Job, str]]]:
        """The next job worth starting at race time `now`, oldest first."""
        shed: list[tuple[Job, str]] = []
        if self.max_age is not None:
            fresh = []
            for j in self.jobs:
                if now - j.t > self.max_age:
                    shed.append((j, STALE))
                else:
                    fresh.append(j)
            self.jobs = fresh
        if not self.jobs:
            return None, shed
        return self.jobs.pop(0), shed


# --- simulation ---------------------------------------------------------------

@dataclass
class SimResult:
    served: list[tuple[Job, float]]       # job, race seconds from message to prose done
    shed: list[tuple[Job, str]]

    @property
    def n(self) -> int:
        return len(self.served) + len(self.shed)

    def pct(self, q: float) -> float:
        lat = sorted(d for _, d in self.served)
        if not lat:
            return float("nan")
        return lat[min(len(lat) - 1, int(q * len(lat)))]


def simulate(arrivals: list[Job], service: Callable[[Job], float],
             queue: CardQueue, speed: float = 1.0) -> SimResult:
    """One model, serving the prose queue, against real arrival times.

    `service(job)` is wall seconds of generation. A replay at `speed`x packs that many
    race seconds into each wall second, so one card costs `service * speed` of race time.
    That is the whole reason a fast replay cannot get prose on every card.
    """
    arrivals = sorted(arrivals, key=lambda j: j.t)
    served: list[tuple[Job, float]] = []
    shed: list[tuple[Job, str]] = []
    free = float("-inf")    # race time at which the model next becomes idle
    i = 0
    while True:
        if not queue.jobs and i < len(arrivals):
            free = max(free, arrivals[i].t)
        while i < len(arrivals) and arrivals[i].t <= free:
            shed += queue.put(arrivals[i])
            i += 1
        job, dropped = queue.get(free)
        shed += dropped
        if job is None:
            if i >= len(arrivals):
                break
            continue
        free += service(job) * speed
        served.append((job, free - job.t))
    return SimResult(served, shed)


def _arrivals(rounds: list[int], year: int = 2026) -> dict[int, list[Job]]:
    """Every card-worthy message of each race, as a job at its real race time."""
    from engine.detect import classify
    from ingest.events import RACE_CONTROL
    from ingest.replay import build_events, load_session

    out: dict[int, list[Job]] = {}
    n = 0
    for rnd in rounds:
        events = build_events(load_session(year, rnd, "R", offline=True))
        jobs = []
        for e in events:
            if e.kind != RACE_CONTROL:
                continue
            f = classify(e["message"], lap=e.lap, car=e["car"])
            if not f.explainable or f.stage == "track_condition":
                continue
            jobs.append(Job(id=n, t=e.t, stage=f.stage, offence=f.offence,
                            cars=frozenset(f.cars), payload=f.raw))
            n += 1
        out[rnd] = jobs
    return out


def main() -> None:
    import logging
    logging.getLogger("fastf1").setLevel(logging.ERROR)

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sim", action="store_true", required=True)
    p.add_argument("--service", default="8",
                   help="wall seconds per card: one number, or a comma list to sample from")
    p.add_argument("--speeds", default="1,5,20")
    p.add_argument("--max-age", type=float, default=120.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    samples = [float(x) for x in args.service.split(",")]
    from ingest.replay import cached_races
    rounds = [r["round"] for r in cached_races()]
    by_round = _arrivals(rounds)
    all_jobs = [j for js in by_round.values() for j in js]

    # Burstiness, measured: how many card-worthy messages land inside one minute.
    worst = 0
    for js in by_round.values():
        ts = sorted(j.t for j in js)
        lo = 0
        for hi in range(len(ts)):
            while ts[hi] - ts[lo] > 60:
                lo += 1
            worst = max(worst, hi - lo + 1)
    print(f"\n{len(all_jobs)} card-worthy messages across {len(rounds)} races "
          f"(rounds {', '.join(map(str, rounds))})")
    print(f"busiest single minute: {worst} messages\n")

    policies = {
        "fifo": dict(supersede=False, max_age=None),
        "fifo + supersede": dict(supersede=True, max_age=None),
        f"+ max_age {args.max_age:.0f}s": dict(supersede=True, max_age=args.max_age),
    }
    print(f"service time: {args.service}s per card\n")
    print(f"{'speed':>5}  {'policy':<22} {'prose':>6} {'p50':>7} {'p95':>7} {'worst':>7}"
          f"  {'superseded':>10} {'stale':>6}")
    for speed in [float(s) for s in args.speeds.split(",")]:
        for name, kw in policies.items():
            rng = random.Random(args.seed)
            served, shed = [], []
            for js in by_round.values():
                r = simulate(js, lambda _j: rng.choice(samples), CardQueue(**kw), speed)
                served += r.served
                shed += r.shed
            res = SimResult(served, shed)
            sup = sum(1 for _, why in shed if why == SUPERSEDED)
            stale = sum(1 for _, why in shed if why == STALE)
            worst_lat = max((d for _, d in served), default=float("nan"))
            print(f"{speed:>4.0f}x  {name:<22} {100 * len(served) / res.n:>5.0f}% "
                  f"{res.pct(0.5):>6.0f}s {res.pct(0.95):>6.0f}s {worst_lat:>6.0f}s"
                  f"  {sup:>10} {stale:>6}")
        print()
    print("prose = cards that get an explanation at all; p50/p95/worst = race seconds from")
    print("the message to its explanation finishing. Sources are on screen in ~1s either way.")


if __name__ == "__main__":
    main()
