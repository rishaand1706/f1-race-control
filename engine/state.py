"""Race state, as a fold over the event stream.

There is no "current state" stored anywhere upstream. There is only a sequence of things
that happened, and the state is whatever you get by applying all of them in order:

    state = reduce(apply, events, RaceState())

That is event sourcing, and it buys three things that matter here. Replaying the same
events always reproduces the same state, so a bug at lap 54 is reachable in seconds
rather than by waiting for a race. The state at any past moment is recoverable by
stopping the fold early, which is what "what was the situation when this incident
happened" needs. And live and replay cannot drift apart, because both feed the same
reducer the same event type.

The rule that keeps all of that true: **`apply` must be a pure function of the event and
the current state.** No clock reads, no network, no randomness. The moment it depends on
anything outside the stream, replay stops being deterministic and the guarantees go.

Kept in memory rather than in DuckDB: twenty cars and a few counters is a dictionary, and
persisting it would buy nothing that re-running the fold does not. The corpus lives in
DuckDB because it is queried with filters; this is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ingest.events import (
    Event, RACE_CONTROL, LAP_COMPLETED, TRACK_STATUS, SESSION_STATUS,
)


@dataclass
class CarState:
    car: int
    driver: str = ""
    team: str = ""
    position: int | None = None
    lap: int = 0
    last_lap_time: float | None = None
    compound: str | None = None
    tyre_life: int | None = None
    stint: int | None = None
    in_pit: bool = False
    # Session time at which this car completed each lap. This is what makes gaps
    # computable: a gap is only meaningful between two cars on the *same* lap.
    t_at_lap: dict[int, float] = field(default_factory=dict)


class RaceState:
    def __init__(self) -> None:
        self.cars: dict[int, CarState] = {}
        self.lap: int = 0                    # leader's lap
        self.track_status: str = "-"
        self.session_status: str = "-"
        self.t: float = 0.0
        self.last_message: str | None = None
        # First car to complete each lap is the leader at that lap; its crossing time is
        # the reference every gap is measured against.
        self.leader_t_at_lap: dict[int, float] = {}

    # --- the fold -----------------------------------------------------------

    def apply(self, e: Event) -> "RaceState":
        self.t = e.t

        if e.kind == LAP_COMPLETED:
            c = self.cars.get(e["car"])
            if c is None:
                c = CarState(car=e["car"])
                self.cars[e["car"]] = c
            c.driver = e["driver"] or c.driver
            c.team = e["team"] or c.team
            c.position = e["position"] if e["position"] is not None else c.position
            c.lap = e.lap or c.lap
            c.last_lap_time = e["lap_time"]
            c.compound = e["compound"] or c.compound
            c.tyre_life = e["tyre_life"]
            c.stint = e["stint"] or c.stint
            c.in_pit = bool(e["pit_in"])
            if e.lap is not None:
                c.t_at_lap[e.lap] = e.t
                prev = self.leader_t_at_lap.get(e.lap)
                if prev is None or e.t < prev:
                    self.leader_t_at_lap[e.lap] = e.t
                self.lap = max(self.lap, e.lap)

        elif e.kind == TRACK_STATUS:
            self.track_status = e["label"]

        elif e.kind == SESSION_STATUS:
            self.session_status = e["status"]

        elif e.kind == RACE_CONTROL:
            self.last_message = e["message"]
            # A race control message carries a lap number of its own, and it can be
            # ahead of the last lap we have timing for. Trusting it keeps the displayed
            # lap correct during the gap between the leader crossing and the timing feed
            # catching up.
            if e.lap:
                self.lap = max(self.lap, e.lap)

        return self

    # --- derived views ------------------------------------------------------

    def order(self) -> list[CarState]:
        """Cars by current position; cars without a position sort last."""
        return sorted(
            self.cars.values(),
            key=lambda c: (c.position is None, c.position or 999),
        )

    def gap_to_leader(self, car: int) -> float | None:
        c = self.cars.get(car)
        if c is None or c.lap not in c.t_at_lap:
            return None
        leader = self.leader_t_at_lap.get(c.lap)
        return None if leader is None else c.t_at_lap[c.lap] - leader

    def gap_ahead(self, car: int) -> float | None:
        """Gap to the car one position ahead, measured on the last lap they share."""
        c = self.cars.get(car)
        if c is None or c.position is None:
            return None
        ahead = next((o for o in self.cars.values() if o.position == c.position - 1), None)
        if ahead is None:
            return None
        shared = set(c.t_at_lap) & set(ahead.t_at_lap)
        if not shared:
            return None
        lap = max(shared)
        return c.t_at_lap[lap] - ahead.t_at_lap[lap]

    def context_for(self, cars: list[int]) -> dict:
        """The live context that turns a terse message into a specific query.

        "INCIDENT INVOLVING CARS 1 AND 44 NOTED - TURN 4" says almost nothing on its own.
        With the state attached - who they are, what positions, how far apart, on what
        tyres, under what track status - it becomes a query that can actually retrieve
        the right precedent. Stage 3 consumes this.
        """
        out = {
            "lap": self.lap,
            "track_status": self.track_status,
            "session_status": self.session_status,
            "cars": [],
        }
        for n in cars:
            c = self.cars.get(n)
            if c is None:
                continue
            out["cars"].append({
                "car": c.car, "driver": c.driver, "team": c.team,
                "position": c.position, "compound": c.compound,
                "tyre_life": c.tyre_life, "stint": c.stint,
                "last_lap_time": c.last_lap_time,
                "gap_to_leader": self.gap_to_leader(c.car),
                "gap_ahead": self.gap_ahead(c.car),
                "in_pit": c.in_pit,
            })
        return out

    def summary(self, top: int = 10) -> str:
        rows = [c for c in self.order() if c.position][:top]
        if not rows:
            return "no timing data seen"
        lines = [f"final state - lap {self.lap}, track {self.track_status}, "
                 f"session {self.session_status}",
                 f"  {'P':>2}  {'CAR':>3} {'DRV':4} {'TYRE':8} {'AGE':>3} "
                 f"{'GAP':>9}  TEAM"]
        for c in rows:
            gap = self.gap_to_leader(c.car)
            gap_s = "leader" if c.position == 1 else (f"+{gap:.3f}" if gap is not None else "-")
            lines.append(f"  {c.position:>2}  {c.car:>3} {c.driver:4} "
                         f"{(c.compound or '-'):8} {c.tyre_life or 0:>3} {gap_s:>9}  {c.team}")
        return "\n".join(lines)
