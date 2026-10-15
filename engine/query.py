"""Facets plus live state -> a retrieval query.

Nobody types anything into this system. The query has to be *built*, from a forty-character
message and whatever the state reducer knows at that moment. That makes it a different
retrieval problem from a chatbot's, and a more interesting one: the quality of the answer
is decided here, before retrieval runs at all.

    TURN 4 INCIDENT INVOLVING CARS 1 (NOR) AND 44 (HAM) NOTED - CAUSING A COLLISION

on its own retrieves anything about collisions. Enriched from `engine/state.py` it becomes
"causing a collision at turn 4, cars fighting for P2, 0.4s apart, on hard tyres, lap 34
under green flag" - which can actually find the precedent that matters.

One query object drives three *separate* filtered retrievals, not one blended top-k. See
`engine/retrieve.py` for why that is not an optimisation but a correctness requirement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from corpus.fetch_regs import in_force
from engine.detect import Facets


@dataclass
class Query:
    facets: Facets
    season: int
    date_iso: str
    text: str                       # the semantic query
    terms: str                      # the lexical query
    context: dict = field(default_factory=dict)
    reg_issue: str | None = None    # the Sporting Regulations issue in force that day
    # Circuit-local time the message was issued, 'YYYY-MM-DDTHH:MM'. Precedent search
    # only sees decisions published before it. Unknown -> the start of that day, which
    # errs towards hiding a same-day precedent rather than leaking a later one.
    asof: str = ""

    def __post_init__(self) -> None:
        self.asof = self.asof or f"{self.date_iso}T00:00"

    @property
    def signature(self) -> str:
        """Cache key. The situation plus the rulebook it is judged against."""
        return f"{self.season}|{self.reg_issue or '-'}|{self.facets.signature}"


def _seconds(x: float | None) -> str:
    return f"{x:.1f}s" if x is not None else "?"


def _car_phrase(car: dict) -> str:
    bits = [f"car {car['car']}"]
    if car.get("driver"):
        bits.append(car["driver"])
    if car.get("position"):
        bits.append(f"P{car['position']}")
    if car.get("compound"):
        age = car.get("tyre_life")
        bits.append(f"{car['compound'].lower()} tyres" + (f" {age} laps old" if age else ""))
    return " ".join(bits)


def build(facets: Facets, state=None, season: int = 2026,
          date_iso: str = "2026-01-01", asof: str = "") -> Query:
    """Compose the query. `state` is a RaceState, or None when replaying without one."""
    offence = facets.offence or "race control incident"

    # --- the semantic query -------------------------------------------------
    parts = [offence]
    if facets.turn:
        parts.append(f"at turn {facets.turn}")

    context: dict = {}
    if state is not None and facets.cars:
        context = state.context_for(list(facets.cars))
        cars = context.get("cars", [])
        if cars:
            parts.append("involving " + " and ".join(_car_phrase(c) for c in cars))
        # A gap is what separates "two cars racing each other" from "a car alone", and
        # the stewards weigh that heavily. It is also the single most useful thing the
        # message itself never says.
        if len(cars) >= 2:
            gaps = [c["gap_to_leader"] for c in cars if c["gap_to_leader"] is not None]
            if len(gaps) == 2:
                parts.append(f"{abs(gaps[0] - gaps[1]):.1f}s apart")
        elif len(cars) == 1 and cars[0].get("gap_ahead") is not None:
            parts.append(f"{_seconds(cars[0]['gap_ahead'])} behind the car ahead")
        if context.get("track_status") and context["track_status"] != "AllClear":
            parts.append(f"under {context['track_status']}")
    if facets.lap:
        parts.append(f"on lap {facets.lap}")

    text = ", ".join(parts)

    # --- the lexical query --------------------------------------------------
    # Deliberately not the same string. BM25 wants the high-signal identifiers and the
    # offence's own words, not the prose scaffolding around them: "involving", "under",
    # "apart" appear in half the corpus and only dilute the scoring.
    terms = [offence]
    if facets.turn:
        terms.append(f"turn {facets.turn}")
    terms += [f"car {c}" for c in facets.cars]
    terms += list(facets.drivers)

    reg = in_force(season, "sporting", date_iso)

    return Query(
        facets=facets, season=season, date_iso=date_iso,
        text=text, terms=" ".join(terms),
        context=context, reg_issue=reg.issue if reg else None, asof=asof,
    )
