"""Race control message -> facets.

A race control message is about forty characters of shouted capitals:

    TURN 4 INCIDENT INVOLVING CARS 1 (NOR) AND 44 (HAM) NOTED - CAUSING A COLLISION

Everything the retrieval layer needs to be precise is in there - the offence, who, where,
what stage of the stewards' process - but as prose, not as fields. This module turns it
into structure, because filtering on structure is what makes retrieval accurate
(docs/concepts/02-metadata-and-versioning.md), and because no LLM should be spent on a
job that a dozen regexes do exactly.

The vocabulary is not guessed. It was read off 1,228 real messages from six 2026 races,
which is also how the surprise showed up: **510 of those 1,228 are waved blue flags**.
Firing a card on every message would mean spending almost half the budget explaining that
a backmarker is being lapped. `explainable` is the filter that stops it.

Usage:
    python -m engine.detect --round 12       # classify a race and print the breakdown
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field

# --- Stages of the stewards' process ----------------------------------------
# A single incident walks through several of these, minutes apart, and each one is a
# separate message. The stage matters as much as the offence: "NOTED" means the
# stewards may look at it, "UNDER INVESTIGATION" means they are, and only a penalty
# message is an outcome.
NOTED = "noted"
INVESTIGATING = "under_investigation"
NO_FURTHER_ACTION = "no_further_action"
PENALTY_ISSUED = "penalty_issued"
PENALTY_SERVED = "penalty_served"
LAP_DELETED = "lap_deleted"
WARNING = "warning"
TRACK_CONDITION = "track_condition"
INFO = "info"

# Ordered: the first match wins, so specific forms are tested before the general ones
# they contain. "NO FURTHER INVESTIGATION" contains "INVESTIGATION"; "PENALTY SERVED"
# contains "PENALTY".
STAGE_PATTERNS: list[tuple[str, re.Pattern]] = [
    (PENALTY_SERVED, re.compile(r"PENALTY SERVED", re.I)),
    (NO_FURTHER_ACTION, re.compile(r"NO FURTHER INVESTIGATION|NO FURTHER ACTION", re.I)),
    (INVESTIGATING, re.compile(r"UNDER INVESTIGATION", re.I)),
    (LAP_DELETED, re.compile(r"\bDELETED\b", re.I)),
    (WARNING, re.compile(r"BLACK AND WHITE FLAG", re.I)),
    (PENALTY_ISSUED, re.compile(
        r"\d+\s*SECOND\s+(?:TIME\s+)?PENALTY|DRIVE[\s-]*THROUGH\s+PENALTY"
        r"|STOP[\s-]*AND[\s-]*GO|GRID\s+PENALTY|\bPENALTY\s+FOR\b|DISQUALIFI", re.I)),
    (NOTED, re.compile(r"\bNOTED\b", re.I)),
    (TRACK_CONDITION, re.compile(
        r"SAFETY CAR|\bVSC\b|VIRTUAL SAFETY CAR|RED FLAG|CHEQUERED FLAG"
        r"|SESSION (?:SUSPENDED|RESUMED)|RACE (?:SUSPENDED|RESUMED|WILL RESUME)", re.I)),
]

# Stages that deserve an explanation card. Blue flags, per-sector yellows and paddock
# notices are the overwhelming majority of traffic and explain themselves.
EXPLAINABLE = {
    NOTED, INVESTIGATING, NO_FURTHER_ACTION,
    PENALTY_ISSUED, PENALTY_SERVED, LAP_DELETED, WARNING, TRACK_CONDITION,
}

CAR_RE = re.compile(r"CARS?\s+(\d+)\s*\(([A-Z]{3})\)(?:\s+AND\s+(\d+)\s*\(([A-Z]{3})\))?")
ANY_CAR_RE = re.compile(r"CAR\s+(\d+)\b")
TURN_RE = re.compile(r"\bTURN\s+(\d+)", re.I)
LAP_RE = re.compile(r"\bLAP\s+(\d+)", re.I)
SECTOR_RE = re.compile(r"TRACK SECTOR\s+(\d+)", re.I)
TIMESTAMP_RE = re.compile(r"\s*\(\d{1,2}:\d{2}:\d{2}\)\s*$")

# Penalty magnitude, when the message states one.
PENALTY_FORMS: list[tuple[str, re.Pattern]] = [
    ("stop_and_go", re.compile(r"STOP[\s-]*AND[\s-]*GO", re.I)),
    ("drive_through", re.compile(r"DRIVE[\s-]*THROUGH", re.I)),
    ("time_penalty", re.compile(r"(\d+)\s*SECOND\s+(?:TIME\s+)?PENALTY", re.I)),
    ("grid_penalty", re.compile(r"(\d+)\s*(?:PLACE|POSITION)\s+GRID", re.I)),
    ("disqualified", re.compile(r"DISQUALIFI", re.I)),
    ("black_and_white", re.compile(r"BLACK AND WHITE FLAG", re.I)),
]

# The offence phrase sits after the last " - ". These tails are noise on the end of it.
OFFENCE_TAIL_RE = re.compile(
    r"\s+(?:AT\s+TURN\s+\d+.*|LAP\s+\d+.*|\d{1,2}:\d{2}:\d{2}.*|\([A-Z]{3}\))\s*$", re.I)


@dataclass(frozen=True)
class Facets:
    stage: str
    offence: str | None = None
    cars: tuple[int, ...] = ()
    drivers: tuple[str, ...] = ()
    turn: int | None = None
    lap: int | None = None
    sector: int | None = None
    penalty: str | None = None
    penalty_seconds: int | None = None
    raw: str = ""

    @property
    def explainable(self) -> bool:
        return self.stage in EXPLAINABLE

    @property
    def signature(self) -> str:
        """Identity of the *situation*, not of the message.

        Two messages with the same signature deserve the same explanation, which is what
        makes caching worth anything: a yellow flag in sector 7 recurs constantly, and
        re-retrieving and re-generating for each one is pure waste.

        Deliberately excludes lap and timestamp - the same offence by the same driver at
        the same corner is the same explanation whether it happened on lap 12 or lap 40.
        """
        return "|".join([
            self.stage,
            self.offence or "-",
            ",".join(str(c) for c in self.cars) or "-",
            str(self.turn or "-"),
        ])

    def describe(self) -> str:
        who = ", ".join(f"#{c}" for c in self.cars) or "-"
        where = f" T{self.turn}" if self.turn else ""
        return f"{self.stage}: {self.offence or '-'} [{who}{where}]"


def _offence(message: str) -> str | None:
    """The offence phrase, taken from after the last ' - ' and cleaned of its tails."""
    if " - " not in message:
        return None
    tail = message.rsplit(" - ", 1)[1].strip()
    tail = TIMESTAMP_RE.sub("", tail).strip()
    tail = OFFENCE_TAIL_RE.sub("", tail).strip()
    tail = tail.rstrip(".").strip()
    # A bare car reference is not an offence: "PENALTY FOR CAR 11 (PER)" splits that way.
    if not tail or ANY_CAR_RE.fullmatch(tail) or len(tail) < 4:
        return None
    return " ".join(tail.split()).title()


def classify(message: str, lap: int | None = None, car: int | None = None) -> Facets:
    """Classify one race control message. `lap` and `car` come from the feed's own columns."""
    msg = " ".join(str(message).split())

    stage = INFO
    for name, pattern in STAGE_PATTERNS:
        if pattern.search(msg):
            stage = name
            break

    cars: list[int] = []
    drivers: list[str] = []
    if m := CAR_RE.search(msg):
        cars.append(int(m.group(1)))
        drivers.append(m.group(2))
        if m.group(3):
            cars.append(int(m.group(3)))
            drivers.append(m.group(4))
    else:
        cars = [int(x) for x in ANY_CAR_RE.findall(msg)]
    if not cars and car:
        cars = [car]

    penalty = penalty_seconds = None
    for name, pattern in PENALTY_FORMS:
        if pm := pattern.search(msg):
            penalty = name
            if name == "time_penalty" and pm.groups() and pm.group(1):
                penalty_seconds = int(pm.group(1))
            break

    # A lap number inside the text refers to the lap the incident happened on, which is
    # more precise than the feed's column when the two disagree.
    lap_in_text = int(m.group(1)) if (m := LAP_RE.search(msg)) else None

    return Facets(
        stage=stage,
        offence=_offence(msg),
        cars=tuple(dict.fromkeys(cars)),
        drivers=tuple(drivers),
        turn=int(m.group(1)) if (m := TURN_RE.search(msg)) else None,
        lap=lap_in_text or lap,
        sector=int(m.group(1)) if (m := SECTOR_RE.search(msg)) else None,
        penalty=penalty,
        penalty_seconds=penalty_seconds,
        raw=msg,
    )


def main() -> None:
    from collections import Counter
    from ingest.replay import load_session

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--year", type=int, default=2026)
    p.add_argument("--round", type=int, default=12)
    p.add_argument("--session", default="R")
    p.add_argument("--show", type=int, default=12, help="print N explainable messages")
    args = p.parse_args()

    s = load_session(args.year, args.round, args.session)
    facets = [
        classify(m["Message"], lap=m["Lap"], car=m["RacingNumber"])
        for _, m in s.race_control_messages.iterrows()
    ]

    print(f"\n{s.event['EventName']} {args.year} - {len(facets)} messages\n")
    print("by stage:")
    for stage, n in Counter(f.stage for f in facets).most_common():
        mark = "  <- card" if stage in EXPLAINABLE else ""
        print(f"   {stage:22s} {n:4d}{mark}")

    exp = [f for f in facets if f.explainable]
    print(f"\nexplainable: {len(exp)} of {len(facets)} "
          f"({100 * len(exp) / len(facets):.0f}%)")
    print(f"distinct situations (signatures): {len(set(f.signature for f in exp))}")

    print("\noffences seen:")
    for off, n in Counter(f.offence for f in exp if f.offence).most_common(12):
        print(f"   {n:4d}  {off}")

    print(f"\nfirst {args.show} explainable:")
    for f in exp[:args.show]:
        print(f"   {f.describe()}")
        print(f"        {f.raw[:96]}")


if __name__ == "__main__":
    main()
