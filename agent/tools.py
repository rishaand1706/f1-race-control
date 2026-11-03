"""The retrieval layer, re-cut as tools an agent can call.

Two versions of the same three searches, and the difference between them is the whole
stage (docs/concepts/09-agents-and-guardrails.md):

- `NaiveTools` puts the filters in the schema. `search_rules` asks the model for a season
  and a date, because the rulebook really is versioned by date and that looks like the
  model's job. Precedent search has no published-before filter, since nothing asks for one.
- `BoundTools` closes over the incident. The model writes the search text and nothing
  else, so it cannot pick the wrong rulebook or read a ruling from the future: there is
  no parameter through which to do either.

Both share a `Ledger`: every passage any call returns gets one number for the whole run, and
those numbers are the only citations the answer may use.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field
from typing import Annotated

from semantic_kernel.functions import kernel_function

from agent import telemetry
from engine.query import Query
from engine.retrieve import GUIDELINE, PRECEDENT, RULE, Hit, Retriever

SHOWN = 3    # passages per call the model reads; the pipeline shows 2 per slot
SCORED = 5   # rule results kept per call, so the evaluation can score hit@5 like --check
NO_FILTER = "9999-12-31T23:59"


@dataclass
class Call:
    tool: str
    args: dict
    shown: list[int]                 # ledger numbers handed to the model
    articles: list[str | None]       # top SCORED articles, for scoring rule searches
    urls: list[str | None]           # top SCORED urls, for spotting precedent leaks
    ms: float
    season: int | None = None        # the filters this call actually ran with
    date_iso: str | None = None


@dataclass
class Ledger:
    """Every passage this run retrieved, each numbered once, in first-seen order."""
    hits: list[Hit] = field(default_factory=list)
    calls: list[Call] = field(default_factory=list)
    _num: dict[int, int] = field(default_factory=dict)

    def number(self, hit: Hit) -> int:
        if hit.pid not in self._num:
            self.hits.append(hit)
            self._num[hit.pid] = len(self.hits)
        return self._num[hit.pid]

    def searched(self, tool: str) -> bool:
        return any(c.tool == tool for c in self.calls)


def render(numbered: list[tuple[int, Hit]]) -> str:
    if not numbered:
        return "No passages matched."
    # Truncated as in engine/explain.py: the operative clause is almost always up front.
    return "\n\n".join(f"[{n}] {h.ref}\n{' '.join(h.text.split())[:700]}" for n, h in numbered)


class _Tools:
    def __init__(self, retriever: Retriever, q: Query, ledger: Ledger) -> None:
        self.retriever = retriever
        self.q = q
        self.ledger = ledger

    def _search(self, tool: str, slot: str, text: str, q: Query, args: dict) -> str:
        t0 = time.monotonic()
        with telemetry.tracer().start_as_current_span(f"retrieve {slot}") as span:
            hits = self.retriever._rank(self.retriever._candidates(slot, q), text, text, SCORED)
            span.set_attribute("f1.query", text)
            span.set_attribute("f1.season", q.season)
            span.set_attribute("f1.date", q.date_iso)
            span.set_attribute("f1.asof", q.asof)
            span.set_attribute("f1.hits", len(hits))
        shown = [(self.ledger.number(h), h) for h in hits[:SHOWN]]
        self.ledger.calls.append(Call(
            tool=tool, args=args, shown=[n for n, _ in shown],
            articles=[h.article for h in hits], urls=[h.url for h in hits],
            ms=(time.monotonic() - t0) * 1000, season=q.season, date_iso=q.date_iso))
        return render(shown)

    @kernel_function(name="search_guidelines",
                     description="Search the FIA Penalty Guidelines and Driving Standards "
                                 "Guidelines for the usual penalty for an offence.")
    def search_guidelines(self, offence: Annotated[str, "the offence, e.g. 'causing a collision'"]) -> str:
        return self._search("search_guidelines", GUIDELINE, offence, self.q, {"offence": offence})


class BoundTools(_Tools):
    """The model chooses the words. The incident chooses the rulebook and the cut-off."""

    @kernel_function(name="search_rules",
                     description="Search the F1 Sporting Regulations in force on the day of "
                                 "the incident for the rule that governs it.")
    def search_rules(self, query: Annotated[str, "what happened, in plain words"]) -> str:
        return self._search("search_rules", RULE, query, self.q, {"query": query})

    @kernel_function(name="search_precedents",
                     description="Search earlier stewards' decisions about similar incidents.")
    def search_precedents(self, query: Annotated[str, "what happened, in plain words"]) -> str:
        return self._search("search_precedents", PRECEDENT, query, self.q, {"query": query})


class NaiveTools(_Tools):
    """What a tutorial would write: the filters are parameters, so the model fills them."""

    @kernel_function(name="search_rules",
                     description="Search the F1 Sporting Regulations for the rule that governs "
                                 "an incident. The regulations change between and within "
                                 "seasons, so give the season and the date of the incident.")
    def search_rules(self, query: Annotated[str, "what happened, in plain words"],
                     season: Annotated[int, "the championship season, e.g. 2025"],
                     date: Annotated[str, "the date of the incident, YYYY-MM-DD"]) -> str:
        q = dataclasses.replace(self.q, season=int(season), date_iso=str(date))
        return self._search("search_rules", RULE, query, q,
                            {"query": query, "season": season, "date": date})

    @kernel_function(name="search_precedents",
                     description="Search stewards' decisions about similar incidents.")
    def search_precedents(self, query: Annotated[str, "what happened, in plain words"]) -> str:
        q = dataclasses.replace(self.q, asof=NO_FILTER)
        return self._search("search_precedents", PRECEDENT, query, q, {"query": query})
