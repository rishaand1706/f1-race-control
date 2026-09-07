"""Retrieved passages -> a grounded explanation card.

This is the first point in the whole system where a language model is involved, and it is
given deliberately little to do: read these passages, explain this message, cite what you
used. It never sees the corpus, only what retrieval selected, and it is never asked for a
fact it was not handed.

Three things make the output trustworthy rather than merely fluent:

**Grounding.** The prompt contains the passages and nothing else. The model is told, in
terms, that anything not in them is not available.

**Citation binding.** The model writes `[2]`; we resolve `[2]` to the actual passage and
publish its reference. A marker pointing at a source that was never supplied is *removed*,
not rendered - because an invented citation is worse than none, it launders a hallucination
into something that looks checkable.

**Caching by situation, not by string.** Keyed on `Query.signature`, which is the offence,
the cars, the corner and the rulebook in force - not the message text and not the lap. The
same offence by the same driver at the same corner is the same explanation on lap 12 and on
lap 40.

That was justified in advance by the claim that a race issues these in bursts, and measuring
it says otherwise: **11% hit rate** over 542 card-worthy messages from 12 races of 2026.
Dropping car numbers from the key would give 46%, and is not available, because the card
names the car and a cached card would name the wrong one. The expensive half of the work is
exactly the car-specific half, which is why caching cannot fix the latency here. Kept, and
kept honest - see docs/concepts/07-grounding-and-citations.md.

Usage:
    python -m engine.explain --round 12 --max 5
"""

from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass, field

import httpx

import config
from engine.detect import Facets
from engine.query import Query
from engine.retrieve import Hit, Retriever, RULE, GUIDELINE, PRECEDENT

# qwen3 emits its reasoning inside <think> tags. Ollama can be asked to disable it, but
# older builds ignore the flag, so the output is stripped defensively either way.
# Belt and braces. With think=True this build returns reasoning in its own field and
# never inside the response, but other builds and models do emit the tags inline.
THINK_RE = re.compile(r"<think>.*?</think>\s*", re.S | re.I)
CITE_RE = re.compile(r"\[(\d+)\]")


class BudgetExhausted(RuntimeError):
    """The model reasoned until it ran out of tokens and never wrote an answer."""

PROMPT = """You are explaining an F1 race control message to a knowledgeable viewer.

RACE CONTROL SAID:
{message}

SITUATION:
{situation}

SOURCES (the only information you may use):
{sources}

Write at most 3 short sentences explaining what this means and why. Rules:
- Use ONLY the sources above. If they do not cover something, do not say it.
- Cite every factual claim with its source number, like [1] or [2].
- Name the specific article or guideline when a source gives one.
- No preamble, no restating the message, no speculation about what happens next.
"""


@dataclass
class Card:
    facets: Facets
    query: Query
    sources: list[Hit]
    explanation: str
    citations: list[Hit] = field(default_factory=list)
    dropped_citations: list[int] = field(default_factory=list)
    cached: bool = False
    retrieve_ms: float = 0.0
    generate_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.retrieve_ms + self.generate_ms

    def render(self) -> str:
        head = f"  {self.facets.stage.upper().replace('_', ' ')}"
        if self.facets.offence:
            head += f" - {self.facets.offence}"
        if self.facets.cars:
            head += "  " + " ".join(f"#{c}" for c in self.facets.cars)
        if self.facets.turn:
            head += f"  turn {self.facets.turn}"

        lines = ["-" * 76, head, "-" * 76, ""]
        for line in self.explanation.split("\n"):
            lines.append(f"  {line}" if line.strip() else "")
        if self.citations:
            lines.append("")
            for i, h in enumerate(self.citations, 1):
                lines.append(f"  [{i}] {h.cite()}")
        if self.dropped_citations:
            lines.append(f"  ! dropped invented citations: {self.dropped_citations}")
        tag = "cached" if self.cached else f"{self.retrieve_ms:.0f}ms retrieve + {self.generate_ms:.0f}ms generate"
        lines.append(f"  ({tag})")
        return "\n".join(lines)


def _situation(q: Query) -> str:
    ctx = q.context or {}
    bits = []
    if ctx.get("lap"):
        bits.append(f"Lap {ctx['lap']}")
    if ctx.get("track_status"):
        bits.append(f"track {ctx['track_status']}")
    for c in ctx.get("cars", []):
        s = f"Car {c['car']}"
        if c.get("driver"):
            s += f" ({c['driver']})"
        if c.get("position"):
            s += f" P{c['position']}"
        if c.get("compound"):
            s += f" on {c['compound'].lower()}"
            if c.get("tyre_life"):
                s += f", {c['tyre_life']} laps old"
        if c.get("gap_to_leader") is not None:
            s += f", {c['gap_to_leader']:.1f}s off the lead"
        bits.append(s)
    return "\n".join(bits) if bits else "No live timing context available."


def _sources_block(hits: list[Hit]) -> str:
    out = []
    for i, h in enumerate(hits, 1):
        body = " ".join(h.text.split())
        # Truncated hard. A 4B model given eight full articles loses the thread, and the
        # operative clause of a regulation is almost always in its opening lines.
        out.append(f"[{i}] {h.ref}\n{body[:700]}")
    return "\n\n".join(out)


def generate(prompt: str, timeout: float = 300.0) -> str:
    """Ask the model for the card text, and get back only the card text.

    `think=True` looks backwards for a system that does not want reasoning, and it is the
    whole fix. qwen3 reasons whether or not it is asked to; the only question is where the
    reasoning *goes*. Asked to think, Ollama routes it to a separate `thinking` field and
    leaves `response` clean. Told `think=False`, this build (0.32.9) ignores the flag, the
    model reasons anyway, and the reasoning arrives untagged inside `response` - where no
    amount of stripping can find it, because there is no longer a tag to strip. The first
    version of this file asked for no thinking and published the model's private
    deliberation as the explanation.

    `/no_think` in the prompt was measured too, and is a no-op here: 285 tokens against
    275 for a thinking run of the same trivial prompt.
    """
    with httpx.Client(timeout=timeout) as client:
        r = client.post(
            f"{config.OLLAMA_HOST}/api/generate",
            json={
                "model": config.LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "think": True,
                # Low temperature: this is a reading-comprehension task over supplied
                # text, not a writing task. Sampling variety buys nothing and costs
                # faithfulness.
                #
                # num_predict has to cover the reasoning as well as the answer, since both
                # are drawn from one budget. At 320 the reasoning consumed all of it and
                # every card came back empty or truncated mid-thought.
                "options": {"temperature": 0.2, "num_predict": config.NUM_PREDICT},
            },
        )
        r.raise_for_status()
        body = r.json()
        text = (body.get("response") or "").strip()

        # An empty answer after a full budget means reasoning ate it. Say so, rather than
        # letting the card render blank and look like a retrieval failure.
        if not text and body.get("done_reason") == "length":
            raise BudgetExhausted(
                f"model spent all {config.NUM_PREDICT} tokens reasoning and never answered"
            )
        return text


def bind_citations(text: str, sources: list[Hit]) -> tuple[str, list[Hit], list[int]]:
    """Resolve [n] markers to real sources; delete the ones that point nowhere.

    Renumbered to the order they are actually used, so the card's footnotes read 1, 2, 3
    rather than exposing retrieval's internal ordering.
    """
    used: list[int] = []
    dropped: list[int] = []

    for m in CITE_RE.finditer(text):
        n = int(m.group(1))
        if 1 <= n <= len(sources):
            if n not in used:
                used.append(n)
        elif n not in dropped:
            dropped.append(n)

    renumber = {old: new for new, old in enumerate(used, 1)}

    def sub(m: re.Match) -> str:
        n = int(m.group(1))
        return f"[{renumber[n]}]" if n in renumber else ""

    cleaned = CITE_RE.sub(sub, text)
    cleaned = re.sub(r" +([.,;])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    return cleaned, [sources[n - 1] for n in used], dropped


class Explainer:
    def __init__(self, retriever: Retriever | None = None, k: int = 2) -> None:
        self.retriever = retriever or Retriever()
        self.k = k
        self.cache: dict[str, Card] = {}
        self.hits = 0
        self.misses = 0

    def explain(self, q: Query) -> Card:
        if (hit := self.cache.get(q.signature)) is not None:
            self.hits += 1
            return Card(facets=q.facets, query=q, sources=hit.sources,
                        explanation=hit.explanation, citations=hit.citations,
                        dropped_citations=hit.dropped_citations, cached=True)
        self.misses += 1

        t0 = time.monotonic()
        results = self.retriever.search(q, k=self.k)
        # Order matters: the rule first, then the FIA's own guideline, then precedent.
        # It is the order the card reads in, and small models cite earlier sources more.
        sources = results[RULE] + results[GUIDELINE] + results[PRECEDENT]
        retrieve_ms = (time.monotonic() - t0) * 1000

        if not sources:
            card = Card(facets=q.facets, query=q, sources=[],
                        explanation="No sources retrieved for this message.",
                        retrieve_ms=retrieve_ms)
            self.cache[q.signature] = card
            return card

        prompt = PROMPT.format(
            message=q.facets.raw,
            situation=_situation(q),
            sources=_sources_block(sources),
        )

        t1 = time.monotonic()
        try:
            raw = THINK_RE.sub("", generate(prompt)).strip()
        except (BudgetExhausted, httpx.HTTPError) as exc:
            # Degrade, do not crash. A replay that dies on one slow generation is worse
            # than a card that says what was retrieved without prose around it: the
            # citations are the part that carries the value, and they are already here.
            generate_ms = (time.monotonic() - t1) * 1000
            card = Card(
                facets=q.facets, query=q, sources=sources,
                explanation=f"({type(exc).__name__}: no explanation generated. "
                            f"Retrieved sources below.)",
                citations=sources, retrieve_ms=retrieve_ms, generate_ms=generate_ms,
            )
            self.cache[q.signature] = card
            return card
        generate_ms = (time.monotonic() - t1) * 1000

        text, citations, dropped = bind_citations(raw, sources)
        card = Card(facets=q.facets, query=q, sources=sources, explanation=text,
                    citations=citations, dropped_citations=dropped,
                    retrieve_ms=retrieve_ms, generate_ms=generate_ms)
        self.cache[q.signature] = card
        return card


def main() -> None:
    from engine.detect import classify
    from engine.state import RaceState
    from engine import query as qmod
    from ingest.events import RACE_CONTROL
    from ingest.replay import build_events, load_session

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--year", type=int, default=2026)
    p.add_argument("--round", type=int, default=12)
    p.add_argument("--session", default="R")
    p.add_argument("--max", type=int, default=5)
    p.add_argument("-k", type=int, default=2, help="passages per card slot")
    args = p.parse_args()

    session = load_session(args.year, args.round, args.session)
    events = build_events(session)
    date_iso = str(session.date.date())

    state = RaceState()
    explainer = Explainer(k=args.k)
    print(f"\n{session.event['EventName']} {args.year}   regs as of {date_iso}\n")

    made = 0
    for e in events:
        state.apply(e)
        if e.kind != RACE_CONTROL:
            continue
        facets = classify(e["message"], lap=e.lap, car=e["car"])
        if not facets.explainable or facets.stage == "track_condition":
            continue
        q = qmod.build(facets, state=state, season=args.year, date_iso=date_iso)
        print(explainer.explain(q).render())
        print()
        made += 1
        if made >= args.max:
            break

    total = explainer.hits + explainer.misses
    print(f"cards: {total}   cache hits: {explainer.hits} "
          f"({100 * explainer.hits / max(total, 1):.0f}%)")


if __name__ == "__main__":
    main()
