"""An incident -> a grounded card, with the model deciding what to retrieve.

`explain(message, q, variant)` runs one agent over one incident. `variant` is "naive" or
"guarded"; the difference is the four rules in docs/concepts/09-agents-and-guardrails.md,
and every one of them is enforced here or in `agent/tools.py`, not in the prompt:

1. Filters are bound to the incident (`BoundTools`), so the model never chooses them.
2. No retrieval, no answer: a run that never called `search_rules` gets one retry, then
   no published prose.
3. Citations are bound to the run's ledger (`bind_citations`, from Stage 3).
4. The tool loop is capped by an auto-function-invocation filter.

The naive variant gets none of these, and is measured with the same checks so the
difference is a number rather than an opinion.

Usage:
    python -m agent.steward "car 44 left the track at turn 4 and gained an advantage"
    python -m agent.steward --naive "..."
"""

from __future__ import annotations

import argparse
import asyncio
import re
import time
from dataclasses import dataclass, field

from agent import telemetry  # before semantic_kernel, see agent/__init__.py
from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.ollama import (OllamaChatCompletion,
                                                  OllamaChatPromptExecutionSettings)
from semantic_kernel.filters import AutoFunctionInvocationContext, FilterTypes
from semantic_kernel.functions import KernelArguments

import config
from agent.tools import BoundTools, Ledger, NaiveTools
from engine.explain import _situation, bind_citations
from engine.query import Query
from engine.retrieve import Hit, Retriever

NAIVE, GUARDED = "naive", "guarded"

# The tutorial version: ask nicely, once.
NAIVE_INSTRUCTIONS = """You explain F1 race control messages to viewers.
Search the regulations first, then explain the message in at most 3 short sentences,
citing sources like [1]."""

# Stage 3's card prompt, re-cut for a model that fetches its own sources. The first line
# is there because measuring showed it is needed: told only to "search first", this model
# answered from memory and invented an article number (docs/concepts/09).
INSTRUCTIONS = """You explain F1 race control messages to a knowledgeable viewer.
You do not know the regulations yourself. Before answering you MUST call search_rules.
You may also call search_guidelines and search_precedents. Use ONLY what the tools return.

Then write at most 3 short sentences explaining what this means and why. Rules:
- If the tools do not cover something, do not say it.
- Cite every factual claim with the source number the tool gave, like [1] or [4].
- Name the specific article or guideline when a source gives one.
- No preamble, no restating the message, no speculation about what happens next."""

RETRY = ("You answered without calling search_rules, so none of that can be checked. "
         "Call search_rules now and answer only from what it returns.")

# SK stops its own loop after 5 rounds by default. This is tighter, and counts calls rather
# than rounds, because one round can request several calls at once.
MAX_TOOL_CALLS = 4
# "Article 33.3", "Art. 33.3", and "Rule 24.1": the naive agent's first answer invented the
# last form, which an Article-only pattern let straight through.
ARTICLE_RE = re.compile(r"\b(?:Art(?:icle|\.)?|Rule)\s*([A-Z]?\d+(?:\.\d+)*[a-z]?)", re.I)
MARKER_RE = re.compile(r"\[(\d+)\]")


@dataclass
class AgentCard:
    variant: str
    message: str
    answer: str                      # as published: bound for guarded, raw for naive
    raw: str                         # what the model wrote, before any guard
    ledger: Ledger
    citations: list[Hit] = field(default_factory=list)
    dropped: list[int] = field(default_factory=list)       # markers pointing at nothing
    unsupported: list[str] = field(default_factory=list)   # articles named, never retrieved
    retried: bool = False
    grounded: bool = True            # did the published answer come after a rule search
    capped: bool = False             # did the tool-call cap stop the loop
    error: str | None = None
    ms: float = 0.0
    trace_id: int = 0


def chat_service() -> OllamaChatCompletion:
    return OllamaChatCompletion(ai_model_id=config.AGENT_MODEL, host=config.OLLAMA_HOST)


def unsupported_articles(text: str, hits: list[Hit]) -> list[str]:
    """Article numbers named in the prose that no retrieved passage contains.

    Citation binding checks that a marker points at a real source. This checks the claim
    sitting next to it: "Article 33.1 [2]" passes binding whenever [2] exists, and is still
    invented if 33.1 appears in nothing the tools returned.
    """
    seen = " ".join(f"{h.ref} {h.article or ''} {h.text}" for h in hits)
    out = []
    for m in ARTICLE_RE.finditer(text):
        num = m.group(1).rstrip(".")
        if not re.search(rf"(?<![\d.]){re.escape(num)}(?![\d])", seen) and num not in out:
            out.append(num)
    return out


def _message(message: str, q: Query) -> str:
    return (f"RACE CONTROL SAID:\n{message}\n\n"
            f"DATE: {q.date_iso} ({q.season} season)\n\n"
            f"SITUATION:\n{_situation(q)}")


def _build(variant: str, retriever: Retriever, q: Query, ledger: Ledger,
           service) -> tuple[ChatCompletionAgent, dict]:
    kernel = Kernel()
    kernel.add_service(service)
    tools = (NaiveTools if variant == NAIVE else BoundTools)(retriever, q, ledger)
    kernel.add_plugin(tools, plugin_name="fia")
    state = {"capped": False}

    if variant == GUARDED:
        @kernel.filter(FilterTypes.AUTO_FUNCTION_INVOCATION)
        async def cap(ctx: AutoFunctionInvocationContext, next):
            # Rule 4. Counted from the ledger, which is the record of calls that ran.
            if len(ledger.calls) >= MAX_TOOL_CALLS:
                state["capped"] = True
                ctx.terminate = True
                return
            await next(ctx)

    settings = OllamaChatPromptExecutionSettings(
        function_choice_behavior=FunctionChoiceBehavior.Auto(),
        # Same settings as the pipeline's cards: a reading task, not a writing one.
        options={"temperature": 0.2, "num_predict": 512},
    )
    agent = ChatCompletionAgent(
        kernel=kernel, name="steward",
        instructions=NAIVE_INSTRUCTIONS if variant == NAIVE else INSTRUCTIONS,
        arguments=KernelArguments(settings=settings),
    )
    return agent, state


async def explain(message: str, q: Query, variant: str = GUARDED,
                  retriever: Retriever | None = None, service=None) -> AgentCard:
    retriever = retriever or Retriever()
    service = service or chat_service()
    ledger = Ledger()
    agent, state = _build(variant, retriever, q, ledger, service)
    card = AgentCard(variant=variant, message=message, answer="", raw="", ledger=ledger)

    t0 = time.monotonic()
    with telemetry.tracer().start_as_current_span("f1.card") as span:
        card.trace_id = span.get_span_context().trace_id
        span.set_attribute("f1.variant", variant)
        span.set_attribute("f1.message", message)
        try:
            thread = ChatHistoryAgentThread()
            resp = await agent.get_response(messages=_message(message, q), thread=thread)
            raw = str(resp.content or "")
            # Rule 2: one retry, in the same thread, so the model sees what it skipped.
            if variant == GUARDED and not ledger.searched("search_rules"):
                card.retried = True
                resp = await agent.get_response(messages=RETRY, thread=resp.thread)
                raw = str(resp.content or "")
        except Exception as exc:  # degrade, as engine/explain.py does: never crash a replay
            card.error = f"{type(exc).__name__}: {exc}"
            raw = ""

        card.raw = raw.strip()
        card.capped = state["capped"]
        card.grounded = ledger.searched("search_rules")
        card.unsupported = unsupported_articles(card.raw, ledger.hits)

        if variant == NAIVE:
            card.answer = card.raw
            card.dropped = sorted({int(n) for n in MARKER_RE.findall(card.raw)
                                   if not 1 <= int(n) <= len(ledger.hits)})
        elif not card.grounded:
            card.answer = "(No grounded explanation: the model did not search the regulations.)"
        elif card.capped:
            # The loop was stopped mid-search, so what came back is not a finished answer.
            # Degrade as engine/explain.py does: the sources are the part with the value.
            card.answer = "(Tool-call limit reached before an answer. Retrieved sources below.)"
            card.citations = list(ledger.hits)
        else:
            # Rule 3: resolve markers against everything this run retrieved, and nothing else.
            card.answer, card.citations, card.dropped = bind_citations(card.raw, ledger.hits)

        for k, v in (("f1.grounded", card.grounded), ("f1.retried", card.retried),
                     ("f1.tool_calls", len(ledger.calls)), ("f1.dropped", len(card.dropped)),
                     ("f1.unsupported", len(card.unsupported)), ("f1.capped", card.capped)):
            span.set_attribute(k, v)
        if card.error:
            span.set_attribute("f1.error", card.error)
    card.ms = (time.monotonic() - t0) * 1000
    return card


def render(card: AgentCard) -> str:
    lines = ["-" * 76, f"  {card.variant.upper()} agent", "-" * 76, ""]
    for c in card.ledger.calls:
        args = ", ".join(f"{k}={v!r}" for k, v in c.args.items())
        lines.append(f"  -> {c.tool}({args})  returned {c.shown}")
    lines += ["", f"  {card.answer}", ""]
    if card.citations:
        lines += [f"  [{i}] {h.cite()}" for i, h in enumerate(card.citations, 1)]
    if card.dropped:
        lines.append(f"  ! markers pointing at nothing: {card.dropped}")
    if card.unsupported:
        lines.append(f"  ! articles named but never retrieved: {card.unsupported}")
    if card.retried:
        lines.append("  ! answered without searching; retried once")
    if card.error:
        lines.append(f"  ! {card.error}")
    lines.append(f"  ({card.ms / 1000:.1f}s, {len(card.ledger.calls)} tool calls)")
    return "\n".join(lines)


def main() -> None:
    from engine import query as qmod
    from engine.detect import classify

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("message", nargs="+")
    p.add_argument("--naive", action="store_true")
    p.add_argument("--season", type=int, default=2026)
    p.add_argument("--date", default="2026-08-23")
    args = p.parse_args()

    telemetry.setup()
    text = " ".join(args.message)
    q = qmod.build(classify(f"INCIDENT NOTED - {text.upper()}"), season=args.season, date_iso=args.date)
    card = asyncio.run(explain(text, q, NAIVE if args.naive else GUARDED))
    print(render(card))


if __name__ == "__main__":
    main()
