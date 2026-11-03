"""The agent's guardrails, with the model replaced by a script. No Ollama needed.

Each guard in docs/concepts/09 gets a test that fails if the guard is removed: the filters
the model cannot set, the retry, the no-search refusal, ledger-bound citations, the
unsupported-article check, and the tool-call cap.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from agent import telemetry  # noqa: F401  (before semantic_kernel)
from pydantic import PrivateAttr
from semantic_kernel.connectors.ai.ollama import OllamaChatCompletion
from semantic_kernel.contents import ChatMessageContent, FunctionCallContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.functions import KernelPlugin

from agent import steward
from agent.tools import NO_FILTER, BoundTools, Ledger, NaiveTools
from engine.detect import classify
from engine import query as qmod
from engine.retrieve import GUIDELINE, PRECEDENT, RULE, Hit


def hit(pid: int, article: str = "33.3", text: str = "Drivers must use the track.") -> Hit:
    return Hit(pid=pid, kind="sporting", ref=f"2025 Sporting Regulations, Art. {article}",
               article=article, title="Track limits", text=text, score=1.0, url=f"u{pid}")


class FakeRetriever:
    """Records the query each search was filtered with; returns fixed passages."""

    def __init__(self, hits: list[Hit] | None = None) -> None:
        self.hits = hits or [hit(1), hit(2, "33.4"), hit(3, "54.3")]
        self.seen: list[tuple[str, str, int, str, str]] = []

    def _candidates(self, slot, q):
        self.seen.append((slot, q.date_iso, q.season, q.asof, q.reg_issue))
        return list(range(len(self.hits)))

    def _rank(self, cand, text, terms, k):
        return self.hits[:k]


class Scripted(OllamaChatCompletion):
    """A chat model that replies from a script: a string is an answer, a dict a tool call."""

    _script: list = PrivateAttr(default_factory=list)

    def __init__(self, script: list) -> None:
        super().__init__(ai_model_id="scripted", host="http://127.0.0.1:9")
        self._script = list(script)

    async def _inner_get_chat_message_contents(self, chat_history, settings):
        step = self._script.pop(0) if self._script else "done"
        if isinstance(step, dict):
            items = [FunctionCallContent(id=f"c{len(chat_history)}", plugin_name="fia",
                                         function_name=step["fn"], arguments=json.dumps(step["args"]))]
            return [ChatMessageContent(role=AuthorRole.ASSISTANT, items=items)]
        return [ChatMessageContent(role=AuthorRole.ASSISTANT, content=step)]


def incident(season: int = 2025, date: str = "2025-07-06"):
    return qmod.build(classify("INCIDENT NOTED - LEAVING THE TRACK AND GAINING AN ADVANTAGE"),
                      season=season, date_iso=date)


def run(script: list, variant: str = steward.GUARDED, retriever=None):
    retriever = retriever or FakeRetriever()
    card = asyncio.run(steward.explain("car 44 left the track", incident(), variant,
                                       retriever=retriever, service=Scripted(script)))
    return card, retriever


SEARCH = {"fn": "search_rules", "args": {"query": "leaving the track"}}


# --- rule 1: the model cannot set the filters ---------------------------------

def test_bound_tools_offer_the_model_no_filter_parameters():
    tools = KernelPlugin.from_object("fia", BoundTools(FakeRetriever(), incident(), Ledger()))
    for name in ("search_rules", "search_precedents"):
        assert [p.name for p in tools[name].parameters] == ["query"]


def test_naive_tools_do_offer_them():
    tools = KernelPlugin.from_object("fia", NaiveTools(FakeRetriever(), incident(), Ledger()))
    assert [p.name for p in tools["search_rules"].parameters] == ["query", "season", "date"]


def test_bound_search_uses_the_incidents_rulebook_and_cutoff():
    r = FakeRetriever()
    q = incident()
    BoundTools(r, q, Ledger()).search_rules("anything")
    BoundTools(r, q, Ledger()).search_precedents("anything")
    assert r.seen[0][:3] == (RULE, "2025-07-06", 2025)
    assert r.seen[1][0] == PRECEDENT and r.seen[1][3] == q.asof


def test_naive_search_uses_whatever_the_model_said():
    r = FakeRetriever()
    tools = NaiveTools(r, incident(), Ledger())
    tools.search_rules("anything", season=2024, date="2024-07-06")
    tools.search_precedents("anything")
    assert r.seen[0][:3] == (RULE, "2024-07-06", 2024)
    assert r.seen[1][3] == NO_FILTER


# --- the ledger ---------------------------------------------------------------

def test_ledger_numbers_each_passage_once_across_calls():
    r = FakeRetriever()
    ledger = Ledger()
    tools = BoundTools(r, incident(), ledger)
    first = tools.search_rules("a")
    second = tools.search_guidelines("b")      # same three passages again
    assert first == second
    assert [c.shown for c in ledger.calls] == [[1, 2, 3], [1, 2, 3]]
    assert len(ledger.hits) == 3
    assert r.seen[1][0] == GUIDELINE


# --- rules 2 and 3 ------------------------------------------------------------

def test_guarded_retries_once_when_the_model_skips_search():
    card, _ = run(["Article 33.3 obviously applies.", SEARCH, "Leaving the track breaches 33.3 [1]."])
    assert card.retried and card.grounded
    assert card.answer == "Leaving the track breaches 33.3 [1]."
    assert [h.pid for h in card.citations] == [1]


def test_guarded_publishes_no_prose_if_the_model_never_searches():
    card, _ = run(["From memory: Article 33.1.", "Still from memory."])
    assert card.retried and not card.grounded
    assert card.answer.startswith("(No grounded explanation")
    assert "33.1" not in card.answer


def test_guarded_deletes_markers_for_passages_this_run_never_returned():
    card, _ = run([SEARCH, "It breaches Article 33.3 [1], as decided before [7]."])
    assert card.dropped == [7]
    assert "[7]" not in card.answer and "[1]" in card.answer


def test_naive_publishes_what_the_model_wrote_and_the_checks_still_see_it():
    card, _ = run(["Under Rule 24.1 this is a penalty [1]."], variant=steward.NAIVE)
    assert card.answer == "Under Rule 24.1 this is a penalty [1]."
    assert not card.grounded and not card.retried
    assert card.dropped == [1] and card.unsupported == ["24.1"]


# --- the unsupported-article check --------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("Article 33.3 applies [1].", []),
    ("Art. 33.4 applies.", []),
    ("Article 33.1 applies [1].", ["33.1"]),        # 33.1 is not 33.10 or 33.3
    ("Rule 24.1 applies.", ["24.1"]),
    ("Article 54.3a applies.", ["54.3a"]),
])
def test_unsupported_articles(text, expected):
    hits = [hit(1, "33.3"), hit(2, "33.4", "see also Article 33.10"), hit(3, "54.3")]
    assert steward.unsupported_articles(text, hits) == expected


# --- rule 4 -------------------------------------------------------------------

def test_tool_call_cap_stops_the_loop_and_publishes_sources_not_prose():
    card, _ = run([SEARCH] * 10 + ["finally an answer [1]"])
    assert card.capped
    assert len(card.ledger.calls) == steward.MAX_TOOL_CALLS
    assert card.answer.startswith("(Tool-call limit reached")
    assert card.citations == card.ledger.hits
