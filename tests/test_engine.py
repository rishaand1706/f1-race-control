"""Stage 3's claims, locked in.

Three things are worth a test here, and they are not the ranking quality - that is
measured, not asserted (`python -m engine.retrieve --check`). What these lock in is the
behaviour that would break silently:

  - classification, because every filter downstream reads its output
  - citation binding, because a citation bug produces plausible-looking wrong output
  - the metadata filters, because the temporal bug is invisible when it happens

No Ollama and no index needed for the first two; the filter tests build a Retriever from
fake rows rather than loading 30MB.

    python -m pytest tests/test_engine.py -q
"""

from __future__ import annotations

import pytest

from engine.detect import (
    INVESTIGATING, LAP_DELETED, NO_FURTHER_ACTION, NOTED, PENALTY_ISSUED,
    TRACK_CONDITION, classify,
)
from engine.explain import bind_citations
from engine.retrieve import GUIDELINE, PRECEDENT, RULE, Hit, Retriever


# --- classification ---------------------------------------------------------

@pytest.mark.parametrize("message, stage", [
    ("TURN 4 INCIDENT INVOLVING CARS 1 (NOR) AND 44 (HAM) NOTED - CAUSING A COLLISION", NOTED),
    ("INCIDENT INVOLVING CAR 16 (LEC) UNDER INVESTIGATION - TRACK LIMITS", INVESTIGATING),
    ("CAR 4 (NOR) TIME 1:21.334 LAP 12 DELETED - TRACK LIMITS AT TURN 4", LAP_DELETED),
    ("FIA STEWARDS: 5 SECOND TIME PENALTY FOR CAR 11 (PER) - CAUSING A COLLISION", PENALTY_ISSUED),
    ("SAFETY CAR DEPLOYED", TRACK_CONDITION),
])
def test_stage_classification(message, stage):
    assert classify(message).stage == stage


def test_specific_stages_beat_the_general_ones_they_contain():
    """Ordering bugs here are the whole reason STAGE_PATTERNS is a list, not a dict.

    'NO FURTHER INVESTIGATION' contains 'INVESTIGATION'; a penalty served is not a penalty
    issued. Both would be misfiled by naive substring order.
    """
    assert classify(
        "INCIDENT INVOLVING CAR 1 (VER) - NO FURTHER INVESTIGATION"
    ).stage == NO_FURTHER_ACTION
    assert classify("CAR 44 (HAM) 5 SECOND PENALTY SERVED").stage != PENALTY_ISSUED


def test_two_car_incident_extracts_both_cars_and_drivers():
    f = classify("TURN 4 INCIDENT INVOLVING CARS 1 (NOR) AND 44 (HAM) NOTED - CAUSING A COLLISION")
    assert f.cars == (1, 44)
    assert f.drivers == ("NOR", "HAM")
    assert f.turn == 4
    assert f.offence == "Causing A Collision"


def test_penalty_magnitude_is_parsed():
    f = classify("FIA STEWARDS: 10 SECOND TIME PENALTY FOR CAR 81 (PIA) - CAUSING A COLLISION")
    assert (f.penalty, f.penalty_seconds) == ("time_penalty", 10)


def test_bare_car_reference_is_not_an_offence():
    """'PENALTY FOR CAR 11 (PER)' splits on ' - ' into a car, not an offence phrase."""
    assert classify("FIA STEWARDS: DRIVE THROUGH PENALTY FOR CAR 11 (PER)").offence is None


def test_blue_flags_are_not_explainable():
    """510 of 1,228 messages in six 2026 races were waved blue flags. This is the filter
    that stops half the compute budget going on 'a backmarker is being lapped'."""
    assert not classify("WAVED BLUE FLAG FOR CAR 5 (BOR)").explainable
    assert classify("INCIDENT INVOLVING CAR 5 (BOR) NOTED - TRACK LIMITS").explainable


def test_signature_ignores_lap_but_not_cars():
    """The cache key is the identity of the situation. Same offence, same corner, same
    driver on a different lap is one situation; a different driver is not."""
    a = classify("TURN 4 INCIDENT INVOLVING CAR 1 (VER) NOTED - TRACK LIMITS", lap=12)
    b = classify("TURN 4 INCIDENT INVOLVING CAR 1 (VER) NOTED - TRACK LIMITS", lap=40)
    c = classify("TURN 4 INCIDENT INVOLVING CAR 44 (HAM) NOTED - TRACK LIMITS", lap=12)
    assert a.signature == b.signature
    assert a.signature != c.signature


# --- citation binding -------------------------------------------------------

def _hits(n: int) -> list[Hit]:
    return [Hit(pid=i, kind="sporting", ref=f"REF-{i}", article=f"A{i}",
                title="t", text="x", score=1.0) for i in range(1, n + 1)]


def test_citations_resolve_and_renumber_by_first_use():
    text, cites, dropped = bind_citations("Breach of the rule [3]. Precedent agrees [1].", _hits(3))
    assert text == "Breach of the rule [1]. Precedent agrees [2]."
    assert [h.ref for h in cites] == ["REF-3", "REF-1"]
    assert dropped == []


def test_invented_citations_are_deleted_not_rendered():
    """The load-bearing test. A marker pointing at a source that was never supplied
    launders a hallucination into something that looks checkable, so it is removed."""
    text, cites, dropped = bind_citations("This is true [1] and this is invented [7].", _hits(2))
    assert "[7]" not in text
    assert text == "This is true [1] and this is invented."
    assert dropped == [7]
    assert len(cites) == 1


def test_repeated_marker_maps_to_one_source():
    text, cites, _ = bind_citations("First [2], and again [2].", _hits(2))
    assert text == "First [1], and again [1]."
    assert len(cites) == 1


def test_no_citations_is_not_an_error():
    text, cites, dropped = bind_citations("An ungrounded sentence.", _hits(2))
    assert (text, cites, dropped) == ("An ungrounded sentence.", [], [])


# --- metadata filters -------------------------------------------------------

class FakeRetriever(Retriever):
    """Rows only. Bypasses __init__ so no index or DuckDB is touched."""

    # pid, kind, season, ref, article, title, text, effective_from, effective_to, url,
    # published
    ROWS = [
        (0, "sporting", 2025, "2025 Issue 4", "33.3", "t", "x", "2025-01-01", "2025-07-01", None, None),
        (1, "sporting", 2025, "2025 Issue 5", "33.3", "t", "x", "2025-07-01", None, None, None),
        (2, "sporting", 2026, "2026 Issue 8", "B1.8.4", "t", "x", "2026-01-01", None, None, None),
        (3, "penalty_guidelines", 2026, "PG", None, "t", "x", None, None, None, None),
        (4, "driving_standards", 2026, "DSG", None, "t", "x", None, None, None, None),
        (5, "decision", 2025, "Doc 12", None, "t", "x", None, None, None, "2025-06-01T18:00"),
        # Race day, 2026-07-05: one ruling from qualifying, one on the race incident itself.
        (6, "decision", 2026, "Doc 40", None, "t", "x", None, None, None, "2026-07-04T17:10"),
        (7, "decision", 2026, "Doc 71", None, "t", "x", None, None, None, "2026-07-05T16:32"),
        (8, "decision", 2026, "Doc 99", None, "t", "x", None, None, None, None),
    ]

    def __init__(self):
        self.rows = self.ROWS


class FakeQuery:
    def __init__(self, season, date_iso, asof=""):
        self.season, self.date_iso = season, date_iso
        self.asof = asof or f"{date_iso}T00:00"


@pytest.fixture
def r():
    return FakeRetriever()


def test_rule_slot_takes_only_the_issue_in_force_that_day(r):
    """The temporal filter. Both issues are 2025 and both contain Article 33.3; only the
    one in force on the incident's own date may answer for it."""
    assert r._candidates(RULE, FakeQuery(2025, "2025-03-01")) == [0]
    assert r._candidates(RULE, FakeQuery(2025, "2025-08-01")) == [1]


def test_rule_slot_never_crosses_a_season(r):
    """The failure that has no symptom: a 2026 rulebook answering a 2025 incident."""
    assert 2 not in r._candidates(RULE, FakeQuery(2025, "2025-08-01"))


def test_guideline_slot_spans_both_guideline_kinds(r):
    assert r._candidates(GUIDELINE, FakeQuery(2026, "2026-08-01")) == [3, 4]


def test_precedent_slot_is_decisions_only_and_crosses_seasons(r):
    """Precedent is the one slot where an older season is the point, not a bug."""
    assert r._candidates(PRECEDENT, FakeQuery(2026, "2026-08-01")) == [5, 6, 7]


def test_a_ruling_published_after_the_message_is_not_a_precedent(r):
    """The leak Stage 4's replay exposed. At 15:04 the incident is only noted; the ruling
    on it (Doc 71) is published at 16:32. Before this filter, the card for the 15:04
    message retrieved Doc 71 and announced the penalty an hour and a half early."""
    during = r._candidates(PRECEDENT, FakeQuery(2026, "2026-07-05", "2026-07-05T15:04"))
    after = r._candidates(PRECEDENT, FakeQuery(2026, "2026-07-05", "2026-07-05T17:00"))
    assert 7 not in during
    assert 6 in during, "an earlier ruling from the same weekend is a legitimate precedent"
    assert 7 in after


def test_unknown_message_time_hides_the_whole_day_and_undated_rulings(r):
    """No time of day: err towards hiding a same-day precedent, never towards leaking."""
    got = r._candidates(PRECEDENT, FakeQuery(2026, "2026-07-05"))
    assert got == [5, 6]


def test_empty_candidate_set_ranks_to_nothing_rather_than_raising(r):
    assert r._rank([], "anything", "anything", 3) == []


# --- the vector store -------------------------------------------------------

def test_dense_channel_returns_the_passage_whose_vector_matched(tmp_path, monkeypatch):
    """The vector lives in its passage's own row (docs/concepts/10-vector-store.md).
    Passages go in out of pid order and with gaps in the ids, which a matrix paired with
    the table by position could not survive, and the dense channel must still score and
    return the passage whose vector matched. A real DuckDB file; no Ollama."""
    import pickle

    import duckdb
    import numpy as np
    from rank_bm25 import BM25Okapi

    import config
    import engine.retrieve as retrieve

    def unit(axis):
        v = np.zeros(config.EMBED_DIM, dtype=np.float32)
        v[axis] = 1.0
        return v

    db = tmp_path / "corpus.duckdb"
    con = duckdb.connect(str(db))
    con.execute(f"""
        CREATE TABLE passages (
            pid INTEGER PRIMARY KEY, kind TEXT, season INTEGER, ref TEXT, article TEXT,
            title TEXT, text TEXT, effective_from TEXT, effective_to TEXT, url TEXT,
            published TEXT, embedding FLOAT[{config.EMBED_DIM}]
        )
    """)
    for pid, axis in [(42, 2), (7, 0), (19, 1)]:
        con.execute("INSERT INTO passages VALUES (?, 'sporting', 2026, ?, ?, 't', 'x', "
                    "NULL, NULL, NULL, NULL, ?)", [pid, f"Doc {pid}", str(pid), unit(axis).tolist()])
    con.close()
    bm25 = tmp_path / "bm25.pkl"
    bm25.write_bytes(pickle.dumps(BM25Okapi([["x"], ["x"], ["x"]])))

    monkeypatch.setattr(config, "DB", db)
    monkeypatch.setattr(retrieve, "BM25_PKL", bm25)
    monkeypatch.setattr(retrieve, "embed_query", lambda text: unit(1))

    r = retrieve.Retriever()
    assert [h.pid for h in r.dense_only("anything", k=1)] == [19]
    scores = r._dense(list(range(len(r.rows))), "anything")
    assert dict(zip((row[0] for row in r.rows), scores.tolist())) == {7: 0.0, 19: 1.0, 42: 0.0}
