"""Filtered hybrid retrieval: three searches, two channels, one fusion rule.

**Three searches, not one.** The card needs three different things - the rule that was
breached, the FIA's own recommended sanction, and what happened to people who did this
before. A single top-k over the whole index cannot deliver that, and not because of
ranking quality: 2,776 sporting regulation chunks would simply crowd out the 110 penalty
guideline entries every time. Faceted retrieval means running one filtered search per slot
the answer has.

**Filter first, then rank.** Every search here narrows the candidate set with metadata
before a single score is computed. That is a pre-filter, and it is both faster and more
accurate than ranking everything and discarding afterwards - see the note on post-filter
in docs/concepts/06-hybrid-retrieval.md, and the temporal demo below for what happens
without it.

**Two channels, fused with RRF.** Dense handles paraphrase, BM25 handles identifiers, and
Stage 1 measured them failing on disjoint queries. Reciprocal rank fusion combines them
without ever comparing their scores, which is the only honest way to do it.

Usage:
    python -m engine.retrieve "causing a collision at turn 4"
    python -m engine.retrieve --demo        # the two failures this stage exists to fix
    python -m engine.retrieve --leak        # precedents from the future, before Stage 4
"""

from __future__ import annotations

import argparse
import pickle
import re
import sys
from dataclasses import dataclass

import duckdb
import numpy as np

import config
from corpus.build_index import VECTORS, BM25_PKL, tokenize
from engine.embed import embed_query
from engine.query import Query

# Slots in the explanation card, and the filter that fills each one.
RULE, GUIDELINE, PRECEDENT = "rule", "guideline", "precedent"


@dataclass
class Hit:
    pid: int
    kind: str
    ref: str
    article: str | None
    title: str
    text: str
    score: float
    dense_rank: int | None = None
    lexical_rank: int | None = None
    url: str | None = None

    def cite(self) -> str:
        return self.ref

    def channels(self) -> str:
        d = f"d{self.dense_rank}" if self.dense_rank is not None else "d-"
        l = f"l{self.lexical_rank}" if self.lexical_rank is not None else "l-"
        return f"{d}/{l}"


class Retriever:
    """Holds the index in memory. Built once, queried per event.

    Loading is ~30MB and a few hundred milliseconds; doing it per query would blow the
    latency budget on its own, which is why this is a class and not a function.
    """

    def __init__(self) -> None:
        if not VECTORS.exists():
            raise SystemExit("No index. Run: python -m corpus.build_index")
        con = duckdb.connect(str(config.DB), read_only=True)
        self.rows = con.execute("""
            SELECT pid, kind, season, ref, article, title, text,
                   effective_from, effective_to, url, published
            FROM passages ORDER BY pid
        """).fetchall()
        con.close()
        self.vectors = np.load(VECTORS)
        self.bm25 = pickle.loads(BM25_PKL.read_bytes())

    # --- filtering ----------------------------------------------------------

    def _candidates(self, slot: str, q: Query) -> list[int]:
        """Row indices surviving the metadata filter for this slot."""
        out: list[int] = []
        for i, r in enumerate(self.rows):
            kind, season, eff_from, eff_to = r[1], r[2], r[7], r[8]

            if slot == RULE:
                if kind != "sporting" or season != q.season:
                    continue
                # The temporal filter, in two lines. The 2026 Sporting Regulations were
                # reissued five times inside the season; an incident is governed by the
                # issue in force on its own date, not by the newest PDF on disk.
                if eff_from and q.date_iso < eff_from:
                    continue
                if eff_to and q.date_iso >= eff_to:
                    continue
            elif slot == GUIDELINE:
                if kind not in ("penalty_guidelines", "driving_standards"):
                    continue
            elif slot == PRECEDENT:
                if kind != "decision":
                    continue
                # A precedent has to exist before the message it explains. Without this,
                # replaying a race retrieves the stewards' ruling on the very incident
                # being explained, and the card announces a penalty nobody has given yet.
                published = r[10]
                if not published or published >= q.asof:
                    continue
            out.append(i)
        return out

    # --- ranking ------------------------------------------------------------

    def _rank(self, cand: list[int], text: str, terms: str, k: int) -> list[Hit]:
        """Score the candidates on both channels and fuse by reciprocal rank."""
        if not cand:
            return []

        dense = self.vectors[cand] @ embed_query(text)
        lexical = np.asarray(self.bm25.get_scores(tokenize(terms)))[cand]

        # Fuse deeper than we return: a passage ranked 1st on one channel and 30th on the
        # other should still win, and it cannot if the pools are cut at k first.
        depth = min(len(cand), max(k * 10, 50))
        d_order = np.argsort(-dense)[:depth]
        l_order = np.argsort(-lexical)[:depth]

        d_rank = {int(j): r for r, j in enumerate(d_order, 1)}
        l_rank = {int(j): r for r, j in enumerate(l_order, 1)}

        # Reciprocal rank fusion. Note what is absent: the two channels' scores. Cosine
        # runs 0..1 and BM25 runs 0..30+ on a scale that shifts with the corpus, so any
        # attempt to add them needs a normalisation that is itself a guess. RRF uses only
        # rank, which both channels agree on the meaning of.
        fused: dict[int, float] = {}
        for j, r in d_rank.items():
            fused[j] = fused.get(j, 0.0) + 1.0 / (config.RRF_K + r)
        for j, r in l_rank.items():
            fused[j] = fused.get(j, 0.0) + 1.0 / (config.RRF_K + r)

        hits: list[Hit] = []
        for j, score in sorted(fused.items(), key=lambda kv: -kv[1])[:k]:
            row = self.rows[cand[j]]
            hits.append(Hit(
                pid=row[0], kind=row[1], ref=row[3], article=row[4],
                title=row[5], text=row[6], url=row[9],
                score=score, dense_rank=d_rank.get(j), lexical_rank=l_rank.get(j),
            ))
        return hits

    # --- public -------------------------------------------------------------

    def search(self, q: Query, k: int = 3) -> dict[str, list[Hit]]:
        """One filtered, fused search per card slot."""
        # The guideline slot is queried with the bare offence name. Its entries are
        # literally titled "Causing a collision", so the enriched prose that helps
        # precedent search actively hurts here.
        offence = q.facets.offence or q.text
        return {
            RULE: self._rank(self._candidates(RULE, q), q.text, q.terms, k),
            GUIDELINE: self._rank(self._candidates(GUIDELINE, q), offence, offence, k),
            PRECEDENT: self._rank(self._candidates(PRECEDENT, q), q.text, q.terms, k),
        }

    def dense_only(self, text: str, k: int = 3, kind: str | None = None) -> list[Hit]:
        """No fusion, no filter. Kept so the failure stays reproducible."""
        cand = [i for i, r in enumerate(self.rows) if kind is None or r[1] == kind]
        scores = self.vectors[cand] @ embed_query(text)
        return [
            Hit(pid=self.rows[cand[j]][0], kind=self.rows[cand[j]][1],
                ref=self.rows[cand[j]][3], article=self.rows[cand[j]][4],
                title=self.rows[cand[j]][5], text=self.rows[cand[j]][6],
                url=self.rows[cand[j]][9], score=float(scores[j]))
            for j in np.argsort(-scores)[:k]
        ]


# --- the staged failures ----------------------------------------------------

def demo() -> None:
    """The two bugs this stage exists to fix, both reproduced on real data."""
    from engine.detect import classify
    from engine import query as qmod

    r = Retriever()

    print("=" * 78)
    print("FAILURE 1: dense retrieval cannot look up an identifier")
    print("=" * 78)
    print('query: "Article 33.4"   (2025 track limits)\n')
    print("  dense only, no lexical channel:")
    for h in r.dense_only("Article 33.4", k=3, kind="sporting"):
        print(f"    [{h.score:.3f}] {h.ref:52s} {h.title[:44]}")
    print("\n  hybrid, RRF fused:")
    facets = classify("INCIDENT NOTED - TRACK LIMITS")
    q = qmod.build(facets, season=2025, date_iso="2025-08-01")
    q.text, q.terms = "Article 33.4 track limits", "article 33.4 track limits"
    for h in r._rank(r._candidates(RULE, q), q.text, q.terms, 3):
        print(f"    [{h.score:.4f} {h.channels()}] {h.ref:52s} {h.title[:40]}")

    print()
    print("=" * 78)
    print("FAILURE 2: no temporal filter, so the wrong rulebook answers")
    print("=" * 78)
    msg = "TURN 4 INCIDENT INVOLVING CAR 44 (HAM) NOTED - LEAVING THE TRACK AND GAINING AN ADVANTAGE"
    facets = classify(msg)
    print(f'incident: "{msg[:70]}..."')
    print("         judged as if it happened on 2025-08-01\n")

    q25 = qmod.build(facets, season=2025, date_iso="2025-08-01")
    print("  unfiltered - any season, any issue:")
    for h in r.dense_only(q25.text, k=3, kind="sporting"):
        print(f"    [{h.score:.3f}] {h.ref}")
    print(f"\n  filtered to the issue in force on 2025-08-01 ({q25.reg_issue}):")
    for h in r._rank(r._candidates(RULE, q25), q25.text, q25.terms, 3):
        print(f"    [{h.score:.4f} {h.channels()}] {h.ref}")

    print("\nThe unfiltered answer is not merely a worse ranking. It cites a rulebook that "
          "did not\nexist when the incident happened, in a system whose entire value is "
          "citing the right one.")


# --- the acceptance check ---------------------------------------------------

# "Alleged breach of Article B5.2.4" - if that reaches the query, the check measures
# nothing. The `fact` field is normally clean prose, but this is not left to trust.
LEAK_RE = re.compile(r"\b(?:article|art\.?|appendix|chapter)\b[^.;,]{0,40}", re.I)


def check(sample: int = 60, k: int = 5, seed: int = 0) -> None:
    """Did retrieval find the article the stewards actually cited?

    Stage 3's acceptance test, run on real decisions. Each one states the article it
    breached, so the ground truth is written down rather than judged: take the stewards'
    own description of what happened, hand retrieval nothing else, and ask whether the
    article they cited comes back.

    **This is a smoke check, not the evaluation.** The sample is drawn from the same
    decisions Stage 5 will evaluate on, so it is not held out, and a number produced this
    way can be reported inside the repo and nowhere else. Stage 5 builds the golden set.
    """
    import random
    import duckdb

    con = duckdb.connect(str(config.DB), read_only=True)
    rows = con.execute("""
        SELECT season, event, doc_date, fact,
               list_transform(list_filter(articles, x -> x.source = 'sporting'),
                              x -> x.article) AS want
        FROM decisions
        WHERE len(want) > 0 AND fact IS NOT NULL AND length(fact) > 20
    """).fetchall()
    con.close()

    random.Random(seed).shuffle(rows)
    rows = rows[:sample]

    r = Retriever()
    from engine.detect import classify
    from engine import query as qmod

    hybrid = [0] * k
    dense = [0] * k
    scored = leaked = 0

    for season, event, doc_date, fact, want in rows:
        text = " ".join(str(fact).split())
        clean = LEAK_RE.sub("", text).strip()
        if clean != text:
            leaked += 1
        if len(clean) < 15:
            continue

        date_iso = _date_iso(doc_date, season)
        q = qmod.build(classify(f"INCIDENT NOTED - {clean.upper()}"),
                       season=season, date_iso=date_iso)
        # The offence phrase parsed out of a synthetic message is a poor stand-in for the
        # stewards' full description, so the description itself is the query.
        q.text, q.terms = clean, clean

        want_set = set(want)
        got = [h.article for h in r._rank(r._candidates(RULE, q), q.text, q.terms, k)]
        base = [h.article for h in r.dense_only(q.text, k=k, kind="sporting")]

        scored += 1
        for i in range(k):
            if want_set & set(got[:i + 1]):
                hybrid[i] += 1
            if want_set & set(base[:i + 1]):
                dense[i] += 1

    if not scored:
        print("no scorable decisions")
        return

    print(f"\nStage 3 check - {scored} real decisions, query = the stewards' own "
          f"description only")
    print(f"(article references stripped from {leaked} of them before querying)\n")
    print(f"  {'':6s}  {'hybrid + filtered':>18s}  {'dense only, unfiltered':>23s}")
    for i in range(k):
        print(f"  hit@{i + 1:<2d}  {100 * hybrid[i] / scored:17.1f}%  "
              f"{100 * dense[i] / scored:22.1f}%")
    print("\nNot held out - same pool Stage 5 evaluates on. A sanity check, not a result.")


def leak(k: int = 2) -> None:
    """How often precedent search handed the model a ruling from the future.

    Replays every cached race, and for each card-worthy message ranks precedents the way
    Stage 3 did - every decision in the corpus - then asks whether any of the top k was
    published after the message was issued. Most of those are not merely later: they are
    the stewards' ruling on the very incident being explained, so the card "predicts" a
    penalty by reading the verdict. The time filter in `_candidates` makes the post-fix
    count zero by construction; this measures what it was.
    """
    import logging

    import pandas as pd

    from engine import query as qmod
    from engine.detect import TRACK_CONDITION, classify
    from engine.state import RaceState
    from ingest.events import RACE_CONTROL
    from ingest.replay import build_events, cached_races, load_session, local_start

    logging.getLogger("fastf1").setLevel(logging.ERROR)
    r = Retriever()
    published = {row[0]: row[10] for row in r.rows}
    every_decision = [i for i, row in enumerate(r.rows) if row[1] == "decision"]

    n = future = same_event = 0
    examples = []
    for race in cached_races():
        session = load_session(2026, race["round"], "R", offline=True)
        t0 = local_start(session)
        date_iso = str(session.date.date())
        state = RaceState()
        for e in build_events(session):
            state.apply(e)
            if e.kind != RACE_CONTROL:
                continue
            f = classify(e["message"], lap=e.lap, car=e["car"])
            if not f.explainable or f.stage == TRACK_CONDITION:
                continue
            asof = (t0 + pd.Timedelta(seconds=e.t)).strftime("%Y-%m-%dT%H:%M")
            q = qmod.build(f, state=state, season=2026, date_iso=date_iso, asof=asof)
            hits = r._rank(every_decision, q.text, q.terms, k)
            late = [h for h in hits if (published[h.pid] or "9999") >= asof]
            n += 1
            if late:
                future += 1
                if any(h.ref.startswith(f"2026 {race['name']}") for h in late):
                    same_event += 1
                    if len(examples) < 4:
                        examples.append((asof, f.raw, late[0].ref, published[late[0].pid]))
        print(f"  {race['name']:<24} running total: {future}/{n} leaked")

    print(f"\n{n} card-worthy messages across {len(cached_races())} races, top-{k} precedents")
    print(f"  a precedent published after the message:  {future} ({100 * future / n:.0f}%)")
    print(f"  ...and from the same race weekend:        {same_event} ({100 * same_event / n:.0f}%)")
    print("\nexamples:")
    for asof, msg, ref, pub in examples:
        print(f"  {asof}  {msg[:70]}")
        print(f"      -> {ref}, published {pub}")


def _date_iso(doc_date: str | None, season: int) -> str:
    """'20 September 2025' -> '2025-09-20'. Falls back to mid-season."""
    from datetime import datetime
    if doc_date:
        try:
            return datetime.strptime(str(doc_date).strip(), "%d %B %Y").strftime("%Y-%m-%d")
        except ValueError:
            pass
    return f"{season}-07-01"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("query", nargs="*", help="free-text query")
    p.add_argument("--season", type=int, default=2026)
    p.add_argument("--date", default="2026-08-23")
    p.add_argument("-k", type=int, default=3)
    p.add_argument("--demo", action="store_true")
    p.add_argument("--check", action="store_true", help="article-match against real decisions")
    p.add_argument("--sample", type=int, default=60)
    p.add_argument("--leak", action="store_true",
                   help="how often precedent search retrieved a ruling from the future")
    args = p.parse_args()

    if args.leak:
        leak()
        return

    if args.demo:
        demo()
        return
    if args.check:
        check(sample=args.sample)
        return
    if not args.query:
        raise SystemExit(__doc__)

    from engine.detect import classify
    from engine import query as qmod

    text = " ".join(args.query)
    q = qmod.build(classify(f"INCIDENT NOTED - {text.upper()}"),
                   season=args.season, date_iso=args.date)
    q.text = text
    results = Retriever().search(q, k=args.k)

    print(f'\nquery : "{text}"')
    print(f"season: {args.season}   date: {args.date}   regs in force: {q.reg_issue}\n")
    for slot in (RULE, GUIDELINE, PRECEDENT):
        print(f"{slot.upper()}")
        for h in results[slot] or []:
            print(f"  [{h.score:.4f} {h.channels()}] {h.ref}")
            print(f"       {' '.join(h.text.split())[:130]}")
        if not results[slot]:
            print("  (nothing)")
        print()


if __name__ == "__main__":
    main()
