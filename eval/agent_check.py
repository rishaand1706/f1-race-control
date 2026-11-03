"""The agent against the pipeline, on the decisions `engine.retrieve --check` uses.

Same 120 decisions (seed 0), same query (the stewards' own description of what happened,
article references stripped), same ground truth (the Sporting Regulations article they
cited). The pipeline's number is recomputed here on exactly these rows, so the comparison
is like for like rather than against a figure from another run.

Per card, it records whether the agent searched at all, whether its first rule search
found the cited article in the top 5, which rulebook and cut-off its searches ran with,
whether precedent search returned the ruling on this very incident, the citations it
invented, and where the time went (OpenTelemetry spans, agent/telemetry.py).

Not held out, for the reason `--check` is not: Stage 5 evaluates on the same pool.

Usage:
    python -m eval.agent_check --variant naive --sample 120
    python -m eval.agent_check --variant guarded --sample 120
    python -m eval.agent_check --summary data/eval/agent-guarded-*.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
from datetime import datetime
from pathlib import Path

from agent import telemetry  # before semantic_kernel
from agent import steward
from agent.tools import RULE
import config
from corpus.fetch_regs import in_force
from engine import query as qmod
from engine.detect import classify
from engine.retrieve import LEAK_RE, Retriever, _date_iso

OUT = config.DATA / "eval"


def rows(sample: int, seed: int = 0) -> list[tuple]:
    """The --check sample, in the same order, plus each decision's URL."""
    import duckdb
    con = duckdb.connect(str(config.DB), read_only=True)
    got = con.execute("""
        SELECT season, event, doc_date, fact,
               list_transform(list_filter(articles, x -> x.source = 'sporting'),
                              x -> x.article) AS want, url
        FROM decisions
        WHERE len(want) > 0 AND fact IS NOT NULL AND length(fact) > 20
    """).fetchall()
    con.close()
    # Same query, same seed, same list length: shuffle's permutation depends on nothing
    # else, so row i here is row i in engine.retrieve.check.
    random.Random(seed).shuffle(got)
    return got[:sample]


def issue(season: int, date_iso: str) -> str | None:
    try:
        reg = in_force(int(season), "sporting", str(date_iso))
    except Exception:
        return None
    return reg.issue if reg else None


async def run(variant: str, sample: int, path: Path) -> None:
    memory = telemetry.setup(to_file=False)
    retriever = Retriever()
    published = {r[9]: r[10] for r in retriever.rows if r[1] == "decision"}
    service = steward.chat_service()

    with path.open("w", encoding="utf-8") as out:
        for i, (season, event, doc_date, fact, want, url) in enumerate(rows(sample)):
            clean = LEAK_RE.sub("", " ".join(str(fact).split())).strip()
            if len(clean) < 15:
                continue
            date_iso = _date_iso(doc_date, season)
            q = qmod.build(classify(f"INCIDENT NOTED - {clean.upper()}"),
                           season=season, date_iso=date_iso)
            want_set = set(want)

            # The pipeline on this row, exactly as --check runs it.
            pipe = [h.article for h in retriever._rank(retriever._candidates(RULE, q), clean, clean, 5)]

            card = await steward.explain(clean, q, variant, retriever=retriever, service=service)
            rule_calls = [c for c in card.ledger.calls if c.tool == "search_rules"]
            prec_calls = [c for c in card.ledger.calls if c.tool == "search_precedents"]
            true_issue = issue(season, date_iso)

            rec = {
                "i": i, "event": event, "season": season, "date": date_iso, "want": sorted(want_set),
                "pipeline_hit5": bool(want_set & set(pipe)),
                "searched": card.grounded, "retried": card.retried, "capped": card.capped,
                "calls": {t: sum(c.tool == t for c in card.ledger.calls)
                          for t in ("search_rules", "search_guidelines", "search_precedents")},
                "first_rule_query": rule_calls[0].args.get("query") if rule_calls else None,
                "agent_hit5_first": bool(rule_calls and want_set & set(rule_calls[0].articles)),
                "agent_hit5_any": any(want_set & set(c.articles) for c in rule_calls),
                "rule_filters": [{"season": c.season, "date": c.date_iso,
                                  "wrong_rulebook": issue(c.season, c.date_iso) != true_issue}
                                 for c in rule_calls],
                "self_leak": any(url in c.urls for c in prec_calls),
                "future_leak": any((published.get(u) or "") >= q.asof
                                   for c in prec_calls for u in c.urls if u),
                "dropped": card.dropped, "unsupported": card.unsupported,
                "answer": card.answer, "error": card.error,
                **telemetry.breakdown(memory.get_finished_spans(), card.trace_id),
            }
            memory.clear()
            out.write(json.dumps(rec) + "\n")
            out.flush()
            print(f"  {i + 1:3d}/{sample}  searched={card.grounded!s:5}  "
                  f"hit5={rec['agent_hit5_first']!s:5}  pipe={rec['pipeline_hit5']!s:5}  "
                  f"{card.ms / 1000:5.1f}s", flush=True)


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:5.1f}%  ({n}/{d})" if d else "  n/a"


def summary(paths: list[Path]) -> None:
    for path in paths:
        recs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        n = len(recs)
        if not n:
            continue
        searched = [r for r in recs if r["searched"]]
        rule_calls = [f for r in recs for f in r["rule_filters"]]
        prec = [r for r in recs if r["calls"]["search_precedents"]]
        card_ms = sorted(r.get("card_ms", 0) for r in recs)
        chat_n = [r.get("chat_n", 0) for r in recs]
        share = [r.get("chat_ms", 0) / r["card_ms"] for r in recs if r.get("card_ms")]

        print(f"\n{path.name}: {n} decisions")
        print(f"  answered after a rule search      {pct(len(searched), n)}")
        print(f"  ...only after the retry guard     {pct(sum(r['retried'] and r['searched'] for r in recs), n)}")
        print(f"  rule hit@5, agent's first search  {pct(sum(r['agent_hit5_first'] for r in recs), n)}")
        print(f"  rule hit@5, any agent search      {pct(sum(r['agent_hit5_any'] for r in recs), n)}")
        print(f"  rule hit@5, pipeline, same rows   {pct(sum(r['pipeline_hit5'] for r in recs), n)}")
        print(f"  rule searches, wrong rulebook     {pct(sum(f['wrong_rulebook'] for f in rule_calls), len(rule_calls))}")
        print(f"  precedent searches run            {pct(len(prec), n)}")
        print(f"  ...returned this incident's ruling {pct(sum(r['self_leak'] for r in prec), len(prec))}")
        print(f"  cards with an invented marker     {pct(sum(bool(r['dropped']) for r in recs), n)}")
        print(f"  cards naming an unretrieved article {pct(sum(bool(r['unsupported']) for r in recs), n)}")
        print(f"  tool-call cap reached             {pct(sum(r['capped'] for r in recs), n)}")
        print(f"  errors                            {pct(sum(bool(r['error']) for r in recs), n)}")
        print(f"  card time p50 / p95               {card_ms[n // 2] / 1000:.1f}s / "
              f"{card_ms[min(n - 1, int(0.95 * n))] / 1000:.1f}s")
        print(f"  model calls per card, mean        {statistics.mean(chat_n):.2f}")
        if share:
            print(f"  share of card time in the model   {100 * statistics.mean(share):.0f}%")
    print("\nNot held out - same pool as engine.retrieve --check and Stage 5.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", choices=[steward.NAIVE, steward.GUARDED])
    p.add_argument("--sample", type=int, default=120)
    p.add_argument("--summary", nargs="*", type=Path)
    args = p.parse_args()

    if args.summary is not None:
        summary(args.summary or sorted(OUT.glob("agent-*.jsonl")))
        return
    if not args.variant:
        sys.exit("give --variant or --summary")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"agent-{args.variant}-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    asyncio.run(run(args.variant, args.sample, path))
    summary([path])


if __name__ == "__main__":
    main()
