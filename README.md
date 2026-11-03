# F1 Race Control Copilot

Watches an F1 session's timing feed and, every time race control issues a message,
explains it by retrieving the governing regulation, the FIA's own penalty guideline,
and similar past stewards' decisions. Every claim carries a citation.

```
INCIDENT INVOLVING CARS 1 AND 44 NOTED - TURN 4
        │
        ▼  classify -> enrich with live race state -> retrieve -> ground
┌──────────────────────────────────────────────────┐
│ ⚠  UNDER INVESTIGATION · Lap 34 · Cars 1 & 44    │
│                                                  │
│ The rule    2026 Sporting Regs, Art. 33.3  [src] │
│ Guideline   Penalty Guidelines, p.12       [src] │
│ Precedent   3 similar decisions found      [src] │
└──────────────────────────────────────────────────┘
```

## Why this project exists

It is a RAG project built to be *defensible*, not just demoable. The domain forces four
retrieval patterns that a chat-with-PDF project never touches:

1. **Temporal versioning** - a 2021 incident must retrieve the 2021 rulebook, not the current one
2. **Hybrid lexical/semantic** - "Article 33.3" and "Car 44" are exact tokens; "forced another driver wide" is semantic
3. **Machine-generated queries** - no human types anything; the query is built from a 40-character message plus live race state
4. **Faceted precedent search** - "similar past incidents" is filter-then-rank, not top-k cosine over everything

And it evaluates against real ground truth: stewards' decisions state the article breached
and the penalty given, so we can measure whether the system cites what the stewards cited.

Concept notes for each of these live in [`docs/concepts/`](docs/concepts/).

## Cost

Zero. FastF1, DuckDB and the FIA documents are free; generation and embeddings run
locally in Ollama. No API keys anywhere.

## Setup

```bash
pip install -r requirements.txt
ollama pull nomic-embed-text
```

On this machine the dependencies live in the Anaconda 3.12 install, which is `python`.
`python3` is the Homebrew build and has none of them, so every command below uses
`python`.

## Building the corpus

Each step is idempotent and safe to re-run; already-fetched PDFs and already-computed
embeddings are skipped.

```bash
python -m corpus.fetch_fia manifest     # ~40 HTML requests -> 2,195 document records
python -m corpus.fetch_fia download     # 868 substantive PDFs, fetched once each
python -m corpus.fetch_regs             # 10 rulebook/guideline issues + version registry
python -m corpus.parse_decisions        # -> decisions.jsonl, with per-field parse rates
python -m corpus.parse_regs             # -> reg_chunks.jsonl  (--compare for the demo)
python -m corpus.build_index            # passages + embeddings in DuckDB, BM25 alongside
```

Current corpus: **868 stewards' decisions** across 2025-2026 and **2,896 regulation
chunks** spanning 8 issues of the Sporting Regulations, the Penalty Guidelines and the
Driving Standards Guidelines - **3,764 passages** in total.

DuckDB is the vector store: each passage's text, metadata and 768-dimensional embedding
sit in one row, and dense search is a SQL query over them. Why that replaced a NumPy
matrix kept beside the table is in
[`docs/concepts/10-vector-store.md`](docs/concepts/10-vector-store.md).

## Querying

```bash
python -m corpus.query_cli "unsafe release in the pit lane"
python -m corpus.query_cli "Article 33.3" -k 5
python -m corpus.query_cli "B1.8.4b" --kind sporting --season 2026
```

Both retrieval channels are printed separately, unfused, because the way each one fails
is the argument for Stage 3. `"Article 33.3"` is the case worth running: the dense
channel confidently returns Article 13.3, third-party insurance, while BM25 returns the
track-limits decisions. See [`docs/concepts/04-lexical-retrieval.md`](docs/concepts/04-lexical-retrieval.md).

## Replaying a session

A finished session, re-emitted as a live event stream at a chosen speed. Same `Event` type
live mode will produce in Stage 6, so everything downstream is written once.

```bash
python -m ingest.replay --round 12 --speed 900              # paced against a wall clock
python -m ingest.replay --round 12 --speed 0 --only race_control
python -m ingest.replay --round 12 --from-lap 40 --max 30
python -m ingest.replay --round 12 --verify-clock
python -m pytest tests/test_replay.py -q
```

The two feeds inside one session disagree on time by **three and a half minutes**, because
race control messages are absolute UTC while lap data is a timedelta on a session timeline
whose origin is only implied. The offset is measured from events appearing in both feeds,
and verified against timestamps race control prints inside its own messages: **111 of them
in the Dutch GP, all within one second**. It is not a constant, either - 208.4s at Zandvoort,
197.2s at the Red Bull Ring. See [`docs/concepts/05-event-streams.md`](docs/concepts/05-event-streams.md).

## Explaining an incident

The stage that turns retrieval into a card. Classify the message into facets, enrich it
with live race state, run one filtered search per slot the card has, then ground a local
model on those passages and bind every citation it writes.

```bash
python -m engine.detect --round 12                 # message -> facets, and the traffic breakdown
python -m engine.retrieve "unsafe release in the pit lane"
python -m engine.retrieve --demo                   # the two failures this stage fixes
python -m engine.retrieve --check --sample 120     # article-match against real decisions
python -m engine.explain --round 12 --max 5        # full cards, needs Ollama
python -m pytest tests/test_engine.py -q
```

**Retrieval is three searches, not one.** 110 penalty-guideline entries cannot compete
with 2,776 regulation chunks in a single top-k; they get crowded out by volume regardless
of relevance. Each search pre-filters on metadata, then fuses BM25 and dense results with
reciprocal rank fusion at `K=10` rather than the literature's 60, which flattens two
strong channels into consensus mediocrity.

Measured with `--check` over **120 real decisions**, given only the stewards' own
description of the incident:

| | hybrid + filtered | dense only, unfiltered |
|---|---|---|
| hit@1 | **49.2%** | 8.3% |
| hit@3 | **58.3%** | 21.7% |
| hit@5 | **64.2%** | 25.8% |

Not held out - it draws from the same pool Stage 5 evaluates on, so it is a sanity check
on plumbing rather than a result. The 36% miss at k=5 is mostly a corpus gap, not a
ranking one: 532 of the articles stewards cite are in the International Sporting Code and
its Appendix L, which are not in the corpus at all.

Two things measured here came out against the design and are recorded that way. Asking
Ollama for `think: False` does not stop qwen3 reasoning, it only removes the `<think>`
tags - so the reasoning arrived untagged inside the answer and the regex written to strip
it had nothing to match. And the situation cache returns an **11% hit rate** over 542 real
messages, not the burst-heavy traffic assumed, because the cache key contains car numbers
and the expensive half of the work is precisely the car-specific half. Both are written up
in [`docs/concepts/07-grounding-and-citations.md`](docs/concepts/07-grounding-and-citations.md).

Cards are correct but slow: retrieval runs 0.4-1.5s, generation 43-114s, because ~92% of
the generated tokens are reasoning that gets discarded. That is Stage 4's problem, now with
numbers attached.

## Data sources and licensing

- **FIA decision documents, regulations and guidelines** - public PDFs from fia.com, fetched
  once each and cached locally. Cited, not redistributed.
- **FastF1** - timing, lap and race control data.
- **Live timing** is F1 commercial data on an unofficial endpoint. Live mode is local only;
  the public demo runs on recorded replay, and no timing data is committed or rehosted.
