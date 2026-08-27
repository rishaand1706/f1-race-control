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
python -m corpus.build_index            # embeddings + BM25 + DuckDB
```

Current corpus: **868 stewards' decisions** across 2025-2026 and **2,896 regulation
chunks** spanning 8 issues of the Sporting Regulations, the Penalty Guidelines and the
Driving Standards Guidelines - **3,764 passages** in total.

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

## Data sources and licensing

- **FIA decision documents, regulations and guidelines** - public PDFs from fia.com, fetched
  once each and cached locally. Cited, not redistributed.
- **FastF1** - timing, lap and race control data.
- **Live timing** is F1 commercial data on an unofficial endpoint. Live mode is local only;
  the public demo runs on recorded replay, and no timing data is committed or rehosted.
