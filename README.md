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

## Data sources and licensing

- **FIA decision documents, regulations and guidelines** - public PDFs from fia.com, fetched
  once each and cached locally. Cited, not redistributed.
- **FastF1** - timing, lap and race control data.
- **Live timing** is F1 commercial data on an unofficial endpoint. Live mode is local only;
  the public demo runs on recorded replay, and no timing data is committed or rehosted.
