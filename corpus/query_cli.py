"""Query the index from the terminal. Stage 1's acceptance test.

Both channels are printed side by side rather than fused, on purpose. Fusion is Stage 3,
and the case for it is much more convincing after you have watched each channel fail in
its own characteristic way:

    "car forced another driver off the track"   dense wins - no shared vocabulary
    "Article 33.3"                              BM25 wins - dense has no notion of an
                                                identifier, only of what it resembles

Usage:
    python -m corpus.query_cli "unsafe release in the pit lane"
    python -m corpus.query_cli "B1.1.3" --kind sporting --season 2026 -k 5
"""

from __future__ import annotations

import pickle
import sys

import duckdb
import numpy as np

import config
from corpus.build_index import BM25_PKL, tokenize
from engine.embed import embed_query


def load():
    if not config.DB.exists():
        raise SystemExit("No index. Run: python -m corpus.build_index")
    con = duckdb.connect(str(config.DB), read_only=True)
    # Rows and vectors come out of one query, so row i and vector i are the same passage.
    got = con.execute(
        "SELECT pid, kind, season, ref, article, title, text, embedding FROM passages ORDER BY pid"
    ).fetchall()
    con.close()
    rows = [r[:-1] for r in got]
    vectors = np.array([r[-1] for r in got], dtype=np.float32)
    return rows, vectors, pickle.loads(BM25_PKL.read_bytes())


def search(query: str, k: int = 5, kind: str | None = None, season: int | None = None):
    rows, vectors, bm25 = load()

    # Filter first, rank second. Shrinking the candidate set before scoring is the
    # cheapest precision win available - see docs/concepts/02-metadata-and-versioning.md.
    keep = [
        i for i, r in enumerate(rows)
        if (kind is None or r[1] == kind) and (season is None or r[2] == season)
    ]
    if not keep:
        raise SystemExit("No passages match that filter.")

    # Vectors are unit-norm, so the dot product IS the cosine similarity.
    dense = vectors[keep] @ embed_query(query)
    lexical = np.asarray(bm25.get_scores(tokenize(query)))[keep]

    def top(scores):
        order = np.argsort(-scores)[:k]
        return [(rows[keep[i]], float(scores[i])) for i in order]

    return top(dense), top(lexical)


def render(label: str, hits) -> None:
    print(f"\n{label}")
    print("-" * 78)
    for rank, (row, score) in enumerate(hits, 1):
        _pid, kind, season, ref, article, title, text = row
        head = (title or text.split("\n")[0])[:70]
        print(f" {rank}. [{score:6.3f}] {kind:18s} {ref}")
        print(f"           {head}")
        snippet = " ".join(text.split())[:150]
        print(f"           {snippet}...")


def main() -> None:
    args = [a for a in sys.argv[1:]]
    if not args:
        raise SystemExit(__doc__)

    k, kind, season = 5, None, None
    query_parts: list[str] = []
    i = 0
    while i < len(args):
        if args[i] in ("-k", "--k") and i + 1 < len(args):
            k = int(args[i + 1]); i += 2
        elif args[i] == "--kind" and i + 1 < len(args):
            kind = args[i + 1]; i += 2
        elif args[i] == "--season" and i + 1 < len(args):
            season = int(args[i + 1]); i += 2
        else:
            query_parts.append(args[i]); i += 1

    query = " ".join(query_parts)
    print(f'query: "{query}"'
          + (f"   [kind={kind}]" if kind else "")
          + (f"   [season={season}]" if season else ""))

    dense_hits, lexical_hits = search(query, k=k, kind=kind, season=season)
    render("DENSE  (nomic-embed-text, cosine)", dense_hits)
    render("LEXICAL (BM25)", lexical_hits)

    overlap = {h[0][0] for h in dense_hits} & {h[0][0] for h in lexical_hits}
    print(f"\noverlap between channels: {len(overlap)}/{k}")


if __name__ == "__main__":
    main()
