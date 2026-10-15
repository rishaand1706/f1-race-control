"""Build the retrieval index: one passage table, two search channels.

Decisions and regulation chunks are different documents with different metadata, but
retrieval wants one surface to filter and rank over, so both are flattened into a single
`passages` table carrying every field the query layer filters on (season, kind, article,
the effective_from/effective_to window, car, penalty type).

Two channels are built over the same passages:

    dense   nomic-embed-text vectors, L2-normalised, stored as a flat (n, 768) matrix
    lexical BM25 over a tokeniser that keeps "33.4" and "b1.1.3" as single tokens

At a few thousand passages a flat exhaustive dense index is the right answer, not a
concession. One matrix multiply scores the whole corpus in single-digit milliseconds.
HNSW and IVF buy sublinear search at the cost of recall and build complexity, and both
only start paying at a scale we are two or three orders of magnitude below.

Embeddings are cached by content hash, so re-running after a chunker change re-embeds
only what actually changed.

Usage:
    python -m corpus.build_index
    python -m corpus.build_index --rebuild     # ignore the embedding cache
"""

from __future__ import annotations

import hashlib
import json
import pickle
import re
import sys

import duckdb
import numpy as np
from rank_bm25 import BM25Okapi

import config
from corpus.parse_decisions import DECISIONS
from corpus.parse_regs import REG_CHUNKS
from engine.embed import embed_documents

VECTORS = config.INDEX / "vectors.npy"
BM25_PKL = config.INDEX / "bm25.pkl"
EMBED_CACHE_VEC = config.INDEX / "embed_cache.npy"
EMBED_CACHE_KEYS = config.INDEX / "embed_cache_keys.json"

# Keeps "33.4", "b1.1.3" and "12.2.1" as single tokens instead of shattering them into
# meaningless integers. Article numbers are the highest-signal terms in this corpus and
# they are exactly what the dense channel is worst at, so the lexical channel must not
# destroy them on the way in.
TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _decision_passage(r: dict) -> dict:
    """Flatten one decision into a passage, with the text that gets embedded.

    The embedded text leads with what a query is likely to look like - the event, the
    session and the incident itself - rather than with administrative fields.
    """
    parts = [
        f"{r['season']} {r['event'].replace('_', ' ').title()}",
        f"Session: {r['session']}" if r["session"] else "",
        f"Car {r['car']} {r['driver']}" if r["car"] and r["driver"] else "",
        f"Incident: {r['fact'] or r['subject']}",
        f"Infringement: {r['infringement']}" if r["infringement"] else "",
        f"Decision: {r['decision']}" if r["decision"] else "",
        f"Reason: {r['reason']}" if r["reason"] else "",
    ]
    ref = f"{r['season']} {r['event'].replace('_', ' ').title()}, Doc {r['doc_number']}"
    return {
        "published": _published(r),
        "kind": "decision",
        "season": r["season"],
        "event": r["event"],
        "article": ", ".join(a["id"] for a in r["articles"]) or None,
        "title": r["subject"] or r["fact"] or "",
        "ref": ref,
        "effective_from": None,
        "effective_to": None,
        "doc_type": r["doc_type"],
        "car": r["car"],
        "session": r["session"],
        "penalty_type": r["penalty_type"],
        "articles_cited": json.dumps([a["id"] for a in r["articles"]]),
        "url": r["url"],
        "text": "\n".join(p for p in parts if p),
    }


def _published(r: dict) -> str | None:
    """'05 July 2026' + '17:32' -> '2026-07-05T17:32', circuit local time.

    When the decision became public, and so the earliest moment it can be anybody's
    precedent. Without it, replaying a race retrieves the stewards' ruling on the very
    incident being explained - published an hour after the message it now "predicts".
    """
    from datetime import datetime
    try:
        day = datetime.strptime(str(r["doc_date"]).strip(), "%d %B %Y")
    except (TypeError, ValueError):
        return None
    hhmm = str(r.get("doc_time") or "").strip()
    return f"{day:%Y-%m-%d}T{hhmm}" if re.fullmatch(r"\d{2}:\d{2}", hhmm) else f"{day:%Y-%m-%d}T23:59"


def _reg_passage(c: dict) -> dict:
    kind_label = {
        "sporting": "Sporting Regulations",
        "penalty_guidelines": "Penalty Guidelines",
        "driving_standards": "Driving Standards Guidelines",
    }[c["kind"]]
    ref = f"{c['season']} {kind_label} {c['issue']}, Art. {c['article']}"
    # The article number is repeated into the embedded text on purpose: a query naming
    # "B1.1.3" should have something to match lexically inside the passage itself.
    header = f"{kind_label} {c['issue']} - Article {c['article']}"
    return {
        "kind": c["kind"],
        "season": c["season"],
        "event": None,
        "article": c["article"],
        "title": c["title"],
        "ref": ref,
        "effective_from": c["effective_from"],
        "effective_to": c["effective_to"],
        "doc_type": None,
        "car": None,
        "session": None,
        "penalty_type": None,
        "articles_cited": None,
        "url": None,
        "text": f"{header}\n{c['text']}",
    }


def load_passages() -> list[dict]:
    if not DECISIONS.exists():
        raise SystemExit("No decisions. Run: python -m corpus.parse_decisions")
    if not REG_CHUNKS.exists():
        raise SystemExit("No reg chunks. Run: python -m corpus.parse_regs")

    passages = [_decision_passage(json.loads(l)) for l in DECISIONS.open()]
    passages += [_reg_passage(json.loads(l)) for l in REG_CHUNKS.open()]
    for i, p in enumerate(passages):
        p["pid"] = i
    return passages


def embed_with_cache(texts: list[str], rebuild: bool = False) -> np.ndarray:
    """Embed, reusing vectors for text we have embedded before.

    Keyed on a hash of the text, not on position: a chunker change reshuffles every
    index but usually leaves most passages byte-identical, and re-embedding those is
    minutes of CPU for no change in the result.
    """
    keys = [hashlib.sha256(t.encode()).hexdigest() for t in texts]

    cache: dict[str, int] = {}
    cached_vecs = np.zeros((0, config.EMBED_DIM), dtype=np.float32)
    if not rebuild and EMBED_CACHE_VEC.exists() and EMBED_CACHE_KEYS.exists():
        cached_vecs = np.load(EMBED_CACHE_VEC)
        cache = {k: i for i, k in enumerate(json.loads(EMBED_CACHE_KEYS.read_text()))}

    todo = [i for i, k in enumerate(keys) if k not in cache]
    print(f"  {len(texts) - len(todo)} cached, {len(todo)} to embed")

    out = np.zeros((len(texts), config.EMBED_DIM), dtype=np.float32)
    for i, k in enumerate(keys):
        if k in cache:
            out[i] = cached_vecs[cache[k]]

    for start in range(0, len(todo), 256):
        batch = todo[start : start + 256]
        out[batch] = embed_documents([texts[i] for i in batch])
        print(f"     embedded {min(start + 256, len(todo))}/{len(todo)}")

    # Rewrite the cache as exactly the current corpus, so it cannot grow without bound
    # across chunker iterations.
    np.save(EMBED_CACHE_VEC, out)
    EMBED_CACHE_KEYS.write_text(json.dumps(keys))
    return out


def build(rebuild: bool = False) -> None:
    passages = load_passages()
    print(f"{len(passages)} passages "
          f"({sum(p['kind'] == 'decision' for p in passages)} decisions, "
          f"{sum(p['kind'] != 'decision' for p in passages)} regulation chunks)\n")

    print("dense channel:")
    vectors = embed_with_cache([p["text"] for p in passages], rebuild=rebuild)
    np.save(VECTORS, vectors)
    print(f"  vectors {vectors.shape} -> {VECTORS.name}")

    print("\nlexical channel:")
    corpus_tokens = [tokenize(p["text"]) for p in passages]
    bm25 = BM25Okapi(corpus_tokens)
    BM25_PKL.write_bytes(pickle.dumps(bm25))
    avg = sum(len(t) for t in corpus_tokens) / len(corpus_tokens)
    print(f"  BM25 over {len(corpus_tokens)} docs, avg {avg:.0f} tokens -> {BM25_PKL.name}")

    print("\nstore:")
    con = duckdb.connect(str(config.DB))
    con.execute("DROP TABLE IF EXISTS passages")
    con.execute("""
        CREATE TABLE passages (
            pid INTEGER PRIMARY KEY, kind TEXT, season INTEGER, event TEXT,
            article TEXT, title TEXT, ref TEXT,
            effective_from TEXT, effective_to TEXT,
            doc_type TEXT, car INTEGER, session TEXT, penalty_type TEXT,
            articles_cited TEXT, url TEXT, text TEXT, published TEXT
        )
    """)
    cols = ["pid", "kind", "season", "event", "article", "title", "ref",
            "effective_from", "effective_to", "doc_type", "car", "session",
            "penalty_type", "articles_cited", "url", "text", "published"]
    con.executemany(
        f"INSERT INTO passages ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
        [[p.get(c) for c in cols] for p in passages],
    )

    # The full decision rows are kept alongside, because Stage 5's evaluation needs the
    # parsed penalty and article fields as ground truth, not the flattened passage text.
    con.execute("DROP TABLE IF EXISTS decisions")
    con.execute(f"CREATE TABLE decisions AS SELECT * FROM read_json_auto('{DECISIONS}')")

    n_pass = con.execute("SELECT count(*) FROM passages").fetchone()[0]
    n_dec = con.execute("SELECT count(*) FROM decisions").fetchone()[0]
    print(f"  passages: {n_pass}")
    print(f"  decisions: {n_dec}")
    print("\n  by kind:")
    for kind, n in con.execute(
        "SELECT kind, count(*) FROM passages GROUP BY kind ORDER BY 2 DESC"
    ).fetchall():
        print(f"     {kind:22s} {n:5d}")
    con.close()
    print(f"\n-> {config.DB}")


if __name__ == "__main__":
    build(rebuild="--rebuild" in sys.argv)
