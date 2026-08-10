"""Local embeddings via Ollama.

The only place in the codebase that talks to the embedding model. Everything goes
through embed_documents() or embed_query() so the task prefixes can never be forgotten
(see docs/concepts/01-embeddings.md for why that matters).

Vectors are returned L2-normalised, which means cosine similarity is just a dot product
and the whole index becomes one matrix multiply.
"""

from __future__ import annotations

import httpx
import numpy as np

import config

# Ollama handles batching internally, but very large batches risk a timeout on CPU.
BATCH = 64


def _embed(texts: list[str], prefix: str) -> np.ndarray:
    """Embed `texts` with the given task prefix. Returns (n, EMBED_DIM), unit-norm rows."""
    if not texts:
        return np.zeros((0, config.EMBED_DIM), dtype=np.float32)

    out: list[np.ndarray] = []
    with httpx.Client(timeout=300) as client:
        for i in range(0, len(texts), BATCH):
            chunk = [prefix + t for t in texts[i : i + BATCH]]
            r = client.post(
                f"{config.OLLAMA_HOST}/api/embed",
                json={"model": config.EMBED_MODEL, "input": chunk},
            )
            r.raise_for_status()
            out.append(np.array(r.json()["embeddings"], dtype=np.float32))

    v = np.vstack(out)
    # Normalise so dot product == cosine similarity. Guard against zero vectors.
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(norms, 1e-12)


def embed_documents(texts: list[str]) -> np.ndarray:
    """Embed text being stored in the index."""
    return _embed(texts, config.EMBED_PREFIX_DOC)


def embed_query(text: str) -> np.ndarray:
    """Embed a single search query. Returns a 1-D vector of length EMBED_DIM."""
    return _embed([text], config.EMBED_PREFIX_QUERY)[0]
