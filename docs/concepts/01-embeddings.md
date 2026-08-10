# 01. Embeddings, cosine similarity, and the prefix trap

## What an embedding is

An embedding model maps text to a fixed-length vector (768 numbers for
`nomic-embed-text`). It is trained so texts used in similar contexts land near each
other in that space.

"Near" = small angle between vectors, measured by **cosine similarity**:

```
normalise both to unit length, then take the dot product  ->  roughly -1 .. 1
```

We normalise *first* so that magnitude (which mostly tracks text length and word
frequency) drops out, leaving only direction, which is the semantic content.

## The trap we actually hit

Real output from our first embedding call, no prefixes:

| Pair | Cosine |
|---|---|
| "leaving the track and gaining a lasting advantage" ↔ "causing a collision" | 0.457 |
| "causing a collision" ↔ "pit lane speed limit is 80 km/h" | **0.524** |

Two driving-standards offences scored **less** similar to each other than one did to an
unrelated pit lane rule. Shipping that would have made precedent search silently return
wrong infringements, with plausible-looking output.

**Cause:** `nomic-embed-text` is trained with **task prefixes**. Prepend
`search_document: ` to stored text and `search_query: ` to queries. The prefix tells the
model which task region of its space to use. Without it you are running the model
off-distribution.

With prefixes: 0.640 vs 0.541. Ordering corrected.

Encoded in `config.py` as `EMBED_PREFIX_DOC` / `EMBED_PREFIX_QUERY`.

## Two lessons

**1. Cosine scores are not absolute.** 0.457 does not mean "46% related". Scores are only
comparable against other scores from the same model on the same task. Setting a fixed
threshold ("keep everything above 0.7") without calibrating on your own data is guessing.

**2. This is a bi-encoder, trained asymmetrically.** Query and document get different
prefixes because a short question and a long passage are different kinds of text even
when they mean the same thing.

That asymmetry points at the architecture question:

| | Bi-encoder | Cross-encoder |
|---|---|---|
| How | Embeds query and doc **separately** | Embeds query and doc **together**, jointly attended |
| Speed | Doc vectors precomputed once; search is a matrix multiply | Must run the model on every (query, doc) pair at query time |
| Accuracy | Lower: never sees the pair interact | Higher: can reason about how they relate |
| Use | Retrieve top ~100 from millions | Rerank those ~100 down to ~10 |

This tension is the entire reason **two-stage retrieval** exists: a fast bi-encoder for
recall, a slow cross-encoder for precision.

## Interview drill

**Q: Why normalise vectors before comparing?**
So magnitude does not influence the score. Magnitude largely reflects length and token
frequency, not meaning. After normalising, dot product *is* cosine similarity, which is a
pure measure of direction.

**Q: Your retrieval quality is poor. How do you know if the embedding model is the problem?**
Check for off-distribution use first (missing task prefixes, wrong language, domain far
from training data). Then test whether a lexical baseline like BM25 beats it on your
queries. If BM25 wins, your queries are probably identifier-heavy and dense retrieval was
the wrong tool alone.

**Q: Can you compare cosine scores from two different embedding models?**
No. Different models have different geometries. Scores are only meaningful within one
model, and even then only relatively.
