# 10. DuckDB as the vector store

Until this change the dense channel lived beside the database, not in it. `build_index`
wrote the passages to DuckDB and the embeddings to `vectors.npy`, an (n, 768) matrix, and
`Retriever` paired them up by position: row *i* of the matrix was taken to be the vector of
passage *i*. Nothing checked that. This stage puts each passage's vector in the passage's
own row and computes similarity in SQL.

## The failure a side file invites

Position is an implicit foreign key. Rebuild the table without regenerating the matrix, or
the other way round, or change the `ORDER BY` that loads the rows, and the dense channel
scores one passage and returns another. No exception, no empty result: a confident card
citing a rule that was never the one that matched. It never happened here, because both
files have always been written by the same run of the same script. But "the build is
always run end to end" is a convention, and a citation system should not rest its
correctness on one.

A vector store, whatever product it is, earns its name by keeping three things together:
the text you will cite, the metadata you filter on, and the vector you rank by. DuckDB
already held the first two.

## What changed

- `passages.embedding FLOAT[768]`, written by the same `INSERT` as the text and metadata.
  `vectors.npy` is gone. `embed_cache.npy` stays: it is a build cache keyed by content
  hash, so re-embedding skips unchanged text, not a second copy of the store.
- Dense scoring is a query:

  ```sql
  SELECT pid, array_inner_product(embedding, $query::FLOAT[768])
  FROM passages
  WHERE pid IN (SELECT unnest($candidates))
  ```

  The vectors are unit length (brief 01), so the inner product is the cosine. Scores come
  back keyed by `pid`, so a score can only belong to the passage it was computed from.
- `dense_only`, the unfiltered baseline kept to reproduce Stage 3's failures, is now the
  plainest possible vector query: `ORDER BY` similarity, `LIMIT k`.

## What did not change, and why

**Exact scan, no approximate index.** The argument in `corpus/build_index.py` still holds.
At 3,764 passages a filtered exhaustive scan takes milliseconds, and an approximate index
such as HNSW buys sublinear search by giving up some recall, a trade that only pays at a
scale two or three orders of magnitude above this one. It would also complicate the
filters: brief 06 is about narrowing the candidates before ranking, and an exact scan over
the survivors does exactly that.

**The filters stay in Python.** `_candidates` holds the rulebook-in-force and
published-before rules, the two filters whose absence produced Stage 3's and Stage 4's
worst bugs, and six tests in `tests/test_engine.py` pin it down. It passes the
surviving `pid`s to the query. Translating it into a `WHERE` clause is possible, but it
would replace tested code with a second copy of the same rules for no measured gain.

**BM25 stays in `rank_bm25`.** The lexical channel depends on a tokeniser that keeps
"33.4" and "b1.1.3" whole (brief 04). Moving it into the database means re-creating that
tokeniser in a second system, for a channel that already works.

## Acceptance

The move must change nothing a user can see. Measured on the 120 decisions of
`python -m engine.retrieve --check --sample 120`, before and after:

| | before (`vectors.npy`) | after (DuckDB) |
|---|---|---|
| hybrid + filtered hit@1 / hit@5 | 49.2% / 64.2% | 49.2% / 64.2% |
| dense only hit@1 / hit@5 | 8.3% / 25.8% | 8.3% / 25.8% |
| stored vectors equal to the old matrix | | bit for bit, all 3,764 |

Identical summary numbers could hide reshuffled rankings, so the full top 10 was compared
too, query by query: the rule slot with each hit's dense and BM25 rank, dense-only with and
without the `sporting` filter, and all three slots of a full `search`. All 120 queries
matched in every field. The scores are computed by different code, NumPy's matrix product
before and DuckDB's `array_inner_product` now, and on this corpus that difference never
changed an order.

`tests/test_engine.py` adds a test that inserts passages out of `pid` order with
deliberately non-contiguous ids, and checks that the dense channel returns the passage
whose vector matched. Under the side-file design that test cannot even be written, because
the pairing lives in no single place a test could check.
