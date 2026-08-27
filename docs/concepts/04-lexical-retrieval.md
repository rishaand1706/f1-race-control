# 04. BM25, and the query where dense retrieval falls apart

## The failure, first

Both channels are built over the same 3,764 passages. Ask for an article by number:

```
python -m corpus.query_cli "Article 33.3" -k 3
```

**Dense (nomic-embed-text, cosine):**

```
1. [0.644] sporting   2025 Sporting Regulations Issue 3, Art. 13.3
              Third party insurance arranged by the promoter shall be...
2. [0.640] sporting   2025 Sporting Regulations Issue 5, Art. 13.3
3. [0.636] sporting   2025 Sporting Regulations Issue 4, Art. 13.3
```

Article 33.3 is track limits. Article 13.3 is **third-party insurance**. The embedding
model returned the wrong rule three times, with high confidence, because `13.3` *looks
like* `33.3`.

**BM25 on the same query** returns track-limits decisions instead. Overlap between the
two channels: **0 of 3**.

That is the whole argument for hybrid retrieval in one query.

## Why dense retrieval cannot do this

An embedding places text by **resemblance**. That is exactly what you want for "car
forced another driver off the track", where the corpus says "forcing another driver off
of the track" and no useful word is shared beyond "track".

But an identifier is not a resemblance, it is an **identity**. `33.3` and `13.3` differ
in one character and are unrelated rules; `33.3` and `Article 33.3` are the same thing.
There is no continuous space in which those relationships hold, because identifiers are
arbitrary labels. Embedding them puts near-identical strings near each other, which is
precisely the wrong answer.

The corpus is full of them: article numbers, car numbers, document numbers, `12.2.1 i)`,
`B1.8.4b`. They are the highest-signal terms we have and the ones the dense channel is
worst at.

## What BM25 actually computes

BM25 is the mature descendant of TF-IDF. TF-IDF says a term matters in proportion to how
often it appears in this document (**term frequency**) and how rare it is across the
corpus (**inverse document frequency**). BM25 keeps both ideas and fixes two flaws.

For query `q` against document `D`:

```
score(D, q) = Σ   IDF(t) ·        f(t,D) · (k1 + 1)
             t∈q             ─────────────────────────────────
                             f(t,D) + k1 · (1 − b + b · |D|/avgdl)
```

```
IDF(t) = ln( (N − n(t) + 0.5) / (n(t) + 0.5) + 1 )
```

Two parameters, and each fixes a specific TF-IDF flaw:

**`k1` controls term saturation** (default 1.5 in `rank_bm25`). Under plain TF-IDF, a
document containing "collision" twenty times scores twenty times higher than one
containing it once. That is wrong: the second occurrence tells you a lot, the twentieth
tells you almost nothing. The `f/(f + k1·…)` shape is a saturating curve - the score
approaches a ceiling as frequency grows. Low `k1` saturates almost immediately (presence
is nearly all that matters); high `k1` behaves more like raw TF.

**`b` controls length normalisation** (default 0.75). Long documents contain more of
every term by accident, so they would win everything. `b` scales the penalty by
`|D|/avgdl`, this document's length over the corpus average. At `b=0` length is ignored
entirely; at `b=1` it is fully normalised. 0.75 is the usual compromise, and it matters
here because our passages run from 25 to 7,956 characters.

**`IDF`** is what makes "the" free and "unsafe release" expensive. Terms appearing in
almost every document contribute almost nothing - the boilerplate we stripped in
`parse_decisions.py` would otherwise have been the highest-frequency text in the corpus.

Our index: **3,764 documents, 137 tokens on average**.

## The tokeniser is load-bearing

BM25 matches tokens exactly, so the tokeniser decides what can be matched at all. The
obvious `\w+` split turns `33.4` into `33` and `4`, and `B1.1.3` into `b1`, `1`, `3` -
destroying exactly the terms the lexical channel exists to catch.

```python
TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)*")
```

Keeps `33.4`, `b1.1.3` and `12.2.1` whole. `corpus/build_index.py::tokenize`.

**The general point:** a lexical index is only as good as its analyser. Every decision in
there - casing, stemming, stopwords, how numbers split - silently sets the ceiling on
what can ever be retrieved.

## Where each channel wins

| Query | Dense | BM25 |
|---|---|---|
| "car forced another driver off the track" | wins - matches "forcing another driver off of the track" with no shared phrasing | also works: enough shared vocabulary |
| "Article 33.3" | **fails** - returns Article 13.3 | wins - exact token |
| "B1.8.4b" | fails - no notion of identity | wins |
| "when can the stewards delete a lap time" | wins - the corpus never phrases it that way | weak - few shared terms |

On the semantic query the two channels agreed on 2 of 3 hits. On the identifier query they
agreed on 0 of 3. **The disagreement is the signal**: two channels failing in uncorrelated
ways is exactly the condition under which fusing them beats either alone. If they always
agreed, the second channel would be pure cost.

That fusion is Stage 3, and the reason to wait is that reciprocal rank fusion is much
easier to justify once you have seen both failure modes in your own corpus.

## Lessons

**1. Dense retrieval is not a superset of lexical retrieval.** It is a different failure
profile. Anyone who tells you embeddings made BM25 obsolete has not tried querying an
identifier.

**2. Hybrid is the default for technical corpora**, not an optimisation. Anything with
part numbers, article numbers, error codes, SKUs or names needs a lexical channel.

**3. Check the analyser before blaming the model.** A tokeniser that shreds your
identifiers looks exactly like "BM25 doesn't work on our data".

## Interview drill

**Q: What do `k1` and `b` do in BM25?**
`k1` sets term-frequency saturation: how quickly repeated occurrences stop adding score.
`b` sets length normalisation strength, penalising long documents that contain more of
every term by accident. Defaults 1.5 and 0.75. Tune `b` down if your documents are
uniform in length, up if they vary wildly.

**Q: Why is BM25 better than TF-IDF?**
It saturates term frequency instead of scaling linearly with it, and it normalises for
document length in a tunable way. Both fix cases where TF-IDF's linearity gives obviously
wrong rankings - keyword-stuffed and long documents respectively.

**Q: You have dense retrieval working. Why add BM25?**
Because they fail differently. Dense handles paraphrase and fails on identifiers; lexical
handles identifiers and fails on paraphrase. Our corpus is dense with article numbers, and
a dense-only index returned Article 13.3 for a query naming Article 33.3. The measurable
justification is channel overlap: near-zero overlap on a query class means one channel is
contributing something the other cannot.

**Q: When would you *not* add a lexical channel?**
When queries and documents share no vocabulary by design - cross-lingual retrieval, or
matching a natural-language question against code. There BM25 contributes noise, and the
effort belongs in the embedding model or in query rewriting instead.
