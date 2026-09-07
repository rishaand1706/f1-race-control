# 06. Fusion, filtering, and what retrieval is actually asked for

Stage 1 built two channels and measured them failing on different queries. This stage has
to combine them, and to answer a question that turns out to matter more than the
combination rule: *what is being retrieved, and how many separate things are there?*

## The card has three slots, so there are three searches

The obvious design is one search. Take the incident, retrieve the top 8 passages, hand
them to the model. It fails, and not for a subtle ranking reason. Count the corpus:

| kind | passages |
|---|---|
| `sporting` | 2,776 |
| `decision` | 868 |
| `penalty_guidelines` | 110 |
| `driving_standards` | 10 |

A single top-k over all 3,764 is a popularity contest that the Sporting Regulations win by
volume. The 110 penalty guideline entries - the FIA's own table of *what sanction this
offence normally attracts*, which is the single most useful thing in the corpus for
predicting an outcome - are outnumbered 25 to 1. They do not need to rank badly to be
crowded out. They just need to not be in the top 8, and there are 2,776 sporting chunks
competing for those slots.

So the card's structure decides retrieval's structure. The card answers three questions:

- **What rule was broken?** → `sporting`, this season, the issue in force that day
- **What does the FIA say the penalty should be?** → `penalty_guidelines` + `driving_standards`
- **What happened to people who did this before?** → `decision`

Three questions, three filtered searches, `k` results each. `Retriever.search` runs them
and returns a dict keyed by slot. This is **faceted retrieval**, and the general lesson is
that when your answer has named parts, retrieving for the parts beats retrieving for the
whole - because a single ranked list has no way to guarantee coverage of anything.

## Pre-filter, not post-filter

Every one of those searches narrows the candidate set with metadata *before* any score is
computed (`Retriever._candidates`). The alternative - rank all 3,764, then discard what
does not match - is a **post-filter**, and it is worse in both directions:

- **Cheaper.** The precedent search scores 868 rows, not 3,764.
- **More accurate**, which is the real point. A post-filter cannot return more than `k`
  results, because it starts from `k` and removes some. Ask for 3 precedents from a global
  top-3 and you may get zero, since all three may be regulations. The recall you needed is
  gone before the filter ever runs.

Post-filtering is only correct when the filter is cheap to apply late and rarely removes
anything. Here it removes almost everything, so it has to come first.

Pre-filtering has one real cost worth naming: the metadata has to be right at index time.
A filter is only as good as the field it reads, which is why Stage 1 treated metadata as
first-class rather than decoration (`02-metadata-and-versioning.md`).

## Reciprocal rank fusion

Two channels return two ranked lists. Combining them is where the tempting mistake lives.

**The tempting mistake is adding the scores.** Cosine similarity runs 0 to 1. BM25 runs 0
to 30-something, on a scale that depends on corpus statistics and shifts when documents
are added. `0.65 + 12.4` is not a number that means anything. The usual patch is
min-max normalising each list first, which fails differently: the top BM25 score becomes
1.0 whether it was a perfect match or the least bad of a bad set, so normalisation
manufactures confidence out of nothing.

**RRF throws the scores away and keeps only the ranks.**

```
score(d) = Σ  1 / (K + rank(d, channel))
        channels
```

That is the whole algorithm. It works because rank is the one thing both channels agree on
the meaning of: "this channel's 1st choice" is comparable across systems in a way that
"0.65" and "12.4" are not. A passage ranked highly by both wins. A passage ranked 1st by
one and ignored by the other still scores well, which is exactly the behaviour we need,
since the two channels are *supposed* to disagree.

### `K`, and why the literature's default was wrong here

`K` is a damping constant. Large `K` flattens the curve, so rank 1 and rank 20 score
almost the same, and only agreement across channels moves anything. Small `K` sharpens it,
so a single channel's top hit can win alone.

The standard value is 60, from the original TREC work - fusing many systems over huge
collections, where damping any one system's confident-but-wrong top rank is the entire
point. Our situation is the opposite: two strong channels over a few hundred pre-filtered
candidates. At `K=60`, rank 1 scores only 1.4x rank 26, and *consensus mediocrity outranks
single-channel certainty*.

The query `Article 33.4` shows it. BM25 ranks the correct article **1st**; dense ranks it
**26th**, because `13.3` looks like `33.4` to an embedding model. Fused at `K=60`, the
right answer came 3rd, beaten by two passages both channels found mildly relevant.

Measured over 50 real incidents (`config.RRF_K`):

| K | 60 | 30 | 20 | 10 | 5 | 2 | 1 |
|---|---|---|---|---|---|---|---|
| MRR | 0.621 | 0.633 | 0.644 | 0.681 | 0.715 | 0.723 | 0.731 |

`K=10` takes most of the gain while still damping. Note the shape: it is monotone toward
`K=1`, which is suspicious - `K=1` is "trust whichever channel is most confident", nearly
no fusion at all. That is a sign the sample is too small and too easy to push the value
further, so the choice stops at 10 and Stage 5 settles it on held-out data.

**One more subtlety.** The fusion pools are cut at `depth = max(10k, 50)`, not at `k`. If
each channel's list were truncated to `k` before fusing, the `Article 33.4` case would be
unfixable: the passage is 26th on the dense channel and would simply not be in the pool.
Fuse deeper than you return.

## Two failures, reproduced

`python -m engine.retrieve --demo` runs both on real data.

**Failure 1 - dense retrieval cannot look up an identifier.** Query `Article 33.4`:

```
dense only:              hybrid, RRF fused:
  Art. 13.3                Art. 33.4   (d26/l1)   <- correct
  Art. 33.1                Art. 55.8   (d8/l7)
  Art. B8.2.11             Art. 26.3   (d15/l4)
```

The `d26/l1` annotation is the argument for hybrid in six characters: 26th on one channel,
1st on the other, and the fusion rule recovered it.

**Failure 2 - the temporal bug, which is invisible unless you look.** A 2025 incident,
retrieved with no season filter:

```
unfiltered:                                 filtered to the issue in force:
  2026 Sporting Regs Issue 04, Art. B1.9.6    2025 Sporting Regs Issue 5, Art. 33.3
  2025 Sporting Regs Issue 4, Art. 33.3       2025 Sporting Regs Issue 5, Art. 53.1
  2025 Sporting Regs Issue 5, Art. 33.3       2025 Sporting Regs Issue 5, Art. 56.6
```

The unfiltered top hit is a 2026 article, cited with cosine 0.729, for an incident from
2025. It is not a worse ranking - it is a rulebook that **did not exist when the incident
happened**, in a system whose entire value proposition is citing the right one. Worse, the
2025 regulations were reissued five times *inside* the season, so "filter to 2025" is not
enough either; the filter is on the issue in force on the incident's own date.

Nothing about the output looks wrong. There is no error, no low score, no warning. This is
the failure mode that makes temporal metadata worth the trouble.

## Does any of it work?

`python -m engine.retrieve --check` asks the only question that matters. Stewards'
decisions state the article they found was breached, so the ground truth is written down
rather than judged. Take the stewards' own description of the incident, give retrieval
nothing else, and see whether the article they cited comes back.

Over **120 real decisions**:

| | hybrid + filtered | dense only, unfiltered |
|---|---|---|
| hit@1 | **49.2%** | 8.3% |
| hit@3 | **58.3%** | 21.7% |
| hit@5 | **64.2%** | 25.8% |

Roughly 2.5x at every depth, and 6x at rank 1. That is the stage justified on its own
terms.

**Two caveats, both load-bearing.**

First, this is **not held out**. The sample comes from the same pool Stage 5 will evaluate
on, so it is a sanity check on plumbing, not a result. It belongs in this repo and nowhere
else. (The same caution applies to any earlier retrieval figure from this project that has
no committed harness behind it - if it cannot be regenerated by a command, it cannot be
cited.)

Second, **35% is still missing at k=5**, and the reason is a corpus gap rather than a
ranking one. Of the articles stewards cite, 562 are Sporting Regulations - which we have -
but 291 are the International Sporting Code and 241 are ISC Appendix L, which are *not in
the corpus at all*. Those are unwinnable by definition. The ceiling here is set by what
Stage 1 fetched, not by how well Stage 3 ranks, and adding the ISC is the highest-value
retrieval change available. Stage 5 should report it that way instead of blaming the
ranker.

## Interview drill

**Q: Why not one retrieval over everything?**
Because the answer has three named parts and one ranked list cannot guarantee coverage of
any of them. Concretely, 110 penalty-guideline entries compete with 2,776 regulation
chunks; they get crowded out by volume regardless of relevance. Faceted retrieval runs one
filtered search per slot the answer has.

**Q: Pre-filter or post-filter?**
Pre-filter whenever the filter is selective, which is nearly always. Post-filtering starts
from a global top-k and removes from it, so it can return fewer than k or none at all - the
recall is destroyed before the filter runs. Post-filtering is only acceptable when the
filter rarely matches nothing and is expensive to apply at index time.

**Q: Why RRF instead of normalising and adding the scores?**
Because the scores are not commensurable. Cosine is bounded 0-1; BM25 is unbounded and
corpus-dependent. Min-max normalisation makes the top hit 1.0 whether it is excellent or
merely least-bad, inventing confidence. RRF uses only rank, which both channels define
identically, and it needs no tuning beyond a single damping constant.

**Q: What does K do, and why is your K not 60?**
K damps the contribution of top ranks. 60 comes from fusing many systems over large TREC
collections, where suppressing any single system's confident errors is the goal. With two
channels over a few hundred pre-filtered candidates, 60 flattens the curve so far that
consensus mediocrity beats single-channel certainty - measured, MRR 0.621 at K=60 against
0.681 at K=10. I use 10, and I would not defend it as tuned: the sample was not held out.

**Q: Your retrieval misses 35% of cited articles. Is that a ranking problem?**
Mostly not. 532 of the cited articles are in the International Sporting Code and its
Appendix L, which are not in the corpus, so no ranker could return them. That is the
useful form of the answer - separating "ranked badly" from "was never there" is what
turns an evaluation number into a decision about what to build next.
