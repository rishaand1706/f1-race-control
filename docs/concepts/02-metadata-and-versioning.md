# 02. Metadata is the main lever (and the temporal versioning trap)

## Structure beats content

We made ~40 HTTP requests and produced **2,195 structured records**, each already carrying
season, event, document number, type, car number and subject.

The alternative: download 2,195 PDFs, extract text, ask an LLM to pull those fields out.
Slower by orders of magnitude, more expensive, and *less* accurate.

> **Principle:** extract metadata from **structure** wherever structure carries it.
> Content extraction is the fallback, not the default.

Filenames, listing pages, directory layouts, database columns: all free, exact metadata.
People routinely reach for an LLM to parse what a regex over a filename handles perfectly.

Implemented in `corpus/fetch_fia.py::_parse_title`.

## What we actually got

```
total documents: 2195
  infringement       440
  summons            258
  decision           170
  other             1327   (classifications, grids, scrutineering - excluded)

substantive by season:   2025 -> 495     2026 -> 373
```

## Why metadata matters more than the embedding model

Query: *"has anyone been penalised for pit lane speeding this season"*

- **Pure semantic search** over 868 docs surfaces things that *sound* like speeding
  penalties, ranked by cosine, from any season, in any order.
- **`WHERE season = 2026 AND subject ILIKE '%pit lane speeding%'`** answers it exactly,
  instantly, with perfect recall.

Filtering shrinks the candidate set *before* ranking happens.

> **Retrieval precision is usually won by filtering, not by a better embedding model.**

Teams burn weeks swapping embedding models for a few points of accuracy when one metadata
filter would have doubled it.

## The temporal versioning trap

I expected this to mean "2021 rules vs 2026 rules". It is worse. The **2026** Sporting
Regulations alone were reissued five times *within the season*:

| Issue | Effective from |
|---|---|
| 04 | 10 Dec 2025 |
| 05 | 27 Feb 2026 |
| 06 | 28 Apr 2026 |
| 07 | 25 Jun 2026 |
| 08 | 5 Aug 2026 |

An incident at the **Australian GP in March 2026** is governed by **Issue 05**.

A naive index holding only the latest rulebook answers from Issue 08 — confidently, with
a citation, and wrong.

> The citation makes a wrong answer *more* convincing, not less. Retrieval-grounded
> answers fail with more authority than ungrounded ones.

**Fix:** every regulation chunk carries `effective_from` / `effective_to`. Every query
carries an incident date. Filter to the version in force at that date, *then* rank.

## A small signal worth noticing

Subject counts show `Pit Lane Speeding` (29) and `Pit lane speeding` (10) as separate
entries. Same offence, different capitalisation.

Lexical matching cares. Semantic matching does not. That is a preview of why we need
both channels (see `03-hybrid-retrieval.md`).

## Interview drill

**Q: A RAG system returns irrelevant results. Where do you look first?**
Filtering, before models. Is there metadata not being used to narrow candidates? Most
precision problems are unfiltered-search problems, not embedding problems.

**Q: What danger is specific to retrieval-augmented answers?**
They carry citations, so they read as verified. A confidently wrong answer sourced to a
real document is harder for a user to catch than an obviously ungrounded one.

**Q: Pre-filter or post-filter?**
Pre-filter when the filter is selective and indexed: you shrink the search space and
guarantee correctness. Post-filter risks returning nothing, because your top-k may be
entirely filtered away. Post-filtering is only acceptable when the filter is weak or
cheap to widen with a larger k.
