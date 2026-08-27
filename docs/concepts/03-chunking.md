# 03. Chunking, and why the document's own structure beats any window size

## The idea

Retrieval does not return documents, it returns **passages**. So before anything is
embedded, every document has to be cut into pieces. That cut decides the ceiling on
retrieval quality, because a passage is the smallest thing the system can ever hand the
model. If the right rule is split across two chunks, no reranker, no better embedding
model and no larger `k` will put it back together.

The default move is a fixed window: 500 characters, 50 of overlap, slide. It is one line
of code and it is wrong here.

## The mechanism, and the tradeoff

Every chunking scheme trades **precision against context**:

- **Small chunks** embed cleanly. One vector represents one idea, so cosine similarity
  is meaningful. But the passage may not contain enough to answer with.
- **Large chunks** carry context, but their embedding is an average of everything inside.
  A 4,000-character chunk covering six topics has a vector near none of them. This is
  sometimes called semantic dilution, and it is why "just use bigger chunks" fails.

Overlap is the patch for boundaries: repeat the last ~50 characters at the start of the
next chunk so a sentence severed at the seam survives somewhere. It costs index size and
duplicate hits, and it only ever reduces the damage rather than removing it.

Structural chunking sidesteps the trade entirely **when the document has structure**. A
regulation is already written as numbered, self-contained rules. The author has done the
segmentation, and it is better than any window we would pick, because it matches the
granularity people actually cite at.

## What we measured

The 2026 Sporting Regulations, Issue 08, is 258,615 characters. Run `python -m
corpus.parse_regs --compare` to reproduce:

| | fixed 500 chars | structural (per article) |
|---|---|---|
| chunks | 575 | 365 |
| straddling an article boundary | **275 (48%)** | 0 by construction |
| starting mid-sentence | **541 (94%)** | 0 by construction |
| size | 500 by definition | median 308, min 25, max 7956 |

Nearly half of all fixed chunks contain the end of one rule and the beginning of another.
Here is a real one, ending mid-word:

```
...d using the Document Management System. Any decision or communication
concerning a particular Competitor should be given to them within twenty-five (25)
minutes of such decision, and receipt must be acknowledged.
B1.1.3
Unless written permission has been given by the FIA to do otherwise, the circuit may
only be used for purposes other than the Competition after the last free practice
session on each day of practice, after Sprint Qualifying, after Qualifying, and on the
day of the Race no less tha
```

**Why this is worse than returning nothing.** That chunk is a plausible top hit for a
query about either rule. Retrieved for a question about B1.1.2 it trails off into an
unrelated rule; retrieved for B1.1.3 it answers with a truncated fragment that omits the
operative clause ("no less than one hour before the Pit Lane exit is opened"). The model
then writes fluent prose from it. The failure is invisible, cited, and wrong.

## The part nobody warns you about: structure is a lie until you look

Structural chunking is only better if you parse the structure correctly. Four separate
traps in one PDF, each found by reading the output rather than by reasoning about it:

**1. Headings sit alone on their line.** The number is on one line, the title beneath it,
the rule beneath that. A regex expecting `B1.6.2 Pit Lane Safety` on one line matched
almost nothing and turned a 98-page rulebook into **20 chunks**. The tell was the number
being implausible, not an error.

**2. Margin line numbers look exactly like article numbers.** The 2026 PDF prints `B9`,
`B21`, `B22` down the margin. Requiring at least one dot separates `B21` (noise) from
`B2.1` (article). Without it the document shatters into hundreds of empty fragments.

**3. Top-level headings use a different form.** Sub-articles are `33.4`; top-level ones
are `34)`. Missing that form meant **184 chunks** ended with the next article's opening
line dangling off them.

**4. The contents page is structurally identical to the document.** It lists every
article number followed by its title, which is exactly what a heading followed by its
title looks like. It produced ~50 one-line chunks carrying real article numbers and real
titles with none of the rule text - and they compete for the same queries. Fixed by
keeping the longest body per article number, since a contents entry is always shorter
than the rule.

None of these are visible from the extracted text at a glance. All four are obvious the
moment you print chunk counts and sizes, which is the actual lesson: **inspect the
distribution, not a sample.**

## When the structure is a table

The Penalty Guidelines are the most valuable document we have - the FIA's own recommended
sanction per offence per session - and plain text extraction destroys them, because the
columns interleave into unreadable prose:

```
Causing a collision Note: Penalty points... App. L, Ch. IV, 2(d) Reprimand or
Grid Place penalty 0 to 3* Reprimand or Grid Place penalty 0 to 3* 10s Penalty...
```

Read as a table instead (`page.find_tables()`), one chunk per offence, relabelled:

```
Offence: Causing a collision
SR Article: App. L, Ch. IV, 2(d)
Free Practice: Reprimand or Grid Place penalty
Qualifying: Reprimand or Grid Place penalty
TTCS: 10s Penalty (Baseline), Drive-Through or 10s stop and go...
```

The wrinkle: one logical guideline spans several physical rows, and a continuation row's
first cell is usually **not** empty - it holds a `Note:` or a dashed sub-case. Treating
only empty-first-cell rows as continuations cut every entry off at its first line, so
"Causing a collision" cited article "App. L, Ch. IV," with the "2(d)" stranded on the row
beneath. Fragments now accumulate per column and are joined at the end.

`corpus/parse_regs.py::chunk_articles`, `::chunk_penalty_table`, `::chunk_sections`.

## Lessons

**1. Chunk on the document's own boundaries when it has them.** Regulations, code, and
markdown all carry authored structure. Fixed windows are for genuinely unstructured prose.

**2. Match the chunk to the unit people cite.** Stewards cite `B1.8.4b`. If that is the
question's unit, it should be the retrieval unit. Anything else forces the system to
reassemble what the author already separated.

**3. Verify the chunker on counts and size distribution, never on one example.** Every
bug above produced output that looked fine in a spot check.

## Interview drill

**Q: How do you choose a chunk size?**
Ideally you do not. If the document has structure, chunk on it. If it genuinely does not,
choose by the unit of meaning in your queries and validate with a retrieval metric on a
golden set, not by intuition. Fixed size is a fallback, not a default.

**Q: What does chunk overlap actually buy, and what does it cost?**
It buys boundary insurance: a sentence cut at a seam appears whole in the neighbour. It
costs index size, and it puts near-duplicate passages into your results, which eats slots
in top-k and skews any diversity you were relying on. It reduces the boundary problem; it
does not solve it. Structural chunking removes it.

**Q: Your chunks are large and retrieval is returning irrelevant ones. Why?**
Probably semantic dilution: a large chunk's vector is an average over everything in it,
so it sits near nothing in particular and matches weakly but broadly. Split to one idea
per chunk, then use a parent-document or context-expansion step if the model needs the
surrounding text to answer.

**Q: How do you chunk a table?**
Not as text. Extract it as a table and emit one chunk per row with the column headers
rewritten as labels, so each chunk is self-describing. A row stripped of its header is
uninterpretable, and a flattened table is worse than useless: it reads as fluent prose
that means nothing.
