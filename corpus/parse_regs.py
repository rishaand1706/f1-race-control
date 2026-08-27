"""Chunk the rulebooks and guidelines.

Three documents, three genuinely different shapes, and the shape decides the chunker:

    sporting            numbered articles, deep hierarchy -> chunk per article
    penalty_guidelines  a table: offence x session -> penalty -> chunk per offence row
    driving_standards   lettered prose sections -> chunk per section

Every chunk carries season, kind, issue and the effective_from/effective_to window of the
issue it came from. That window is what makes the temporal filter possible: an incident
in March 2026 must retrieve Issue 05, not the latest PDF sitting on disk. See
docs/concepts/02-metadata-and-versioning.md, and `fetch_regs.in_force()` for the lookup.

Usage:
    python -m corpus.parse_regs
    python -m corpus.parse_regs --compare    # fixed-size vs structural, with the damage
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, asdict

import fitz  # pymupdf

import config
from corpus.fetch_regs import load_registry, local_path, Reg

REG_CHUNKS = config.INDEX / "reg_chunks.jsonl"

FIXED_CHUNK_CHARS = 500  # the naive baseline, kept so the failure stays reproducible

# Article headings sit alone on their line, with the article's title on the line beneath:
#
#     B1.6.2
#     Pit Lane Safety
#     In all of the cases detailed in this article, an F1 Car will be deemed...
#
# Requiring the number and title on one line matched almost nothing and turned a 98-page
# rulebook into 20 chunks.
#
# The dot is required, and that is not cosmetic: the 2026 PDF prints margin line numbers
# as "B9", "B21", "B22" alone on their own lines. Without the dot they parse as articles
# and shatter the document into hundreds of empty fragments.
#   2025:  33.4      19.3      42.3      and top-level as "34)"
#   2026:  B5.3.3    B2.4.1    B5.14.4a
#
# The "34)" form matters: without it a top-level heading is not a boundary, so 184
# chunks ended with the opening line of the next article dangling off them.
HEADING_RE = re.compile(r"^(B?\d+(?:\.\d+){1,3}\s?[a-z]?|\d{1,3}\))\s*$")

# Standalone margin line numbers and page numbers.
MARGIN_RE = re.compile(r"^\s*B?\d{1,3}\s*$")

SECTION_RE = re.compile(r"^([A-Z])\.\s+(.{3,80})$")  # driving standards: "C. Chicanes"


@dataclass
class Chunk:
    reg_key: str          # e.g. "2026/sporting/Issue 08"
    season: int
    kind: str
    issue: str
    effective_from: str
    effective_to: str | None
    article: str | None   # "B5.3.3", "33.4", or the offence name for guidelines
    title: str
    text: str

    @property
    def n_chars(self) -> int:
        return len(self.text)


def _page_lines(path) -> list[list[str]]:
    with fitz.open(path) as doc:
        return [
            [ln.rstrip() for ln in page.get_text().replace("\xa0", " ").split("\n")]
            for page in doc
        ]


def strip_furniture(pages: list[list[str]]) -> str:
    """Drop the header, footer and margin numbering that repeats on every page.

    Detected rather than hardcoded: any line appearing on more than a third of pages is
    furniture ("2026 Formula 1: Sporting Regulations", the copyright line, the issue
    date). Left in, it lands in most chunks and becomes the highest-frequency text in
    the corpus - which is exactly the text BM25 will then treat as uninformative while
    it inflates every document's length.
    """
    n_pages = len(pages)
    counts = Counter(ln.strip() for page in pages for ln in page if ln.strip())
    furniture = {ln for ln, c in counts.items() if c > max(2, n_pages * 0.33)}

    out: list[str] = []
    for page in pages:
        for ln in page:
            s = ln.strip()
            if not s or s in furniture or MARGIN_RE.match(s):
                continue
            out.append(s)
    return "\n".join(out)


# --- The naive baseline -----------------------------------------------------

def chunk_fixed(text: str, size: int = FIXED_CHUNK_CHARS, overlap: int = 50) -> list[str]:
    """Fixed-width character chunking. Ignores every boundary the document gives us."""
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i : i + size])
        i += size - overlap
    return chunks


# --- The structural chunkers ------------------------------------------------

def chunk_articles(text: str) -> list[tuple[str, str, str]]:
    """Split the sporting regulations at article headings.

    Returns (article, title, body). A sub-article stays with its parent's number, so
    "B5.3.3" is retrievable on its own - which matters because that is the granularity
    the stewards cite at.
    """
    out: list[tuple[str, str, str]] = []
    current_art: str | None = None
    current_title = ""
    buf: list[str] = []
    expect_title = False

    for line in text.split("\n"):
        m = HEADING_RE.match(line)
        if m:
            if current_art is not None:
                out.append((current_art, current_title, "\n".join(buf).strip()))
            current_art = m.group(1).strip().rstrip(")")
            current_title = ""
            buf = []
            expect_title = True
        elif current_art is not None:
            if expect_title and line.strip():
                # The line after the number is the article's title - but only on articles
                # that have one. Sub-articles usually go straight into the rule text, so
                # require something short and unpunctuated before calling it a title,
                # rather than labelling the rule's opening clause as the heading.
                s = line.strip()
                if len(s) <= 80 and not s.endswith((".", ",", ";", ":")):
                    current_title = s
                expect_title = False
            buf.append(line)

    if current_art is not None:
        out.append((current_art, current_title, "\n".join(buf).strip()))

    # The contents page lists every article number followed by its title, which is
    # structurally indistinguishable from an article heading followed by its first line.
    # Left alone it produces ~50 one-line chunks that duplicate real article numbers and
    # win retrieval on title keywords while containing none of the rule.
    #
    # Keeping the longest body per article number drops them without needing to locate
    # the contents page: a table-of-contents entry is always shorter than the rule itself.
    best: dict[str, tuple[str, str, str]] = {}
    for art, title, body in out:
        if len(body) < 25:
            continue
        if art not in best or len(body) > len(best[art][2]):
            best[art] = (art, title, body)
    return list(best.values())


def chunk_penalty_table(path) -> list[tuple[str, str, str]]:
    """One chunk per offence in the penalty guidelines table.

    The table is the single most useful document we have - it states the FIA's own
    recommended sanction per offence per session type - and it is the one that plain
    text extraction destroys, because the columns interleave into unreadable prose.
    So it is read as a table and rewritten as labelled text.

    Rows wrap, and not tidily. One logical guideline spans several physical rows, and a
    continuation row's first cell is often *not* empty - it carries a "Note:" or a "-"
    sub-case. Treating only empty-first-cell rows as continuations cut every entry off
    at its first line, so "Causing a collision" cited article "App. L, Ch. IV," with the
    "2(d)" stranded on the row beneath.

    So fragments are accumulated per column and joined at the end, which reassembles both
    the wrapped cells and the sub-cases into one chunk per offence.
    """
    out: list[tuple[str, str, str]] = []
    header: list[str] = []
    current: str | None = None
    cols: dict[int, list[str]] = {}

    def flush() -> None:
        nonlocal current, cols
        if current and cols:
            # Column 0 holds the sub-cases and notes, not another offence name, so it is
            # labelled as such rather than emitting a second "Offence:" line.
            body = "\n".join(
                f"{'Notes' if i == 0 else header[i]}: {' '.join(cols[i]).strip()}"
                for i in sorted(cols)
                if i < len(header) and " ".join(cols[i]).strip()
            )
            if body:
                out.append((current, current, f"Offence: {current}\n{body}"))
        current, cols = None, {}

    # A row opens a new offence only if its first cell names one. Empty cells, notes and
    # dashed sub-cases all continue the offence above.
    # Scoped (?i:) rather than a global re.I flag: the trailing [a-z] must stay
    # case-sensitive, because "starts with a lowercase letter" is precisely the signal
    # that a cell is the tail of a wrapped sentence rather than a new offence name.
    is_continuation = re.compile(r"^\s*((?i:note\b|as above)|[-–*(]|[a-z])").search

    with fitz.open(path) as doc:
        for page in doc:
            for table in page.find_tables().tables:
                rows = table.extract()
                if not rows:
                    continue
                if rows[0] and (rows[0][0] or "").strip().lower() == "offence":
                    header = [(c or "").strip() for c in rows[0]]
                    rows = rows[1:]
                if not header:
                    continue

                for row in rows:
                    cells = [(c or "").replace("\n", " ").strip() for c in row]
                    if not any(cells):
                        continue
                    first = cells[0]
                    if first and not is_continuation(first):
                        flush()
                        current = first
                    elif current is None:
                        continue  # a continuation before any offence: nothing to attach to
                    elif first:
                        cols.setdefault(0, []).append(first)
                    for i in range(1, min(len(cells), len(header))):
                        if cells[i]:
                            cols.setdefault(i, []).append(cells[i])
    flush()
    return out


def chunk_sections(text: str) -> list[tuple[str, str, str]]:
    """Split the driving standards guidelines at its lettered sections."""
    out: list[tuple[str, str, str]] = []
    letter, title, buf = None, "", []
    for line in text.split("\n"):
        m = SECTION_RE.match(line)
        if m:
            if letter:
                out.append((letter, title, "\n".join(buf).strip()))
            letter, title = m.group(1), m.group(2).strip()
            buf = [m.group(2).strip()]
        elif letter:
            buf.append(line)
    if letter:
        out.append((letter, title, "\n".join(buf).strip()))
    return [c for c in out if c[2]] or [("full", "Driving Standards Guidelines", text)]


# --- Driver -----------------------------------------------------------------

def chunks_for(reg: Reg) -> list[Chunk]:
    path = local_path(reg)
    if not path.exists() or path.stat().st_size == 0:
        print(f"  ! missing PDF for {reg.season} {reg.kind} {reg.issue}")
        return []

    if reg.kind == "penalty_guidelines":
        parts = chunk_penalty_table(path)
    else:
        text = strip_furniture(_page_lines(path))
        parts = (chunk_sections(text) if reg.kind == "driving_standards"
                 else chunk_articles(text))

    key = f"{reg.season}/{reg.kind}/{reg.issue}"
    return [
        Chunk(reg_key=key, season=reg.season, kind=reg.kind, issue=reg.issue,
              effective_from=reg.effective_from, effective_to=reg.effective_to,
              article=art, title=title, text=body)
        for art, title, body in parts
    ]


def parse_all() -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for reg in sorted(load_registry(), key=lambda r: (r.season, r.kind, r.effective_from)):
        cs = chunks_for(reg)
        all_chunks.extend(cs)
        sizes = [c.n_chars for c in cs] or [0]
        print(f"  {reg.season} {reg.kind:20s} {reg.issue:9s} "
              f"{len(cs):5d} chunks   median {sorted(sizes)[len(sizes)//2]:5d} chars")

    with REG_CHUNKS.open("w") as f:
        for c in all_chunks:
            f.write(json.dumps(asdict(c)) + "\n")

    print(f"\n{len(all_chunks)} chunks -> {REG_CHUNKS}")
    return all_chunks


# --- The staged failure -----------------------------------------------------

def compare() -> None:
    """Show what fixed-size chunking does to a rulebook, then what structure fixes.

    This is Stage 1's deliberate failure. Build the naive version, look at the damage,
    then fix it - the bug is more memorable than the advice.
    """
    reg = next(r for r in load_registry() if r.season == 2026 and r.kind == "sporting"
               and r.issue == "Issue 08")
    text = strip_furniture(_page_lines(local_path(reg)))

    fixed = chunk_fixed(text)
    structural = chunk_articles(text)

    # How many fixed chunks straddle an article boundary? If a chunk contains a heading
    # anywhere but its first line, it has swallowed the start of a new rule while still
    # carrying the tail of the previous one.
    straddling = 0
    for c in fixed:
        lines = c.split("\n")
        if any(HEADING_RE.match(ln) for ln in lines[1:]):
            straddling += 1

    # And how many begin mid-sentence, with no heading at all to say what rule they are?
    orphan = sum(
        1 for c in fixed
        if not HEADING_RE.match(c.split("\n")[0]) and c[:1] and not c[:1].isupper()
    )

    print(f"2026 Sporting Regulations, Issue 08 - {len(text):,} characters\n")
    print(f"  fixed {FIXED_CHUNK_CHARS}-char chunks : {len(fixed):5d}")
    print(f"     straddling an article boundary: {straddling:5d}  "
          f"({100 * straddling / len(fixed):.0f}%)")
    print(f"     starting mid-sentence         : {orphan:5d}  "
          f"({100 * orphan / len(fixed):.0f}%)")
    print(f"  structural article chunks       : {len(structural):5d}")
    sizes = sorted(len(b) for _, _, b in structural)
    print(f"     size: median {sizes[len(sizes)//2]}, min {sizes[0]}, max {sizes[-1]} chars")

    # Find a fixed chunk that visibly severs a rule, and show the same rule intact.
    # Skipping the contents page: it straddles headings on every line, so it is the
    # easiest example to find and the least representative one to show.
    by_art = {a: b for a, _, b in structural}
    for c in fixed:
        lines = c.split("\n")
        hits = [ln for ln in lines[1:] if HEADING_RE.match(ln)]
        if hits and len(c.strip()) > 300 and "CONTENTS" not in c:
            art = HEADING_RE.match(hits[0]).group(1).rstrip(")")
            if len(by_art.get(art, "")) < 200:
                continue
            print("\n" + "=" * 78)
            print(f"  FIXED-SIZE CHUNK - two different rules in one chunk, "
                  f"the second ({art}) cut off:")
            print("=" * 78)
            print("   " + c.strip()[:520].replace("\n", "\n   "))
            whole = next((b for a, _, b in structural if a == art), None)
            if whole:
                print("\n" + "=" * 78)
                print(f"  STRUCTURAL CHUNK - article {art}, whole:")
                print("=" * 78)
                print("   " + whole[:520].replace("\n", "\n   "))
            break

    print("\nThe retrieval consequence: the fixed chunk can be the top hit for a query "
          "about\neither rule and answers both incorrectly - half of one, the opening "
          "line of the\nother. It is confidently wrong, which is worse than empty.")


if __name__ == "__main__":
    compare() if "--compare" in sys.argv else parse_all()
