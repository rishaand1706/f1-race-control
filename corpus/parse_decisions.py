"""Turn stewards' decision PDFs into structured rows.

Metadata that the FIA document listing already gave us - season, event, document type,
car, subject - is taken from `manifest.jsonl`, never re-derived from the PDF. See
docs/concepts/02-metadata-and-versioning.md: extract from structure where structure
carries it. This module's job is only the four things the listing cannot tell us:

    session   Practice / Qualifying / Race / Sprint - decides which regs were in force
    articles  the regulation(s) the stewards actually cited
    penalty   what they actually handed down
    reason    their written justification

`articles` and `penalty` are the ground truth for Stage 5's evaluation: we hold documents
out, feed the system only the incident description, and ask whether it cites the article
the stewards cited and predicts the penalty they gave. So a parse miss here is a silent
hole in the eval, which is why every field's success rate is reported and nothing is
dropped - a row that fails structured parsing keeps its full text and stays searchable.

Usage:
    python -m corpus.parse_decisions
    python -m corpus.parse_decisions --show 3     # print 3 parsed rows in full
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, asdict, field

import fitz  # pymupdf

import config
from corpus.fetch_fia import load_manifest, SUBSTANTIVE_TYPES

DECISIONS = config.INDEX / "decisions.jsonl"

# The FIA's PDF generator emits non-breaking spaces between almost every word. Without
# this, `^Session` matches but `^No / Driver` never does, and roughly half the corpus
# looks unparseable for a reason that has nothing to do with the document structure.
# Normalise first, always.
NBSP = " "

# Labels on the stewards' decision form. They sit at the start of a line, but the value
# may follow on the same line or on the lines beneath, depending on how the two-column
# layout flattened - so the parser accepts both.
FIELD_LABELS = [
    "No / Driver", "Competitor", "Time", "Session", "Fact", "Offence",
    "Infringement", "Decision", "Reason", "Description",
]
# The trailing lookahead stops the boilerplate sentence "Decisions of the Stewards are
# taken independently of the FIA..." from being captured as the Decision field, which it
# otherwise is on every document where that paragraph starts a line.
LABEL_RE = re.compile(
    r"^(" + "|".join(re.escape(x) for x in FIELD_LABELS) + r")(?![A-Za-z])\s*:?[ \t]*(.*)$",
    re.M,
)

# Header block, above the incident form.
DOC_NO_RE = re.compile(r"^Document\s*\n\s*(\d+)", re.M)
DOC_DATE_RE = re.compile(r"^Date\s*\n\s*(\d{1,2}\s+\w+\s+\d{4})", re.M)
DOC_TIME_RE = re.compile(r"^Time\s*\n\s*(\d{1,2}:\d{2})", re.M)
EVENT_DATES_RE = re.compile(r"^(\d{1,2})\s*[-–]\s*(\d{1,2})\s+(\w+)\s+(\d{4})\s*$", re.M)

DRIVER_RE = re.compile(r"^(\d+)\s*[-–]\s*(.+)$")

# Two regulation namespaces get cited, and they resolve to different documents:
# the International Sporting Code (driving standards live in Appendix L) and the
# season's Formula One Sporting Regulations. Keeping them apart is what lets retrieval
# jump from a decision to the right rulebook.
# The 2026 regulations renumbered every article with a Section B prefix - B1.8.6, B5.5.5
# - so a pattern anchored on a leading digit silently lost almost every 2026 citation
# (article extraction sat at 53% on infringements, against 82% for 2025 alone).
#
# Sub-letters arrive in three styles across the two seasons: "2 d)", "12.4.1e" and "2c".
# The alternation covers all three, and stays case-sensitive on the letter: with re.I,
# "Article 15 The Stewards..." captures "T" as a sub-article.
ARTICLE_RE = re.compile(r"[Aa]rticle\s+(B?\d+(?:\.\d+)*)([a-z]\)?|\s+[a-z]\))?")
APPENDIX_L_RE = re.compile(
    r"Appendix\s+L,?\s+Chapter\s+([IVX]+),?\s+Article\s+(\d+(?:\.\d+)*)\s*([a-z]\)?)?", re.I
)

SESSION_MAP = [
    (re.compile(r"\bsprint\s+quali", re.I), "Sprint Qualifying"),
    (re.compile(r"\bsprint\b", re.I), "Sprint"),
    (re.compile(r"\bqualifying\b|\bQ[123]\b", re.I), "Qualifying"),
    (re.compile(r"\brace\b", re.I), "Race"),
    (re.compile(r"\bpractice\b|\bFP[123]\b", re.I), "Practice"),
]

# Penalty taxonomy. Ordered: the first pattern that matches wins, so the specific
# outcomes are checked before the generic ones.
PENALTY_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("no_further_action", re.compile(r"no\s+further\s+action|no\s+penalty\s+(?:is\s+)?appli", re.I)),
    ("disqualified", re.compile(r"\bdisqualif", re.I)),
    ("time_penalty", re.compile(r"(\d+)\s*second\s+time\s+penalt", re.I)),
    ("stop_go", re.compile(r"stop\s*(?:and|/|&)?\s*go", re.I)),
    ("drive_through", re.compile(r"drive[\s-]*through", re.I)),
    # The FIA writes grid drops both ways: "5 grid place penalty" and "Drop of 3 grid
    # positions for the next Race". Matching only the first form lost every instance of
    # the second, which is the phrasing used on power-unit and gearbox penalties.
    ("grid_penalty", re.compile(
        r"(\d+)\s*(?:grid\s+)?(?:position|place|grid)\s+(?:grid\s+)?penalt"
        r"|drop\s+of\s+(\d+)\s*(?:grid\s+)?(?:position|place)", re.I)),
    ("pit_lane_start", re.compile(r"start\s+(?:the\s+race\s+)?from\s+the\s+pit\s+lane", re.I)),
    # Both voices appear: "lap times are deleted" and "Deletion of the lap times shown".
    ("lap_times_deleted", re.compile(
        r"lap\s+times?\s+(?:are\s+|be\s+)?delet|delet\w*\s+of\s+the\s+lap\s+times?", re.I)),
    ("fine", re.compile(r"fine[ds]?\s+(?:of\s+)?€?\s?([\d,]+)", re.I)),
    ("reprimand", re.compile(r"\breprimand", re.I)),
    ("warning", re.compile(r"\bwarning\b", re.I)),
]
SECONDS_RE = re.compile(r"(\d+)\s*second", re.I)
GRID_PLACES_RE = re.compile(r"(\d+)\s*(?:grid\s+)?(?:position|place)", re.I)
POINTS_RE = re.compile(r"(\d+)\s+penalty\s+point", re.I)
FINE_RE = re.compile(r"€\s?([\d,]+)", re.I)


@dataclass
class Decision:
    # --- from the manifest (free, exact) ---
    season: int
    event: str
    doc_type: str
    doc_number: int | None
    car: int | None
    team: str | None
    subject: str
    filename: str
    url: str

    # --- from the PDF body ---
    driver: str | None = None
    competitor: str | None = None
    session: str | None = None
    doc_date: str | None = None
    doc_time: str | None = None
    incident_time: str | None = None
    fact: str | None = None
    infringement: str | None = None
    decision: str | None = None
    reason: str | None = None
    articles: list[dict] = field(default_factory=list)
    penalty_type: str | None = None
    penalty_seconds: int | None = None
    penalty_grid_places: int | None = None
    penalty_points: int | None = None
    fine_eur: int | None = None
    text: str = ""

    @property
    def article_ids(self) -> list[str]:
        return [a["id"] for a in self.articles]


def normalise(text: str) -> str:
    """Undo the FIA PDF generator's non-breaking spaces and tidy line ends."""
    text = text.replace(NBSP, " ").replace("‑", "-")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n[ \t]+", "\n", text)


def read_pdf(path) -> str:
    with fitz.open(path) as doc:
        return normalise("\n".join(page.get_text() for page in doc))


def split_fields(text: str) -> dict[str, str]:
    """Split the labelled stewards' form into {label: value}.

    A value runs from its label to the start of the next label, which handles both
    layouts the extractor produces (value on the label's own line, or beneath it).
    """
    matches = list(LABEL_RE.finditer(text))
    out: dict[str, str] = {}
    for i, m in enumerate(matches):
        label = m.group(1)
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        value = (m.group(2) + "\n" + text[m.end() : end]).strip()
        # "Time" appears twice: once in the header (document published time) and once
        # for the incident. The header block is parsed separately, so keep the first
        # occurrence of every other label but let the *incident* Time win.
        if label not in out or label == "Time":
            out[label] = value
    return out


def strip_boilerplate(reason: str) -> str:
    """Drop the right-of-appeal paragraph and steward signatures.

    It is identical on every document, so leaving it in means several hundred chunks
    that are ~40% the same text. That inflates every BM25 document length and gives the
    dense index hundreds of near-duplicate neighbours competing with real content.
    """
    cut = re.split(
        r"Competitors are reminded that they have the right to appeal", reason, maxsplit=1
    )[0]
    return cut.strip()


def extract_articles(text: str) -> list[dict]:
    """Every regulation reference, tagged with which rulebook it points at."""
    found: list[dict] = []
    seen: set[str] = set()

    for m in APPENDIX_L_RE.finditer(text):
        chapter, art, sub = m.group(1), m.group(2), (m.group(3) or "").strip()
        aid = f"ISC App.L Ch.{chapter} Art.{art}{' ' + sub if sub else ''}"
        if aid not in seen:
            seen.add(aid)
            found.append({"source": "isc_appendix_l", "id": aid, "article": art, "sub": sub or None})

    matches = list(ARTICLE_RE.finditer(text))
    for i, m in enumerate(matches):
        art, sub = m.group(1), (m.group(2) or "").strip()
        # The citation style names the rulebook *after* the number ("Article 33.3 of the
        # FIA Formula One Sporting Regulations"), so look ahead - but bound the lookahead
        # at the next "Article" and let the nearest rulebook win.
        #
        # A fixed 140-character window run through a priority chain got this wrong on
        # every compound citation. "Article 33.4 of the FIA Formula One Sporting
        # Regulations and Article 12.2.1 i) of the FIA International Sporting Code" put
        # both rulebooks inside 33.4's window, and the chain checked the Code first, so
        # 33.4 was filed as an ISC article. Proximity decides, not priority.
        stop = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        tail = text[m.end() : min(stop, m.end() + 200)].lower()
        head = text[max(0, m.start() - 90) : m.start()].lower()
        if "appendix l" in head + " " + tail and "sporting regulations" not in tail[:60]:
            continue  # already captured, with its chapter, above

        # A "B" prefix is definitional: Section B *is* the Sporting Regulations, so the
        # number identifies its own rulebook without the citation naming one.
        if art.startswith("B"):
            aid = f"sporting:{art}{' ' + sub if sub else ''}"
            if aid not in seen:
                seen.add(aid)
                found.append({"source": "sporting", "id": aid, "article": art,
                              "sub": sub or None})
            continue

        source, nearest = "unknown", len(tail) + 1
        for phrase, name in (
            ("technical regulations", "technical"),
            ("financial regulations", "financial"),
            ("international sporting code", "isc"),
            ("sporting code", "isc"),
            ("sporting regulations", "sporting"),
        ):
            pos = tail.find(phrase)
            if pos != -1 and pos < nearest:
                source, nearest = name, pos
        aid = f"{source}:{art}{' ' + sub if sub else ''}"
        if aid not in seen:
            seen.add(aid)
            found.append({"source": source, "id": aid, "article": art, "sub": sub or None})

    return found


def classify_penalty(decision_text: str) -> dict:
    """Map the Decision field onto a small taxonomy plus its magnitude."""
    out: dict = {
        "penalty_type": None, "penalty_seconds": None,
        "penalty_grid_places": None, "penalty_points": None, "fine_eur": None,
    }
    if not decision_text:
        return out

    for name, pattern in PENALTY_PATTERNS:
        if pattern.search(decision_text):
            out["penalty_type"] = name
            break

    if out["penalty_type"] == "time_penalty":
        m = SECONDS_RE.search(decision_text)
        if m:
            out["penalty_seconds"] = int(m.group(1))
    if out["penalty_type"] == "grid_penalty":
        m = GRID_PLACES_RE.search(decision_text)
        if m:
            out["penalty_grid_places"] = int(m.group(1))

    # Penalty points are additive to any other sanction, so they are read independently
    # of the type rather than as one of its branches.
    m = POINTS_RE.search(decision_text)
    if m:
        out["penalty_points"] = int(m.group(1))
    m = FINE_RE.search(decision_text)
    if m:
        out["fine_eur"] = int(m.group(1).replace(",", ""))

    return out


def infer_session(fields: dict[str, str], subject: str, text: str) -> str | None:
    explicit = fields.get("Session", "").strip()
    if explicit:
        first = explicit.split("\n")[0].strip()
        if first:
            return first
    # Summons documents carry no Session field, so fall back to the language of the
    # allegation itself ("during Q1", "in the Race").
    for pattern, name in SESSION_MAP:
        if pattern.search(subject) or pattern.search(text[:1500]):
            return name
    return None


def parse_one(doc, path) -> Decision:
    text = read_pdf(path)
    fields = split_fields(text)

    d = Decision(
        season=doc.season, event=doc.event, doc_type=doc.doc_type,
        doc_number=doc.doc_number, car=doc.car, team=doc.team,
        subject=doc.subject, filename=doc.filename, url=doc.url,
        text=text,
    )

    if m := DOC_DATE_RE.search(text):
        d.doc_date = m.group(1)
    if m := DOC_TIME_RE.search(text):
        d.doc_time = m.group(1)

    if driver_field := fields.get("No / Driver", ""):
        first = driver_field.split("\n")[0].strip()
        if m := DRIVER_RE.match(first):
            d.driver = m.group(2).strip()
            if d.car is None:
                d.car = int(m.group(1))
        elif first:
            d.driver = first

    if competitor := fields.get("Competitor", ""):
        d.competitor = competitor.split("\n")[0].strip() or None
    if incident_time := fields.get("Time", ""):
        if m := re.match(r"(\d{1,2}:\d{2})", incident_time.strip()):
            d.incident_time = m.group(1)

    d.session = infer_session(fields, doc.subject, text)
    d.fact = (fields.get("Fact") or fields.get("Description") or "").strip() or None
    d.infringement = (fields.get("Infringement") or fields.get("Offence") or "").strip() or None
    d.decision = (fields.get("Decision") or "").strip() or None
    if reason := fields.get("Reason", "").strip():
        d.reason = strip_boilerplate(reason) or None

    # Articles are cited in the Infringement line on decisions, but in the Reason on
    # summons documents, so search the union rather than one designated field.
    citation_text = " ".join(x for x in (d.infringement, d.fact, d.reason) if x)
    d.articles = extract_articles(citation_text) if citation_text else []
    if not d.articles:
        # The tabular documents - "Race Deleted Lap Times" and friends - put the incident
        # in a table of cars and turns and cite the article elsewhere in the body, so the
        # labelled fields come back empty of citations while the page is full of them.
        # Boilerplate is stripped first, or every one of these would "cite" Article 15,
        # which is only the right-of-appeal paragraph printed on every document.
        d.articles = extract_articles(strip_boilerplate(text))

    # A summons alleges rather than sanctions, so a missing Decision is correct there,
    # not a parse failure.
    for k, v in classify_penalty(d.decision or "").items():
        setattr(d, k, v)

    return d


def parse_all() -> list[Decision]:
    docs = [d for d in load_manifest() if d.doc_type in SUBSTANTIVE_TYPES]
    rows: list[Decision] = []
    missing = failed = 0

    for doc in docs:
        path = config.PDFS / f"{doc.season}_{doc.filename}"
        if not path.exists() or path.stat().st_size == 0:
            missing += 1
            continue
        try:
            rows.append(parse_one(doc, path))
        except Exception as e:  # noqa: BLE001 - one unreadable PDF must not stop the run
            failed += 1
            print(f"  ! {doc.filename}: {type(e).__name__}: {e}")

    with DECISIONS.open("w") as f:
        for r in rows:
            f.write(json.dumps(asdict(r)) + "\n")

    report(rows, missing, failed)
    return rows


def report(rows: list[Decision], missing: int, failed: int) -> None:
    from collections import Counter

    n = len(rows)
    print(f"\nparsed {n} documents  (PDF absent: {missing}, unreadable: {failed})")
    print(f"-> {DECISIONS}\n")
    if not n:
        return

    # Broken out by document type on purpose. A summons alleges and calls the team in;
    # it has no Fact, Infringement or Decision because no decision has been taken yet.
    # Averaging those absences into one rate makes a correct parse look 65% broken.
    fields = ["driver", "competitor", "session", "fact", "infringement",
              "decision", "reason", "articles"]
    types = sorted({r.doc_type for r in rows})
    print(f"field extraction rate by document type:\n")
    print("   " + " " * 14 + "".join(f"{t + ' (' + str(sum(r.doc_type == t for r in rows)) + ')':>18s}"
                                     for t in types))
    for name in fields:
        cells = ""
        for t in types:
            group = [r for r in rows if r.doc_type == t]
            ok = sum(bool(getattr(r, name)) for r in group)
            cells += f"{ok:6d}/{len(group):<4d}{100 * ok / len(group):5.1f}%"
        print(f"   {name:14s}{cells}")

    sanctioned = [r for r in rows if r.doc_type in {"infringement", "decision"}]
    if sanctioned:
        got = sum(bool(r.penalty_type) for r in sanctioned)
        held = [r for r in sanctioned if r.decision]
        print(f"\n   penalty classified: {got}/{len(sanctioned)} of sanctioning docs "
              f"({100 * got / len(sanctioned):.1f}%), "
              f"{got}/{len(held)} of those with a Decision field "
              f"({100 * got / max(len(held), 1):.1f}%)")

    print("\npenalty types:")
    for t, c in Counter(r.penalty_type for r in rows if r.penalty_type).most_common():
        print(f"   {t:20s} {c:4d}")

    print("\nmost-cited articles:")
    for a, c in Counter(a for r in rows for a in r.article_ids).most_common(12):
        print(f"   {c:4d}  {a}")


def show(k: int = 2) -> None:
    rows = [json.loads(l) for l in DECISIONS.open()]
    for r in rows[:k]:
        print("=" * 88)
        for key in ("season", "event", "doc_type", "car", "driver", "competitor",
                    "session", "subject", "infringement", "decision",
                    "penalty_type", "penalty_seconds", "penalty_points", "articles"):
            print(f"  {key:16s} {r[key]}")
        print(f"  {'reason':16s} {(r['reason'] or '')[:200]}...")


if __name__ == "__main__":
    if "--show" in sys.argv:
        i = sys.argv.index("--show")
        show(int(sys.argv[i + 1]) if len(sys.argv) > i + 1 else 2)
    else:
        parse_all()
