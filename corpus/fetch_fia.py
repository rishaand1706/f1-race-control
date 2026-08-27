"""Fetch the FIA F1 document index, then the decision PDFs themselves.

Two phases, deliberately separated:

  1. build_manifest()  - HTML only. Cheap, fast, re-runnable. Produces one record per
                         published document with metadata parsed from the *listing*,
                         not from the PDF.
  2. download()        - fetches the PDFs we actually want, exactly once each.

Why the split matters: the listing already gives us season, event, document number,
document type, car number and subject. That is nearly all our retrieval metadata, for
the cost of ~36 HTML requests instead of ~2000 PDF parses. Extracting metadata from
structure is always cheaper and more reliable than extracting it from content, so do it
there whenever the structure carries it.

Usage:
    python -m corpus.fetch_fia manifest
    python -m corpus.fetch_fia download
    python -m corpus.fetch_fia stats
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, asdict

import httpx
from bs4 import BeautifulSoup

import config

MANIFEST = config.INDEX / "manifest.jsonl"

# Season landing pages carry an opaque numeric id we cannot derive, so they are read
# from the championship page's season selector at run time.
SEASON_LINK = re.compile(r"/season/season-(\d{4})-(\d+)")

# Document types worth retrieving over. Everything else on an FIA event page is
# administrative (classifications, grids, scrutineering, points) and would only dilute
# the index. See docs/concepts/02-corpus-and-metadata.md.
SUBSTANTIVE_TYPES = {"infringement", "decision", "summons", "offence"}

TITLE_RE = re.compile(r"^\s*Doc\s+(\d+)\s*[-–]\s*(.+?)\s*$", re.I)
CAR_RE = re.compile(r"^car\s+(\d+)$", re.I)
CARS_RE = re.compile(r"\bcar\s+(\d+)\b", re.I)


@dataclass
class Doc:
    season: int
    event: str
    doc_number: int | None
    title: str
    doc_type: str          # "infringement" | "decision" | "summons" | ... | "other"
    car: int | None        # car the document is addressed to
    team: str | None       # some documents are addressed to a team instead
    subject: str           # e.g. "Alleged failure to slow for yellow flags"
    cars_mentioned: list[int]
    url: str
    filename: str
    published: str | None  # as printed: "26.07.26 19:25" CET


def _client() -> httpx.Client:
    # fia.com routinely takes 20-30s to render a listing page, so the timeout is
    # generous by design. Measured, not guessed: curl and httpx both sit at ~20-30s.
    return httpx.Client(
        headers={"User-Agent": config.USER_AGENT},
        timeout=httpx.Timeout(connect=15.0, read=120.0, write=30.0, pool=15.0),
        follow_redirects=True,
    )


def _get(client: httpx.Client, url: str, attempts: int = 3) -> httpx.Response:
    """GET with exponential backoff. A slow public site will time out occasionally;
    that is a transient condition, not a reason to lose 30 minutes of scraping."""
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            r = client.get(url)
            r.raise_for_status()
            return r
        except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as e:
            last = e
            if attempt < attempts - 1:
                wait = 2 ** attempt * 3
                print(f"      retry {attempt + 1}/{attempts - 1} in {wait}s ({type(e).__name__})")
                time.sleep(wait)
    raise last  # type: ignore[misc]


def _parse_title(title: str, season: int, event: str, href: str, published: str | None) -> Doc:
    """Turn 'Doc 60 - Infringement - Car 11 - Starting procedure infringement' into fields."""
    doc_number = None
    body = title.strip()

    m = TITLE_RE.match(body)
    if m:
        doc_number = int(m.group(1))
        body = m.group(2)

    parts = [p.strip() for p in re.split(r"\s+[-–]\s+", body) if p.strip()]
    doc_type = parts[0].lower() if parts else "other"
    rest = parts[1:]

    car: int | None = None
    team: str | None = None
    if rest:
        cm = CAR_RE.match(rest[0])
        if cm:
            car = int(cm.group(1))
            rest = rest[1:]
        elif doc_type in SUBSTANTIVE_TYPES and len(rest) > 1:
            # e.g. "Infringement - Ferrari - Unsafe release of car 44"
            team = rest[0]
            rest = rest[1:]

    subject = " - ".join(rest).strip()

    # Normalise the long administrative titles into a single bucket.
    if doc_type not in SUBSTANTIVE_TYPES:
        doc_type = "other"

    cars_mentioned = sorted({int(c) for c in CARS_RE.findall(title)})
    if car is not None and car not in cars_mentioned:
        cars_mentioned.append(car)

    return Doc(
        season=season,
        event=event,
        doc_number=doc_number,
        title=title.strip(),
        doc_type=doc_type,
        car=car,
        team=team,
        subject=subject,
        cars_mentioned=sorted(cars_mentioned),
        url=config.FIA_BASE + href,
        filename=href.rsplit("/", 1)[-1],
        published=published,
    )


def _season_urls(client: httpx.Client) -> dict[int, str]:
    """Map season year -> its landing page URL, read from the championship page."""
    r = _get(client, config.FIA_F1_DOCS)
    found: dict[int, str] = {}
    for m in SEASON_LINK.finditer(r.text):
        found[int(m.group(1))] = f"{config.FIA_F1_DOCS}/season/season-{m.group(1)}-{m.group(2)}"
    return found


def _events(client: httpx.Client, season_url: str) -> list[str]:
    """Event names available within one season."""
    r = _get(client, season_url)
    names = re.findall(r'<option value="[^"]*/event/([^"]+)"', r.text)
    from urllib.parse import unquote
    seen, out = set(), []
    for n in names:
        name = unquote(n).strip()
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _event_docs(client: httpx.Client, season: int, season_url: str, event: str) -> list[Doc]:
    from urllib.parse import quote
    r = _get(client, f"{season_url}/event/{quote(event)}")
    soup = BeautifulSoup(r.text, "html.parser")

    docs: list[Doc] = []
    for li in soup.select("li.document-row"):
        a = li.find("a", href=True)
        if not a or not a["href"].lower().endswith(".pdf"):
            continue
        title_el = a.find("div", class_="title")
        if not title_el:
            continue
        pub_el = a.find("span", class_="date-display-single")
        docs.append(
            _parse_title(
                title=title_el.get_text(" ", strip=True),
                season=season,
                event=event,
                href=a["href"],
                published=pub_el.get_text(strip=True) if pub_el else None,
            )
        )
    return docs


def build_manifest(seasons: list[int] | None = None) -> list[Doc]:
    seasons = seasons or config.SEASONS
    all_docs: list[Doc] = []

    with _client() as client:
        available = _season_urls(client)
        for season in seasons:
            if season not in available:
                print(f"  ! season {season} not published by the FIA, skipping")
                continue
            season_url = available[season]
            events = _events(client, season_url)
            print(f"{season}: {len(events)} events")
            for event in events:
                time.sleep(config.FETCH_DELAY_SECONDS)
                docs = _event_docs(client, season, season_url, event)
                n_sub = sum(d.doc_type in SUBSTANTIVE_TYPES for d in docs)
                print(f"   {event:38s} {len(docs):3d} docs  ({n_sub} substantive)")
                all_docs.extend(docs)

    with MANIFEST.open("w") as f:
        for d in all_docs:
            f.write(json.dumps(asdict(d)) + "\n")
    print(f"\nwrote {len(all_docs)} records -> {MANIFEST}")
    return all_docs


def load_manifest() -> list[Doc]:
    if not MANIFEST.exists():
        raise SystemExit("No manifest. Run: python -m corpus.fetch_fia manifest")
    return [Doc(**json.loads(line)) for line in MANIFEST.open()]


def download(types: set[str] | None = None) -> None:
    """Fetch each wanted PDF exactly once. Safe to re-run; already-present files skip."""
    types = types or SUBSTANTIVE_TYPES
    docs = [d for d in load_manifest() if d.doc_type in types]
    print(f"{len(docs)} documents match {sorted(types)}")

    fetched = skipped = failed = 0
    with _client() as client:
        for i, d in enumerate(docs, 1):
            dest = config.PDFS / f"{d.season}_{d.filename}"
            if dest.exists() and dest.stat().st_size > 0:
                skipped += 1
                continue
            try:
                time.sleep(config.FETCH_DELAY_SECONDS)
                r = _get(client, d.url)
                dest.write_bytes(r.content)
                fetched += 1
                if fetched % 25 == 0:
                    print(f"   [{i}/{len(docs)}] fetched={fetched} skipped={skipped}")
            except Exception as e:  # noqa: BLE001 - one bad PDF must not kill the run
                failed += 1
                print(f"   ! {d.filename}: {type(e).__name__}: {e}")

    print(f"\nfetched={fetched} already-had={skipped} failed={failed}")


def stats() -> None:
    from collections import Counter
    docs = load_manifest()
    print(f"total documents: {len(docs)}\n")
    print("by type:")
    for t, n in Counter(d.doc_type for d in docs).most_common():
        print(f"   {t:16s} {n:5d}")
    print("\nby season (substantive only):")
    for s, n in sorted(Counter(d.season for d in docs if d.doc_type in SUBSTANTIVE_TYPES).items()):
        print(f"   {s} {n:5d}")
    print("\nmost common subjects:")
    for subj, n in Counter(
        d.subject for d in docs if d.doc_type in SUBSTANTIVE_TYPES and d.subject
    ).most_common(15):
        print(f"   {n:3d}  {subj[:70]}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "manifest"
    {"manifest": build_manifest, "download": download, "stats": stats}[cmd]()
