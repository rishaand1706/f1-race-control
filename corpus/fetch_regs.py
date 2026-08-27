"""Fetch the rulebooks, every issue of them.

The FIA reissues the Sporting Regulations several times per season. A March 2026
incident is governed by the issue in force in March 2026, not by the latest PDF. So we
store *every* issue with the date range it was in force, and retrieval filters on that
range before ranking. See docs/concepts/02-metadata-and-versioning.md.

The registry below is hand-verified: every URL was confirmed to return HTTP 200. The FIA
does not expose a machine-readable regulation index, and guessing URL patterns produced
404s, so an explicit list is both more honest and more reliable than a scraper here.

Usage:
    python -m corpus.fetch_regs
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict

import httpx

import config

REGS_MANIFEST = config.INDEX / "regulations.jsonl"
FIA = "https://www.fia.com"


@dataclass
class Reg:
    season: int
    kind: str            # "sporting" | "penalty_guidelines" | "driving_standards"
    issue: str
    effective_from: str  # ISO date this issue was published
    url: str
    effective_to: str | None = None   # filled in below: the next issue's start


# --- Hand-verified registry -------------------------------------------------
# Gap noted honestly: the FIA published later 2025 Sporting Regulations issues that are
# no longer linked from the current regulations pages. Issues 3-5 cover Dec 2024 to the
# end of the season for our purposes; a 2025 incident after Apr 2025 resolves to Issue 5,
# which may not be the exact issue in force. Flagged rather than silently assumed.
_REGISTRY: list[Reg] = [
    # 2025 Sporting Regulations
    Reg(2025, "sporting", "Issue 3", "2024-12-11",
        f"{FIA}/sites/default/files/fia_2025_formula_1_sporting_regulations_-_issue_3_-_2024-12-11.pdf"),
    Reg(2025, "sporting", "Issue 4", "2025-02-26",
        f"{FIA}/system/files/documents/fia_2025_formula_1_sporting_regulations_-_issue_4_-_2025-02-26.pdf"),
    Reg(2025, "sporting", "Issue 5", "2025-04-30",
        f"{FIA}/system/files/documents/fia_2025_formula_1_sporting_regulations_-_issue_5_-_2025-04-30.pdf"),

    # 2026 Sporting Regulations (Section B) - reissued five times inside the season
    Reg(2026, "sporting", "Issue 04", "2025-12-10",
        f"{FIA}/system/files/documents/fia_2026_f1_regulations_-_section_b_sporting_-_iss_04_-_2025-12-10_0.pdf"),
    Reg(2026, "sporting", "Issue 05", "2026-02-27",
        f"{FIA}/system/files/documents/fia_2026_f1_regulations_-_section_b_sporting_-_iss_05_-_2026-02-27.pdf"),
    Reg(2026, "sporting", "Issue 06", "2026-04-28",
        f"{FIA}/system/files/documents/fia_2026_f1_regulations_-_section_b_sporting_-_iss_06_-_2026-04-28.pdf"),
    Reg(2026, "sporting", "Issue 07", "2026-06-25",
        f"{FIA}/system/files/documents/fia_2026_f1_regulations_-_section_b_sporting_-_iss_07_-_2026-06-25.pdf"),
    Reg(2026, "sporting", "Issue 08", "2026-08-05",
        f"{FIA}/system/files/documents/fia_2026_f1_regulations_-_section_b_sporting_-_iss_08_-_2026-08-05_7.pdf"),

    # Stewards' guidelines. Not regulations, but they state the *recommended* sanction
    # per infringement type, which is exactly what a penalty prediction should cite.
    Reg(2026, "penalty_guidelines", "2026", "2026-03-01",
        f"{FIA}/sites/default/files/2026_f1_penalty_guidelines.pdf"),
    Reg(2026, "driving_standards", "2026", "2026-03-01",
        f"{FIA}/sites/default/files/2026_f1_driving_standards_guidelines.pdf"),
]


def registry() -> list[Reg]:
    """Registry with effective_to derived from the next issue of the same season+kind."""
    regs = [Reg(**asdict(r)) for r in _REGISTRY]
    for key in {(r.season, r.kind) for r in regs}:
        group = sorted(
            [r for r in regs if (r.season, r.kind) == key],
            key=lambda r: r.effective_from,
        )
        for a, b in zip(group, group[1:]):
            a.effective_to = b.effective_from
    return regs


def local_path(reg: Reg) -> "config.Path":  # type: ignore[name-defined]
    from pathlib import Path
    stem = f"{reg.season}_{reg.kind}_{reg.issue.replace(' ', '')}"
    return Path(config.PDFS) / f"{stem}.pdf"


def fetch_all() -> list[Reg]:
    regs = registry()
    headers = {"User-Agent": config.USER_AGENT}
    with httpx.Client(headers=headers, timeout=httpx.Timeout(read=180.0, connect=15.0,
                                                             write=30.0, pool=15.0),
                      follow_redirects=True) as client:
        for r in regs:
            dest = local_path(r)
            if dest.exists() and dest.stat().st_size > 0:
                print(f"  have  {dest.name}")
                continue
            time.sleep(config.FETCH_DELAY_SECONDS)
            resp = client.get(r.url)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            print(f"  got   {dest.name}  ({len(resp.content) / 1024:.0f} KB)")

    with REGS_MANIFEST.open("w") as f:
        for r in regs:
            f.write(json.dumps(asdict(r)) + "\n")

    print(f"\n{len(regs)} regulation documents -> {REGS_MANIFEST}")
    print("\nversions in force:")
    for r in sorted(regs, key=lambda r: (r.season, r.kind, r.effective_from)):
        until = r.effective_to or "current"
        print(f"  {r.season} {r.kind:20s} {r.issue:9s} {r.effective_from} -> {until}")
    return regs


def load_registry() -> list[Reg]:
    if not REGS_MANIFEST.exists():
        raise SystemExit("No regulations manifest. Run: python -m corpus.fetch_regs")
    return [Reg(**json.loads(line)) for line in REGS_MANIFEST.open()]


def in_force(season: int, kind: str, date_iso: str) -> Reg | None:
    """The issue governing `date_iso`. This is the temporal filter, in one function."""
    candidates = [
        r for r in load_registry()
        if r.season == season and r.kind == kind and r.effective_from <= date_iso
    ]
    return max(candidates, key=lambda r: r.effective_from) if candidates else None


if __name__ == "__main__":
    fetch_all()
