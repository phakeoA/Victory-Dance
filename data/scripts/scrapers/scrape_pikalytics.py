"""
scrape_pikalytics.py
====================
Scrapes Pikalytics for [Gen 9 Champions] VGC 2026 Reg M-A and produces
a single JSON file:  pikalytics_regma.json

Uses Crawl4AI (v0.8.x) AsyncWebCrawler — no BeautifulSoup, no requests lib.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT SCHEMA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{
  "format":     "gen9championsvgc2026regmb",
  "scraped_at": "2026-06-09T...",
  "pokemon": {
    "Kingambit": {
      "usage_pct": 38.0,
      "moves":     [{"name": "Sucker Punch",  "pct": 99.3}, ...],
      "items":     [{"name": "Chople Berry",  "pct": 51.9}, ...],
      "abilities": [{"name": "Defiant",       "pct": 94.1}, ...],
      "spreads":   [
        {"nature": "Adamant", "evs": [32,32,0,0,2,0], "pct": 7.82},
        ...                          # HP/Atk/Def/SpA/SpD/Spe
      ],
      "teammates": [{"name": "Sneasler", "pct": 52.1}, ...]
    },
    ...
  }
}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  pip install "crawl4ai>=0.8" && crawl4ai-setup

  # Full scrape (all mons)
  python scrape_pikalytics.py

  # Quick test — first 5 mons only
  python scrape_pikalytics.py --limit 5

  # Resume a partial scrape (skips already-scraped mons)
  python scrape_pikalytics.py --resume

  # Tune concurrency (default 2 — be polite to the server)
  python scrape_pikalytics.py --concurrency 3

HOW IT WORKS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1.  The full roster is enumerated from Smogon's published usage stats
    (the upstream source of Pikalytics' data). The Pikalytics index page
    only server-renders the top ~50 mons, so scraping links from it
    misses everything below that cutoff (e.g. regular Scizor). Index
    links are still extracted and merged in as a fallback.
2.  Individual mon pages are fetched one-at-a-time through a Semaphore
    so we never hammer the server with more than --concurrency tabs at once.
3.  Random jitter (1–4 s) is injected between every request and a longer
    pause (8–18 s) is added between batches so the traffic pattern looks
    human and won't trigger rate-limiting.
4.  BrowserConfig uses enable_stealth=True and a realistic user-agent.
    CrawlerRunConfig uses magic=True + max_retries=2 for resilience.
5.  Results are saved incrementally so --resume always works.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai import JsonCssExtractionStrategy

# ── Config ─────────────────────────────────────────────────────────────────────
FORMAT_SLUG  = "gen9championsvgc2026regma"
BASE_URL     = "https://www.pikalytics.com"
INDEX_URL    = f"{BASE_URL}/pokedex/{FORMAT_SLUG}"

# Smogon publishes the usage stats Pikalytics is built on. The "-0" file
# (unweighted baseline) lists every Pokémon that appeared in the format,
# which lets us enumerate the full roster instead of just the top ~50
# that the Pikalytics index page renders.
SMOGON_STATS_BASE = "https://www.smogon.com/stats"

# Output goes to  <project_root>/data/pikalytics_regma.json
# __file__ is     <project_root>/data/scripts/scrape_pikalytics.py
# data/scripts/scrapers/ → parents[1]==data/ , parents[2]==project root
_SCRIPT_DIR = Path(__file__).resolve().parent        # data/scripts/scrapers/
OUTPUT_DIR  = _SCRIPT_DIR.parents[2] / "data"        # <project_root>/data/
OUTPUT_FILE = OUTPUT_DIR / "pikalytics_regma.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)        # create data/ if missing

# Realistic Chrome 124 user-agent — stealth mode will also randomise navigator
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# Delay ranges (seconds) — tweak to taste
JITTER_MIN   = 1.0   # minimum pause between individual requests
JITTER_MAX   = 4.0   # maximum pause between individual requests
BATCH_MIN    = 8.0   # minimum pause between batches
BATCH_MAX    = 18.0  # maximum pause between batches


# ── CSS extraction schema for the index page ──────────────────────────────────

INDEX_SCHEMA = {
    "name": "pokemon_links",
    "baseSelector": f"a[href*='/pokedex/{FORMAT_SLUG}/']",
    "fields": [
        {"name": "href", "type": "attribute", "attribute": "href"},
    ],
}


# ── Regex parsers — applied to result.markdown.raw_markdown ───────────────────

_PCT_PATTERN = re.compile(
    r"([A-Za-z][A-Za-z0-9 '\-\(\)]+?)\s+(\d+\.\d+)%"
)

_SPREAD_PATTERN = re.compile(
    r"(Adamant|Modest|Jolly|Timid|Brave|Quiet|Bold|Impish|Careful|Calm"
    r"|Sassy|Relaxed|Gentle|Hasty|Naive|Lax|Rash|Naughty|Lonely|Mild"
    r"|Hardy|Docile|Serious|Bashful|Quirky)"
    r"\s+(\d+)\s*/\s*(\d+)\s*/\s*(\d+)\s*/\s*(\d+)\s*/\s*(\d+)\s*/\s*(\d+)"
    r"\s+(\d+\.\d+)%",
    re.IGNORECASE,
)

_USAGE_PATTERN = re.compile(r"Usage Percent\s*\n+\s*(\d+(?:\.\d+)?)%", re.IGNORECASE)

# Types to skip when parsing section entries
_TYPE_NOISE = {
    "normal","fire","water","grass","electric","ice","fighting","poison",
    "ground","flying","psychic","bug","rock","ghost","dragon","dark",
    "steel","fairy","other",
}

# New-layout (2026) section boundaries: the spreads block has a PLAIN-TEXT header
# "Nature / EV Spreads" (not a ## heading) and the page ends with an FAQ — both must
# bound the ##-delimited sections, else 'Best Abilities' bleeds into the nature list.
_SECTION_END_RE = re.compile(
    r"^#{1,4}\s|Nature\s*/\s*EV Spreads|Frequently Asked Questions",
    re.MULTILINE | re.IGNORECASE,
)

_NATURES_ALT = (
    "Adamant|Modest|Jolly|Timid|Brave|Quiet|Bold|Impish|Careful|Calm|Sassy|"
    "Relaxed|Gentle|Hasty|Naive|Lax|Rash|Naughty|Lonely|Mild|Hardy|Docile|"
    "Serious|Bashful|Quirky"
)
# New layout: EV spreads are bare 32-notation rows (hp/atk/def/spa/spd/spe + pct),
# NO LONGER paired inline with a nature; natures are a separate pct-weighted list.
_EV_ROW_RE = re.compile(
    r"(\d+)\s*/\s*(\d+)\s*/\s*(\d+)\s*/\s*(\d+)\s*/\s*(\d+)\s*/\s*(\d+)\s+(\d+\.\d+)\s*%"
)
_NATURE_PCT_RE = re.compile(rf"\b({_NATURES_ALT})\b\s+(\d+\.\d+)\s*%", re.IGNORECASE)

# A nature is NEVER a valid ability/item name. The "Best Abilities"/"Best Items" section
# slice is bounded by the next heading or the "Nature / EV Spreads" marker; when that block
# is captured mid lazy-render the boundary can be missed and the nature distribution bleeds
# into the section, so e.g. "Adamant 89.9%" parses as an ability (verified: Swampert M-B —
# 'Adamant' even outranked the real 'Torrent'). Filter the finite nature set out defensively,
# the same way _TYPE_NOISE drops the type-row labels.
_NATURE_NOISE = frozenset(n.lower() for n in _NATURES_ALT.split("|"))


def _parse_section(markdown: str, header: str) -> list[dict]:
    """
    Extract (name, pct) pairs from a named section.

    Actual Pikalytics heading format: "## Best Moves for Sableye"
    i.e. the section keyword appears after the ## but the full heading includes
    " for <PokemonName>" — so we match the keyword anywhere in the heading line.

    Strategy: find the heading line, slice from there to the next ## heading,
    then run the PCT pattern over that slice. No DOTALL on the body capture to
    avoid catastrophic backtracking.
    """
    base = header.split()[-1]           # "Moves", "Items", "Abilities", etc.
    alternates = [
        re.escape(header),              # "Best Moves"
        re.escape(f"Top {base}"),       # "Top Moves"
        re.escape(base),                # "Moves"
    ]
    keyword_pat = "(?:" + "|".join(alternates) + ")"

    # Match a heading line that contains the keyword (with optional suffix like " for Sableye")
    heading_re = re.compile(
        r"^#{1,4}\s[^\n]*" + keyword_pat + r"[^\n]*$",
        re.IGNORECASE | re.MULTILINE,
    )
    m = heading_re.search(markdown)
    if not m:
        return []

    # Slice from end of that heading line to the next heading (or EOF)
    start = m.end()
    nxt = _SECTION_END_RE.search(markdown, start)
    end = nxt.start() if nxt else len(markdown)
    section_text = markdown[start:end]

    results: list[dict] = []
    seen: set[str] = set()
    for match in _PCT_PATTERN.finditer(section_text):
        name = match.group(1).strip()
        pct  = float(match.group(2))
        if name.lower() in _TYPE_NOISE or name.lower() in _NATURE_NOISE:
            continue
        if name not in seen:
            seen.add(name)
            results.append({"name": name, "pct": pct})
    return results


def _parse_moves(markdown: str) -> list[dict]:
    """
    Pikalytics move section layout (each entry spans 3 content lines):

        Light Screen
        psychic
        74.917%

    Standard _PCT_PATTERN grabs "psychic 74.917%" (the type, not the name).
    Instead we find the section, split into lines, and walk them as a triplet:
    name-line → type-line → pct-line.
    """
    heading_re = re.compile(
        r"^#{1,4}\s[^\n]*(?:Best\s+Moves|Top\s+Moves|Moves)[^\n]*$",
        re.IGNORECASE | re.MULTILINE,
    )
    m = heading_re.search(markdown)
    if not m:
        return []
    start = m.end()
    nxt = _SECTION_END_RE.search(markdown, start)
    end = nxt.start() if nxt else len(markdown)
    section_text = markdown[start:end]

    # Strip blank lines, collect non-empty content lines
    lines = [l.strip() for l in section_text.splitlines() if l.strip()]

    _TYPE_SET = {
        "normal","fire","water","grass","electric","ice","fighting","poison",
        "ground","flying","psychic","bug","rock","ghost","dragon","dark",
        "steel","fairy","other",
    }
    _PCT_RE = re.compile(r"^(\d+\.\d+)%$")
    _NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9 '\-\(\)]+$")

    results: list[dict] = []
    seen: set[str] = set()
    i = 0
    while i < len(lines):
        line = lines[i]
        # Expect: move name (not a type word, not a percentage)
        if _NAME_RE.match(line) and line.lower() not in _TYPE_SET and not _PCT_RE.match(line):
            name = line
            # Next non-empty line should be the type, then the pct
            if i + 2 < len(lines):
                type_line = lines[i + 1]
                pct_line  = lines[i + 2]
                if type_line.lower() in _TYPE_SET and _PCT_RE.match(pct_line):
                    pct = float(pct_line.rstrip("%"))
                    if name not in seen:
                        seen.add(name)
                        results.append({"name": name, "pct": pct})
                    i += 3
                    continue
        i += 1
    return results


def _parse_spreads(markdown: str) -> tuple[list[dict], list[dict]]:
    """Parse the new-layout 'Nature / EV Spreads' block.

    The page no longer pairs a nature with each EV spread inline — it lists EV
    spreads (bare hp/atk/def/spa/spd/spe 32-notation rows + pct) and natures
    (name + pct) as TWO separate distributions. We therefore:
      * extract the EV spreads (with pct),
      * extract the nature distribution,
      * attach the MODAL (most-common) nature to each EV spread so the
        {nature, evs, pct} schema stays identical to the M-A scrape (downstream
        belief/est-stat code is unchanged). The true nature distribution is also
        returned separately (new ``natures`` field) so nothing is lost.

    Returns (spreads, natures).
    """
    m = re.search(r"Nature\s*/\s*EV Spreads", markdown, re.IGNORECASE)
    if not m:
        return [], []
    start = m.end()
    nxt = re.search(r"Frequently Asked Questions|^#{1,4}\s", markdown[start:], re.MULTILINE)
    end = start + nxt.start() if nxt else len(markdown)
    section = re.sub(r"\s+", " ", markdown[start:end])     # collapse line-split rows

    ev_rows = [
        ([int(g) for g in row[:6]], float(row[6]))
        for row in _EV_ROW_RE.findall(section)
    ]
    natures = [
        {"nature": n.capitalize(), "pct": float(p)}
        for (n, p) in _NATURE_PCT_RE.findall(section)
    ]
    modal = max(natures, key=lambda x: x["pct"])["nature"] if natures else None
    spreads = [
        {"nature": modal, "evs": evs, "pct": pct}          # HP Atk Def SpA SpD Spe
        for (evs, pct) in ev_rows
    ]
    return spreads, natures


def _parse_usage(markdown: str) -> Optional[float]:
    m = _USAGE_PATTERN.search(markdown)
    return float(m.group(1)) if m else None


def _parse_teammates(markdown: str) -> list[dict]:
    """
    Teammate links look like:
      [ ![Charizard-Mega-Y](https://cdn...png) Mega Charizard Y fireflying 43.893% ](url)

    The image alt text carries the canonical form name (matches the
    Smogon/JSON-key naming, e.g. "Charizard-Mega-Y"), so we use that
    rather than the display name.

    Heading is "## Best Teammates for <Name>" — use the same slice approach
    as _parse_section.
    """
    teammates: list[dict] = []
    seen: set[str] = set()

    heading_re = re.compile(r"^#{1,4}\s[^\n]*Teammates[^\n]*$", re.IGNORECASE | re.MULTILINE)
    m = heading_re.search(markdown)
    if not m:
        return []

    start = m.end()
    nxt = _SECTION_END_RE.search(markdown, start)
    end = nxt.start() if nxt else len(markdown)
    section_text = markdown[start:end]

    # New layout: each teammate is a sprite-link carrying a RANK (#1..#10), not a
    # pct (the FAQ literally renders "undefined%"). Capture name (image alt) + rank.
    for match in re.finditer(
        r"!\[([^\]]+)\]\([^)]*\)[^#\]]*?#(\d+)", section_text
    ):
        name = match.group(1).strip()
        rank = int(match.group(2))
        if name not in seen:
            seen.add(name)
            teammates.append({"name": name, "rank": rank})
    return teammates


def _parse_mon_page(markdown: str) -> dict:
    spreads, natures = _parse_spreads(markdown)
    return {
        "usage_pct": _parse_usage(markdown),
        "moves":     _parse_moves(markdown),
        "items":     _parse_section(markdown, "Best Items"),
        "abilities": _parse_section(markdown, "Best Abilities"),
        "spreads":   spreads,
        "natures":   natures,
        "teammates": _parse_teammates(markdown),
    }


# ── Crawl4AI browser / run configs ────────────────────────────────────────────

def _make_browser_config() -> BrowserConfig:
    """
    Stealth mode masks WebDriver flags, randomises navigator properties,
    and makes the browser look like an ordinary Chrome window.
    """
    return BrowserConfig(
        headless=True,
        user_agent=USER_AGENT,
        enable_stealth=True,   # patch navigator.webdriver, plugins, etc.
        verbose=False,
    )


def _make_run_config(*, use_extraction: bool = False) -> CrawlerRunConfig:
    """
    magic=True  – randomise timings, mouse movement, remove overlay elements.
    max_retries – retry on HTTP 429/403 with automatic back-off.
    wait_until  – wait for full page load (important for JS-rendered stats).
    """
    kwargs: dict = dict(
        cache_mode  = CacheMode.BYPASS,
        magic       = False,
        # The Nature/EV-Spreads + Teammates blocks are LAZY JS-rendered (they land
        # AFTER the `load` event). Single/low-concurrency fetches give the JS time to
        # render; under high concurrency the tabs starve and pages get captured
        # spread-less. ``networkidle`` is unusable (Pikalytics keeps persistent
        # connections, so it never settles) — use low concurrency instead.
        wait_until  = "load",
        page_timeout= 30_000,   # ms
        max_retries = 2,
        # NOTE: the Nature/EV-Spreads + Teammates blocks are async-rendered AFTER
        # `load` and crawl4ai captures the HTML at an inconsistent moment, so a
        # subset of pages come back spread-less (flaky lazy-render). Tried
        # networkidle / delay_before_return_html / scan_full_page / in-page JS poll
        # — each hung, errored (antibot), or stayed flaky. Reliable fix TBD (likely a
        # targeted single-fetch RETRY pass for data-mons missing spreads, since the
        # single --debug fetch reliably renders them). Young M-B meta is thin anyway.
    )
    if use_extraction:
        kwargs["extraction_strategy"] = JsonCssExtractionStrategy(INDEX_SCHEMA)
    return CrawlerRunConfig(**kwargs)


# ── Roster enumeration ─────────────────────────────────────────────────────────

def fetch_smogon_roster() -> list[tuple[str, float]]:
    """
    Return [(name, usage_pct), ...] for every Pokémon in the format, taken
    from the most recent Smogon usage-stats month available (walking back
    up to 6 months). Names use the same convention as Pikalytics URLs and
    our JSON keys ("Scizor-Mega", "Charizard-Mega-Y", ...).

    Returns [] if Smogon is unreachable — caller falls back to index links.
    """
    now = datetime.now(timezone.utc)
    year, month = now.year, now.month
    for _ in range(6):
        url = f"{SMOGON_STATS_BASE}/{year:04d}-{month:02d}/{FORMAT_SLUG}-0.txt"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=30) as resp:
                text = resp.read().decode("utf-8", errors="replace")
        except Exception:
            text = ""

        roster: list[tuple[str, float]] = []
        for line in text.splitlines():
            # | 78   | Scizor             |  1.69063% | 113462 |  1.691% | ...
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 4 and parts[1].isdigit():
                try:
                    usage = float(parts[3].rstrip("%"))
                except ValueError:
                    usage = 0.0
                roster.append((parts[2], usage))
        if roster:
            print(f"[roster] {len(roster)} Pokémon from Smogon stats {year:04d}-{month:02d}.")
            return roster

        # Walk back one month
        month -= 1
        if month == 0:
            month, year = 12, year - 1

    print("[roster] Smogon stats unavailable; falling back to index links only.")
    return []


def _load_champions_roster(format_slug: str, refresh: bool = False) -> dict[str, str]:
    """Return {pikalytics_slug (toID): display_name} for ``format_slug``'s FULL
    in-format roster, enumerated from the LOCAL Showdown champions mod (offline,
    authoritative, future-reg-proof).

    Replaces the old Smogon-usage roster, which DOES NOT EXIST for a freshly
    launched reg (Smogon publishes monthly in arrears). The slug (key) is the
    Pikalytics URL path (pure toID, no escaping); the display name (value) is the
    OUTPUT JSON key, kept byte-identical to the M-A scrape's keys (e.g.
    'Charizard-Mega-Y', 'Rotom-Wash', 'Mr. Rime') so downstream name-keyed code
    (TP features / encoders) keeps matching.  Cached to
    data/champions_roster_<reg>.json — pass ``refresh`` to regenerate.
    """
    import subprocess

    reg_m = re.search(r"(reg[a-z0-9]+)", format_slug)
    reg = reg_m.group(1) if reg_m else "unknown"
    cache = OUTPUT_DIR / f"champions_roster_{reg}.json"
    if cache.exists() and not refresh:
        with cache.open(encoding="utf-8") as f:
            roster = json.load(f)
        print(f"[roster] {len(roster)} species from cache {cache.name} "
              f"(use --refresh-roster to re-enumerate).")
        return roster

    node = _SCRIPT_DIR.parents[2] / ".venv" / "node" / "node.exe"
    node_cmd = str(node) if node.exists() else "node"
    enum_js = _SCRIPT_DIR / "enum_champions_roster.js"
    proc = subprocess.run(
        [node_cmd, str(enum_js), format_slug],
        capture_output=True, text=True, encoding="utf-8",
    )
    if proc.returncode != 0 or not (proc.stdout or "").strip():
        raise RuntimeError(
            f"roster enumeration failed for {format_slug} "
            f"(is pokemon-showdown present + built?): {(proc.stderr or '').strip()}"
        )
    roster = json.loads(proc.stdout)
    with cache.open("w", encoding="utf-8") as f:
        json.dump(roster, f, ensure_ascii=False, indent=0)
    print(f"[roster] {len(roster)} in-format species for {format_slug} "
          f"(enumerated from Showdown mod; cached {cache.name}).")
    return roster


async def get_pokemon_list(crawler=None, min_usage: float = 0.0,
                           refresh_roster: bool = False) -> list[tuple[str, str]]:
    """Return [(pikalytics_slug, display_name), ...] for the FULL in-format roster,
    enumerated from the local Showdown champions mod (see _load_champions_roster).

    ``crawler`` / ``min_usage`` are accepted for call-site compatibility but unused:
    the roster is legality-derived (complete), not usage-thresholded.
    """
    roster = _load_champions_roster(FORMAT_SLUG, refresh=refresh_roster)
    pairs = sorted(roster.items(), key=lambda kv: kv[1].lower())
    print(f"[roster] {len(pairs)} Pokémon to scrape for {FORMAT_SLUG}.")
    return pairs


# ── Individual page fetching (rate-limited) ────────────────────────────────────

async def _fetch_one(
    mon_id: str,
    mon_name: str,
    sem: asyncio.Semaphore,
) -> tuple[str, dict]:
    """
    Fetch a single mon page with a FRESH crawler. The lazy Nature/EV-Spreads +
    Teammates sections render reliably on a fresh browser but FLAKE when one
    crawler is reused across many pages (verified: fresh-per-fetch = 100% spread
    coverage on high-usage mons; reused crawler degrades after a handful). Retries
    ONCE on a suspected lazy-render miss (moves present but spreads empty).

    The Pikalytics URL slug is the DISPLAY name (e.g. 'Abomasnow-Mega', 'Rotom-Wash',
    'Mr. Rime'), NOT the toID — mega/forme usage lives on the hyphenated-name page.
    Returns (mon_name, parsed_data) keyed by the DISPLAY name.
    """
    url = f"{INDEX_URL}/{urllib.parse.quote(mon_name)}"

    async def _attempt() -> tuple[bool, str, dict]:
        async with AsyncWebCrawler(config=_make_browser_config()) as crawler:
            result = await crawler.arun(url, config=_make_run_config())
        if not result.success:
            return False, result.error_message, {}
        md = result.markdown.raw_markdown if result.markdown else ""
        return True, "", _parse_mon_page(md)

    async with sem:
        await asyncio.sleep(random.uniform(JITTER_MIN, JITTER_MAX))   # human-ish jitter
        ok, err, data = await _attempt()
        # Retry once on a likely lazy-render miss: page loaded (has moves) but the
        # async spread block hadn't rendered when the HTML was captured.
        if ok and data.get("moves") and not data.get("spreads"):
            await asyncio.sleep(random.uniform(JITTER_MIN, JITTER_MAX))
            ok2, _err2, data2 = await _attempt()
            if ok2 and data2.get("spreads"):
                data = data2

    if not ok:
        print(f"    [!] {mon_name}: fetch failed — {err}")
        return mon_name, {"error": err}

    print(f"    ✓ {mon_name}: {len(data['moves'])} moves, {len(data['spreads'])} spreads")
    return mon_name, data


async def scrape_all(
    roster: list[tuple[str, str]],
    concurrency: int,
) -> dict[str, dict]:
    """
    Fetch all mon pages. ``roster`` is a list of (slug_id, display_name) pairs.
    Each fetch uses a FRESH crawler (reliable lazy-render); the semaphore caps how
    many browsers run at once.
    """
    sem = asyncio.Semaphore(concurrency)
    tasks = [_fetch_one(mon_id, mon_name, sem) for (mon_id, mon_name) in roster]
    pairs = await asyncio.gather(*tasks)
    return dict(pairs)


# ── Main ───────────────────────────────────────────────────────────────────────

async def main(limit: int, concurrency: int, resume: bool, min_usage: float,
               refresh_roster: bool = False) -> None:
    # ── Load existing data if resuming ────────────────────────────────────────
    existing: dict = {}
    already: set[str] = set()
    if resume and OUTPUT_FILE.exists():
        with OUTPUT_FILE.open(encoding="utf-8") as f:
            existing = json.load(f)
        already = set(existing.get("pokemon", {}).keys())
        print(f"[resume] Loaded {len(already)} existing entries.")

    browser_conf = _make_browser_config()

    async with AsyncWebCrawler(config=browser_conf) as crawler:
        # ── Step 1: roster ────────────────────────────────────────────────────
        roster = await get_pokemon_list(crawler, min_usage, refresh_roster=refresh_roster)

        if limit:
            roster = roster[:limit]
            print(f"[roster] Limited to first {limit}.")

        to_scrape = [(mid, mname) for (mid, mname) in roster if mname not in already]
        print(f"[scrape] {len(to_scrape)} mons to scrape (concurrency={concurrency}).")

        if not to_scrape:
            print("[scrape] Nothing to do.")
            return

        # ── Step 2: prepare output dict ───────────────────────────────────────
        result = existing if resume else {
            "format":     FORMAT_SLUG,
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "pokemon":    {},
        }
        # Ensure metadata even on a --resume that started from an empty/absent file
        # (otherwise `result = existing` is a bare {} missing format/scraped_at).
        result.setdefault("format", FORMAT_SLUG)
        result.setdefault("scraped_at", datetime.now(timezone.utc).isoformat())
        result.setdefault("pokemon", {})

        # ── Step 3: scrape in batches (incremental saves + inter-batch pause) ─
        BATCH_SIZE = max(concurrency, 5)
        total_batches = (len(to_scrape) + BATCH_SIZE - 1) // BATCH_SIZE

        for batch_idx in range(total_batches):
            batch_start = batch_idx * BATCH_SIZE
            batch = to_scrape[batch_start : batch_start + BATCH_SIZE]

            print(
                f"\n[batch {batch_idx + 1}/{total_batches}] "
                f"mons {batch_start + 1}–{batch_start + len(batch)}"
            )

            batch_data = await scrape_all(batch, concurrency)
            result["pokemon"].update(batch_data)

            # Incremental save after every batch
            with OUTPUT_FILE.open("w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"    [saved] {OUTPUT_FILE} ({OUTPUT_FILE.stat().st_size / 1024:.1f} KB)")

            # Inter-batch pause — give the server a breather
            if batch_idx < total_batches - 1:
                pause = random.uniform(BATCH_MIN, BATCH_MAX)
                print(f"    [pause] Waiting {pause:.1f}s before next batch…")
                await asyncio.sleep(pause)

    total = len(result["pokemon"])
    size_kb = OUTPUT_FILE.stat().st_size / 1024
    print(f"\n[done] Saved {total} entries → {OUTPUT_FILE}  ({size_kb:.1f} KB)")


# ── Debug helper ───────────────────────────────────────────────────────────────

async def _debug_markdown(mon_slug: str) -> None:
    """
    Fetch one mon page and dump its raw markdown so you can inspect section
    headers and diagnose parse failures.

    Usage:  python scrape_pikalytics.py --debug-markdown kingambit
    Output: debug_<mon_slug>.md  (written next to the script)
    """
    url = f"{INDEX_URL}/{urllib.parse.quote(mon_slug)}"
    print(f"[debug] Fetching {url}")
    async with AsyncWebCrawler(config=_make_browser_config()) as crawler:
        result = await crawler.arun(url, config=_make_run_config())

    if not result.success:
        print(f"[debug] FAILED: {result.error_message}")
        return

    markdown = result.markdown.raw_markdown if result.markdown else ""
    out_path = _SCRIPT_DIR / f"debug_{mon_slug}.md"
    out_path.write_text(markdown, encoding="utf-8")
    print(f"[debug] Saved {len(markdown)} chars → {out_path}")

    # Also run the parser and show what it found
    data = _parse_mon_page(markdown)
    print(f"\n[debug] Parsed result:")
    for key, val in data.items():
        print(f"  {key}: {val!r}")

    # Show first 120 chars around the word "moves" in the markdown
    lower_md = markdown.lower()
    idx = lower_md.find("move")
    if idx >= 0:
        snippet = markdown[max(0, idx-20):idx+200]
        print(f"\n[debug] Context around first 'move' mention:\n{snippet!r}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def cli() -> None:
    # Windows consoles often default to cp1252, which can't print ✓/→/–
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    parser = argparse.ArgumentParser(
        description="Scrape Pikalytics Champions-doubles usage data using Crawl4AI (v0.8.x).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--limit",
        type=int, default=0,
        help="Only scrape the first N Pokémon (0 = all)",
    )
    parser.add_argument(
        "--concurrency",
        type=int, default=2,
        help="Max simultaneous browser tabs (keep ≤3 to avoid rate-limiting)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip Pokémon already present in the output file",
    )
    parser.add_argument(
        "--min-usage",
        type=float, default=0.0,
        help="Skip Pokémon below this Smogon usage %% (0 = scrape everything)",
    )
    parser.add_argument(
        "--debug-markdown",
        metavar="MON",
        default=None,
        help=(
            "Fetch a single mon page, print its raw markdown, and exit. "
            "Useful for diagnosing section-header mismatches. "
            "Example: --debug-markdown kingambit"
        ),
    )
    parser.add_argument(
        "--format", "-f", default=None, dest="format_slug",
        help="Champions-doubles format id to scrape (default: the active format). "
             "e.g. gen9championsvgc2026regma for M-A. Drives the Pikalytics URL + "
             "the output filename pikalytics_<reg>.json (lockstep).",
    )
    parser.add_argument(
        "--refresh-roster", action="store_true",
        help="Re-enumerate the roster from the Showdown mod (ignore the cached "
             "data/champions_roster_<reg>.json).",
    )
    args = parser.parse_args()

    # Select the format and REBIND the URL + output filename in LOCKSTEP, so a scrape
    # can never write one reg's data into another reg's file.
    global FORMAT_SLUG, INDEX_URL, OUTPUT_FILE
    slug = args.format_slug
    if slug is None:
        try:
            from v_dance.formats import default_format
            slug = default_format()
        except Exception:
            slug = FORMAT_SLUG
    FORMAT_SLUG = slug
    INDEX_URL = f"{BASE_URL}/pokedex/{FORMAT_SLUG}"
    _reg_m = re.search(r"(reg[a-z0-9]+)", FORMAT_SLUG)
    OUTPUT_FILE = OUTPUT_DIR / (f"pikalytics_{_reg_m.group(1)}.json" if _reg_m
                               else "pikalytics_regma.json")
    print(f"[config] format={FORMAT_SLUG}  url={INDEX_URL}  output={OUTPUT_FILE.name}")

    # ⚠ Playwright must SPAWN its driver subprocess, which the Windows SelectorEventLoop cannot
    # do (asyncio raises NotImplementedError) — and the ``v_dance.formats`` import above just
    # installed the Selector policy GLOBALLY (v_dance/__init__'s poke-env Ctrl-C fix). This
    # process runs no poke-env, so restore the Proactor policy before the crawl — the same
    # dance as play_vs_human_browser._use_proactor_loop. (2026-07-10: without this, every
    # scrape died at AsyncWebCrawler start the moment the format-lockstep import existed.)
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    if args.debug_markdown:
        asyncio.run(_debug_markdown(args.debug_markdown))
    else:
        asyncio.run(main(args.limit, args.concurrency, args.resume, args.min_usage,
                         refresh_roster=args.refresh_roster))


if __name__ == "__main__":
    cli()
