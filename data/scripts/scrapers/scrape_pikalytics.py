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
FORMAT_SLUG  = "gen9championsvgc2026regmb"
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
    next_heading = re.search(r"^#{1,4}\s", markdown[start:], re.MULTILINE)
    end = start + next_heading.start() if next_heading else len(markdown)
    section_text = markdown[start:end]

    results: list[dict] = []
    seen: set[str] = set()
    for match in _PCT_PATTERN.finditer(section_text):
        name = match.group(1).strip()
        pct  = float(match.group(2))
        if name.lower() in _TYPE_NOISE:
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
    next_heading = re.search(r"^#{1,4}\s", markdown[start:], re.MULTILINE)
    end = start + next_heading.start() if next_heading else len(markdown)
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


def _parse_spreads(markdown: str) -> list[dict]:
    spreads: list[dict] = []
    for m in _SPREAD_PATTERN.finditer(markdown):
        spreads.append({
            "nature": m.group(1).capitalize(),
            "evs":    [int(m.group(i)) for i in range(2, 8)],  # HP Atk Def SpA SpD Spe
            "pct":    float(m.group(8)),
        })
    return spreads


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
    next_heading = re.search(r"^#{1,4}\s", markdown[start:], re.MULTILINE)
    end = start + next_heading.start() if next_heading else len(markdown)
    section_text = markdown[start:end]

    for match in re.finditer(
        r"!\[([^\]]+)\]\([^)]+\)[^\[\]]*?(\d+\.\d+)%", section_text
    ):
        name = match.group(1).strip()
        pct  = float(match.group(2))
        if name not in seen:
            seen.add(name)
            teammates.append({"name": name, "pct": pct})
    return teammates


def _parse_mon_page(markdown: str) -> dict:
    return {
        "usage_pct": _parse_usage(markdown),
        "moves":     _parse_moves(markdown),
        "items":     _parse_section(markdown, "Best Items"),
        "abilities": _parse_section(markdown, "Best Abilities"),
        "spreads":   _parse_spreads(markdown),
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
        wait_until  = "load",
        page_timeout= 30_000,   # ms
        max_retries = 2,
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


async def get_pokemon_list(crawler: AsyncWebCrawler, min_usage: float = 0.0) -> list[str]:
    """
    Return the full list of Pokémon names to scrape.

    Primary source: Smogon usage stats (complete roster).
    Merged with: links extracted from the Pikalytics index page, which only
    server-renders the top ~50 mons but costs us nothing to include.
    """
    print(f"[index] Fetching {INDEX_URL}")
    result = await crawler.arun(INDEX_URL, config=_make_run_config(use_extraction=True))

    if not result.success:
        raise RuntimeError(f"Index page fetch failed: {result.error_message}")

    raw = json.loads(result.extracted_content or "[]")
    slug_re = re.compile(rf"/pokedex/{FORMAT_SLUG}/([^/?#]+)")
    seen: set[str] = set()
    names: list[str] = []
    for item in raw:
        href = item.get("href", "")
        m = slug_re.search(href)
        if m:
            slug = urllib.parse.unquote(m.group(1))
            if slug not in seen:
                seen.add(slug)
                names.append(slug)

    print(f"[index] Found {len(names)} Pokémon on the index page.")

    roster = fetch_smogon_roster()
    if min_usage > 0:
        skipped = [n for n, u in roster if u < min_usage]
        roster  = [(n, u) for n, u in roster if u >= min_usage]
        if skipped:
            print(f"[roster] Skipping {len(skipped)} mons below {min_usage}% usage.")

    for name, _usage in roster:
        if name not in seen:
            seen.add(name)
            names.append(name)

    print(f"[index] {len(names)} Pokémon total after merging Smogon roster.")
    return names


# ── Individual page fetching (rate-limited) ────────────────────────────────────

async def _fetch_one(
    crawler: AsyncWebCrawler,
    name: str,
    sem: asyncio.Semaphore,
) -> tuple[str, dict]:
    """
    Fetch a single mon page, honouring the semaphore and adding jitter.
    Returns (name, parsed_data).
    """
    # Names like "Mr. Rime" need the space percent-encoded in the URL
    url = f"{INDEX_URL}/{urllib.parse.quote(name)}"

    async with sem:
        # Random jitter before the request — looks human, avoids burst detection
        await asyncio.sleep(random.uniform(JITTER_MIN, JITTER_MAX))

        result = await crawler.arun(url, config=_make_run_config())

    if not result.success:
        print(f"    [!] {name}: fetch failed — {result.error_message}")
        return name, {"error": result.error_message}

    markdown = result.markdown.raw_markdown if result.markdown else ""
    data = _parse_mon_page(markdown)

    moves_n   = len(data["moves"])
    spreads_n = len(data["spreads"])
    print(f"    ✓ {name}: {moves_n} moves, {spreads_n} spreads")
    return name, data


async def scrape_all(
    crawler: AsyncWebCrawler,
    names: list[str],
    concurrency: int,
) -> dict[str, dict]:
    """
    Fetch all mon pages with bounded concurrency.
    Tasks are created upfront but the semaphore keeps at most
    `concurrency` requests in-flight at any moment.
    """
    sem = asyncio.Semaphore(concurrency)
    tasks = [_fetch_one(crawler, name, sem) for name in names]
    pairs = await asyncio.gather(*tasks)
    return dict(pairs)


# ── Main ───────────────────────────────────────────────────────────────────────

async def main(limit: int, concurrency: int, resume: bool, min_usage: float) -> None:
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
        names = await get_pokemon_list(crawler, min_usage)

        if limit:
            names = names[:limit]
            print(f"[index] Limited to first {limit}.")

        to_scrape = [n for n in names if n not in already]
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

            batch_data = await scrape_all(crawler, batch, concurrency)
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
        description="Scrape Pikalytics Reg M-A data using Crawl4AI (v0.8.x).",
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
    args = parser.parse_args()

    if args.debug_markdown:
        asyncio.run(_debug_markdown(args.debug_markdown))
    else:
        asyncio.run(main(args.limit, args.concurrency, args.resume, args.min_usage))


if __name__ == "__main__":
    cli()
