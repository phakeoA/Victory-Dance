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
  "format":     "gen9championsvgc2026regma",
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
1.  Crawl4AI fetches the index page and extracts all Pokémon slugs via
    JsonCssExtractionStrategy (zero LLM calls).
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai import JsonCssExtractionStrategy

# ── Config ─────────────────────────────────────────────────────────────────────
FORMAT_SLUG  = "gen9championsvgc2026regma"
BASE_URL     = "https://www.pikalytics.com"
INDEX_URL    = f"{BASE_URL}/pokedex/{FORMAT_SLUG}"
OUTPUT_FILE  = Path("pikalytics_regma.json")

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
    """Extract (name, pct) pairs from a named section."""
    header_re = re.compile(
        r"#{1,3}\s+" + re.escape(header) + r".*?\n(.*?)(?=#{1,3}\s|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    m = header_re.search(markdown)
    if not m:
        return []
    section_text = m.group(1)
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
      [Sneasler Sneasler fightingpoison 52.142%](url)
    """
    teammates: list[dict] = []
    seen: set[str] = set()
    section_m = re.search(
        r"Best Teammates.*?\n(.*?)(?=##|\Z)",
        markdown, re.DOTALL | re.IGNORECASE,
    )
    if not section_m:
        return []
    for match in re.finditer(r"\[(\S+)\s+\S+\s+\S+\s+(\d+\.\d+)%\]", section_m.group(1)):
        name = match.group(1)
        pct  = float(match.group(2))
        if name not in seen:
            seen.add(name)
            teammates.append({"name": name, "pct": pct})
    return teammates


def _parse_mon_page(markdown: str) -> dict:
    return {
        "usage_pct": _parse_usage(markdown),
        "moves":     _parse_section(markdown, "Best Moves"),
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
        magic       = True,
        wait_until  = "load",
        page_timeout= 30_000,   # ms
        max_retries = 2,
    )
    if use_extraction:
        kwargs["extraction_strategy"] = JsonCssExtractionStrategy(INDEX_SCHEMA)
    return CrawlerRunConfig(**kwargs)


# ── Index page ─────────────────────────────────────────────────────────────────

async def get_pokemon_list(crawler: AsyncWebCrawler) -> list[str]:
    """Fetch the index and return all Pokémon name slugs in order."""
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
            slug = m.group(1)
            if slug not in seen:
                seen.add(slug)
                names.append(slug)

    print(f"[index] Found {len(names)} Pokémon.")
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
    url = f"{INDEX_URL}/{name}"

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

async def main(limit: int, concurrency: int, resume: bool) -> None:
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
        # ── Step 1: index ─────────────────────────────────────────────────────
        names = await get_pokemon_list(crawler)

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


# ── CLI ────────────────────────────────────────────────────────────────────────

def cli() -> None:
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
    args = parser.parse_args()
    asyncio.run(main(args.limit, args.concurrency, args.resume))


if __name__ == "__main__":
    cli()
