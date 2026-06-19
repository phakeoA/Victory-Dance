"""
scrape_serebii_items.py
=======================
Scrapes Serebii for [Pokémon Champions] items and produces a single JSON file.
Uses Crawl4AI's markdown engine to safely track categories and filter out 
the "Miscellaneous" items.

Uses Crawl4AI (v0.8.x) AsyncWebCrawler.
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

# ── Config ─────────────────────────────────────────────────────────────────────
TARGET_URL   = "https://www.serebii.net/pokemonchampions/items.shtml"

_SCRIPT_DIR = Path(__file__).resolve().parent          # data/scripts/scrapers/
OUTPUT_DIR  = _SCRIPT_DIR.parents[2] / "data"          # <project_root>/data/
# Per-reg output (Serebii's page is single + tracks the CURRENT reg). The active
# format's reg by default; --reg overrides in __main__.
try:
    from v_dance.formats import default_format as _default_format, reg_token as _reg_token
    _DEFAULT_REG = _reg_token(_default_format()) or "regmb"
except Exception:
    _DEFAULT_REG = "regmb"
OUTPUT_FILE = OUTPUT_DIR / f"serebii_champions_items_{_DEFAULT_REG}.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# ── Crawl4AI Configs ──────────────────────────────────────────────────────────

def _make_browser_config() -> BrowserConfig:
    return BrowserConfig(
        headless=True,
        user_agent=USER_AGENT,
        enable_stealth=True,
        verbose=False,
    )

def _make_run_config() -> CrawlerRunConfig:
    return CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        magic=True,
        wait_until="load",
        page_timeout=30_000,
        max_retries=2,
        # We drop the JsonCssExtractionStrategy and parse the clean markdown instead
    )

# ── Markdown Parsing Logic ────────────────────────────────────────────────────

def parse_markdown_items(markdown_text: str) -> list[dict]:
    results = []
    current_category = "Uncategorized"
    
    # Simple regex to identify when we enter a miscellaneous section
    misc_pattern = re.compile(r"misc", re.IGNORECASE)

    lines = markdown_text.splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 1. Capture Category Headers (e.g., "## Miscellaneous Items")
        if line.startswith("#"):
            current_category = line.lstrip("# ").strip()
            continue

        # 2. Process Markdown Table Rows (e.g., "| [img] | Potion | Restores HP |")
        if line.startswith("|") and line.endswith("|"):
            # Split by pipe and strip whitespace from columns
            columns = [col.strip() for col in line.split("|")[1:-1]]
            
            # Skip markdown table dividers (e.g., "| --- | --- |")
            if any("---" in col for col in columns):
                continue
                
            # Ensure we have enough columns to work with
            if len(columns) < 2:
                continue

            # Identify columns based on Serebii's layout:
            # col[0] = Picture/Icon info, col[1] = Item Name, col[2] = Effect
            name_text = columns[1]
            effect_text = columns[2] if len(columns) > 2 else ""

            # Skip table subheaders ("Name", "Picture", "Effect")
            if name_text.lower() in ("name", "picture", "item") or effect_text.lower() == "effect":
                continue

            # 3. Filter checks
            # Filter A: Skip if the tracked markdown category is Miscellaneous
            if misc_pattern.search(current_category):
                continue

            # Filter B: Hard safety check on item characteristics (Affinity Tickets / Coupons)
            if "ticket" in name_text.lower() or "coupon" in name_text.lower():
                continue

            # 4. Save valid items
            if name_text:
                results.append({
                    "category": current_category,
                    "name": name_text,
                    "effect": effect_text
                })

    return results

# ── Main ───────────────────────────────────────────────────────────────────────

async def main(reg: str = "regmb") -> None:
    print(f"[scrape] Fetching Serebii [{reg}] items via Markdown from {TARGET_URL}...")
    
    browser_conf = _make_browser_config()
    run_conf = _make_run_config()

    async with AsyncWebCrawler(config=browser_conf) as crawler:
        result = await crawler.arun(TARGET_URL, config=run_conf)

        if not result.success:
            print(f"[!] Fetch failed: {result.error_message}")
            return

        # Use the built-in clean markdown output from Crawl4AI
        markdown_content = result.markdown or ""
        valid_items = parse_markdown_items(markdown_content)
        
        output_payload = {
            "game": "pokemonchampions",
            "reg": reg,                       # Serebii's page is single + shows the CURRENT reg; tag explicitly
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "items": valid_items
        }

        with OUTPUT_FILE.open("w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2, ensure_ascii=False)
            
        size_kb = OUTPUT_FILE.stat().st_size / 1024
        print(f"[done] Successfully saved {len(valid_items)} non-miscellaneous items.")
        print(f"       Saved to → {OUTPUT_FILE} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Scrape Serebii [Pokémon Champions] items for a regulation. NOTE: "
                    "Serebii's page is a SINGLE page that shows the CURRENT reg (no per-reg "
                    "URL), so --reg only sets the output name + tag — run it while the page "
                    "is showing the reg you want. Output: serebii_champions_items_<reg>.json."
    )
    ap.add_argument("--format", "--reg", default=None, dest="reg",
                    help="Reg/format to tag the scrape (e.g. gen9championsvgc2026regmb or "
                         "regmb). Default: the active format.")
    args = ap.parse_args()

    reg = args.reg
    if reg is None:
        try:
            from v_dance.formats import default_format, reg_token
            reg = reg_token(default_format()) or "regmb"
        except Exception:
            reg = "regmb"
    else:
        _m = re.search(r"(reg[a-z0-9]+)", reg)
        reg = _m.group(1) if _m else reg

    OUTPUT_FILE = OUTPUT_DIR / f"serebii_champions_items_{reg}.json"
    print(f"[config] reg={reg}  output={OUTPUT_FILE.name}")
    asyncio.run(main(reg))