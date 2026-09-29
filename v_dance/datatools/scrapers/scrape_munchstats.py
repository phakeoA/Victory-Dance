"""Scrape MunchStats' live Champions in-game doubles stats into the BeliefState schema.

B5 (USER GO 2026-09-29): the MAIN belief source during a regulation. Pikalytics publishes no M-C
spreads / natures / teammates, and Smogon's monthly files only appear long after a regulation
starts (M-B's June + July files were published 2026-08-23), so both are unusable mid-reg.
MunchStats (https://www.munchstats.com/champions/doubles/<Name>) serves the official game's
ranked-doubles data, refreshed during the reg, as server-rendered HTML with per species:
Moves % · Teammates #rank · Items % · Abilities % · Stat Point Spreads (32-scale) % · Natures %.
Megas are NOT separate pages: the stone sits in the base species' Items (Salamence -> Salamencite),
which is the convention BeliefState already expects. There is no usage %, only a usage rank, so
``usage_pct`` stays None and ``datatools.belief_sources`` takes it from Pikalytics.

robots.txt: Crawl-delay 10 and no ``?month=`` URLs (historical months are Smogon re-downloads),
so this fetches only the current pages, one every >= 10 s (~45 min for a full reg).

Output ``data/munchstats_<reg>.json`` (same per-species schema as ``pikalytics_<reg>.json``),
saved every 10 species and resumable: re-running skips species already present unless --refresh.

Usage::

    python -m v_dance.datatools.scrapers.scrape_munchstats                    # active reg, resume
    python -m v_dance.datatools.scrapers.scrape_munchstats --only Rillaboom Salamence
    python -m v_dance.datatools.scrapers.scrape_munchstats --refresh          # re-fetch everything
"""
from __future__ import annotations

import argparse
import html
import json
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import requests

_REPO = Path(__file__).resolve().parents[3]
_DATA = _REPO / "data"
BASE_URL = "https://www.munchstats.com/champions/doubles/"
USER_AGENT = "Mozilla/5.0 (compatible; VictoryDance-research/1.0; honours robots.txt crawl-delay)"
CRAWL_DELAY = 10.0

# Section headers in page order; a section runs until the next header or the first END marker.
_SECTIONS = {"Moves": "moves", "Teammates": "teammates", "Items": "items", "Abilities": "abilities",
             "Stat Point Spreads": "spreads", "Natures": "natures"}
_END = ("Stats Graph (Beta)", "Usage", "Top Teams", "Export Pokemon")
_PCT = re.compile(r"^(\d+(?:\.\d+)?)%$")
_RANK = re.compile(r"^#(\d+)$")
_SPREAD = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{1,2})/(\d{1,2})/(\d{1,2})/(\d{1,2})$")
_STAMP = re.compile(r"Game data last scraped:\s*([^<\n]+?)\s*<")


def _text_lines(page: str) -> list[str]:
    body = re.sub(r"(?s)<script.*?</script>|<style.*?</style>", "", page)
    txt = html.unescape(re.sub(r"<[^>]+>", "\n", body))
    return [ln.strip() for ln in txt.splitlines() if ln.strip()]


def parse_page(page: str) -> Optional[dict]:
    """One species page -> BeliefState-schema dict, or None when the page has no stats."""
    lines = _text_lines(page)
    try:
        start = lines.index("Base Stats")          # everything before it is nav + the sidebar index
    except ValueError:
        return None
    rank = None
    for i, ln in enumerate(lines[:start]):
        if ln == "Rank" and i + 1 < len(lines) and _RANK.match(lines[i + 1]):
            rank = int(_RANK.match(lines[i + 1]).group(1))
            break

    raw: dict[str, list[tuple[str, str]]] = {}
    cur = None
    body = lines[start:]
    i = 0
    while i < len(body):
        ln = body[i]
        if ln in _SECTIONS:
            cur = _SECTIONS[ln]
            raw[cur] = []
            i += 1
            continue
        if ln in _END and cur is not None:
            break
        if cur is not None and i + 1 < len(body) and (_PCT.match(body[i + 1]) or _RANK.match(body[i + 1])):
            raw[cur].append((ln, body[i + 1]))
            i += 2
            continue
        i += 1
    if not raw.get("moves"):
        return None

    def pct_list(key: str) -> list[dict]:
        return [{"name": n, "pct": float(_PCT.match(v).group(1))} for n, v in raw.get(key, []) if _PCT.match(v)]

    natures = [{"nature": n, "pct": float(_PCT.match(v).group(1))}
               for n, v in raw.get("natures", []) if _PCT.match(v)]
    # Spreads and natures are listed separately (like Pikalytics' 2026 layout): attach the MODAL
    # nature to every spread, the same rule scrape_pikalytics._parse_spreads applies.
    modal = max(natures, key=lambda x: x["pct"])["nature"] if natures else None
    spreads = []
    for s, v in raw.get("spreads", []):
        m, p = _SPREAD.match(s), _PCT.match(v)
        if m and p:
            spreads.append({"nature": modal, "evs": [int(x) for x in m.groups()], "pct": float(p.group(1))})
    teammates = [{"name": n, "rank": int(_RANK.match(v).group(1))}
                 for n, v in raw.get("teammates", []) if _RANK.match(v)]
    return {"usage_pct": None, "usage_rank": rank, "moves": pct_list("moves"), "items": pct_list("items"),
            "abilities": pct_list("abilities"), "spreads": spreads, "natures": natures, "teammates": teammates}


def species_to_fetch(reg: str) -> list[str]:
    """Display names from the reg's Showdown-enumerated roster, minus Mega/Primal formes (no own page)."""
    roster = json.loads((_DATA / f"champions_roster_{reg}.json").read_text(encoding="utf-8"))
    return [n for n in roster.values() if not re.search(r"-(Mega|Primal)\b", n)]


def _save(out_path: Path, doc: dict) -> None:
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, out_path)


def main() -> None:
    from v_dance.formats import default_format, reg_token

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--format", default=None, help="format id (default: the active format)")
    ap.add_argument("--only", nargs="*", help="fetch just these display names")
    ap.add_argument("--limit", type=int, default=0, help="stop after N fetches (0 = all)")
    ap.add_argument("--refresh", action="store_true", help="re-fetch species already in the output")
    ap.add_argument("--delay", type=float, default=CRAWL_DELAY, help="seconds between requests (>= 10)")
    a = ap.parse_args()

    fmt = a.format or default_format()
    reg = reg_token(fmt)
    out_path = _DATA / f"munchstats_{reg}.json"
    doc = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {"pokemon": {}, "missing": []}
    doc.update(format=fmt, source=BASE_URL)
    names = a.only or species_to_fetch(reg)
    todo = [n for n in names if a.refresh or a.only or (n not in doc["pokemon"] and n not in doc["missing"])]
    delay = max(a.delay, CRAWL_DELAY)
    print(f"[munch] {fmt}: {len(todo)} to fetch ({len(names) - len(todo)} already done), {delay:.0f}s apart "
          f"-> ~{len(todo) * delay / 60:.0f} min")

    sess = requests.Session()
    sess.headers["User-Agent"] = USER_AGENT
    for k, name in enumerate(todo[: a.limit or None], 1):
        url = BASE_URL + quote(name)
        try:
            r = sess.get(url, timeout=30)
            page = r.text if r.status_code == 200 else ""
        except requests.RequestException as e:
            print(f"  [{k}/{len(todo)}] {name}: request failed ({e}); will retry next run")
            page = None
        if page is not None:
            m = _STAMP.search(page)
            if m:
                doc["game_data_last_scraped"] = m.group(1)
            parsed = parse_page(page) if page else None
            if parsed:
                doc["pokemon"][name] = parsed
                if name in doc["missing"]:
                    doc["missing"].remove(name)
                print(f"  [{k}/{len(todo)}] {name}: {len(parsed['moves'])} moves, {len(parsed['spreads'])} spreads")
            else:
                if name not in doc["missing"]:
                    doc["missing"].append(name)
                print(f"  [{k}/{len(todo)}] {name}: no stats (HTTP {r.status_code})")
        if k % 10 == 0:
            doc["scraped_at"] = datetime.now(timezone.utc).isoformat()
            _save(out_path, doc)
        time.sleep(delay + random.uniform(0, 1.5))
    doc["scraped_at"] = datetime.now(timezone.utc).isoformat()
    _save(out_path, doc)
    print(f"[munch] {len(doc['pokemon'])} species with stats, {len(doc['missing'])} without -> {out_path}")


if __name__ == "__main__":
    main()
