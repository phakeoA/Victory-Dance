"""
scrape_replays.py
=================
Downloads Pokémon Showdown ladder replays for ANY format, sorted by rating,
into  data/vods/Type_B/<format_slug>/  as static replay .html files that the
Victory-Dance vod_parser can ingest unchanged.

It is the replay-harvesting companion to scrape_pikalytics.py.  Where that
script scrapes usage stats with a headless browser (Crawl4AI), this one only
needs Showdown's plain JSON endpoints, so it uses `requests` directly:

  * Search index :  https://replay.pokemonshowdown.com/search.json?format=<slug>&page=<n>&sort=rating
                    → 51 entries/page, rating-descending (the 51st row is a
                      cursor that repeats as the next page's first row, so we
                      de-duplicate by replay id).
  * Replay log   :  https://replay.pokemonshowdown.com/<id>.json
                    → {id, format, players, log, uploadtime, rating, ...}

The bare replay URL serves a JS single-page app with NO embedded battle log,
and the historical `.html` export endpoint is gone (404).  So we rebuild the
classic static replay page (inline CSS + `name="replayid"` input + a
`battle-log-data` script) from the `.json` payload.  vod_parser only reads the
`battle-log-data` block and the `replayid`, so the reconstructed file parses
identically to the replays already in the corpus.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT IT GUARANTEES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1.  No duplicates.  On startup it scans every .html already in the target
    folder, pulls each one's replay id (vod_parser.extract_replay_id_from_html),
    and skips any search hit whose id is already present — across re-runs too.
2.  Team variety.  The same scan also reads each existing replay's teampreview
    (vod_parser.extract_log_from_html → the |poke| lines) and seeds a per-team
    counter.  While downloading, a replay is skipped only when BOTH players'
    6-mon teams have already appeared `--max-per-team` times in the corpus, so
    a handful of meta teams can't flood the dataset while rarer teams are still
    welcomed (good for training on unexpected situations).  Replays from
    formats without teampreview bypass this cap automatically.
3.  Politeness.  Random jitter between every request and a short pause between
    search pages; automatic retry/back-off on network errors and HTTP 429.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  # Smoke test — grab 50 replays for the default format
  python scrape_replays.py --target 50

  # Full run — ~1000 replays for the default format
  python scrape_replays.py --target 1000

  # A different format (anything Showdown's replay search accepts)
  python scrape_replays.py --format gen9vgc2024regh --target 500

  # See what WOULD be downloaded without writing anything
  python scrape_replays.py --target 20 --dry-run

  # Loosen / tighten the team-variety cap (0 = no cap)
  python scrape_replays.py --target 1000 --max-per-team 20
"""

from __future__ import annotations

import argparse
import html
import json
import random
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

# ── vod_parser bootstrap (so `import vod_parser` works from anywhere) ─────────
_SCRIPT_DIR = Path(__file__).resolve().parent          # data/scripts/scrapers/
from v_dance.parser.vod_parser.replay_parser import (  # noqa: E402  (after sys.path tweak)
    extract_log_from_html,
    extract_replay_id_from_html,
)

# ── Config / constants ────────────────────────────────────────────────────────
BASE_URL    = "https://replay.pokemonshowdown.com"
SEARCH_URL  = f"{BASE_URL}/search.json"

# data/vods/Type_B  (the per-format slug subfolder is appended at runtime)
# scrapers/ → parents[1]==data/  (data/scripts/scrapers/ -> data/)
VODS_ROOT   = _SCRIPT_DIR.parents[1] / "vods" / "Type_B"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# Request pacing (seconds) — tune via CLI if the server pushes back
DELAY_MIN      = 0.6   # min pause between replay downloads
DELAY_MAX      = 1.4   # max pause between replay downloads
PAGE_DELAY_MIN = 1.0   # min pause between search pages
PAGE_DELAY_MAX = 2.0   # max pause between search pages

MAX_RETRIES = 4        # per-request retry budget
RETRY_BASE  = 2.0      # back-off base (seconds): RETRY_BASE * 2**attempt

# The exact inline stylesheet shipped by Showdown's "version 1" static replay
# export — copied verbatim so reconstructed files look identical in a browser.
_REPLAY_CSS = (
    "\nhtml,body {font-family:Verdana, sans-serif;font-size:10pt;margin:0;padding:0;}"
    "body{padding:12px 0;} .battle-log {font-family:Verdana, sans-serif;font-size:10pt;}"
    " .battle-log-inline {border:1px solid #AAAAAA;background:#EEF2F5;color:black;"
    "max-width:640px;margin:0 auto 80px;padding-bottom:5px;} .battle-log .inner "
    "{padding:4px 8px 0px 8px;} .battle-log .inner-preempt {padding:0 8px 4px 8px;}"
    " .battle-log .inner-after {margin-top:0.5em;} .battle-log h2 {margin:0.5em -8px;"
    "padding:4px 8px;border:1px solid #AAAAAA;background:#E0E7EA;border-left:0;"
    "border-right:0;font-family:Verdana, sans-serif;font-size:13pt;} .battle-log .chat "
    "{vertical-align:middle;padding:3px 0 3px 0;font-size:8pt;} .battle-log .chat strong "
    "{color:#40576A;} .battle-log .chat em {padding:1px 4px 1px 3px;color:#000000;"
    "font-style:normal;} .chat.mine {background:rgba(0,0,0,0.05);margin-left:-8px;"
    "margin-right:-8px;padding-left:8px;padding-right:8px;} .spoiler {color:#BBBBBB;"
    "background:#BBBBBB;padding:0px 3px;} .spoiler:hover, .spoiler:active, .spoiler-shown "
    "{color:#000000;background:#E2E2E2;padding:0px 3px;} .spoiler a {color:#BBBBBB;}"
    " .spoiler:hover a, .spoiler:active a, .spoiler-shown a {color:#2288CC;} .chat code, "
    ".chat .spoiler:hover code, .chat .spoiler:active code, .chat .spoiler-shown code "
    "{border:1px solid #C0C0C0;background:#EEEEEE;color:black;padding:0 2px;} .chat "
    ".spoiler code {border:1px solid #CCCCCC;background:#CCCCCC;color:#CCCCCC;} "
    ".battle-log .rated {padding:3px 4px;} .battle-log .rated strong {color:white;"
    "background:#89A;padding:1px 4px;border-radius:4px;} .spacer {margin-top:0.5em;}"
    " .message-announce {background:#6688AA;color:white;padding:1px 4px 2px;}"
    " .message-announce a, .broadcast-green a, .broadcast-blue a, .broadcast-red a "
    "{color:#DDEEFF;} .broadcast-green {background-color:#559955;color:white;"
    "padding:2px 4px;} .broadcast-blue {background-color:#6688AA;color:white;"
    "padding:2px 4px;} .infobox {border:1px solid #6688AA;padding:2px 4px;}"
    " .infobox-limited {max-height:200px;overflow:auto;overflow-x:hidden;} "
    ".broadcast-red {background-color:#AA5544;color:white;padding:2px 4px;}"
    " .message-learn-canlearn {font-weight:bold;color:#228822;text-decoration:underline;}"
    " .message-learn-cannotlearn {font-weight:bold;color:#CC2222;"
    "text-decoration:underline;} .message-effect-weak {font-weight:bold;color:#CC2222;}"
    " .message-effect-resist {font-weight:bold;color:#6688AA;} .message-effect-immune "
    "{font-weight:bold;color:#666666;} .message-learn-list {margin-top:0;margin-bottom:0;}"
    " .message-throttle-notice, .message-error {color:#992222;} .message-overflow, "
    ".chat small.message-overflow {font-size:0pt;} .message-overflow::before "
    "{font-size:9pt;content:'...';} .subtle {color:#3A4A66;}\n"
)

# Trailing embed loader, verbatim from the static export.
_REPLAY_EMBED_SCRIPT = (
    '<script>\n'
    "let daily = Math.floor(Date.now()/1000/60/60/24);"
    "document.write('<script src=\"https://play.pokemonshowdown.com/js/"
    "replay-embed.js?version'+daily+'\"></'+'script>');\n"
    '</script>'
)

# |poke|p1|Floette-Eternal, L50, F|   → capture player + species (pre-comma)
_POKE_RE = re.compile(r"^\|poke\|(p[12])\|([^,|]+)", re.MULTILINE)


# ── Small helpers ──────────────────────────────────────────────────────────────

def to_id(name: str) -> str:
    """Showdown's toID(): lowercase, strip every non-alphanumeric char."""
    return re.sub(r"[^a-z0-9]+", "", (name or "").lower())


def format_prefix(display_format: str) -> str:
    """Display format name → the alnum prefix used in filenames.

    "[Gen 9 Champions] VGC 2026 Reg M-A" → "Gen9ChampionsVGC2026RegMA"
    """
    return re.sub(r"[^A-Za-z0-9]+", "", display_format or "")


def team_sigs_from_log(log: str) -> tuple[frozenset, frozenset]:
    """Return (p1_team, p2_team) as frozensets of teampreview species.

    Empty frozensets when the format has no teampreview — callers treat those
    as "unknown team" and never apply the variety cap to them.
    """
    teams: dict[str, set] = {"p1": set(), "p2": set()}
    for m in _POKE_RE.finditer(log):
        teams[m.group(1)].add(m.group(2).strip())
    return frozenset(teams["p1"]), frozenset(teams["p2"])


def utc_date(uploadtime: int) -> str:
    """uploadtime (unix secs) → 'YYYY-MM-DD' in UTC (matches Showdown's date)."""
    return datetime.fromtimestamp(uploadtime, tz=timezone.utc).strftime("%Y-%m-%d")


# ── HTML reconstruction ─────────────────────────────────────────────────────────

def build_replay_html(data: dict) -> str:
    """Rebuild the classic static replay page from a replay .json payload.

    Only the `battle-log-data` script and the `replayid` input matter to
    vod_parser; the rest mirrors the real export so the file renders normally.
    """
    replay_id = data["id"]
    disp_fmt  = data.get("format") or ""
    players   = data.get("players") or ["", ""]
    p1_name   = players[0] if len(players) > 0 else ""
    p2_name   = players[1] if len(players) > 1 else ""
    log       = data.get("log") or ""

    # Escape every '/' as '\/' — matches the original export and makes a stray
    # "</script>" in chat impossible.  extract_log_from_html reverses this.
    log_escaped = log.replace("/", "\\/")

    disp_e = html.escape(disp_fmt)
    p1_e   = html.escape(p1_name)
    p2_e   = html.escape(p2_name)

    return (
        "<!DOCTYPE html>\n"
        '<meta charset="utf-8" />\n'
        "<!-- version 1 -->\n"
        f"<title>{disp_e} replay: {p1_e} vs. {p2_e}</title>\n"
        f"<style>{_REPLAY_CSS}</style>\n"
        '<div class="wrapper replay-wrapper" style="max-width:1180px;margin:0 auto">\n'
        f'<input type="hidden" name="replayid" value="{html.escape(replay_id)}" />\n'
        '<div class="battle"></div><div class="battle-log"></div>'
        '<div class="replay-controls"></div><div class="replay-controls-2"></div>\n'
        '<h1 style="font-weight:normal;text-align:center"><strong>'
        f'{disp_e}</strong><br />'
        f'<a href="http://pokemonshowdown.com/users/{to_id(p1_name)}" '
        f'class="subtle" target="_blank">{p1_e}</a> vs. '
        f'<a href="http://pokemonshowdown.com/users/{to_id(p2_name)}" '
        f'class="subtle" target="_blank">{p2_e}</a></h1>\n'
        f'<script type="text/plain" class="battle-log-data">{log_escaped}</script>\n'
        "</div>\n"
        f"{_REPLAY_EMBED_SCRIPT}\n"
    )


def replay_filename(data: dict) -> str:
    """`<Prefix>-<YYYY-MM-DD>-<p1id>-<p2id>.html`, matching the existing corpus."""
    players = data.get("players") or ["", ""]
    p1id = to_id(players[0] if len(players) > 0 else "") or "unknown"
    p2id = to_id(players[1] if len(players) > 1 else "") or "unknown"
    prefix = format_prefix(data.get("format") or "")
    date = utc_date(data["uploadtime"]) if data.get("uploadtime") else "unknown"
    return f"{prefix}-{date}-{p1id}-{p2id}.html"


# ── Network (with retry / back-off) ──────────────────────────────────────────────

def _get_json(session: requests.Session, url: str, params: Optional[dict] = None) -> Optional[object]:
    """GET a JSON endpoint with retry + exponential back-off (longer on 429)."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 404:
                return None  # gone / private — caller skips
            if resp.status_code == 429:
                wait = RETRY_BASE * (2 ** attempt) + random.uniform(2, 6)
                print(f"    [429] rate-limited; backing off {wait:.1f}s")
                time.sleep(wait)
                continue
            print(f"    [!] HTTP {resp.status_code} for {url}")
        except (requests.RequestException, json.JSONDecodeError) as e:
            print(f"    [!] {type(e).__name__}: {e}")
        time.sleep(RETRY_BASE * (2 ** attempt) + random.uniform(0, 1))
    return None


def fetch_search_page(session: requests.Session, fmt: str, page: int) -> list[dict]:
    data = _get_json(session, SEARCH_URL, {"format": fmt, "page": page, "sort": "rating"})
    return data if isinstance(data, list) else []


def fetch_replay_json(session: requests.Session, replay_id: str) -> Optional[dict]:
    data = _get_json(session, f"{BASE_URL}/{replay_id}.json")
    return data if isinstance(data, dict) and data.get("log") else None


# ── Existing-corpus scan (dedup ids + seed team counts) ──────────────────────────

def scan_existing(out_dir: Path) -> tuple[set[str], Counter]:
    """Read every .html already in `out_dir`; return (replay ids, team counts).

    Uses vod_parser to stay consistent with how the rest of the pipeline reads
    these files.  Malformed files are skipped with a warning.
    """
    existing_ids: set[str] = set()
    team_counts: Counter = Counter()
    if not out_dir.exists():
        return existing_ids, team_counts

    files = sorted(out_dir.glob("*.html"))
    print(f"[scan] Inspecting {len(files)} existing replay(s) in {out_dir} …")
    for fp in files:
        try:
            raw = fp.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            print(f"    [!] could not read {fp.name}: {e}")
            continue

        rid = extract_replay_id_from_html(raw)
        if rid:
            existing_ids.add(rid)

        try:
            log = extract_log_from_html(raw)
        except ValueError:
            continue  # not a parseable replay file — ignore for team counts
        s1, s2 = team_sigs_from_log(log)
        if s1:
            team_counts[s1] += 1
        if s2:
            team_counts[s2] += 1

    print(f"[scan] {len(existing_ids)} known replay id(s); "
          f"{len(team_counts)} distinct team(s) already represented.")
    return existing_ids, team_counts


# ── Main scrape loop ─────────────────────────────────────────────────────────────

def scrape(
    fmt: str,
    target: int,
    out_dir: Path,
    *,
    start_page: int,
    max_pages: int,
    max_per_team: int,
    dry_run: bool,
    delay_range: tuple[float, float],
    page_delay_range: tuple[float, float],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    existing_ids, team_counts = scan_existing(out_dir)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})

    seen_run: set[str] = set()          # ids already handled this run
    kept = skipped_dup = skipped_div = failed = 0
    empty_pages = 0

    def saturated(sig: frozenset) -> bool:
        return bool(sig) and max_per_team > 0 and team_counts[sig] >= max_per_team

    print(f"\n[scrape] format={fmt}  target={target}  max_per_team="
          f"{max_per_team or 'off'}  dry_run={dry_run}")

    page = start_page
    pages_done = 0
    while kept < target and pages_done < max_pages:
        entries = fetch_search_page(session, fmt, page)
        pages_done += 1
        if not entries:
            empty_pages += 1
            print(f"[page {page}] no results.")
            if empty_pages >= 2:
                print("[scrape] two empty pages in a row — reached the end.")
                break
            page += 1
            continue
        empty_pages = 0

        new_on_page = 0
        print(f"[page {page}] {len(entries)} hits "
              f"(rating {entries[0].get('rating')}–{entries[-1].get('rating')})")

        for entry in entries:
            if kept >= target:
                break
            rid = entry.get("id")
            if not rid or rid in seen_run:
                continue
            seen_run.add(rid)
            if entry.get("private") or entry.get("password"):
                continue
            if rid in existing_ids:
                skipped_dup += 1
                continue
            new_on_page += 1

            # Polite pause before each replay fetch.
            time.sleep(random.uniform(*delay_range))

            data = fetch_replay_json(session, rid)
            if data is None:
                failed += 1
                print(f"    [!] {rid}: log unavailable — skipped")
                continue

            s1, s2 = team_sigs_from_log(data.get("log") or "")
            if saturated(s1) and saturated(s2):
                skipped_div += 1
                continue  # both teams already well-represented

            fname = replay_filename(data)
            path = out_dir / fname
            # Distinct replay that collides on players+date → disambiguate.
            if path.exists():
                path = out_dir / f"{fname[:-5]}-{rid.rsplit('-', 1)[-1]}.html"

            if not dry_run:
                path.write_text(build_replay_html(data), encoding="utf-8")

            existing_ids.add(rid)
            if s1:
                team_counts[s1] += 1
            if s2:
                team_counts[s2] += 1
            kept += 1

            rate = entry.get("rating")
            tag = "(dry) " if dry_run else ""
            print(f"    ✓ {tag}[{kept}/{target}] {rate} {path.name}")

        if new_on_page == 0:
            empty_pages += 1
            if empty_pages >= 2:
                print("[scrape] no new replays across recent pages — stopping.")
                break

        if kept < target and pages_done < max_pages:
            time.sleep(random.uniform(*page_delay_range))
        page += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(f"[done] format={fmt}")
    print(f"   downloaded : {kept}{' (dry-run, nothing written)' if dry_run else ''}")
    print(f"   skipped (already had)  : {skipped_dup}")
    print(f"   skipped (team variety) : {skipped_div}")
    print(f"   failed fetches         : {failed}")
    print(f"   pages fetched          : {pages_done}")
    print(f"   distinct teams now     : {len(team_counts)}")
    if team_counts:
        top = team_counts.most_common(1)[0]
        print(f"   most common team count : {top[1]}")
    print(f"   output dir : {out_dir}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def cli() -> None:
    # Windows consoles often default to cp1252, which can't print ✓/–
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    ap = argparse.ArgumentParser(
        description="Download rating-sorted Showdown replays into data/vods/Type_B/<format>/.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--format", default="gen9championsvgc2026regma",
                    help="Showdown format slug (as it appears in the replay-search URL)")
    ap.add_argument("--target", type=int, default=3000,
                    help="Number of NEW (not-yet-downloaded) replays to fetch")
    ap.add_argument("--max-per-team", type=int, default=12,
                    help="Skip a replay only if BOTH teams already appear this many "
                         "times in the corpus (0 = no variety cap)")
    ap.add_argument("--out-dir", default=None,
                    help="Override output root (default: <repo>/data/vods/Type_B). "
                         "The format slug is always appended as a subfolder.")
    ap.add_argument("--start-page", type=int, default=1, help="First search page to fetch")
    ap.add_argument("--max-pages", type=int, default=200, help="Safety cap on pages fetched")
    ap.add_argument("--delay-min", type=float, default=DELAY_MIN,
                    help="Min seconds between replay downloads")
    ap.add_argument("--delay-max", type=float, default=DELAY_MAX,
                    help="Max seconds between replay downloads")
    ap.add_argument("--page-delay-min", type=float, default=PAGE_DELAY_MIN,
                    help="Min seconds between search pages")
    ap.add_argument("--page-delay-max", type=float, default=PAGE_DELAY_MAX,
                    help="Max seconds between search pages")
    ap.add_argument("--dry-run", action="store_true",
                    help="List what would be downloaded without writing files")
    args = ap.parse_args()

    fmt = args.format.strip().lower()
    out_root = Path(args.out_dir).resolve() if args.out_dir else VODS_ROOT
    out_dir = out_root / fmt

    scrape(
        fmt, args.target, out_dir,
        start_page=args.start_page,
        max_pages=args.max_pages,
        max_per_team=args.max_per_team,
        dry_run=args.dry_run,
        delay_range=(args.delay_min, args.delay_max),
        page_delay_range=(args.page_delay_min, args.page_delay_max),
    )


if __name__ == "__main__":
    cli()
