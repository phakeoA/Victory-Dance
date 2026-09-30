"""The ladder's OWN numbers — ``https://pokemonshowdown.com/users/<userid>.json`` — the ground truth the bot's chained
rating is checked against (USER 2026-09-29: "the site elo is the absolute truth").

The bot's rating is chained from each battle's rating-exchange line, so every battle whose exchange it never sees (a
reconnect gap, a room decided while the link was down, the 09-29 frozen consumer) silently drops out. On 09-29 the
site said M-C 478 W / 469 L while the bench had logged 477 / 448 — 22 games missing, 21 of them LOSSES, which flatters
the logged win rate and the bandit's per-arm rewards. The online bot polls this endpoint, appends a ``site_rating``
row to ``human_bench.jsonl`` whenever the numbers change, and logs the drift from the panel's chained rating.
"""
from __future__ import annotations

import json
import re
import urllib.request
from typing import Optional

USER_URL = "https://pokemonshowdown.com/users/{uid}.json"


def userid(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def fetch(username: str, fmt: str, timeout: float = 10.0) -> Optional[dict]:
    """{elo, gxe, w, l} for ``fmt`` on the official ladder, or None when the account has no rating there yet."""
    req = urllib.request.Request(USER_URL.format(uid=userid(username)),
                                 headers={"User-Agent": "VictoryDance-bot (rating sync)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    rt = (data.get("ratings") or {}).get(fmt)
    if not rt:
        return None
    return {"elo": round(float(rt["elo"]), 1), "gxe": rt.get("gxe"), "w": int(rt.get("w") or 0),
            "l": int(rt.get("l") or 0)}


def drift_line(site: dict, panel_rating: Optional[float], persistent: bool = True) -> str:
    """``persistent`` = the gap was also > 5 at the previous poll. With 5 battles at once the panel and the site are
    often a game apart for a moment (09-30: 79 of 266 polls > 5 Elo, median gap 1.1), so only a gap that SURVIVES
    a poll is called a desync."""
    gap = "" if panel_rating is None else f" · panel {panel_rating:.0f} (Δ {site['elo'] - panel_rating:+.0f})"
    flag = "  ⚠ DESYNC — the panel missed games" if panel_rating is not None and abs(site["elo"] - panel_rating) > 5 and persistent else ""
    return f"[online] SITE (truth): elo {site['elo']:.0f} · {site['w']}W-{site['l']}L · GXE {site['gxe']}{gap}{flag}"
