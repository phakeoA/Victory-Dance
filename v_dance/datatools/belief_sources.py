"""Merge the per-source stat files into the ONE belief file the stack loads: ``data/belief_<reg>.json``.

B5 (USER GO 2026-09-29): MunchStats is the MAIN source during a regulation (live in-game data,
spreads included); Pikalytics fills what MunchStats lacks (usage %, species it has no page for);
the previous reg's Pikalytics file is the last resort for spreads / natures / teammates, which
neither live source may have early in a reg. Per species and PER FIELD the first non-empty source
wins, and ``sources`` on every species records which one did, so a later re-scrape + re-merge
replaces borrowed data. ``formats.pikalytics_path_for`` prefers ``belief_<reg>.json`` over the raw
``pikalytics_<reg>.json``, so re-running a scraper never clobbers the merge — re-run this after
any scrape.

Move % scale: MunchStats and M-B-era Pikalytics give PER-SET % (Grassy Glide 97.4 = 97.4 % of
Rillaboom sets), but Pikalytics' M-C pages give a PER-SLOT share (the list sums to ~100). A
per-slot list is rescaled x4 (four move slots, capped at 100) and tagged ``moves_scale``.

Usage::

    python -m v_dance.datatools.belief_sources                    # active reg, previous reg = regmb
    python -m v_dance.datatools.belief_sources --reg regmc --prev regmb --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_DATA = Path(__file__).resolve().parents[2] / "data"
FIELDS = ("usage_pct", "moves", "items", "abilities", "spreads", "natures", "teammates")
PREV_ONLY = ("spreads", "natures", "teammates")      # the only fields borrowed from an older reg


def _load(path: Path) -> Optional[dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _per_set_moves(moves: list[dict]) -> tuple[list[dict], Optional[str]]:
    total = sum(m.get("pct") or 0 for m in moves)
    if moves and total <= 105:                       # a per-slot share: sums to ~100 across moves
        return [{**m, "pct": min(100.0, round(4 * (m.get("pct") or 0), 3))} for m in moves], "per-slot x4"
    return moves, None


def merge(munch: Optional[dict], pika: Optional[dict], prev: Optional[dict], prev_reg: str) -> dict:
    """Return {name: species_entry} merged field by field (first non-empty source wins)."""
    tiers = [("munchstats", (munch or {}).get("pokemon", {})),
             ("pikalytics", (pika or {}).get("pokemon", {})),
             (f"pikalytics_{prev_reg}", (prev or {}).get("pokemon", {}))]
    names = list(dict.fromkeys([*tiers[0][1], *tiers[1][1]]))   # prev reg never ADDS species
    out = {}
    for name in names:
        entry, sources = {}, {}
        for field in FIELDS:
            for label, mons in tiers:
                if label.startswith("pikalytics_") and field not in PREV_ONLY:
                    continue
                val = (mons.get(name) or {}).get(field)
                if val not in (None, [], {}):
                    if field == "moves" and label != "munchstats":
                        val, scale = _per_set_moves(val)
                        if scale:
                            entry["moves_scale"] = scale
                    entry[field], sources[field] = val, label
                    break
            entry.setdefault(field, None if field == "usage_pct" else [])
        rank = (tiers[0][1].get(name) or {}).get("usage_rank")
        if rank is not None:
            entry["usage_rank"] = rank
        entry["sources"] = sources
        out[name] = entry
    return out


def main() -> None:
    from v_dance.formats import default_format, reg_token

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reg", default=None, help="reg token, e.g. regmc (default: the active format's)")
    ap.add_argument("--prev", default="regmb", help="previous reg token for the last-resort tier")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    reg = a.reg or reg_token(default_format())
    paths = {"munchstats": _DATA / f"munchstats_{reg}.json", "pikalytics": _DATA / f"pikalytics_{reg}.json",
             "prev": _DATA / f"pikalytics_{a.prev}.json"}
    munch, pika, prev = (_load(p) for p in paths.values())
    if not munch and not pika:
        raise SystemExit(f"no source file for {reg}: run scrape_munchstats and/or scrape_pikalytics first")
    mons = merge(munch, pika, prev, a.prev)

    counts: dict[str, dict[str, int]] = {}
    for e in mons.values():
        for field, label in e["sources"].items():
            counts.setdefault(field, {}).setdefault(label, 0)
            counts[field][label] += 1
    live = [e for e in mons.values() if e["moves"]]
    print(f"belief_{reg}: {len(mons)} species ({len(live)} with moves); "
          f"no spreads: {sum(1 for e in live if not e['spreads'])}")
    for field in FIELDS:
        print(f"  {field:10} {counts.get(field, {})}")
    if a.dry_run:
        return
    doc = {"format": (munch or pika).get("format"), "built_at": datetime.now(timezone.utc).isoformat(),
           "inputs": {k: {"file": p.name, "scraped_at": (d or {}).get("scraped_at"),
                          "game_data_last_scraped": (d or {}).get("game_data_last_scraped")}
                      for (k, p), d in zip(paths.items(), (munch, pika, prev)) if d},
           "pokemon": mons}
    out = _DATA / f"belief_{reg}.json"
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
