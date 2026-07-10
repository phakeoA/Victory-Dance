"""Observed-meta aggregator — the opponents the bot ACTUALLY faces, as a Pikalytics-format file.

USER request 2026-07-10: aggregate the per-opponent dossiers (``artifacts/dossiers/*.json`` —
revealed moves / items / abilities per species, written passively by every benchmark/online/
ladder game) into ONE usage file in the SAME schema as ``data/pikalytics_<reg>.json``, so it is
drop-in loadable by ``BeliefState`` and blendable with the scraped prior once volume justifies.

What the dossiers can and cannot support:
  * usage_pct / moves / items / abilities / teammates — computed here. Percentages are
    RECORD-level approximations: one record = one (opponent, species) dossier entry; a move's
    pct = share of that species' records revealing it (reveals are partial — real usage is ≥).
  * spreads / natures — NOT observable from dossiers (that needs replay stat-mining) → emitted
    empty; BeliefState's empty-entry fallbacks handle absent sections.

CLI (repo root):
  python -m v_dance.datatools.observed_meta                # writes data/observed_meta_<reg>.json
  python -m v_dance.datatools.observed_meta --min-records 2 --top 20
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

_REPO = Path(__file__).resolve().parents[2]
DOSSIER_DIR = _REPO / "artifacts" / "dossiers"


def _display_names(reg: str) -> Dict[str, str]:
    """{slug: Display Name} from the scraper's roster cache; {} when absent (keys then stay
    slugs — BeliefState lowercases keys at load, so lookups still resolve)."""
    p = _REPO / "data" / f"champions_roster_{reg}.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def aggregate(dossier_dir: Path = DOSSIER_DIR, *, fmt: str = "", min_records: int = 1,
              names: Optional[Dict[str, str]] = None) -> dict:
    """Fold every dossier into one Pikalytics-schema dict. ``min_records`` drops species seen in
    fewer than that many (opponent, species) records — a noise floor for tiny samples."""
    names = names or {}
    n_games = n_opponents = 0
    records: Counter = Counter()                   # species -> # of (opponent, species) records
    seen_games: Counter = Counter()                # species -> Σ times_seen (game-level sightings)
    moves: Dict[str, Counter] = defaultdict(Counter)
    items: Dict[str, Counter] = defaultdict(Counter)
    items_n: Counter = Counter()                   # species -> records WITH a revealed item
    abilities: Dict[str, Counter] = defaultdict(Counter)
    abilities_n: Counter = Counter()
    mates: Dict[str, Counter] = defaultdict(Counter)

    for f in sorted(Path(dossier_dir).glob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        mons = d.get("mons") or {}
        if not mons:
            continue
        n_opponents += 1
        n_games += len(d.get("games") or [])
        species_here = sorted(mons.keys())
        for sp, rec in mons.items():
            records[sp] += 1
            seen_games[sp] += int(rec.get("times_seen") or 1)
            for mv in rec.get("moves") or []:
                moves[sp][mv] += 1
            if rec.get("item"):
                items[sp][rec["item"]] += 1
                items_n[sp] += 1
            if rec.get("ability"):
                abilities[sp][rec["ability"]] += 1
                abilities_n[sp] += 1
            for other in species_here:
                if other != sp:
                    mates[sp][other] += 1

    def _disp(slug: str) -> str:
        return names.get(slug) or names.get(re.sub(r"[^a-z0-9]", "", slug.lower())) or slug

    def _pct_list(counter: Counter, denom: int, key: str = "name", top: int = 30) -> list:
        denom = max(denom, 1)
        return [{key: _disp(n) if key == "name" else n, "pct": round(100.0 * c / denom, 1)}
                for n, c in counter.most_common(top)]

    pokemon = {}
    for sp, n_rec in records.most_common():
        if n_rec < min_records:
            continue
        pokemon[_disp(sp)] = {
            "usage_pct": round(100.0 * seen_games[sp] / max(n_games, 1), 1),
            "moves": _pct_list(moves[sp], n_rec),
            "items": _pct_list(items[sp], items_n[sp]),
            "abilities": _pct_list(abilities[sp], abilities_n[sp]),
            "spreads": [],                          # not observable from dossiers (see docstring)
            "natures": [],
            "teammates": _pct_list(mates[sp], n_rec, top=10),
        }
    return {"format": fmt, "scraped_at": datetime.now(timezone.utc).isoformat(),
            "source": "observed_dossiers", "n_games": n_games, "n_opponents": n_opponents,
            "pokemon": pokemon}


def main() -> int:
    from v_dance.formats import default_format
    fmt = default_format() or ""
    m = re.search(r"(reg[a-z0-9]+)", fmt)
    reg = m.group(1) if m else "regma"
    ap = argparse.ArgumentParser(description="Aggregate opponent dossiers into a Pikalytics-format usage file.")
    ap.add_argument("--dossiers", type=Path, default=DOSSIER_DIR)
    ap.add_argument("--out", type=Path, default=_REPO / "data" / f"observed_meta_{reg}.json")
    ap.add_argument("--min-records", type=int, default=1,
                    help="drop species seen in fewer (opponent, species) records than this.")
    ap.add_argument("--top", type=int, default=15, help="summary rows to print.")
    args = ap.parse_args()

    data = aggregate(args.dossiers, fmt=fmt, min_records=args.min_records,
                     names=_display_names(reg))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[observed-meta] {data['n_opponents']} opponents / {data['n_games']} games "
          f"/ {len(data['pokemon'])} species -> {args.out}")
    print(f"  {'species':<22}{'use%':>6}  top item (pct)          top move (pct)")
    for sp, rec in list(data["pokemon"].items())[:args.top]:
        it = rec["items"][0] if rec["items"] else {"name": "-", "pct": 0}
        mv = rec["moves"][0] if rec["moves"] else {"name": "-", "pct": 0}
        print(f"  {sp:<22}{rec['usage_pct']:>6.1f}  {it['name']:<18}{it['pct']:>5.1f}  "
              f"{mv['name']:<18}{mv['pct']:>5.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
