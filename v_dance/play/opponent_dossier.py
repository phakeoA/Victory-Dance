"""B-L2 OpponentDossier (capture half) — Phase-3 design lever L2 (docs/phase3_adaptation_loop_design.md).

One JSON per opponent under ``artifacts/dossiers/<opp_id>.json``, updated passively at every
finished benchmark/online battle (same hooks as bench recording): revealed species → moves /
item / ability, W-L vs us, per-game history. This is the DATA SUBSTRATE for cross-game
adaptation; the consumer (belief/TP warm-start = L2b) is deliberately deferred until ≥3 USER
sets exist to gate it (design §Build order). Capture is pure IO — it can never change play.

CLI:  python -m v_dance.play.opponent_dossier            # list dossiers
      python -m v_dance.play.opponent_dossier <name>     # print one

⚠ mega/forme note: ``mon.species`` is the BASE species (poke-env, memory 2026-06-22) — dossier
keys are base-species; forme/mega shows up in the revealed moves/items instead.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

_REPO = Path(__file__).resolve().parents[2]
DOSSIER_DIR = _REPO / "artifacts" / "dossiers"
MAX_GAMES = 200          # per-opponent history cap (newest kept)


def _toid(s: str) -> str:
    return "".join(c for c in (s or "").lower() if c.isalnum())


def dossier_path(opponent: str) -> Path:
    return DOSSIER_DIR / f"{_toid(opponent) or 'unknown'}.json"


def load(opponent: str) -> dict:
    p = dossier_path(opponent)
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass                                        # corrupt → start fresh (capture only)
    return {"opponent": opponent, "games": [], "mons": {}, "wins_vs_us": 0, "losses_vs_us": 0,
            "draws_vs_us": 0}


def update_from_battle(battle, result: str, our_team: Optional[str] = None,
                       note: str = "") -> Optional[Path]:
    """Merge one FINISHED poke-env battle into the opponent's dossier and save it.

    ``result`` is OUR result string ("ai"|"human"|"draw" — bench-row convention, where
    "ai" = WE won ⇒ the opponent LOST). Never raises (capture must never break play);
    returns the written path or None."""
    try:
        opp = getattr(battle, "opponent_username", None)
        if not opp:
            return None
        d = load(opp)
        d["opponent"] = opp                             # keep the display name fresh
        # W-L bookkeeping from THEIR side: our "ai" win = their loss.
        key = {"ai": "losses_vs_us", "human": "wins_vs_us", "draw": "draws_vs_us"}.get(result)
        if key:
            d[key] = int(d.get(key, 0)) + 1
        # Revealed sets: union moves, last-seen item/ability, sighting count per base species.
        for mon in (getattr(battle, "opponent_team", None) or {}).values():
            sp = _toid(getattr(mon, "species", "") or "")
            if not sp:
                continue
            rec = d["mons"].setdefault(sp, {"species": getattr(mon, "species", sp),
                                            "moves": [], "item": None, "ability": None,
                                            "times_seen": 0})
            rec["times_seen"] += 1
            moves = set(rec.get("moves") or [])
            moves.update((getattr(mon, "moves", None) or {}).keys())
            rec["moves"] = sorted(moves)
            item = getattr(mon, "item", None)
            if item and item not in ("unknown_item",):
                rec["item"] = item
            ability = getattr(mon, "ability", None)
            if ability:
                rec["ability"] = ability
        d["games"].append({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                           "battle_tag": getattr(battle, "battle_tag", None),
                           "result": result, "turns": getattr(battle, "turn", None),
                           "our_team": our_team, "note": note})
        d["games"] = d["games"][-MAX_GAMES:]
        DOSSIER_DIR.mkdir(parents=True, exist_ok=True)
        p = dossier_path(opp)
        p.write_text(json.dumps(d, indent=1), encoding="utf-8")
        return p
    except Exception:
        return None                                     # capture is best-effort by contract


def summary(opponent: str) -> str:
    d = load(opponent)
    n = len(d.get("games") or [])
    return (f"{d.get('opponent', opponent)}: {n} game(s) — them {d.get('wins_vs_us', 0)}W · "
            f"us {d.get('losses_vs_us', 0)}W · {d.get('draws_vs_us', 0)}D, "
            f"{len(d.get('mons') or {})} revealed mons")


def main() -> None:                                      # tiny CLI: list / print one
    import sys
    if len(sys.argv) > 1:
        name = sys.argv[1]
        p = dossier_path(name)
        if not p.is_file():
            raise SystemExit(f"no dossier for {name!r} ({p})")
        print(p.read_text(encoding="utf-8"))
        return
    DOSSIER_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(DOSSIER_DIR.glob("*.json"))
    if not files:
        print(f"(no dossiers yet — they accumulate under {DOSSIER_DIR})")
    for f in files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            print(f"  {f.stem:<24} {len(d.get('games') or []):>3} games   "
                  f"{len(d.get('mons') or {}):>2} revealed mons")
        except (json.JSONDecodeError, OSError):
            print(f"  {f.stem:<24} (unreadable)")


if __name__ == "__main__":
    main()
