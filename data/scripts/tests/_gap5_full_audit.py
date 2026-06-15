"""Gap #5 full per-mon / global parity audit (diagnostic, NOT a pytest module).

Drives BOTH encode paths (live poke-env vs offline vod_parser) from a replay log
and diffs EVERY named feature block of the publicly-comparable OPPONENT slots
(opp active 2,3 + opp bench 8-11) and the GLOBAL block, per turn.

Run:  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe data/scripts/tests/_gap5_full_audit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "data" / "scripts"))
sys.path.insert(0, str(ROOT / "data" / "scripts" / "tests"))

import _parity_harness as H  # noqa: E402
from vod_parser.replay_parser import extract_log_from_html  # noqa: E402
from state_encoder import (  # noqa: E402
    STATE_DIM, NUM_TYPES, NUM_STATUS, NUM_BOOSTS, NUM_MOVES, MOVE_FEATURES,
    NUM_WEATHER, NUM_FIELDS, NUM_SIDE_CONDS, ACTIVE_SLOTS, BENCH_SLOTS,
    OPP_BENCH_SLOTS, POKEMON_FEATURES,
)

# ── per-mon named blocks (slice within one POKEMON_FEATURES slot) ──────────────
PMON_BLOCKS = {
    "hp":     (H.OFF_HP,    H.OFF_HP + 1),
    "type1":  (H.OFF_TYPE1, H.OFF_TYPE1 + NUM_TYPES),
    "type2":  (H.OFF_TYPE2, H.OFF_TYPE2 + NUM_TYPES),
    "base":   (H.OFF_BASE,  H.OFF_BASE + 6),
    "est":    (H.OFF_EST,   H.OFF_EST + 6),
    "known":  (H.OFF_KNOWN, H.OFF_KNOWN + 1),
    "mega":   (H.OFF_MEGA,  H.OFF_MEGA + 1),
    "tera":   (H.OFF_TERA,  H.OFF_TERA + 1),
    "status": (H.OFF_STATUS, H.OFF_STATUS + NUM_STATUS),
    "boost":  (H.OFF_BOOST, H.OFF_BOOST + NUM_BOOSTS),
    "moves":  (H.OFF_MOVES, H.OFF_MOVES + NUM_MOVES * MOVE_FEATURES),
    "active": (H.OFF_ACTIVE, H.OFF_ACTIVE + 1),
    "rev":    (H.OFF_REV,   H.OFF_REV + 1),
    "fnt":    (H.OFF_FNT,   H.OFF_FNT + 1),
    "trans":  (H.OFF_TRANS, H.OFF_TRANS + 1),
}

# ── global named blocks (absolute indices) ────────────────────────────────────
_G = (ACTIVE_SLOTS + BENCH_SLOTS + OPP_BENCH_SLOTS) * POKEMON_FEATURES  # 1320
GLOBAL_BLOCKS = {
    "weather":     (_G,                       _G + NUM_WEATHER),
    "fields":      (_G + NUM_WEATHER,          _G + NUM_WEATHER + NUM_FIELDS),
    "our_side_sc": (_G + NUM_WEATHER + NUM_FIELDS,
                    _G + NUM_WEATHER + NUM_FIELDS + NUM_SIDE_CONDS),
    "opp_side_sc": (_G + NUM_WEATHER + NUM_FIELDS + NUM_SIDE_CONDS,
                    _G + NUM_WEATHER + NUM_FIELDS + 2 * NUM_SIDE_CONDS),
    "turn":        (STATE_DIM - 6, STATE_DIM - 5),
    "trickroom":   (STATE_DIM - 5, STATE_DIM - 4),
    "team_counts": (STATE_DIM - 4, STATE_DIM),
}

OPP_SLOTS = (*H.ACTIVE_OPP, *H.OPP_BENCH)   # 2,3,8,9,10,11


def audit_log(log: str, perspective: str, atol: float = 1e-3, splice: bool = False):
    """Return {scope: {block: [(turn, slot, live, off), ...]}} of divergences.

    ``splice=True`` uses the gap-#6 production-honest live path (opponent side
    reconstructed from the public log via vod_parser, spliced into the live
    vector); ``splice=False`` uses the raw poke-env opponent view."""
    user = H.player_usernames(log)[perspective]
    live = H.live_vectors_per_turn(log, user, splice=splice)
    off = H.offline_vectors_per_turn(log, perspective)

    pmon_diffs: dict = {b: [] for b in PMON_BLOCKS}
    glob_diffs: dict = {b: [] for b in GLOBAL_BLOCKS}

    for turn in sorted(off):
        if turn not in live:
            continue
        lv, ov = live[turn], off[turn]
        for slot in OPP_SLOTS:
            b = H.slot_base(slot)
            for name, (lo, hi) in PMON_BLOCKS.items():
                ls, os_ = lv[b + lo:b + hi], ov[b + lo:b + hi]
                if not np.allclose(ls, os_, atol=atol):
                    pmon_diffs[name].append((turn, slot, np.round(ls, 3), np.round(os_, 3)))
        for name, (lo, hi) in GLOBAL_BLOCKS.items():
            ls, os_ = lv[lo:hi], ov[lo:hi]
            if not np.allclose(ls, os_, atol=atol):
                glob_diffs[name].append((turn, np.round(ls, 3), np.round(os_, 3)))

    return pmon_diffs, glob_diffs


def report(label: str, log: str, perspective: str, show_examples: int = 2):
    pmon, glob = audit_log(log, perspective)
    print(f"\n=== {label}  (perspective {perspective}) ===")
    any_div = False
    for name, lst in pmon.items():
        if lst:
            any_div = True
            print(f"  [per-mon] {name}: {len(lst)} slot-turns differ")
            for (t, s, ls, os_) in lst[:show_examples]:
                print(f"      t{t} slot{s}: live={list(ls)}  off={list(os_)}")
    for name, lst in glob.items():
        if lst:
            any_div = True
            print(f"  [global] {name}: {len(lst)} turns differ")
            for (t, ls, os_) in lst[:show_examples]:
                print(f"      t{t}: live={list(ls)}  off={list(os_)}")
    if not any_div:
        print("  NO OPP-SIDE / GLOBAL DIVERGENCES")
    return pmon, glob


if __name__ == "__main__":
    VODS = ROOT / "data" / "vods"
    targets = [
        # (label, filename, perspective)
        ("dual Zoroark+Ditto", "Gen9ChampionsVGC2026RegMA-2026-06-13-victoriousdancing-kronomono.html", "p1"),
        ("dual Zoroark+Ditto", "Gen9ChampionsVGC2026RegMA-2026-06-13-victoriousdancing-kronomono.html", "p2"),
        ("real-HP (kronomono=opp)", "Gen9ChampionsVGC2026RegMA-2026-06-12-kronomono-guts1100.html", "p2"),
        ("real-HP (kronomono=own)", "Gen9ChampionsVGC2026RegMA-2026-06-12-kronomono-guts1100.html", "p1"),
        ("clean", "Gen9ChampionsVGC2026RegMA-2026-04-20-stevenhevgc-speedyturtle87.html", "p1"),
    ]
    for label, fname, persp in targets:
        hit = next(VODS.rglob(fname), None)
        if hit is None:
            print(f"!! fixture not found: {fname}")
            continue
        log = extract_log_from_html(hit.read_text(encoding="utf-8"))
        report(label, log, persp)
