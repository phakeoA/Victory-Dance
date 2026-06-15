"""GAP #4 empirical sweep: compare the 4 team-count globals + per-mon is_fainted
flag (offset 108, all 12 slots) between LIVE and OFFLINE encoders, for BOTH
fixtures and BOTH perspectives, every turn present in BOTH paths.
"""
import sys
sys.path.insert(0, 'data/scripts')
sys.path.insert(0, 'data/scripts/tests')

from pathlib import Path
import numpy as np

import _parity_harness as H
from vod_parser.replay_parser import extract_log_from_html, ShowdownReplayParser
from state_encoder import STATE_DIM, POKEMON_FEATURES

OFF_FNT = 108  # POKEMON_FEATURES - 2
SLOT_NAMES = {
    0: "own_act0", 1: "own_act1", 2: "opp_act0", 3: "opp_act1",
    4: "own_bench0", 5: "own_bench1", 6: "own_bench2", 7: "own_bench3",
    8: "opp_bench0", 9: "opp_bench1", 10: "opp_bench2", 11: "opp_bench3",
}

FIXTURES = {
    "ZOROARK_DITTO": "Gen9ChampionsVGC2026RegMA-2026-06-13-victoriousdancing-kronomono.html",
    "CLEAN": "Gen9ChampionsVGC2026RegMA-2026-04-20-stevenhevgc-speedyturtle87.html",
}


def recover_counts(vec):
    """Return (own_live, opp_live, own_fnt, opp_fnt) recovered counts."""
    return tuple(round(float(vec[STATE_DIM - 4 + k]) * 4) for k in range(4))


def fnt_flags(vec):
    return [int(round(float(vec[H.slot_base(s) + OFF_FNT]))) for s in range(12)]


def run_fixture(label, fname):
    path = next(Path('data/vods').rglob(fname))
    log = extract_log_from_html(path.read_text(encoding='utf-8'))
    names = H.player_usernames(log)
    print(f"\n{'='*78}\nFIXTURE {label}: {fname}\n  players: {names}\n{'='*78}")

    divergences = []
    for persp in ("p1", "p2"):
        user = names[persp]
        live = H.live_vectors_per_turn(log, user)
        off = H.offline_vectors_per_turn(log, persp)
        turns = sorted(set(live) & set(off))
        print(f"\n--- perspective {persp} (user={user}) "
              f"turns in both: {turns[0]}..{turns[-1]} ({len(turns)}) ---")

        for t in turns:
            lv, ov = live[t], off[t]
            lc, oc = recover_counts(lv), recover_counts(ov)
            lf, of = fnt_flags(lv), fnt_flags(ov)
            counts_differ = lc != oc
            flag_diffs = [s for s in range(12) if lf[s] != of[s]]
            if counts_differ or flag_diffs:
                print(f"\n  TURN {t} DIVERGES:")
                if counts_differ:
                    print(f"    counts (own_live,opp_live,own_fnt,opp_fnt):")
                    print(f"      live={lc}  offline={oc}")
                for s in flag_diffs:
                    print(f"    is_fainted[{s}={SLOT_NAMES[s]}]: "
                          f"live={lf[s]} offline={of[s]}")
                divergences.append((label, persp, t, lc, oc, flag_diffs, lf, of))
    return divergences


def main():
    all_div = []
    for label, fname in FIXTURES.items():
        all_div += run_fixture(label, fname)
    print(f"\n\n{'#'*78}\nTOTAL DIVERGING (turn,persp,fixture) rows: {len(all_div)}\n{'#'*78}")
    for (label, persp, t, lc, oc, fd, lf, of) in all_div:
        fds = ",".join(f"{s}:{SLOT_NAMES[s]}" for s in fd)
        print(f"  {label:14s} {persp} turn {t:3d} "
              f"counts live={lc} off={oc}  fnt_diff=[{fds}]")


if __name__ == "__main__":
    main()
