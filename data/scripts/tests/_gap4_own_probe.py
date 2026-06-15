import sys
sys.path.insert(0, 'data/scripts')
sys.path.insert(0, 'data/scripts/tests')

import numpy as np
from pathlib import Path
import _parity_harness as H
from vod_parser.replay_parser import extract_log_from_html, ShowdownReplayParser
from state_encoder import STATE_DIM, POKEMON_FEATURES

FIXTURES = {
    'zoroark_ditto': 'Gen9ChampionsVGC2026RegMA-2026-06-13-victoriousdancing-kronomono.html',
    'clean': 'Gen9ChampionsVGC2026RegMA-2026-04-20-stevenhevgc-speedyturtle87.html',
}

OWN_LIVE = STATE_DIM - 4
OPP_LIVE = STATE_DIM - 3
OWN_FNT  = STATE_DIM - 2
OPP_FNT  = STATE_DIM - 1

OFF_FNT = POKEMON_FEATURES - 2  # 108


def find_fix(name):
    return next(Path('data/vods').rglob(name))


def recover(vec, idx):
    return round(float(vec[idx]) * 4)


def slot_fnt(vec, slot):
    b = H.slot_base(slot)
    return float(vec[b + OFF_FNT])


def slot_empty(vec, slot):
    return H.slot_is_empty(vec, slot)


def run(name, fname, persp):
    print("=" * 90)
    print(f"FIXTURE {name}  persp={persp}  file={fname}")
    log = extract_log_from_html(find_fix(fname).read_text(encoding='utf-8'))
    users = H.player_usernames(log)
    user = users[persp]
    print("players:", users, " our user:", user)

    live = H.live_vectors_per_turn(log, user)
    off  = H.offline_vectors_per_turn(log, persp)

    turns = sorted(set(live) & set(off))
    print(f"turns compared: {turns[0]}..{turns[-1]} (n={len(turns)})")
    print()
    hdr = f"{'turn':>4} | {'L own_live':>10} {'O own_live':>10} | {'L own_fnt':>9} {'O own_fnt':>9} | own-bench-empty L/O"
    print(hdr)
    print("-" * len(hdr))
    diverge_count_live = []
    diverge_count_fnt = []
    diverge_bench = []
    for t in turns:
        lv, ov = live[t], off[t]
        l_live, o_live = recover(lv, OWN_LIVE), recover(ov, OWN_LIVE)
        l_fnt,  o_fnt  = recover(lv, OWN_FNT),  recover(ov, OWN_FNT)
        # own bench slots 4-7: which are non-empty
        l_bench = [s for s in (4,5,6,7) if not slot_empty(lv, s)]
        o_bench = [s for s in (4,5,6,7) if not slot_empty(ov, s)]
        flag_live = "  <== own_live" if l_live != o_live else ""
        flag_fnt  = "  <== own_fnt"  if l_fnt  != o_fnt  else ""
        flag_bench = "  <== bench" if l_bench != o_bench else ""
        if l_live != o_live: diverge_count_live.append(t)
        if l_fnt != o_fnt: diverge_count_fnt.append(t)
        if l_bench != o_bench: diverge_bench.append(t)
        print(f"{t:>4} | {l_live:>10} {o_live:>10} | {l_fnt:>9} {o_fnt:>9} | "
              f"{str(l_bench):>10}/{str(o_bench):<10}{flag_live}{flag_fnt}{flag_bench}")

    print()
    print(f"own_live divergent turns: {diverge_count_live}")
    print(f"own_fnt  divergent turns: {diverge_count_fnt}")
    print(f"own-bench-occupancy divergent turns: {diverge_bench}")
    return log, live, off, turns


if __name__ == "__main__":
    run('zoroark_ditto', FIXTURES['zoroark_ditto'], 'p2')
    print()
    run('clean', FIXTURES['clean'], 'p2')
    print()
    run('clean', FIXTURES['clean'], 'p1')
