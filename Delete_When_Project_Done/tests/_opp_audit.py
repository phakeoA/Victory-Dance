import sys
sys.path.insert(0, 'data/scripts')
sys.path.insert(0, 'data/scripts/tests')
import numpy as np
from pathlib import Path
import _parity_harness as H
from vod_parser.replay_parser import extract_log_from_html, ShowdownReplayParser
from state_encoder import STATE_DIM, POKEMON_FEATURES

OFF_FNT = H.OFF_FNT  # 108
IDX = {
    'own_live': STATE_DIM - 4,
    'opp_live': STATE_DIM - 3,
    'own_fnt':  STATE_DIM - 2,
    'opp_fnt':  STATE_DIM - 1,
}
OPP_ACTIVE_SLOTS = (2, 3)
OPP_BENCH_SLOTS = (8, 9, 10, 11)

FIXTURES = {
    'dual_zoroark_ditto': 'Gen9ChampionsVGC2026RegMA-2026-06-13-victoriousdancing-kronomono.html',
    'clean': 'Gen9ChampionsVGC2026RegMA-2026-04-20-stevenhevgc-speedyturtle87.html',
}

def fnt_flag(vec, slot):
    return vec[H.slot_base(slot) + OFF_FNT]

def slot_empty(vec, slot):
    b = H.slot_base(slot)
    return bool(np.all(vec[b:b + POKEMON_FEATURES] == 0.0))

def run(fixname, fname):
    log = extract_log_from_html(next(Path('data/vods').rglob(fname)).read_text(encoding='utf-8'))
    names = H.player_usernames(log)
    print(f"\n========== FIXTURE {fixname}  players={names} ==========")
    for persp in ('p1', 'p2'):
        user = names[persp]
        live = H.live_vectors_per_turn(log, user)
        off  = H.offline_vectors_per_turn(log, persp)
        turns = sorted(set(live) & set(off))
        print(f"\n--- perspective {persp} (user={user}) turns={turns[0]}..{turns[-1]} ({len(turns)}) ---")
        any_div = False
        for t in turns:
            lv, ov = live[t], off[t]
            # opp_live / opp_fnt globals (recover counts: *4)
            l_ol, o_ol = lv[IDX['opp_live']] * 4, ov[IDX['opp_live']] * 4
            l_of, o_of = lv[IDX['opp_fnt']] * 4,  ov[IDX['opp_fnt']] * 4
            msgs = []
            if abs(l_ol - o_ol) > 1e-3:
                msgs.append(f"opp_live live={l_ol:.2f} off={o_ol:.2f}")
            if abs(l_of - o_of) > 1e-3:
                msgs.append(f"opp_fnt live={l_of:.2f} off={o_of:.2f}")
            # opp active is_fainted (slots 2,3)
            for s in OPP_ACTIVE_SLOTS:
                lf, of_ = fnt_flag(lv, s), fnt_flag(ov, s)
                if abs(lf - of_) > 1e-3:
                    msgs.append(f"opp-active slot{s} is_fnt live={lf} off={of_}")
            # opp bench is_fainted (slots 8-11), only meaningful when both non-empty?
            for s in OPP_BENCH_SLOTS:
                lf, of_ = fnt_flag(lv, s), fnt_flag(ov, s)
                le, oe = slot_empty(lv, s), slot_empty(ov, s)
                if abs(lf - of_) > 1e-3:
                    msgs.append(f"opp-bench slot{s} is_fnt live={lf}(empty={le}) off={of_}(empty={oe})")
            if msgs:
                any_div = True
                print(f"  turn {t}: " + " | ".join(msgs))
        if not any_div:
            print("  OPP-SIDE CONSISTENT on all turns (opp_live/opp_fnt + opp is_fainted slots 2,3,8-11)")

for fn, f in FIXTURES.items():
    run(fn, f)
