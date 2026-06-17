import sys
sys.path.insert(0, 'data/scripts')
sys.path.insert(0, 'data/scripts/tests')

from pathlib import Path
import numpy as np
import _parity_harness as H
from vod_parser.replay_parser import extract_log_from_html
from state_encoder import STATE_DIM, POKEMON_FEATURES, _TYPE_IDX

FNAME = 'Gen9ChampionsVGC2026RegMA-2026-06-13-victoriousdancing-kronomono.html'
PERSP = 'p2'
OFF_HP = 0
OFF_FNT = POKEMON_FEATURES - 2
OFF_REV = POKEMON_FEATURES - 3
OFF_ACTIVE = POKEMON_FEATURES - 4
TYPE1_BASE = 1
IDX2NAME = {v: k for k, v in _TYPE_IDX.items()}


def find_fix(name):
    return next(Path('data/vods').rglob(name))


def type1_of(vec, slot):
    b = H.slot_base(slot)
    for i in range(20):
        if vec[b + TYPE1_BASE + i] > 0.5:
            return IDX2NAME[i]
    return None


if __name__ == "__main__":
    log = extract_log_from_html(find_fix(FNAME).read_text(encoding='utf-8'))
    user = H.player_usernames(log)[PERSP]
    live = H.live_vectors_per_turn(log, user)
    off = H.offline_vectors_per_turn(log, PERSP)

    for t in (2, 3):
        lv, ov = live[t], off[t]
        print(f"=== turn {t} : own bench slot 4 ===")
        for label, vec in (("LIVE", lv), ("OFFLINE", ov)):
            b = H.slot_base(4)
            empty = bool(np.all(vec[b:b+POKEMON_FEATURES] == 0.0))
            print(f"  {label:<8} empty={empty} type1={type1_of(vec,4)} "
                  f"hp={vec[b+OFF_HP]:.3f} is_active={vec[b+OFF_ACTIVE]:.0f} "
                  f"is_revealed={vec[b+OFF_REV]:.0f} is_fainted={vec[b+OFF_FNT]:.0f}")
        print(f"  globals: own_live L={lv[STATE_DIM-4]*4:.0f} O={ov[STATE_DIM-4]*4:.0f} | "
              f"own_fnt L={lv[STATE_DIM-2]*4:.0f} O={ov[STATE_DIM-2]*4:.0f}")
        print()
