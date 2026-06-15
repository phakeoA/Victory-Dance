"""GAP #4: dump is_fainted flag for every PRESENT slot, both paths, all turns."""
import sys
from pathlib import Path
sys.path.insert(0, 'data/scripts')
sys.path.insert(0, 'data/scripts/tests')

import numpy as np
import _parity_harness as H
from vod_parser.replay_parser import extract_log_from_html, ShowdownReplayParser
from state_encoder import STATE_DIM, POKEMON_FEATURES

OFF_FNT = H.OFF_FNT
OFF_STATUS = H.OFF_STATUS
FNT_STATUS_IDX = 1

SLOT_NAMES = {
    0: "ownA0", 1: "ownA1", 2: "oppA0", 3: "oppA1",
    4: "ownB0", 5: "ownB1", 6: "ownB2", 7: "ownB3",
    8: "oppB0", 9: "oppB1", 10: "oppB2", 11: "oppB3",
}

FIXTURES = {
    "zoroark_ditto": "Gen9ChampionsVGC2026RegMA-2026-06-13-victoriousdancing-kronomono.html",
    "clean": "Gen9ChampionsVGC2026RegMA-2026-04-20-stevenhevgc-speedyturtle87.html",
}


def fnt(vec, slot):
    return vec[H.slot_base(slot) + OFF_FNT]


def present(vec, slot):
    return not H.slot_is_empty(vec, slot)


def run(label, fname):
    print(f"\n##### {label} #####")
    path = next(Path('data/vods').rglob(fname))
    log = extract_log_from_html(path.read_text(encoding='utf-8'))
    users = H.player_usernames(log)
    for persp in ("p1", "p2"):
        user = users[persp]
        live = H.live_vectors_per_turn(log, user)
        off = H.offline_vectors_per_turn(log, persp)
        turns = sorted(set(live) & set(off))
        print(f"\n-- persp {persp} ({user}) --")
        for t in turns:
            lv, ov = live[t], off[t]
            # collect any slot where EITHER path has the mon present, summarising fnt flag
            cells = []
            for slot in range(12):
                lp, op = present(lv, slot), present(ov, slot)
                if not (lp or op):
                    continue
                lf, of = fnt(lv, slot), fnt(ov, slot)
                mark = ""
                if lp != op:
                    mark = " <PRESENCE_DIFF>"
                elif abs(lf - of) > 1e-6:
                    mark = " <FNT_DIFF>"
                if lf > 0.5 or of > 0.5 or mark:
                    cells.append(f"{SLOT_NAMES[slot]}:L{int(lf)}O{int(of)}p{int(lp)}{int(op)}{mark}")
            if cells:
                print(f"  turn {t:2d}: " + "  ".join(cells))


if __name__ == "__main__":
    for label, fname in FIXTURES.items():
        run(label, fname)
