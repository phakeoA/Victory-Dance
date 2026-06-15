"""GAP #4 lens: IS_FAINTED flag (offset 108) across all 12 slots, both paths."""
import sys
from pathlib import Path
sys.path.insert(0, 'data/scripts')
sys.path.insert(0, 'data/scripts/tests')

import numpy as np
import _parity_harness as H
from vod_parser.replay_parser import extract_log_from_html, ShowdownReplayParser
from state_encoder import STATE_DIM, POKEMON_FEATURES

OFF_FNT = H.OFF_FNT  # 108
OFF_STATUS = H.OFF_STATUS  # 56
NUM_STATUS = 7
FNT_STATUS_IDX = 1  # STATUS_NAMES.index("FNT")

SLOT_NAMES = {
    0: "own_active_0", 1: "own_active_1",
    2: "opp_active_0", 3: "opp_active_1",
    4: "own_bench_0", 5: "own_bench_1", 6: "own_bench_2", 7: "own_bench_3",
    8: "opp_bench_0", 9: "opp_bench_1", 10: "opp_bench_2", 11: "opp_bench_3",
}

FIXTURES = {
    "zoroark_ditto": "Gen9ChampionsVGC2026RegMA-2026-06-13-victoriousdancing-kronomono.html",
    "clean": "Gen9ChampionsVGC2026RegMA-2026-04-20-stevenhevgc-speedyturtle87.html",
}


def fnt_flag(vec, slot):
    return vec[H.slot_base(slot) + OFF_FNT]


def fnt_status_onehot(vec, slot):
    """Return the FNT one-hot bit of the status block for this slot."""
    b = H.slot_base(slot) + OFF_STATUS
    return vec[b + FNT_STATUS_IDX]


def slot_present(vec, slot):
    return not H.slot_is_empty(vec, slot)


def run_fixture(label, fname):
    print(f"\n{'='*78}\nFIXTURE: {label}  ({fname})\n{'='*78}")
    path = next(Path('data/vods').rglob(fname))
    log = extract_log_from_html(path.read_text(encoding='utf-8'))
    users = H.player_usernames(log)
    print(f"players: {users}")

    for persp in ("p1", "p2"):
        user = users[persp]
        print(f"\n----- PERSPECTIVE {persp} (user={user}) -----")
        live = H.live_vectors_per_turn(log, user)
        off = H.offline_vectors_per_turn(log, persp)
        turns = sorted(set(live) & set(off))
        print(f"turns compared: {turns[0]}..{turns[-1]} (n={len(turns)})")

        divergences = []
        for t in turns:
            lv, ov = live[t], off[t]
            for slot in range(12):
                lf, of = fnt_flag(lv, slot), fnt_flag(ov, slot)
                lp, op = slot_present(lv, slot), slot_present(ov, slot)
                # is_fainted flag mismatch
                flag_diff = abs(lf - of) > 1e-6
                # presence-of-fainted mismatch: a fainted mon present on one path only
                present_diff = (lp != op)
                if flag_diff or present_diff:
                    divergences.append((t, slot, lf, of, lp, op))

        if not divergences:
            print("  NO is_fainted / presence divergence across all slots/turns.")
        else:
            print(f"  {len(divergences)} divergent (turn,slot) cells:")
            for (t, slot, lf, of, lp, op) in divergences:
                print(f"    turn {t:2d} slot {slot:2d} ({SLOT_NAMES[slot]:13s}) "
                      f"live_fnt={lf:.1f} off_fnt={of:.1f} "
                      f"live_present={lp} off_present={op}")

        # Status one-hot FNT cross-check: wherever is_fainted=1, status must be FNT
        status_mismatch = []
        for t in turns:
            for vec, name in ((live[t], "live"), (off[t], "offline")):
                for slot in range(12):
                    if abs(fnt_flag(vec, slot) - 1.0) < 1e-6:
                        if abs(fnt_status_onehot(vec, slot) - 1.0) > 1e-6:
                            status_mismatch.append((t, slot, name))
        if status_mismatch:
            print(f"  STATUS-FNT one-hot NOT set on fainted mon:")
            for (t, slot, name) in status_mismatch:
                print(f"    turn {t} slot {slot} ({SLOT_NAMES[slot]}) path={name}")
        else:
            print("  status FNT one-hot consistent with is_fainted on both paths.")


if __name__ == "__main__":
    for label, fname in FIXTURES.items():
        run_fixture(label, fname)
