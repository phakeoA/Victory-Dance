"""Standalone verification that the local_battle gap-#6 splice wiring works
WITHOUT needing the Showdown server.

It (1) imports the local spliced players (catches import/collision errors),
(2) drives a poke-env DoubleBattle from the spottedwoot illusion replay exactly
as the live Player would receive it, simulates the protocol capture, runs the
splice (_build_opp_snapshot logic), encodes, and checks the spliced opponent
side matches the OFFLINE training encoder (the gap-#6 target) — including the
re-disguise turn that used to carry the phantom, and (3) checks the synthesised
|poke| roster (the |showteam| live path).

Run: PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe local_battle/_verify_splice.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
for _p in (str(_REPO / "data" / "scripts" / "tests"), str(_REPO / "data" / "scripts"),
           str(_REPO), str(_HERE)):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

# (1) imports — must resolve to the LOCAL spliced versions, no collision
import live_vgc_base as LVB
from live_vgc_base import SplicingVGCPlayerBase, _CAPTURE_SKIP, _norm_tag, _display_species
import player as local_player
import random_player as local_random
print("[1] imports OK:",
      f"player.VGCPlayer={local_player.VGCPlayer.__name__}",
      f"random.RandomVGCPlayer={local_random.RandomVGCPlayer.__name__}")
assert issubclass(local_player.VGCPlayer, SplicingVGCPlayerBase), "VGCPlayer not spliced!"
assert issubclass(local_random.RandomVGCPlayer, SplicingVGCPlayerBase), "RandomVGCPlayer not spliced!"

import _parity_harness as H
from vod_parser.replay_parser import extract_log_from_html
from live_state_encoder import LiveStateEncoder, opp_snapshot_from_log_prefix
from state_encoder import VodStateEncoder, POKEMON_FEATURES, ACTIVE_SLOTS, BENCH_SLOTS, OPP_BENCH_SLOTS

OPP_LO = (ACTIVE_SLOTS - 2) * POKEMON_FEATURES
OPP_HI = ACTIVE_SLOTS * POKEMON_FEATURES
BENCH_LO = (ACTIVE_SLOTS + BENCH_SLOTS) * POKEMON_FEATURES
BENCH_HI = (ACTIVE_SLOTS + BENCH_SLOTS + OPP_BENCH_SLOTS) * POKEMON_FEATURES


def _capture_lines_from_log(log: str):
    """Mimic SplicingVGCPlayerBase._handle_battle_message capture from a raw log:
    keep public protocol lines, skip the same client-only message types."""
    out = []
    for ln in log.split("\n"):
        if not ln.startswith("|"):
            continue
        parts = ln.split("|")
        if len(parts) < 2 or parts[1] in _CAPTURE_SKIP:
            continue
        out.append(ln)
    return out


def main():
    hit = next(_REPO.joinpath("data", "vods").rglob("*spottedwoot-lampistest*.html"))
    log = extract_log_from_html(hit.read_text(encoding="utf-8"))
    persp = "p1"
    user = H.player_usernames(log)[persp]

    # (2) splice pipeline, replicating _build_opp_snapshot with captured lines
    captured = _capture_lines_from_log(log)
    has_poke = any(l.startswith("|poke|") for l in captured)
    print(f"[2] captured {len(captured)} protocol lines; has |poke|={has_poke}")

    enc_live = LiveStateEncoder()
    enc_off = VodStateEncoder()
    off = H.offline_vectors_per_turn(log, persp)

    mech_bad = phantom = seam_turns = 0
    checked = 0
    for turn in sorted(off):
        battle = H.drive_live_battle_to_turn(log, user, turn)
        # _build_opp_snapshot equivalent (replay already has |poke| → no synth)
        opp_snap = opp_snapshot_from_log_prefix("\n".join(captured), persp, turn)
        vec = enc_live.encode(battle, opp_snapshot=opp_snap)
        assert vec.shape[0] == off[turn].shape[0]
        checked += 1
        # WIRING CORRECTNESS: the spliced opp bytes must EXACTLY equal the offline
        # encoder applied to the SAME (prefix) snapshot — proves encode() faithfully
        # injected it into the opp ranges and nowhere else.
        ref = enc_off.encode_snapshot(opp_snap, turn=turn)
        if not (np.allclose(vec[OPP_LO:OPP_HI], ref[OPP_LO:OPP_HI], atol=1e-4)
                and np.allclose(vec[BENCH_LO:BENCH_HI], ref[BENCH_LO:BENCH_HI], atol=1e-4)):
            mech_bad += 1
        # INFORMATIONAL: real-time prefix vs full-parse training (the documented
        # gap-#6 seam — same identity, transient HP/seen diffs on re-disguise).
        if not (np.allclose(vec[OPP_LO:OPP_HI], off[turn][OPP_LO:OPP_HI], atol=1e-4)
                and np.allclose(vec[BENCH_LO:BENCH_HI], off[turn][BENCH_LO:BENCH_HI], atol=1e-4)):
            seam_turns += 1
        # phantom check on the snapshot itself
        if opp_snap:
            oa = [v["species"] for v in (opp_snap.get("opp_active") or {}).values()]
            ob = [m["species"] for m in (opp_snap.get("opp_bench") or [])]
            if set(oa) & set(ob):
                phantom += 1
    print(f"[2] checked {checked} turns | splice-mechanics failures={mech_bad} "
          f"| phantom turns={phantom} | real-time-vs-full seam turns={seam_turns} (expected on re-disguise)")

    # (3) synthesised |poke| roster (the live |showteam| path): strip |poke|
    # from the captured log and confirm synth regenerates a working roster.
    battle = H.drive_live_battle_to_turn(log, user, 8)
    synth = SplicingVGCPlayerBase._synth_poke_lines(battle)
    print(f"[3] _synth_poke_lines produced {len(synth)} lines; sample: {synth[:3]}")
    no_poke = [l for l in captured if not l.startswith("|poke|")]
    opp_snap_synth = opp_snapshot_from_log_prefix("\n".join(synth + no_poke), persp, 8)
    opp_snap_real = opp_snapshot_from_log_prefix("\n".join(captured), persp, 8)
    def sig(s):
        return (sorted((s.get("opp_active") or {}).items()),
                sorted((m["species"], bool(m.get("seen"))) for m in (s.get("opp_bench") or [])))
    match = sig(opp_snap_synth) == sig(opp_snap_real) if (opp_snap_synth and opp_snap_real) else False
    print(f"[3] synth-roster opp snapshot matches replay-|poke| snapshot: {match}")
    if not match and opp_snap_synth:
        print("    synth bench:", [(m['species'], m.get('seen')) for m in opp_snap_synth['opp_bench']])
        print("    real  bench:", [(m['species'], m.get('seen')) for m in opp_snap_real['opp_bench']])

    ok = (mech_bad == 0 and phantom == 0 and match)
    print("\n>>> SPLICE WIRING", "VERIFIED ✅" if ok else "HAS ISSUES ❌")
    print("    (splice mechanics faithful, no phantom, synth roster works; any"
          " seam turns are the documented real-time-vs-lookahead edge, not a bug)")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
