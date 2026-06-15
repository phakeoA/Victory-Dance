"""Full-pipeline verification across Zoroark/Ditto edge-case replays.

For every replay that contains an Illusion (Zoroark) or Transform (Ditto), and
for BOTH perspectives, this exercises all three modules end to end:

  vod_parser.py         → parse the replay to snapshots          (no crash)
  state_encoder.py      → encode_snapshot offline → (1398,)      (shape + finite)
  live_state_encoder.py → poke-env DoubleBattle → encode (1398,) (shape + finite)

and runs the per-feature-block parity audit (opponent side + globals), tallying
which blocks still diverge so we can confirm only the KNOWN-remaining ones
(side conditions, boosts, reveal-order moves) appear — not the fixed ones
(opp-bench, is_transformed, PP, team counts, is_fainted, HP scale, mega).

Run:  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe data/scripts/tests/_gap5_pipeline_verify.py
"""
from __future__ import annotations

import os
import sys
import traceback
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "data" / "scripts"))
sys.path.insert(0, str(ROOT / "data" / "scripts" / "tests"))

# Gap #6: exercise the production-honest live path (vod_parser opponent splice).
# Set GAP6_SPLICE=0 to measure the raw poke-env opponent view instead.
SPLICE = os.environ.get("GAP6_SPLICE", "1") != "0"

import _parity_harness as H            # noqa: E402
import _gap5_full_audit as A           # noqa: E402
from state_encoder import STATE_DIM, VodStateEncoder  # noqa: E402
from live_state_encoder import LiveStateEncoder        # noqa: E402
from vod_parser.replay_parser import ShowdownReplayParser, extract_log_from_html  # noqa: E402


def find_edge_replays():
    out = []
    for v in (ROOT / "data" / "vods").rglob("*.html"):
        try:
            log = extract_log_from_html(v.read_text(encoding="utf-8")).lower()
        except Exception:
            continue
        if any(t in log for t in ("zoroark", "|replace|", "ditto", "imposter",
                                   "|-transform|")):
            out.append(v)
    return out


def verify_one(path: Path):
    """Return (ok, stats_dict, block_divergences:Counter)."""
    log = extract_log_from_html(path.read_text(encoding="utf-8"))
    stats = {"offline_vecs": 0, "live_vecs": 0, "turns": 0}
    blocks: Counter = Counter()
    offenc, liveenc = VodStateEncoder(), LiveStateEncoder()

    for persp in ("p1", "p2"):
        # ── vod_parser + offline encoder ──
        res = ShowdownReplayParser(log, our_player=persp).parse()
        for turn in res["turns"]:
            for key in ("state_before_actions", "state_after_actions"):
                snap = turn.get(key, {}).get(persp)
                if not snap:
                    continue
                vec = offenc.encode_snapshot(snap, turn=turn["turn"])
                assert vec.shape == (STATE_DIM,), f"offline shape {vec.shape}"
                assert np.isfinite(vec).all(), "offline non-finite"
                stats["offline_vecs"] += 1

        # ── live encoder (gap-#6 production-honest spliced opponent path) ──
        user = H.player_usernames(log)[persp]
        live = H.live_vectors_per_turn(log, user, encoder=liveenc, splice=SPLICE)
        for v in live.values():
            assert v.shape == (STATE_DIM,), f"live shape {v.shape}"
            assert np.isfinite(v).all(), "live non-finite"
        stats["live_vecs"] += len(live)

        # ── per-block parity audit (opp side + globals) ──
        pmon, glob = A.audit_log(log, persp, splice=SPLICE)
        for name, lst in pmon.items():
            if lst:
                blocks[f"pmon:{name}"] += len(lst)
        for name, lst in glob.items():
            if lst:
                blocks[f"global:{name}"] += len(lst)
        stats["turns"] += len(live)

    return True, stats, blocks


def main():
    replays = find_edge_replays()
    print(f"Zoroark/Ditto edge replays found: {len(replays)}\n")
    total = Counter()
    agg_blocks: Counter = Counter()
    crashes = []
    ok_count = 0
    for path in replays:
        try:
            ok, stats, blocks = verify_one(path)
            ok_count += 1
            for k, v in stats.items():
                total[k] += v
            agg_blocks += blocks
        except Exception:
            crashes.append((path.name, traceback.format_exc().splitlines()[-1]))

    print(f"replays processed OK : {ok_count}/{len(replays)}")
    print(f"offline vectors      : {total['offline_vecs']}  (vod_parser + state_encoder, all (1398,) finite)")
    print(f"live vectors         : {total['live_vecs']}  (live_state_encoder, all (1398,) finite)")
    if crashes:
        print(f"\n!! CRASHES ({len(crashes)}):")
        for n, e in crashes:
            print(f"   {n}: {e}")
    else:
        print("\nNO CRASHES — all three modules ran clean on every edge replay.")

    print("\nResidual opp-side/global parity divergences by block (slot-turns):")
    if not agg_blocks:
        print("   NONE")
    for blk, cnt in agg_blocks.most_common():
        print(f"   {blk:24s} {cnt}")
    print("\n(Expected residuals only: global:*_side_sc [side-condition turns],"
          " pmon:boost [poke-env obs], pmon:moves [reveal-order/belief].)")


if __name__ == "__main__":
    main()
