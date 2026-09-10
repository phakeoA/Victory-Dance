"""Head-to-head TP eval: (v11 battle net + SBDA TP)  vs  (same battle net + legacy 46-dim TP).

MIRROR design — both players use the SAME 6-mon team and the SAME battle net, so the ONLY
difference is which 4-of-6 + lead order each TP net brings. SBDA's win-rate vs legacy isolates
team-preview quality (battle execution is identical). Run across several M-B teams (active regmb
format) and aggregate. >50% (with margin) => the SBDA TP picks better brings/leads in practice.
"""
from __future__ import annotations
import argparse, asyncio, math
from pathlib import Path

import v_dance  # noqa: F401  (sets the Windows SelectorEventLoop policy — poke-env POKE_LOOP fix)
from v_dance.play.run_local_battle import (make_player, start_showdown, stop_showdown,
                                           load_team, BATTLE_FORMAT)
from v_dance.play.parallel_battles import play_pairing, close_players

_REPO = Path(__file__).resolve().parents[2]          # v_dance/eval/ -> repo root (moved from scratch/ 2026-09-10)
assert (_REPO / "pyproject.toml").is_file(), _REPO
BATTLE_NET = _REPO / "ai_train_scripts" / "BC_model" / "checkpoints_attn" / "battle_selfplay_gen141.pt"
BATTLE_NET_B = BATTLE_NET   # side-B (legacy-TP) pilot; defaults to side-A. Set via --battle-net-b for a PAIR eval.
TP_SBDA    = _REPO / "ai_train_scripts" / "teamPreview_model" / "checkpoints_sbda" / "teampreview_sbda.pt"
TP_LEGACY  = _REPO / "ai_train_scripts" / "teamPreview_model" / "checkpoints_pre_sbda" / "teampreview_base.pt"
TEAMS_DIR  = _REPO / "teams" / "Champions" / "M-B"


async def run_eval(team_names, per_team: int, concurrency: int = 3) -> None:
    proc = start_showdown()
    sbda_won = legacy_won = total_fin = 0
    tp_sbda, tp_legacy = {}, {}
    print(f"[tp-eval] format={BATTLE_FORMAT} | A={BATTLE_NET.parent.name}+SBDA  vs  "
          f"B={BATTLE_NET_B.parent.name}+legacy | {len(team_names)} teams x {per_team} battles | "
          f"concurrency={concurrency}")
    try:
        for i, tname in enumerate(team_names):
            team_str = load_team(TEAMS_DIR / tname)
            pA = make_player(f"SBDA{i}", team_str, model_path=BATTLE_NET, team_chooser_path=TP_SBDA,
                             max_concurrent_battles=concurrency)
            pB = make_player(f"Legacy{i}", team_str, model_path=BATTLE_NET_B, team_chooser_path=TP_LEGACY,
                             max_concurrent_battles=concurrency)
            won, fin = await play_pairing(pA, pB, per_team, label=tname)   # pA(SBDA) wins / finished
            l_won = pB.n_won_battles
            sbda_won += won; legacy_won += l_won; total_fin += fin
            for d, p in ((tp_sbda, pA), (tp_legacy, pB)):
                for k, v in (getattr(p, "_tp_source", {}) or {}).items():
                    d[k] = d.get(k, 0) + v
            draws = fin - won - l_won
            print(f"  [{tname:22s}] SBDA {won:2d} | legacy {l_won:2d} | draw {draws:2d}  (of {fin})")
            await close_players(pA, pB)
    finally:
        stop_showdown(proc)

    print("\n" + "=" * 56)
    print(f"  HEAD-TO-HEAD TP EVAL  (battles finished: {total_fin})")
    draws = total_fin - sbda_won - legacy_won
    if total_fin:
        rate = sbda_won / total_fin
        # 95% binomial CI on SBDA win-rate (over decisive+draw games; draws counted as not-won)
        se = math.sqrt(rate * (1 - rate) / total_fin)
        lo, hi = rate - 1.96 * se, rate + 1.96 * se
        # decisive-only rate (ignore draws) — the cleaner head-to-head number
        dec = sbda_won + legacy_won
        drate = (sbda_won / dec) if dec else float("nan")
        print(f"  SBDA  TP wins : {sbda_won}  ({rate:.1%} of all; 95% CI [{lo:.1%}, {hi:.1%}])")
        print(f"  legacy TP wins: {legacy_won}  ({legacy_won/total_fin:.1%} of all)")
        print(f"  draws         : {draws}")
        print(f"  SBDA decisive win-rate (draws excluded): {drate:.1%} of {dec}")
        verdict = ("SBDA WINS MORE — swap candidate" if lo > 0.5 else
                   "legacy wins more" if hi < 0.5 else
                   "INCONCLUSIVE (CI spans 50%) — more games or a swap-design eval needed")
        print(f"  >>> VERDICT: {verdict}")
    print(f"  TP source (SBDA player) : {tp_sbda}")
    print(f"  TP source (legacy player): {tp_legacy}")
    print("  (want both ~all 'model' = the TP NET drove preview, not the heuristic fallback)")
    print("=" * 56)


def main():
    global BATTLE_NET, BATTLE_NET_B
    ap = argparse.ArgumentParser()
    ap.add_argument("--teams", type=int, default=8, help="number of M-B teams (mirror) to sweep")
    ap.add_argument("--per-team", type=int, default=25, help="battles per team")
    ap.add_argument("--smoke", action="store_true", help="tiny run: 1 team x 4 battles")
    ap.add_argument("--concurrency", type=int, default=3, help="parallel battles per pairing")
    ap.add_argument("--battle-net", default=str(BATTLE_NET),
                    help="side-A pilot (SBDA-TP side); default = BC anchor. Pass a self-play champion, "
                         "e.g. artifacts/self_play_archive/checkpoints/gen141.pt, to gate rl.2.")
    ap.add_argument("--battle-net-b", default=None,
                    help="side-B pilot (legacy-TP side); default = same as --battle-net. Set differently "
                         "for a PAIR eval, e.g. (gen141+SBDA) vs (BC-anchor+legacy).")
    args = ap.parse_args()
    BATTLE_NET = Path(args.battle_net)
    BATTLE_NET_B = Path(args.battle_net_b) if args.battle_net_b else BATTLE_NET
    names = sorted(p.name for p in TEAMS_DIR.iterdir() if p.is_file())
    if args.smoke:
        names, per, conc = names[:1], 4, 2
    else:
        names, per, conc = names[: args.teams], args.per_team, args.concurrency
    asyncio.run(run_eval(names, per, conc))


if __name__ == "__main__":
    main()
