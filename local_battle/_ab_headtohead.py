"""Behavioral A/B: AUX opponent-head checkpoint vs the BASELINE checkpoint,
head-to-head on a MIRROR team (so the only difference is the in-battle policy).

    python local_battle/_ab_headtohead.py [n_battles] [team_name]

Reuses run_local_battle's server/player machinery; reuses an already-running
Showdown server if port 8000 is open (else starts+stops one).
"""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
for _p in (str(_REPO / "data" / "scripts"), str(_REPO), str(_HERE)):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import run_local_battle as R  # noqa: E402  (server + make_player + team helpers)

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("ab")
log.setLevel(logging.INFO)

_CK = _REPO / "ai_train_scripts" / "BC_model" / "checkpoints"
AUX = _CK / "bc_aux_opp_EXPERIMENT.pt"
BASE = _CK / "bc_best.pt"
TC = _REPO / "ai_train_scripts" / "teamPreview_model" / "checkpoints" / "teampreview_best.pt"


async def main(n: int, team_name: str, swap: bool = False) -> None:
    proc = R.start_showdown()                 # None if a server is already running
    team_path = R.resolve_team_path(team_name)
    team = R.load_team(team_path)
    # ``swap`` puts AUX on the Blue seat instead of Red, to cancel any side bias.
    red_ck, blue_ck = (BASE, AUX) if swap else (AUX, BASE)
    log.info("RED=%s  BLUE=%s  mirror-team=%s  n=%d  swap=%s",
             red_ck.name, blue_ck.name, team_name, n, swap)

    p_red = R.make_player("AbRed", team, model_path=red_ck, team_chooser_path=TC)
    p_blue = R.make_player("AbBlue", team, model_path=blue_ck, team_chooser_path=TC)
    aux = p_blue if swap else p_red
    base = p_red if swap else p_blue
    try:
        await aux.battle_against(base, n_battles=n)
    finally:
        await aux.ps_client.stop_listening()
        await base.ps_client.stop_listening()
        aux.close()
        base.close()
        R.stop_showdown(proc)

    total = aux.n_finished_battles
    aw, bw = aux.n_won_battles, base.n_won_battles
    print("\n" + "=" * 56)
    print(f"  HEAD-TO-HEAD  AUX vs BASELINE  ({total} battles, mirror '{team_name}')")
    print("=" * 56)
    if total:
        print(f"  AUX  (bc_aux_opp_EXPERIMENT) wins : {aw}  ({aw/total*100:.1f} %)")
        print(f"  BASE (bc_best)               wins : {bw}  ({bw/total*100:.1f} %)")
        print(f"  draws                              : {total - aw - bw}")
    print("─" * 56)
    print(f"  AUX  decisions by source : {dict(getattr(aux, '_source_counts', {}))}")
    print(f"  BASE decisions by source : {dict(getattr(base, '_source_counts', {}))}")
    print("  (want mostly 'model'; 'retry'/'model_error'/'no_model' = a problem)")
    print("=" * 56)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    team = sys.argv[2] if len(sys.argv) > 2 else "team1"
    swap = len(sys.argv) > 3 and sys.argv[3].lower() in ("swap", "1", "true")
    asyncio.run(main(n, team, swap))
