"""Absolute-skill probe: the production BC checkpoint vs a RANDOM-legal player,
mirror team.  How much better than random is the behavior-cloned policy?

    python local_battle/_vs_random.py [n] [team]
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

import run_local_battle as R  # noqa: E402

logging.basicConfig(level=logging.WARNING)
BASE = _REPO / "ai_train_scripts" / "BC_model" / "checkpoints" / "bc_best.pt"
TC = _REPO / "ai_train_scripts" / "teamPreview_model" / "checkpoints" / "teampreview_best.pt"


async def main(n: int, team_name: str) -> None:
    proc = R.start_showdown()
    team = R.load_team(R.resolve_team_path(team_name))
    bc = R.make_player("BcModel", team, model_path=BASE, team_chooser_path=TC)
    rnd = R.make_player("Random", team, model_path=None)   # model_path=None → RandomVGCPlayer
    try:
        await bc.battle_against(rnd, n_battles=n)
    finally:
        await bc.ps_client.stop_listening()
        await rnd.ps_client.stop_listening()
        bc.close()
        rnd.close()
        R.stop_showdown(proc)
    total = bc.n_finished_battles
    print("\n" + "=" * 50)
    print(f"  BC MODEL vs RANDOM  ({total} battles, mirror '{team_name}')")
    print(f"  BC model wins : {bc.n_won_battles}  ({bc.n_won_battles/max(total,1)*100:.1f} %)")
    print(f"  Random  wins  : {rnd.n_won_battles}  ({rnd.n_won_battles/max(total,1)*100:.1f} %)")
    print("=" * 50)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    team = sys.argv[2] if len(sys.argv) > 2 else "team1"
    asyncio.run(main(n, team))
