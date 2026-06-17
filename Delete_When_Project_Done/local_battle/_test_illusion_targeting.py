"""Behavioral test of the #15 deliberate illusion-targeting fix.

Runs a Kronomono3 MIRROR (both sides bring Zoroark-Hisui, so a same-species
illusion routinely makes poke-env merge two foes and lose a target slot) with
BOTH players model-driven.  Measures:
  * how many times the gap-#6 reconstruction let the codec DELIBERATELY aim a
    foe slot poke-env had lost (vgc_base._ILLUSION_DELIBERATE_TARGETS), and
  * that Showdown rejected 0 orders ('retry'/'model_error'/'no_model' = a bug).

    python local_battle/_test_illusion_targeting.py [n_battles] [team]
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
import vgc_base                # noqa: E402  (holds the deliberate-target counter)

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("illusion")
log.setLevel(logging.INFO)

CK = _REPO / "ai_train_scripts" / "BC_model" / "checkpoints" / "bc_best.pt"
TC = _REPO / "ai_train_scripts" / "teamPreview_model" / "checkpoints" / "teampreview_best.pt"


def _bad(counts: dict) -> int:
    """Decisions that indicate a Showdown rejection / policy failure."""
    return sum(counts.get(k, 0) for k in ("retry", "model_error", "no_model"))


async def main(n: int, team_name: str) -> None:
    proc = R.start_showdown()
    team = R.load_team(R.resolve_team_path(team_name))
    vgc_base._ILLUSION_DELIBERATE_TARGETS = 0     # reset the counter for this run
    log.info("Kronomono3 illusion targeting test: both AIs, mirror '%s', n=%d", team_name, n)

    p1 = R.make_player("IllRed", team, model_path=CK, team_chooser_path=TC)
    p2 = R.make_player("IllBlue", team, model_path=CK, team_chooser_path=TC)
    try:
        await p1.battle_against(p2, n_battles=n)
    finally:
        await p1.ps_client.stop_listening()
        await p2.ps_client.stop_listening()
        p1.close()
        p2.close()
        R.stop_showdown(proc)

    total = p1.n_finished_battles
    c1 = dict(getattr(p1, "_source_counts", {}))
    c2 = dict(getattr(p2, "_source_counts", {}))
    delib = vgc_base._ILLUSION_DELIBERATE_TARGETS
    bad = _bad(c1) + _bad(c2)

    print("\n" + "=" * 60)
    print(f"  #15 ILLUSION-TARGETING TEST  ({total} battles, mirror '{team_name}')")
    print("=" * 60)
    print(f"  deliberate opp_a/opp_b targets via reconstruction : {delib}")
    print(f"  Showdown rejections / policy failures (must be 0)  : {bad}")
    print("─" * 60)
    print(f"  IllRed  decisions by source : {c1}")
    print(f"  IllBlue decisions by source : {c2}")
    print("─" * 60)
    verdict = "PASS" if bad == 0 else "FAIL (rejections occurred)"
    if delib == 0:
        verdict += "  (note: no same-species-merge window occurred this run — "\
                   "fix correct but untriggered; try more battles)"
    print(f"  VERDICT: {verdict}")
    print("=" * 60)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    team = sys.argv[2] if len(sys.argv) > 2 else "Kronomono3"
    asyncio.run(main(n, team))
