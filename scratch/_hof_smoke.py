"""P2.3 bounded live smoke for hof_eval — play a CANDIDATE vs the archived gen snapshots as HoF
suspects, then run the breadth veto. Confirms the model-vs-model per-suspect plumbing works end
to end on a real Showdown server. Bounded by --games (tiny) + the per-battle watchdog.

Run (repo root):
  .venv/Scripts/python.exe scratch/_hof_smoke.py --games 4 --workers 4
"""
from __future__ import annotations

import argparse
import asyncio
import glob
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate",
                    default=str(_REPO / "ai_train_scripts/BC_model/checkpoints_v4/bc_best.pt"))
    ap.add_argument("--archive", default=str(_REPO / "artifacts/self_play_archive"))
    ap.add_argument("--games", type=int, default=4, help="battles vs EACH suspect (tiny for a smoke)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--teams", nargs="+",
                    default=["team1", "WolfeGlick", "Kronomono1", "Kronomono3"])
    ap.add_argument("--team-chooser",
                    default=str(_REPO / "ai_train_scripts/teamPreview_model/checkpoints/teampreview_best.pt"))
    args = ap.parse_args()

    from v_dance.selfplay.generation import hof_eval
    from v_dance.selfplay.gate import hall_of_fame_gate, HoFConfig
    from v_dance.selfplay.league import LeagueSnapshot

    paths = sorted(glob.glob(str(Path(args.archive) / "gen*.pt")))
    suspects = [LeagueSnapshot(Path(p).stem, p, i) for i, p in enumerate(paths)]
    print(f"[hof-smoke] candidate={Path(args.candidate).name}  "
          f"suspects={[s.snapshot_id for s in suspects]}", flush=True)
    if not suspects:
        print("[hof-smoke] no gen*.pt in the archive — run a generation smoke first.", flush=True)
        return

    results = asyncio.run(hof_eval(
        args.candidate, suspects, team_pool=args.teams, team_chooser=args.team_chooser,
        games_per_snapshot=args.games, n_workers=args.workers, manage_server=True))
    print(f"[hof-smoke] per-suspect (id, wins, games): {results}", flush=True)

    # min_pool=1 / min_games=1 so even a tiny smoke renders a verdict (production uses 4 / 40).
    v, st = hall_of_fame_gate(results, cfg=HoFConfig(min_pool=1, min_games_per_snap=1))
    wu = st.get("worst_upper")
    print(f"[hof-smoke] HoF verdict={v} reason={st.get('reason')} "
          f"worst={st.get('worst_snapshot_id')} worst_upper="
          f"{wu:.3f}" if wu is not None else f"[hof-smoke] HoF verdict={v} reason={st.get('reason')}",
          flush=True)
    for r in st.get("suspects", []):
        print(f"    {r['snapshot_id']}: {r['wins']}/{r['games']} "
              f"upper={r['wilson_upper']:.3f} vetoed={r['vetoed']}", flush=True)


if __name__ == "__main__":
    main()
