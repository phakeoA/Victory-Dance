"""Fresh, PARALLEL head-to-head re-test of candidate checkpoints vs one opponent (2026-10-01).

Picking the best generation by its in-run panel score carries selection luck (the winner's curse); this plays each
candidate a FRESH sample (a new matchup seed; the SAME pairings for every candidate, so they compare paired) against
the opponent, fanned across a worker-process pool + a Showdown server pool — the self-play eval machinery
(``mp_eval.eval_with_pool``'s panel path), so 400 games take minutes, not the single-process gauntlet's ~20 min.

Two team shapes: ``random`` (both sides random pool teams = general skill) and ``own`` (the candidate always plays
``--own-team`` vs random opponent teams = the LADDER shape). Put the opponent itself in ``--candidates`` as the
control: in ``random`` it must read ~50 % (a tool-bias check), in ``own`` it is the baseline the others must beat.

    .venv/Scripts/python.exe -X utf8 -m v_dance.eval.panel_eval --opponent era2=<ckpt> \\
        --candidates era2=<ckpt> g42=<ckpt> ... --battles 400 --modes random own --own-team Baltimore_Sand_Psy
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]


def wilson(w: int, n: int, z: float = 1.96):
    if n <= 0:
        return (0.0, 1.0)
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (c - h, c + h)


def main(argv=None) -> int:
    from v_dance.selfplay.generation import parse_panel
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", nargs="+", required=True, metavar="NAME=CKPT")
    ap.add_argument("--opponent", required=True, metavar="NAME=CKPT")
    ap.add_argument("--battles", type=int, default=400, help="games per candidate per mode")
    ap.add_argument("--modes", nargs="+", default=["random"], choices=["random", "own"])
    ap.add_argument("--own-team", default=None, help="the candidate's team in mode 'own' (the ladder team)")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--async-per-proc", type=int, default=3)
    ap.add_argument("--servers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1000, help="matchup seed (self-play runs use 0 + generation)")
    ap.add_argument("--out", default=str(_REPO / "scratch" / "panel_eval.json"))
    a = ap.parse_args(argv)
    if "own" in a.modes and not a.own_team:
        ap.error("--modes own needs --own-team")
    cands, opp = parse_panel(a.candidates), parse_panel([a.opponent])

    import v_dance.play.run_local_battle as R
    from v_dance.formats import default_format
    from v_dance.play.model_io import DEFAULT_TP_CHECKPOINT
    from v_dance.selfplay import mp_collect as MP
    from v_dance.selfplay import mp_eval as ME
    teams = sorted(R.discover_teams(reg=default_format()))
    (oname, _), = opp.items()
    print(f"[panel_eval] {len(cands)} candidate(s) x {a.modes} x {a.battles} games vs {oname}; {len(teams)} "
          f"{default_format()} teams; {a.procs} procs x {a.async_per_proc} async, {a.servers} server(s); seed {a.seed}")
    pool = MP.CollectionPool(a.procs)
    servers = R.ServerPool(max(1, a.servers), manage=True).start_all()
    ports = servers.ports if a.servers > 1 else None
    rows = []
    try:
        for mode in a.modes:
            for i, (name, ck) in enumerate(cands.items()):
                t0 = time.perf_counter()
                res, _src = ME.eval_with_pool(
                    ck, opponents=[], team_pool=teams, battles_per_opponent=0, team_chooser=str(DEFAULT_TP_CHECKPOINT),
                    submit_fn=pool.submit, n_procs=a.procs, async_per_proc=a.async_per_proc,
                    generation=900 + 10 * a.modes.index(mode) + i, ports=ports, matchup_seed=a.seed,
                    panel=opp, panel_battles=a.battles, panel_team_pool=teams,
                    panel_own_team=(a.own_team if mode == "own" else None))
                w, f = res.get(ME.PANEL_PREFIX + oname, (0, 0))
                lo, hi = wilson(w, f)
                dt = time.perf_counter() - t0
                rows.append({"mode": mode, "candidate": name, "ckpt": ck, "wins": w, "games": f,
                             "rate": (w / f) if f else None, "ci95": [lo, hi], "secs": round(dt, 1)})
                print(f"  [{mode:6s}] {name:10s} vs {oname}: {w:4d}/{f:<4d} = {w / max(1, f) * 100:5.1f}%  "
                      f"95% CI [{lo * 100:4.1f}, {hi * 100:4.1f}]  ({f / dt * 60 if dt else 0:.0f} games/min)", flush=True)
    finally:
        pool.close()
        servers.stop_all()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"opponent": opp, "seed": a.seed, "rows": rows}, indent=2), encoding="utf-8")
    print(f"[panel_eval] -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
