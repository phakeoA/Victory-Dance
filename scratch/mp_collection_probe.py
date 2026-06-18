"""Throwaway throughput probe for task #14 (multicore collection).

Measures collection **games/min + system CPU%** as we spread the EXISTING per-battle
collection (``SelfPlayVGCPlayer`` model-vs-model — the real per-turn model inference +
state encoding that the GIL serialises) across N OS PROCESSES, all sharing ONE Showdown
server. This answers the load-bearing #14 question BEFORE building the real integration:
does multiprocessing beat the asyncio ceiling, or is the Node server / websocket latency
the wall?

NOT wired into anything — it's an isolated measurement. Outputs a table + a verdict, and
auto-cleans the server via the (now tree-killing) stop_showdown.

Memory note: each worker is CPU-only (``CUDA_VISIBLE_DEVICES=""``) with 1 torch thread,
~0.6 GB; default cap is 4 procs (~2.4 GB transient, released between configs).

Run:
    .venv/Scripts/python.exe scratch/mp_collection_probe.py --procs 1 2 4 --conc 3 --per 2
Cleanup afterwards:
    rm -rf artifacts/logs/task13/probe_rb
"""
from __future__ import annotations

import argparse
import statistics
import threading
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_RB = _REPO / "artifacts" / "logs" / "task13" / "probe_rb"

# Per-worker-process state: one ActorCritic loaded ONCE by the pool initializer, so the
# real timing excludes spawn + model-load.
_STATE: dict = {}


def _init_worker(ckpt: str) -> None:
    """Pool initializer — runs once per worker process at spawn."""
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""          # CPU-only worker (no VRAM, lighter)
    import logging
    logging.getLogger("poke_env").setLevel(logging.ERROR)
    logging.getLogger("websockets").setLevel(logging.ERROR)
    import torch
    torch.set_num_threads(1)                          # avoid N×all-cores oversubscription
    from v_dance.selfplay.actor_critic import ActorCritic
    _STATE["ac"] = ActorCritic.from_bc_checkpoint(ckpt, device="cpu")


def _noop(_):
    """Warm-up task — forces the worker to spawn + run the initializer before timing."""
    return None


def _collect(args):
    """One worker task: play ``conc`` concurrent model-vs-model pairings of ``per`` battles
    each against the shared server; return the finished-game count."""
    team_a, team_b, conc, per, tag = args
    import asyncio
    import logging
    import v_dance.play.run_local_battle as R
    from poke_env import AccountConfiguration
    from v_dance.play import parallel_battles as PB
    from v_dance.selfplay.game_runner import SelfPlayVGCPlayer

    ac = _STATE["ac"]
    ta = R.load_team(R.resolve_team_path(team_a))
    tb = R.load_team(R.resolve_team_path(team_b))
    _RB.mkdir(parents=True, exist_ok=True)
    done = {"n": 0}

    def _mk(side, i):
        team = ta if side == "A" else tb
        return SelfPlayVGCPlayer(
            ac, tau=1.0, sample_seed=tag * 1000 + i * 4 + (0 if side == "A" else 1),
            replay_path=_RB / f"p{tag}{side}{i}.jsonl",
            account_configuration=AccountConfiguration(f"PB{tag}{side}{i}", None),
            battle_format=R.BATTLE_FORMAT, team=team, max_concurrent_battles=1,
            log_level=logging.ERROR)

    async def one(i):
        p1, p2 = _mk("A", i), _mk("B", i)
        try:
            _w, f = await PB.play_pairing(p1, p2, per, battle_timeout=60, label=f"probe{tag}")
            done["n"] += f
        finally:
            await PB.close_players(p1, p2)

    asyncio.run(PB.run_jobs([lambda i=i: one(i) for i in range(conc)], workers=conc))
    return done["n"]


def main() -> int:
    ap = argparse.ArgumentParser(description="task #14 multicore collection throughput probe")
    ap.add_argument("--procs", type=int, nargs="+", default=[1, 2, 4],
                    help="process counts to sweep (default 1 2 4; keep <= ~6 for memory)")
    ap.add_argument("--conc", type=int, default=3, help="concurrent battles per worker (asyncio)")
    ap.add_argument("--per", type=int, default=2, help="battles per pairing")
    ap.add_argument("--ckpt", default=str(_REPO / "ai_train_scripts" / "BC_model"
                                          / "checkpoints_v4" / "bc_best.pt"))
    ap.add_argument("--teams", nargs=2, default=["team1", "WolfeGlick"])
    args = ap.parse_args()

    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    import psutil
    import v_dance.play.run_local_battle as R

    ctx = mp.get_context("spawn")
    print(f"== multicore collection probe ==  conc/worker={args.conc} per={args.per} "
          f"({args.conc * args.per} games/worker)  shared Showdown server")
    print(f"   NOTE: CPU% is SYSTEM-WIDE (includes your Chrome/VSCode ~8% baseline).")
    server = R.start_showdown()
    rows = []
    tag = 0
    try:
        for nproc in args.procs:
            ex = ProcessPoolExecutor(max_workers=nproc, mp_context=ctx,
                                     initializer=_init_worker, initargs=(args.ckpt,))
            try:
                list(ex.map(_noop, range(nproc)))     # force spawn + AC load (excluded from timing)
                samples: list = []
                stop = threading.Event()

                def _sampler(_s=samples, _stop=stop):
                    while not _stop.is_set():
                        _s.append(psutil.cpu_percent(interval=0.5))

                th = threading.Thread(target=_sampler, daemon=True)
                th.start()
                t0 = time.perf_counter()
                futs = []
                for _ in range(nproc):
                    tag += 1
                    futs.append(ex.submit(_collect, (args.teams[0], args.teams[1],
                                                     args.conc, args.per, tag)))
                games = sum(f.result() for f in futs)
                dt = time.perf_counter() - t0
                stop.set()
                th.join()
            finally:
                ex.shutdown(wait=True)
            gpm = games / dt * 60 if dt > 0 else float("nan")
            cpu_avg = statistics.mean(samples) if samples else 0.0
            cpu_peak = max(samples) if samples else 0.0
            rows.append((nproc, games, dt, gpm, cpu_avg, cpu_peak))
            print(f"   procs={nproc:>2}  games={games:>3}  {dt:5.1f}s  {gpm:6.1f} games/min  "
                  f"CPU avg {cpu_avg:4.0f}% peak {cpu_peak:4.0f}%")
    finally:
        R.stop_showdown(server)

    if len(rows) >= 2:
        base, top = rows[0][3], rows[-1][3]
        speedup = top / base if base else float("nan")
        print(f"\n   speedup {rows[0][0]}->{rows[-1][0]} procs: {speedup:.2f}x   "
              f"(CPU {rows[0][4]:.0f}% -> {rows[-1][4]:.0f}%)")
        if speedup >= 1.6:
            print("   VERDICT: multiprocessing SCALES -> build the real integration (#14b).")
        elif speedup >= 1.2:
            print("   VERDICT: PARTIAL scaling -> some win; the Node server may be the next "
                  "wall (try N servers on N ports).")
        else:
            print("   VERDICT: little/no gain -> server/latency-bound; multiprocessing won't "
                  "help much (stay on asyncio, or add Showdown servers).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
