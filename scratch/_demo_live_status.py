"""Dev demo: SIMULATE a self-play run by writing status.json + a growing
manifest.json, so the live dashboard (3c.6g) can be watched WITHOUT a real run.

It does NOT touch Showdown — the Spectate iframes will be blank (fake battle
tags / no local server), but the LIVE badge, the in-generation progress bar, the
running win-rate, and the Overview/Health charts all update in real time, which
is the point: it exercises the dashboard's polling end-to-end.

Run it alongside the dashboard server, then open the dashboard:

    # terminal 1
    .venv/Scripts/python.exe -m v_dance.datatools.dashboard_server --port 5175
    # terminal 2
    .venv/Scripts/python.exe scratch/_demo_live_status.py
    # browser
    http://127.0.0.1:5175/      (watch the LIVE bar + charts move)
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

from v_dance.selfplay.archive import write_manifest
from v_dance.selfplay.generation import GenerationHistory, GenerationRecord
from v_dance.selfplay.status import LiveStatus

_REPO = Path(__file__).resolve().parents[1]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Simulate a live self-play run for the dashboard")
    ap.add_argument("--archive", default=str(_REPO / "artifacts" / "self_play_archive"))
    ap.add_argument("--gens", type=int, default=4)
    ap.add_argument("--games", type=int, default=40, help="collection games per generation")
    ap.add_argument("--speed", type=float, default=0.25, help="seconds per simulated game")
    args = ap.parse_args(argv)

    arch = Path(args.archive)
    status = LiveStatus(arch / "status.json")
    history = GenerationHistory()
    status.start_run(args.gens)
    wr = 0.40
    print(f"[demo] simulating {args.gens} generations -> {arch}/status.json + manifest.json")

    for g in range(args.gens):
        status.phase("collecting", generation=g, games_total=args.games)
        tag_a = f"battle-gen9championsvgc2026regma-{1000 + g * 2}"
        tag_b = f"battle-gen9championsvgc2026regma-{1001 + g * 2}"
        wins = 0
        for i in range(1, args.games + 1):
            wins += 1 if random.random() < (0.45 + 0.03 * g) else 0
            status.games(i, wins / i)
            if i % 4 == 0:                              # battle rooms rotate as games run
                status.set_active_battles([
                    {"tag": tag_a, "p1": "SP1", "p2": "SP2", "turn": i // 3},
                    {"tag": tag_b, "p1": "SP1", "p2": "SP2", "turn": i // 4},
                ])
            time.sleep(args.speed)

        status.phase("updating", generation=g)
        loss = max(0.30, 0.55 - 0.03 * g)
        halted = (g == 2)                              # simulate one collapse-guard trip
        us = {"loss": loss, "kl_to_bc": 0.012 if halted else 0.003 + 0.0006 * g,
              "explained_variance": 0.55 if halted else min(0.92, 0.80 + 0.02 * g),
              "clip_fraction": 0.09 if halted else 0.03, "entropy": max(0.85, 1.25 - 0.04 * g),
              "halted": halted}
        status.set_update(us)
        time.sleep(1.0)

        status.phase("evaluating", generation=g)
        time.sleep(1.0)

        # commit the generation to the manifest the dashboard charts read
        wr = wins / args.games
        elo = 980 + g * 45 + random.randint(-12, 12)
        verdict = "revert" if halted else "promote"
        promoted = not halted
        history.add(GenerationRecord(g, args.games, round(wr * 100), 100, float(elo),
                                     verdict, promoted, update_stats=us))
        if promoted:
            history.best_path = f"gen{g}.pt"
            history.best_scripted = (round(wr * 100), 100)
        write_manifest(arch, history)
        status.set_update(us, last_verdict=verdict)
        status.phase("idle", generation=g)
        print(f"[demo] gen {g}: wr {wr:.2f} elo {elo} {verdict}{' (halted)' if halted else ''}")

    status.finish_run()
    print("[demo] run finished (status.live=false). Reload the dashboard to see the final state.")


if __name__ == "__main__":
    main()
