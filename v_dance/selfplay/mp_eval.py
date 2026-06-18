"""Multiprocessing EVAL — the gauntlet across worker processes (task #19, Fix A).

Once collection is multiprocessed (#14), the per-gen gauntlet (scripted ladder + the 360-game
prev_best mirror, ~540 games) becomes the bottleneck — it's the same GIL-bound asyncio shape the
old collection had. This module fans the gauntlet's independent (opponent × team-pairing) battles
across the SAME persistent worker pool, so eval gets the same ~2x.

It's SIMPLER than collection: the candidate is already a checkpoint FILE on disk (``gen{N}.pt``,
saved before eval) and ``prev_best`` is the champion's path, so there's NO weight handoff; workers
just play their slice and TALLY ``(wins, finished)`` per opponent — no critic, no trajectories, no
league/PFSP. Mirrors ``gauntlet.run_gauntlet``'s per-descriptor logic exactly so the result is the
same ``{kind: (wins, finished)}``.

The plan (``build_eval_specs``) and the dispatch (``_eval_specs`` with an injected player factory)
are offline-tested; only ``eval_worker`` / ``_build_eval_players_real`` touch poke-env. Top level
stays torch-/poke-env-free."""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Callable, List, Optional

from v_dance.play.parallel_battles import close_players, play_pairing, run_jobs
from v_dance.selfplay.mp_collect import partition_specs   # round-robin split (generic over any list)

log = logging.getLogger(__name__)


@dataclass
class EvalSpec:
    """One gauntlet chunk: ``n`` battles of the candidate vs ``kind`` on the ``team_a`` vs
    ``team_b`` pairing. ``uid`` is globally unique so worker account names never collide."""
    kind: str
    team_a: str
    team_b: str
    n: int
    uid: int


def build_eval_specs(opponents, team_pool, battles_per_opponent: int, *, matchup_seed: int = 0,
                     mirror_battles: Optional[int] = None) -> List[EvalSpec]:
    """Plan the gauntlet as flat picklable specs — the same descriptor set ``run_gauntlet`` builds
    (the prev_best mirror gets ``mirror_battles`` instead of the scripted count). Torch-free."""
    from v_dance.eval.gauntlet import team_matchups
    specs: List[EvalSpec] = []
    uid = 0
    for kind in opponents:
        nb = mirror_battles if (kind == "prev_best" and mirror_battles) else battles_per_opponent
        for team_a, team_b, n in team_matchups(team_pool, nb, seed=matchup_seed):
            uid += 1
            specs.append(EvalSpec(kind, team_a, team_b, n, uid))
    return specs


def merge_eval_results(results):
    """Fold per-worker ``{"acc": {kind:(w,f)}, "source": {...}}`` dicts into
    ``({kind: (wins, finished)}, source_counts)``. ``None`` (a dead worker) is skipped."""
    acc: dict = {}
    source: Counter = Counter()
    for r in results:
        if r is None:
            continue
        for k, (w, f) in r.get("acc", {}).items():
            a = acc.setdefault(k, [0, 0])
            a[0] += w
            a[1] += f
        source.update(r.get("source", {}))
    return {k: (v[0], v[1]) for k, v in acc.items()}, dict(source)


# ── dispatch (injected player factory → offline-testable) ──────────────────────
async def _eval_specs(candidate, prev_best, team_chooser, specs: List[EvalSpec], *,
                      battle_timeout: Optional[float], async_workers: int,
                      build_players: Callable, live_dir=None, save_replays: bool = False):
    """Play every EvalSpec (bounded to ``async_workers`` concurrent battles in this process) and
    tally ``(wins, finished)`` per opponent kind + the decision-source counts. A build error skips
    THAT chunk (not the whole worker batch)."""
    acc: dict = {}
    source: Counter = Counter()

    async def _run(spec: EvalSpec):
        acc.setdefault(spec.kind, [0, 0])
        try:
            model_player, opp = build_players(candidate, prev_best, team_chooser, spec,
                                              live_dir, save_replays)
        except Exception:
            log.warning("mp eval player-build failed (vs %s) — skipping chunk.",
                        spec.kind, exc_info=True)
            return
        try:
            w, f = await play_pairing(model_player, opp, spec.n,
                                      battle_timeout=battle_timeout, label=spec.kind)
            acc[spec.kind][0] += w
            acc[spec.kind][1] += f
        finally:
            source.update(getattr(model_player, "_source_counts", {}) or {})
            for k, v in (getattr(model_player, "_tp_source", {}) or {}).items():
                source[f"tp_{k}"] += v                       # team-preview tally (#4)
            await close_players(model_player, opp)

    await run_jobs([lambda s=s: _run(s) for s in specs], workers=async_workers)
    return {"acc": {k: (v[0], v[1]) for k, v in acc.items()}, "source": dict(source)}


def _build_eval_players_real(candidate, prev_best, team_chooser, spec: EvalSpec,
                             live_dir=None, save_replays: bool = False):
    """The real per-kind eval players (poke-env). Mirrors ``run_gauntlet``'s ``_run`` exactly:
    a model player on the candidate vs a scripted/prev_best opponent, account names keyed to the
    global uid so they're unique across worker processes. ``live_dir`` (#18b) makes the model
    player publish its eval match to the spectate feed."""
    import v_dance.play.run_local_battle as R
    from v_dance.eval.gauntlet import _make_opponent
    uid = spec.uid
    model_team = R.load_team(R.resolve_team_path(spec.team_a))
    opp_team = R.load_team(R.resolve_team_path(spec.team_b))
    model_player = R.make_player(f"BC{uid}", model_team, model_path=candidate,
                                 team_chooser_path=team_chooser,
                                 live_dir=live_dir, save_replays=save_replays)
    opp = _make_opponent(spec.kind, f"OP{spec.kind[:4]}{uid}", opp_team,
                         model_path=prev_best, team_chooser_path=team_chooser)
    return model_player, opp


# ── stateless spawn-worker entry ───────────────────────────────────────────────
def eval_worker(payload):
    """Spawn-worker entry: play the assigned EvalSpecs against the SHARED server and return the
    picklable ``{"acc": {kind:(w,f)}, "source": {...}}`` tally. ``payload`` =
    ``(candidate, prev_best, team_chooser, specs, battle_timeout, async_workers)`` — all picklable
    (paths + EvalSpecs). The candidate / prev_best load straight from disk (no weight handoff)."""
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""          # CPU-only worker
    import asyncio
    import logging as _logging
    import torch
    torch.set_num_threads(1)
    _logging.getLogger("poke_env").setLevel(_logging.ERROR)
    _logging.getLogger("websockets").setLevel(_logging.ERROR)
    candidate, prev_best, team_chooser, specs, battle_timeout, async_workers, live_dir, save_replays = payload
    return asyncio.run(_eval_specs(
        candidate, prev_best, team_chooser, specs, battle_timeout=battle_timeout,
        async_workers=async_workers, build_players=_build_eval_players_real,
        live_dir=live_dir, save_replays=save_replays))


# ── main-process orchestration ─────────────────────────────────────────────────
def eval_with_pool(candidate, *, opponents, team_pool, battles_per_opponent: int, team_chooser,
                   submit_fn: Callable, prev_best=None, mirror_battles: Optional[int] = None,
                   matchup_seed: int = 0, battle_timeout: Optional[float] = 90.0,
                   n_procs: int = 4, async_per_proc: int = 3,
                   live_dir=None, save_replays: bool = False):
    """Multiprocess analogue of ``gauntlet.run_gauntlet`` (task #19). Pre-validates the candidate
    (+ prev_best) load LOUDLY in main, plans + partitions the gauntlet descriptors, ships them to
    the pool via the injected ``submit_fn`` (``pool.submit(payloads, worker_fn=eval_worker)``), and
    merges the tallies. Returns ``({kind: (wins, finished)}, source_counts)`` — the caller computes
    ``model_elo``. ``submit_fn`` is injected so this is offline-tested without a pool/server."""
    import v_dance.play.model_io as model_io
    model_io.load_bc_policy(str(candidate))          # fail LOUD before fanning out (3c.3b lesson)
    if prev_best:
        model_io.load_bc_policy(str(prev_best))
    specs = build_eval_specs(opponents, team_pool, battles_per_opponent,
                             matchup_seed=matchup_seed, mirror_battles=mirror_battles)
    batches = partition_specs(specs, n_procs)
    pb = str(prev_best) if prev_best else None
    tc = str(team_chooser) if team_chooser else None
    ld = str(live_dir) if live_dir else None
    payloads = [(str(candidate), pb, tc, batch, battle_timeout, async_per_proc, ld,
                 bool(save_replays)) for batch in batches]
    results, source = merge_eval_results(submit_fn(payloads, worker_fn=eval_worker))
    for kind in opponents:
        results.setdefault(kind, (0, 0))             # a fully-dropped opponent reads 0/0, not absent
    return results, source
