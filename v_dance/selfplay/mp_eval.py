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
from pathlib import Path
from typing import Callable, List, Optional

from v_dance.play.parallel_battles import (close_players, eval_account_names, gen_salt,
                                           play_pairing, run_jobs)
from v_dance.selfplay.mp_collect import partition_specs   # round-robin split (generic over any list)

log = logging.getLogger(__name__)


@dataclass
class EvalSpec:
    """One gauntlet chunk: ``n`` battles of the candidate vs ``kind`` on the ``team_a`` vs
    ``team_b`` pairing. ``uid`` is globally unique within the gen so worker account names never
    collide across processes; ``gen`` (22d) folds the GENERATION into those names so they don't
    collide across gens either (a stale challenge from one gen vs the next gen's reuse)."""
    kind: str
    team_a: str
    team_b: str
    n: int
    uid: int
    gen: int = 0


def build_eval_specs(opponents, team_pool, battles_per_opponent: int, *, matchup_seed: int = 0,
                     mirror_battles: Optional[int] = None, gen: int = 0) -> List[EvalSpec]:
    """Plan the gauntlet as flat picklable specs — the same descriptor set ``run_gauntlet`` builds
    (the prev_best mirror gets ``mirror_battles`` instead of the scripted count). ``gen`` (22d) is
    stamped onto every spec so the worker account names fold in the generation. Torch-free."""
    from v_dance.eval.gauntlet import team_matchups
    specs: List[EvalSpec] = []
    uid = 0
    for kind in opponents:
        nb = mirror_battles if (kind == "prev_best" and mirror_battles) else battles_per_opponent
        for team_a, team_b, n in team_matchups(team_pool, nb, seed=matchup_seed):
            uid += 1
            specs.append(EvalSpec(kind, team_a, team_b, n, uid, gen))
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
                      build_players: Callable, live_dir=None, save_replays: bool = False,
                      port: Optional[int] = None):
    """Play every EvalSpec (bounded to ``async_workers`` concurrent battles in this process) and
    tally ``(wins, finished)`` per opponent kind + the decision-source counts. A build error skips
    THAT chunk (not the whole worker batch)."""
    acc: dict = {}
    source: Counter = Counter()

    async def _run(spec: EvalSpec):
        acc.setdefault(spec.kind, [0, 0])
        try:
            model_player, opp = build_players(candidate, prev_best, team_chooser, spec,
                                              live_dir, save_replays, port=port)  # 22f: pool server
        except Exception:
            log.warning("mp eval player-build failed (vs %s) — skipping chunk.",
                        spec.kind, exc_info=True)
            return
        try:
            w, f = await play_pairing(model_player, opp, spec.n,
                                      battle_timeout=battle_timeout, label=spec.kind)
            # A backstop-forfeit (forced-switch loop-guard → ForfeitBattleOrder) is an engineering
            # escape, NOT a skill result, yet poke-env scores it as a finished WIN for the other side.
            # Discard such battles from BOTH numerator and denominator so they don't bias the gate: an
            # OPP (champion) forfeit is otherwise a spurious candidate WIN — a ONE-DIRECTIONAL push
            # toward PROMOTE that poisons the frozen-champion lineage — and a CANDIDATE forfeit a
            # spurious loss. Collection drops these via drop_fallback_pairs (mp_collect ~170); eval has
            # no trajectory, so adjust the (w, f) tally here. `_forfeited_tags` is fresh per chunk
            # (players are rebuilt per spec), deduped one-entry-per-battle (live_vgc_base ~440), and the
            # two sides' sets are battle-disjoint (the first forfeit ends the battle). `_forfeited_tags`
            # is tagged at DECISION time (when ForfeitBattleOrder is returned), while `w` only counts a
            # SERVER-ACKED win, so in the rare stall-abandon window (an opp forfeit issued but wedged, then
            # folded into the FINISHED delta as a non-win by the watchdog) opp_ff can exceed the reflected
            # wins. Clamp non-negative so one such chunk can never push a negative tally into the gate
            # (the f-side already nets to real_finished >= 0; the clamp there is belt-and-suspenders).
            opp_ff = len(getattr(opp, "_forfeited_tags", None) or ())
            own_ff = len(getattr(model_player, "_forfeited_tags", None) or ())
            acc[spec.kind][0] += max(0, w - opp_ff)
            acc[spec.kind][1] += max(0, f - opp_ff - own_ff)
        finally:
            source.update(getattr(model_player, "_source_counts", {}) or {})
            for k, v in (getattr(model_player, "_tp_source", {}) or {}).items():
                source[f"tp_{k}"] += v                       # team-preview tally (#4)
            await close_players(model_player, opp)

    await run_jobs([lambda s=s: _run(s) for s in specs], workers=async_workers)
    return {"acc": {k: (v[0], v[1]) for k, v in acc.items()}, "source": dict(source)}


def _build_eval_players_real(candidate, prev_best, team_chooser, spec: EvalSpec,
                             live_dir=None, save_replays: bool = False, port: Optional[int] = None):
    """The real per-kind eval players (poke-env). Mirrors ``run_gauntlet``'s ``_run`` exactly:
    a model player on the candidate vs a scripted/prev_best opponent, account names keyed to the
    global uid AND ``spec.gen`` (22d) so they're unique across worker processes AND across
    generations. ``live_dir`` (#18b) makes the model player publish its eval match to the
    spectate feed."""
    import v_dance.play.run_local_battle as R
    from v_dance.eval.gauntlet import _make_opponent, eval_replay_routing, _ckpt_gen
    uid = spec.uid
    model_team = R.load_team(R.resolve_team_path(spec.team_a))
    opp_team = R.load_team(R.resolve_team_path(spec.team_b))
    model_name, opp_name = eval_account_names(spec.kind, uid, salt=gen_salt(spec.gen))
    # task E: file the saved replay under eval/<kind>/ (scripted) or eval/league/ (gen-vs-gen),
    # named gen<N>_vs_<kind|genM>; the live spectate JSON stays flat in live_dir (dashboard).
    subdir, label = eval_replay_routing(spec.kind, _ckpt_gen(candidate),
                                        opp_ref=(prev_best if spec.kind == "prev_best" else None))
    rdir = str(Path(live_dir) / subdir) if (live_dir and save_replays) else None
    model_player = R.make_player(model_name, model_team, model_path=candidate,
                                 team_chooser_path=team_chooser,
                                 live_dir=live_dir, save_replays=save_replays,
                                 replay_dir=rdir, replay_label=label, port=port)
    opp = _make_opponent(spec.kind, opp_name, opp_team,
                         model_path=prev_best, team_chooser_path=team_chooser, port=port)
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
    candidate, prev_best, team_chooser, specs, battle_timeout, async_workers, live_dir, save_replays, port = payload
    return asyncio.run(_eval_specs(
        candidate, prev_best, team_chooser, specs, battle_timeout=battle_timeout,
        async_workers=async_workers, build_players=_build_eval_players_real,
        live_dir=live_dir, save_replays=save_replays, port=port))


# ── main-process orchestration ─────────────────────────────────────────────────
def eval_with_pool(candidate, *, opponents, team_pool, battles_per_opponent: int, team_chooser,
                   submit_fn: Callable, prev_best=None, mirror_battles: Optional[int] = None,
                   matchup_seed: int = 0, battle_timeout: Optional[float] = 90.0,
                   n_procs: int = 4, async_per_proc: int = 3,
                   live_dir=None, save_replays: bool = False, generation: int = 0, ports=None):
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
                             matchup_seed=matchup_seed, mirror_battles=mirror_battles,
                             gen=generation)
    batches = partition_specs(specs, n_procs)
    pb = str(prev_best) if prev_best else None
    tc = str(team_chooser) if team_chooser else None
    ld = str(live_dir) if live_dir else None
    # 22f: spread the eval worker batches round-robin across the pool servers (batch i -> ports[i%K]).
    _ports = [int(p) for p in ports] if ports else None
    payloads = [(str(candidate), pb, tc, batch, battle_timeout, async_per_proc, ld,
                 bool(save_replays), (_ports[i % len(_ports)] if _ports else None))
                for i, batch in enumerate(batches)]
    results, source = merge_eval_results(submit_fn(payloads, worker_fn=eval_worker))
    for kind in opponents:
        results.setdefault(kind, (0, 0))             # a fully-dropped opponent reads 0/0, not absent
    return results, source
