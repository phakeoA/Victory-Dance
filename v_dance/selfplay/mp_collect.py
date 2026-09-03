"""Multiprocessing collection — picklable plan + stateless worker (task #14b.1).

The probe (14a) showed collection scales ~5x across OS processes because the asyncio path is
GIL-bound on the per-turn model inference. This module is the foundation for the real
integration (14b.2 wires the pool):

  * ``ChunkSpec``        — a flat, PICKLABLE descriptor of one collection chunk (team pairing +
                           league-sampled opponent), so the MAIN process can plan the batch
                           (``build_chunk_specs`` — league sampling stays in main, race-free) and
                           ship the work to worker processes without sending league objects.
  * ``partition_specs``  — split the flat plan across workers (round-robin → balanced load).
  * ``worker_collect``   — the STATELESS spawn-worker entry: load the AC (CPU, cached), collect
                           its assigned ChunkSpecs against the SHARED Showdown server, return a
                           picklable ``WorkerResult``. Reuses the tested per-battle primitives
                           (``play_pairing`` / ``close_players`` / ``run_jobs``) — zero new battle
                           code — and mirrors ``generation.collect_with_league``'s per-kind logic.
  * ``merge_results``    — fold the per-worker results back together in main (where league
                           ``record_result`` / PFSP then runs, in 14b.2).

The pure scaffolding (specs / partition / merge / picklability) and the collection DISPATCH
(``_collect_specs`` with an injected ``build_players``) are unit-tested offline with fakes; only
``worker_collect`` / ``_build_players_real`` touch poke-env (live-tested in 14b.4). Top-level
imports stay torch-/poke-env-free so the plan is importable in the main process and in tests."""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from v_dance.play.parallel_battles import (close_players, collect_account_names, gen_salt,
                                           play_pairing, run_jobs)
from v_dance.selfplay.collector import align_paired_trajectories   # pure (poke-env-free)

_REPO_ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger(__name__)


# ── picklable plan ─────────────────────────────────────────────────────────────
@dataclass
class ChunkSpec:
    """One collection chunk, flattened to picklable primitives (no league objects).

    ``kind`` ∈ {latest, snapshot, scripted}; ``opp_ref`` is the opponent checkpoint PATH for a
    snapshot, the scripted KIND ("max_damage"/…) for scripted, and ``None`` for latest (the
    opponent is another recorder sharing the live AC). ``snapshot_id`` is set only for snapshots
    so PFSP results attribute back to the right league entry. ``uid`` is GLOBALLY unique across
    the whole batch, so worker account names / replay paths never collide WITHIN a gen; ``gen``
    (22d) folds the GENERATION into those names too, so they don't collide ACROSS gens either (a
    stale server-side challenge from one gen vs the next gen's reuse)."""
    team_a: str
    team_b: str
    n: int
    kind: str
    opp_ref: Optional[str]
    snapshot_id: Optional[str]
    uid: int
    gen: int = 0
    # W2 throughput step 2b (2026-09-03): > 0 → this chunk's pair plays through the SERVER-SIDE spawner
    # (rlspawn plugin; spawn_client.play_pairing_spawned) keeping this many rooms alive, instead of the
    # challenge protocol. 0 = challenge path, byte-identical. Picklable primitive like the rest.
    spawn_rooms: int = 0


@dataclass
class WorkerResult:
    """One worker's picklable return: collected trajectories, decision-source tally, PFSP
    outcomes ``(snapshot_id, latest_won)`` for the main process to fold into the league, and the
    finished-game count."""
    trajectories: list = field(default_factory=list)
    source_counts: dict = field(default_factory=dict)
    pfsp: List[Tuple[str, bool]] = field(default_factory=list)
    n_games: int = 0


def _spec_from_sample(sample, team_a: str, team_b: str, n: int, uid: int, gen: int) -> ChunkSpec:
    kind = sample[0]
    if kind == "snapshot":
        snap = sample[1]
        return ChunkSpec(team_a, team_b, n, "snapshot", snap.path, snap.snapshot_id, uid, gen)
    if kind == "scripted":
        return ChunkSpec(team_a, team_b, n, "scripted", str(sample[1]), None, uid, gen)
    return ChunkSpec(team_a, team_b, n, "latest", None, None, uid, gen)   # latest (or degenerate)


def build_chunk_specs(league, team_pool, n_games: int, *, chunk_size: int = 10,
                      matchup_seed: int = 0, seed: int = 0, gen: int = 0,
                      spawn_rooms: int = 0) -> List[ChunkSpec]:
    """Plan the collection batch as a flat list of PICKLABLE ChunkSpecs — the multiprocessing
    analogue of ``generation.build_collection_chunks``. League sampling + uid assignment happen
    HERE, in the main process (race-free), so the workers never touch league state. ``gen`` (22d)
    is stamped onto every spec so the worker account names fold in the generation (no cross-gen
    reuse collisions). Pure w.r.t. poke-env (torch-free)."""
    import numpy as np
    from v_dance.eval.gauntlet import team_matchups
    rng = np.random.default_rng(seed)
    specs: List[ChunkSpec] = []
    uid = 0
    for team_a, team_b, n in team_matchups(team_pool, n_games, seed=matchup_seed):
        remaining = n
        while remaining > 0:
            cn = min(chunk_size, remaining)
            remaining -= cn
            uid += 1
            spec = _spec_from_sample(league.sample(rng), team_a, team_b, cn, uid, gen)
            spec.spawn_rooms = max(0, int(spawn_rooms or 0))
            specs.append(spec)
    return specs


def partition_specs(specs: List[ChunkSpec], n_workers: int) -> List[List[ChunkSpec]]:
    """Split the flat plan across ``n_workers`` worker batches, ROUND-ROBIN so each worker gets a
    balanced chunk count (chunks are ~equal cost). Empty batches are dropped (fewer specs than
    workers ⇒ fewer workers spawned)."""
    n = max(1, int(n_workers))
    buckets: List[List[ChunkSpec]] = [[] for _ in range(n)]
    for i, s in enumerate(specs):
        buckets[i % n].append(s)
    return [b for b in buckets if b]


def merge_results(results) -> WorkerResult:
    """Fold per-worker ``WorkerResult``s into one (trajectories concatenated, source counts
    summed, PFSP outcomes concatenated, games summed). ``None`` results (a worker that died) are
    skipped."""
    trajs: list = []
    sc: Counter = Counter()
    pfsp: List[Tuple[str, bool]] = []
    games = 0
    for r in results:
        if r is None:
            continue
        trajs.extend(r.trajectories)
        sc.update(r.source_counts)
        pfsp.extend(r.pfsp)
        games += r.n_games
    return WorkerResult(trajs, dict(sc), pfsp, games)


# ── collection dispatch (injected player factory → offline-testable) ───────────
async def _play_spawned_real(our, opp, n: int, *, rooms: int, battle_timeout, label: str):
    """The real spawner pairing (poke-env + the rlspawn plugin) — lazy import keeps this module's
    top level poke-env-free. Returns the SpawnStats."""
    from v_dance.play.run_local_battle import BATTLE_FORMAT
    from v_dance.selfplay import spawn_client as SC
    return await SC.play_pairing_spawned(
        our, opp, n, fmt=BATTLE_FORMAT,
        cfg=SC.SpawnConfig(rooms=int(rooms), battle_timeout=battle_timeout), label=label)


async def _collect_specs(ac, specs: List[ChunkSpec], *, tau: float, seed: int,
                         team_chooser, battle_timeout: Optional[float], async_workers: int,
                         build_players: Callable, live_dir=None, save_replays: bool = False,
                         status=None, port: Optional[int] = None,
                         play_spawned: Optional[Callable] = None) -> WorkerResult:
    """Play every ChunkSpec in ``specs`` (bounded to ``async_workers`` concurrent battles within
    this process), collecting trajectories + source counts + PFSP outcomes.
    ``build_players(ac, spec, tau, seed, team_chooser) -> (our, opp)`` is INJECTED so the per-kind
    dispatch is offline-testable with fakes. Mirrors ``collect_with_league``'s per-kind logic:
    latest ⇒ collect BOTH perspectives; snapshot ⇒ our trajectory + record (snapshot_id, won) for
    PFSP; scripted ⇒ our trajectory only. A spec with ``spawn_rooms > 0`` plays through the
    server-side spawner (``play_spawned(our, opp, n, rooms=, battle_timeout=, label=)``, injected
    for the offline fakes; default = the real ``spawn_client`` pairing) — every opponent kind is a
    poke-env account of ours, so all three kinds can spawn. Its stats land in ``source_counts`` as
    ``spawn_*`` (bookkeeping keys, stripped by the Phase-0 report)."""
    trajectories: list = []
    source_counts: Counter = Counter()
    pfsp: List[Tuple[str, bool]] = []
    games = {"n": 0}
    _spawn_fn = play_spawned or _play_spawned_real

    async def _run(spec: ChunkSpec):
        try:
            our, opp = build_players(ac, spec, tau, seed, team_chooser, live_dir, save_replays,
                                     port=port)   # 22f: bind to this worker's assigned pool server
        except Exception:
            # 14b.3 review: a transient build error (team resolve/load) skips THIS chunk only —
            # otherwise it bubbles out of run_jobs and drops the whole worker's batch (~1/n_procs).
            log.warning("mp collect player-build failed (vs %s) — skipping chunk.",
                        spec.kind, exc_info=True)
            return
        try:
            rooms = int(getattr(spec, "spawn_rooms", 0) or 0)
            if rooms > 0:
                st = await _spawn_fn(our, opp, spec.n, rooms=rooms, battle_timeout=battle_timeout,
                                     label=f"{spec.kind} spawn uid{spec.uid}")
                for k in ("pairs", "decisions", "stalls_forfeited", "ghosts_rescued", "abandoned"):
                    source_counts["spawn_" + k.split("_")[0]] += int(getattr(st, k, 0) or 0)
                source_counts["spawn_elapsed_s"] += int(round(float(getattr(st, "elapsed_s", 0.0) or 0.0)))
            else:
                await play_pairing(our, opp, spec.n, battle_timeout=battle_timeout, label=spec.kind)
        except Exception:
            log.warning("mp collect chunk failed (vs %s) — continuing.", spec.kind, exc_info=True)
        finally:
            source_counts.update(getattr(our, "_source_counts", {}) or {})
            our_trajs = our.finished_trajectories()
            # #4 (T3.1 extension): a NON-recording opponent (snapshot/scripted) that backstop-forfeited
            # makes OUR +1 win there spurious → tag our trajectory FALLBACK so drop_fallback_pairs discards
            # it. (latest is handled by the opp SelfPlayVGCPlayer's own FALLBACK + shared battle_id.) Inline
            # the tag-normalize (strip leading '>') to keep this module poke-env-free, matching _norm_tag.
            if spec.kind != "latest":
                _opp_ff = getattr(opp, "_forfeited_tags", None)
                if _opp_ff:
                    for _t in our_trajs.values():
                        if (str(_t.meta.battle_id or "").strip().lstrip(">").strip()) in _opp_ff:
                            _t.meta.terminal_type = "fallback"
            trajectories.extend(our_trajs.values())
            games["n"] += len(our_trajs)
            if spec.kind == "latest":
                source_counts.update(getattr(opp, "_source_counts", {}) or {})
                opp_trajs = opp.finished_trajectories()
                # Piece 3 (Level-A): align opp-action targets across both mirror perspectives
                # (shared battle_tag). In-place -> the our_trajs already extended above update too.
                align_paired_trajectories(our_trajs, opp_trajs)
                trajectories.extend(opp_trajs.values())
            elif spec.kind == "snapshot":
                for t in our_trajs.values():
                    # #1: skip a FALLBACK (opp backstop-forfeit) game — dropped from PPO, so it must
                    # not bias PFSP either. Gate on is_trainable (same as the PPO buffer). getattr keeps
                    # minimal offline fakes (no is_trainable) treated as trainable = the old behavior.
                    if t.meta.won is not None and getattr(t.meta, "is_trainable", True):
                        pfsp.append((spec.snapshot_id, bool(t.meta.won)))
            await close_players(our, opp)

    await run_jobs([lambda s=s: _run(s) for s in specs], workers=async_workers)
    return WorkerResult(trajectories, dict(source_counts), pfsp, games["n"])


def _build_players_real(ac, spec: ChunkSpec, tau: float, seed: int, team_chooser, live_dir=None,
                        save_replays: bool = False, port: Optional[int] = None):
    """The real per-kind player factory (poke-env). Mirrors ``collect_with_league._sp`` /
    opponent construction exactly, so multiprocess collection is byte-equivalent to the asyncio
    path — the global ``uid`` keeps account names unique across workers and ``spec.gen`` (22d)
    keeps them unique across generations. ``live_dir`` (#18) is the shared spectate dir each worker
    writes its battles to."""
    import logging as _logging
    import v_dance.play.run_local_battle as R
    from poke_env import AccountConfiguration
    from v_dance.eval.gauntlet import _make_opponent
    from v_dance.selfplay.game_runner import SelfPlayVGCPlayer
    uid, cn = spec.uid, spec.n
    ta = R.load_team(R.resolve_team_path(spec.team_a))
    tb = R.load_team(R.resolve_team_path(spec.team_b))
    rb = _REPO_ROOT / "artifacts" / "replay_buffer"
    our_name, opp_name = collect_account_names(spec.kind, uid, salt=gen_salt(spec.gen),
                                               opp_ref=spec.opp_ref)
    # 22f: bind these players to the worker's assigned pool server (None = poke-env's 8000 default).
    _server = {"server_configuration": R.localhost_server_config(port)} if port is not None else {}
    # W2 spawn path: the poke-env battle-count queue must sit ABOVE the rooms the spawner keeps alive
    # (spawn_client.max_concurrent_for — a full queue blocks the listen loop); challenge path = cn.
    _rooms = int(getattr(spec, "spawn_rooms", 0) or 0)
    if _rooms > 0:
        from v_dance.selfplay.spawn_client import max_concurrent_for
        _mcb = max(1, cn, max_concurrent_for(_rooms))
    else:
        _mcb = max(1, cn)

    def _sp(team, who, name, live=None):
        return SelfPlayVGCPlayer(
            ac, tau=tau, sample_seed=seed + who * 10_000 + uid, live_dir=live,
            save_replays=save_replays,
            replay_path=rb / f"{name}.jsonl",         # diagnostic trace keyed by the (gen-unique) name
            account_configuration=AccountConfiguration(name, None),
            battle_format=R.BATTLE_FORMAT, team=team, max_concurrent_battles=_mcb,
            log_level=_logging.WARNING, **_server)

    our = _sp(ta, 0, our_name, live=live_dir)         # #18: worker publishes its battle to live_dir
    if spec.kind == "latest":
        opp = _sp(tb, 1, opp_name)
    elif spec.kind == "snapshot":
        opp = R.make_player(opp_name, tb, model_path=spec.opp_ref,
                            team_chooser_path=team_chooser, max_concurrent_battles=_mcb,
                            port=port)
    else:   # scripted
        opp = _make_opponent(spec.opp_ref, opp_name, tb,
                             max_concurrent_battles=_mcb, port=port)
    return our, opp


# ── stateless spawn-worker entry (top-level so spawn can pickle it by name) ─────
_WORKER_CACHE: dict = {}


def _worker_ac(ckpt):
    """Load the AC on the CPU, cached by (path, mtime) so a persistent worker reuses it across
    chunks but RELOADS when the main process writes new per-gen weights (a fresh path/mtime)."""
    import os
    key = (str(ckpt), os.path.getmtime(ckpt))
    ac = _WORKER_CACHE.get(key)
    if ac is None:
        from v_dance.selfplay.actor_critic import ActorCritic
        ac = ActorCritic.from_bc_checkpoint(ckpt, device="cpu")
        # CRITICAL (14b.3 review): from_bc_checkpoint RE-CLONES the critic from the policy's value
        # head and IGNORES the saved critic_state. The recorded ``value`` is the GAE baseline (read
        # from ac.critic), and the asyncio path records it from the TRAINED critic (inference_copy
        # deep-copies it). Load the trained critic_state so mp baselines match — else GAE drifts.
        ac.restore_from(ckpt, device="cpu")
        _WORKER_CACHE.clear()                    # only the latest weights matter
        _WORKER_CACHE[key] = ac
    return ac


def worker_collect(payload):
    """Spawn-worker entry: load the AC (CPU) and collect the assigned ChunkSpecs against the
    SHARED server, returning a picklable ``WorkerResult``. ``payload`` =
    ``(ckpt, specs, tau, seed, team_chooser, battle_timeout, async_workers)`` — all picklable."""
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""      # CPU-only worker (no VRAM / CUDA init)
    import asyncio
    import logging as _logging
    import torch
    torch.set_num_threads(1)                      # avoid N×all-cores oversubscription
    _logging.getLogger("poke_env").setLevel(_logging.ERROR)
    _logging.getLogger("websockets").setLevel(_logging.ERROR)
    ckpt, specs, tau, seed, team_chooser, battle_timeout, async_workers, live_dir, save_replays, port = payload
    ac = _worker_ac(ckpt)
    return asyncio.run(_collect_specs(
        ac, specs, tau=tau, seed=seed, team_chooser=team_chooser,
        battle_timeout=battle_timeout, async_workers=async_workers,
        build_players=_build_players_real, live_dir=live_dir, save_replays=save_replays,
        port=port))


# ── real process-pool backend (task #14b.2b) ───────────────────────────────────
def _default_executor(n_procs: int):
    """A spawn-context ProcessPoolExecutor (Windows requires spawn; spawn also keeps workers
    free of the parent's imported CUDA/torch state)."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    return ProcessPoolExecutor(max_workers=max(1, int(n_procs)),
                               mp_context=mp.get_context("spawn"))


def _pool_selftest_worker(payload):
    """Trivial picklable worker for the offline pool smoke (NO poke-env): ``payload = (label, n)``
    -> a ``WorkerResult`` with ``n_games=n``. Lets the real ``ProcessPoolExecutor(spawn)`` wiring
    be confirmed without battles."""
    label, n = payload
    return WorkerResult(trajectories=[label] * int(n), source_counts={"games": int(n)},
                        n_games=int(n))


class CollectionPool:
    """Persistent worker pool for multiprocess collection (task #14b.2b).

    Holds a spawn ``ProcessPoolExecutor`` across generations (so spawn + torch import are paid
    ONCE, not per gen) and exposes ``submit(payloads) -> [WorkerResult | None]`` as the backend the
    orchestrator (``collect_with_pool``) injects. CRASH RECOVERY: a worker that raises comes back
    ``None`` (its chunks are dropped, the gen survives); a worker that DIES (BrokenProcessPool)
    also yields ``None`` and the pool is torn down so the NEXT ``submit`` respawns a fresh one —
    a single crash can't poison the whole run. ``executor_factory`` / ``worker_fn`` are injectable
    so the submit + recovery logic is unit-tested with a fake executor (no real processes)."""

    def __init__(self, n_procs: int, *, worker_fn: Callable = worker_collect,
                 executor_factory: Callable = _default_executor):
        self.n_procs = max(1, int(n_procs))
        self._worker_fn = worker_fn
        self._factory = executor_factory
        self._ex = None

    def _ensure(self):
        if self._ex is None:
            self._ex = self._factory(self.n_procs)
        return self._ex

    def submit(self, payloads, worker_fn=None, on_result=None):
        """Run a worker over ``payloads`` and return per-payload results (``None`` for a worker
        that raised or died), in submission order. ``worker_fn`` overrides the default for THIS
        submit — ProcessPoolExecutor workers run any picklable function per call, so the SAME pool
        serves both collection (``worker_collect``) and eval (``mp_eval.eval_worker``) without a
        second set of processes. ``on_result(result)`` (optional) is invoked as each batch's
        result is collected (submission order) for live progress; results are returned in
        submission order."""
        if not payloads:
            return []
        from concurrent.futures.process import BrokenProcessPool
        fn = worker_fn or self._worker_fn
        ex = self._ensure()
        futs = [ex.submit(fn, p) for p in payloads]
        results, broken = [], False
        for f in futs:                               # collected in submission order
            try:
                r = f.result()
            except BrokenProcessPool:
                r, broken = None, True
            except Exception:
                log.warning("collection worker raised — dropping its chunk(s).", exc_info=True)
                r = None
            results.append(r)
            if on_result is not None:                # live progress as each batch lands
                try:
                    on_result(r)
                except Exception:
                    log.debug("on_result progress callback failed (non-fatal)", exc_info=True)
        if broken:
            log.warning("collection pool BROKE (a worker died) — respawning on the next batch.")
            try:
                self._ex.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            self._ex = None                          # next submit() spawns a fresh pool
        return results

    def close(self):
        if self._ex is not None:
            try:
                self._ex.shutdown(wait=True, cancel_futures=True)
            except Exception:
                pass
            self._ex = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ── per-gen weight handoff (task #14b.2c) ──────────────────────────────────────
def mp_ckpt_path(archive_dir, generation: int) -> Path:
    """Path for a generation's worker checkpoint. A FRESH path per gen busts the worker's
    ``(path, mtime)`` AC cache so it picks up the new weights."""
    return Path(archive_dir) / f"_mp_collect_gen{generation}.pt"


def sweep_mp_ckpts(archive_dir) -> int:
    """Delete any leftover per-gen worker checkpoints (``_mp_collect_gen*.pt``) in ``archive_dir``.

    The per-gen unlink in the live loop is skipped if a generation's collection raises (Ctrl-C
    lands inside the blocking ``pool.submit``; KeyboardInterrupt is a BaseException the worker's
    ``except Exception`` doesn't catch), orphaning a FULL-weight ckpt each crashed gen. Call this
    at startup (reclaim prior-run orphans) and in the loop's finally. Returns the count removed."""
    n = 0
    try:
        for f in Path(archive_dir).glob("_mp_collect_gen*.pt"):
            try:
                f.unlink()
                n += 1
            except Exception:
                pass
    except Exception:
        pass
    return n


def save_inference_ckpt(ac, path, *, generation: int = 0) -> str:
    """Freeze the AC's CURRENT weights to ``path`` as a worker-loadable checkpoint.

    Writes a CPU INFERENCE COPY (so a GPU-resident trainer's weights serialize as CPU tensors the
    CPU workers load cleanly) and VERIFIES the round-trip — a broken save raises loudly instead of
    silently degrading the workers to a no-model picker (the 3c.3b lesson). Used by 14b.3 as the
    ``save_ckpt_fn`` injected into ``collect_with_pool``. Returns ``str(path)``."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ac.inference_copy("cpu").save(path, generation=generation, verify=True)
    return str(path)


# ── main-process orchestration (task #14b.2a) ──────────────────────────────────
def collect_with_pool(ac, league, n_games: int, *, team_pool, ckpt_path,
                      submit_fn: Callable, save_ckpt_fn: Callable,
                      n_procs: int = 4, async_per_proc: int = 3, tau: float = 1.0,
                      seed: int = 0, matchup_seed: int = 0, chunk_size: int = 10,
                      team_chooser=None, battle_timeout: Optional[float] = 90.0, status=None,
                      live_dir=None, save_replays: bool = False, generation: int = 0,
                      ports=None, spawn_rooms: int = 0):
    """Multiprocess analogue of ``generation.collect_with_league`` (task #14b.2).

    Plans the batch + samples the league in the MAIN process (race-free), freezes the current
    weights to ``ckpt_path`` for the workers, ships PARTITIONED ChunkSpecs to worker processes via
    the injected ``submit_fn``, folds the per-worker results, and applies PFSP
    ``league.record_result`` HERE — league state never leaves the main process.

    ``submit_fn(payloads) -> List[Optional[WorkerResult]]`` (the real ProcessPoolExecutor backend
    is 14b.2b; a dead worker comes back ``None`` and is skipped) and ``save_ckpt_fn(ac, path)``
    (the real inference-ckpt save is 14b.2c) are INJECTED, so this orchestration is offline-tested
    with fakes. Returns the SAME ``(trajectories, source_counts)`` shape as
    ``collect_with_league`` so the 14b.3 wiring is a drop-in."""
    save_ckpt_fn(ac, ckpt_path)                          # freeze current weights for the workers
    # W2 spawn path (2026-09-03): ``spawn_rooms`` > 0 stamps every chunk so each worker process drives
    # its pair through the server-side spawner (the plugin must already be in the clone — the caller
    # installs it BEFORE the pool servers start; the launcher's build compiles it).
    specs = build_chunk_specs(league, team_pool, n_games, chunk_size=chunk_size,
                              matchup_seed=matchup_seed, seed=seed, gen=generation,
                              spawn_rooms=spawn_rooms)
    batches = partition_specs(specs, n_procs)
    _ld = str(live_dir) if live_dir else None
    # 22f: spread the worker batches round-robin across the pool servers (batch i -> ports[i % K]);
    # ports=None keeps the single-server default (poke-env's localhost:8000).
    _ports = [int(p) for p in ports] if ports else None
    payloads = [(str(ckpt_path), batch, tau, seed, team_chooser, battle_timeout, async_per_proc,
                 _ld, bool(save_replays), (_ports[i % len(_ports)] if _ports else None))
                for i, batch in enumerate(batches)]
    # Live progress: report games_done to the status writer AS EACH worker batch lands (was a
    # single update at the very END -> the dashboard sat frozen at 0/N for the whole multiprocess
    # collect). The on_result hook fires per completed batch; a submit_fn that doesn't accept it
    # (offline fakes) falls back to the end-of-collect reconcile below.
    prog = {"games": 0, "decided": 0, "won": 0}

    def _on_result(res):
        if status is None or res is None:
            return
        prog["games"] += int(getattr(res, "n_games", 0) or 0)
        for t in getattr(res, "trajectories", None) or []:
            w = getattr(getattr(t, "meta", None), "won", None)
            if w is not None:
                prog["decided"] += 1
                prog["won"] += 1 if w else 0
        try:
            status.games(prog["games"],
                         (prog["won"] / prog["decided"]) if prog["decided"] else None)
        except Exception:
            log.debug("status games() progress update failed (non-fatal)", exc_info=True)

    import inspect
    try:
        _sig = inspect.signature(submit_fn)
        _accepts = ("on_result" in _sig.parameters
                    or any(p.kind == p.VAR_KEYWORD for p in _sig.parameters.values()))
    except (TypeError, ValueError):
        _accepts = False
    raw = submit_fn(payloads, on_result=_on_result) if _accepts else submit_fn(payloads)
    merged = merge_results(raw)
    for snapshot_id, won in merged.pfsp:                 # PFSP update in MAIN (league stays here)
        league.record_result(snapshot_id, won)
    if status is not None:
        decided = won = 0
        for t in merged.trajectories:                    # final reconcile (covers the no-hook path)
            w = getattr(getattr(t, "meta", None), "won", None)
            if w is not None:
                decided += 1
                won += 1 if w else 0
        try:
            status.games(merged.n_games, (won / decided) if decided else None)
        except Exception:
            log.debug("status games() update failed (non-fatal)", exc_info=True)
    return merged.trajectories, merged.source_counts
