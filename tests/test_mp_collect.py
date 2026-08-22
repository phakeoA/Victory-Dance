"""Offline tests for multiprocessing-collection scaffolding (task #14b.1).

No server / spawn: covers the PICKLABLE plan (ChunkSpec / WorkerResult / Trajectory),
``build_chunk_specs`` flattening the league sample, round-robin ``partition_specs``,
``merge_results`` aggregation, and the per-kind collection DISPATCH (``_collect_specs`` with
an injected fake player factory — latest⇒both perspectives, snapshot⇒PFSP, scripted⇒ours)."""
from __future__ import annotations

import asyncio
import pickle
import types

import numpy as np
import pytest

from v_dance.selfplay import mp_collect as MP
from v_dance.selfplay.mp_collect import ChunkSpec, WorkerResult


# ── fake league + snapshot ────────────────────────────────────────────────────
class _FakeSnap:
    def __init__(self, sid, path):
        self.snapshot_id = sid
        self.path = path


class _FakeLeague:
    """Returns canned sample() results in order (round-robin if exhausted)."""
    def __init__(self, samples):
        self._samples = list(samples)
        self._i = 0

    def sample(self, rng):
        s = self._samples[self._i % len(self._samples)]
        self._i += 1
        return s


# ── ChunkSpec / build_chunk_specs ─────────────────────────────────────────────
def test_chunkspec_is_picklable():
    s = ChunkSpec("A", "B", 3, "snapshot", "gen2.pt", "gen2", 7)
    assert pickle.loads(pickle.dumps(s)) == s


def test_build_chunk_specs_flattens_each_league_kind():
    league = _FakeLeague([("latest", "latest.pt"),
                          ("snapshot", _FakeSnap("gen3", "g3.pt")),
                          ("scripted", "max_damage")])
    specs = MP.build_chunk_specs(league, ["A", "B"], 3, chunk_size=1, matchup_seed=0, seed=0)
    assert len(specs) == 3                       # 3 single-battle chunks
    by_kind = {s.kind: s for s in specs}
    assert by_kind["latest"].opp_ref is None and by_kind["latest"].snapshot_id is None
    assert by_kind["snapshot"].opp_ref == "g3.pt" and by_kind["snapshot"].snapshot_id == "gen3"
    assert by_kind["scripted"].opp_ref == "max_damage" and by_kind["scripted"].snapshot_id is None


def test_build_chunk_specs_uids_are_unique_and_sequential():
    league = _FakeLeague([("latest", "x.pt")])
    specs = MP.build_chunk_specs(league, ["A", "B", "C"], 6, chunk_size=1, seed=1)
    uids = [s.uid for s in specs]
    assert uids == list(range(1, len(specs) + 1))      # 1..N, globally unique


def test_build_chunk_specs_stamps_generation_and_defaults_zero():
    """22d: every chunk carries the generation (folded into the worker account names); default 0."""
    league = _FakeLeague([("latest", "x.pt")])
    specs = MP.build_chunk_specs(league, ["A", "B", "C"], 6, chunk_size=1, seed=1, gen=4)
    assert specs and all(s.gen == 4 for s in specs)
    league2 = _FakeLeague([("latest", "x.pt")])
    assert all(s.gen == 0 for s in
               MP.build_chunk_specs(league2, ["A", "B", "C"], 6, chunk_size=1, seed=1))


def test_collect_with_pool_threads_generation_onto_specs():
    """generation reaches the ChunkSpecs shipped to the workers (so they build gen-keyed names)."""
    captured = {}

    def submit(payloads):
        captured["specs"] = [s for p in payloads for s in p[1]]   # payload[1] = the ChunkSpec batch
        return [WorkerResult(n_games=1) for _ in payloads]

    league = _RecordingLeague([("latest", "x.pt")])
    MP.collect_with_pool(object(), league, 2, team_pool=["A", "B"], ckpt_path="t.pt",
                         n_procs=1, chunk_size=1, submit_fn=submit,
                         save_ckpt_fn=lambda a, p: None, generation=9)
    assert captured["specs"] and all(s.gen == 9 for s in captured["specs"])


def test_collect_with_pool_assigns_pool_ports_round_robin():
    """22f.3: worker batches are spread round-robin across the pool servers (batch i -> ports[i%K])."""
    captured = {}

    def submit(payloads):
        captured["ports"] = [p[9] for p in payloads]    # index 9 = the assigned server port
        return [WorkerResult(n_games=1) for _ in payloads]

    league = _RecordingLeague([("latest", "x.pt")])
    MP.collect_with_pool(object(), league, 6, team_pool=["A", "B", "C"], ckpt_path="t.pt",
                         n_procs=3, chunk_size=1, submit_fn=submit,
                         save_ckpt_fn=lambda a, p: None, ports=[8000, 8001])
    assert captured["ports"] == [8000, 8001, 8000]      # 3 batches over 2 servers, round-robin


def test_collect_with_pool_no_ports_is_single_server():
    captured = {}

    def submit(payloads):
        captured["ports"] = [p[9] for p in payloads]
        return [WorkerResult(n_games=1) for _ in payloads]

    league = _RecordingLeague([("latest", "x.pt")])
    MP.collect_with_pool(object(), league, 4, team_pool=["A", "B"], ckpt_path="t.pt",
                         n_procs=2, chunk_size=1, submit_fn=submit, save_ckpt_fn=lambda a, p: None)
    assert captured["ports"] and all(port is None for port in captured["ports"])   # 8000 default


# ── partition_specs ───────────────────────────────────────────────────────────
def test_partition_specs_round_robin_balances():
    specs = [ChunkSpec("A", "B", 1, "latest", None, None, i) for i in range(5)]
    parts = MP.partition_specs(specs, 2)
    assert [len(p) for p in parts] == [3, 2]            # round-robin: 0,2,4 | 1,3
    assert [s.uid for s in parts[0]] == [0, 2, 4]


def test_partition_specs_drops_empty_buckets():
    specs = [ChunkSpec("A", "B", 1, "latest", None, None, 0)]
    parts = MP.partition_specs(specs, 4)               # 1 spec, 4 workers
    assert len(parts) == 1 and len(parts[0]) == 1      # 3 empty buckets dropped


# ── merge_results ─────────────────────────────────────────────────────────────
def test_merge_results_aggregates_and_skips_none():
    r1 = WorkerResult(trajectories=["t1", "t2"], source_counts={"model": 5},
                      pfsp=[("gen1", True)], n_games=2)
    r2 = WorkerResult(trajectories=["t3"], source_counts={"model": 3, "retry": 1},
                      pfsp=[("gen1", False)], n_games=1)
    merged = MP.merge_results([r1, None, r2])
    assert merged.trajectories == ["t1", "t2", "t3"]
    assert merged.source_counts == {"model": 8, "retry": 1}
    assert merged.pfsp == [("gen1", True), ("gen1", False)]
    assert merged.n_games == 3


# ── picklability of the wire types ────────────────────────────────────────────
def test_worker_result_is_picklable():
    r = WorkerResult(trajectories=[1, 2], source_counts={"model": 4},
                     pfsp=[("g", True)], n_games=2)
    assert pickle.loads(pickle.dumps(r)) == r


def test_trajectory_is_picklable():
    """Trajectories cross the process boundary as pickled return values — confirm a real one
    (numpy state + masks + meta) round-trips."""
    from v_dance.selfplay.collector import TrajectoryCollector
    c = TrajectoryCollector("battle-x", "p1")
    c.add_step(state=np.zeros(4, np.float32), action_s0=0, action_s1=1, gimmick_s0=0,
               gimmick_s1=0, logprob=-1.0, value=0.5, mask_s0=[1, 1, 1, 1],
               mask_s1=[1, 1, 1, 1], gmask_s0=None, gmask_s1=None, decision_type="turn", turn=1)
    traj = c.finish(own_team=["A"], opp_team=["B"], tp_bring=[0], tp_leads=[0], won=True,
                    terminal_type="win", n_turns=1)
    back = pickle.loads(pickle.dumps(traj))
    assert back.meta.won is True and len(back.transitions) == 1
    assert np.allclose(back.transitions[0].state, traj.transitions[0].state)


# ── _collect_specs dispatch (injected fake player factory; no server) ─────────
class _FakeTraj:
    def __init__(self, won):
        self.meta = types.SimpleNamespace(won=won)


class _FakePlayer:
    def __init__(self, trajs=None, sources=None, proto=None):
        self.n_won_battles = 0
        self.n_finished_battles = 0
        self.closed = False
        self._trajs = trajs or {}
        self._source_counts = sources or {}
        self._proto_log = proto or {}
        async def _stop():
            return None
        self.ps_client = types.SimpleNamespace(stop_listening=_stop)

    async def battle_against(self, opp, n_battles):
        self.n_finished_battles += n_battles

    def finished_trajectories(self):
        return dict(self._trajs)

    def close(self):
        self.closed = True


def _fake_build_players(ac, spec, tau, seed, tc, live_dir=None, save_replays=False, port=None):
    our = _FakePlayer(trajs={f"b{spec.uid}": _FakeTraj(True)}, sources={"model": 5})
    if spec.kind == "latest":
        opp = _FakePlayer(trajs={f"b{spec.uid}o": _FakeTraj(False)}, sources={"model": 3})
    else:
        opp = _FakePlayer()
    return our, opp


def _collect(specs):
    return asyncio.run(MP._collect_specs(
        None, specs, tau=1.0, seed=0, team_chooser=None, battle_timeout=None,
        async_workers=2, build_players=_fake_build_players))


def test_collect_specs_latest_collects_both_perspectives():
    res = _collect([ChunkSpec("A", "B", 1, "latest", None, None, 1)])
    assert len(res.trajectories) == 2                  # our + opp
    assert res.source_counts["model"] == 8             # 5 (our) + 3 (opp)
    assert res.pfsp == []                              # latest doesn't feed PFSP
    assert res.n_games == 1                            # one game (our perspective count)


def test_collect_specs_snapshot_records_pfsp():
    res = _collect([ChunkSpec("A", "B", 1, "snapshot", "g3.pt", "gen3", 2)])
    assert len(res.trajectories) == 1                  # only our perspective
    assert res.pfsp == [("gen3", True)]                # (snapshot_id, latest_won)
    assert res.source_counts["model"] == 5


def test_collect_specs_scripted_only_our_trajectory():
    res = _collect([ChunkSpec("A", "B", 1, "scripted", "max_damage", None, 3)])
    assert len(res.trajectories) == 1
    assert res.pfsp == []


def test_collect_specs_scripted_opp_forfeit_marks_our_traj_fallback():
    """#4-ext: when a SCRIPTED/root opponent backstop-forfeits (its tag is in opp._forfeited_tags), the
    collection path tags OUR +1-win trajectory FALLBACK so drop_fallback_pairs (tested separately) discards
    the spurious win. Previously the fake opponents never set _forfeited_tags, so this branch was untested."""
    tag = "b9"
    our_traj = _FakeTraj(True)
    our_traj.meta.battle_id = tag
    our_traj.meta.terminal_type = "win"

    def _build(ac, spec, tau, seed, tc, live_dir=None, save_replays=False, port=None):
        our = _FakePlayer(trajs={tag: our_traj}, sources={"model": 5})
        opp = _FakePlayer()
        opp._forfeited_tags = {tag}        # the scripted opp backstop-forfeited this battle
        return our, opp

    res = asyncio.run(MP._collect_specs(
        None, [ChunkSpec("A", "B", 1, "scripted", "max_damage", None, 9)], tau=1.0, seed=0,
        team_chooser=None, battle_timeout=None, async_workers=2, build_players=_build))
    assert len(res.trajectories) == 1
    assert our_traj.meta.terminal_type == "fallback"      # our +1 is now discardable


def test_collect_specs_scripted_no_forfeit_keeps_win():
    """Control: a scripted opp that did NOT forfeit (no _forfeited_tags) leaves our win trajectory intact."""
    tag = "b7"
    our_traj = _FakeTraj(True)
    our_traj.meta.battle_id = tag
    our_traj.meta.terminal_type = "win"

    def _build(ac, spec, tau, seed, tc, live_dir=None, save_replays=False, port=None):
        return _FakePlayer(trajs={tag: our_traj}, sources={"model": 5}), _FakePlayer()

    asyncio.run(MP._collect_specs(
        None, [ChunkSpec("A", "B", 1, "scripted", "max_damage", None, 7)], tau=1.0, seed=0,
        team_chooser=None, battle_timeout=None, async_workers=2, build_players=_build))
    assert our_traj.meta.terminal_type == "win"           # untouched


def test_collect_specs_aggregates_across_specs():
    res = _collect([ChunkSpec("A", "B", 1, "latest", None, None, 1),
                    ChunkSpec("A", "B", 1, "snapshot", "g3.pt", "gen3", 2)])
    assert len(res.trajectories) == 3                  # 2 (latest) + 1 (snapshot)
    assert res.source_counts["model"] == 13            # 8 + 5
    assert res.pfsp == [("gen3", True)]
    assert res.n_games == 2


# ── collect_with_pool orchestration (task #14b.2a; injected submit + save) ────
class _RecordingLeague:
    """Fake league that both serves canned samples AND records PFSP record_result calls."""
    def __init__(self, samples):
        self._fl = _FakeLeague(samples)
        self.recorded = []

    def sample(self, rng):
        return self._fl.sample(rng)

    def record_result(self, snapshot_id, latest_won):
        self.recorded.append((snapshot_id, latest_won))


def test_collect_with_pool_freezes_weights_partitions_and_updates_league():
    saved = {}
    captured = {}

    def save_ckpt(ac, path):
        saved["call"] = (ac, str(path))

    def submit(payloads):
        captured["payloads"] = payloads
        return [WorkerResult(trajectories=[f"t{i}"], source_counts={"model": 1},
                             pfsp=[("gen2", True)], n_games=1)
                for i, _ in enumerate(payloads)]

    league = _RecordingLeague([("snapshot", _FakeSnap("gen2", "g2.pt"))])
    ac = object()
    trajs, src = MP.collect_with_pool(
        ac, league, 4, team_pool=["A", "B"], ckpt_path="tmp_gen.pt", n_procs=2,
        async_per_proc=2, chunk_size=1, submit_fn=submit, save_ckpt_fn=save_ckpt)

    assert saved["call"] == (ac, "tmp_gen.pt")                  # weights frozen for workers
    payloads = captured["payloads"]
    assert 1 <= len(payloads) <= 2                              # partitioned across <= n_procs
    # every payload carries the SAME ckpt path + a list of ChunkSpecs + the async fan-out
    assert all(p[0] == "tmp_gen.pt" and isinstance(p[1], list)
               and isinstance(p[1][0], ChunkSpec) and p[6] == 2 for p in payloads)
    # PFSP applied in MAIN — one record_result per worker's outcome
    assert league.recorded == [("gen2", True)] * len(payloads)
    assert sorted(trajs) == [f"t{i}" for i in range(len(payloads))]
    assert src == {"model": len(payloads)}                      # source counts summed


def test_collect_with_pool_tolerates_dead_worker():
    """A worker that died comes back as None from submit_fn — merge skips it, the gen survives."""
    def submit(payloads):
        return [WorkerResult(trajectories=["t"], source_counts={"model": 1}, pfsp=[], n_games=1),
                None]

    league = _RecordingLeague([("latest", "x.pt")])
    trajs, src = MP.collect_with_pool(
        ac=object(), league=league, n_games=2, team_pool=["A", "B"], ckpt_path="t.pt",
        n_procs=2, chunk_size=1, submit_fn=submit, save_ckpt_fn=lambda a, p: None)
    assert trajs == ["t"] and src == {"model": 1}              # only the live worker's output


# ── CollectionPool backend (task #14b.2b; fake executor → no real processes) ──
class _FakeFuture:
    def __init__(self, result=None, exc=None):
        self._r, self._e = result, exc

    def result(self):
        if self._e is not None:
            raise self._e
        return self._r


class _FakeExecutor:
    """Returns canned results/exceptions per submit() in order; records shutdown."""
    def __init__(self, results):
        self._results = list(results)
        self._i = 0
        self.shutdown_called = False

    def submit(self, fn, payload):
        r = self._results[self._i]
        self._i += 1
        return _FakeFuture(exc=r) if isinstance(r, BaseException) else _FakeFuture(result=r)

    def shutdown(self, wait=True, cancel_futures=False):
        self.shutdown_called = True


def test_pool_submit_returns_results_in_order_and_closes():
    made = []

    def factory(n):
        e = _FakeExecutor([WorkerResult(n_games=1), WorkerResult(n_games=2)])
        made.append(e)
        return e

    pool = MP.CollectionPool(2, executor_factory=factory)
    res = pool.submit(["p1", "p2"])
    assert [r.n_games for r in res] == [1, 2]
    pool.close()
    assert made[0].shutdown_called and pool._ex is None


def test_pool_drops_raising_worker_but_keeps_pool():
    def factory(n):
        return _FakeExecutor([WorkerResult(n_games=1), RuntimeError("boom"),
                              WorkerResult(n_games=3)])

    pool = MP.CollectionPool(3, executor_factory=factory)
    res = pool.submit(["a", "b", "c"])
    assert res[0].n_games == 1 and res[1] is None and res[2].n_games == 3
    assert pool._ex is not None              # a normal exception does NOT tear down the pool


def test_pool_respawns_after_broken_process_pool():
    from concurrent.futures.process import BrokenProcessPool
    queue = [_FakeExecutor([BrokenProcessPool("worker died")]),
             _FakeExecutor([WorkerResult(n_games=9)])]
    made = []

    def factory(n):
        e = queue[len(made)]
        made.append(e)
        return e

    pool = MP.CollectionPool(1, executor_factory=factory)
    assert pool.submit(["x"]) == [None]      # crash → None
    assert pool._ex is None and len(made) == 1   # pool torn down for respawn
    res2 = pool.submit(["y"])
    assert res2[0].n_games == 9 and len(made) == 2   # next submit spawned a fresh pool


def test_pool_empty_payloads_is_noop():
    pool = MP.CollectionPool(2, executor_factory=lambda n: pytest.fail("must not spawn"))
    assert pool.submit([]) == []             # no payloads → no executor created


def test_pool_submit_uses_per_call_worker_fn():
    """#19: the same pool serves collection AND eval — submit(worker_fn=...) overrides the default
    per call (ProcessPoolExecutor runs any picklable fn per submit)."""
    seen = []

    class _Rec(_FakeExecutor):
        def submit(self, fn, payload):
            seen.append(fn)
            return super().submit(fn, payload)

    def my_eval_worker(p):
        return None

    pool = MP.CollectionPool(1, worker_fn=lambda p: "collect",
                             executor_factory=lambda n: _Rec([WorkerResult(n_games=1)]))
    pool.submit(["x"], worker_fn=my_eval_worker)
    assert seen == [my_eval_worker]                # the per-call override, not the default worker_fn


def test_pool_real_spawn_smoke():
    """Confirm the REAL ProcessPoolExecutor(spawn) wiring end-to-end with a trivial torch-free
    worker (no battles): two processes, two payloads, results in order."""
    with MP.CollectionPool(2, worker_fn=MP._pool_selftest_worker) as pool:
        res = pool.submit([("a", 3), ("b", 5)])
    assert [r.n_games for r in res] == [3, 5]
    assert res[0].source_counts == {"games": 3}


# ── per-gen weight handoff (task #14b.2c) ─────────────────────────────────────
def _tiny_ac(tmp_path):
    """Build a tiny ActorCritic from a synthetic value-trained BC checkpoint."""
    from conftest import write_attn_ckpt
    from v_dance.encoders.state_encoder import get_action_dim, get_gimmick_dim, get_state_dim
    from v_dance.selfplay.actor_critic import ActorCritic
    sd, ad, gd = get_state_dim(), get_action_dim(), get_gimmick_dim()
    p = write_attn_ckpt(tmp_path / "bc.pt")
    return ActorCritic.from_bc_checkpoint(p), (sd, ad, gd)


def test_mp_ckpt_path_is_fresh_per_generation(tmp_path):
    assert MP.mp_ckpt_path(tmp_path, 3) != MP.mp_ckpt_path(tmp_path, 4)   # cache-busting per gen


def test_save_inference_ckpt_roundtrips_to_worker_loadable(tmp_path):
    """The per-gen handoff: save the AC's current weights, then reload exactly how a worker does
    (from_bc_checkpoint on CPU) and confirm the policy weights match bit-for-bit."""
    import torch
    from v_dance.selfplay.actor_critic import ActorCritic
    ac, _ = _tiny_ac(tmp_path)
    out = MP.mp_ckpt_path(tmp_path, 0)
    ret = MP.save_inference_ckpt(ac, out, generation=7)
    assert ret == str(out) and out.exists()

    loaded = ActorCritic.from_bc_checkpoint(out, device="cpu")
    src, dst = ac.policy.state_dict(), loaded.policy.state_dict()
    assert src.keys() == dst.keys()
    for k in src:
        assert torch.allclose(src[k].cpu(), dst[k].cpu())          # weights survive the handoff


def test_worker_ac_loads_the_trained_critic_not_the_bc_clone(tmp_path):
    """14b.3 review (CRITICAL): the recorded `value` is the GAE baseline, read from ac.critic. The
    worker MUST load the TRAINED critic_state — from_bc_checkpoint alone re-clones the critic from
    the policy value head, so a warm-started/PPO-updated critic would be silently discarded."""
    import torch
    from v_dance.selfplay.actor_critic import ActorCritic
    ac, _ = _tiny_ac(tmp_path)
    with torch.no_grad():                          # diverge the trained critic from the policy head
        for p in ac.critic.parameters():
            p.add_(0.5)
    out = MP.mp_ckpt_path(tmp_path, 0)
    MP.save_inference_ckpt(ac, out, generation=0)

    MP._WORKER_CACHE.clear()
    wac = MP._worker_ac(out)                        # exactly how the worker loads it
    for k, v in ac.critic.state_dict().items():
        assert torch.allclose(wac.critic.state_dict()[k].cpu(), v.cpu())   # == the TRAINED critic

    bc_only = ActorCritic.from_bc_checkpoint(out, device="cpu")            # the buggy load
    differs = any(not torch.allclose(bc_only.critic.state_dict()[k].cpu(), v.cpu())
                  for k, v in ac.critic.state_dict().items())
    assert differs                                 # proves restore_from is load-bearing
    MP._WORKER_CACHE.clear()


def test_collect_specs_build_error_skips_only_that_chunk():
    """14b.3 review: a transient player-build error skips THAT chunk, not the whole worker batch."""
    def build(ac, spec, tau, seed, tc, live_dir=None, save_replays=False, port=None):
        if spec.uid == 2:
            raise RuntimeError("team resolve failed")
        return _fake_build_players(ac, spec, tau, seed, tc)

    res = asyncio.run(MP._collect_specs(
        None, [ChunkSpec("A", "B", 1, "scripted", "max_damage", None, 1),
               ChunkSpec("A", "B", 1, "scripted", "max_damage", None, 2),   # build fails here
               ChunkSpec("A", "B", 1, "scripted", "max_damage", None, 3)],
        tau=1.0, seed=0, team_chooser=None, battle_timeout=None, async_workers=2,
        build_players=build))
    assert len(res.trajectories) == 2              # uids 1 + 3 collected; uid 2 skipped, not all-dropped


def test_sweep_mp_ckpts_removes_only_worker_ckpts(tmp_path):
    """14b.3 review: reclaim orphaned per-gen worker ckpts WITHOUT touching league/resume files."""
    (tmp_path / "_mp_collect_gen0.pt").write_text("x")
    (tmp_path / "_mp_collect_gen5.pt").write_text("x")
    (tmp_path / "gen3.pt").write_text("keep")          # a real league snapshot — must survive
    (tmp_path / "resume.pt").write_text("keep")
    n = MP.sweep_mp_ckpts(tmp_path)
    assert n == 2
    assert not (tmp_path / "_mp_collect_gen0.pt").exists()
    assert not (tmp_path / "_mp_collect_gen5.pt").exists()
    assert (tmp_path / "gen3.pt").exists() and (tmp_path / "resume.pt").exists()


# ── live progress: status.games() updates PER BATCH, not just once at the end ──
def test_collect_with_pool_reports_incremental_progress():
    """The dashboard sat frozen at 0/N because status.games() only fired ONCE at merge.
    With an on_result-aware submit_fn it must now update games_done per completed batch."""
    calls = []

    class _Status:
        def games(self, games_done, winrate=None):
            calls.append(games_done)

    def submit(payloads, on_result=None):           # fire the hook per batch (as workers finish)
        out = []
        for p in payloads:
            r = WorkerResult(n_games=1, trajectories=[], source_counts={"model": 1})
            out.append(r)
            if on_result is not None:
                on_result(r)
        return out

    league = _RecordingLeague([("latest", "x.pt")])
    MP.collect_with_pool(object(), league, 3, team_pool=["A", "B"], ckpt_path="t.pt",
                         n_procs=3, chunk_size=1, submit_fn=submit,
                         save_ckpt_fn=lambda a, p: None, status=_Status())
    assert len(calls) >= 2                           # incremental (the OLD behaviour fired exactly once)
    assert calls == sorted(calls)                    # monotonic non-decreasing games_done
    assert calls[-1] >= 1


def test_collect_with_pool_progress_falls_back_without_on_result():
    """A submit_fn that doesn't accept on_result (offline fakes / older backends) must still
    update games_done once at the end — the inspect-guard must not crash."""
    calls = []

    class _Status:
        def games(self, games_done, winrate=None):
            calls.append(games_done)

    def submit(payloads):                            # old-style: NO on_result parameter
        return [WorkerResult(n_games=1, trajectories=[], source_counts={"model": 1})
                for _ in payloads]

    league = _RecordingLeague([("latest", "x.pt")])
    MP.collect_with_pool(object(), league, 2, team_pool=["A", "B"], ckpt_path="t.pt",
                         n_procs=2, chunk_size=1, submit_fn=submit,
                         save_ckpt_fn=lambda a, p: None, status=_Status())
    assert calls and calls[-1] >= 1                  # final reconcile fired; no crash
