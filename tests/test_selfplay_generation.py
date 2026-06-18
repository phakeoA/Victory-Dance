"""Task 3c.3: generation loop — promotion gate, history, and the collect -> update ->
eval -> gate -> admit/revert orchestration (with injected live steps). Pure (no
poke-env / no torch beyond a fake trainer).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
from v_dance.selfplay.league import OpponentLeague  # noqa: E402
from v_dance.selfplay import generation as GN  # noqa: E402
from v_dance.selfplay.generation import (GateConfig, GenConfig, GenerationHistory,  # noqa: E402
                                  promotion_gate, aggregate_scripted, run_generation)


# ── 3c.7b: training-team-pool resolution (explicit path needs no poke-env) ─────
def test_resolve_train_pool_explicit_is_verbatim():
    assert GN.resolve_train_pool(["Trickery", "UB_Perish_trap"]) == ["Trickery", "UB_Perish_trap"]


def test_default_eval_pool_is_controlled_curated_set():
    # sec 15: eval is a controlled CURATED set — well under the full ~71-team pool, no dups
    n = len(GN.DEFAULT_EVAL_TEAMS)
    assert 4 <= n < 30
    assert len(set(GN.DEFAULT_EVAL_TEAMS)) == n          # no duplicate teams


def test_resolve_eval_battles_auto_sizes_to_full_coverage():
    # None => 2x ordered pairs = full both-orientation side-balanced coverage (sec 15/16)
    assert GN.resolve_eval_battles(None, 6) == 2 * 6 * 5         # 60
    assert GN.resolve_eval_battles(None, 8) == 2 * 8 * 7         # 112
    assert GN.resolve_eval_battles(None, 1) == 2                 # degenerate-pool guard


def test_resolve_eval_battles_explicit_passthrough():
    assert GN.resolve_eval_battles(12, 6) == 12                  # fast-smoke override untouched


# ── 3c.7c: collection tau exploration anneal ──────────────────────────────────
def test_tau_for_generation_anneals_linearly():
    assert GN.tau_for_generation(0, 1.3, 1.0, 12) == pytest.approx(1.3)   # gen0 = start
    assert GN.tau_for_generation(12, 1.3, 1.0, 12) == pytest.approx(1.0)  # reaches end
    assert GN.tau_for_generation(24, 1.3, 1.0, 12) == pytest.approx(1.0)  # held after
    assert GN.tau_for_generation(6, 1.3, 1.0, 12) == pytest.approx(1.15)  # midpoint linear


def test_tau_for_generation_flat_when_disabled():
    assert GN.tau_for_generation(5, 1.3, 1.0, 0) == pytest.approx(1.0)    # anneal off -> end
    assert GN.tau_for_generation(5, 1.0, 1.0, 12) == pytest.approx(1.0)   # equal endpoints


def test_tau_for_generation_monotone_nonincreasing_within_bounds():
    taus = [GN.tau_for_generation(g, 1.3, 1.0, 10) for g in range(15)]
    assert taus[0] == pytest.approx(1.3) and taus[-1] == pytest.approx(1.0)
    assert all(1.0 - 1e-9 <= t <= 1.3 + 1e-9 for t in taus)              # within [end, start]
    assert all(a >= b - 1e-9 for a, b in zip(taus, taus[1:]))            # never increases


def test_target_kl_for_generation_relax_and_cap():
    # default relax=0 -> the bar is the base, unchanged at every gen (no behaviour change)
    assert GN.target_kl_for_generation(0, 0.15) == pytest.approx(0.15)
    assert GN.target_kl_for_generation(50, 0.15) == pytest.approx(0.15)
    # linear relaxation per gen
    assert GN.target_kl_for_generation(0, 0.15, 0.01) == pytest.approx(0.15)
    assert GN.target_kl_for_generation(10, 0.15, 0.01) == pytest.approx(0.25)
    # cap clamps the relaxed bar
    assert GN.target_kl_for_generation(100, 0.15, 0.01, cap=0.30) == pytest.approx(0.30)
    # base None (guard off) stays off regardless of relax
    assert GN.target_kl_for_generation(10, None, 0.01) is None


# ── 3c.8c: parallel-collection chunk planner ──────────────────────────────────
def test_build_collection_chunks_plans_all_games_unique_uids():
    from v_dance.selfplay.league import OpponentLeague
    lg = OpponentLeague(latest_path="x")                     # no snapshots -> latest/scripted
    pool = [f"T{i}" for i in range(71)]
    chunks = GN.build_collection_chunks(lg, pool, 12, chunk_size=10, matchup_seed=0, seed=0)
    assert sum(c["cn"] for c in chunks) == 12                # plans exactly n_games battles
    assert len({c["uid"] for c in chunks}) == len(chunks)    # unique uids (no account clash)
    assert all(c["cn"] == 1 for c in chunks)                 # big pool -> 1 battle/pairing
    assert all(c["spec"][0] in ("latest", "snapshot", "scripted") for c in chunks)


def test_build_collection_chunks_seed_reproducible():
    from v_dance.selfplay.league import OpponentLeague
    lg = OpponentLeague(latest_path="x")
    pool = [f"T{i}" for i in range(8)]
    key = lambda cs: [(c["team_a"], c["team_b"], c["cn"], c["spec"][0]) for c in cs]
    a = GN.build_collection_chunks(lg, pool, 20, chunk_size=10, matchup_seed=1, seed=2)
    b = GN.build_collection_chunks(lg, pool, 20, chunk_size=10, matchup_seed=1, seed=2)
    assert key(a) == key(b)                                  # deterministic plan (resumable)


# ── promotion gate ────────────────────────────────────────────────────────────
def test_gate_no_baseline_promotes():
    v, st = promotion_gate(50, 100, 0, 0)
    assert v == "promote" and st["reason"] == "no_baseline"


def test_gate_significant_improvement_promotes():
    # 0.65 vs 0.50 over 400 games each -> clearly beyond noise
    v, st = promotion_gate(260, 400, 200, 400, GateConfig(z=1.0))
    assert v == "promote" and st["delta"] == pytest.approx(0.15)


def test_gate_within_noise_holds():
    # 0.52 vs 0.50 over 120 games -> not significant
    v, _ = promotion_gate(62, 120, 60, 120, GateConfig(z=1.0))
    assert v == "hold"


def test_gate_significant_regression_reverts():
    v, _ = promotion_gate(220, 400, 260, 400, GateConfig(z=1.0))   # 0.55 vs 0.65
    assert v == "revert"


def test_gate_no_revert_when_disabled():
    v, _ = promotion_gate(220, 400, 260, 400, GateConfig(z=1.0, revert_on_regression=False))
    assert v == "hold"


def test_aggregate_scripted():
    results = {"random": (30, 40), "max_damage": (20, 40), "heuristic": (10, 40),
               "prev_best": (5, 10)}                                # mirror excluded
    assert aggregate_scripted(results) == (60, 120)


# ── history ───────────────────────────────────────────────────────────────────
def test_history_round_trip_and_elo_curve():
    h = GenerationHistory()
    h.add(GN.GenerationRecord(0, 200, 60, 120, 1040.0, "promote", True))
    h.add(GN.GenerationRecord(1, 200, 70, 120, 1080.0, "hold", False))
    h.best_path, h.best_scripted = "gen0.pt", (60, 120)
    assert h.generation == 2
    assert h.elo_curve() == [(0, 1040.0), (1, 1080.0)]
    back = GenerationHistory.from_obj(h.to_obj())
    assert back.generation == 2 and back.best_path == "gen0.pt"
    assert back.best_scripted == (60, 120) and back.records[1].verdict == "hold"


# ── orchestration (injected live steps) ───────────────────────────────────────
class _FakeTrainer:
    def __init__(self):
        self.warmups = 0
        self.updates = 0

    def warmup_critic(self, trajs, n):
        self.warmups += 1
        return {"value_loss": 0.2}

    def ppo_update(self, trajs):
        self.updates += 1
        return {"loss": 0.1, "halted": False}


def _harness(win_rates, eval_games=400):
    """Build the injected fns; eval returns the next win-rate from ``win_rates``."""
    ac = object()
    league = OpponentLeague(latest_path="bc_best.pt")
    history = GenerationHistory()
    calls = {"restore": [], "saved": []}

    def collect_fn(ac, lg, gen):
        return [object()] * 10, {"model": 100}

    def save_fn(ac, gen):
        p = f"gen{gen}.pt"
        calls["saved"].append(p)
        return p

    def eval_fn(path, prev_best_path=None):
        gen = len(history.records)
        wr = win_rates[gen]
        return {"random": (round(wr * eval_games), eval_games)}, 1000 + (wr - 0.5) * 800

    def restore_fn(ac, path):
        calls["restore"].append(path)

    return ac, league, history, calls, dict(
        collect_fn=collect_fn, save_fn=save_fn, eval_fn=eval_fn, restore_fn=restore_fn)


def test_first_generation_warms_up_and_promotes():
    ac, league, history, calls, fns = _harness([0.55])
    tr = _FakeTrainer()
    rep = run_generation(ac, tr, league, history, cfg=GenConfig(warmup_updates=3), **fns)
    assert tr.warmups == 1 and tr.updates == 1                  # warm-up only on gen 0
    assert rep["verdict"] == "promote" and rep["promoted"]
    assert len(league.snapshots) == 1 and league.latest_path == "gen0.pt"
    assert history.best_path == "gen0.pt"


def test_second_generation_no_warmup():
    ac, league, history, calls, fns = _harness([0.55, 0.62])
    tr = _FakeTrainer()
    run_generation(ac, tr, league, history, cfg=GenConfig(warmup_updates=3), **fns)
    run_generation(ac, tr, league, history, cfg=GenConfig(warmup_updates=3), **fns)
    assert tr.warmups == 1 and tr.updates == 2                  # NO warm-up on gen 1


def test_hold_admits_snapshot_but_keeps_champion():
    """Decoupled admission (sec 16): a HELD (competent) gen is still admitted as a league
    opponent for PFSP diversity — but it is NOT marked champion, so the champion pointer and
    high-water are unchanged and there's no revert."""
    ac, league, history, calls, fns = _harness([0.55, 0.555])   # gen1 within noise of gen0
    tr = _FakeTrainer()
    run_generation(ac, tr, league, history, **fns)              # gen0 promote
    rep = run_generation(ac, tr, league, history, **fns)        # gen1 hold
    assert rep["verdict"] == "hold" and not rep["promoted"]
    assert len(league.snapshots) == 2                           # gen1 ADMITTED (decoupled)
    assert sum(s.is_champion for s in league.snapshots) == 1    # only gen0 is the champion
    assert calls["restore"] == []                               # no revert
    assert history.best_path == "gen0.pt"                       # champion unchanged


def test_decoupled_admission_prunes_and_cleans_up(monkeypatch):
    """Over many held gens the league admits every competent gen but stays BOUNDED by prune,
    and cleanup_fn is handed the evicted snapshots (so the live runner can delete their files).
    A reverted/collapsed gen is NOT admitted."""
    from v_dance.selfplay.generation import run_generation, GenConfig, GenerationHistory
    from v_dance.selfplay.league import OpponentLeague
    league = OpponentLeague(latest_path="bc.pt")
    history = GenerationHistory()
    history.best_path = "champ.pt"
    history.best_scripted = (180, 200)
    history.scripted_high_water = 0.60
    evicted_seen = []

    def eval_fn(path, prev_best_path):
        # healthy scripted + a flat 55% mirror (below the 70% bar, not a collapse) → mostly HOLD
        return ({"random": (60, 67), "max_damage": (60, 67), "heuristic": (60, 66),
                 "prev_best": (132, 240)}, 1500.0)

    cfg = GenConfig(league_cap=5, keep_recent=2)
    for i in range(12):
        run_generation(object(), _FakeTrainer(), league, history,
                       collect_fn=lambda ac, lg, gen: ([], {}), eval_fn=eval_fn,
                       save_fn=lambda ac, gen: f"g{len(history.records)}.pt",
                       cleanup_fn=lambda ev: evicted_seen.extend(ev), cfg=cfg)

    assert len(league.snapshots) <= 5            # bounded by the cap despite 12 admits
    assert evicted_seen                          # eviction happened and cleanup_fn was called
    # the champion (a plateau re-anchor will have occurred) survives in the pool
    assert any(s.is_champion for s in league.snapshots)


def test_regression_reverts_to_best():
    ac, league, history, calls, fns = _harness([0.65, 0.50])    # gen1 clear regression
    tr = _FakeTrainer()
    run_generation(ac, tr, league, history, **fns)              # gen0 promote, best=gen0
    rep = run_generation(ac, tr, league, history, **fns)        # gen1 revert
    assert rep["verdict"] == "revert" and not rep["promoted"]
    assert calls["restore"] == ["gen0.pt"]                      # restored from the best
    assert len(league.snapshots) == 1


def test_run_generation_writes_live_status(tmp_path):
    from v_dance.selfplay.status import LiveStatus, read_status
    ac, league, history, calls, fns = _harness([0.55])
    tr = _FakeTrainer()
    ls = LiveStatus(tmp_path / "status.json")
    rep = run_generation(ac, tr, league, history, status=ls,
                         cfg=GenConfig(warmup_updates=0), **fns)
    s = read_status(ls.path)
    assert s["run"]["phase"] == "evaluating"                 # last phase run_generation sets
    assert s["run"]["generation"] == 0
    assert s["update"]["loss"] == pytest.approx(0.1) and s["update"]["halted"] == 0.0
    assert s["run"]["last_verdict"] == rep["verdict"]        # verdict captured after the gate


def test_dry_run_smoke(capsys):
    GN._dry_run(n_generations=6)
    out = capsys.readouterr().out
    assert "Generation-loop dry run" in out
    assert "PROMOTE" in out and "HOLD" in out and "REVERT" in out   # all verdicts shown
    assert "league snapshots admitted" in out


def test_live_wiring_importable():
    """The live functions exist with the expected signatures (exercised live by the smoke)."""
    import inspect
    for fn in ("collect_with_league", "gauntlet_eval", "run_live_generations"):
        assert hasattr(GN, fn)
    assert inspect.iscoroutinefunction(GN.collect_with_league)
    assert set(inspect.signature(GN.run_live_generations).parameters) >= {
        "ckpt", "n_generations", "team_pool", "team_chooser", "archive_dir", "n_servers"}


# ── 22f.4: staggered multi-server recycle schedule ────────────────────────────
def test_server_recycle_index_single_server_matches_legacy():
    """K=1 must reproduce the old behaviour: recycle the (only) server every N gens, nothing else."""
    fired = [g for g in range(1, 61) if GN.server_recycle_index(g, 20, 1) is not None]
    assert fired == [20, 40, 60]                                # every 20 gens
    assert all(GN.server_recycle_index(g, 20, 1) == 0 for g in fired)   # always the single server


def test_server_recycle_index_staggers_a_pool():
    """K=2, N=20 → stride 10: one server every 10 gens, ROUND-ROBIN, so each server is refreshed
    every 20 gens and they never recycle in the same gen (no all-down stall)."""
    sched = {g: GN.server_recycle_index(g, 20, 2) for g in range(1, 61)}
    assert sched[10] == 0 and sched[20] == 1 and sched[30] == 0 and sched[40] == 1
    fired = {g: i for g, i in sched.items() if i is not None}
    assert set(fired) == {10, 20, 30, 40, 50, 60}              # one recycle every stride (10) gens
    # each individual server is recycled every 20 gens (every OTHER stride)
    assert [g for g, i in fired.items() if i == 0] == [10, 30, 50]
    assert [g for g, i in fired.items() if i == 1] == [20, 40, 60]


def test_server_recycle_index_disabled_and_gen0():
    assert GN.server_recycle_index(0, 20, 3) is None           # gen 0 boundary: never
    assert GN.server_recycle_index(20, 0, 3) is None           # restart_server_every=0 disables it
