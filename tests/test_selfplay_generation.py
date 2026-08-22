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
        self.rebases = 0

    def warmup_critic(self, trajs, n):
        self.warmups += 1
        return {"value_loss": 0.2}

    def rebase_values(self, trajs):
        # #23: gen-0 post-warmup value rebase; the fake has no critic so it's a counted no-op.
        self.rebases += 1
        return 0

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
    assert tr.rebases == 1                                      # #23: value-rebase fires with the warm-up
    assert rep["verdict"] == "promote" and rep["promoted"]
    assert len(league.snapshots) == 1 and league.latest_path == "gen0.pt"
    assert history.best_path == "gen0.pt"


def test_second_generation_no_warmup():
    ac, league, history, calls, fns = _harness([0.55, 0.62])
    tr = _FakeTrainer()
    run_generation(ac, tr, league, history, cfg=GenConfig(warmup_updates=3), **fns)
    run_generation(ac, tr, league, history, cfg=GenConfig(warmup_updates=3), **fns)
    assert tr.warmups == 1 and tr.updates == 2                  # NO warm-up on gen 1
    assert tr.rebases == 1                                      # #23: value-rebase only with the gen-0 warm-up


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


# ── __main__ entrypoint: re-exec-as-module + cp1252-safe output ────────────────
import os          # noqa: E402
import subprocess  # noqa: E402


def _run_entrypoint(mode: str, *cli, timeout: int = 120):
    """Invoke the generation entrypoint as a subprocess and return (rc, combined_out).

    ``mode`` "module" -> ``python -m v_dance.selfplay.generation``;
             "script" -> ``python <repo>/v_dance/selfplay/generation.py`` (must re-exec
             itself via -m). PYTHONIOENCODING is forced to cp1252 to reproduce the
             Windows console the utf-8 reconfigure guards against.
    """
    gen_py = _REPO / "v_dance" / "selfplay" / "generation.py"
    cmd = ([sys.executable, "-m", "v_dance.selfplay.generation", *cli] if mode == "module"
           else [sys.executable, str(gen_py), *cli])
    env = dict(os.environ, PYTHONIOENCODING="cp1252")
    p = subprocess.run(cmd, cwd=str(_REPO), env=env, capture_output=True,
                       text=True, encoding="utf-8", errors="replace", timeout=timeout)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def test_help_is_cp1252_safe_as_module():
    # regression: --help formatted a string with U+2248 (≈); a cp1252 stdout used to
    # crash with UnicodeEncodeError. The __main__ utf-8 reconfigure must prevent it.
    rc, out = _run_entrypoint("module", "--help")
    assert rc == 0, out
    assert "usage:" in out
    assert "UnicodeEncodeError" not in out


def test_script_path_reexecs_as_module():
    # regression: running the bare file path must re-exec via -m (package context +
    # mp-spawn parity) AND inherit the cp1252-safe output, producing crash-free help
    # identical to the -m invocation.
    rc_s, out_s = _run_entrypoint("script", "--help")
    rc_m, out_m = _run_entrypoint("module", "--help")
    assert rc_s == 0, out_s
    assert "usage:" in out_s
    assert "UnicodeEncodeError" not in out_s
    assert out_s == out_m                                       # re-exec -> identical help


# ── wizard: EVAL team selection (pick-your-own / choose-a-count) ───────────────
from types import SimpleNamespace  # noqa: E402


def _fresh_args():
    return SimpleNamespace(pick_eval_teams=False, eval_teams_dir=None, n_eval_teams=None)


def test_wizard_eval_team_selection_pick_via_explorer():
    args = _fresh_args()
    GN._wizard_eval_team_selection(
        args, ask=lambda *a, **k: None, ask_yn=lambda *a, **k: True, repo_root=Path("/repo"))
    assert args.pick_eval_teams is True
    assert args.n_eval_teams is None        # picking -> exact set, no count cap
    assert args.eval_teams_dir is None      # explorer, not a directory scan


def test_wizard_eval_team_selection_count_samples_full_pool():
    args = _fresh_args()
    GN._wizard_eval_team_selection(
        args, ask=lambda *a, **k: 6, ask_yn=lambda *a, **k: False, repo_root=Path("/repo"))
    assert args.pick_eval_teams is False
    assert args.n_eval_teams == 6                               # the chosen count
    assert args.eval_teams_dir is not None                     # sampled from the FULL pool
    assert "teams" in args.eval_teams_dir and "Champions" in args.eval_teams_dir


def test_wizard_eval_team_selection_default_curated():
    args = _fresh_args()
    GN._wizard_eval_team_selection(
        args, ask=lambda *a, **k: None, ask_yn=lambda *a, **k: False, repo_root=Path("/repo"))
    assert args.pick_eval_teams is False
    assert args.n_eval_teams is None        # blank count -> the curated default eval set
    assert args.eval_teams_dir is None


# ── --config run-config: override threading + fail-loud validation ─────────────
import json as _json  # noqa: E402

_GEN_DESTS = {"games", "tau", "kl_coef", "mirror_battles", "hof_champions"}


def test_build_train_configs_applies_overrides():
    ppo, train = GN.build_train_configs(
        kl_coef=0.25, tau=1.1,
        ppo_overrides={"entropy_coef": 0.05, "clip_eps": 0.25},
        train_overrides={"actor_lr": 1e-4, "ppo_epochs": 6})
    assert ppo.entropy_coef == 0.05 and ppo.clip_eps == 0.25
    assert ppo.kl_coef == 0.25 and ppo.tau == 1.1            # explicit CLI-derived still set
    assert train.actor_lr == 1e-4 and train.ppo_epochs == 6


def test_build_train_configs_explicit_beats_file_override():
    # CLI > config: an explicit kl_coef must WIN over kl_coef placed in the ppo override
    ppo, _ = GN.build_train_configs(kl_coef=0.3, ppo_overrides={"kl_coef": 0.9})
    assert ppo.kl_coef == 0.3


def test_load_run_config_valid(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text(_json.dumps({
        "_comment": "ignored",
        "generation": {"games": 400, "kl_coef": 0.25},
        "ppo": {"entropy_coef": 0.02},
        "train": {"actor_lr": 1e-4},
        "gate": {"promote_threshold": 0.6},
    }), encoding="utf-8")
    cfg = GN._load_run_config(str(p), _GEN_DESTS)
    assert cfg["generation"]["games"] == 400
    assert cfg["ppo"]["entropy_coef"] == 0.02
    assert cfg["train"]["actor_lr"] == 1e-4
    assert cfg["gate"]["promote_threshold"] == 0.6


def test_load_run_config_unknown_section_fails(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text(_json.dumps({"nonsense": {"x": 1}}), encoding="utf-8")
    with pytest.raises(SystemExit):
        GN._load_run_config(str(p), _GEN_DESTS)


def test_load_run_config_unknown_key_fails(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text(_json.dumps({"train": {"actor_lrr": 1e-4}}), encoding="utf-8")   # typo
    with pytest.raises(SystemExit):
        GN._load_run_config(str(p), _GEN_DESTS)


def test_load_run_config_unknown_generation_key_fails(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text(_json.dumps({"generation": {"not_a_flag": 1}}), encoding="utf-8")
    with pytest.raises(SystemExit):
        GN._load_run_config(str(p), _GEN_DESTS)


def test_example_config_is_valid():
    # the committed config.example.json must parse + validate against the real configs
    ex = Path(__file__).resolve().parents[1] / "config.example.json"
    if not ex.exists():
        pytest.skip("config.example.json absent")
    full_dests = {"games", "tau", "tau_start", "tau_anneal_gens", "kl_coef",
                  "mirror_battles", "hof_champions"}
    cfg = GN._load_run_config(str(ex), full_dests)
    assert "ppo" in cfg and "train" in cfg


def test_dump_config_round_trips(tmp_path):
    # --dump-config emits the FULL default config; it must be clean JSON with all 4 sections
    # and round-trip through _load_run_config (every key valid by construction).
    repo = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, "-m", "v_dance.selfplay.generation", "--dump-config"],
                       cwd=str(repo), env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=180)
    assert r.returncode == 0, r.stderr
    cfg = _json.loads(r.stdout)
    assert set(cfg) >= {"generation", "ppo", "train", "gate"}
    assert "kl_coef" in cfg["generation"] and "entropy_coef" in cfg["ppo"]
    assert "actor_lr" in cfg["train"] and "promote_threshold" in cfg["gate"]
    p = tmp_path / "dumped.json"
    p.write_text(r.stdout, encoding="utf-8")
    loaded = GN._load_run_config(str(p), set(cfg["generation"]))   # must validate
    assert loaded["train"]["actor_lr"] == cfg["train"]["actor_lr"]


# ── fs-monitor: edge tally = the COMPLEMENT of model-driven sources ────────────
def test_fs_monitor_counts_extracts_non_model_sources_with_total():
    sources = {"model": 480, "forced_switch_model": 6, "forced_default": 3, "forfeit": 1, "retry": 12}
    fs = GN.fs_monitor_counts(sources)
    assert fs["forced_default"] == 3
    assert fs["forfeit"] == 1
    assert fs["retry"] == 12                         # retry IS a model-not-driving edge event
    # model-driven sources (model / forced_switch_model) are EXCLUDED from the edge tally
    assert "model" not in fs and "forced_switch_model" not in fs
    assert fs["total"] == 16                         # 3 + 1 + 12 (every non-model source)


def test_fs_monitor_counts_captures_unknown_non_model_label():
    # a NEW non-model fallback label (not in _FS_MONITOR_KEYS) must still be surfaced (complement,
    # not a hand-list) — model_error / a future label are exactly this class
    fs = GN.fs_monitor_counts({"model": 100, "model_error": 4, "some_future_fallback": 2})
    assert fs["model_error"] == 4
    assert fs["some_future_fallback"] == 2           # dynamically captured
    assert fs["total"] == 6


def test_fs_monitor_counts_excludes_bookkeeping_and_teampreview():
    fs = GN.fs_monitor_counts({"games": 500, "tp_model": 40, "forfeit": 2})
    assert "games" not in fs and "tp_model" not in fs   # bookkeeping + teampreview excluded
    assert fs["total"] == 2


def test_fs_monitor_counts_stable_schema_when_empty():
    fs = GN.fs_monitor_counts({})
    assert set(fs) == set(GN._FS_MONITOR_KEYS) | {"total"}
    assert fs["total"] == 0
    assert all(fs[k] == 0 for k in GN._FS_MONITOR_KEYS)


def test_fs_monitor_counts_handles_none_and_nonint():
    fs = GN.fs_monitor_counts(None)
    assert fs["total"] == 0
    fs2 = GN.fs_monitor_counts({"forfeit": None, "forced_default": 2})
    assert fs2["forfeit"] == 0 and fs2["forced_default"] == 2
