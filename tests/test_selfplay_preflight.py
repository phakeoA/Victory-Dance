"""2026-09-30: the self-play PREFLIGHT (v_dance/selfplay/preflight.py) + the PLATEAU STOP (gate.plateau_stop)
+ the per-opponent finished-game count both collection paths fold into OpponentLeague.played."""
from types import SimpleNamespace

import numpy as np

from v_dance.selfplay import preflight as PF
from v_dance.selfplay.gate import GenerationHistory, GenerationRecord, plateau_stop
from v_dance.selfplay.league import OpponentLeague, LeagueConfig, opp_key
from v_dance.selfplay.mp_collect import WorkerResult, collect_with_pool, merge_results


def _hist(n, promotes):
    """``promotes`` = {gen: reason}; every other gen holds."""
    h = GenerationHistory()
    for g in range(n):
        p = g in promotes
        h.add(GenerationRecord(generation=g, n_trajectories=10, scripted_wins=1, scripted_games=2,
                               model_elo=None, verdict="promote" if p else "hold", promoted=p,
                               update_stats={"loss": 0.1}, reason=promotes.get(g, "hold")))
    return h


def _first_stop(n, promotes, patience):
    h = GenerationHistory()
    full = _hist(n, promotes)
    for r in full.records:
        h.add(r)
        if plateau_stop(h, patience):
            return r.generation
    return None


# ── the plateau stop, replayed on the two real W2 runs (manifest.json, 40 gens each) ──
ERA5B = {0: "no_baseline", 6: "beat_champion", 8: "beat_champion", 28: "beat_champion"}
BCFT = {0: "no_baseline", 2: "beat_champion", 4: "beat_champion", 23: "plateau_reanchor", 34: "plateau_reanchor"}


def test_default_patience_keeps_era5b_gen28_and_stops_bcft_at_29():
    assert _first_stop(40, ERA5B, 25) is None          # the only ladder-positive champion survives
    assert _first_stop(40, BCFT, 25) == 29             # re-anchors don't reset the clock → 10 gens saved


def test_patience_20_keeps_gen28_by_one_generation_and_19_loses_it():
    assert _first_stop(40, ERA5B, 20) is None
    assert _first_stop(40, ERA5B, 19) == 27


def test_plateau_stop_off_and_old_records_count_as_real():
    assert plateau_stop(_hist(40, {0: "no_baseline"}), 0) is None
    h = _hist(30, {0: "no_baseline", 20: "plateau_reanchor"})
    assert "last real new champion: gen 0" in (plateau_stop(h, 25) or "")
    h.records[20].reason = None                        # a pre-2026-09-30 record: no stored reason, no HoF
    assert plateau_stop(h, 25) is None                 # fail-safe: counted as a real champion


def test_record_reason_round_trips_and_old_dicts_load():
    r = GenerationRecord(generation=3, n_trajectories=1, scripted_wins=0, scripted_games=0,
                         model_elo=None, verdict="promote", promoted=True, reason="plateau_reanchor")
    assert GenerationRecord.from_obj(r.to_obj()).reason == "plateau_reanchor"
    old = r.to_obj()
    old.pop("reason")
    assert GenerationRecord.from_obj(old).reason is None


# ── coverage: rotation + finished-game counts ──
def test_cover_kinds_rotates_through_every_opponent_and_is_not_persisted():
    lg = OpponentLeague(latest_path="l.pt", clones=("a/c1.pt", "b/c1.pt"), cfg=LeagueConfig(clone_frac=0.3))
    lg.admit("g0", "g0.pt", 0)
    lg.cover_kinds = True
    rng = np.random.default_rng(0)
    got = {opp_key(k, v if k in ("clone", "scripted") else None) for k, v in (lg.sample(rng) for _ in range(7))}
    assert got == set(lg.opponent_keys()) and len(got) == 7   # latest, snapshot, 2 clones, 3 scripted
    assert "cover_kinds" not in lg.to_obj() and "played" not in lg.to_obj()


def test_clones_without_a_share_are_not_expected():
    lg = OpponentLeague(latest_path="l.pt", clones=("c.pt",), cfg=LeagueConfig(clone_frac=0.0))
    assert not any(k.startswith("clone:") for k in lg.opponent_keys())


def test_worker_counts_fold_into_the_league():
    merged = merge_results([WorkerResult(n_games=2, kind_games={"latest": 1, "clone:c.pt": 1}), None,
                            WorkerResult(n_games=1, kind_games={"clone:c.pt": 1})])
    assert merged.kind_games == {"latest": 1, "clone:c.pt": 2}
    lg = OpponentLeague(latest_path="l.pt")
    collect_with_pool(object(), lg, 2, team_pool=["t1", "t2"], ckpt_path="x.pt",
                      submit_fn=lambda payloads: [WorkerResult(n_games=2, kind_games={"scripted:random": 2})],
                      save_ckpt_fn=lambda a, p: None)
    assert lg.played == {"scripted:random": 2}


# ── the preflight ──
def _args(**kw):
    base = dict(generations=40, games=300, eval_battles=None, mirror_battles=360, hof_games=60, warmup=5,
                archive="real_archive", resume=None, resume_gen=None, hours=6, stop_after_stale=25,
                restart_server_every=20, save_replays=True, keep_snapshots=25, league_clones=["c1.pt", "c2.pt"],
                clone_frac=0.3, register_arms=False, register_prefix="era5b_g", bandit_config=None,
                hof=True, preflight="auto", preflight_only=False, run_cfg_gate={"plateau_window": 5})
    base.update(kw)
    return SimpleNamespace(**base)


def test_wanted():
    assert PF.wanted(_args()) and PF.wanted(_args(generations=0))
    assert not PF.wanted(_args(generations=1)) and not PF.wanted(_args(preflight="off"))
    assert PF.wanted(_args(generations=1, preflight="on")) and PF.wanted(_args(preflight="off", preflight_only=True))


def test_preflight_args_shrink_a_copy_and_sandbox_the_bandit(tmp_path):
    src = _args(register_arms=True)
    a = PF.preflight_args(src, tmp_path)
    assert (a.generations, a.games, a.hours, a.stop_after_stale, a.cover_kinds) == (3, 10, None, 0, True)
    assert a.run_cfg_gate["plateau_window"] == 5 and a.run_cfg_gate["min_h2h_games"] == 1
    assert a.bandit_config.startswith(str(tmp_path)) and a.register_prefix == "preflight_g"
    assert src.generations == 40 and src.bandit_config is None and src.archive == "real_archive"
    b = PF.resume_args(a)
    assert (b.generations, b.resume_gen, a.resume_gen) == (1, "latest", None)


def _result(n_gens, lg, promotes=(0, 1)):
    return {"history": _hist(n_gens, {g: "beat_champion" for g in promotes}), "league": lg}


def _league(played):
    lg = OpponentLeague(latest_path="l.pt", clones=("c1.pt",), cfg=LeagueConfig(clone_frac=0.3))
    lg.admit("g0", "g0.pt", 0)
    lg.note_played(played)
    return lg


ALL = {"latest": 3, "snapshot": 2, "clone:c1.pt": 2, "scripted:random": 1,
       "scripted:max_damage": 1, "scripted:heuristic": 1}


def test_check_passes_a_clean_run_and_names_a_silent_zero():
    ok, warns, _ = PF.check(_result(3, _league(ALL)), _result(4, _league({})))
    assert ok == [] and warns == []
    bad = dict(ALL)
    bad.pop("clone:c1.pt")
    problems, _, _ = PF.check(_result(3, _league(bad)), _result(4, _league({})))
    assert problems == ["opponent clone:c1.pt finished 0 games (its chunks failed — see the warnings above)"]


def test_check_catches_a_short_run_a_failed_resume_and_a_missing_update():
    fresh = _result(3, _league(ALL))
    fresh["history"].records[2].update_stats = {"halted": False}
    problems, _, _ = PF.check(fresh, _result(3, _league({})))
    assert any("PPO update did not run" in p for p in problems)
    assert any("resume did not continue" in p for p in problems)
    problems, _, _ = PF.check(_result(2, _league(ALL)), _result(3, _league({})))
    assert any("only 2 of 3" in p for p in problems)


def test_run_preflight_pass_removes_its_folder_and_a_crash_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(PF, "PREFLIGHT_DIR", tmp_path)
    calls = []

    def launch(a):
        calls.append((a.generations, a.resume_gen))
        return _result(3, _league(ALL)) if a.resume_gen is None else _result(4, _league({}))
    assert PF.run_preflight(_args(hof=False), launch) is True
    assert calls == [(3, None), (1, "latest")] and list(tmp_path.iterdir()) == []

    def boom(a):
        raise SystemExit(2)
    assert PF.run_preflight(_args(), boom) is False
    assert len(list(tmp_path.iterdir())) == 1          # kept for debugging
