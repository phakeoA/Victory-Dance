"""Drill league (2026-10-02) — reinforcement-learning invariants, offline.

The drill's opponent PRESSURE is a logit bias. PPO stays on-policy only if it never reaches a RECORDING player
(SelfPlayVGCPlayer: the learner on seat 0, and the 'latest' opponent whose trajectories PPO also trains on), and a
run without --drill must be untouched. These tests pin that, plus the RL-side defects found in review (xfail strict):

  * PFSP: outcomes vs PRESSURE-BIASED snapshots are folded into the league's persisted PFSP tally;
  * mirror chunks (own team vs own team) against a snapshot still get pressure, though the drill scoreboard drops them;
  * (pre-existing, reachable only on the multiprocess path the drill requires) the spawn path's ``spawn_*`` counters
    ride ``source_counts`` and sink ``model_driven_fraction`` under the 0.75 hard-fail.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from v_dance.rl.reward import model_driven_fraction
from v_dance.selfplay import mp_collect as MP
from v_dance.selfplay.mp_collect import ChunkSpec

OUR = "LG0x"


# ── fakes ──────────────────────────────────────────────────────────────────────
def _traj(tag, won=True, terminal="win"):
    return SimpleNamespace(meta=SimpleNamespace(won=won, battle_id=tag, terminal_type=terminal,
                                                is_trainable=terminal != "fallback"))


class _Player:
    def __init__(self, *, username="", trajs=None, sources=None, pressure_stats=None):
        self.username = username
        self.n_won_battles = 0
        self.n_finished_battles = 0
        self._trajs = trajs or {}
        self._source_counts = dict(sources or {})
        self.battles = {}
        if pressure_stats is not None:
            self._pressure_stats = dict(pressure_stats)

        async def _stop():
            return None
        self.ps_client = SimpleNamespace(stop_listening=_stop)

    async def battle_against(self, opp, n_battles):
        self.n_finished_battles += n_battles

    def finished_trajectories(self):
        return dict(self._trajs)

    def close(self):
        pass


def _builder(our_sources=None, opp_stats=None):
    def build(ac, spec, tau, seed, tc, live_dir=None, save_replays=False, port=None):
        tag = f"battle-{spec.uid}"
        our = _Player(username=OUR, trajs={tag: _traj(tag)}, sources=our_sources or {"model": 40})
        if spec.kind == "latest":
            opp = _Player(username="LG1x", trajs={tag: _traj(tag, won=False, terminal="loss")},
                          sources={"model": 40}, pressure_stats={"fired": 0, "taken": 0})
        else:
            opp = _Player(username="OP0y", pressure_stats=opp_stats or {"fired": 7, "taken": 5})
        return our, opp
    return build


def _collect(specs, build, **kw):
    return asyncio.run(MP._collect_specs(None, specs, tau=1.0, seed=0, team_chooser=None, battle_timeout=None,
                                         async_workers=2, build_players=build, **kw))


class _Snap:
    def __init__(self, sid, path):
        self.snapshot_id, self.path = sid, path


class _League:
    def __init__(self, samples):
        self._s, self._i = list(samples), 0

    def sample(self, rng):
        s = self._s[self._i % len(self._s)]
        self._i += 1
        return s


# ── 1. the bias never reaches a recorder (the real per-kind factory) ─────────────
@pytest.fixture
def real_factory(monkeypatch):
    """``_build_players_real`` with every poke-env / IO touch point replaced by kwarg recorders."""
    pytest.importorskip("poke_env")
    import v_dance.eval.gauntlet as G
    import v_dance.play.run_local_battle as R
    import v_dance.selfplay.game_runner as GR

    calls = {"selfplay": [], "make_player": [], "scripted": []}

    class _RecSelfPlay:
        def __init__(self, ac, **kw):
            calls["selfplay"].append(kw)

    def _make_player(name, team, **kw):
        calls["make_player"].append(kw)
        return SimpleNamespace(name=name)

    def _make_opponent(kind, name, team, **kw):
        calls["scripted"].append(kw)
        return SimpleNamespace(name=name)

    monkeypatch.setattr(GR, "SelfPlayVGCPlayer", _RecSelfPlay)
    monkeypatch.setattr(R, "make_player", _make_player)
    monkeypatch.setattr(R, "load_team", lambda p: "team-text")
    monkeypatch.setattr(R, "resolve_team_path", lambda t: t)
    monkeypatch.setattr(G, "_make_opponent", _make_opponent)
    return calls


@pytest.mark.parametrize("kind,opp_ref,snap", [("latest", None, None), ("snapshot", "g3.pt", "gen3"),
                                                ("clone", "clones/c1.pt", None), ("scripted", "max_damage", None)])
def test_pressure_never_reaches_a_recording_player_even_if_stamped(real_factory, kind, opp_ref, snap):
    # defence in depth: stamp pressure on EVERY kind by hand (build_chunk_specs never stamps latest)
    spec = ChunkSpec("own", "opp", 2, kind, opp_ref, snap, 7, 3, 0, "field", 2.5, True)
    MP._build_players_real(None, spec, 1.0, 0, None)
    for kw in real_factory["selfplay"]:                         # the learner (+ the latest opponent)
        assert "pressure" not in kw and "pressure_bias" not in kw
    assert len(real_factory["selfplay"]) == (2 if kind == "latest" else 1)
    for kw in real_factory["scripted"]:
        assert "pressure" not in kw and "pressure_bias" not in kw
    if kind in ("snapshot", "clone"):
        (kw,) = real_factory["make_player"]
        assert kw["pressure"] == "field" and kw["pressure_bias"] == 2.5
    else:
        assert real_factory["make_player"] == []


def test_unstamped_snapshot_builds_with_pressure_off(real_factory):
    spec = ChunkSpec("own", "opp", 2, "snapshot", "g3.pt", "gen3", 7, 3)
    MP._build_players_real(None, spec, 1.0, 0, None)
    (kw,) = real_factory["make_player"]
    assert kw["pressure"] is None and kw["pressure_bias"] == 0.0


def test_real_selfplay_player_refuses_any_pressure_kwarg():
    pytest.importorskip("poke_env")
    from v_dance.selfplay.game_runner import SelfPlayVGCPlayer
    for kw in ({"pressure": "field"}, {"pressure_bias": 1.0}, {"pressure_bias": float("nan")},
               {"pressure": "field", "pressure_bias": 0.0}):
        with pytest.raises(ValueError):
            SelfPlayVGCPlayer(object(), **kw)


# ── 2. the MODEL-DRIVEN guard never sees the pressure counters ───────────────────
def test_pressure_stats_never_move_the_model_driven_fraction():
    specs = [ChunkSpec("own", "opp", 1, "snapshot", "g3.pt", "gen3", u, 0, 0, "field", 2.5, True) for u in (1, 2)]
    on = _collect(specs, _builder(our_sources={"model": 40, "retry": 1}, opp_stats={"fired": 10_000, "taken": 9_000}))
    off = _collect([ChunkSpec("own", "opp", 1, "snapshot", "g3.pt", "gen3", u, 0) for u in (1, 2)],
                   _builder(our_sources={"model": 40, "retry": 1}, opp_stats={"fired": 0, "taken": 0}))
    assert on.source_counts == off.source_counts
    assert model_driven_fraction(on.source_counts) == model_driven_fraction(off.source_counts)
    assert on.pressure_stats == {"fired": 20_000, "taken": 18_000}


# ── 3. a run WITHOUT --drill is unchanged ────────────────────────────────────────
def test_no_drill_collect_with_pool_is_unchanged_by_the_new_kwargs():
    def run(**kw):
        league = _League([("snapshot", _Snap("gen3", "g3.pt")), ("latest", None)])
        league.record_result = lambda sid, won: None
        got = {}

        def submit(payloads):
            got["specs"] = [s for p in payloads for s in p[1]]
            return [MP.WorkerResult(trajectories=[1], source_counts={"model": 3}, n_games=1)]
        out = MP.collect_with_pool(None, league, 20, team_pool=["a", "b", "c"], ckpt_path="x.pt", submit_fn=submit,
                                   save_ckpt_fn=lambda a, p: None, n_procs=2, chunk_size=5, seed=1, **kw)
        return got["specs"], out
    s0, o0 = run()
    s1, o1 = run(pressure=None, pressure_bias=0.0, score=False, drill_sink=None)
    assert s0 == s1 and o0 == o1
    assert all(s.pressure is None and s.pressure_bias == 0.0 and s.score is False for s in s0)


# ── 4. defects (review 2026-10-02) ───────────────────────────────────────────────
def test_biased_snapshot_games_do_not_feed_pfsp():
    spec = ChunkSpec("own", "opp", 1, "snapshot", "g3.pt", "gen3", 1, 0, 0, "field", 2.5, True)
    res = _collect([spec], _builder())
    assert res.pfsp == []


def test_mirror_chunks_are_not_pressured():
    league = _League([("snapshot", _Snap("gen3", "g3.pt"))])
    specs = MP.build_chunk_specs(league, ["own", "rain", "grassy"], 20, chunk_size=5, own_team="own",
                                 own_mirror_frac=1.0, pressure="field", pressure_bias=2.5, score=True)
    mirrors = [s for s in specs if s.team_a == s.team_b]
    assert mirrors, "the plan must contain mirror chunks for this check"
    assert all(s.pressure is None for s in mirrors)


def test_spawn_counters_do_not_trip_the_model_driven_guard():
    async def fake_spawn(our, opp, n, *, rooms, battle_timeout, label):
        return SimpleNamespace(pairs=n, decisions=80, stalls_forfeited=0, ghosts_rescued=0, abandoned=0,
                               elapsed_s=45.0)
    specs = [ChunkSpec("own", "opp", 1, "snapshot", "g3.pt", "gen3", u, 0, 4) for u in (1, 2)]
    res = _collect(specs, _builder(), play_spawned=fake_spawn)
    assert model_driven_fraction(res.source_counts) >= 0.75
