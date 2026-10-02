"""Drill league (2026-10-02): the COLLECTION + RECORD plumbing, offline.

Covers the path a drill's data travels: ``ChunkSpec`` (pressure / score stamps, picklable) ->
``build_chunk_specs`` (pressure ONLY on non-recording model opponents) -> the worker's
``_collect_specs`` (battle logs -> ``drill_rows``, opponent ``_pressure_stats`` -> ``pressure_stats``,
never ``source_counts``) -> ``merge_results`` -> ``collect_with_pool``'s ``drill_sink`` -> the
generation record (``GenerationRecord.drill``) -> the manifest, the preflight check, the per-gen
report line and the CLI. No server, no GPU, no network: players are fakes with ``battles[tag]._replay_data``
shaped like poke-env's (a list of split protocol messages)."""
from __future__ import annotations

import asyncio
import inspect
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from v_dance.selfplay import mp_collect as MP
from v_dance.selfplay import preflight as PF
from v_dance.selfplay.archive import build_manifest
from v_dance.selfplay.gate import GenerationHistory, GenerationRecord
from v_dance.selfplay.league import LeagueConfig, OpponentLeague
from v_dance.selfplay.mp_collect import ChunkSpec, WorkerResult

_REPO_ROOT = Path(__file__).resolve().parents[1]
OUR = "LG0x"


# ── fakes ──────────────────────────────────────────────────────────────────────
def _replay_data(our: str = OUR, opp: str = "OP0y", winner: str = OUR, our_mon: str = "Salamence",
                 opp_mon: str = "Rillaboom"):
    """poke-env ``battle._replay_data``: each protocol message stored as its ``|``-split list."""
    return [["", "player", "p1", our, "1"],
            ["", "player", "p2", opp, "2"],
            ["", "switch", "p1a: " + our_mon, our_mon + ", L50, M", "100/100"],
            ["", "switch", "p2a: " + opp_mon, opp_mon + ", L50, F", "100/100"],
            ["", "turn", "1"],
            ["", "win", winner]]


def _traj(tag, won=True, terminal="win"):
    return SimpleNamespace(meta=SimpleNamespace(won=won, battle_id=tag, terminal_type=terminal))


class _Player:
    """Minimal stand-in for a poke-env player as ``_collect_specs`` / ``_drill_rows`` touch it."""

    def __init__(self, *, username="", trajs=None, sources=None, battles=None,
                 pressure_stats=None, forfeited=None):
        self.username = username
        self.n_won_battles = 0
        self.n_finished_battles = 0
        self.closed = False
        self._trajs = trajs or {}
        self._source_counts = sources or {}
        self.battles = battles if battles is not None else {}
        if pressure_stats is not None:
            self._pressure_stats = dict(pressure_stats)
        if forfeited is not None:
            self._forfeited_tags = set(forfeited)

        async def _stop():
            return None
        self.ps_client = SimpleNamespace(stop_listening=_stop)

    async def battle_against(self, opp, n_battles):
        self.n_finished_battles += n_battles

    def finished_trajectories(self):
        return dict(self._trajs)

    def close(self):
        self.closed = True


def _builder(*, opp_stats=None, forfeit_uids=(), replay=None, made=None):
    """A ``build_players`` fake: our recorder ``LG0x`` with one finished game (tag ``battle-<uid>``)
    whose ``_replay_data`` is a real-shaped log; the opponent carries ``_pressure_stats`` (model
    kinds) and ``_forfeited_tags`` for the uids in ``forfeit_uids``."""
    opp_stats = {"fired": 2, "taken": 1} if opp_stats is None else opp_stats

    def build(ac, spec, tau, seed, tc, live_dir=None, save_replays=False, port=None):
        tag = f"battle-{spec.uid}"
        data = replay if replay is not None else _replay_data()
        our = _Player(username=OUR, trajs={tag: _traj(tag)}, sources={"model": 5},
                      battles={tag: SimpleNamespace(_replay_data=data)})
        ff = {tag} if spec.uid in forfeit_uids else None
        if spec.kind == "latest":       # a SelfPlayVGCPlayer: zero pressure (refused), its own trajectory
            opp = _Player(username="LG0y", trajs={tag + "o": _traj(tag + "o", won=False, terminal="loss")},
                          sources={"model": 3}, pressure_stats={"fired": 0, "taken": 0}, forfeited=ff)
        elif spec.kind in ("snapshot", "clone"):
            opp = _Player(username="OP0y", pressure_stats=opp_stats, forfeited=ff)
        else:                           # scripted: no model, no pressure stats
            opp = _Player(username="OP0y", forfeited=ff)
        if made is not None:
            made.append((spec, our, opp))
        return our, opp
    return build


def _collect(specs, build):
    return asyncio.run(MP._collect_specs(
        None, specs, tau=1.0, seed=0, team_chooser=None, battle_timeout=None,
        async_workers=2, build_players=build))


class _Snap:
    def __init__(self, sid, path):
        self.snapshot_id, self.path = sid, path


class _CyclingLeague:
    """Canned ``sample()`` results in order, round-robin; records PFSP calls (collect_with_pool)."""

    def __init__(self, samples):
        self._s, self._i, self.recorded = list(samples), 0, []

    def sample(self, rng):
        s = self._s[self._i % len(self._s)]
        self._i += 1
        return s

    def record_result(self, snapshot_id, won):
        self.recorded.append((snapshot_id, won))


def _four_kinds():
    return _CyclingLeague([("latest", "latest.pt"), ("snapshot", _Snap("gen3", "g3.pt")),
                           ("clone", "clones/c1.pt"), ("scripted", "max_damage")])


# ── ChunkSpec ──────────────────────────────────────────────────────────────────
def test_chunkspec_drill_fields_default_off():
    s = ChunkSpec("A", "B", 3, "snapshot", "g3.pt", "gen3", 7)       # the old 7-positional ctor
    assert (s.pressure, s.pressure_bias, s.score) == (None, 0.0, False)
    assert (s.gen, s.spawn_rooms) == (0, 0)


def test_chunkspec_full_positional_order_is_append_only():
    s = ChunkSpec("A", "B", 2, "clone", "c.pt", None, 4, 5, 6, "field", 2.5, True)
    assert (s.gen, s.spawn_rooms) == (5, 6)
    assert (s.pressure, s.pressure_bias, s.score) == ("field", 2.5, True)


def test_chunkspec_with_pressure_pickles():
    s = ChunkSpec("A", "B", 2, "clone", "c.pt", None, 4, gen=3, pressure="field", pressure_bias=2.5, score=True)
    back = pickle.loads(pickle.dumps(s))
    assert back == s
    assert (back.pressure, back.pressure_bias, back.score) == ("field", 2.5, True)


def test_worker_result_drill_fields_default_empty_and_pickle():
    r = WorkerResult()
    assert r.drill_rows == [] and r.pressure_stats == {}
    assert WorkerResult().drill_rows is not r.drill_rows               # no shared mutable default
    full = WorkerResult(n_games=1, drill_rows=[{"our": OUR, "text": "|win|LG0x", "mirror": False}],
                        pressure_stats={"fired": 3, "taken": 1})
    assert pickle.loads(pickle.dumps(full)) == full


# ── build_chunk_specs: pressure only on snapshot + clone; score everywhere ──────
def test_build_chunk_specs_stamps_pressure_only_on_snapshot_and_clone():
    specs = MP.build_chunk_specs(_four_kinds(), ["A", "B"], 8, chunk_size=1, pressure="field",
                                 pressure_bias=2.5, score=True)
    assert len(specs) == 8
    assert {s.kind for s in specs} == {"latest", "snapshot", "clone", "scripted"}
    for s in specs:
        assert s.score is True, s.kind
        if s.kind in ("snapshot", "clone"):
            assert (s.pressure, s.pressure_bias) == ("field", 2.5), s.kind
        else:
            assert (s.pressure, s.pressure_bias) == (None, 0.0), s.kind
    clone = next(s for s in specs if s.kind == "clone")
    assert clone.opp_ref == "clones/c1.pt"


@pytest.mark.parametrize("pressure,bias", [("field", 0.0), ("field", -1.0), ("field", float("nan")),
                                           (None, 2.5), ("", 2.5)])
def test_build_chunk_specs_no_pressure_when_off(pressure, bias):
    specs = MP.build_chunk_specs(_four_kinds(), ["A", "B"], 8, chunk_size=1, pressure=pressure,
                                 pressure_bias=bias, score=True)
    assert all(s.pressure is None and s.pressure_bias == 0.0 for s in specs)
    assert all(s.score is True for s in specs)


def test_build_chunk_specs_defaults_stamp_nothing():
    specs = MP.build_chunk_specs(_four_kinds(), ["A", "B"], 8, chunk_size=1)
    assert all((s.pressure, s.pressure_bias, s.score) == (None, 0.0, False) for s in specs)


def test_build_chunk_specs_pressure_does_not_change_the_plan():
    """Pressure is a flag on existing kinds: the pairings, kinds, uids and opponents are identical."""
    plain = MP.build_chunk_specs(_four_kinds(), ["A", "B", "C"], 9, chunk_size=1, seed=3, matchup_seed=1)
    drill = MP.build_chunk_specs(_four_kinds(), ["A", "B", "C"], 9, chunk_size=1, seed=3, matchup_seed=1,
                                 pressure="field", pressure_bias=2.5, score=True)
    key = [(s.team_a, s.team_b, s.n, s.kind, s.opp_ref, s.snapshot_id, s.uid) for s in plain]
    assert key == [(s.team_a, s.team_b, s.n, s.kind, s.opp_ref, s.snapshot_id, s.uid) for s in drill]


# ── _drill_rows (pure helper) ───────────────────────────────────────────────────
def _our_with(tag, data=None, battle_key=None):
    return _Player(username=OUR, trajs={tag: _traj(tag)},
                   battles={battle_key or tag: SimpleNamespace(_replay_data=data or _replay_data())})


def test_drill_rows_builds_the_row_from_replay_data():
    tag = "battle-gen9-1"
    our = _our_with(tag)
    spec = ChunkSpec("Own", "OppTeam", 1, "clone", "clones/c1.pt", None, 1, pressure="field", pressure_bias=2.5,
                     score=True)
    rows = MP._drill_rows(our, _Player(), our.finished_trajectories(), spec)
    assert len(rows) == 1
    r = rows[0]
    assert r["our"] == OUR
    assert r["text"].splitlines()[0] == "|player|p1|LG0x|1"
    assert "|win|LG0x" in r["text"].splitlines()
    assert (r["kind"], r["opp_ref"], r["team_b"]) == ("clone", "clones/c1.pt", "OppTeam")
    assert (r["mirror"], r["fallback"], r["pressure"]) == (False, False, True)
    json.dumps(rows, allow_nan=False)                                   # rows are plain JSON data


@pytest.mark.parametrize("kind,opp_ref,want", [("latest", None, None), ("snapshot", "g3.pt", None),
                                               ("clone", "c.pt", "c.pt"), ("scripted", "max_damage", "max_damage")])
def test_drill_rows_opp_ref_only_for_clone_and_scripted(kind, opp_ref, want):
    tag = "b1"
    our = _our_with(tag)
    rows = MP._drill_rows(our, _Player(), our.finished_trajectories(),
                          ChunkSpec("A", "B", 1, kind, opp_ref, None, 1))
    assert rows[0]["opp_ref"] == want and rows[0]["kind"] == kind and rows[0]["pressure"] is False


def test_drill_rows_mirror_flag():
    tag = "b1"
    our = _our_with(tag)
    rows = MP._drill_rows(our, _Player(), our.finished_trajectories(),
                          ChunkSpec("Own", "Own", 1, "snapshot", "g.pt", "g", 1))
    assert rows[0]["mirror"] is True


def test_drill_rows_fallback_from_terminal_type_or_opponent_forfeit():
    tag = "b1"
    our = _our_with(tag)
    trajs = our.finished_trajectories()
    spec = ChunkSpec("A", "B", 1, "latest", None, None, 1)
    assert MP._drill_rows(our, _Player(), trajs, spec)[0]["fallback"] is False
    assert MP._drill_rows(our, _Player(forfeited={tag}), trajs, spec)[0]["fallback"] is True
    assert MP._drill_rows(our, _Player(forfeited={"other"}), trajs, spec)[0]["fallback"] is False
    trajs[tag].meta.terminal_type = "fallback"                          # our own backstop forfeit
    assert MP._drill_rows(our, _Player(), trajs, spec)[0]["fallback"] is True


def test_drill_rows_normalizes_a_leading_gt_in_the_tag():
    """Trajectory keyed ``>battle-1`` finds ``battles['battle-1']``; a forfeit recorded as the
    normalized tag still flags it."""
    our = _our_with(">battle-1", battle_key="battle-1")
    rows = MP._drill_rows(our, _Player(forfeited={"battle-1"}), our.finished_trajectories(),
                          ChunkSpec("A", "B", 1, "scripted", "random", None, 1))
    assert len(rows) == 1 and rows[0]["fallback"] is True


def test_drill_rows_skips_games_without_a_battle_and_tolerates_missing_attrs():
    our = _our_with("b1")
    trajs = {"b1": _traj("b1"), "b2": _traj("b2")}                      # b2 has no battle object
    rows = MP._drill_rows(our, None, trajs, ChunkSpec("A", "B", 2, "latest", None, None, 1))
    assert len(rows) == 1                                               # opp=None is fine too
    bare = SimpleNamespace(username=OUR)                                # no .battles at all
    assert MP._drill_rows(bare, None, trajs, ChunkSpec("A", "B", 2, "latest", None, None, 1)) == []
    empty = SimpleNamespace(username=OUR, battles={"b1": SimpleNamespace()})   # no _replay_data yet
    assert MP._drill_rows(empty, None, {"b1": _traj("b1")},
                          ChunkSpec("A", "B", 1, "latest", None, None, 1))[0]["text"] == ""


# ── _collect_specs: rows + pressure stats come home, source_counts stays clean ──
def test_collect_specs_score_true_returns_rows_and_folds_pressure_stats():
    spec = ChunkSpec("Own", "OppTeam", 1, "clone", "clones/c1.pt", None, 1, pressure="field",
                     pressure_bias=2.5, score=True)
    res = _collect([spec], _builder())
    assert len(res.drill_rows) == 1
    row = res.drill_rows[0]
    assert "|player|p1|LG0x" in row["text"]
    assert (row["kind"], row["opp_ref"], row["team_b"]) == ("clone", "clones/c1.pt", "OppTeam")
    assert (row["mirror"], row["fallback"], row["pressure"]) == (False, False, True)
    assert res.pressure_stats == {"fired": 2, "taken": 1}
    assert res.source_counts == {"model": 5}                            # NEVER the pressure counters
    assert res.n_games == 1 and len(res.trajectories) == 1


def test_collect_specs_score_false_returns_no_rows():
    spec = ChunkSpec("Own", "OppTeam", 1, "snapshot", "g3.pt", "gen3", 1, pressure="field", pressure_bias=2.5)
    res = _collect([spec], _builder())
    assert res.drill_rows == []
    assert res.pressure_stats == {"fired": 2, "taken": 1}               # stats fold regardless of score
    assert "fired" not in res.source_counts and "taken" not in res.source_counts
    assert res.pfsp == []            # review RL-1: a PRESSURED snapshot game never feeds PFSP (it is not the real snapshot)


def test_collect_specs_mirror_and_forfeit_flags_through_the_worker():
    specs = [ChunkSpec("Own", "Own", 1, "snapshot", "g3.pt", "gen3", 1, score=True),       # mirror
             ChunkSpec("Own", "Opp", 1, "latest", None, None, 2, score=True),             # opp forfeits
             ChunkSpec("Own", "Opp", 1, "scripted", "max_damage", None, 3, score=True)]   # opp forfeits
    made = []
    res = _collect(specs, _builder(forfeit_uids=(2, 3), made=made))
    rows = {r["kind"]: r for r in res.drill_rows}
    assert set(rows) == {"snapshot", "latest", "scripted"}
    assert rows["snapshot"]["mirror"] is True and rows["snapshot"]["fallback"] is False
    # latest: the collector does NOT re-tag our trajectory (the opp recorder's own FALLBACK does), so the
    # row's fallback flag must come from the opponent's _forfeited_tags
    assert rows["latest"]["fallback"] is True and rows["latest"]["mirror"] is False
    latest_our = next(o for s, o, _ in made if s.kind == "latest")
    assert latest_our._trajs["battle-2"].meta.terminal_type == "win"
    assert rows["scripted"]["fallback"] is True
    assert rows["scripted"]["opp_ref"] == "max_damage"


def test_collect_specs_pressure_stats_sum_across_chunks_and_never_touch_source_counts():
    specs = [ChunkSpec("A", "B", 1, "clone", "c1.pt", None, 1, pressure="field", pressure_bias=2.5, score=True),
             ChunkSpec("A", "B", 1, "snapshot", "g.pt", "g", 2, pressure="field", pressure_bias=2.5, score=True),
             ChunkSpec("A", "B", 1, "latest", None, None, 3, score=True),
             ChunkSpec("A", "B", 1, "scripted", "random", None, 4, score=True)]
    res = _collect(specs, _builder(opp_stats={"fired": 3, "taken": 2}))
    assert res.pressure_stats == {"fired": 6, "taken": 4}
    assert res.source_counts == {"model": 5 * 4 + 3}                    # our x4 + the latest opp recorder
    assert len(res.drill_rows) == 4
    assert sorted(r["pressure"] for r in res.drill_rows) == [False, False, True, True]


def test_collect_specs_a_broken_log_is_non_fatal():
    """A battle log that cannot be joined (non-str entries) loses only its drill row — the game is
    still collected and the players still closed."""
    made = []
    res = _collect([ChunkSpec("A", "B", 1, "clone", "c.pt", None, 1, score=True)],
                   _builder(replay=[["", 1, None]], made=made))
    assert res.drill_rows == []
    assert len(res.trajectories) == 1 and res.n_games == 1
    _, our, opp = made[0]
    assert our.closed and opp.closed


# ── merge_results ──────────────────────────────────────────────────────────────
def test_merge_results_concatenates_rows_sums_stats_skips_none():
    a, b, c = {"our": "a"}, {"our": "b"}, {"our": "c"}
    merged = MP.merge_results([WorkerResult(n_games=1, drill_rows=[a], pressure_stats={"fired": 1, "taken": 1}),
                               None,
                               WorkerResult(n_games=2, drill_rows=[b, c], pressure_stats={"fired": 2})])
    assert merged.drill_rows == [a, b, c]
    assert merged.pressure_stats == {"fired": 3, "taken": 1}
    assert merged.n_games == 3


def test_merge_results_tolerates_old_style_results_and_empty_input():
    old = SimpleNamespace(trajectories=["t"], source_counts={"model": 1}, pfsp=[], n_games=1)   # pre-drill shape
    merged = MP.merge_results([old, None])
    assert merged.drill_rows == [] and merged.pressure_stats == {} and merged.trajectories == ["t"]
    empty = MP.merge_results([])
    assert empty.drill_rows == [] and empty.pressure_stats == {} and empty.n_games == 0


# ── collect_with_pool: the drill sink ───────────────────────────────────────────
def test_collect_with_pool_calls_drill_sink_once_and_returns_a_two_tuple():
    calls, captured = [], {}

    def submit(payloads):
        captured["payloads"] = payloads
        return [WorkerResult(trajectories=[f"t{i}"], source_counts={"model": 1}, n_games=1,
                             drill_rows=[{"our": OUR, "i": i}], pressure_stats={"fired": 1, "taken": i})
                for i, _ in enumerate(payloads)]

    out = MP.collect_with_pool(object(), _four_kinds(), 8, team_pool=["A", "B"], ckpt_path="t.pt",
                               n_procs=2, chunk_size=1, submit_fn=submit, save_ckpt_fn=lambda a, p: None,
                               pressure="field", pressure_bias=2.5, score=True,
                               drill_sink=lambda rows, ps: calls.append((rows, ps)))
    assert isinstance(out, tuple) and len(out) == 2
    trajs, src = out
    assert sorted(trajs) == ["t0", "t1"] and src == {"model": 2}
    assert len(calls) == 1
    rows, ps = calls[0]
    assert rows == [{"our": OUR, "i": 0}, {"our": OUR, "i": 1}]
    assert ps == {"fired": 2, "taken": 1}
    # the plan reached the workers: pressure only on snapshot/clone, score on all; payload shape unchanged
    specs = [s for p in captured["payloads"] for s in p[1]]
    assert all(len(p) == 10 for p in captured["payloads"])
    assert {s.kind for s in specs} == {"latest", "snapshot", "clone", "scripted"}
    assert all(s.score for s in specs)
    assert all((s.pressure == "field") == (s.kind in ("snapshot", "clone")) for s in specs)


def test_collect_with_pool_sink_gets_empty_data_when_every_worker_died():
    calls = []
    trajs, src = MP.collect_with_pool(object(), _four_kinds(), 2, team_pool=["A", "B"], ckpt_path="t.pt",
                                      n_procs=2, chunk_size=1, submit_fn=lambda payloads: [None for _ in payloads],
                                      save_ckpt_fn=lambda a, p: None, score=True,
                                      drill_sink=lambda rows, ps: calls.append((rows, ps)))
    assert calls == [([], {})]
    assert trajs == [] and src == {}


def test_collect_with_pool_sink_failure_is_non_fatal_and_no_sink_is_fine():
    def submit(payloads):
        return [WorkerResult(trajectories=["t"], source_counts={"model": 1}, n_games=1, pfsp=[("gen3", True)],
                             drill_rows=[{"our": OUR}]) for _ in payloads]

    def boom(rows, ps):
        raise RuntimeError("scoreboard exploded")

    league = _CyclingLeague([("snapshot", _Snap("gen3", "g3.pt"))])
    out = MP.collect_with_pool(object(), league, 1, team_pool=["A", "B"], ckpt_path="t.pt", n_procs=1,
                               chunk_size=1, submit_fn=submit, save_ckpt_fn=lambda a, p: None, drill_sink=boom)
    assert out == (["t"], {"model": 1})
    assert league.recorded == [("gen3", True)]                          # PFSP still applied after the sink
    out2 = MP.collect_with_pool(object(), _CyclingLeague([("latest", "x.pt")]), 1, team_pool=["A", "B"],
                                ckpt_path="t.pt", n_procs=1, chunk_size=1, submit_fn=submit,
                                save_ckpt_fn=lambda a, p: None)
    assert out2 == (["t"], {"model": 1})


# ── GenerationRecord.drill ─────────────────────────────────────────────────────
_SB = {"drill": "field", "games": 12, "win": 0.583, "trick_per_game": 0.25, "weather_fights": 4,
       "weather_fight_win": 0.5, "weather_share": None, "skipped": 1, "mirror": 2, "fallback": 0,
       "targets": {"games": 5, "win": 0.4},
       "by_kind": {"clone": {"games": 7, "win": 0.571}, "latest": {"games": 5, "win": 0.6}},
       "pressure": {"games": 7, "fired": 9, "taken": 4}}


def _rec(g, drill=None, **kw):
    base = dict(generation=g, n_trajectories=10, scripted_wins=1, scripted_games=2, model_elo=None,
                verdict="hold", promoted=False, update_stats={"loss": 0.1})
    base.update(kw)
    return GenerationRecord(drill=drill, **base)


def test_generation_record_drill_round_trips_through_json():
    r = _rec(4, drill=_SB)
    obj = r.to_obj()
    assert obj["drill"] == _SB
    back = GenerationRecord.from_obj(json.loads(json.dumps(obj, allow_nan=False)))
    assert back.drill == _SB


def test_generation_record_without_drill_loads_none():
    r = _rec(2)
    assert r.drill is None and r.to_obj()["drill"] is None
    old = r.to_obj()
    old.pop("drill")                                                    # a pre-drill snapshot dict
    assert GenerationRecord.from_obj(old).drill is None


# ── archive.build_manifest ─────────────────────────────────────────────────────
def test_manifest_per_gen_dict_carries_drill_and_is_strict_json():
    h = GenerationHistory()
    h.add(_rec(0))
    h.add(_rec(1, drill=_SB, promoted=True, verdict="promote"))
    m = build_manifest(h)
    gens = m["generations"]
    assert [g["drill"] for g in gens] == [None, _SB]
    txt = json.dumps(m, allow_nan=False)
    assert json.loads(txt)["generations"][1]["drill"]["targets"] == {"games": 5, "win": 0.4}


def test_manifest_drill_error_scoreboard_is_strict_json():
    """The generation loop stores ``{"games": 0, "error": repr(exc)}`` when scoring fails — still JSON."""
    h = GenerationHistory()
    h.add(_rec(0, drill={"games": 0, "error": "KeyError('ours')"}))
    json.dumps(build_manifest(h), allow_nan=False)


# ── preflight.check with drill_on / pressure_on ─────────────────────────────────
_ALL = {"latest": 3, "snapshot": 2, "clone:c1.pt": 2, "scripted:random": 1,
        "scripted:max_damage": 1, "scripted:heuristic": 1}


def _league(played):
    lg = OpponentLeague(latest_path="l.pt", clones=("c1.pt",), cfg=LeagueConfig(clone_frac=0.3))
    lg.admit("g0", "g0.pt", 0)
    lg.note_played(played)
    return lg


def _hist(n, drill):
    """``drill``: a dict for every gen, a callable ``gen -> dict|None``, or None."""
    h = GenerationHistory()
    for g in range(n):
        d = drill(g) if callable(drill) else drill
        h.add(_rec(g, drill=(dict(d) if isinstance(d, dict) else d), promoted=(g == 1),
                   verdict="promote" if g == 1 else "hold"))
    return h


def _run(drill_fresh, drill_resumed=None, **kw):
    fresh = {"history": _hist(3, drill_fresh), "league": _league(_ALL)}
    resumed = {"history": _hist(4, drill_fresh if drill_resumed is None else drill_resumed),
               "league": _league({})}
    return PF.check(fresh, resumed, **kw)


_GOOD = {"games": 6, "targets": {"games": 2}, "pressure": {"games": 3, "fired": 4, "taken": 1}}


def test_preflight_drill_clean_run_passes():
    problems, warns, _ = _run(_GOOD, drill_on=True, pressure_on=True)
    assert problems == [] and warns == []


def test_preflight_drill_zero_or_missing_games_is_a_problem():
    problems, _, _ = _run(lambda g: {**_GOOD, "games": 0} if g == 2 else _GOOD, drill_on=True)
    assert problems == ["gen 2: the drill scoreboard scored 0 games"]
    problems, _, _ = _run(lambda g: None if g == 1 else _GOOD, drill_on=True)
    assert problems == ["gen 1: the drill scoreboard scored 0 games"]
    problems, _, _ = _run({"games": 0, "error": "KeyError('ours')"}, drill_on=True)
    assert len(problems) == 4 and all("KeyError('ours')" in p for p in problems)


def test_preflight_drill_checks_the_resumed_history():
    """The resumed history is loaded back from the snapshot: a drill that did not persist fails there."""
    problems, _, _ = _run(_GOOD, drill_resumed=lambda g: None, drill_on=True)
    assert problems == [f"gen {g}: the drill scoreboard scored 0 games" for g in range(4)]


def test_preflight_drill_targets_summing_to_zero_is_a_problem():
    problems, _, _ = _run({**_GOOD, "targets": {"games": 0}}, drill_on=True)
    assert len(problems) == 1 and "TARGET" in problems[0]
    no_split = {k: v for k, v in _GOOD.items() if k != "targets"}       # no targets split at all -> no check
    problems, _, _ = _run(no_split, drill_on=True)
    assert problems == []
    one = _run(lambda g: {**_GOOD, "targets": {"games": 1 if g == 3 else 0}}, drill_on=True)[0]
    assert one == []                                                    # summed across gens


def test_preflight_pressure_never_firing_only_warns():
    quiet = {**_GOOD, "pressure": {"games": 3, "fired": 0, "taken": 0}}
    problems, warns, _ = _run(quiet, drill_on=True, pressure_on=True)
    assert problems == []
    assert len(warns) == 1 and "pressure never fired" in warns[0]
    no_pressure_block = {k: v for k, v in _GOOD.items() if k != "pressure"}
    assert len(_run(no_pressure_block, drill_on=True, pressure_on=True)[1]) == 1
    assert _run(quiet, drill_on=True, pressure_on=False)[1] == []       # pressure off -> no warning


def test_preflight_drill_off_is_unchanged():
    base = _run(None)
    assert base == _run(None, drill_on=False, pressure_on=True)
    assert base == _run({"games": 0}, drill_on=False, pressure_on=True)
    assert base[0] == [] and base[1] == []


def test_run_preflight_passes_drill_flags_through(tmp_path, monkeypatch):
    seen = {}
    real = PF.check

    def spy(fresh, resumed, **kw):
        seen.update(kw)
        return real(fresh, resumed, **kw)

    monkeypatch.setattr(PF, "PREFLIGHT_DIR", tmp_path)
    monkeypatch.setattr(PF, "check", spy)

    def launch(a):
        n = 3 if a.resume_gen is None else 4
        return {"history": _hist(n, _GOOD), "league": _league(_ALL if n == 3 else {})}

    args = SimpleNamespace(generations=40, games=300, league_clones=["c1.pt"], clone_frac=0.3,
                           register_arms=False, hof=False, preflight="auto", preflight_only=False,
                           run_cfg_gate={}, drill="field", drill_bias=2.5)
    assert PF.run_preflight(args, launch) is True
    assert seen["drill_on"] is True and seen["pressure_on"] is True
    args.drill_bias = 0.0
    PF.run_preflight(args, launch)
    assert seen["drill_on"] is True and seen["pressure_on"] is False
    args.drill = None
    PF.run_preflight(args, launch)
    assert seen["drill_on"] is False and seen["pressure_on"] is False


# ── the per-gen report line ─────────────────────────────────────────────────────
def _rep(**kw):
    base = {"generation": 3, "n_trajectories": 120, "scripted_win_rate": 0.5, "model_elo": None,
            "verdict": "hold", "league_size": 2}
    base.update(kw)
    return base


def _drill_lines(out):
    return [ln for ln in out.splitlines() if "DRILL" in ln]


def test_report_prints_a_drill_line_only_when_set(capsys):
    from v_dance.selfplay.generation import print_generation_report
    print_generation_report(_rep())
    assert _drill_lines(capsys.readouterr().out) == []
    print_generation_report(_rep(drill=None))
    assert _drill_lines(capsys.readouterr().out) == []
    print_generation_report(_rep(drill={}))
    assert _drill_lines(capsys.readouterr().out) == []
    print_generation_report(_rep(drill=_SB))
    lines = _drill_lines(capsys.readouterr().out)
    assert len(lines) == 1
    assert lines[0].startswith("        DRILL field: ")
    assert "games 12" in lines[0] and "targets:" in lines[0] and "fired 9" in lines[0]


def test_report_prints_the_scoring_error_line(capsys):
    from v_dance.selfplay.generation import print_generation_report
    print_generation_report(_rep(drill={"games": 0, "error": "KeyError('ours')"}))
    lines = _drill_lines(capsys.readouterr().out)
    assert len(lines) == 1 and "games 0" in lines[0] and "KeyError" in lines[0]


# ── end to end: worker rows -> merge -> sink -> scoreboard -> record -> manifest ──
def test_rows_from_the_worker_score_and_persist_end_to_end(capsys):
    from v_dance.selfplay.drills import DrillSetup, gen_scoreboard, get_drill
    from v_dance.selfplay.generation import print_generation_report
    specs = [ChunkSpec("Own", "Opp_Rilla", 1, "clone", "c1.pt", None, 1, pressure="field", pressure_bias=2.5,
                       score=True),
             ChunkSpec("Own", "Other", 1, "latest", None, None, 2, score=True),
             ChunkSpec("Own", "Own", 1, "snapshot", "g.pt", "g", 3, score=True)]            # mirror: left out
    res = _collect(specs, _builder())
    sunk = []
    MP.collect_with_pool(object(), _CyclingLeague([("latest", "x.pt")]), 1, team_pool=["A", "B"],
                         ckpt_path="t.pt", n_procs=1, chunk_size=1, submit_fn=lambda p: [res, None],
                         save_ckpt_fn=lambda a, p: None, drill_sink=lambda rows, ps: sunk.append((rows, ps)))
    rows, ps = sunk[0]
    setup = DrillSetup(drill=get_drill("focus:opp=rillaboom,mon=salamence"), pool=None, opp_weights={},
                       target_keys=frozenset({"opp_rilla"}), pressure=None, bias=0.0)
    sb = gen_scoreboard(setup, rows, ps)
    assert sb["drill"] == "focus" and sb["games"] == 2 and sb["win"] == 1.0
    assert sb["mirror"] == 1 and sb["fallback"] == 0 and sb["skipped"] == 0
    assert sb["targets"]["games"] == 1
    assert set(sb["by_kind"]) == {"clone", "latest"}
    assert sb["pressure"] == {"games": 1, "fired": 4, "taken": 2}     # clone 2/1 + mirror snapshot 2/1 + latest 0/0
    assert sb["salamence_brought"] == 1.0
    h = GenerationHistory()
    h.add(_rec(0, drill=sb))
    back = GenerationRecord.from_obj(json.loads(json.dumps(h.records[0].to_obj(), allow_nan=False)))
    assert back.drill == sb
    json.dumps(build_manifest(h), allow_nan=False)
    print_generation_report(_rep(drill=sb))
    assert _drill_lines(capsys.readouterr().out)[0].lstrip().startswith("DRILL focus:")


# ── the CLI + run_live_generations contract ─────────────────────────────────────
def test_cli_help_lists_the_drill_flags():
    env = {**os.environ, "PYTHONUTF8": "1"}
    r = subprocess.run([sys.executable, "-m", "v_dance.selfplay.generation", "--help"], cwd=str(_REPO_ROOT),
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180, env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "--drill " in r.stdout or "--drill\n" in r.stdout
    assert "--drill-bias" in r.stdout


def test_run_live_generations_drill_param_defaults_none():
    from v_dance.selfplay.generation import run_live_generations
    params = inspect.signature(run_live_generations).parameters
    assert "drill" in params and params["drill"].default is None
    assert params["drill"].kind == inspect.Parameter.KEYWORD_ONLY


def test_run_live_generations_refuses_a_drill_without_the_process_pool(tmp_path, capsys):
    """The drill rides the multiprocess collector: a single-process run exits 2 before any model,
    server or pool is built."""
    from v_dance.selfplay.generation import run_live_generations
    with pytest.raises(SystemExit) as ei:
        run_live_generations("no_such.pt", team_pool=["A", "B"], team_chooser=None,
                             archive_dir=tmp_path / "arch", collect_procs=1, drill=object())
    assert ei.value.code == 2
    assert "--drill needs --collect-procs >= 2" in capsys.readouterr().err
