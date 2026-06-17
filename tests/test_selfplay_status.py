"""3c.6e-1: live run status writer for the dashboard. Pure — no torch/poke-env."""
from __future__ import annotations

import json

from v_dance.selfplay.status import LiveStatus, read_status, _numeric


def _ls(tmp_path):
    ticks = iter(range(1000, 1000000))
    return LiveStatus(tmp_path / "status.json", clock=lambda: float(next(ticks)))


def test_blank_until_started(tmp_path):
    ls = _ls(tmp_path)
    ls._write()
    s = read_status(ls.path)
    assert s["live"] is False and s["run"]["phase"] == "idle"
    assert s["showdown_url"] == "http://localhost:8000"
    assert s["updated_at"] is not None            # clock stamped


def test_start_run_sets_live_and_target(tmp_path):
    ls = _ls(tmp_path)
    ls.start_run(n_generations=5, hours=2.0)
    s = read_status(ls.path)
    assert s["live"] is True
    assert s["run"]["phase"] == "starting" and s["run"]["n_generations"] == 5
    assert s["run"]["hours_budget"] == 2.0 and s["run"]["started_at"] is not None


def test_phase_collecting_sets_total_and_resets_progress(tmp_path):
    ls = _ls(tmp_path)
    ls.start_run(3)
    ls.phase("collecting", generation=2, games_total=300)
    s = read_status(ls.path)
    assert s["run"]["phase"] == "collecting" and s["run"]["generation"] == 2
    assert s["run"]["games_total"] == 300 and s["run"]["games_done"] == 0


def test_games_progress_and_running_winrate(tmp_path):
    ls = _ls(tmp_path)
    ls.start_run(1); ls.phase("collecting", generation=0, games_total=300)
    ls.games(137, running_p1_winrate=0.51)
    s = read_status(ls.path)
    assert s["run"]["games_done"] == 137 and abs(s["run"]["running_p1_winrate"] - 0.51) < 1e-9


def test_active_battles_filtered_and_coerced(tmp_path):
    ls = _ls(tmp_path)
    ls.set_active_battles([
        {"tag": "battle-gen9-1", "p1": "SP1", "p2": "SP2", "turn": "7"},
        {"p1": "no-tag"},                                   # dropped (no tag)
        {"tag": "battle-gen9-2"},                           # defaults filled
    ])
    s = read_status(ls.path)
    b = s["active_battles"]
    assert len(b) == 2
    assert b[0]["tag"] == "battle-gen9-1" and b[0]["turn"] == 7
    assert b[1]["p1"] == "p1" and b[1]["turn"] == 0


def test_phase_updating_clears_active_battles(tmp_path):
    ls = _ls(tmp_path)
    ls.set_active_battles([{"tag": "battle-x-1"}])
    ls.phase("updating")
    assert read_status(ls.path)["active_battles"] == []


def test_set_update_keeps_numeric_only(tmp_path):
    ls = _ls(tmp_path)
    ls.set_update({"loss": 0.4, "halted": True, "halt_reason": "kl", "explained_variance": 0.86},
                  last_verdict="promote")
    s = read_status(ls.path)
    assert s["update"]["loss"] == 0.4 and s["update"]["halted"] == 1.0
    assert "halt_reason" not in s["update"]                 # non-numeric dropped
    assert s["run"]["last_verdict"] == "promote"


def test_finish_run(tmp_path):
    ls = _ls(tmp_path)
    ls.start_run(2); ls.set_active_battles([{"tag": "battle-y-1"}])
    ls.finish_run()
    s = read_status(ls.path)
    assert s["live"] is False and s["run"]["phase"] == "done" and s["active_battles"] == []


def test_atomic_write_leaves_no_tmp(tmp_path):
    ls = _ls(tmp_path)
    ls.start_run(1)
    assert ls.path.exists()
    assert not ls.path.with_name(ls.path.name + ".tmp").exists()
    json.loads(ls.path.read_text(encoding="utf-8"))         # always valid JSON


def test_read_status_missing_is_none(tmp_path):
    assert read_status(tmp_path / "nope.json") is None


def test_numeric_helper():
    assert _numeric({"a": 1, "b": 2.5, "c": True, "d": "x"}) == {"a": 1.0, "b": 2.5, "c": 1.0}


def test_write_live_log(tmp_path):
    ls = _ls(tmp_path)
    ls.write_live_log("battle-x-7", ["|turn|1", "|move|p1a: A|Tackle|p2a: B"], turn=1)
    p = ls.path.with_name("live_log.json")
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["tag"] == "battle-x-7" and d["turn"] == 1 and d["n_lines"] == 2
    assert d["log"][0] == "|turn|1"
    assert not p.with_name(p.name + ".tmp").exists()        # atomic, no tmp left
