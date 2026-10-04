"""2026-10-04: the per-turn trace (``vgc_base.ReplayBuffer``) is APPEND-ONLY and eval players write none.

Before, every battle end READ AND REWROTE the player's whole ``artifacts/replay_buffer/<name>.jsonl`` to back-fill the
outcome, and panel_eval re-uses its player names on every call, so those files reached 200-330 MB and every re-test
call ran ~40 s slower than the one before (the mega-hold re-test: 82 s → 610 s per 400 games). Now a battle's turns
are held in memory and appended once, outcome filled; the line format is unchanged; eval players pass trace=False."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from v_dance.play.vgc_base import NullReplayBuffer, ReplayBuffer


def _lines(p: Path) -> list:
    return p.read_text(encoding="utf-8").splitlines()


def _rec(buf, battle, turn, val=0.25):
    buf.record(battle, turn, np.full(3, val, dtype=np.float32), 7, 11, "model")


def test_finalise_appends_only_that_battle_with_its_outcome(tmp_path):
    p = tmp_path / "BC900x1.jsonl"
    buf = ReplayBuffer(p)
    _rec(buf, "battle-a", 1)
    _rec(buf, "battle-b", 1)                 # two battles interleaved, as with max_concurrent_battles > 1
    _rec(buf, "battle-a", 2)
    assert p.read_text(encoding="utf-8") == ""          # nothing is written mid-battle
    buf.finalise("battle-a", 1)
    rows = [json.loads(x) for x in _lines(p)]
    assert [(r["battle_id"], r["turn"], r["outcome"]) for r in rows] == [("battle-a", 1, 1), ("battle-a", 2, 1)]
    buf.finalise("battle-b", 0)
    rows = [json.loads(x) for x in _lines(p)]
    assert [(r["battle_id"], r["turn"], r["outcome"]) for r in rows][2:] == [("battle-b", 1, 0)]
    buf.close()


def test_line_format_is_byte_identical_to_the_old_back_filled_lines(tmp_path):
    """The old writer dumped each turn with outcome null, then at the battle end json.loads + set outcome + json.dumps;
    the new one dumps once with the outcome set — the lines must be the same bytes."""
    p = tmp_path / "t.jsonl"
    buf = ReplayBuffer(p)
    state = np.array([0.1, -2.5, 3.0000001, 0.0], dtype=np.float32)
    buf.record("battle-x", 4, state, 3, 9, "model")
    buf.finalise("battle-x", -1)
    buf.close()
    old = json.dumps({"battle_id": "battle-x", "turn": 4, "state": state.tolist(), "action_s0": 3,
                      "action_s1": 9, "source": "model", "outcome": None})
    obj = json.loads(old)
    obj["outcome"] = -1
    assert _lines(p) == [json.dumps(obj)]


def test_an_existing_trace_is_never_read_or_rewritten(tmp_path, monkeypatch):
    p = tmp_path / "OPpane900x1.jsonl"
    old = b'{"old": 1}\n{"old": 2}\n'
    p.write_bytes(old)

    def _no_read(self, *a, **k):              # the old writer re-read the WHOLE file at every battle end
        raise AssertionError(f"read_text({self}) — the trace must be append-only")
    monkeypatch.setattr(Path, "read_text", _no_read)
    buf = ReplayBuffer(p)
    _rec(buf, "battle-a", 1)
    buf.finalise("battle-a", 1)
    buf.close()
    monkeypatch.undo()
    data = p.read_bytes()
    assert data.startswith(old)                                     # the old bytes are untouched
    assert json.loads(data[len(old):].decode("utf-8"))["outcome"] == 1   # one new line, appended


def test_close_writes_a_battle_that_never_finished_with_outcome_null(tmp_path):
    p = tmp_path / "t.jsonl"
    buf = ReplayBuffer(p)
    _rec(buf, "battle-crashed", 3)
    buf.finalise("battle-unknown", 1)                    # a battle with no turns: nothing to write
    assert p.read_text(encoding="utf-8") == ""
    buf.close()
    rows = [json.loads(x) for x in _lines(p)]
    assert [(r["battle_id"], r["outcome"]) for r in rows] == [("battle-crashed", None)]


def test_null_buffer_records_nothing():
    buf = NullReplayBuffer()
    _rec(buf, "battle-a", 1)
    buf.finalise("battle-a", 1)
    buf.close()
    assert buf.path is None


def test_a_player_built_with_trace_false_opens_no_trace_file(tmp_path):
    pytest.importorskip("poke_env")
    import v_dance.play.run_local_battle as R
    from poke_env import AccountConfiguration
    from v_dance.play.player import VGCPlayer
    team = R.load_team(R.resolve_team_path("WolfeGlick"))
    off, on = tmp_path / "BCnotrace.jsonl", tmp_path / "BCtrace.jsonl"
    p_off = VGCPlayer(model_path=None, replay_path=off, trace=False, battle_format=R.BATTLE_FORMAT, team=team,
                      account_configuration=AccountConfiguration("BCnotrace", None), start_listening=False)
    p_on = VGCPlayer(model_path=None, replay_path=on, battle_format=R.BATTLE_FORMAT, team=team,
                     account_configuration=AccountConfiguration("BCtrace", None), start_listening=False)
    assert isinstance(p_off._replay, NullReplayBuffer) and not off.exists()
    assert isinstance(p_on._replay, ReplayBuffer) and on.exists()       # the default keeps tracing
    p_off.close()
    p_on.close()


def test_every_eval_player_is_built_with_trace_false(monkeypatch):
    """mp_eval (the gauntlet's parallel path, the panel, panel_eval) builds the candidate and every opponent kind
    without a trace."""
    import v_dance.eval.gauntlet as G
    import v_dance.play.run_local_battle as R
    from v_dance.selfplay import mp_eval as ME
    calls = []
    monkeypatch.setattr(R, "make_player", lambda *a, **k: calls.append(("make_player", k)) or object())
    monkeypatch.setattr(G, "_make_opponent", lambda *a, **k: calls.append(("_make_opponent", k)) or object())
    monkeypatch.setattr(R, "load_team", lambda p: "TEAMSTR")
    monkeypatch.setattr(R, "resolve_team_path", lambda n: n)
    specs = [ME.EvalSpec("panel:era2", "A", "B", 4, 1, 0, opp_ckpt="era2.pt"),
             ME.EvalSpec("heuristic", "A", "B", 4, 2, 0),
             ME.EvalSpec("prev_best", "A", "A", 4, 3, 0)]
    for s in specs:
        ME._build_eval_players_real("cand.pt", "best.pt", "tc.pt", s)
    assert len(calls) == 6
    assert all(k.get("trace") is False for _, k in calls), calls


def test_the_single_process_gauntlet_builds_its_players_without_a_trace(monkeypatch):
    import asyncio
    import types as _t
    pytest.importorskip("poke_env")
    import v_dance.eval.gauntlet as G
    import v_dance.play.run_local_battle as R

    class FakePlayer:
        def __init__(self):
            self.n_won_battles = self.n_finished_battles = 0

            async def _stop():
                return None
            self.ps_client = _t.SimpleNamespace(stop_listening=_stop)

        async def battle_against(self, opp, n_battles):
            self.n_finished_battles += n_battles

        def close(self):
            pass

    traces = []
    monkeypatch.setattr(R, "start_showdown", lambda: None)
    monkeypatch.setattr(R, "stop_showdown", lambda proc: None)
    monkeypatch.setattr(R, "make_player", lambda *a, **k: traces.append(k.get("trace")) or FakePlayer())
    monkeypatch.setattr(G, "_make_opponent", lambda *a, **k: traces.append(k.get("trace")) or FakePlayer())
    monkeypatch.setattr(R, "load_team", lambda p: "TEAMSTR")
    monkeypatch.setattr(R, "resolve_team_path", lambda n: n)
    asyncio.run(G.run_gauntlet(opponents=["random"], team_pool=["A", "B"], battles_per_opponent=2,
                               ckpt=Path("bc.pt"), team_chooser=Path("tc.pt"), manage_server=True))
    assert traces and all(t is False for t in traces), traces
