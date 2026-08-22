"""15b.0: the self-play collector records the TRUE team-preview decision.

Before this, `finalize_trajectory` was called without the bring/leads/opp roster,
so every trajectory carried the first-4 placeholder + an EMPTY opponent team —
useless for outcome-driven TP (sec 14). Now `teampreview()` captures the ACTUAL
submitted order + rosters into ``_tp_decision[battle_tag]`` and the SelfPlay
finalize threads it into the Trajectory's EpisodeMeta.

Built with ``__new__`` players + SimpleNamespace battle fakes (no Showdown server),
mirroring tests/test_teampreview_serve.py + tests/test_selfplay_game_runner.py.
"""
from __future__ import annotations

import sys
import types
from collections import Counter
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("poke_env")
import numpy as np  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
import v_dance.play.player as P            # noqa: E402
from v_dance.play import vgc_base          # noqa: E402
from v_dance.selfplay import game_runner as G  # noqa: E402
from v_dance.selfplay.collector import TrajectoryCollector  # noqa: E402
from v_dance.selfplay.schema import PASS_ACTION  # noqa: E402
from v_dance.encoders.state_encoder import get_state_dim, get_action_dim  # noqa: E402

SD, AD = get_state_dim(), get_action_dim()
_OWN = ("Incineroar", "Rillaboom", "FlutterMane", "Amoonguss", "Urshifu", "Landorus")


def _player():
    obj = P.VGCPlayer.__new__(P.VGCPlayer)   # skip the networked __init__
    obj._team_chooser = object()
    obj._tc_vocab = {}
    obj._tc_cfg = {"lead_k": 2}
    obj._device = "cpu"
    obj._tp_source = Counter()
    obj._tp_decision = {}
    return obj


def _battle(opp=("Charizard", "Whimsicott"), tag="b1"):
    team = {f"p1: {s}": types.SimpleNamespace(species=s) for s in _OWN}
    return types.SimpleNamespace(
        teampreview_team=[],                 # this format sends no |poke| → empty (falls back to .team)
        team=team,
        max_team_size=4,
        teampreview_opponent_team=[types.SimpleNamespace(species=s) for s in opp],
        opponent_team={},
        battle_tag=tag)


# ── teampreview captures the ACTUAL submitted decision ─────────────────────────
def test_teampreview_captures_net_decision(monkeypatch):
    p = _player()
    monkeypatch.setattr(P._M, "team_order", lambda *a, **k: [2, 3, 0, 1])  # net's leads-first order
    cmd = p.teampreview(_battle())
    assert cmd == "/team 3412"                                   # indices 2,3,0,1 → 1-based digits
    d = p._tp_decision["b1"]
    assert d["bring"] == [2, 3, 0, 1]                            # full submitted order = the bring
    assert d["leads"] == [2, 3]                                  # first two submitted = the leads
    assert d["own_team"] == list(_OWN)                           # roster the indices point into
    assert d["opp_team"] == ["Charizard", "Whimsicott"]          # real opponent roster (was empty before)


def test_teampreview_captures_heuristic_first4_with_real_opp(monkeypatch):
    # No chooser → first-4 heuristic (today's collection reality) — but we STILL capture
    # the real opponent roster, which the old placeholder dropped.
    p = _player()
    p._team_chooser = None
    monkeypatch.setattr(P, "_heuristic_team_order", lambda b: [0, 1, 2, 3])
    p.teampreview(_battle(opp=("Miraidon", "Flutter Mane", "Ogerpon")))
    d = p._tp_decision["b1"]
    assert d["bring"] == [0, 1, 2, 3] and d["leads"] == [0, 1]
    assert d["opp_team"] == ["Miraidon", "Flutter Mane", "Ogerpon"]


def test_teampreview_capture_never_breaks_the_order(monkeypatch):
    # A capture failure must not disturb the returned /team command (best-effort bookkeeping).
    p = _player()
    monkeypatch.setattr(P._M, "team_order", lambda *a, **k: [0, 1, 2, 3])
    b = _battle()
    b.teampreview_opponent_team = 999                            # non-iterable → capture raises internally
    cmd = p.teampreview(b)
    assert cmd == "/team 1234"                                   # order still returned cleanly
    assert p._tp_decision.get("b1") is None                      # capture skipped, nothing recorded


# ── the SelfPlay finalize threads the captured decision into EpisodeMeta ───────
def _one_step_collector(tag="b1", role="p1"):
    c = TrajectoryCollector(tag, role)
    c.add_step(state=np.zeros(SD, np.float32), action_s0=0, action_s1=PASS_ACTION,
               mask_s0=[1] * AD, logprob=-1.0, value=0.0, turn=5)
    return c


def test_selfplay_finalize_uses_captured_decision(monkeypatch):
    sp = G.SelfPlayVGCPlayer.__new__(G.SelfPlayVGCPlayer)
    sp._tp_decision = {"b1": {"bring": [2, 3, 0, 1], "leads": [2, 3],
                              "own_team": list(_OWN),
                              "opp_team": ["Charizard", "Whimsicott"]}}
    sp._collectors = {"b1": _one_step_collector()}
    sp._finished = {}
    # The real base callback logs poke-env counters (not available offline); stub it with
    # the same cleanup the real one does (pop the consumed decision) so we can assert both.
    monkeypatch.setattr(vgc_base.VGCPlayerBase, "_battle_finished_callback",
                        lambda self, b: self._tp_decision.pop(b.battle_tag, None))
    battle = types.SimpleNamespace(battle_tag="b1", won=True, lost=False, turn=7, team={})
    sp._battle_finished_callback(battle)

    meta = sp._finished["b1"].meta
    assert meta.tp_bring == [2, 3, 0, 1]                         # TRUE bring, not the first-4 placeholder
    assert meta.tp_leads == [2, 3]
    assert meta.own_team == list(_OWN)
    assert meta.opp_team == ["Charizard", "Whimsicott"]          # opp roster recorded (was () before)
    assert "b1" not in sp._tp_decision                          # consumed + cleaned up


def test_selfplay_finalize_falls_back_when_no_capture(monkeypatch):
    # Defensive: no captured decision → falls back to the battle roster + first-4 default,
    # exactly the pre-15b.0 behaviour (never crashes the finalize).
    sp = G.SelfPlayVGCPlayer.__new__(G.SelfPlayVGCPlayer)
    sp._tp_decision = {}
    sp._collectors = {"b2": _one_step_collector("b2")}
    sp._finished = {}
    monkeypatch.setattr(vgc_base.VGCPlayerBase, "_battle_finished_callback",
                        lambda self, b: None)
    team = {f"p1: {s}": types.SimpleNamespace(species=s) for s in _OWN}
    battle = types.SimpleNamespace(battle_tag="b2", won=False, lost=True, turn=3, team=team)
    sp._battle_finished_callback(battle)
    meta = sp._finished["b2"].meta
    assert meta.tp_bring == [0, 1, 2, 3]                         # finalize's first-4 default
    assert meta.own_team == list(_OWN)
