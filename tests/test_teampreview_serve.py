"""#4: the team-preview NET drives the live bring-4/leads selection, with a
counter (_tp_source) + logging so its live behaviour is measurable.

Tests VGCPlayer._choose_team_order: the net path (counted as 'model') and the
heuristic fallbacks (no chooser / inference error / invalid indices), built via
__new__ so no Showdown connection is needed.
"""

from __future__ import annotations

import sys
import types
from collections import Counter
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("poke_env")

_REPO = Path(__file__).resolve().parents[1]
import v_dance.play.player as P  # noqa: E402  (local_battle/player.py)


def _player():
    obj = P.VGCPlayer.__new__(P.VGCPlayer)      # skip the networked __init__
    obj._team_chooser = object()
    obj._tc_vocab = {}
    obj._tc_cfg = {"lead_k": 2}
    obj._device = "cpu"
    obj._tp_source = Counter()
    return obj


def _team(*species):
    return [types.SimpleNamespace(species=s) for s in species]


def _battle(opp=("Charizard", "Whimsicott")):
    return types.SimpleNamespace(
        teampreview_opponent_team=[types.SimpleNamespace(species=s) for s in opp],
        battle_tag="b1")


def test_tp_net_drives_and_is_counted(monkeypatch):
    p = _player()
    monkeypatch.setattr(P._M, "team_order", lambda *a, **k: [2, 3, 0, 1])
    team = _team("Incineroar", "Rillaboom", "FlutterMane", "Amoonguss", "Urshifu", "Landorus")
    order = p._choose_team_order(_battle(), team, 4)
    assert order == [2, 3, 0, 1]                 # the net's leads-first order
    assert p._tp_source["model"] == 1
    assert p._tp_source["heuristic"] == 0


def test_no_chooser_falls_back_to_heuristic(monkeypatch):
    p = _player()
    p._team_chooser = None
    monkeypatch.setattr(P, "_heuristic_team_order", lambda b: [0, 1, 2, 3])
    order = p._choose_team_order(_battle(), _team("a", "b", "c", "d", "e", "f"), 4)
    assert order == [0, 1, 2, 3]
    assert p._tp_source["heuristic"] == 1 and p._tp_source["model"] == 0


def test_net_inference_error_falls_back(monkeypatch):
    p = _player()

    def boom(*a, **k):
        raise RuntimeError("species not in vocab")

    monkeypatch.setattr(P._M, "team_order", boom)
    monkeypatch.setattr(P, "_heuristic_team_order", lambda b: [0, 1, 2, 3])
    order = p._choose_team_order(_battle(), _team("a", "b", "c", "d", "e", "f"), 4)
    assert order == [0, 1, 2, 3]
    assert p._tp_source["heuristic"] == 1


def test_net_invalid_indices_falls_back(monkeypatch):
    p = _player()
    monkeypatch.setattr(P._M, "team_order", lambda *a, **k: [2, 2, 0, 1])  # duplicate idx
    monkeypatch.setattr(P, "_heuristic_team_order", lambda b: [0, 1, 2, 3])
    order = p._choose_team_order(_battle(), _team("a", "b", "c", "d", "e", "f"), 4)
    assert order == [0, 1, 2, 3]
    assert p._tp_source["heuristic"] == 1
