"""Tests for the scripted gauntlet opponents (MaxDamage / type-speed Heuristic).

Uses real poke_env Move / Pokemon objects (so type-effectiveness is genuine) and
a duck-typed battle, with player instances built via __new__ so no Showdown
connection is needed.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("poke_env")

_REPO = Path(__file__).resolve().parents[1]
import v_dance.eval.eval_opponents as EO  # noqa: E402
from poke_env.battle import Move, Pokemon  # noqa: E402
from v_dance.encoders.state_encoder import SWITCH_OFFSET  # noqa: E402


def _mk(cls):
    """A player instance without poke-env's networked __init__."""
    return cls.__new__(cls)


def _battle(active_moves, foe_species="ferrothorn", active_types=None):
    """Duck DoubleBattle: one active mon with the given moves + one foe."""
    moves = {m: Move(m, gen=9) for m in active_moves}
    fire = Move("flamethrower", gen=9).type
    mon = types.SimpleNamespace(
        moves=moves, types=active_types or [fire], fainted=False)
    foe = Pokemon(gen=9, species=foe_species)
    return types.SimpleNamespace(active_pokemon=[mon, None],
                                 opponent_active_pokemon=[foe, None])


# ── MaxDamage ─────────────────────────────────────────────────────────────────
def test_maxdamage_scores_by_base_power():
    p = _mk(EO.MaxDamageVGCPlayer)
    b = _battle(["flamethrower", "earthquake"])     # bp 90 vs 100
    # action i//3 = move idx, i%3 = target bucket (0 = foe0)
    s_flame = p._score(b, 0, 0)                      # move0 flamethrower, foe0
    s_quake = p._score(b, 0, 3)                      # move1 earthquake, foe0
    assert s_quake > s_flame                         # picks the bigger base power
    assert p._score(b, 0, 2) == EO._ALLY_PENALTY     # move0 at ally bucket forbidden
    assert p._score(b, 0, SWITCH_OFFSET) == EO._SWITCH_SCORE


def test_maxdamage_best_action_picks_strongest_move(monkeypatch):
    p = _mk(EO.MaxDamageVGCPlayer)
    b = _battle(["tackle", "flamethrower", "protect", "earthquake"])  # 40/90/0/100

    def fake_mask(battle, slot):
        row = [False] * 16
        for i in (0, 3, 6, 9):       # all four moves legal at foe0
            row[i] = True
        row[SWITCH_OFFSET] = True     # a switch is also legal
        return row

    monkeypatch.setattr(EO, "build_legal_action_mask", fake_mask)
    assert p._best_action(b, 0) == 9                  # earthquake (bp100) at foe0


def test_maxdamage_switches_only_when_no_move(monkeypatch):
    p = _mk(EO.MaxDamageVGCPlayer)
    b = _battle(["flamethrower"])
    monkeypatch.setattr(EO, "build_legal_action_mask",
                        lambda battle, slot: [i == SWITCH_OFFSET for i in range(16)])
    assert p._best_action(b, 0) == SWITCH_OFFSET      # only a switch is legal


# ── Heuristic (type + speed) ──────────────────────────────────────────────────
def test_heuristic_prefers_super_effective_over_higher_base_power():
    """vs Ferrothorn (Grass/Steel): Fire flamethrower is 4x (and STAB) → it beats
    the higher-base-power Earthquake (2x).  This is exactly where the heuristic
    diverges from MaxDamage."""
    h = _mk(EO.HeuristicVGCPlayer)
    md = _mk(EO.MaxDamageVGCPlayer)
    b = _battle(["flamethrower", "earthquake"])       # 90 fire vs 100 ground
    assert h._score(b, 0, 0) > h._score(b, 0, 3)      # heuristic: flamethrower
    assert md._score(b, 0, 3) > md._score(b, 0, 0)    # max-dmg: earthquake


def test_heuristic_status_move_low_priority():
    h = _mk(EO.HeuristicVGCPlayer)
    b = _battle(["protect", "flamethrower"])
    assert h._score(b, 0, 0) == 0.0                   # protect (status)
    assert h._score(b, 0, 3) > 0.0                    # flamethrower scores positive


def test_heuristic_never_targets_ally():
    h = _mk(EO.HeuristicVGCPlayer)
    b = _battle(["flamethrower"])
    assert h._score(b, 0, 2) == EO._ALLY_PENALTY      # move0 ally bucket forbidden


def test_select_actions_dedups_cross_slot_switch(monkeypatch):
    """Both slots picking the same bench switch is illegal — slot 1 re-picks."""
    p = _mk(EO.MaxDamageVGCPlayer)
    b = _battle(["flamethrower"])

    def fake_mask(battle, slot):
        row = [False] * 16
        row[SWITCH_OFFSET] = True
        if slot == 1:
            row[SWITCH_OFFSET + 1] = True   # slot 1 has a second bench option
        return row

    monkeypatch.setattr(EO, "build_legal_action_mask", fake_mask)
    a0, a1, src = p._select_actions(b, None)
    assert a0 == SWITCH_OFFSET
    assert a1 == SWITCH_OFFSET + 1 and a1 != a0       # deduped
    assert src == "max_damage"
