"""v11 Phase B.1b (#3) — priority-aware who-moves-first move-block channels.

_moves_first gains a prio_delta arg: a nonzero priority bracket hard-overrides speed (and is TR-invariant);
an equal bracket keeps the soft speed/TR margin (so the move-agnostic GLOBAL turn-order block is unchanged).
The move block emits 2 channels/move (vs each enemy active), folding THIS move's priority against the
enemy's resolved speed. Layout: MOVE_FEATURES 52→54, STATE_DIM 4421→4517. Design: wf_1c1bdc85-206.
"""
from __future__ import annotations

import math

from v_dance.encoders.state_encoder import (
    StateEncoder, MOVE_FEATURES, _MOVE_BLOCK_REL, move_slots_for_mon, norm_species,
    get_state_dim, get_state_layout_version, _moves_first,
)

OFF_MOVES = _MOVE_BLOCK_REL
OFF_MOVESFIRST = 17   # within a move block: after type-eff (9-12) + damage band (13-16); e0 at +17, e1 at +18


# ── the helper ────────────────────────────────────────────────────────────────
def test_moves_first_priority_overrides_speed():
    # slower 'a' (30) vs faster 'b' (142): speed says a moves second, but +1 priority overrides.
    assert _moves_first(30, 142, False, prio_delta=1) == 1.0
    assert _moves_first(142, 30, False, prio_delta=-1) == -1.0


def test_moves_first_priority_is_trick_room_invariant():
    # Trick Room reorders WITHIN a bracket only — it must NOT flip a priority bracket.
    assert _moves_first(30, 142, True, prio_delta=1) == 1.0
    assert _moves_first(142, 30, True, prio_delta=2) == 1.0


def test_moves_first_equal_bracket_is_legacy_behaviour():
    # prio_delta == 0 → the original soft tanh margin + TR flip (GLOBAL block unchanged).
    assert math.isclose(_moves_first(100, 50, False), math.tanh(math.log(2.0)))
    assert math.isclose(_moves_first(100, 50, False, prio_delta=0), math.tanh(math.log(2.0)))
    assert math.isclose(_moves_first(100, 50, True), -math.tanh(math.log(2.0)))   # TR flip
    assert _moves_first(0, 50, False) == 0.0                                       # missing speed → 0


# ── move-block emission through encode_snapshot ───────────────────────────────
def _mon(species, ability, *, moves=(), spe=100):
    return {
        "species": species, "base_species": species, "hp_pct": 100.0,
        "seen": True, "is_fainted": False, "known_moves": list(moves), "revealed_moves": [],
        "boosts": {}, "status": None, "known_ability": ability,
        "stats_estimate": {"mode": "exact",
                           "stats": {"atk": 120, "spa": 100, "def": 90, "spd": 90, "hp": 180, "spe": spe}},
    }


def _movesfirst(attacker, opp_a, opp_b, move):
    """(vs_opp_a, vs_opp_b) who-moves-first channels of `move` used by our_a."""
    snap = {"our_active": {"our_a": attacker, "our_b": None},
            "opp_active": {"opp_a": opp_a, "opp_b": opp_b},
            "our_bench": [], "opp_bench": [], "field": {}, "side_conditions": {}}
    vec = StateEncoder().encode_snapshot(snap, turn=1)
    for m_idx, (mv, _c) in enumerate(move_slots_for_mon(attacker)):
        if norm_species(mv) == norm_species(move):
            base = OFF_MOVES + m_idx * MOVE_FEATURES + OFF_MOVESFIRST
            return float(vec[base]), float(vec[base + 1])
    raise AssertionError(f"{move} not in attacker slots")


def test_priority_move_reads_first_vs_faster_enemy():
    slow = _mon("Incineroar", "Intimidate", moves=["Fake Out", "Knock Off"], spe=30)
    fast = _mon("Dragapult", "Clear Body", spe=142)
    fo_a, fo_b = _movesfirst(slow, fast, None, "Fake Out")     # Fake Out priority +3
    assert fo_a == 1.0, "a +priority move must read moves-first even when slower"
    assert fo_b == 0.0, "empty opp_b slot → 0"


def test_zero_priority_move_uses_speed_margin():
    slow = _mon("Incineroar", "Intimidate", moves=["Fake Out", "Knock Off"], spe=30)
    fast = _mon("Dragapult", "Clear Body", spe=142)
    ko_a, _ = _movesfirst(slow, fast, None, "Knock Off")       # Knock Off priority 0
    assert ko_a < 0.0, "a 0-priority slower move should read moves-second (negative margin)"


def test_negative_priority_move_reads_last():
    fast = _mon("Dragapult", "Clear Body", moves=["Dragon Tail", "Dragon Darts"], spe=142)
    slow = _mon("Incineroar", "Intimidate", spe=30)
    dt_a, _ = _movesfirst(fast, slow, None, "Dragon Tail")     # Dragon Tail priority -6
    assert dt_a == -1.0, "a negative-priority move reads moves-last regardless of speed"


# ── layout ────────────────────────────────────────────────────────────────────
def test_layout_v12():
    assert MOVE_FEATURES == 59           # v16 (B2b: +2 per-move per-enemy hit-chance channels)
    assert get_state_dim() == 5057
    assert get_state_layout_version() == 19
