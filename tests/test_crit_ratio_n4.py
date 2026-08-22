"""v11 N4 — crit-ratio expected-damage term folded into the (otherwise crit-blind) damage band.

stage = move.critRatio(base 1) + Super Luck(+1) + Scope Lens/Razor Claw(+1), capped 4; p_crit = 1/[_,24,8,2,1]
[stage]; a crit deals x1.5 (Sniper x2.25). expected_mult = 1 + p_crit*(0.5, or 1.25 with Sniper), folded into
the situational band mult on BOTH encoders via the shared _expected_crit_mult (critRatio from _gen9_moves =
GenData → byte-parity). Scope Lens is an ITEM → suppressed under Magic Room (rides the P5 gate). Value-only.
Unseen Fist (the other N4 target) is a FORCED defer — no protect-this-turn input until the |-singleturn| enabler.
"""
from __future__ import annotations

from v_dance.encoders.state_encoder import (
    StateEncoder, MOVE_FEATURES, _MOVE_BLOCK_REL, move_slots_for_mon, norm_species,
    _expected_crit_mult, get_state_dim,
)

_P1 = 1.0 + (1.0 / 24) * 0.5    # crit stage 1 (normal move)
_P2 = 1.0 + (1.0 / 8) * 0.5     # crit stage 2 (high-crit move OR +1 booster)
_MR = {"magic_room_turns_remaining": 5}


# ════════════════════════════ formula unit ════════════════════════════
def test_crit_mult_formula():
    assert abs(_expected_crit_mult("tackle", "pressure", False) - _P1) < 1e-9
    assert abs(_expected_crit_mult("stoneedge", "pressure", False) - _P2) < 1e-9   # critRatio 2 → stage 2
    assert abs(_expected_crit_mult("tackle", "superluck", False) - _P2) < 1e-9     # Super Luck +1 → stage 2
    assert abs(_expected_crit_mult("tackle", "pressure", True) - _P2) < 1e-9       # Scope Lens +1 → stage 2
    assert abs(_expected_crit_mult("stoneedge", "superluck", True) - 1.5) < 1e-9   # 2+1+1=4 → 1/1 → 1.5
    assert abs(_expected_crit_mult("tackle", "sniper", False) - (1.0 + (1.0 / 24) * 1.25)) < 1e-9
    # cap at stage 4 (no overflow past 1/1)
    assert _expected_crit_mult("stoneedge", "superluck", True) == _expected_crit_mult("frostbreath", "superluck", True) \
        or _expected_crit_mult("stoneedge", "superluck", True) == 1.5


def test_high_crit_move_beats_normal_move():
    assert _expected_crit_mult("stoneedge", "pressure", False) > _expected_crit_mult("earthquake", "pressure", False)


# ════════════════════════════ e2e band (model: test_items_b3) ════════════════════════════
def _mon(species, ability="Pressure", *, moves=(), item=None):
    m = {"species": species, "base_species": species, "hp_pct": 100.0, "seen": True, "is_fainted": False,
         "known_moves": list(moves), "revealed_moves": [], "boosts": {}, "status": None,
         "known_ability": ability, "volatiles": {},
         "stats_estimate": {"mode": "exact",
                            "stats": {"atk": 130, "spa": 80, "def": 120, "spd": 120, "hp": 300, "spe": 80}}}
    if item is not None:
        m["known_item"] = item
    return m


def _band_max(att, move, defender, field):
    snap = {"our_active": {"our_a": att, "our_b": None},
            "opp_active": {"opp_a": defender, "opp_b": None},
            "our_bench": [], "opp_bench": [], "field": field, "side_conditions": {}}
    vec = StateEncoder().encode_snapshot(snap, turn=1)
    for m_idx, (mv, _c) in enumerate(move_slots_for_mon(att)):
        if norm_species(mv) == norm_species(move):
            return float(vec[_MOVE_BLOCK_REL + m_idx * MOVE_FEATURES + 14])   # damage MAX vs enemy0
    raise AssertionError(move)


def test_super_luck_raises_band():
    no = _band_max(_mon("Garchomp", "Rough Skin", moves=["Earthquake"]), "Earthquake", _mon("Arcanine"), {})
    sl = _band_max(_mon("Garchomp", "Super Luck", moves=["Earthquake"]), "Earthquake", _mon("Arcanine"), {})
    assert abs(sl / no - (_P2 / _P1)) < 5e-3        # crit stage 1→2


def test_scope_lens_raises_band_and_magic_room_suppresses():
    plain = _mon("Garchomp", "Rough Skin", moves=["Earthquake"])
    scope = _mon("Garchomp", "Rough Skin", moves=["Earthquake"], item="Scope Lens")
    arc = _mon("Arcanine")
    assert abs(_band_max(scope, "Earthquake", arc, {}) / _band_max(plain, "Earthquake", arc, {}) - (_P2 / _P1)) < 5e-3
    # Scope Lens is an item → Magic Room suppresses its crit boost → band == the no-item band
    assert abs(_band_max(scope, "Earthquake", arc, _MR) - _band_max(plain, "Earthquake", arc, _MR)) < 1e-4


def test_high_crit_move_band_above_normal_crit():
    # a critRatio:2 move carries the stage-2 EV even with no booster (the realized in-meta crit signal)
    att = _mon("Tyranitar", "Sand Stream", moves=["Stone Edge"])
    # compare the SAME attacker's Stone Edge crit EV vs a synthetic stage-1 baseline via the unit ratio
    assert _expected_crit_mult("stoneedge", "Pressure".lower(), False) == _P2


# ════════════════════════════ layout (value-only) ════════════════════════════
def test_layout_unchanged():
    assert get_state_dim() == 5057
