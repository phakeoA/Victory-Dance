"""v11 B2b — per-enemy defender-evasion HIT-CHANCE channels (+96 layout, STATE_DIM 4913, v16).

Each move gets 2 NEW channels (one per enemy active) = the realized hit chance of that move vs that enemy,
folding the defender's evasion stage / Sand Veil / Snow Cloak / Tangled Feet / Bright Powder + No Guard + the
Showdown acc−eva COMBINED net boost (battle-actions.ts:706-722). Shared `_per_enemy_hit_chance` → byte-parity.
Design: scratch/v11_b2b_evasion_design_synthesis.md (wf_afb03b5a).
"""
from __future__ import annotations

from v_dance.encoders.state_encoder import (
    StateEncoder, MOVE_FEATURES, _MOVE_BLOCK_REL, move_slots_for_mon, norm_species,
    _per_enemy_hit_chance, _move_ignores_evasion, get_state_dim, get_state_layout_version, POKEMON_FEATURES,
)


def _D(ability=None, eva_stage=0, confused=False, no_guard=False, evasion_item=False):
    return dict(ability=ability, eva_stage=eva_stage, confused=confused, no_guard=no_guard,
                evasion_item=evasion_item)


def _h(base=1.0, ab=None, stage=0, wl=False, phys=False, oh=False, d=None, mid="x", w=None):
    return _per_enemy_hit_chance(base, attacker_ability=ab, acc_stage=stage, wide_lens=wl, is_physical=phys,
                                 is_ohko=oh, defender=d if d is not None else _D(), move_id=mid, weather=w)


# ════════════════════════════ layout pin ════════════════════════════
def test_b2b_layout_v16():
    assert MOVE_FEATURES == 59
    assert POKEMON_FEATURES == 413
    assert get_state_dim() == 5057
    assert get_state_layout_version() == 19


# ════════════════════════════ _per_enemy_hit_chance unit ════════════════════════════
def test_evasion_stage_boost_table():
    assert abs(_h(d=_D(eva_stage=2)) - 3 / 5) < 1e-9         # +2 eva → ×3/(3+2)
    assert _h(d=_D(eva_stage=-2)) == 1.0                     # -2 eva → ×5/3, capped at 1.0
    assert abs(_h(0.5, d=_D(eva_stage=-2)) - 0.5 * 5 / 3) < 1e-9


def test_no_guard_either_side_overrides():
    assert _h(0.3, d=_D(no_guard=True, eva_stage=6)) == 1.0   # defender No Guard
    assert _h(0.3, ab="noguard", d=_D(eva_stage=6)) == 1.0    # attacker No Guard


def test_weather_evasion_abilities():
    assert abs(_h(d=_D(ability="sandveil"), w="SANDSTORM") - 3277 / 4096) < 1e-9
    assert _h(d=_D(ability="sandveil")) == 1.0                # no sand → no effect
    assert abs(_h(d=_D(ability="snowcloak"), w="SNOWSCAPE") - 3277 / 4096) < 1e-9
    assert abs(_h(d=_D(ability="snowcloak"), w="HAIL") - 3277 / 4096) < 1e-9   # Snow Cloak: hail too
    assert _h(d=_D(ability="sandveil"), w="SNOWSCAPE") == 1.0  # wrong weather


def test_tangled_feet_and_evasion_item():
    assert _h(d=_D(ability="tangledfeet", confused=True)) == 0.5
    assert _h(d=_D(ability="tangledfeet")) == 1.0             # not confused → no effect
    assert abs(_h(d=_D(evasion_item=True)) - 3686 / 4096) < 1e-9   # Bright Powder / Lax Incense


def test_ignore_evasion_negates_stage_only():
    # Sacred Sword (ignoreEvasion) + Keen Eye attacker both negate the eva STAGE…
    assert _h(d=_D(eva_stage=6), mid="sacredsword") == 1.0
    assert _h(ab="keeneye", d=_D(eva_stage=6)) == 1.0
    # …but do NOT negate Sand Veil (a separate accuracy mult).
    assert abs(_h(d=_D(ability="sandveil", eva_stage=6), mid="sacredsword", w="SANDSTORM") - 3277 / 4096) < 1e-9


def test_combined_boost_not_two_multipliers():
    # the regression for the workflow-caught bug: att +2 acc vs def +1 eva = ONE net boost clamp(2-1)=+1 → ×4/3,
    # NOT (3+2)/3 × 3/(3+1) = 5/4. So 0.75 → 1.0 (clamped), not 0.9375.
    assert _h(0.75, stage=2, d=_D(eva_stage=1)) == 1.0
    assert abs(_h(0.6, stage=2, d=_D(eva_stage=1)) - 0.6 * 4 / 3) < 1e-9


def test_ohko_bypasses_evasion():
    assert _h(0.3, oh=True, d=_D(eva_stage=6)) == 0.3        # OHKO accuracy untouched by evasion


def test_move_ignores_evasion_helper():
    assert _move_ignores_evasion("sacredsword") is True
    assert _move_ignores_evasion("chipaway") is True
    assert _move_ignores_evasion("tackle") is False


# ════════════════════════════ e2e: channel placement + caller logic ════════════════════════════
def _mon(species, ability, *, moves=(), boosts=None):
    return {"species": species, "base_species": species, "hp_pct": 100.0, "seen": True, "is_fainted": False,
            "known_moves": list(moves), "revealed_moves": [], "boosts": boosts or {}, "status": None,
            "known_ability": ability, "volatiles": {},
            "stats_estimate": {"mode": "exact",
                               "stats": {"atk": 130, "spa": 100, "def": 100, "spd": 100, "hp": 300, "spe": 100}}}


def _hitchance_channels(att, move, opp_a, opp_b=None):
    snap = {"our_active": {"our_a": att, "our_b": None},
            "opp_active": {"opp_a": opp_a, "opp_b": opp_b},
            "our_bench": [], "opp_bench": [], "field": {}, "side_conditions": {}}
    vec = StateEncoder().encode_snapshot(snap, turn=1)
    for m_idx, (mv, _c) in enumerate(move_slots_for_mon(att)):
        if norm_species(mv) == norm_species(move):
            b = _MOVE_BLOCK_REL + m_idx * MOVE_FEATURES
            return float(vec[b + 23]), float(vec[b + 24])     # hit-chance vs opp_a, vs opp_b (core offsets 23,24)
    raise AssertionError(move)


def test_b2b_e2e_evasion_reduces_and_empty_slot_zero():
    att = _mon("Garchomp", "Rough Skin", moves=["Stone Edge"])     # Stone Edge = 80% accuracy
    evasive = _mon("Blissey", "Natural Cure", boosts={"evasion": 2})
    c0, c1 = _hitchance_channels(att, "Stone Edge", evasive, opp_b=None)
    assert abs(c0 - 0.80 * 3 / 5) < 1e-3       # 80% × 3/(3+2) = 0.48
    assert c1 == 0.0                           # empty opp_b slot → 0.0


def test_b2b_e2e_always_hit_is_one():
    att = _mon("Garchomp", "Rough Skin", moves=["Aerial Ace"])     # always-hit
    evasive = _mon("Blissey", "Natural Cure", boosts={"evasion": 6})
    c0, _ = _hitchance_channels(att, "Aerial Ace", evasive)
    assert c0 == 1.0                           # always-hit bypasses evasion entirely
