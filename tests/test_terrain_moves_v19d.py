"""v19d — TERRAIN-CONDITIONAL move mechanics (USER 2026-10-02: "terrain wars and weather wars are concepts it
doesn't understand"; "do the expanding force fix").

Value-only fixes (no layout change; _CACHE_SCHEMA 5), every rule read from the pinned server's data/moves.ts:
  · the Electric / Grassy / Psychic terrain ×1.3 keys on the ATTACKER being grounded (it keyed on the defender:
    a grounded Indeedee's Psychic hit into Corviknight / Salamence lost the boost, a Flying attacker gained one)
  · Expanding Force ×1.5 AND hits both foes (is_spread → the doubles ×0.75) under Psychic Terrain, grounded user
  · Grassy Glide +1 priority under Grassy Terrain (grounded user) → the who-moves-first channel
Offline↔live parity is by construction (shared battle_mechanics helpers) + tests/test_encoder_parity.py.

2026-10-02 (bisect): the v19d values are now OPT-IN (VD_TERRAIN_V19D=1) — they cost the served g50 ~8 pp on
Baltimore because it was trained on the pre-v19d values. These locks run with the switch ON; the last test locks
the default (pre-v19d) values.
"""
from __future__ import annotations

import math

import pytest

import v_dance.encoders.battle_mechanics as _BM
from v_dance.encoders.battle_mechanics import (
    _situational_damage_mult, terrain_bp_mult, terrain_priority, terrain_spread,
)


@pytest.fixture(autouse=True)
def _v19d_on(request, monkeypatch):
    if "pre_v19d" not in request.node.name:
        monkeypatch.setattr(_BM, "TERRAIN_V19D", True)   # the helpers read the module flag at call time
from v_dance.encoders.state_encoder import (
    StateEncoder, MOVE_FEATURES, _MOVE_BLOCK_REL, move_slots_for_mon, norm_species,
)

OFF_PRIO = 3        # raw move-data priority channel (stays the data value)
OFF_SPREAD = 8      # is_spread channel
OFF_DAMAGE = 13     # band min/max vs enemy0
OFF_FIRST = 17      # who-moves-first vs enemy0
_TANK = {"atk": 80, "spa": 80, "def": 300, "spd": 300, "hp": 700, "spe": 40}
_FAST_TANK = dict(_TANK, spe=400)


# ── helper-level locks ────────────────────────────────────────────────────────
def test_expanding_force_helpers():
    assert terrain_bp_mult("expandingforce", "PSYCHIC_TERRAIN", True) == 1.5
    assert terrain_bp_mult("expandingforce", "PSYCHIC_TERRAIN", None) == 1.5     # unknown = grounded
    assert terrain_bp_mult("expandingforce", "PSYCHIC_TERRAIN", False) == 1.0    # Levitate / Flying user
    assert terrain_bp_mult("expandingforce", "GRASSY_TERRAIN", True) == 1.0
    assert terrain_bp_mult("expandingforce", None, True) == 1.0
    assert terrain_bp_mult("psychic", "PSYCHIC_TERRAIN", True) == 1.0            # ordinary moves untouched
    assert terrain_spread("expandingforce", "PSYCHIC_TERRAIN", True) is True
    assert terrain_spread("expandingforce", "PSYCHIC_TERRAIN", False) is False
    assert terrain_spread("expandingforce", None, True) is False
    assert terrain_spread("psychic", "PSYCHIC_TERRAIN", True) is False


def test_grassy_glide_priority_helper():
    assert terrain_priority("grassyglide", 0, "GRASSY_TERRAIN", True) == 1
    assert terrain_priority("grassyglide", 0, "GRASSY_TERRAIN", False) == 0
    assert terrain_priority("grassyglide", 0, "PSYCHIC_TERRAIN", True) == 0
    assert terrain_priority("grassyglide", 0, None, True) == 0
    assert terrain_priority("fakeout", 3, "GRASSY_TERRAIN", True) == 3


def test_terrain_boost_keys_on_the_attacker():
    flying = {"grounded": False, "types": ["STEEL", "FLYING"]}
    grounded = {"grounded": True, "types": ["NORMAL"]}
    # grounded attacker into an airborne defender: boosted (was 1.0 before v19d)
    assert math.isclose(_situational_damage_mult("PSYCHIC", False, None, "PSYCHIC_TERRAIN", flying,
                                                 False, False, False, attacker_grounded=True), 1.3)
    # airborne attacker into a grounded defender: NOT boosted (was 1.3 before v19d)
    assert math.isclose(_situational_damage_mult("GRASS", True, None, "GRASSY_TERRAIN", grounded,
                                                 False, False, False, attacker_grounded=False), 1.0)
    # the defender-keyed terrain effects stay on the defender
    assert math.isclose(_situational_damage_mult("DRAGON", False, None, "MISTY_TERRAIN", grounded,
                                                 False, False, False, attacker_grounded=False), 0.5)
    assert math.isclose(_situational_damage_mult("GROUND", True, None, "GRASSY_TERRAIN", flying,
                                                 False, False, False, grassy_eq=True, attacker_grounded=True), 1.0)
    # a caller that does not pass the attacker: the pre-v19d defender-keyed behaviour
    assert math.isclose(_situational_damage_mult("PSYCHIC", False, None, "PSYCHIC_TERRAIN", flying,
                                                 False, False, False), 1.0)


# ── encoder integration (offline writer; live is the parity twin) ────────────
def _mon(species, ability, *, moves=(), stats=None):
    return {
        "species": species, "base_species": species, "hp_pct": 100.0,
        "seen": True, "is_fainted": False, "known_moves": list(moves), "revealed_moves": [],
        "boosts": {}, "status": None, "known_ability": ability,
        "stats_estimate": {"mode": "exact",
                           "stats": stats or {"atk": 100, "spa": 100, "def": 100,
                                              "spd": 100, "hp": 200, "spe": 100}},
    }


def _move_block(attacker, defender, move, terrain):
    snap = {"our_active": {"our_a": attacker, "our_b": None},
            "opp_active": {"opp_a": defender, "opp_b": None},
            "our_bench": [], "opp_bench": [],
            "field": ({"terrain": terrain} if terrain else {}), "side_conditions": {}}
    vec = StateEncoder().encode_snapshot(snap, turn=3)
    for m_idx, (mv, _c) in enumerate(move_slots_for_mon(attacker)):
        if norm_species(mv) == norm_species(move):
            b = _MOVE_BLOCK_REL + m_idx * MOVE_FEATURES
            return vec[b: b + MOVE_FEATURES]
    raise AssertionError(f"{move} not in attacker slots")


def _close(a, b):
    return math.isclose(a, b, rel_tol=2e-3, abs_tol=1e-6)


def test_expanding_force_hits_both_and_hits_harder_in_psychic_terrain():
    indeedee = _mon("Indeedee-F", "Psychic Surge", moves=["Expanding Force"])
    tank = _mon("Snorlax", "Thick Fat", stats=_TANK)                      # Normal: neutral, grounded
    clear = _move_block(indeedee, tank, "Expanding Force", None)
    pt = _move_block(indeedee, tank, "Expanding Force", "psychic")
    assert float(clear[OFF_SPREAD]) == 0.0 and float(pt[OFF_SPREAD]) == 1.0
    assert float(clear[OFF_DAMAGE + 1]) > 0
    # terrain ×1.3 · Expanding Force ×1.5 · spread ×0.75 (it now hits BOTH foes)
    assert _close(float(pt[OFF_DAMAGE + 1]), float(clear[OFF_DAMAGE + 1]) * 1.3 * 1.5 * 0.75)


def test_expanding_force_keeps_the_terrain_boost_into_a_flying_defender():
    indeedee = _mon("Indeedee-F", "Psychic Surge", moves=["Expanding Force"])
    corv = _mon("Corviknight", "Mirror Armor", stats=_TANK)               # Steel/Flying: NOT grounded
    clear = float(_move_block(indeedee, corv, "Expanding Force", None)[OFF_DAMAGE + 1])
    pt = float(_move_block(indeedee, corv, "Expanding Force", "psychic")[OFF_DAMAGE + 1])
    assert clear > 0
    assert _close(pt, clear * 1.3 * 1.5 * 0.75)                          # pre-v19d: × 1.5 × 0.75 only


def test_an_airborne_user_gets_nothing_from_psychic_terrain():
    latios = _mon("Latios", "Levitate", moves=["Expanding Force"])
    tank = _mon("Snorlax", "Thick Fat", stats=_TANK)
    clear = _move_block(latios, tank, "Expanding Force", None)
    pt = _move_block(latios, tank, "Expanding Force", "psychic")
    assert float(pt[OFF_SPREAD]) == 0.0
    assert _close(float(pt[OFF_DAMAGE + 1]), float(clear[OFF_DAMAGE + 1]))   # pre-v19d: ×1.3 (defender keyed)


def test_grassy_glide_moves_first_in_grassy_terrain():
    rilla = _mon("Rillaboom", "Grassy Surge", moves=["Grassy Glide"])
    clear = _move_block(rilla, _mon("Snorlax", "Thick Fat", stats=_FAST_TANK), "Grassy Glide", None)
    gt = _move_block(rilla, _mon("Snorlax", "Thick Fat", stats=_FAST_TANK), "Grassy Glide", "grassy")
    assert float(clear[OFF_FIRST]) < 0                                     # slower, same bracket
    assert float(gt[OFF_FIRST]) == 1.0                                     # priority bracket: moves first
    assert float(gt[OFF_PRIO]) == float(clear[OFF_PRIO]) == 0.0           # the raw data channel is unchanged


def test_default_is_the_pre_v19d_values(monkeypatch):
    # the name carries "pre_v19d" → the autouse fixture leaves the module default (OFF) in place
    monkeypatch.delenv("VD_TERRAIN_V19D", raising=False)
    assert _BM.TERRAIN_V19D is False
    assert terrain_bp_mult("expandingforce", "PSYCHIC_TERRAIN", True) == 1.0      # a plain single-target move
    assert terrain_spread("expandingforce", "PSYCHIC_TERRAIN", True) is False
    assert terrain_priority("grassyglide", 0, "GRASSY_TERRAIN", True) == 0
    flying = {"grounded": False, "types": ["FLYING"]}
    # the terrain x1.3 keys on the DEFENDER again: a grounded attacker's Psychic hit into a Flying target gets none
    assert _situational_damage_mult("PSYCHIC", False, None, "PSYCHIC_TERRAIN", flying,
                                    False, False, False, attacker_grounded=True) == 1.0
    assert "PRE-v19d" in _BM.terrain_values_banner()
