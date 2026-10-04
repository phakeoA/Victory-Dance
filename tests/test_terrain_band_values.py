"""Terrain values in the damage band — the PRE-v19d values every checkpoint was trained on (locked 2026-10-04).

v19d (2026-10-02) tried server-exact terrain mechanics (attacker-keyed terrain ×1.3, Expanding Force ×1.5 + spread,
Grassy Glide +1 priority); they cost g50 ~8 pp on inputs no served checkpoint had seen, went default-OFF the same
day and were REMOVED 2026-10-04 (cleanup pass 2). These locks keep the values the nets know: the terrain ×1.3
keys on the DEFENDER being grounded, Expanding Force is a plain single-target move, Grassy Glide has no priority.
A retrain that wants the server-exact mechanics must re-add them deliberately (and break these locks).
"""
from __future__ import annotations

import math

from v_dance.encoders.battle_mechanics import _situational_damage_mult
from v_dance.encoders.state_encoder import (
    StateEncoder, MOVE_FEATURES, _MOVE_BLOCK_REL, move_slots_for_mon, norm_species,
)

OFF_PRIO = 3        # raw move-data priority channel
OFF_SPREAD = 8      # is_spread channel
OFF_DAMAGE = 13     # band min/max vs enemy0
OFF_FIRST = 17      # who-moves-first vs enemy0
_TANK = {"atk": 80, "spa": 80, "def": 300, "spd": 300, "hp": 700, "spe": 40}
_FAST_TANK = dict(_TANK, spe=400)


def test_terrain_boost_keys_on_the_defender():
    flying = {"grounded": False, "types": ["STEEL", "FLYING"]}
    grounded = {"grounded": True, "types": ["NORMAL"]}
    assert math.isclose(_situational_damage_mult("PSYCHIC", False, None, "PSYCHIC_TERRAIN", grounded,
                                                 False, False, False), 1.3)
    assert math.isclose(_situational_damage_mult("PSYCHIC", False, None, "PSYCHIC_TERRAIN", flying,
                                                 False, False, False), 1.0)
    assert math.isclose(_situational_damage_mult("GRASS", True, None, "GRASSY_TERRAIN", grounded,
                                                 False, False, False), 1.3)
    assert math.isclose(_situational_damage_mult("DRAGON", False, None, "MISTY_TERRAIN", grounded,
                                                 False, False, False), 0.5)


# ── encoder integration (offline writer; the live writer shares the same helpers) ──
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


def test_expanding_force_is_a_plain_single_target_move_in_psychic_terrain():
    indeedee = _mon("Indeedee-F", "Psychic Surge", moves=["Expanding Force"])
    tank = _mon("Snorlax", "Thick Fat", stats=_TANK)                      # Normal: neutral, grounded
    clear = _move_block(indeedee, tank, "Expanding Force", None)
    pt = _move_block(indeedee, tank, "Expanding Force", "psychic")
    assert float(clear[OFF_SPREAD]) == 0.0 and float(pt[OFF_SPREAD]) == 0.0
    assert float(clear[OFF_DAMAGE + 1]) > 0
    assert _close(float(pt[OFF_DAMAGE + 1]), float(clear[OFF_DAMAGE + 1]) * 1.3)   # the terrain ×1.3 only


def test_no_terrain_boost_into_a_flying_defender():
    indeedee = _mon("Indeedee-F", "Psychic Surge", moves=["Expanding Force"])
    corv = _mon("Corviknight", "Mirror Armor", stats=_TANK)               # Steel/Flying: NOT grounded
    clear = float(_move_block(indeedee, corv, "Expanding Force", None)[OFF_DAMAGE + 1])
    pt = float(_move_block(indeedee, corv, "Expanding Force", "psychic")[OFF_DAMAGE + 1])
    assert clear > 0 and _close(pt, clear)


def test_an_airborne_user_still_boosts_into_a_grounded_defender():
    latios = _mon("Latios", "Levitate", moves=["Expanding Force"])
    tank = _mon("Snorlax", "Thick Fat", stats=_TANK)
    clear = _move_block(latios, tank, "Expanding Force", None)
    pt = _move_block(latios, tank, "Expanding Force", "psychic")
    assert float(pt[OFF_SPREAD]) == 0.0
    assert _close(float(pt[OFF_DAMAGE + 1]), float(clear[OFF_DAMAGE + 1]) * 1.3)   # defender-keyed


def test_grassy_glide_has_no_priority_in_grassy_terrain():
    rilla = _mon("Rillaboom", "Grassy Surge", moves=["Grassy Glide"])
    clear = _move_block(rilla, _mon("Snorlax", "Thick Fat", stats=_FAST_TANK), "Grassy Glide", None)
    gt = _move_block(rilla, _mon("Snorlax", "Thick Fat", stats=_FAST_TANK), "Grassy Glide", "grassy")
    assert float(clear[OFF_FIRST]) < 0 and float(gt[OFF_FIRST]) == float(clear[OFF_FIRST])
    assert float(gt[OFF_PRIO]) == float(clear[OFF_PRIO]) == 0.0
