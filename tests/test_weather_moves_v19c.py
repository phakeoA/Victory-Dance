"""v19c — WEATHER-CONDITIONAL move mechanics (online rain-loss defect, 2026-07-10).

The band sold a rain Solar Beam as a full instant 120 BP nuke (real: halved + a charge turn) — the online
bot spammed it into both confirmed rain losses (…2646978812 Sanatrax / …2646982193 jrosenfeld1992).
Value-only fixes (no layout change; _CACHE_SCHEMA 3):
  · Solar Beam/Blade damage band ×0.5 under rain/sand/snow + the two_turn_charge tag now DYNAMIC
    (cleared when sun — or Electro-Shot-rain — lets the move fire instantly)
  · Weather Ball damage band ×2 under any active weather (type morph was already v19b)
  · Hydro Steam ×1.5 in sun (net of the generic Water-in-Sun halving already applied by type)
  · Thunder/Hurricane/Blizzard weather accuracy (rain/snow bypass → always-hit; sun T/H → 50%)
Offline↔live parity is by construction (shared battle_mechanics helpers) + tests/test_encoder_parity.py.
"""
from __future__ import annotations

import math

from v_dance.encoders.battle_mechanics import (
    charge_skipped_now, weather_accuracy, weather_bp_mult,
)
from v_dance.encoders.mechanic_tags import MOVE_TAG_NAMES
from v_dance.encoders.state_encoder import (
    StateEncoder, MOVE_FEATURES, _MOVE_BLOCK_REL, move_slots_for_mon, norm_species,
)

OFF_ACC = 4        # accuracy channel within a move block
OFF_DAMAGE = 13    # band min/max vs enemy0
OFF_HIT = 23       # realized hit-chance vs enemy0
OFF_TAGS = 28      # move-tag multi-hot start
TWO_TURN = MOVE_TAG_NAMES.index("two_turn_charge")

# a high-bulk defender keeps bands under the [0,1] clamp so ratios are meaningful
_TANK = {"atk": 80, "spa": 80, "def": 300, "spd": 300, "hp": 700, "spe": 40}


# ── helper-level locks ────────────────────────────────────────────────────────
def test_weather_bp_mult_solar_family():
    for mid in ("solarbeam", "solarblade"):
        for w in ("RainDance", "PRIMORDIALSEA", "Sandstorm", "Snowscape", "Hail"):
            assert weather_bp_mult(mid, w) == 0.5
        for w in ("SunnyDay", "DESOLATELAND", None):
            assert weather_bp_mult(mid, w) == 1.0


def test_weather_bp_mult_weather_ball_and_hydro_steam():
    assert weather_bp_mult("weatherball", "RainDance") == 2.0
    assert weather_bp_mult("weatherball", "Sandstorm") == 2.0
    assert weather_bp_mult("weatherball", None) == 1.0
    assert weather_bp_mult("hydrosteam", "SunnyDay") == 3.0   # ×1.5 real × cancel the generic sun ×0.5
    assert weather_bp_mult("hydrosteam", "RainDance") == 1.0  # rain's generic Water ×1.5 already applies
    assert weather_bp_mult("hydrosteam", None) == 1.0
    assert weather_bp_mult("surf", "RainDance") == 1.0        # ordinary moves untouched
    assert weather_bp_mult("heatwave", "SunnyDay") == 1.0


def test_charge_skipped_now():
    assert charge_skipped_now("solarbeam", "SunnyDay") is True
    assert charge_skipped_now("solarbeam", "DESOLATELAND") is True
    assert charge_skipped_now("solarbeam", "RainDance") is False
    assert charge_skipped_now("solarbeam", None) is False
    assert charge_skipped_now("solarblade", "SunnyDay") is True
    assert charge_skipped_now("electroshot", "RainDance") is True   # the Archaludon rain staple
    assert charge_skipped_now("electroshot", "PRIMORDIALSEA") is True
    assert charge_skipped_now("electroshot", "SunnyDay") is False
    assert charge_skipped_now("meteorbeam", "SunnyDay") is False    # no weather skip for other chargers
    assert charge_skipped_now("tackle", "SunnyDay") is False


def test_weather_accuracy_hooks():
    assert weather_accuracy("hurricane", "RainDance") == (1.0, True)   # no accuracy check at all
    assert weather_accuracy("hurricane", "SunnyDay") == (0.5, False)
    assert weather_accuracy("hurricane", None) == (None, False)
    assert weather_accuracy("thunder", "RainDance") == (1.0, True)
    assert weather_accuracy("thunder", "SunnyDay") == (0.5, False)
    assert weather_accuracy("blizzard", "Snowscape") == (1.0, True)
    assert weather_accuracy("blizzard", "Hail") == (1.0, True)
    assert weather_accuracy("blizzard", "RainDance") == (None, False)
    assert weather_accuracy("tackle", "RainDance") == (None, False)


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


def _move_block(attacker, defender, move, weather):
    snap = {"our_active": {"our_a": attacker, "our_b": None},
            "opp_active": {"opp_a": defender, "opp_b": None},
            "our_bench": [], "opp_bench": [],
            "field": ({"weather": weather} if weather else {}), "side_conditions": {}}
    vec = StateEncoder().encode_snapshot(snap, turn=3)
    for m_idx, (mv, _c) in enumerate(move_slots_for_mon(attacker)):
        if norm_species(mv) == norm_species(move):
            b = _MOVE_BLOCK_REL + m_idx * MOVE_FEATURES
            return vec[b: b + MOVE_FEATURES]
    raise AssertionError(f"{move} not in attacker slots")


def _close(a, b):
    return math.isclose(a, b, rel_tol=2e-3, abs_tol=1e-6)


def test_solar_beam_band_halved_in_rain_full_in_sun():
    atk = _mon("Venusaur", "Overgrow", moves=["Solar Beam"])
    d = _mon("Politoed", "Drizzle", stats=_TANK)          # Water: Solar Beam is the SE 'trap' click
    clear = float(_move_block(atk, d, "Solar Beam", None)[OFF_DAMAGE + 1])
    rain = float(_move_block(atk, d, "Solar Beam", "RainDance")[OFF_DAMAGE + 1])
    sun = float(_move_block(atk, d, "Solar Beam", "SunnyDay")[OFF_DAMAGE + 1])
    assert clear > 0
    assert _close(rain, clear * 0.5)                       # the band stops selling the rain nuke
    assert _close(sun, clear)                              # sun: full power (and no charge turn)


def test_two_turn_charge_tag_is_dynamic():
    atk = _mon("Venusaur", "Overgrow", moves=["Solar Beam"])
    d = _mon("Politoed", "Drizzle", stats=_TANK)
    assert _move_block(atk, d, "Solar Beam", None)[OFF_TAGS + TWO_TURN] == 1.0
    assert _move_block(atk, d, "Solar Beam", "RainDance")[OFF_TAGS + TWO_TURN] == 1.0
    assert _move_block(atk, d, "Solar Beam", "SunnyDay")[OFF_TAGS + TWO_TURN] == 0.0

    arch = _mon("Archaludon", "Stamina", moves=["Electro Shot"])
    assert _move_block(arch, d, "Electro Shot", None)[OFF_TAGS + TWO_TURN] == 1.0
    assert _move_block(arch, d, "Electro Shot", "RainDance")[OFF_TAGS + TWO_TURN] == 0.0


def test_weather_ball_band_doubles_in_weather():
    # Sandstorm: WB morphs Normal→Rock (v19b) — neutral vs Electric both ways, no sand SpD boost
    # (defender is not Rock), attacker gets STAB neither way → the ratio isolates the ×2 BP.
    atk = _mon("Garchomp", "Rough Skin", moves=["Weather Ball"])
    d = _mon("Pikachu", "Static", stats=_TANK)
    clear = float(_move_block(atk, d, "Weather Ball", None)[OFF_DAMAGE + 1])
    sand = float(_move_block(atk, d, "Weather Ball", "Sandstorm")[OFF_DAMAGE + 1])
    assert clear > 0
    assert _close(sand, clear * 2.0)


def test_hurricane_accuracy_and_hit_chance_follow_weather():
    atk = _mon("Pelipper", "Drizzle", moves=["Hurricane"])
    d = _mon("Politoed", "Damp", stats=_TANK)
    clear = _move_block(atk, d, "Hurricane", None)
    rain = _move_block(atk, d, "Hurricane", "RainDance")
    sun = _move_block(atk, d, "Hurricane", "SunnyDay")
    assert _close(float(clear[OFF_ACC]), 0.7)
    assert float(rain[OFF_ACC]) == 1.0
    assert _close(float(sun[OFF_ACC]), 0.5)
    assert _close(float(clear[OFF_HIT]), 0.7)              # realized hit chance vs enemy0
    assert float(rain[OFF_HIT]) == 1.0                     # rain: the accuracy check is skipped entirely
    assert _close(float(sun[OFF_HIT]), 0.5)
