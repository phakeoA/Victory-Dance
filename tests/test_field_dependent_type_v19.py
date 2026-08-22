"""Regression lock for v19b — FIELD-DEPENDENT move type (Weather Ball / Terrain Pulse).

Without this the encoder rated Weather Ball / Terrain Pulse as their base Normal type, so the type-eff /
damage-band channels mis-rated them (a rain-boosted Water Weather Ball read neutral instead of super-
effective). Value-only fix (no layout change). Offline↔live parity is covered by tests/test_encoder_parity.py.
"""
from __future__ import annotations

from v_dance.encoders.battle_mechanics import field_dependent_move_type as F
from v_dance.encoders.encoder_layout import _TYPE_IDX, NUM_TYPES
from v_dance.encoders.action_codec import _MOVE_BLOCK_REL
from v_dance.encoders.state_encoder import StateEncoder


def test_weather_ball_type_follows_weather():
    assert F("weatherball", "Normal", "RainDance", None) == "Water"
    assert F("weatherball", "Normal", "SunnyDay", None) == "Fire"
    assert F("weatherball", "Normal", "Sandstorm", None) == "Rock"
    assert F("weatherball", "Normal", "Snowscape", None) == "Ice"
    assert F("weatherball", "Normal", "Hail", None) == "Ice"        # gen9 Hail → snow
    assert F("weatherball", "Normal", None, None) == "Normal"       # no weather → base


def test_terrain_pulse_type_follows_terrain():
    assert F("terrainpulse", "Normal", None, "electric") == "Electric"
    assert F("terrainpulse", "Normal", None, "grassy") == "Grass"
    assert F("terrainpulse", "Normal", None, "psychic") == "Psychic"
    assert F("terrainpulse", "Normal", None, "misty") == "Fairy"
    assert F("terrainpulse", "Normal", None, None) == "Normal"


def test_non_field_moves_unchanged():
    assert F("surf", "Water", "RainDance", None) == "Water"          # ordinary Water move untouched
    assert F("flamethrower", "Fire", "SunnyDay", None) == "Fire"
    assert F("tackle", "Normal", "RainDance", None) == "Normal"


def _mon(sp, mv):
    return {"species": sp, "base_species": sp, "hp_pct": 100.0, "seen": True, "is_fainted": False,
            "known_moves": list(mv), "revealed_moves": [], "boosts": {}, "status": None}


def test_encoded_weather_ball_type_flips_with_weather():
    type_rel = _MOVE_BLOCK_REL + 1   # move-0 type feature

    def enc(weather):
        snap = {"our_active": {"our_a": _mon("Charizard", ("Weather Ball", "Air Slash")), "our_b": None},
                "opp_active": {"opp_a": _mon("Incineroar", ("Knock Off",)), "opp_b": None},
                "our_bench": [], "opp_bench": [], "field": ({"weather": weather} if weather else {}),
                "side_conditions": {}}
        return StateEncoder().encode_snapshot(snap, turn=3)

    assert round(enc("RainDance")[type_rel] * (NUM_TYPES - 1)) == _TYPE_IDX["WATER"]    # rain → Water
    assert round(enc(None)[type_rel] * (NUM_TYPES - 1)) == _TYPE_IDX["NORMAL"]          # none → Normal
