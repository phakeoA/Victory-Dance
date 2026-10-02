"""The Champions mega abilities the damage band missed (mega audit gap 4, 2026-10-02).

Value-only (no layout change; _CACHE_SCHEMA 6), every rule read from the pinned server (data/abilities.ts,
sim/pokemon.ts):
  · Fairy Aura (Mega Floette)  — every Fairy move on the field ×5448/4096; ×3072/4096 with Aura Break out
  · Fire Mane (Mega Pyroar)    — Fire moves ×1.5
  · Mega Sol (Mega Meganium)   — the holder's own moves act as in sun (Weather Ball → Fire, Solar Beam not
                                 halved in rain, Fire ×1.5 / Water ×0.5); the redundant-condition bit keeps the
                                 FIELD weather
  · Dragonize (Mega Feraligatr) — the Normal → Dragon retype was there, the ×1.2 boost was not
  · Eelevate (Mega Eelektross) — airborne + Ground-immune, like Levitate
Offline↔live parity is by construction (shared battle_mechanics helpers) + tests/test_encoder_parity.py.
"""
from __future__ import annotations

import math

from v_dance.encoders.battle_mechanics import (
    _ability_damage_mult, _is_grounded, _move_immune, attacker_weather, aura_mult, field_auras,
)
from v_dance.encoders.encoder_layout import NUM_TYPES, _TYPE_IDX
from v_dance.encoders.state_encoder import (
    StateEncoder, MOVE_FEATURES, _MOVE_BLOCK_REL, move_slots_for_mon, norm_species,
)

OFF_TYPE = 1
OFF_IMMUNE = 10     # type-eff immune flag vs enemy0
OFF_DAMAGE = 13     # band min/max vs enemy0
OFF_REDUNDANT = 25  # v18 redundant-condition bit
_TANK = {"atk": 80, "spa": 80, "def": 300, "spd": 300, "hp": 700, "spe": 40}
_AURA = 5448 / 4096


def _mon(species, ability, *, moves=(), stats=None):
    return {
        "species": species, "base_species": species, "hp_pct": 100.0,
        "seen": True, "is_fainted": False, "known_moves": list(moves), "revealed_moves": [],
        "boosts": {}, "status": None, "known_ability": ability,
        "stats_estimate": {"mode": "exact",
                           "stats": stats or {"atk": 100, "spa": 100, "def": 100,
                                              "spd": 100, "hp": 200, "spe": 100}},
    }


def _move_block(attacker, defender, move, *, weather=None, partner=None, opp_partner=None):
    snap = {"our_active": {"our_a": attacker, "our_b": partner},
            "opp_active": {"opp_a": defender, "opp_b": opp_partner},
            "our_bench": [], "opp_bench": [],
            "field": ({"weather": weather} if weather else {}), "side_conditions": {}}
    vec = StateEncoder().encode_snapshot(snap, turn=3)
    for m_idx, (mv, _c) in enumerate(move_slots_for_mon(attacker)):
        if norm_species(mv) == norm_species(move):
            b = _MOVE_BLOCK_REL + m_idx * MOVE_FEATURES
            return vec[b: b + MOVE_FEATURES]
    raise AssertionError(f"{move} not in attacker slots")


def _band(*a, **k) -> float:
    return float(_move_block(*a, **k)[OFF_DAMAGE + 1])


def _close(a, b):
    return math.isclose(a, b, rel_tol=2e-3, abs_tol=1e-6)


def _type_is(block, t: str) -> bool:
    return math.isclose(float(block[OFF_TYPE]), _TYPE_IDX[t] / (NUM_TYPES - 1), rel_tol=1e-6)   # float32 row


# ── helper-level locks ────────────────────────────────────────────────────────
def test_aura_helpers():
    assert field_auras(["fairyaura", "intimidate", None]) == frozenset({"fairyaura"})
    assert field_auras([]) == frozenset()
    assert math.isclose(aura_mult("FAIRY", {"fairyaura"}), _AURA)
    assert math.isclose(aura_mult("FAIRY", {"fairyaura", "aurabreak"}), 3072 / 4096)
    assert aura_mult("DARK", {"fairyaura"}) == 1.0
    assert aura_mult("FAIRY", frozenset()) == 1.0
    assert math.isclose(aura_mult("DARK", {"darkaura"}), _AURA)


def test_mega_sol_weather_helper():
    assert attacker_weather("RAINDANCE", "megasol", "weatherball") == "SUNNYDAY"
    assert attacker_weather(None, "megasol", "solarbeam") == "SUNNYDAY"
    assert attacker_weather("RAINDANCE", "megasol", "electroshot") == "RAINDANCE"   # the one exception
    assert attacker_weather("RAINDANCE", "overgrow", "weatherball") == "RAINDANCE"


def test_fire_mane_and_dragonize_mults():
    kw = dict(is_stab=False, is_physical=False, type_mult=1.0, hp_frac=1.0, att_burned=False, att_statused=False)
    assert math.isclose(_ability_damage_mult("flamethrower", "firemane", None, eff_move_type="FIRE", **kw), 1.5)
    assert _ability_damage_mult("hypervoice", "firemane", None, eff_move_type="NORMAL", **kw) == 1.0
    assert math.isclose(_ability_damage_mult("bodyslam", "dragonize", None, eff_move_type="DRAGON", **kw),
                        4915 / 4096)
    assert _ability_damage_mult("weatherball", "dragonize", None, eff_move_type="NORMAL", **kw) == 1.0


def test_eelevate_floats_like_levitate():
    assert _is_grounded(["ELECTRIC"], "eelevate", "", False, False) is False
    assert _is_grounded(["ELECTRIC"], "eelevate", "ironball", False, False) is True
    assert _move_immune("GROUND", {"ability": "eelevate"}, None, "earthquake") is True
    assert _move_immune("GROUND", {"ability": "eelevate"}, "moldbreaker", "earthquake") is False
    assert _move_immune("GROUND", {"ability": "eelevate", "force_grounded": True}, None, "earthquake") is False


# ── encoder integration (offline writer; live is the parity twin) ────────────
def test_fairy_aura_boosts_every_fairy_move_on_the_field():
    sylveon = _mon("Sylveon", "Pixilate", moves=["Moonblast"])
    tank = _mon("Snorlax", "Thick Fat", stats=_TANK)
    floette = _mon("Floette-Mega", "Fairy Aura")
    plain = _band(sylveon, tank, "Moonblast")
    assert plain > 0
    assert _close(_band(sylveon, tank, "Moonblast", partner=floette), plain * _AURA)        # the partner's aura
    assert _close(_band(sylveon, tank, "Moonblast", opp_partner=floette), plain * _AURA)    # an OPPONENT's aura
    zygarde = _mon("Zygarde-Mega", "Aura Break")
    assert _close(_band(sylveon, tank, "Moonblast", partner=floette, opp_partner=zygarde), plain * 3072 / 4096)


def test_fire_mane_boosts_fire_only():
    pyroar = _mon("Pyroar-Mega", "Fire Mane", moves=["Flamethrower", "Hyper Voice"])
    plain = _mon("Pyroar-Mega", "Unnerve", moves=["Flamethrower", "Hyper Voice"])
    tank = _mon("Snorlax", "Thick Fat", stats=_TANK)
    assert _close(_band(pyroar, tank, "Flamethrower"), _band(plain, tank, "Flamethrower") * 1.5 / 0.5 * 0.5)
    assert _close(_band(pyroar, tank, "Hyper Voice"), _band(plain, tank, "Hyper Voice"))


def test_mega_sol_moves_act_as_in_sun():
    meganium = _mon("Meganium-Mega", "Mega Sol", moves=["Weather Ball", "Solar Beam", "Sunny Day"])
    tank = _mon("Snorlax", "Thick Fat", stats=_TANK)
    rain_wb = _move_block(meganium, tank, "Weather Ball", weather="RainDance")
    assert _type_is(rain_wb, "FIRE")                     # not Water: it sees sun
    plain = _mon("Meganium-Mega", "Overgrow", moves=["Weather Ball", "Solar Beam", "Sunny Day"])
    rain_sb = _band(meganium, tank, "Solar Beam", weather="RainDance")
    assert _close(rain_sb, _band(plain, tank, "Solar Beam", weather="RainDance") * 2.0)   # not halved
    # Mega Sol never SETS sun: Sunny Day stays a real (non-redundant) move in clear weather
    assert float(_move_block(meganium, tank, "Sunny Day")[OFF_REDUNDANT]) == 0.0


def test_dragonize_retypes_and_boosts():
    gatr = _mon("Feraligatr-Mega", "Dragonize", moves=["Body Slam"])
    tank = _mon("Snorlax", "Thick Fat", stats=_TANK)
    blk = _move_block(gatr, tank, "Body Slam")
    assert _type_is(blk, "DRAGON")
    plain = _mon("Feraligatr-Mega", "Torrent", moves=["Body Slam"])
    # ×1.2 Dragonize · ×1.5 Dragon STAB (Feraligatr-Mega is Water/Dragon) vs a Normal body slam with no STAB
    assert _close(float(blk[OFF_DAMAGE + 1]), _band(plain, tank, "Body Slam") * (4915 / 4096) * 1.5)


def test_earthquake_into_mega_eelektross_is_immune():
    chomp = _mon("Garchomp", "Rough Skin", moves=["Earthquake"])
    eel = _mon("Eelektross-Mega", "Eelevate", stats=_TANK)
    blk = _move_block(chomp, eel, "Earthquake")
    assert float(blk[OFF_IMMUNE]) == 1.0
    assert float(blk[OFF_DAMAGE + 1]) == 0.0
