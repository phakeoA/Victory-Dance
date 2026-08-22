"""v11 Phase A.1b — FLAG-gated ability immunities (Bulletproof/Soundproof/Wind Rider) + partial signed
resists/weaknesses (Purifying Salt Ghost ×0.5, Dry Skin Fire ×1.25, Water Bubble Fire ×0.5 / Water ×2).
All six abilities are breakable → Mold Breaker bypasses them. Verified vs abilities.ts; the move flags
(bullet/sound/wind) come from GenData(9) so offline↔live stay byte-identical.
"""
from __future__ import annotations

import math

from v_dance.encoders.state_encoder import (
    StateEncoder, MOVE_FEATURES, _MOVE_BLOCK_REL, move_slots_for_mon, norm_species,
    _ability_immunizes, _ability_damage_mult, _move_damage_props,
)

OFF_MOVES = _MOVE_BLOCK_REL
OFF_TYPEEFF = 9      # within a move block: signed/immune vs enemy0 at +9/+10
OFF_DAMAGE = 13      # band min/max vs enemy0 at +13/+14


def _h(move_id, att=None, dfn=None, *, mt="GHOST", phys=False, tmult=1.0):
    return _ability_damage_mult(move_id, att, dfn, eff_move_type=mt, is_stab=False,
                                is_physical=phys, type_mult=tmult, hp_frac=0.5,
                                att_burned=False, att_statused=False)


def _close(a, b):
    return math.isclose(a, b, rel_tol=1e-4, abs_tol=1e-6)


# ── FLAG-gated immunities (via _ability_immunizes, needs move_id) ─────────────
def test_bulletproof_blocks_bullet_moves():
    assert _ability_immunizes("GHOST", "bulletproof", None, "shadowball") is True
    assert _ability_immunizes("POISON", "bulletproof", None, "sludgebomb") is True
    assert _ability_immunizes("FIGHTING", "bulletproof", None, "aurasphere") is True
    assert _ability_immunizes("GROUND", "bulletproof", None, "earthquake") is False   # not a bullet
    assert _ability_immunizes("GHOST", "bulletproof", None, "shadowball", ) is True
    # Mold Breaker bypasses (breakable ability)
    assert _ability_immunizes("GHOST", "bulletproof", "moldbreaker", "shadowball") is False


def test_soundproof_blocks_sound_moves():
    assert _ability_immunizes("NORMAL", "soundproof", None, "boomburst") is True
    assert _ability_immunizes("NORMAL", "soundproof", None, "hypervoice") is True
    assert _ability_immunizes("GROUND", "soundproof", None, "earthquake") is False
    assert _ability_immunizes("NORMAL", "soundproof", "teravolt", "boomburst") is False


def test_windrider_blocks_wind_moves():
    assert _ability_immunizes("FLYING", "windrider", None, "bleakwindstorm") is True
    assert _ability_immunizes("ICE", "windrider", None, "icywind") is True
    assert _ability_immunizes("FLYING", "windrider", None, "hurricane") is True       # Hurricane has wind flag
    assert _ability_immunizes("GROUND", "windrider", None, "earthquake") is False
    assert _ability_immunizes("FLYING", "windrider", "turboblaze", "bleakwindstorm") is False


def test_flag_immunity_needs_move_id():
    # without a move_id the flag path cannot fire (type path still works for type-immunities)
    assert _ability_immunizes("GHOST", "bulletproof", None) is False
    assert _ability_immunizes("GROUND", "levitate", None) is True     # A.1 type path unaffected


# ── partial signed resists/weaknesses (via _ability_damage_mult) ──────────────
def test_purifying_salt_ghost_resist():
    assert _h("shadowball", None, "purifyingsalt", mt="GHOST") == 0.5
    assert _h("earthquake", None, "purifyingsalt", mt="GROUND") == 1.0       # only Ghost
    assert _h("shadowball", "moldbreaker", "purifyingsalt", mt="GHOST") == 1.0   # MB bypass


def test_dry_skin_fire_weakness():
    assert _h("flamethrower", None, "dryskin", mt="FIRE") == 1.25
    assert _h("earthquake", None, "dryskin", mt="GROUND") == 1.0
    assert _h("flamethrower", "teravolt", "dryskin", mt="FIRE") == 1.0           # MB bypass


def test_water_bubble_both_sides():
    # defensive Fire ×0.5
    assert _h("flamethrower", None, "waterbubble", mt="FIRE") == 0.5
    assert _h("flamethrower", "moldbreaker", "waterbubble", mt="FIRE") == 1.0    # MB bypass
    # offensive Water ×2 (attacker)
    assert _h("surf", "waterbubble", None, mt="WATER", phys=False) == 2.0
    assert _h("earthquake", "waterbubble", None, mt="GROUND") == 1.0


def test_move_props_have_flag_keys():
    assert _move_damage_props("shadowball")["is_bullet"] is True
    assert _move_damage_props("bleakwindstorm")["is_wind"] is True
    assert _move_damage_props("boomburst")["is_sound"] is True
    assert _move_damage_props("earthquake")["is_bullet"] is False


# ── end-to-end through encode_snapshot (offline) ──────────────────────────────
def _mon(species, ability, *, moves=(), stats=None):
    return {
        "species": species, "base_species": species,
        "hp_pct": 100.0, "seen": True, "is_fainted": False,
        "known_moves": list(moves), "revealed_moves": [],
        "boosts": {}, "status": None, "known_ability": ability,
        "stats_estimate": {"mode": "exact",
                           "stats": stats or {"atk": 150, "spa": 150, "def": 100,
                                              "spd": 100, "hp": 200, "spe": 100}},
    }


_TANK = {"atk": 100, "spa": 100, "def": 250, "spd": 250, "hp": 600, "spe": 50}


def _chan(attacker, defender, move):
    """(signed, immune, dmin, dmax) of `move` used by attacker vs enemy0 (defender)."""
    snap = {"our_active": {"our_a": attacker, "our_b": None},
            "opp_active": {"opp_a": defender, "opp_b": None},
            "our_bench": [], "opp_bench": [], "field": {}, "side_conditions": {}}
    vec = StateEncoder().encode_snapshot(snap, turn=1)
    for m_idx, (mv, _c) in enumerate(move_slots_for_mon(attacker)):
        if norm_species(mv) == norm_species(move):
            te = OFF_MOVES + m_idx * MOVE_FEATURES + OFF_TYPEEFF
            dm = OFF_MOVES + m_idx * MOVE_FEATURES + OFF_DAMAGE
            return float(vec[te]), float(vec[te + 1]), float(vec[dm]), float(vec[dm + 1])
    raise AssertionError(f"{move} not in attacker move slots")


def test_e2e_soundproof_zeroes_sound_move():
    # Boomburst (Normal sound) vs Soundproof Gyarados (Water/Flying → Normal neutral) → band 0 + immune.
    signed, immune, dmin, dmax = _chan(_mon("Exploud", "Scrappy", moves=["Boomburst"]),
                                       _mon("Gyarados", "Soundproof"), "Boomburst")
    assert immune == 1.0 and dmin == 0.0 and dmax == 0.0


def test_e2e_bulletproof_zeroes_bullet_move():
    signed, immune, dmin, dmax = _chan(_mon("Gengar", "Cursed Body", moves=["Shadow Ball"]),
                                       _mon("Gyarados", "Bulletproof"), "Shadow Ball")  # Ghost neutral on Water/Flying
    assert immune == 1.0 and dmax == 0.0


def test_e2e_purifying_salt_halves_ghost_band():
    base = _chan(_mon("Gengar", "Cursed Body", moves=["Shadow Ball"]),
                 _mon("Gyarados", "Intimidate", stats=_TANK), "Shadow Ball")
    salted = _chan(_mon("Gengar", "Cursed Body", moves=["Shadow Ball"]),
                   _mon("Gyarados", "Purifying Salt", stats=_TANK), "Shadow Ball")
    assert base[3] > 0 and _close(salted[3], base[3] * 0.5)


def test_e2e_water_bubble_doubles_water_band():
    base = _chan(_mon("Gyarados", "Intimidate", moves=["Surf"]),
                 _mon("Snorlax", "Immunity", stats=_TANK), "Surf")
    bubble = _chan(_mon("Gyarados", "Water Bubble", moves=["Surf"]),
                   _mon("Snorlax", "Immunity", stats=_TANK), "Surf")
    assert base[3] > 0 and _close(bubble[3], base[3] * 2.0)
