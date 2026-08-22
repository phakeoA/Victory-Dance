"""v11 Phase C.2c gap-fix — grounding (Magnet Rise/Telekinesis/Smack Down/Iron Ball/Air Balloon) + the
Ground-move immunity it implies + Tar Shot (Fire ×2).

The static typing-only grounded read mis-handled these for Arena Trap (B.3), terrain ×1.3, and Ground
immunity (a Magnet-Rise'd / Air-Balloon mon is immune to Ground — was showing full damage). Mirrors
pokemon-showdown sim/pokemon.ts isGrounded() (Gravity excluded — not tracked offline). Value-only.
"""
from __future__ import annotations

import numpy as np

from v_dance.encoders.state_encoder import (
    _is_grounded, _ground_immune, _move_immune, StateEncoder, MOVE_FEATURES, _MOVE_BLOCK_REL,
    move_slots_for_mon, norm_species,
)
from v_dance.parser.vod_parser.battle_models import volatile_flags

OFF_TE = _MOVE_BLOCK_REL + 9      # signed/immune vs enemy0
OFF_DMG = _MOVE_BLOCK_REL + 13    # band min/max vs enemy0


# ── volatile_flags grounding keys ─────────────────────────────────────────────
def test_volatile_flags_grounding_keys():
    assert volatile_flags({"magnetrise"})["levitating"] is True
    assert volatile_flags({"telekinesis"})["levitating"] is True
    assert volatile_flags({"smackdown"})["force_grounded"] is True
    assert volatile_flags({"ingrain"})["force_grounded"] is True
    assert volatile_flags({"tarshot"})["tar_shot"] is True
    assert volatile_flags(set())["levitating"] is False


# ── _is_grounded precedence (mirrors isGrounded) ──────────────────────────────
def test_is_grounded_precedence():
    G, L = ["DRAGON", "GROUND"], ["FLYING"]
    assert _is_grounded(G, "roughskin", "", False, False) is True            # plain grounded
    assert _is_grounded(L, "keeneye", "", False, False) is False             # Flying → ungrounded
    assert _is_grounded(G, "levitate", "", False, False) is False            # Levitate → ungrounded
    assert _is_grounded(G, "roughskin", "airballoon", False, False) is False  # Air Balloon
    assert _is_grounded(G, "roughskin", "", True, False) is False            # Magnet Rise/Telekinesis
    # force-grounded OVERRIDES every ungrounder
    assert _is_grounded(L, "keeneye", "", False, True) is True               # Smack Down on a Flyer
    assert _is_grounded(L, "levitate", "ironball", True, False) is True      # Iron Ball overrides all


def test_ground_immune():
    assert _ground_immune("airballoon", False, False) is True
    assert _ground_immune("", True, False) is True                          # Magnet Rise
    assert _ground_immune("", True, True) is False                          # Smack Down forces grounded
    assert _ground_immune("ironball", True, False) is False                 # Iron Ball forces grounded
    assert _ground_immune("", False, False) is False


def test_move_immune_ground_not_moldbreaker_bypassed():
    d = {"types": ["NORMAL"], "ability": "thickfat", "ground_immune": True}
    assert _move_immune("GROUND", d, None, "earthquake") is True
    assert _move_immune("GROUND", d, "moldbreaker", "earthquake") is True   # NOT bypassed (item/volatile)
    assert _move_immune("FIRE", d, None, "flamethrower") is False           # only Ground


# ── end-to-end: Ground immunity + Tar Shot ────────────────────────────────────
def _mon(species, ability, *, moves=(), volatiles=None, stats=None):
    return {
        "species": species, "base_species": species, "hp_pct": 100.0, "seen": True,
        "is_fainted": False, "known_moves": list(moves), "revealed_moves": [], "boosts": {},
        "status": None, "known_ability": ability, "volatiles": volatiles or {},
        "stats_estimate": {"mode": "exact",
                           "stats": stats or {"atk": 130, "spa": 110, "def": 110, "spd": 110,
                                              "hp": 280, "spe": 60}},
    }


def _chan(att, dfn, move):
    snap = {"our_active": {"our_a": att, "our_b": None}, "opp_active": {"opp_a": dfn, "opp_b": None},
            "our_bench": [], "opp_bench": [], "field": {}, "side_conditions": {}}
    vec = StateEncoder().encode_snapshot(snap, turn=1)
    for m_idx, (mv, _c) in enumerate(move_slots_for_mon(att)):
        if norm_species(mv) == norm_species(move):
            b = _MOVE_BLOCK_REL + m_idx * MOVE_FEATURES
            return float(vec[b + 9]), float(vec[b + 10]), float(vec[b + 13 + 1])   # signed, immune, dmax
    raise AssertionError(move)


def test_e2e_magnet_rise_ground_immunity():
    chomp = _mon("Garchomp", "Rough Skin", moves=["Earthquake"])
    grounded = _chan(chomp, _mon("Snorlax", "Immunity"), "Earthquake")
    levitating = _chan(chomp, _mon("Snorlax", "Immunity", volatiles={"levitating": True}), "Earthquake")
    assert grounded[1] == 0.0 and grounded[2] > 0.0          # grounded → hit
    assert levitating[1] == 1.0 and levitating[2] == 0.0     # Magnet Rise → Ground immune


def test_e2e_air_balloon_ground_immunity():
    chomp = _mon("Garchomp", "Rough Skin", moves=["Earthquake"])
    # known_item path: Air Balloon → resolve_item_json → airballoon → ground_immune
    bal = _mon("Heatran", "Flash Fire", moves=[])
    bal["known_item"] = "Air Balloon"
    res = _chan(chomp, bal, "Earthquake")
    assert res[1] == 1.0 and res[2] == 0.0


def test_smack_down_grounds_for_trap_and_terrain():
    # Smack Down/Ingrain force a Flyer GROUNDED → Arena-trappable + terrain-affected. (Negating the
    # Flying *typing* Ground-immunity in the damage band is a separate, rarer chart change — DEFERRED.)
    from v_dance.encoders.state_encoder import _defender_profile
    flyer = _mon("Talonflame", "Flame Body", volatiles={"force_grounded": True})
    assert _defender_profile(flyer)["grounded"] is True
    assert _defender_profile(_mon("Talonflame", "Flame Body"))["grounded"] is False   # Flying ungrounded


def test_e2e_tar_shot_doubles_fire():
    user = _mon("Charizard", "Blaze", moves=["Flamethrower"])
    base = _chan(user, _mon("Garchomp", "Rough Skin"), "Flamethrower")               # Fire vs Dragon/Ground = 1×
    tarred = _chan(user, _mon("Garchomp", "Rough Skin", volatiles={"tar_shot": True}), "Flamethrower")
    assert tarred[2] > base[2]                              # band ~2× higher
    assert tarred[0] > base[0]                              # signed type-eff bumped (+0.5)
