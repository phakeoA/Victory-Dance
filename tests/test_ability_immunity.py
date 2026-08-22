"""v11 Phase A.1 — a DEFENDER's ability TYPE-IMMUNITY (Levitate/Volt Absorb/Water Absorb/…) must ZERO
the move's damage band AND set the type-eff immune flag.

The correctness bug it fixes: a Ground move vs Levitate (Electric vs Volt Absorb, …) used to encode a
full nonzero damage band + non-immune type-eff, identical to a neutral hit. The shared helper
`_ability_immunizes` keeps the offline + live encoders byte-identical; an attacker with Mold Breaker /
Teravolt / Turboblaze bypasses the immunity.
"""
from __future__ import annotations

from v_dance.encoders.state_encoder import (
    StateEncoder, MOVE_FEATURES, _MOVE_BLOCK_REL, move_slots_for_mon, norm_species,
    _ability_immunizes,
)

OFF_MOVES = _MOVE_BLOCK_REL
OFF_TYPEEFF = 9      # within a move block: signed/immune vs enemy0 at +9/+10
OFF_DAMAGE = 13      # band min/max vs enemy0 at +13/+14


def _mon(species, ability, *, moves=()):
    return {
        "species": species, "base_species": species,
        "hp_pct": 100.0, "seen": True, "is_fainted": False,
        "known_moves": list(moves), "revealed_moves": [],
        "boosts": {}, "status": None, "known_ability": ability,
        "stats_estimate": {"mode": "exact",
                           "stats": {"atk": 150, "spa": 100, "def": 100, "spd": 100, "hp": 200, "spe": 100}},
    }


def _eq_channels(attacker_ability, defender_ability):
    """Encode Garchomp(Earthquake=Ground,phys) vs Snorlax(Normal, so Ground is 1x by typing) and return
    (signed, immune, dmin, dmax) of Earthquake vs enemy0 (opp_a). Snorlax's typing is neutral to Ground,
    so any zeroing is purely the defender ABILITY."""
    our_a = _mon("Garchomp", attacker_ability, moves=["Earthquake"])
    snap = {"our_active": {"our_a": our_a, "our_b": None},
            "opp_active": {"opp_a": _mon("Snorlax", defender_ability), "opp_b": None},
            "our_bench": [], "opp_bench": [], "field": {}, "side_conditions": {}}
    vec = StateEncoder().encode_snapshot(snap, turn=1)
    for m_idx, (mv, _c) in enumerate(move_slots_for_mon(our_a)):
        if norm_species(mv) == "earthquake":
            te = OFF_MOVES + m_idx * MOVE_FEATURES + OFF_TYPEEFF
            dm = OFF_MOVES + m_idx * MOVE_FEATURES + OFF_DAMAGE
            return float(vec[te]), float(vec[te + 1]), float(vec[dm]), float(vec[dm + 1])
    raise AssertionError("Earthquake not found in our_a's move slots")


# ── the helper (unit) ─────────────────────────────────────────────────────────
def test_helper_immunity_matrix():
    assert _ability_immunizes("GROUND", "levitate", None) is True
    assert _ability_immunizes("ELECTRIC", "voltabsorb", None) is True
    assert _ability_immunizes("WATER", "waterabsorb", None) is True
    assert _ability_immunizes("FIRE", "flashfire", None) is True
    assert _ability_immunizes("GRASS", "sapsipper", None) is True
    assert _ability_immunizes("GROUND", "levitate", "moldbreaker") is False     # bypass
    assert _ability_immunizes("FIRE", "levitate", None) is False                # wrong type
    assert _ability_immunizes("GROUND", "purifyingsalt", None) is False         # resist, not immunity
    assert _ability_immunizes("GROUND", None, None) is False


# ── end-to-end through encode_snapshot ────────────────────────────────────────
def test_levitate_zeros_ground_band_and_marks_immune():
    signed, immune, dmin, dmax = _eq_channels("Rough Skin", "Levitate")
    assert immune == 1.0, "type-eff immune flag not set for Ground vs Levitate"
    assert signed == -1.0
    assert dmin == 0.0 and dmax == 0.0, "damage band not zeroed vs an immune ability"


def test_non_immunity_ability_keeps_real_damage():
    # Thick Fat is a resist, NOT a Ground immunity → Ground vs Snorlax stays neutral + nonzero
    signed, immune, dmin, dmax = _eq_channels("Rough Skin", "Thick Fat")
    assert immune == 0.0
    assert dmax > 0.0, "neutral Ground hit should have a nonzero band"


def test_mold_breaker_attacker_bypasses_the_immunity():
    signed, immune, dmin, dmax = _eq_channels("Mold Breaker", "Levitate")
    assert immune == 0.0, "Mold Breaker must ignore the defender's immunity ability"
    assert dmax > 0.0, "Mold Breaker Earthquake should still damage a Levitate mon"
