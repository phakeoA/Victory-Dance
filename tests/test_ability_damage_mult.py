"""v11 Phase A.2 — ability DAMAGE MULTIPLIERS (offensive boosts + defensive resists) folded into the
B1.2 damage band. Verifies every multiplier value/condition against the spec (adversarially verified vs
abilities.ts in scratch/v11_a2_ability_damage_mult_synthesis.md), the Mold-Breaker breakable rule, the
two corrected decisions (steelyspirit INCLUDE; shadowshield/prismarmor NOT mold-breakable), and the
GenData-sourced move props that keep offline↔live byte-parity.
"""
from __future__ import annotations

import math
import pytest

from v_dance.encoders.state_encoder import (
    StateEncoder, MOVE_FEATURES, _MOVE_BLOCK_REL, move_slots_for_mon, norm_species,
    _ability_damage_mult, _move_damage_props, _M_1_2, _M_1_3,
)

OFF_MOVES = _MOVE_BLOCK_REL
OFF_DAMAGE = 13      # within a move block: band min/max vs enemy0 at +13/+14


def _h(move_id, att=None, dfn=None, *, mt="GROUND", stab=False, phys=True,
       tmult=1.0, hp=0.5, burned=False, statused=False):
    """Thin wrapper over _ability_damage_mult with the common test defaults."""
    return _ability_damage_mult(move_id, att, dfn, eff_move_type=mt, is_stab=stab,
                                is_physical=phys, type_mult=tmult, hp_frac=hp,
                                att_burned=burned, att_statused=statused)


def _close(a, b):
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


# ── early-out / no relevant ability ──────────────────────────────────────────
def test_no_relevant_ability_is_identity():
    assert _h("earthquake", "roughskin", "snowwarning") == 1.0
    assert _h("earthquake", None, None) == 1.0


# ── OFFENSIVE value matrix ───────────────────────────────────────────────────
def test_adaptability_only_on_stab():
    assert _close(_h("earthquake", "adaptability", stab=True), 2.0 / 1.5)
    assert _h("earthquake", "adaptability", stab=False) == 1.0


def test_technician_bp_threshold():
    assert _move_damage_props("machpunch")["base_power"] == 40
    assert _move_damage_props("bulldoze")["base_power"] == 60     # inclusive boundary
    assert _move_damage_props("earthquake")["base_power"] == 100
    assert _h("machpunch", "technician") == 1.5
    assert _h("bulldoze", "technician") == 1.5                    # BP == 60 still boosts (<=)
    assert _h("earthquake", "technician") == 1.0                  # BP 100


def test_toughclaws_contact():
    assert _close(_h("closecombat", "toughclaws"), _M_1_3)        # contact
    assert _h("earthquake", "toughclaws") == 1.0                  # non-contact


def test_tintedlens_resisted_only():
    assert _h("earthquake", "tintedlens", tmult=0.5) == 2.0       # resisted
    assert _h("earthquake", "tintedlens", tmult=0.25) == 2.0
    assert _h("earthquake", "tintedlens", tmult=1.0) == 1.0       # neutral
    assert _h("earthquake", "tintedlens", tmult=2.0) == 1.0       # super
    assert _h("earthquake", "tintedlens", tmult=0.0) == 1.0       # immune ≠ resisted


def test_huge_pure_power_physical_only():
    assert _h("earthquake", "hugepower", phys=True) == 2.0
    assert _h("earthquake", "hugepower", phys=False) == 1.0
    assert _h("earthquake", "purepower", phys=True) == 2.0


def test_guts_status_and_burn_cancel():
    # net x1.5: unburned status
    assert _h("earthquake", "guts", phys=True, statused=True, burned=False) == 1.5
    # burned: helper returns x3.0 so the external burn x0.5 cancels to net x1.5
    assert _h("earthquake", "guts", phys=True, statused=True, burned=True) == 3.0
    assert _h("earthquake", "guts", phys=True, statused=False) == 1.0   # no status
    assert _h("earthquake", "guts", phys=False, statused=True) == 1.0   # special


def test_hustle_physical():
    assert _h("earthquake", "hustle", phys=True) == 1.5
    assert _h("earthquake", "hustle", phys=False) == 1.0


def test_ironfist_strongjaw_sharpness():
    assert _close(_h("icepunch", "ironfist"), _M_1_2)             # punch
    assert _h("earthquake", "ironfist") == 1.0
    assert _h("crunch", "strongjaw") == 1.5                       # bite
    assert _h("leafblade", "sharpness") == 1.5                    # slicing


def test_reckless_recoil_and_crash():
    assert _close(_h("flareblitz", "reckless"), _M_1_2)           # recoil
    assert _close(_h("highjumpkick", "reckless"), _M_1_2)         # crash
    assert _close(_h("jumpkick", "reckless"), _M_1_2)
    assert _h("earthquake", "reckless") == 1.0


def test_punkrock_offense_sound():
    assert _close(_h("boomburst", "punkrock", phys=False), _M_1_3)
    assert _h("earthquake", "punkrock") == 1.0


def test_type_boost_offense():
    assert _h("x", "rockypayload", mt="ROCK") == 1.5
    assert _h("x", "rockypayload", mt="GROUND") == 1.0
    assert _close(_h("x", "transistor", mt="ELECTRIC"), _M_1_3)  # Gen-9 value, not legacy 1.5
    assert _h("x", "transistor", mt="ELECTRIC") != 1.5
    assert _h("x", "dragonsmaw", mt="DRAGON") == 1.5
    assert _h("x", "steelworker", mt="STEEL") == 1.5


def test_steelyspirit_is_included_like_steelworker():
    # CORRECTED decision: onAllyBasePower dispatches over alliesAndSelf() (incl. self) → self-applies.
    assert _h("x", "steelyspirit", mt="STEEL") == 1.5
    assert _h("x", "steelyspirit", mt="GROUND") == 1.0


def test_ate_boost_raw_normal_and_exclusions():
    # hypervoice is a raw-Normal move → galvanize boosts it x1.2 (effective type now Electric).
    assert _close(_h("hypervoice", "galvanize", mt="ELECTRIC"), _M_1_2)
    assert _close(_h("hypervoice", "pixilate", mt="FAIRY"), _M_1_2)
    # weatherball is raw-Normal but in the noModifyType exclusion list → no boost.
    assert _h("weatherball", "galvanize", mt="NORMAL") == 1.0
    # a non-Normal raw move gets no -ate boost.
    assert _h("earthquake", "galvanize", mt="GROUND") == 1.0


def test_sheerforce_secondary():
    assert _move_damage_props("icepunch")["has_secondaries"] is True
    assert _close(_h("icepunch", "sheerforce"), _M_1_3)
    assert _h("earthquake", "sheerforce") == 1.0                  # no secondary


# ── DEFENSIVE value matrix ───────────────────────────────────────────────────
def test_multiscale_shadowshield_full_hp():
    assert _h("earthquake", None, "multiscale", hp=1.0) == 0.5
    assert _h("earthquake", None, "multiscale", hp=0.9) == 1.0
    assert _h("earthquake", None, "shadowshield", hp=1.0) == 0.5


def test_thickfat_heatproof_type():
    assert _h("x", None, "thickfat", mt="FIRE") == 0.5
    assert _h("x", None, "thickfat", mt="ICE") == 0.5
    assert _h("x", None, "thickfat", mt="WATER") == 1.0
    assert _h("x", None, "heatproof", mt="FIRE") == 0.5
    assert _h("x", None, "heatproof", mt="ICE") == 1.0


def test_filter_family_super_only():
    for ab in ("filter", "solidrock", "prismarmor"):
        assert _h("x", None, ab, tmult=2.0) == 0.75
        assert _h("x", None, ab, tmult=1.0) == 1.0
        assert _h("x", None, ab, tmult=0.5) == 1.0


def test_fluffy_fire_and_contact_stack():
    assert _h("flamethrower", None, "fluffy", mt="FIRE") == 2.0           # Fire non-contact
    assert _h("closecombat", None, "fluffy", mt="FIGHTING") == 0.5        # non-Fire contact
    assert _close(_h("flareblitz", None, "fluffy", mt="FIRE"), 1.0)       # Fire + contact → x1.0
    assert _h("earthquake", None, "fluffy", mt="GROUND") == 1.0


def test_icescales_furcoat_category():
    assert _h("x", None, "icescales", phys=False) == 0.5
    assert _h("x", None, "icescales", phys=True) == 1.0
    assert _h("x", None, "furcoat", phys=True) == 0.5
    assert _h("x", None, "furcoat", phys=False) == 1.0


def test_wonderguard_zeroes_non_super():
    assert _h("x", None, "wonderguard", tmult=1.0) == 0.0     # neutral → no damage
    assert _h("x", None, "wonderguard", tmult=0.5) == 0.0     # resisted → no damage
    assert _h("x", None, "wonderguard", tmult=2.0) == 1.0     # super → lands


# ── MOLD BREAKER (per-ability breakable rule) ────────────────────────────────
def test_mold_breaker_ignores_breakable_defenses():
    for mb in ("moldbreaker", "teravolt", "turboblaze"):
        assert _h("earthquake", mb, "multiscale", hp=1.0) == 1.0      # breakable → ignored
        assert _h("x", mb, "filter", tmult=2.0) == 1.0
        assert _h("x", mb, "wonderguard", tmult=1.0) == 1.0          # bypassed → not zeroed


def test_mold_breaker_does_NOT_ignore_unbreakable_defenses():
    # shadowshield / prismarmor have flags:{} → Mold Breaker must NOT suppress them.
    assert _h("earthquake", "moldbreaker", "shadowshield", hp=1.0) == 0.5
    assert _h("x", "moldbreaker", "prismarmor", tmult=2.0) == 0.75
    assert _h("x", "teravolt", "prismarmor", tmult=2.0) == 0.75


# ── GenData-sourced move props (the offline↔live parity fix) ──────────────────
def test_move_props_match_gendata_truth():
    # firefang: stale data/moves.json had no secondary; GenData has secondaries → True (the parity fix).
    assert _move_damage_props("firefang")["has_secondaries"] is True
    assert _move_damage_props("thunderfang")["has_secondaries"] is True
    # diamondstorm: stale moves.json had secondary True; GenData = self-boost, not a secondary → False.
    assert _move_damage_props("diamondstorm")["has_secondaries"] is False
    # crash / recoil folding
    assert _move_damage_props("highjumpkick")["has_recoil_or_crash"] is True
    assert _move_damage_props("supercellslam")["has_recoil_or_crash"] is True   # missing offline `crash`
    assert _move_damage_props("flareblitz")["has_recoil_or_crash"] is True
    assert _move_damage_props("struggle")["has_recoil_or_crash"] is True        # struggleRecoil
    assert _move_damage_props("earthquake")["has_recoil_or_crash"] is False
    # flags + raw type
    p = _move_damage_props("icepunch")
    assert p["is_punch"] and p["is_contact"]
    assert _move_damage_props("hypervoice")["raw_is_normal"] is True
    assert _move_damage_props("earthquake")["raw_is_normal"] is False


# ── end-to-end wiring through encode_snapshot (offline) ───────────────────────
def _mon(species, ability, *, moves=(), hp_pct=100.0, status=None, stats=None):
    return {
        "species": species, "base_species": species,
        "hp_pct": hp_pct, "seen": True, "is_fainted": False,
        "known_moves": list(moves), "revealed_moves": [],
        "boosts": {}, "status": status, "known_ability": ability,
        "stats_estimate": {"mode": "exact",
                           "stats": stats or {"atk": 150, "spa": 100, "def": 100,
                                              "spd": 100, "hp": 200, "spe": 100}},
    }


# A high-bulk defender keeps the damage band well under the [0,1] clamp so multiplier RATIOS are exact.
_TANK = {"atk": 100, "spa": 100, "def": 250, "spd": 250, "hp": 600, "spe": 50}


def _band(attacker, defender, move):
    """(dmin, dmax) of `move` (by name) used by `attacker` vs enemy0 (`defender`), via encode_snapshot."""
    snap = {"our_active": {"our_a": attacker, "our_b": None},
            "opp_active": {"opp_a": defender, "opp_b": None},
            "our_bench": [], "opp_bench": [], "field": {}, "side_conditions": {}}
    vec = StateEncoder().encode_snapshot(snap, turn=1)
    for m_idx, (mv, _c) in enumerate(move_slots_for_mon(attacker)):
        if norm_species(mv) == norm_species(move):
            dm = OFF_MOVES + m_idx * MOVE_FEATURES + OFF_DAMAGE
            return float(vec[dm]), float(vec[dm + 1])
    raise AssertionError(f"{move} not in attacker move slots")


def _close_band(a, b):
    # the band is stored float32, so compare ratios with a float32-grade tolerance.
    return math.isclose(a, b, rel_tol=1e-4, abs_tol=1e-6)


def test_e2e_offensive_boost_raises_band():
    # Iron Head (Steel) vs a high-bulk Normal defender (Steel neutral). Steelworker raises the band x1.5.
    base = _band(_mon("Metagross", "Clear Body", moves=["Iron Head"]),
                 _mon("Snorlax", "Immunity", stats=_TANK), "Iron Head")
    boosted = _band(_mon("Metagross", "Steelworker", moves=["Iron Head"]),
                    _mon("Snorlax", "Immunity", stats=_TANK), "Iron Head")
    assert base[1] > 0 and _close_band(boosted[1], base[1] * 1.5)


def test_e2e_guts_burn_is_cancelled():
    # A burned Guts attacker and a paralyzed Guts attacker both have net ×1.5 (Guts ignores the burn ½),
    # so their physical bands must be EQUAL — proving Guts cancels the burn.
    burned = _band(_mon("Conkeldurr", "Guts", moves=["Drain Punch"], status="brn"),
                   _mon("Snorlax", "Immunity", stats=_TANK), "Drain Punch")
    para = _band(_mon("Conkeldurr", "Guts", moves=["Drain Punch"], status="par"),
                 _mon("Snorlax", "Immunity", stats=_TANK), "Drain Punch")
    assert burned[1] > 0 and _close_band(burned[1], para[1])
    # For a NON-Guts attacker (Drain Punch has no secondary → Sheer Force is a no-op), burn DOES apply:
    # burned is net ×0.5 of the paralyzed (no-burn) case.
    burned_noguts = _band(_mon("Conkeldurr", "Sheer Force", moves=["Drain Punch"], status="brn"),
                          _mon("Snorlax", "Immunity", stats=_TANK), "Drain Punch")
    para_noguts = _band(_mon("Conkeldurr", "Sheer Force", moves=["Drain Punch"], status="par"),
                        _mon("Snorlax", "Immunity", stats=_TANK), "Drain Punch")
    assert _close_band(burned_noguts[1], para_noguts[1] * 0.5)


def test_e2e_wonderguard_zeroes_neutral_move():
    dmin, dmax = _band(_mon("Garchomp", "Rough Skin", moves=["Earthquake"]),
                       _mon("Shedinja", "Wonder Guard"), "Earthquake")  # Ground neutral on Shedinja typing? see note
    # Shedinja is Bug/Ghost: Ground is 1x (neutral) → Wonder Guard zeroes it.
    assert dmin == 0.0 and dmax == 0.0
