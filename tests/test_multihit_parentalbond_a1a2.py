"""v11 A1 (multi-hit damage band) + A2 (Parental Bond) — correctness-bug fixes from the pre-Phase-D audit.

A1: `_damage_band` applied base power ONCE; a 2-5 multihit move read ~1/N of true damage (only a `/5`
intrinsic channel existed). Now the band scales by hit count (min-hits × low roll, max-hits × high), with
Skill Link (→ variable forced to max) and Loaded Dice ([2,5] → 4-5) folded in. A2: Mega Kangaskhan's Parental
Bond (Gen-9 2nd strike ×0.25 → effective ×1.25) on single-target non-multihit moves. Multihit-ineligibility
keeps A1 and A2 disjoint. Both value-only (no layout). Audit: scratch/v11_pre_phaseD_gap_audit_synthesis.md.
"""
from __future__ import annotations

from v_dance.encoders.state_encoder import (
    StateEncoder, MOVE_FEATURES, _MOVE_BLOCK_REL, move_slots_for_mon, norm_species,
    _move_hit_range, _ability_damage_mult, _DMG_BOOST_AB,
)


# ════════════════════════════ A1 unit: _move_hit_range ════════════════════════════
def test_move_hit_range():
    assert _move_hit_range("bulletseed") == (2, 5)     # variable
    assert _move_hit_range("rockblast") == (2, 5)
    assert _move_hit_range("dragondarts") == (2, 2)    # fixed 2
    assert _move_hit_range("tripleaxel") == (3, 3)     # fixed 3
    assert _move_hit_range("earthquake") == (1, 1)     # single hit
    assert _move_hit_range(None) == (1, 1)


# ════════════════════════════ A2 unit: Parental Bond ════════════════════════════
def _pb(move_id, is_spread):
    return _ability_damage_mult(move_id, "parentalbond", None, eff_move_type="NORMAL", is_stab=False,
                                is_physical=True, type_mult=1.0, hp_frac=1.0, att_burned=False,
                                att_statused=False, is_spread=is_spread)


def test_parentalbond_in_boost_set():
    assert "parentalbond" in _DMG_BOOST_AB


def test_parentalbond_single_target():
    assert _pb("return", False) == 1.25                # single-target, single-hit → ×1.25


def test_parentalbond_spread_ineligible():
    assert _pb("earthquake", True) == 1.0              # spread move → no Parental Bond


def test_parentalbond_multihit_ineligible():
    assert _pb("bulletseed", False) == 1.0             # multihit → no Parental Bond (disjoint from A1)


# ════════════════════════════ e2e through the band ════════════════════════════
def _mon(species, ability, *, moves=(), item=None):
    m = {"species": species, "base_species": species, "hp_pct": 100.0, "seen": True, "is_fainted": False,
         "known_moves": list(moves), "revealed_moves": [], "boosts": {}, "status": None,
         "known_ability": ability, "volatiles": {},
         "stats_estimate": {"mode": "exact",
                            "stats": {"atk": 130, "spa": 100, "def": 100, "spd": 100, "hp": 260, "spe": 80}}}
    if item is not None:
        m["known_item"] = item
    return m


def _band(att, move):
    # vs a high-bulk defender (Blissey) so the band stays under the [0,1] clamp → ratios meaningful
    snap = {"our_active": {"our_a": att, "our_b": None},
            "opp_active": {"opp_a": _mon("Blissey", "Natural Cure"), "opp_b": None},
            "our_bench": [], "opp_bench": [], "field": {}, "side_conditions": {}}
    vec = StateEncoder().encode_snapshot(snap, turn=1)
    for m_idx, (mv, _c) in enumerate(move_slots_for_mon(att)):
        if norm_species(mv) == norm_species(move):
            b = _MOVE_BLOCK_REL + m_idx * MOVE_FEATURES
            return float(vec[b + 13]), float(vec[b + 13 + 1])      # (dmin, dmax)
    raise AssertionError(move)


def test_a1_multihit_band_scales():
    dmin, dmax = _band(_mon("Cinccino", "Cute Charm", moves=["Bullet Seed"]), "Bullet Seed")
    single = _band(_mon("Cinccino", "Cute Charm", moves=["Tackle"]), "Tackle")   # 1-hit reference
    # Bullet Seed BP 25 ×2..5 vs Tackle BP 40 ×1: the multihit max is far above a single 25-BP hit.
    assert dmax > dmin > 0.0
    # max/min ratio ≈ (5 hits × 1.0) / (2 hits × 0.85) = 2.94
    assert abs((dmax / dmin) - (5 / (2 * 0.85))) < 0.05


def test_a1_skill_link_forces_max():
    plain = _band(_mon("Cinccino", "Cute Charm", moves=["Bullet Seed"]), "Bullet Seed")
    sl = _band(_mon("Cinccino", "Skill Link", moves=["Bullet Seed"]), "Bullet Seed")
    assert sl[1] == plain[1]                            # same max (5 hits)
    assert sl[0] > plain[0]                             # higher min (5 hits vs 2)
    assert abs((sl[0] / sl[1]) - 0.85) < 1e-3          # min/max == the 0.85 roll (n_min==n_max==5)


def test_a1_loaded_dice_min_four():
    plain = _band(_mon("Cinccino", "Cute Charm", moves=["Bullet Seed"]), "Bullet Seed")
    ld = _band(_mon("Cinccino", "Cute Charm", moves=["Bullet Seed"], item="Loaded Dice"), "Bullet Seed")
    assert ld[1] == plain[1]                            # max unchanged (5 hits)
    assert abs(ld[0] / plain[0] - 4 / 2) < 0.02        # min 4 hits vs 2 hits → 2×


def test_a1_single_hit_unchanged():
    # a 1-hit move's band is byte-identical to pre-A1 (hits_min=hits_max=1 default).
    dmin, dmax = _band(_mon("Garchomp", "Rough Skin", moves=["Earthquake"]), "Earthquake")
    assert 0.0 < dmin < dmax
    assert abs(dmin / dmax - 0.85) < 1e-3              # just the 0.85-1.0 roll, no hit multiplier


def test_a2_parental_bond_e2e():
    plain = _band(_mon("Kangaskhan", "Scrappy", moves=["Return"]), "Return")
    pbond = _band(_mon("Kangaskhan", "Parental Bond", moves=["Return"]), "Return")
    assert abs(pbond[1] / plain[1] - 1.25) < 1e-3      # ×1.25 effective
