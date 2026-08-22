"""v11 Klutz + Embargo: per-mon item-effect suppression.

Klutz (ability) and Embargo (5-turn volatile) both zero the holder's item effects
(Showdown Pokemon.ignoringItem). The shared ``_item_active`` gate now returns None under
``klutz`` / ``embargo`` (in addition to the field-wide Magic Room ``suppressed``), so every
downstream item read (damage band, speed, grounding, item tags) sees no item. Klutz suppresses
every item EXCEPT the ignoreKlutz set (Ability Shield / Macho Brace / Power items) — none of which
is an encoded effect, so the encoder treats Klutz as full item suppression. Both encoders compute
the per-mon klutz/embargo flags identically → byte-parity. 0 in-meta (forward-compat).
"""
from __future__ import annotations

from v_dance.encoders.state_encoder import (
    StateEncoder, MOVE_FEATURES, _MOVE_BLOCK_REL, move_slots_for_mon, norm_species, _item_active,
)
from v_dance.parser.vod_parser.battle_models import volatile_flags


# ════════════════════════════ unit: the gate ════════════════════════════
def test_item_active_gate():
    assert _item_active("lifeorb", False) == "lifeorb"            # active
    assert _item_active("lifeorb", True) is None                 # Magic Room
    assert _item_active("lifeorb", False, klutz=True) is None    # Klutz
    assert _item_active("lifeorb", False, embargo=True) is None  # Embargo
    assert _item_active("lifeorb", False, klutz=False, embargo=False) == "lifeorb"


def test_volatile_flags_embargo_key():
    assert volatile_flags({"embargo"})["embargo"] is True
    assert volatile_flags({"leechseed"})["embargo"] is False


# ════════════════════════════ e2e: item effect suppressed ════════════════════════════
def _mon(species, ability, *, item=None, moves=(), volatiles=None):
    return {"species": species, "base_species": species, "hp_pct": 100.0, "seen": True, "is_fainted": False,
            "known_moves": list(moves), "revealed_moves": [], "boosts": {}, "status": None,
            "known_ability": ability, "known_item": item, "volatiles": volatiles or {},
            "stats_estimate": {"mode": "exact",
                               "stats": {"atk": 150, "spa": 100, "def": 100, "spd": 100, "hp": 300, "spe": 100}}}


def _damage_block_sum(att, move, opp_a):
    """Sum of the 4 damage channels (min/max vs each enemy) for `move` — drops when the item is suppressed."""
    snap = {"our_active": {"our_a": att, "our_b": None},
            "opp_active": {"opp_a": opp_a, "opp_b": None},
            "our_bench": [], "opp_bench": [], "field": {}, "side_conditions": {}}
    vec = StateEncoder().encode_snapshot(snap, turn=1)
    for m_idx, (mv, _c) in enumerate(move_slots_for_mon(att)):
        if norm_species(mv) == norm_species(move):
            b = _MOVE_BLOCK_REL + m_idx * MOVE_FEATURES
            return sum(float(vec[b + off]) for off in (13, 14, 15, 16))   # damage [min0,max0,min1,max1]
    raise AssertionError(move)


def test_klutz_suppresses_life_orb_damage():
    opp = _mon("Blissey", "Natural Cure")               # grounded, Normal (Earthquake neutral, not immune)
    holds = _mon("Garchomp", "Sand Veil", item="Life Orb", moves=["Earthquake"])
    klutzed = _mon("Garchomp", "Klutz", item="Life Orb", moves=["Earthquake"])
    d_on = _damage_block_sum(holds, "Earthquake", opp)
    d_klutz = _damage_block_sum(klutzed, "Earthquake", opp)
    assert d_on > 0.0
    assert d_klutz < d_on            # Klutz zeroes Life Orb ×1.3 → lower damage


def test_embargo_suppresses_life_orb_damage():
    opp = _mon("Blissey", "Natural Cure")
    holds = _mon("Garchomp", "Sand Veil", item="Life Orb", moves=["Earthquake"])
    embargoed = _mon("Garchomp", "Sand Veil", item="Life Orb", moves=["Earthquake"],
                     volatiles={"embargo": True})
    d_on = _damage_block_sum(holds, "Earthquake", opp)
    d_emb = _damage_block_sum(embargoed, "Earthquake", opp)
    assert d_on > 0.0
    assert d_emb < d_on              # Embargo volatile zeroes Life Orb → lower damage


def test_klutz_equals_embargo_suppression():
    # Both suppression paths must produce the SAME encoded damage (item fully inactive either way).
    opp = _mon("Blissey", "Natural Cure")
    klutzed = _mon("Garchomp", "Sand Veil", item="Life Orb", moves=["Earthquake"],
                   volatiles={"embargo": True})
    klutz_ab = _mon("Garchomp", "Klutz", item="Life Orb", moves=["Earthquake"])
    assert abs(_damage_block_sum(klutzed, "Earthquake", opp)
               - _damage_block_sum(klutz_ab, "Earthquake", opp)) < 1e-9
