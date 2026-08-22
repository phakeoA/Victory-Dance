"""
Level C / B0c follow-up — variable / weight-based base power (2026-06-30, fork-B increment 1).

The stepper now recomputes the REAL base power of variable-BP moves via the shared
``damage_mechanics.variable_base_power`` (Low Kick weight tiers, Eruption HP scaling, Last Respects
fainted-allies). Previously Low Kick returned 0 damage (dex basePower 0) and Eruption/Last Respects used
the static base → systematic faint-recall misses (B0d FN). These lock the fix.
"""
from __future__ import annotations

from v_dance.encoders import white_box_sim as W


def _mon(species, hp=100.0, atk=150, spe=100, hp_stat=220, defn=100, fainted=False, boosts=None):
    return {
        "species": species, "base_species": species, "hp_pct": hp, "status": None,
        "is_fainted": fainted, "boosts": dict(boosts or {}), "known_item": None, "known_ability": None,
        "volatiles": {}, "revealed_moves": [], "times_attacked": 0,
        "stats_estimate": {"mode": "distribution",
                           "stats": {"hp": hp_stat, "atk": atk, "def": defn, "spa": atk, "spd": defn, "spe": spe}},
    }


def _state(our_a=None, our_b=None, opp_a=None, opp_b=None, our_bench=None):
    return {
        "field": {"weather": None, "terrain": None, "trick_room_turns_remaining": 0},
        "side_conditions": {"our_side": {"tailwind_turns_remaining": 0, "screens": {}},
                            "opp_side": {"tailwind_turns_remaining": 0, "screens": {}}},
        "our_active": {"our_a": our_a, "our_b": our_b},
        "opp_active": {"opp_a": opp_a, "opp_b": opp_b},
        "our_bench": our_bench or [], "opp_bench": [],
    }


def _opp_a_hp_after(state, action):
    nxt, _log = W.white_box_sim(state, action)
    return nxt["opp_active"]["opp_a"]["hp_pct"]


def test_low_kick_deals_damage_on_heavy_target():
    # Low Kick (dex basePower 0) vs heavy Snorlax (~460kg → 120 BP), Fighting 2x on Normal — must now hurt.
    s = _state(_mon("Machamp"), None, _mon("Snorlax", hp_stat=260, defn=110), None)
    hp = _opp_a_hp_after(s, {"our_a": {"kind": "move", "move": "Low Kick", "target": "opp_a"}})
    assert hp < 99.0, f"Low Kick did no damage (hp={hp}) — variable BP not applied"


def test_eruption_scales_with_attacker_hp():
    full = _state(_mon("Torkoal", hp=100.0), None, _mon("Garchomp", hp_stat=240, defn=110), None)
    low = _state(_mon("Torkoal", hp=30.0), None, _mon("Garchomp", hp_stat=240, defn=110), None)
    dmg_full = 100.0 - _opp_a_hp_after(full, {"our_a": {"kind": "move", "move": "Eruption", "target": "opp_a"}})
    dmg_low = 100.0 - _opp_a_hp_after(low, {"our_a": {"kind": "move", "move": "Eruption", "target": "opp_a"}})
    assert dmg_full > 0 and dmg_low > 0
    assert dmg_low < dmg_full * 0.6, f"Eruption did not scale with HP (full={dmg_full:.1f} low={dmg_low:.1f})"


def test_last_respects_scales_with_fainted_allies():
    none_faint = _state(_mon("Houndstone"), _mon("Meowscarada"),
                        _mon("Garchomp", hp_stat=260, defn=110), None)
    two_faint = _state(_mon("Houndstone"), _mon("Meowscarada", hp=0.0, fainted=True),
                       _mon("Garchomp", hp_stat=260, defn=110), None,
                       our_bench=[_mon("Kingambit", hp=0.0, fainted=True)])
    act = {"our_a": {"kind": "move", "move": "Last Respects", "target": "opp_a"}}
    dmg0 = 100.0 - _opp_a_hp_after(none_faint, act)
    dmg2 = 100.0 - _opp_a_hp_after(two_faint, act)
    assert dmg2 > dmg0 * 1.8, f"Last Respects did not scale with fainted allies (0={dmg0:.1f} 2={dmg2:.1f})"
