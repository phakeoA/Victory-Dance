"""
Level C — fixes for the B1/audit findings (2026-06-30): live-snapshot enrichment, unseen-switch HP seeding,
consumed-item gates (Leftovers/Focus Sash), and the search trained-head guard.
"""
from __future__ import annotations

from v_dance.encoders import white_box_sim as W
from v_dance.play import search as S


def _mon(species, hp=100.0, item=None, item_consumed=False, ability=None, hp_stat=200):
    return {
        "species": species, "base_species": species, "hp_pct": hp, "status": None, "is_fainted": False,
        "boosts": {}, "known_item": item, "item_consumed": item_consumed, "known_ability": ability,
        "mega_ability": None, "is_mega": False,
        "volatiles": {"has_substitute": False, "perish_norm": 0.0, "residual_damage": False},
        "revealed_moves": [], "times_attacked": 0,
        "stats_estimate": {"mode": "distribution", "stats": {"hp": hp_stat, "atk": 150, "def": 100,
                                                             "spa": 150, "spd": 100, "spe": 100}},
    }


def _state(our_a=None, opp_a=None, opp_bench=None, field=None):
    return {
        "field": field or {"weather": None, "terrain": None, "trick_room_turns_remaining": 0},
        "side_conditions": {"our_side": {"tailwind_turns_remaining": 0, "screens": {}},
                            "opp_side": {"tailwind_turns_remaining": 0, "screens": {}}},
        "our_active": {"our_a": our_a, "our_b": None}, "opp_active": {"opp_a": opp_a, "opp_b": None},
        "our_bench": [], "opp_bench": opp_bench or [],
    }


# ── unseen-switch HP seeding (forward/leaf contradiction fix) ───────────────────
def test_switch_in_seeds_unseen_hp_to_full():
    unseen = _mon("Garchomp"); unseen["hp_pct"] = None      # never-revealed bench mon
    s = _state(_mon("Pikachu"), _mon("Snorlax"), opp_bench=[unseen])
    inc = W.switch_in(s, "opp_a", 0)
    assert inc is not None and inc["hp_pct"] == 100.0 and inc["is_fainted"] is False
    assert not W.is_fainted(s["opp_active"]["opp_a"])       # not treated as fainted


# ── consumed-item gates ─────────────────────────────────────────────────────────
def test_survives_ko_respects_consumed_sash():
    live = _mon("Flutter Mane", item="Focus Sash")
    dead = _mon("Flutter Mane", item="Focus Sash", item_consumed=True)
    assert W._survives_ko(live, "opp_a", frozenset()) is True
    assert W._survives_ko(dead, "opp_a", frozenset()) is False       # popped Sash no longer protects


def test_leftovers_heal_respects_consumed():
    held = _state(our_a=_mon("Snorlax", hp=50.0, item="Leftovers"))
    gone = _state(our_a=_mon("Snorlax", hp=50.0, item="Leftovers", item_consumed=True))
    W.apply_residuals(held, log=W.UnmodelledLog())
    W.apply_residuals(gone, log=W.UnmodelledLog())
    assert held["our_active"]["our_a"]["hp_pct"] > 50.0              # held Leftovers heals
    assert gone["our_active"]["our_a"]["hp_pct"] == 50.0            # consumed/Knocked-Off does not


# ── live-snapshot enrichment (the critical A/B-blocking fix) ─────────────────────
def test_enrich_snapshot_adds_stats_and_belief():
    from v_dance.parser.belief_state import BeliefState
    belief = BeliefState()
    raw = {"species": "Incineroar", "base_species": "Incineroar", "hp_pct": 100.0, "is_fainted": False,
           "revealed_moves": [], "known_item": None, "known_ability": None}
    snap = {"our_active": {"our_a": raw, "our_b": None}, "opp_active": {"opp_a": None, "opp_b": None},
            "our_bench": [], "opp_bench": []}
    S.enrich_snapshot(snap, belief)
    est = raw.get("stats_estimate") or {}
    assert est.get("mode") == "distribution" and (est.get("stats") or {}).get("atk")   # stats now present
    assert (raw.get("belief") or {}).get("spreads")                                      # belief block attached


def test_enrich_snapshot_noop_without_belief():
    raw = {"species": "Incineroar", "base_species": "Incineroar", "hp_pct": 100.0}
    snap = {"our_active": {"our_a": raw}, "opp_active": {}, "our_bench": [], "opp_bench": []}
    S.enrich_snapshot(snap, None)            # no belief → no-op, no crash
    assert "stats_estimate" not in raw
