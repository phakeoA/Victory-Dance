"""
Level C / B0b+B0c — turn sequencer + per-move damage resolution + the turn-stepper (2026-06-30).

Tests ``order_moves`` (priority/speed/Trick-Room ordering) and ``resolve_move`` / ``white_box_sim``
(expected damage, type immunity, spread, faint-drops-move, input-not-mutated) in white_box_sim.py.
"""
from __future__ import annotations

from v_dance.encoders import white_box_sim as W


def _mon(species, hp=100.0, atk=180, spe=100, status=None, item=None, hp_stat=200, defn=120, boosts=None):
    return {
        "species": species, "base_species": species, "hp_pct": hp, "status": status,
        "is_fainted": False, "boosts": dict(boosts or {}), "known_item": item, "known_ability": None,
        "volatiles": {}, "revealed_moves": [],
        "stats_estimate": {"mode": "distribution",
                           "stats": {"hp": hp_stat, "atk": atk, "def": defn, "spa": atk, "spd": defn, "spe": spe}},
    }


def _state(our_a=None, our_b=None, opp_a=None, opp_b=None, field=None):
    return {
        "field": field or {"weather": None, "terrain": None, "trick_room_turns_remaining": 0},
        "side_conditions": {"our_side": {"tailwind_turns_remaining": 0, "screens": {}},
                            "opp_side": {"tailwind_turns_remaining": 0, "screens": {}}},
        "our_active": {"our_a": our_a, "our_b": our_b},
        "opp_active": {"opp_a": opp_a, "opp_b": opp_b},
        "our_bench": [], "opp_bench": [],
    }


# ── B0b sequencer ─────────────────────────────────────────────────────────────
def test_order_by_speed():
    s = _state(_mon("Garchomp", spe=200), None, _mon("Snorlax", spe=30), None)
    order = W.order_moves(s, {"our_a": {"kind": "move", "move": "Tackle"},
                              "opp_a": {"kind": "move", "move": "Tackle"}})
    assert [k for k, _ in order] == ["our_a", "opp_a"]            # faster first


def test_priority_overrides_speed():
    s = _state(_mon("Garchomp", spe=200), None, _mon("Snorlax", spe=30), None)
    order = W.order_moves(s, {"our_a": {"kind": "move", "move": "Tackle"},
                              "opp_a": {"kind": "move", "move": "Quick Attack"}})
    assert order[0][0] == "opp_a"                                # +1 priority moves first despite slow


def test_trick_room_flips_speed():
    s = _state(_mon("Garchomp", spe=200), None, _mon("Snorlax", spe=30), None,
               field={"weather": None, "trick_room_turns_remaining": 5})
    order = W.order_moves(s, {"our_a": {"kind": "move", "move": "Tackle"},
                              "opp_a": {"kind": "move", "move": "Tackle"}})
    assert order[0][0] == "opp_a"                                # slower first under Trick Room


# ── B0c damage resolution ─────────────────────────────────────────────────────
def test_resolve_move_deals_damage():
    s = _state(_mon("Garchomp", atk=200), None, _mon("Kingambit", hp_stat=180), None)  # Dark/Steel → EQ ×2
    W.resolve_move(s, "our_a", {"kind": "move", "move": "Earthquake", "target": "opp_a"})
    assert s["opp_active"]["opp_a"]["hp_pct"] < 100.0


def test_ground_immune_no_damage():
    s = _state(_mon("Garchomp", atk=200), None, _mon("Charizard"), None)               # Fire/Flying → Ground 0×
    W.resolve_move(s, "our_a", {"kind": "move", "move": "Earthquake", "target": "opp_a"})
    assert s["opp_active"]["opp_a"]["hp_pct"] == 100.0


def test_spread_hits_both_foes():
    s = _state(_mon("Garchomp", atk=200), None, _mon("Kingambit"), _mon("Tyranitar"))
    W.resolve_move(s, "our_a", {"kind": "move", "move": "Rock Slide"})                 # allAdjacentFoes
    assert s["opp_active"]["opp_a"]["hp_pct"] < 100.0 and s["opp_active"]["opp_b"]["hp_pct"] < 100.0


# ── B0c full turn ─────────────────────────────────────────────────────────────
def test_white_box_sim_full_turn_no_mutation():
    s = _state(_mon("Garchomp", atk=220, spe=200), None, _mon("Kingambit", hp_stat=170), None)
    nxt, log = W.white_box_sim(s, {"our_a": {"kind": "move", "move": "Earthquake", "target": "opp_a"},
                                   "opp_a": {"kind": "move", "move": "Tackle", "target": "our_a"}})
    assert nxt is not s and s["opp_active"]["opp_a"]["hp_pct"] == 100.0    # input untouched
    assert nxt["opp_active"]["opp_a"]["hp_pct"] < 100.0                    # opp took Earthquake
    assert nxt["our_active"]["our_a"]["hp_pct"] < 100.0                    # our took Tackle


def test_fainted_mover_drops_its_move():
    s = _state(_mon("Garchomp", atk=320, spe=200), None, _mon("Kingambit", hp=10.0, hp_stat=100), None)
    nxt, log = W.white_box_sim(s, {"our_a": {"kind": "move", "move": "Earthquake", "target": "opp_a"},
                                   "opp_a": {"kind": "move", "move": "Tackle", "target": "our_a"}})
    assert nxt["opp_active"]["opp_a"]["is_fainted"] is True
    assert nxt["our_active"]["our_a"]["hp_pct"] == 100.0                   # opp fainted before acting → move dropped
    assert log["mover_fainted_before_acting"] >= 1


def test_protect_blocks_damage():
    s = _state(_mon("Garchomp", atk=250, spe=200), None, _mon("Kingambit", hp_stat=170), None)
    # opp Protects; our Earthquake should deal 0 to it this turn
    nxt, log = W.white_box_sim(s, {"our_a": {"kind": "move", "move": "Earthquake", "target": "opp_a"},
                                   "opp_a": {"kind": "move", "move": "Protect"}})
    assert nxt["opp_active"]["opp_a"]["hp_pct"] == 100.0 and log["blocked_by_protect"] >= 1


def test_heal_move_recovers():
    hurt = _mon("Garchomp", hp=40.0)
    s = _state(hurt, None, _mon("Kingambit"), None)
    nxt, _ = W.white_box_sim(s, {"our_a": {"kind": "move", "move": "Roost"}})
    assert nxt["our_active"]["our_a"]["hp_pct"] == 90.0          # +50% (capped at 100)


def test_switch_resolves_before_move():
    s = _state(_mon("Garchomp", atk=200, spe=200), None, _mon("Kingambit", hp_stat=180), None)
    s["opp_bench"] = [_mon("Skarmory")]                                    # Steel/Flying → Ground 0×
    # opp switches to Skarmory (Ground-immune); our Earthquake then hits the SWITCHED-IN mon for 0
    nxt, log = W.white_box_sim(s, {"our_a": {"kind": "move", "move": "Earthquake", "target": "opp_a"},
                                   "opp_a": {"kind": "switch", "bench_index": 0}})
    assert nxt["opp_active"]["opp_a"]["species"] == "Skarmory"             # switch happened first
    assert nxt["opp_active"]["opp_a"]["hp_pct"] == 100.0                   # Skarmory is Ground-immune
