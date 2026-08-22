"""
Level C / B0c follow-up — mega stat-swap (2026-06-30, fork-B increment 2).

On the mega-decision turn the stepper now swaps the mon to its mega forme (stats via the base-stat delta,
ability → the forme's fixed ability, species → forme for STAB/typing) BEFORE moves resolve, so that turn's
damage + speed use the mega forme. Previously the turn used pre-mega stats (`mega_not_applied`) → a B0d
faint-recall gap (~10k notes). These lock the swap.
"""
from __future__ import annotations

from v_dance.encoders import white_box_sim as W


def _mon(species, hp=100.0, atk=200, spa=120, spe=120, hp_stat=220, defn=120, known_ability=None):
    return {
        "species": species, "base_species": species, "hp_pct": hp, "status": None, "is_fainted": False,
        "boosts": {}, "known_item": None, "known_ability": known_ability, "mega_ability": None,
        "is_mega": False, "volatiles": {}, "revealed_moves": [], "times_attacked": 0,
        "stats_estimate": {"mode": "distribution",
                           "stats": {"hp": hp_stat, "atk": atk, "def": defn, "spa": spa, "spd": defn, "spe": spe}},
    }


def _state(our_a=None, our_b=None, opp_a=None, opp_b=None):
    return {
        "field": {"weather": None, "terrain": None, "trick_room_turns_remaining": 0},
        "side_conditions": {"our_side": {"tailwind_turns_remaining": 0, "screens": {}},
                            "opp_side": {"tailwind_turns_remaining": 0, "screens": {}}},
        "our_active": {"our_a": our_a, "our_b": our_b}, "opp_active": {"opp_a": opp_a, "opp_b": opp_b},
        "our_bench": [], "opp_bench": [],
    }


def test_apply_mega_swaps_stats_species_ability():
    m = _mon("Scizor", atk=200)
    assert W.apply_mega(m) is True
    assert m["species"] == "Scizor-Mega" and m["is_mega"] is True
    assert m["mega_ability"] == "Technician"
    assert m["stats_estimate"]["stats"]["atk"] == 200 + (150 - 130)   # base delta +20
    assert m["stats_estimate"]["stats"]["def"] == 120 + (140 - 100)   # +40


def test_apply_mega_floette_special():
    m = _mon("Floette-Eternal", spa=185)
    assert W.apply_mega(m) is True
    assert m["species"] == "Floette-Mega"
    assert m["stats_estimate"]["stats"]["spa"] == 185 + (155 - 125)   # +30


def test_apply_mega_noop_when_already_mega():
    m = _mon("Scizor"); m["is_mega"] = True
    assert W.apply_mega(m) is False


def test_mega_action_increases_damage_end_to_end():
    # Scizor Bullet Punch (40 BP Steel) vs a Normal defender: mega adds Atk +20 AND Technician (x1.5 on <=60 BP).
    base = _state(_mon("Scizor", atk=200), None, _mon("Snorlax", hp_stat=260, defn=120), None)
    mega = _state(_mon("Scizor", atk=200), None, _mon("Snorlax", hp_stat=260, defn=120), None)
    act = {"kind": "move", "move": "Bullet Punch", "target": "opp_a"}
    hp_base = W.white_box_sim(base, {"our_a": dict(act)})[0]["opp_active"]["opp_a"]["hp_pct"]
    hp_mega = W.white_box_sim(mega, {"our_a": {**act, "mega": True}})[0]["opp_active"]["opp_a"]["hp_pct"]
    assert hp_mega < hp_base, f"mega did not raise damage (base hp={hp_base} mega hp={hp_mega})"


def test_mega_ambiguous_forme_logged():
    # Charizard has Mega-X and Mega-Y → pick first + log ambiguity (never silent).
    log = W.UnmodelledLog()
    m = _mon("Charizard")
    W.apply_mega(m, log=log)
    assert any(k.startswith("mega_ambiguous_forme") for k in log)
