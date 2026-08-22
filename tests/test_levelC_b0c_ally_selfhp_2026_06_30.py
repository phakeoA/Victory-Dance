"""
Level C / B0c follow-ups — ally damage modifiers (inc 3) + self-HP recoil/drain/crash/self-destruct (inc 4).
(2026-06-30, fork-B.)

Inc 3: Helping Hand ×1.5, ally Power Spot/Battery/Steely Spirit, def-ally Friend Guard ×0.75.
Inc 4: data-driven from GenData(9) move fields — recoil (frac of dmg), drain (heal), Mind Blown/Steel Beam
(50% flat), Struggle (25%), HJK crash on miss, Life Orb 10%, self-destruct → faint.
"""
from __future__ import annotations

from v_dance.encoders import white_box_sim as W


def _mon(species, hp=100.0, atk=200, spa=200, spe=120, hp_stat=240, defn=100, item=None, ability=None):
    return {
        "species": species, "base_species": species, "hp_pct": hp, "status": None, "is_fainted": False,
        "boosts": {}, "known_item": item, "known_ability": ability, "mega_ability": None, "is_mega": False,
        "volatiles": {}, "revealed_moves": [], "times_attacked": 0,
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


def _opp_a_hp(state, actions):
    return W.white_box_sim(state, actions)[0]["opp_active"]["opp_a"]["hp_pct"]


def _our_a_hp(state, actions):
    return W.white_box_sim(state, actions)[0]["our_active"]["our_a"]["hp_pct"]


# ── inc 3: ally damage modifiers ──────────────────────────────────────────────
def test_helping_hand_boosts_ally_damage():
    # our_b uses Helping Hand on our_a; our_a's attack should hit harder than without.
    base = _state(_mon("Garchomp"), _mon("Whimsicott"), _mon("Snorlax", hp_stat=260, defn=110), None)
    hh = _state(_mon("Garchomp"), _mon("Whimsicott"), _mon("Snorlax", hp_stat=260, defn=110), None)
    no = _opp_a_hp(base, {"our_a": {"kind": "move", "move": "Earthquake", "target": "opp_a"}})
    yes = _opp_a_hp(hh, {"our_a": {"kind": "move", "move": "Dragon Claw", "target": "opp_a"},
                         "our_b": {"kind": "move", "move": "Helping Hand", "target": "our_a"}})
    # compare each move to itself: re-run Dragon Claw without HH
    base2 = _state(_mon("Garchomp"), _mon("Whimsicott"), _mon("Snorlax", hp_stat=260, defn=110), None)
    plain = _opp_a_hp(base2, {"our_a": {"kind": "move", "move": "Dragon Claw", "target": "opp_a"}})
    assert (100 - yes) > (100 - plain) * 1.4, f"Helping Hand did not boost (plain dmg {100-plain:.1f} hh {100-yes:.1f})"


def test_helping_hand_not_logged_as_gap():
    s = _state(_mon("Garchomp"), _mon("Whimsicott"), _mon("Snorlax", hp_stat=260), None)
    _, log = W.white_box_sim(s, {"our_a": {"kind": "move", "move": "Dragon Claw", "target": "opp_a"},
                                 "our_b": {"kind": "move", "move": "Helping Hand", "target": "our_a"}})
    assert not any(k == "status_move:helpinghand" for k in log)
    assert any(k == "applied:helpinghand" for k in log)


def test_friend_guard_reduces_damage():
    plain = _state(_mon("Garchomp"), None, _mon("Snorlax", hp_stat=260, defn=110), _mon("Clefable"))
    fg = _state(_mon("Garchomp"), None, _mon("Snorlax", hp_stat=260, defn=110),
                _mon("Clefable", ability="Friend Guard"))
    act = {"our_a": {"kind": "move", "move": "Dragon Claw", "target": "opp_a"}}
    assert _opp_a_hp(fg, act) > _opp_a_hp(plain, act), "Friend Guard did not reduce damage"


# ── inc 4: self-HP ────────────────────────────────────────────────────────────
def test_recoil_damages_attacker():
    s = _state(_mon("Talonflame", hp=100.0), None, _mon("Snorlax", hp_stat=260, defn=110), None)
    hp = _our_a_hp(s, {"our_a": {"kind": "move", "move": "Brave Bird", "target": "opp_a"}})
    assert hp < 100.0, f"Brave Bird recoil not applied (attacker hp={hp})"


def test_drain_heals_attacker():
    # attacker at 50%, uses Giga Drain → should heal back some.
    s = _state(_mon("Ferrothorn", hp=50.0, spa=180), None, _mon("Gastrodon", hp_stat=300, defn=90), None)
    hp = _our_a_hp(s, {"our_a": {"kind": "move", "move": "Giga Drain", "target": "opp_a"}})
    assert hp > 50.0, f"Giga Drain did not heal attacker (hp={hp})"


def test_self_destruct_faints_attacker():
    s = _state(_mon("Drifblim", hp=100.0), None, _mon("Snorlax", hp_stat=260), None)
    nxt, log = W.white_box_sim(s, {"our_a": {"kind": "move", "move": "Explosion", "target": "opp_a"}})
    assert nxt["our_active"]["our_a"]["hp_pct"] <= 0, "Explosion did not faint the user"
    assert any(k.startswith("applied:selfdestruct") for k in log)


def test_mind_blown_recoil_flat_50():
    s = _state(_mon("Blacephalon", hp=100.0, spa=220), None, _mon("Snorlax", hp_stat=260, defn=80), None)
    hp = _our_a_hp(s, {"our_a": {"kind": "move", "move": "Mind Blown", "target": "opp_a"}})
    assert hp <= 50.0, f"Mind Blown 50% recoil not applied (hp={hp})"
