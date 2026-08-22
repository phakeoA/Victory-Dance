"""
Level C / B0d — option B: survival mechanics + non-damage faints (2026-06-30, fork-B "fully fleshed").

Focus Sash / Sturdy (survive a KO from full HP), Endure (survive from any HP), Substitute (absorb the first
hit), Perish Song (perish_norm 1.0 → faint), generic residual chip (Leech Seed / Salt Cure bool → 1/8). These
reduce the gate's genuine_over / genuine_gap. Snapshot-representation-bound (see the design doc).
"""
from __future__ import annotations

from v_dance.encoders import white_box_sim as W


def _vol(**over):
    v = {"has_substitute": False, "perish_norm": 0.0, "residual_damage": False}
    v.update(over)
    return v


def _mon(species, hp=100.0, atk=200, spa=200, spe=100, hp_stat=200, defn=120, item=None, ability=None, vol=None):
    return {
        "species": species, "base_species": species, "hp_pct": hp, "status": None, "is_fainted": False,
        "boosts": {}, "known_item": item, "known_ability": ability, "mega_ability": None, "is_mega": False,
        "volatiles": vol if vol is not None else _vol(), "revealed_moves": [], "times_attacked": 0,
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


def _glass_cannon():
    return _mon("Rampardos", atk=520, hp_stat=200)          # huge attacker → OHKOs a frail full-HP mon


def _frail(item=None, ability=None, hp=100.0, vol=None):
    return _mon("Flutter Mane", hp=hp, hp_stat=120, defn=45, item=item, ability=ability, vol=vol)


def test_focus_sash_survives_lethal_from_full():
    s = _state(_glass_cannon(), None, _frail(item="Focus Sash"), None)
    nxt, log = W.white_box_sim(s, {"our_a": {"kind": "move", "move": "Earthquake", "target": "opp_a"}})
    d = nxt["opp_active"]["opp_a"]
    assert not W.is_fainted(d) and 0 < d["hp_pct"] <= 1.0
    assert any(k == "applied:survive_ko" for k in log)


def test_sturdy_survives_lethal_from_full():
    s = _state(_glass_cannon(), None, _frail(ability="Sturdy"), None)
    d = W.white_box_sim(s, {"our_a": {"kind": "move", "move": "Earthquake", "target": "opp_a"}})[0]["opp_active"]["opp_a"]
    assert not W.is_fainted(d) and d["hp_pct"] > 0


def test_sash_does_not_save_a_second_hit():
    # two attackers both Earthquake the Sash mon: first leaves 1 HP (no longer full), second KOs. The second
    # attacker LEVITATES so it is immune to our_a's Earthquake (an allAdjacent move now self-hits the ally,
    # audit 2026-07-01) and survives to land the second hit.
    s = _state(_glass_cannon(), _mon("Rampardos", atk=520, hp_stat=200, ability="Levitate"),
               _frail(item="Focus Sash"), None)
    d = W.white_box_sim(s, {"our_a": {"kind": "move", "move": "Earthquake", "target": "opp_a"},
                            "our_b": {"kind": "move", "move": "Earthquake", "target": "opp_a"}})[0]
    assert W.is_fainted(d["opp_active"]["opp_a"])


def test_alladjacent_move_hits_own_ally():
    # audit 2026-07-01: Earthquake ('allAdjacent') must damage the user's OWN ally, not just the foes.
    # Our Rampardos EQs the foe; our ally Flutter Mane (Ground-neutral, frail) must take real self-damage.
    ally = _frail(hp=100.0)
    s = _state(_glass_cannon(), ally, _frail(item="Focus Sash"), None)
    nxt = W.white_box_sim(s, {"our_a": {"kind": "move", "move": "Earthquake", "target": "opp_a"}})[0]
    our_b = nxt["our_active"]["our_b"]
    assert our_b["hp_pct"] < 100.0            # the ally took spread self-damage (was untouched before the fix)


def test_endure_survives_from_low_hp():
    # opp_a at 20% uses Endure; our_a OHKO-strength hit → survives at floor.
    s = _state(_glass_cannon(), None, _frail(hp=20.0), None)
    nxt = W.white_box_sim(s, {"our_a": {"kind": "move", "move": "Earthquake", "target": "opp_a"},
                              "opp_a": {"kind": "move", "move": "Endure", "target": "opp_a"}})[0]
    assert not W.is_fainted(nxt["opp_active"]["opp_a"])


def test_substitute_absorbs_first_hit():
    sub = _frail(vol=_vol(has_substitute=True))
    s = _state(_glass_cannon(), None, sub, None)
    nxt, log = W.white_box_sim(s, {"our_a": {"kind": "move", "move": "Earthquake", "target": "opp_a"}})
    d = nxt["opp_active"]["opp_a"]
    assert d["hp_pct"] == 100.0                              # mon untouched (sub ate the hit)
    assert d["volatiles"]["has_substitute"] is False         # sub broke
    assert any(k == "applied:substitute_absorb" for k in log)


def test_perish_faints_in_residuals():
    m = _mon("Gengar", vol=_vol(perish_norm=1.0))
    snap = _state(opp_a=m)
    W.apply_residuals(snap, log=W.UnmodelledLog())
    assert W.is_fainted(snap["opp_active"]["opp_a"])


def test_residual_damage_chips():
    m = _mon("Gengar", hp=10.0, vol=_vol(residual_damage=True))   # 10% HP, 1/8 chip → faints
    snap = _state(opp_a=m)
    log = W.UnmodelledLog()
    W.apply_residuals(snap, log=log)
    assert W.is_fainted(snap["opp_active"]["opp_a"])
    assert any(k == "applied:residual_chip" for k in log)


def test_survival_flips_must_faint_bracket():
    s = _state(_glass_cannon(), None, _frail(item="Focus Sash"), None)
    d = W.white_box_sim(s, {"our_a": {"kind": "move", "move": "Earthquake", "target": "opp_a"}})[0]["opp_active"]["opp_a"]
    assert d["_hp_hi"] > 0 and d["_hp_lo"] > 0 and d["_hp_floor"] > 0   # no shadow faints under Sash
