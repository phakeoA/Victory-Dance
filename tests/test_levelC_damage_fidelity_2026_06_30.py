"""
Level C — damage-fidelity audit fixes (2026-06-30): Body Press / Foul Play stat overrides, Psyshock-family
hits-Def, Tar Shot, fainted_allies (Supreme Overlord), Life-Orb-before-drain ordering, double-switch bench.
"""
from __future__ import annotations

from v_dance.encoders import white_box_sim as W


def _mon(species, hp=100.0, atk=120, spa=120, defn=120, spd=120, spe=100, hp_stat=220,
         item=None, ability=None, vol=None, fainted=False):
    return {
        "species": species, "base_species": species, "hp_pct": hp, "status": None, "is_fainted": fainted,
        "boosts": {}, "known_item": item, "item_consumed": False, "known_ability": ability, "mega_ability": None,
        "is_mega": False, "volatiles": vol or {"has_substitute": False, "perish_norm": 0.0, "residual_damage": False,
                                               "tar_shot": False},
        "revealed_moves": [], "times_attacked": 0,
        "stats_estimate": {"mode": "distribution",
                           "stats": {"hp": hp_stat, "atk": atk, "def": defn, "spa": spa, "spd": spd, "spe": spe}},
    }


def _state(our_a=None, our_b=None, opp_a=None, opp_b=None, our_bench=None, field=None):
    return {
        "field": field or {"weather": None, "terrain": None, "trick_room_turns_remaining": 0},
        "side_conditions": {"our_side": {"tailwind_turns_remaining": 0, "screens": {}},
                            "opp_side": {"tailwind_turns_remaining": 0, "screens": {}}},
        "our_active": {"our_a": our_a, "our_b": our_b}, "opp_active": {"opp_a": opp_a, "opp_b": opp_b},
        "our_bench": our_bench or [], "opp_bench": [],
    }


def _dmg(state, action_move, target="opp_a"):
    nxt, _ = W.white_box_sim(state, {"our_a": {"kind": "move", "move": action_move, "target": target}})
    return 100.0 - nxt["opp_active"]["opp_a"]["hp_pct"]


def test_body_press_scales_with_attacker_def():
    lo = _state(_mon("Aggron", atk=80, defn=80), None, _mon("Blissey", hp_stat=300, defn=60), None)
    hi = _state(_mon("Aggron", atk=80, defn=300), None, _mon("Blissey", hp_stat=300, defn=60), None)
    assert _dmg(hi, "Body Press") > _dmg(lo, "Body Press") * 1.5    # uses DEF, not Atk


def test_foul_play_scales_with_target_atk():
    weak = _state(_mon("Grimmsnarl"), None, _mon("Chansey", hp_stat=300, atk=40, defn=80), None)
    strong = _state(_mon("Grimmsnarl"), None, _mon("Chansey", hp_stat=300, atk=300, defn=80), None)
    assert _dmg(strong, "Foul Play") > _dmg(weak, "Foul Play") * 1.5   # uses the TARGET's Atk


def test_psyshock_hits_physical_def():
    # target with high SpD, low Def → Psyshock (hits Def) should out-damage a normal special move (hits SpD)
    s1 = _state(_mon("Hatterene", spa=200), None, _mon("Snorlax", hp_stat=300, defn=40, spd=240), None)
    s2 = _state(_mon("Hatterene", spa=200), None, _mon("Snorlax", hp_stat=300, defn=40, spd=240), None)
    assert _dmg(s1, "Psyshock") > _dmg(s2, "Psychic") * 1.5


def test_tar_shot_doubles_fire():
    # bulky Fire-NEUTRAL defender so the base hit isn't already a KO (then Tar Shot's ×2 is visible)
    plain = _state(_mon("Charizard", spa=110), None, _mon("Garchomp", hp_stat=400, spd=260), None)
    tar = _state(_mon("Charizard", spa=110), None,
                 _mon("Garchomp", hp_stat=400, spd=260,
                      vol={"has_substitute": False, "perish_norm": 0.0, "residual_damage": False, "tar_shot": True}), None)
    assert _dmg(tar, "Flamethrower") > _dmg(plain, "Flamethrower") * 1.5


def test_supreme_overlord_scales_with_fainted_allies():
    none_f = _state(_mon("Kingambit", ability="Supreme Overlord"), _mon("Incineroar"),
                    _mon("Garchomp", hp_stat=260, defn=110), None)
    two_f = _state(_mon("Kingambit", ability="Supreme Overlord"),
                   _mon("Incineroar", hp=0.0, fainted=True),
                   _mon("Garchomp", hp_stat=260, defn=110), None,
                   our_bench=[_mon("Amoonguss", hp=0.0, fainted=True)])
    assert _dmg(two_f, "Kowtow Cleave") > _dmg(none_f, "Kowtow Cleave")


def test_life_orb_recoil_after_drain_keeps_attacker_alive():
    # attacker at 10% with Life Orb uses Giga Drain: drain heals BEFORE the 10% LO recoil → must survive.
    s = _state(_mon("Ferrothorn", hp=10.0, spa=170, item="Life Orb"), None,
               _mon("Gastrodon", hp_stat=300, defn=90), None)
    nxt, _ = W.white_box_sim(s, {"our_a": {"kind": "move", "move": "Giga Drain", "target": "opp_a"}})
    assert not W.is_fainted(nxt["our_active"]["our_a"])


def test_type_boost_item_raises_matching_type():
    plain = _state(_mon("Charizard", spa=140), None, _mon("Garchomp", hp_stat=400, spd=260), None)
    boost = _state(_mon("Charizard", spa=140, item="Charcoal"), None, _mon("Garchomp", hp_stat=400, spd=260), None)
    d_plain, d_boost = _dmg(plain, "Flamethrower"), _dmg(boost, "Flamethrower")
    assert d_boost > d_plain and d_boost < d_plain * 1.4    # Charcoal ≈ ×1.2 on a Fire move


def test_type_boost_item_ignores_offtype():
    fire = _state(_mon("Charizard", spa=140, item="Charcoal"), None, _mon("Garchomp", hp_stat=400, spd=260), None)
    none = _state(_mon("Charizard", spa=140), None, _mon("Garchomp", hp_stat=400, spd=260), None)
    assert abs(_dmg(fire, "Air Slash") - _dmg(none, "Air Slash")) < 0.5   # Charcoal does NOT boost a Flying move


def test_double_switch_uses_distinct_bench_mons():
    bench = [_mon("Rotom"), _mon("Kingambit")]
    s = _state(_mon("Garchomp"), _mon("Sneasler"), _mon("Snorlax"), None, our_bench=bench)
    nxt, _ = W.white_box_sim(s, {"our_a": {"kind": "switch", "bench_index": 0},
                                 "our_b": {"kind": "switch", "bench_index": 1}})
    got = {nxt["our_active"]["our_a"]["base_species"], nxt["our_active"]["our_b"]["base_species"]}
    assert got == {"Rotom", "Kingambit"}              # not both Rotom (stale-index bug)
