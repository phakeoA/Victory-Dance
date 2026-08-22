"""v11 B3 — defender Assault Vest / Eviolite + attacker Expert Belt / type-boost items (+ B3b resist berries).

AV onModifySpD ×1.5 → incoming special ×2/3 · Eviolite onModifyDef+SpD ×1.5 (NFE holder) → both sides ×2/3
(both in the shared _situational_damage_mult). Expert Belt ×1.2 on a super-effective hit · type-boost items
×1.2 on a matching-type move (attacker, move-writer band). B3b gap-scan: resist berry ×0.5 on a SE hit of its
type (Chilan = all Normal). VALUE-only, no layout. Design: scratch/v11_b3* (wf_5e1ca4d3).
"""
from __future__ import annotations

from v_dance.encoders.damage_mechanics import is_nfe
from v_dance.encoders.state_encoder import (
    StateEncoder, MOVE_FEATURES, _MOVE_BLOCK_REL, move_slots_for_mon, norm_species,
    _situational_damage_mult, _BAND_ITEM_MULT, _TYPE_BOOST_TYPE, _RESIST_BERRY_TYPE,
)

_TT = 2.0 / 3.0


# ════════════════════════════ is_nfe + maps ════════════════════════════
def test_is_nfe():
    for s in ("chansey", "scyther", "porygon2", "dusclops", "kirlia", "gabite"):
        assert is_nfe(s) is True
    for s in ("garchomp", "blissey", "dragapult", "porygonz", "mimikyu", None, ""):
        assert is_nfe(s) is False


def test_item_maps():
    assert len(_TYPE_BOOST_TYPE) == 40 and len(_RESIST_BERRY_TYPE) == 18
    assert _TYPE_BOOST_TYPE["charcoal"] == "FIRE" and _TYPE_BOOST_TYPE["dracoplate"] == "DRAGON"
    assert _RESIST_BERRY_TYPE["occaberry"] == "FIRE" and _RESIST_BERRY_TYPE["chilanberry"] == "NORMAL"
    assert "expertbelt" not in _TYPE_BOOST_TYPE      # EB is SE-gated, not a type item


# ════════════════════════════ AV / Eviolite (defender D-term) unit ════════════════════════════
def _D(**k):
    d = {"types": ["NORMAL"], "grounded": True, "screen_phys": False, "screen_spec": False,
         "assault_vest": False, "evio": False}
    d.update(k)
    return d


def test_assault_vest_special_only():
    assert _situational_damage_mult("WATER", False, None, None, _D(assault_vest=True), False, False, False) == _TT
    assert _situational_damage_mult("WATER", True, None, None, _D(assault_vest=True), False, False, False) == 1.0
    # Psyshock (special, hits Def) → AV (SpD-only) does NOT fire
    assert _situational_damage_mult("PSYCHIC", False, None, None, _D(assault_vest=True), False, False, False,
                                    hits_def=True) == 1.0


def test_eviolite_both_sides():
    assert _situational_damage_mult("WATER", True, None, None, _D(evio=True), False, False, False) == _TT
    assert _situational_damage_mult("WATER", False, None, None, _D(evio=True), False, False, False) == _TT
    assert _situational_damage_mult("WATER", True, None, None, _D(), False, False, False) == 1.0   # no item


def test_av_stacks_with_sand():
    rock = _D(types=["ROCK"], assault_vest=True)
    # Sand SpD ×1.5 AND AV SpD ×1.5 on a special move → ×2/3 × ×2/3
    assert abs(_situational_damage_mult("WATER", False, "SANDSTORM", None, rock, False, False, False)
               - _TT * _TT) < 1e-9


# ════════════════════════════ e2e: attacker items + resist berry (move-writer band) ════════════════════════════
def _mon(species, ability="Pressure", *, moves=(), item=None, spa=70, hp=520):
    m = {"species": species, "base_species": species, "hp_pct": 100.0, "seen": True, "is_fainted": False,
         "known_moves": list(moves), "revealed_moves": [], "boosts": {}, "status": None,
         "known_ability": ability, "volatiles": {},
         "stats_estimate": {"mode": "exact",
                            "stats": {"atk": 70, "spa": spa, "def": 120, "spd": 120, "hp": hp, "spe": 80}}}
    if item is not None:
        m["known_item"] = item
    return m


def _band(att, move, defender):
    snap = {"our_active": {"our_a": att, "our_b": None},
            "opp_active": {"opp_a": defender, "opp_b": None},
            "our_bench": [], "opp_bench": [], "field": {}, "side_conditions": {}}
    vec = StateEncoder().encode_snapshot(snap, turn=1)
    for m_idx, (mv, _c) in enumerate(move_slots_for_mon(att)):
        if norm_species(mv) == norm_species(move):
            b = _MOVE_BLOCK_REL + m_idx * MOVE_FEATURES
            return float(vec[b + 13]), float(vec[b + 14])
    raise AssertionError(move)


def test_type_boost_item_e2e():
    # Charcoal boosts a Fire move ×1.2 (Flamethrower vs Blissey = neutral, unclamped).
    no = _band(_mon("Charizard", "Blaze", moves=["Flamethrower"]), "Flamethrower", _mon("Blissey"))
    yes = _band(_mon("Charizard", "Blaze", moves=["Flamethrower"], item="Charcoal"), "Flamethrower", _mon("Blissey"))
    assert abs(yes[1] / no[1] - _BAND_ITEM_MULT) < 5e-3
    # Charcoal does NOT boost a non-Fire move.
    no_w = _band(_mon("Blastoise", "Torrent", moves=["Surf"]), "Surf", _mon("Blissey"))
    cha_w = _band(_mon("Blastoise", "Torrent", moves=["Surf"], item="Charcoal"), "Surf", _mon("Blissey"))
    assert abs(cha_w[1] - no_w[1]) < 1e-4


def test_expert_belt_se_only():
    # Earthquake vs Arcanine (Fire, grounded) = 2× SE → Expert Belt ×1.2.
    no = _band(_mon("Garchomp", "Rough Skin", moves=["Earthquake"], spa=70), "Earthquake", _mon("Arcanine"))
    eb = _band(_mon("Garchomp", "Rough Skin", moves=["Earthquake"], item="Expert Belt"), "Earthquake", _mon("Arcanine"))
    assert abs(eb[1] / no[1] - _BAND_ITEM_MULT) < 5e-3
    # Earthquake vs Blissey (Normal) = neutral → Expert Belt does NOT fire.
    no_n = _band(_mon("Garchomp", "Rough Skin", moves=["Earthquake"]), "Earthquake", _mon("Blissey"))
    eb_n = _band(_mon("Garchomp", "Rough Skin", moves=["Earthquake"], item="Expert Belt"), "Earthquake", _mon("Blissey"))
    assert abs(eb_n[1] - no_n[1]) < 1e-4


def test_resist_berry_e2e():
    # Flamethrower vs Glaceon (Ice) = 2× SE; Occa Berry halves the SE Fire damage ×0.5.
    no = _band(_mon("Charizard", "Blaze", moves=["Flamethrower"]), "Flamethrower", _mon("Glaceon"))
    occ = _band(_mon("Charizard", "Blaze", moves=["Flamethrower"]), "Flamethrower", _mon("Glaceon", item="Occa Berry"))
    assert abs(occ[1] / no[1] - 0.5) < 5e-3
    # Occa does NOT resist a non-Fire move.
    no_w = _band(_mon("Blastoise", "Torrent", moves=["Surf"]), "Surf", _mon("Glaceon"))
    occ_w = _band(_mon("Blastoise", "Torrent", moves=["Surf"]), "Surf", _mon("Glaceon", item="Occa Berry"))
    assert abs(occ_w[1] - no_w[1]) < 1e-4
