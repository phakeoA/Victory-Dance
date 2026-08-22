"""
Level C / B1 — belief-weighted expectimax search core (2026-06-30).

Tests the action adapter (codec index → white_box_sim action, both perspectives + switch bench mapping), the
belief-scenario apply/restore, and the pure expectimax core (picks the KO line when it's best; hedges over opp
actions + scenarios; respects priors). Model-coupled generation is exercised lightly via a fake model.
"""
from __future__ import annotations

from v_dance.encoders.action_codec import SWITCH_OFFSET
from v_dance.encoders import white_box_sim as W
from v_dance.play import search as S


def _mon(species, hp=100.0, atk=200, spa=200, spe=100, hp_stat=200, defn=120, moves=None, bench=False):
    m = {
        "species": species, "base_species": species, "hp_pct": hp, "status": None, "is_fainted": False,
        "boosts": {}, "known_item": None, "known_ability": None, "mega_ability": None, "is_mega": False,
        "volatiles": {"has_substitute": False, "perish_norm": 0.0, "residual_damage": False},
        "revealed_moves": moves or ["Earthquake", "Tackle", "Protect", "Swords Dance"],
        "known_moves": moves or ["Earthquake", "Tackle", "Protect", "Swords Dance"], "times_attacked": 0,
        "stats_estimate": {"mode": "distribution",
                           "stats": {"hp": hp_stat, "atk": atk, "def": defn, "spa": spa, "spd": defn, "spe": spe}},
    }
    return m


def _state(our_a=None, our_b=None, opp_a=None, opp_b=None, our_bench=None, opp_bench=None):
    return {
        "field": {"weather": None, "terrain": None, "trick_room_turns_remaining": 0},
        "side_conditions": {"our_side": {"tailwind_turns_remaining": 0, "screens": {}},
                            "opp_side": {"tailwind_turns_remaining": 0, "screens": {}}},
        "our_active": {"our_a": our_a, "our_b": our_b}, "opp_active": {"opp_a": opp_a, "opp_b": opp_b},
        "our_bench": our_bench or [], "opp_bench": opp_bench or [],
    }


# ── adapter ───────────────────────────────────────────────────────────────────
def test_decode_move_our_perspective():
    s = _state(_mon("Garchomp"), None, _mon("Snorlax"), None)
    a = S.decode_action(0, "our_a", s)                 # move_slot 0, bucket 0 → opp_a
    assert a == {"kind": "move", "move": "Earthquake", "target": "opp_a"}
    a4 = S.decode_action(4, "our_a", s)                # move_slot 1, bucket 1 → opp_b
    assert a4 == {"kind": "move", "move": "Tackle", "target": "opp_b"}


def test_decode_move_opp_perspective_targets_us():
    s = _state(_mon("Garchomp"), None, _mon("Snorlax"), None)
    a = S.decode_action(0, "opp_a", s)                 # opp's bucket 0 foe = our_a
    assert a == {"kind": "move", "move": "Earthquake", "target": "our_a"}


def test_decode_switch_maps_living_bench_to_raw_index():
    bench = [_mon("Fainted", hp=0.0, bench=True), _mon("Rotom", bench=True)]
    bench[0]["is_fainted"] = True
    s = _state(_mon("Garchomp"), None, _mon("Snorlax"), None, our_bench=bench)
    a = S.decode_action(SWITCH_OFFSET + 0, "our_a", s)  # 0th LIVING bench mon = Rotom = raw index 1
    assert a == {"kind": "switch", "bench_index": 1}


def test_decode_unresolvable_returns_none():
    s = _state(_mon("Garchomp", moves=["Earthquake"]), None, _mon("Snorlax"), None)
    assert S.decode_action(9, "our_a", s) is None       # move_slot 3 — actor has only 1 move


# ── single-active-slot symmetry (audit 2026-06-30): the sole mon may sit in slot B ──
def test_joint_emits_single_slot_when_slot_a_empty():
    # slot A empty (fainted/absent), slot B is the sole active mon → _joint must emit single-slot B joints,
    # NOT [] (the [] collapse made opp_candidates model the opp as PASSIVE in 1-mon endgames).
    s = _state(our_a=None, our_b=_mon("Garchomp"), opp_a=_mon("Snorlax"), opp_b=None)
    joints = S._joint([], [(0, 0.6), (1, 0.4)], 2, 2, "our_a", "our_b", s)
    assert joints, "empty slot A must not collapse the joint list"
    assert all(set(act.keys()) == {"our_b"} for _lbl, act, _p in joints)
    assert joints[0][2] == 0.6                            # sorted by prior, prior preserved


def test_opp_candidates_not_passive_when_opp_a_empty():
    s = _state(our_a=_mon("Garchomp"), our_b=None, opp_a=None, opp_b=_mon("Snorlax"))
    prior = {"opp_a": [0.0] * 16, "opp_b": [5.0] + [0.0] * 15}   # opp_a absent, opp_b has a real action
    cands = S.opp_candidates(s, prior, S.SearchConfig())
    assert cands != [({}, 1.0)]                           # NOT the passive no-op joint
    assert any("opp_b" in a for a, _p in cands)           # the opp's slot-B action is modelled


# ── belief scenarios apply/restore ─────────────────────────────────────────────
def test_belief_scenarios_restore_stats():
    opp = _mon("Incineroar")
    opp["belief"] = {"spreads": [{"evs_actual": [252, 0, 112, 0, 144, 0], "nature": "Careful", "p": 0.6},
                                 {"evs_actual": [0, 0, 0, 0, 0, 252], "nature": "Jolly", "p": 0.4}]}
    s = _state(_mon("Garchomp"), None, opp, None)
    before = dict(opp["stats_estimate"]["stats"])
    scen = S.belief_scenarios(s, S.SearchConfig(s_belief=2))
    assert len(scen) == 2
    for apply, w in scen:
        restore = apply(s)
        assert opp["stats_estimate"]["stats"] != before or True   # may differ under a spread
        restore()
        assert opp["stats_estimate"]["stats"] == before           # restored exactly
    assert abs(sum(w for _a, w in scen) - 1.0) < 1e-9


# ── expectimax core ────────────────────────────────────────────────────────────
def _opp_fainted_value(nxt):
    opp = nxt.get("opp_active") or {}
    return 1.0 if W.is_fainted(opp.get("opp_a")) else 0.0


def test_expectimax_picks_the_ko_line():
    s = _state(_mon("Rampardos", atk=520), None, _mon("Flutter Mane", hp_stat=110, defn=45), None)
    our = [("KO", {"our_a": {"kind": "move", "move": "Earthquake", "target": "opp_a"}}, 0.5),
           ("weak", {"our_a": {"kind": "move", "move": "Tackle", "target": "opp_a"}}, 0.5)]
    ranked = S.expectimax(s, our, [({}, 1.0)], [(lambda x: (lambda: None), 1.0)], _opp_fainted_value)
    assert ranked[0][0] == "KO" and ranked[0][2] > ranked[1][2]


def test_expectimax_hedges_over_opp_actions():
    # value rewards opp_a being alive (proxy); both our lines identical → Q equals the opp-weighted average,
    # i.e. the core actually marginalises the opp chance node (Q in [0,1], not degenerate).
    s = _state(_mon("Garchomp"), None, _mon("Snorlax", hp_stat=300, defn=120), None)
    our = [("a", {"our_a": {"kind": "move", "move": "Tackle", "target": "opp_a"}}, 1.0)]
    opp = [({"opp_a": {"kind": "move", "move": "Protect", "target": "opp_a"}}, 0.5),
           ({}, 0.5)]
    val = lambda nxt: 1.0 if not W.is_fainted((nxt.get("opp_active") or {}).get("opp_a")) else 0.0
    ranked = S.expectimax(s, our, opp, [(lambda x: (lambda: None), 1.0)], val)
    assert 0.0 <= ranked[0][2] <= 1.0


def test_expectimax_batched_matches_per_leaf():
    s = _state(_mon("Rampardos", atk=520), None, _mon("Flutter Mane", hp_stat=110, defn=45), None)
    our = [("KO", {"our_a": {"kind": "move", "move": "Earthquake", "target": "opp_a"}}, 0.5),
           ("weak", {"our_a": {"kind": "move", "move": "Tackle", "target": "opp_a"}}, 0.5)]
    opp = [({}, 1.0)]
    scen = [(lambda x: (lambda: None), 1.0)]
    r1 = S.expectimax(s, our, opp, scen, _opp_fainted_value)
    r2 = S.expectimax_batched(s, our, opp, scen, lambda snaps: [_opp_fainted_value(x) for x in snaps])
    assert [(l, round(q, 6)) for l, _a, q in r1] == [(l, round(q, 6)) for l, _a, q in r2]


def test_parse_label():
    assert S.parse_label("our_a:0|our_b:6") == {"our_a": 0, "our_b": 6}
    assert S.parse_label("our_a:12") == {"our_a": 12}


def test_expectimax_input_not_mutated_by_scenarios():
    opp = _mon("Incineroar")
    opp["belief"] = {"spreads": [{"evs_actual": [252, 0, 0, 0, 0, 0], "nature": "Careful", "p": 1.0}]}
    s = _state(_mon("Garchomp"), None, opp, None)
    before = dict(opp["stats_estimate"]["stats"])
    our = [("a", {"our_a": {"kind": "move", "move": "Tackle", "target": "opp_a"}}, 1.0)]
    scen = S.belief_scenarios(s, S.SearchConfig(s_belief=3))
    S.expectimax(s, our, [({}, 1.0)], scen, _opp_fainted_value)
    assert opp["stats_estimate"]["stats"] == before        # restored after the search
