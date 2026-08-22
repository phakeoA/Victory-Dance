"""v11 exact-mask — conservatively tighten the OFFLINE training action mask (build_action_mask) to match
the serve mask's legality, WITHOUT ever dropping the actually-taken corpus action.

Invariant (forbidden to violate): taken_action ⊆ exact_offline_mask ⊆ serve_mask. Drops only on certainty;
Encore / KNOWN-Choice locks fail OPEN. Switch-trapping, belief-Choice and Imprison stay a superset. The new
restriction state (taunt / disabled_move / encore_move / stint_moves) is captured by the parser. Design:
workflow wf_a1398bfb-d2e. Corpus mask is STORED → these tighten on the end-of-v11 re-annotation/re-export.
"""
from __future__ import annotations

from v_dance.parser.vod_parser.replay_parser import ShowdownReplayParser
from v_dance.encoders.state_encoder import (
    build_action_mask, _certainly_illegal_move_slots, move_slots_for_mon, norm_species,
)

# ════════════════════════════ parser restriction capture ════════════════════════════
_H = """|player|p1|A|1|
|player|p2|B|1|
|poke|p1|Annihilape, L50, M|
|poke|p1|Garchomp, L50, M|
|poke|p2|Garchomp, L50, M|
|teamsize|p1|2
|teamsize|p2|1
|start
|switch|p1a: Annihilape|Annihilape, L50, M|100/100
|switch|p2a: Garchomp|Garchomp, L50, M|100/100
"""


def _vol(body, turn, key):
    r = ShowdownReplayParser(_H + body + "|turn|999\n", our_player="p1").parse()
    for t in r["turns"]:
        if t["turn"] == turn:
            return t["state_before_actions"]["p1"]["our_active"]["our_a"]["volatiles"].get(key)
    raise AssertionError(turn)


def test_disable_captures_named_move():
    b = "|turn|1\n|move|p1a: Annihilape|Close Combat|p2a: Garchomp\n|-start|p1a: Annihilape|Disable|Close Combat\n|turn|2\n"
    assert _vol(b, 2, "disabled_move") == "Close Combat"


def test_disable_end_clears():
    b = ("|turn|1\n|move|p1a: Annihilape|Close Combat|p2a: Garchomp\n|-start|p1a: Annihilape|Disable|Close Combat\n"
         "|turn|2\n|-end|p1a: Annihilape|Disable\n|turn|3\n")
    assert _vol(b, 3, "disabled_move") is None


def test_encore_derives_last_move():
    b = "|turn|1\n|move|p1a: Annihilape|Bulk Up|p2a: Garchomp\n|-start|p1a: Annihilape|Encore\n|turn|2\n"
    assert _vol(b, 2, "encore_move") == "Bulk Up"


def test_taunt_split_bool():
    b = "|turn|1\n|-start|p1a: Annihilape|move: Taunt\n|turn|2\n"
    assert _vol(b, 2, "taunt") is True
    assert _vol(b, 2, "move_restricted") is True       # the bundle still fires


def test_switch_clears_restrictions():
    # Disable Annihilape, switch it out + back → cleared.
    b = ("|turn|1\n|move|p1a: Annihilape|Close Combat|p2a: Garchomp\n|-start|p1a: Annihilape|Disable|Close Combat\n"
         "|switch|p1a: Garchomp|Garchomp, L50, M|100/100\n|turn|2\n"
         "|switch|p1a: Annihilape|Annihilape, L50, M|100/100\n|turn|3\n")
    assert _vol(b, 3, "disabled_move") is None


# ════════════════════════════ conservative drop helper ════════════════════════════
def _mon(**kw):
    base = {"species": "X", "base_species": "X", "hp_pct": 100.0, "seen": True, "is_fainted": False,
            "revealed_moves": [], "boosts": {}, "status": None, "known_ability": None, "volatiles": {},
            "known_item": None, "item_consumed": False, "move_pp_used": {}, "stint_moves": []}
    base.update(kw)
    return base


def test_zero_pp_dropped_inflated_denominator():
    # Tackle base PP 35 → pp_max = 35*8//5 = 56. pp_used 35 (empty without PP Ups) is KEPT (conservative);
    # only pp_used >= 56 drops.
    assert _certainly_illegal_move_slots(_mon(known_moves=["Tackle"], move_pp_used={"tackle": 35}), {}) == set()
    assert _certainly_illegal_move_slots(_mon(known_moves=["Tackle"], move_pp_used={"tackle": 56}), {}) == {0}


def test_gravity_drops_flagged_move():
    mn = _mon(known_moves=["Fly", "Earthquake"])
    assert _certainly_illegal_move_slots(mn, {"gravity_turns_remaining": 5}) == {0}
    assert _certainly_illegal_move_slots(mn, {}) == set()


def test_first_turn_gating():
    mn = _mon(known_moves=["Fake Out", "Knock Off"])
    assert _certainly_illegal_move_slots(mn, {}) == {0}                                  # not first turn
    mn2 = _mon(known_moves=["Fake Out", "Knock Off"], volatiles={"first_turn": True})
    assert _certainly_illegal_move_slots(mn2, {}) == set()


def test_taunt_drops_status_keeps_damaging():
    mn = _mon(known_moves=["Knock Off", "Will-O-Wisp"], volatiles={"taunt": True})
    assert _certainly_illegal_move_slots(mn, {}) == {1}                                  # WoW is status


def test_taunt_uses_split_bool_not_bundle():
    # move_restricted bundle True (e.g. Disable) but taunt False → status move NOT dropped by Taunt rule.
    mn = _mon(known_moves=["Knock Off", "Will-O-Wisp"],
              volatiles={"move_restricted": True, "taunt": False})
    assert 1 not in _certainly_illegal_move_slots(mn, {})


def test_known_choice_locks_to_stint0():
    mn = _mon(known_moves=["Knock Off", "Sucker Punch"], known_item="Choice Band",
              stint_moves=["Knock Off"])
    assert _certainly_illegal_move_slots(mn, {}) == {1}                                  # only Knock Off legal


def test_belief_choice_does_not_lock():
    mn = _mon(known_moves=["Knock Off", "Sucker Punch"], stint_moves=["Knock Off"])     # item unknown
    assert _certainly_illegal_move_slots(mn, {}) == set()


def test_consumed_choice_does_not_lock():
    mn = _mon(known_moves=["Knock Off", "Sucker Punch"], known_item="Choice Band",
              item_consumed=True, stint_moves=["Knock Off"])
    assert _certainly_illegal_move_slots(mn, {}) == set()


def test_disable_drops_named_move():
    mn = _mon(known_moves=["Close Combat", "Knock Off"], volatiles={"disabled_move": "Close Combat"})
    assert _certainly_illegal_move_slots(mn, {}) == {0}


def test_encore_locks_to_named_move():
    mn = _mon(known_moves=["Bulk Up", "Knock Off"], volatiles={"encore_move": "Bulk Up"})
    assert _certainly_illegal_move_slots(mn, {}) == {1}                                  # only Bulk Up legal


def test_encore_undeterminable_fails_open():
    mn = _mon(known_moves=["Knock Off", "Sucker Punch"], volatiles={"encore_move": "Protect"})
    assert _certainly_illegal_move_slots(mn, {}) == set()                                # not in slots → keep all


def test_assault_vest_known_drops_status():
    mn = _mon(known_moves=["Knock Off", "Bulk Up"], known_item="Assault Vest")
    assert _certainly_illegal_move_slots(mn, {}) == {1}                                  # Bulk Up is status
    # belief-AV (unknown item) must NOT trigger
    assert _certainly_illegal_move_slots(_mon(known_moves=["Knock Off", "Bulk Up"]), {}) == set()


# ════════════════════════════ THE invariant: taken action never dropped ════════════════════════════
def _taken_legal(mon, taken_move, field=None):
    """build_action_mask gives the taken move at least one legal bucket (its slot survives)."""
    snap = {"our_active": {"our_a": mon, "our_b": None}, "opp_active": {"opp_a": _mon(species="Garchomp"),
            "opp_b": None}, "our_bench": [], "opp_bench": [], "field": field or {}, "side_conditions": {}}
    row = build_action_mask(snap)["our_a"]
    for m_idx, (name, _c) in enumerate(move_slots_for_mon(mon)):
        if norm_species(name) == norm_species(taken_move):
            return any(row[m_idx * 3 + b] for b in range(3))
    raise AssertionError(taken_move)


def test_taken_action_never_dropped():
    # Each restriction: the move the player ACTUALLY took stays legal.
    assert _taken_legal(_mon(known_moves=["Fake Out", "Knock Off"], volatiles={"first_turn": True}), "Fake Out")
    assert _taken_legal(_mon(known_moves=["Fly", "Earthquake"]), "Earthquake", {"gravity_turns_remaining": 5})
    assert _taken_legal(_mon(known_moves=["Knock Off", "Will-O-Wisp"], volatiles={"taunt": True}), "Knock Off")
    assert _taken_legal(_mon(known_moves=["Bulk Up", "Knock Off"], volatiles={"encore_move": "Bulk Up"}), "Bulk Up")
    assert _taken_legal(_mon(known_moves=["Knock Off", "Sucker Punch"], known_item="Choice Band",
                             stint_moves=["Knock Off"]), "Knock Off")
    assert _taken_legal(_mon(known_moves=["Close Combat", "Knock Off"],
                             volatiles={"disabled_move": "Close Combat"}), "Knock Off")


# ════════════════════════════ regression: no-restriction byte-identical + switches unchanged ════════════════════════════
def test_no_restriction_switch_buckets_unchanged():
    mon = _mon(known_moves=["Earthquake", "Rock Slide"])
    snap = {"our_active": {"our_a": mon, "our_b": None}, "opp_active": {"opp_a": _mon(species="Garchomp"),
            "opp_b": None},
            "our_bench": [_mon(species="Rotom"), _mon(species="Heatran")],
            "opp_bench": [], "field": {}, "side_conditions": {}}
    row = build_action_mask(snap)["our_a"]
    assert row[12] == 1 and row[13] == 1               # both living bench switches legal (superset retained)
