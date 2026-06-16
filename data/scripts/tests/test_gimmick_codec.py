"""Tests for the gimmick (mega-evolution) codec in state_encoder.py.

The gimmick head is a SEPARATE per-slot 2-way decision {0=none, 1=mega},
orthogonal to the frozen 16-way move/switch head.  These tests cover:
  2a constants + accessor (ACTION_DIM stays frozen 16)
  2b action_to_gimmick
  2c build_gimmick_mask (mega-capability + team-megaed + empty/fainted)
  2d annotate_transition_actions stamps gimmick_index + gimmick_mask, invariant
"""

from __future__ import annotations

import state_encoder as se


# ── 2a: constants + accessor ────────────────────────────────────────────────
def test_gimmick_constants():
    assert se.GIMMICK_NONE == 0
    assert se.GIMMICK_MEGA == 1
    assert se.GIMMICK_DIM == 2
    assert se.get_gimmick_dim() == 2


def test_action_dim_still_frozen_16():
    # The gimmick head must NOT touch the move/switch action space.
    assert se.ACTION_DIM == 16
    assert se.ACTIONS_PER_SLOT == 16
    assert se.get_action_dim() == 16


# ── 2b: action_to_gimmick ───────────────────────────────────────────────────
def test_action_to_gimmick_mega_move():
    assert se.action_to_gimmick({"action": "move", "move": "Body Press", "mega": True}) == 1


def test_action_to_gimmick_plain_move():
    assert se.action_to_gimmick({"action": "move", "move": "Body Press"}) == 0
    assert se.action_to_gimmick({"action": "move", "move": "Body Press", "mega": False}) == 0


def test_action_to_gimmick_switch_never_megas():
    assert se.action_to_gimmick({"action": "switch", "species": "Incineroar"}) == 0


# ── 2c: build_gimmick_mask ──────────────────────────────────────────────────
def _mon(species, base=None, is_mega=False, is_fainted=False):
    return {"species": species, "base_species": base or species,
            "is_mega": is_mega, "is_fainted": is_fainted}


def test_species_is_mega_capable():
    assert se._species_is_mega_capable("Venusaur") is True
    assert se._species_is_mega_capable("Charizard") is True
    assert se._species_is_mega_capable("Floette-Eternal") is True
    assert se._species_is_mega_capable("Incineroar") is False
    assert se._species_is_mega_capable("Miraidon") is False
    assert se._species_is_mega_capable(None) is False


def test_gimmick_mask_capable_mon_allows_mega():
    snap = {"our_active": {"our_a": _mon("Venusaur")}}
    m = se.build_gimmick_mask(snap)
    assert m["our_a"] == [1, 1]          # none + mega both legal
    assert m["our_b"] == [0, 0]          # empty slot → all-zero


def test_gimmick_mask_noncapable_mon_forbids_mega():
    snap = {"our_active": {"our_a": _mon("Incineroar")}}
    m = se.build_gimmick_mask(snap)
    assert m["our_a"] == [1, 0]          # none legal, mega not


def test_gimmick_mask_fainted_slot_all_zero():
    snap = {"our_active": {"our_a": _mon("Venusaur", is_fainted=True)}}
    assert se.build_gimmick_mask(snap)["our_a"] == [0, 0]


def test_gimmick_mask_team_already_megaed_forbids_mega_everywhere():
    # our_b already mega'd (is_mega) → our_a may no longer mega even though it is
    # mega-capable (one mega per team per game).
    snap = {"our_active": {"our_a": _mon("Charizard"),
                           "our_b": _mon("Gengar-Mega", base="Gengar", is_mega=True)}}
    m = se.build_gimmick_mask(snap)
    assert m["our_a"] == [1, 0]
    assert m["our_b"] == [1, 0]


def test_gimmick_mask_team_megaed_detected_on_bench():
    # the spent mega may have been switched to the bench — still detected.
    snap = {"our_active": {"our_a": _mon("Charizard")},
            "our_bench": [_mon("Venusaur-Mega", base="Venusaur", is_mega=True)]}
    assert se.build_gimmick_mask(snap)["our_a"] == [1, 0]


def test_gimmick_mask_two_capable_mons_both_allow_mega_before_use():
    snap = {"our_active": {"our_a": _mon("Charizard"), "our_b": _mon("Venusaur")}}
    m = se.build_gimmick_mask(snap)
    assert m["our_a"] == [1, 1]
    assert m["our_b"] == [1, 1]


# ── 2d: annotate_transition_actions stamps gimmick_index + gimmick_mask ──────
def _g_mon(species, moves, **over):
    d = {"species": species, "base_species": species, "is_fainted": False,
         "is_mega": False, "known_moves": list(moves), "revealed_moves": []}
    d.update(over)
    return d


def _g_snap():
    return {
        "our_active": {
            "our_a": _g_mon("Venusaur", ("Body Press", "Sludge Bomb", "Protect", "Earth Power")),
            "our_b": _g_mon("Incineroar", ("Knock Off", "Fake Out", "Parting Shot", "Flare Blitz")),
        },
        "opp_active": {"opp_a": _g_mon("Kingambit", ("Sucker Punch",))},
        "our_bench": [],
    }


def test_annotate_stamps_gimmick_index_and_mask():
    t = {"state_before_actions": _g_snap(), "our_actions": [
        {"slot": "our_a", "action": "move", "move": "Body Press",
         "target_slot": "opp_a", "mega": True},
        {"slot": "our_b", "action": "move", "move": "Knock Off", "target_slot": "opp_a"},
    ]}
    se.annotate_transition_actions(t)
    a, b = t["our_actions"]
    assert a["action_index"] is not None and a["gimmick_index"] == 1   # Venusaur mega'd
    assert b["action_index"] is not None and b["gimmick_index"] == 0   # Incineroar plain
    assert t["gimmick_mask"]["our_a"] == [1, 1]
    assert t["gimmick_mask"]["our_b"] == [1, 0]


def test_annotate_drops_gimmick_when_mon_not_mega_capable():
    # Contrived: a mega flag on a non-capable mon (Incineroar).  The gimmick mask
    # forbids mega there, so the invariant drops gimmick_index to None — never a
    # wrong label.  (Does not occur on real data: every real mega is capable.)
    t = {"state_before_actions": _g_snap(), "our_actions": [
        {"slot": "our_b", "action": "move", "move": "Knock Off",
         "target_slot": "opp_a", "mega": True},
    ]}
    se.annotate_transition_actions(t)
    assert t["our_actions"][0]["action_index"] is not None
    assert t["our_actions"][0]["gimmick_index"] is None


def test_annotate_drops_gimmick_when_action_unencodable():
    # A mega move the mon does not have (not in its 4 slots) → action_index None
    # → gimmick None too (a checkbox on an action we could not encode).
    t = {"state_before_actions": _g_snap(), "our_actions": [
        {"slot": "our_a", "action": "move", "move": "Hyper Beam",
         "target_slot": "opp_a", "mega": True},
    ]}
    se.annotate_transition_actions(t)
    assert t["our_actions"][0]["action_index"] is None
    assert t["our_actions"][0]["gimmick_index"] is None


def test_annotate_switch_gimmick_zero():
    snap = _g_snap()
    snap["our_bench"] = [_g_mon("Meganium", ("Giga Drain",))]
    t = {"state_before_actions": snap, "our_actions": [
        {"slot": "our_a", "action": "switch", "species": "Meganium"},
    ]}
    se.annotate_transition_actions(t)
    assert t["our_actions"][0]["action_index"] is not None
    assert t["our_actions"][0]["gimmick_index"] == 0
