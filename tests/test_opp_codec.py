"""Task #9a: opponent-perspective action codec (auxiliary opp head).

The opp codec flips the snapshot (our<->opp) and reuses build_action_mask /
action_to_index, so the opponent's legal actions + chosen-action index are
encoded in the same 0-15 codec from the opponent's seat (our mons = its foes).
"""

from __future__ import annotations

import v_dance.encoders.state_encoder as se


def _mon(species, moves):
    return {"species": species, "base_species": species, "is_fainted": False,
            "known_moves": list(moves), "revealed_moves": []}


def _snap():
    return {
        "our_active": {"our_a": _mon("Incineroar", ["Knock Off", "Fake Out", "Parting Shot", "Flare Blitz"]),
                       "our_b": _mon("Kingambit", ["Sucker Punch"])},
        "opp_active": {"opp_a": _mon("Charizard", ["Heat Wave", "Air Slash", "Protect", "Solar Beam"]),
                       "opp_b": _mon("Venusaur", ["Sludge Bomb"])},
        "our_bench": [_mon("Rillaboom", ["Grassy Glide"])],
        "opp_bench": [_mon("Amoonguss", ["Spore"])],
    }


def test_flip_perspective_swaps_and_relabels():
    f = se._flip_perspective(_snap())
    assert set(f["our_active"]) == {"our_a", "our_b"}
    assert f["our_active"]["our_a"]["species"] == "Charizard"    # opp_a → our_a
    assert f["opp_active"]["opp_a"]["species"] == "Incineroar"   # our_a → opp_a
    assert f["our_bench"][0]["species"] == "Amoonguss"           # opp_bench → our_bench
    assert f["opp_bench"][0]["species"] == "Rillaboom"


def test_flip_action_slots():
    a = se._flip_action_slots({"slot": "opp_a", "target_slot": "our_b"})
    assert a["slot"] == "our_a" and a["target_slot"] == "opp_b"


def test_opp_action_to_index_encodes_and_is_legal():
    snap = _snap()
    idx = se.opp_action_to_index(
        {"slot": "opp_a", "action": "move", "move": "Heat Wave", "target_slot": "our_a"}, snap)
    assert idx is not None
    assert se.build_opp_action_mask(snap)["opp_a"][idx] == 1     # legal under opp mask


def test_opp_mask_symmetric_board_mirrors_our_mask():
    """On a mirror board (identical mons + benches both sides) the opponent's
    legal-action mask must equal ours, relabeled — a real (non-tautological)
    check that the flip reuses the codec correctly."""
    snap = {
        "our_active": {"our_a": _mon("Incineroar", ["Knock Off", "Fake Out"]),
                       "our_b": _mon("Kingambit", ["Sucker Punch", "Iron Head"])},
        "opp_active": {"opp_a": _mon("Incineroar", ["Knock Off", "Fake Out"]),
                       "opp_b": _mon("Kingambit", ["Sucker Punch", "Iron Head"])},
        "our_bench": [_mon("Rillaboom", ["Grassy Glide"])],
        "opp_bench": [_mon("Rillaboom", ["Grassy Glide"])],
    }
    our = se.build_action_mask(snap)
    opp = se.build_opp_action_mask(snap)
    assert opp["opp_a"] == our["our_a"]
    assert opp["opp_b"] == our["our_b"]


def test_annotate_opp_actions_stamps_index_and_mask():
    snap = _snap()
    t = {"state_before_actions": snap, "opp_actions_actual": [
        {"slot": "opp_a", "action": "move", "move": "Heat Wave", "target_slot": "our_a"},
        {"slot": "opp_b", "action": "move", "move": "Sludge Bomb", "target_slot": "our_b"},
    ]}
    se.annotate_opp_actions(t)
    assert set(t["opp_action_mask"]) == {"opp_a", "opp_b"}
    for act in t["opp_actions_actual"]:
        idx = act["opp_action_index"]
        assert idx is not None
        assert t["opp_action_mask"][act["slot"]][idx] == 1       # invariant: legal


def test_annotate_opp_actions_drops_unencodable():
    snap = _snap()
    t = {"state_before_actions": snap, "opp_actions_actual": [
        {"slot": "opp_a", "action": "move", "move": "Hyper Beam", "target_slot": "our_a"},
    ]}
    se.annotate_opp_actions(t)
    assert t["opp_actions_actual"][0]["opp_action_index"] is None  # not Charizard's move


def test_opp_switch_encodes_against_opp_bench():
    snap = _snap()
    idx = se.opp_action_to_index(
        {"slot": "opp_a", "action": "switch", "species": "Amoonguss"}, snap)
    assert idx is not None and idx >= se.SWITCH_OFFSET            # opp bench mon
