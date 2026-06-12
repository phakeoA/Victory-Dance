"""Tests for the state_encoder action codec — string actions ⇄ fixed 0–15
indices + decision-time legality masks.

Frozen convention (state_encoder module docstring):
    0–11  move slot m (0–3) × target bucket t (0=opp_a, 1=opp_b, 2=ally)
    12–15 switch to living-bench slot 0–3
"""

from __future__ import annotations

import pytest

from state_encoder import (
    ACTIONS_PER_SLOT,
    SWITCH_OFFSET,
    action_to_index,
    annotate_transition_actions,
    build_action_mask,
    index_to_action,
    move_slots_for_mon,
)


# ── Synthetic snapshot helpers ─────────────────────────────────────────────

def _mon(species="Sneasler", moves=("Fake Out", "Close Combat", "Dire Claw", "Protect"), **over):
    d = {
        "species": species,
        "base_species": species,
        "is_fainted": False,
        "known_moves": list(moves),
        "revealed_moves": [],
    }
    d.update(over)
    return d


def _snap(**over):
    d = {
        "our_active": {
            "our_a": _mon(),
            "our_b": _mon("Incineroar",
                          ("Knock Off", "Parting Shot", "Helping Hand", "Rock Slide")),
        },
        "opp_active": {
            "opp_a": _mon("Kingambit", ("Sucker Punch",)),
            "opp_b": _mon("Rotom-Wash", ("Hydro Pump",)),
        },
        "our_bench": [
            _mon("Meganium", ("Giga Drain",)),
            _mon("Diggersby", ("Earthquake",)),
            _mon("Palafin", ("Jet Punch",), is_fainted=True),   # dead — not a slot
        ],
    }
    d.update(over)
    return d


# ── move_slots_for_mon ─────────────────────────────────────────────────────

def test_move_slots_known_beats_revealed_and_belief_pads():
    mon = _mon(moves=("Fake Out", "Close Combat"),
               revealed_moves=["Dire Claw"],
               belief={"moves_predicted": [{"name": "Swords Dance", "p": 0.61},
                                           {"name": "Protect", "p": 0.55},
                                           {"name": "U-turn", "p": 0.20}]})
    slots = move_slots_for_mon(mon)
    assert [s[0] for s in slots] == ["Fake Out", "Close Combat",
                                     "Swords Dance", "Protect"]
    assert slots[0][1] == 1.0 and slots[2][1] == pytest.approx(0.61)


def test_move_slots_empty_mon():
    assert move_slots_for_mon(None) == []
    assert move_slots_for_mon({"species": "X"}) == []


# ── action_to_index: moves ─────────────────────────────────────────────────

def test_move_single_target_buckets():
    snap = _snap()
    mon = snap["our_active"]["our_a"]
    act = {"slot": "our_a", "action": "move", "move": "Close Combat"}
    assert action_to_index({**act, "target_slot": "opp_a"}, mon, snap) == 1 * 3 + 0
    assert action_to_index({**act, "target_slot": "opp_b"}, mon, snap) == 1 * 3 + 1
    # targeting the ally (e.g. into a Weakness Policy) → bucket 2
    assert action_to_index({**act, "target_slot": "our_b"}, mon, snap) == 1 * 3 + 2


def test_move_self_and_spread_canonicalise_to_bucket_zero():
    snap = _snap()
    mon_a = snap["our_active"]["our_a"]
    mon_b = snap["our_active"]["our_b"]
    # Protect (self) — slot 3 of our_a → 3*3+0 = 9
    assert action_to_index({"action": "move", "move": "Protect",
                            "target_slot": None}, mon_a, snap) == 9
    # Rock Slide (allAdjacentFoes) — slot 3 of our_b; the log sometimes
    # records one of the struck targets, the bucket must stay canonical
    assert action_to_index({"action": "move", "move": "Rock Slide",
                            "target_slot": "opp_b"}, mon_b, snap) == 9


def test_move_ally_kind_is_bucket_two():
    snap = _snap()
    mon_b = snap["our_active"]["our_b"]
    # Helping Hand (adjacentAlly) — slot 2 → 2*3+2 = 8
    assert action_to_index({"action": "move", "move": "Helping Hand",
                            "target_slot": "our_a"}, mon_b, snap) == 8


def test_move_loose_name_matching():
    snap = _snap()
    mon = snap["our_active"]["our_a"]
    assert action_to_index({"action": "move", "move": "fake out",
                            "target_slot": "opp_a"}, mon, snap) == 0


def test_move_outside_encoded_slots_is_none():
    snap = _snap()
    mon = snap["our_active"]["our_a"]
    assert action_to_index({"action": "move", "move": "Tera Blast",
                            "target_slot": "opp_a"}, mon, snap) is None
    assert action_to_index({"action": "move", "move": "Fake Out",
                            "target_slot": "opp_a"}, None, snap) is None


# ── action_to_index: switches ──────────────────────────────────────────────

def test_switch_indexes_living_bench_order():
    snap = _snap()
    assert action_to_index({"action": "switch", "species": "Meganium"},
                           None, snap) == SWITCH_OFFSET + 0
    assert action_to_index({"action": "switch", "species": "Diggersby"},
                           None, snap) == SWITCH_OFFSET + 1


def test_switch_to_fainted_or_unknown_is_none():
    snap = _snap()
    # Palafin is fainted — not a legal bench slot
    assert action_to_index({"action": "switch", "species": "Palafin"},
                           None, snap) is None
    assert action_to_index({"action": "switch", "species": "Pikachu"},
                           None, snap) is None


def test_switch_matches_base_species_of_mega():
    snap = _snap()
    snap["our_bench"][0]["species"] = "Meganium-Mega"
    snap["our_bench"][0]["base_species"] = "Meganium"
    assert action_to_index({"action": "switch", "species": "Meganium"},
                           None, snap) == SWITCH_OFFSET + 0


# ── index_to_action round trip ─────────────────────────────────────────────

def test_index_to_action_decodes_all_sixteen():
    for idx in range(ACTIONS_PER_SLOT):
        d = index_to_action(idx)
        if idx < SWITCH_OFFSET:
            assert d["kind"] == "move"
            assert d["move_slot"] == idx // 3
            assert d["target_bucket"] == idx % 3
        else:
            assert d["kind"] == "switch"
            assert d["bench_slot"] == idx - SWITCH_OFFSET
    with pytest.raises(ValueError):
        index_to_action(16)


# ── build_action_mask ──────────────────────────────────────────────────────

def test_mask_shape_and_basic_legality():
    mask = build_action_mask(_snap())
    assert set(mask) == {"our_a", "our_b"}
    assert all(len(row) == ACTIONS_PER_SLOT for row in mask.values())

    a = mask["our_a"]
    # Fake Out (normal): both foes + ally reachable
    assert a[0] == a[1] == a[2] == 1
    # Protect (self): only canonical bucket 0
    assert a[9] == 1 and a[10] == 0 and a[11] == 0
    # two living bench mons → switches 12,13 only (Palafin is fainted)
    assert a[12] == 1 and a[13] == 1 and a[14] == 0 and a[15] == 0

    b = mask["our_b"]
    # Helping Hand (adjacentAlly): bucket 2 only
    assert b[2 * 3 + 0] == 0 and b[2 * 3 + 1] == 0 and b[2 * 3 + 2] == 1
    # Rock Slide (spread): bucket 0 only
    assert b[9] == 1 and b[10] == 0 and b[11] == 0


def test_mask_missing_opp_slot_blocks_that_bucket():
    snap = _snap()
    del snap["opp_active"]["opp_b"]
    a = build_action_mask(snap)["our_a"]
    assert a[0] == 1 and a[1] == 0          # Fake Out → opp_b gone
    assert a[2] == 1                         # ally still there


def test_mask_no_ally_blocks_ally_bucket_and_helping_hand():
    snap = _snap()
    del snap["our_active"]["our_b"]
    mask = build_action_mask(snap)
    a = mask["our_a"]
    assert a[2] == 0                         # no ally to point Fake Out at
    assert mask["our_b"] == [0] * ACTIONS_PER_SLOT   # empty slot → all zero

    snap2 = _snap()
    del snap2["our_active"]["our_a"]
    b = build_action_mask(snap2)["our_b"]
    assert b[2 * 3 + 2] == 0                 # Helping Hand with no ally


def test_mask_no_foes_keeps_reachable_targets_only():
    # no foes, ally present → only the ally bucket stays for "normal" moves
    snap = _snap(opp_active={})
    a = build_action_mask(snap)["our_a"]
    assert a[0] == 0 and a[1] == 0 and a[2] == 1

    # nothing reachable at all → canonical bucket-0 fallback (Showdown still
    # lets you select the move; it auto-targets / fails)
    snap2 = _snap(opp_active={})
    del snap2["our_active"]["our_b"]
    a2 = build_action_mask(snap2)["our_a"]
    assert a2[0] == 1 and a2[1] == 0 and a2[2] == 0


def test_mask_unknown_moves_only_belief_padded():
    snap = _snap()
    snap["our_active"]["our_a"] = _mon(moves=(), revealed_moves=[])
    a = build_action_mask(snap)["our_a"]
    assert sum(a[:12]) == 0                  # no known moves → none legal
    assert a[12] == 1                        # switching still allowed


# ── annotate_transition_actions ────────────────────────────────────────────

def test_annotate_stamps_index_and_mask():
    snap = _snap()
    t = {
        "state_before_actions": snap,
        "our_actions": [
            {"slot": "our_a", "action": "move", "move": "Fake Out",
             "target_slot": "opp_b"},
            {"slot": "our_b", "action": "switch", "species": "Diggersby"},
        ],
    }
    annotate_transition_actions(t)
    assert t["our_actions"][0]["action_index"] == 1
    assert t["our_actions"][1]["action_index"] == SWITCH_OFFSET + 1
    assert t["action_mask"]["our_a"][1] == 1
    assert t["action_mask"]["our_b"][SWITCH_OFFSET + 1] == 1


# ── Integration: the real VOD export ──────────────────────────────────────

@pytest.fixture(scope="module")
def vod_transitions(vod_path):
    from vod_parser import replay_to_transitions
    return replay_to_transitions(vod_path, None, None, players=["p1", "p2"])


def test_vod_every_transition_has_mask(vod_transitions):
    for t in vod_transitions:
        mask = t["action_mask"]
        assert set(mask) == {"our_a", "our_b"}
        assert all(len(row) == ACTIONS_PER_SLOT and
                   set(row) <= {0, 1} for row in mask.values())


def test_vod_actions_encode_and_respect_mask(vod_transitions):
    """Every encodable action must be marked legal by its own mask, and the
    real VOD should encode the overwhelming majority of decisions."""
    total = encoded = 0
    for t in vod_transitions:
        for act in t["our_actions"]:
            total += 1
            idx = act["action_index"]
            if idx is None:
                continue
            encoded += 1
            assert 0 <= idx < ACTIONS_PER_SLOT
            assert t["action_mask"][act["slot"]][idx] == 1, (
                f"turn {t['turn']} {t['perspective']} {act} "
                f"mask={t['action_mask'][act['slot']]}"
            )
    assert total > 0
    assert encoded / total >= 0.9, f"only {encoded}/{total} actions encodable"
