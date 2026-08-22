"""Piece 3 (Level-A opp-prediction): opponent-action alignment + schema fields.

Verifies that the post-game aligner stamps each perspective's ``opp_a_action`` /
``opp_b_action`` from the OTHER perspective's same-``(turn, decision_type)`` action
(seat-symmetric codec -> direct copy), leaving the ``PASS_ACTION`` sentinel wherever
the opponent passed or no matching step exists (so the Piece-2 aux loss masks it).
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("numpy")
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
from v_dance.selfplay.schema import (  # noqa: E402
    Transition, EpisodeMeta, Trajectory, PASS_ACTION,
)
from v_dance.selfplay.collector import (  # noqa: E402
    align_opponent_actions, align_paired_trajectories,
)


def _mk(turn, dt, a0, a1):
    return Transition(state=np.zeros(4, dtype=np.float32), action_s0=a0, action_s1=a1,
                      decision_type=dt, turn=turn)


def _traj(battle_id, role, steps):
    meta = EpisodeMeta(battle_id=battle_id, own_role=role, own_team=["x"] * 6,
                       opp_team=["y"] * 6, tp_bring=[0, 1, 2, 3], tp_leads=[0, 1],
                       won=None, terminal_type="draw", n_turns=len(steps))
    return Trajectory(meta=meta, transitions=[_mk(*s) for s in steps])


# ── align_opponent_actions ────────────────────────────────────────────────────
def test_align_copies_each_sides_actions_to_the_other():
    a = _traj("g", "p1", [(1, "turn", 3, 5), (2, "turn", 7, PASS_ACTION)])
    b = _traj("g", "p2", [(1, "turn", 10, 11), (2, "turn", 12, 13)])
    filled = align_opponent_actions(a, b)
    # a learns b's same-turn actions (the opponent's own action IS the opp_a/opp_b target)
    assert (a.transitions[0].opp_a_action, a.transitions[0].opp_b_action) == (10, 11)
    assert (a.transitions[1].opp_a_action, a.transitions[1].opp_b_action) == (12, 13)
    # b learns a's actions; a passed slot 1 on turn 2 -> b keeps the sentinel there
    assert (b.transitions[0].opp_a_action, b.transitions[0].opp_b_action) == (3, 5)
    assert b.transitions[1].opp_a_action == 7
    assert b.transitions[1].opp_b_action == PASS_ACTION
    # filled = a-side 4 (10,11,12,13) + b-side 3 (3,5,7); the PASS slot is NOT counted
    assert filled == 7


def test_unmatched_turn_leaves_sentinel():
    a = _traj("g", "p1", [(1, "turn", 3, 4), (5, "turn", 1, 2)])   # turn 5 has no opp step
    b = _traj("g", "p2", [(1, "turn", 8, 9)])
    align_opponent_actions(a, b)
    assert (a.transitions[0].opp_a_action, a.transitions[0].opp_b_action) == (8, 9)
    assert (a.transitions[1].opp_a_action, a.transitions[1].opp_b_action) == \
        (PASS_ACTION, PASS_ACTION)


def test_decision_type_must_match():
    # same turn but turn-decision vs replacement-decision must NOT cross-align
    a = _traj("g", "p1", [(1, "turn", 3, 4)])
    b = _traj("g", "p2", [(1, "replacement", 8, 9)])
    filled = align_opponent_actions(a, b)
    assert filled == 0
    assert (a.transitions[0].opp_a_action, a.transitions[0].opp_b_action) == \
        (PASS_ACTION, PASS_ACTION)
    assert (b.transitions[0].opp_a_action, b.transitions[0].opp_b_action) == \
        (PASS_ACTION, PASS_ACTION)


def test_matching_replacement_steps_align():
    a = _traj("g", "p1", [(2, "replacement", 14, PASS_ACTION)])
    b = _traj("g", "p2", [(2, "replacement", 15, PASS_ACTION)])
    filled = align_opponent_actions(a, b)
    assert a.transitions[0].opp_a_action == 15
    assert b.transitions[0].opp_a_action == 14
    assert a.transitions[0].opp_b_action == PASS_ACTION   # both passed slot 1
    assert filled == 2


def test_align_is_idempotent():
    a = _traj("g", "p1", [(1, "turn", 3, 4)])
    b = _traj("g", "p2", [(1, "turn", 8, 9)])
    first = align_opponent_actions(a, b)
    again = align_opponent_actions(a, b)   # re-running overwrites with the SAME values
    assert first == again == 4             # 2 slots filled each direction; count is the writes
    assert (a.transitions[0].opp_a_action, a.transitions[0].opp_b_action) == (8, 9)


# ── align_paired_trajectories ─────────────────────────────────────────────────
def test_paired_alignment_matches_by_tag_and_ignores_unpaired():
    our = {"gA": _traj("gA", "p1", [(1, "turn", 1, 2)]),
           "gB": _traj("gB", "p1", [(1, "turn", 5, 6)])}     # gB has no opp pair
    opp = {"gA": _traj("gA", "p2", [(1, "turn", 3, 4)])}
    total = align_paired_trajectories(our, opp)
    assert total == 4   # gA aligned both ways (2 + 2)
    assert (our["gA"].transitions[0].opp_a_action,
            our["gA"].transitions[0].opp_b_action) == (3, 4)
    assert (opp["gA"].transitions[0].opp_a_action,
            opp["gA"].transitions[0].opp_b_action) == (1, 2)
    # gB present only on our side -> untouched sentinel
    assert (our["gB"].transitions[0].opp_a_action,
            our["gB"].transitions[0].opp_b_action) == (PASS_ACTION, PASS_ACTION)


# ── schema: opp fields round-trip + back-compat ───────────────────────────────
def test_opp_fields_default_to_sentinel():
    t = Transition(state=np.zeros(3, dtype=np.float32), action_s0=0, action_s1=0)
    assert (t.opp_a_action, t.opp_b_action) == (PASS_ACTION, PASS_ACTION)


def test_opp_fields_round_trip_through_obj():
    t = Transition(state=np.zeros(3, dtype=np.float32), action_s0=1, action_s1=2,
                   opp_a_action=7, opp_b_action=PASS_ACTION, turn=1)
    back = Transition.from_obj(t.to_obj())
    assert (back.opp_a_action, back.opp_b_action) == (7, PASS_ACTION)


def test_old_jsonl_without_opp_keys_decodes_to_sentinel():
    t = Transition(state=np.zeros(3, dtype=np.float32), action_s0=1, action_s1=2,
                   opp_a_action=7, opp_b_action=8)
    old = t.to_obj()
    del old["opp_a_action"]
    del old["opp_b_action"]
    decoded = Transition.from_obj(old)   # pre-Piece-3 store row
    assert (decoded.opp_a_action, decoded.opp_b_action) == (PASS_ACTION, PASS_ACTION)
