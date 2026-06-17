"""Task 3a.2: two-perspective collector + zero-sum symmetry guard."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
from v_dance.selfplay.collector import TrajectoryCollector, assert_zero_sum  # noqa: E402


def _collect(battle_id, role, won, terminal_type, n_steps=2):
    c = TrajectoryCollector(battle_id, role)
    for t in range(1, n_steps + 1):
        c.add_step(state=np.full(6, float(t), dtype=np.float32),
                   action_s0=t, action_s1=-1, gimmick_s0=0,
                   logprob=-0.5 * t, value=0.5, turn=t)
    return c.finish(own_team=["a"] * 6, opp_team=["b"] * 6,
                    tp_bring=[0, 1, 2, 3], tp_leads=[0, 1],
                    won=won, terminal_type=terminal_type)


def test_finish_marks_last_done_and_infers_turns():
    tr = _collect("g1", "p1", True, "win", n_steps=3)
    assert len(tr.transitions) == 3
    assert tr.transitions[-1].done is True
    assert all(not t.done for t in tr.transitions[:-1])
    assert tr.meta.n_turns == 3                      # inferred from last step's turn
    assert tr.transitions[-1].reward == 0.0          # 3a.2 places NO reward (that's 3a.3)


def test_zero_sum_passes_for_win_loss_pair():
    us = _collect("g", "p1", True, "win")
    opp = _collect("g", "p2", False, "loss")
    assert_zero_sum(us, opp)                          # +1 + (-1) == 0


def test_zero_sum_passes_for_draw_and_adjudicated_and_horizon():
    assert_zero_sum(_collect("g", "p1", None, "draw"),
                    _collect("g", "p2", None, "draw"))            # 0 + 0
    assert_zero_sum(_collect("g", "p1", True, "adjudicated"),
                    _collect("g", "p2", False, "adjudicated"))    # +1 + -1
    assert_zero_sum(_collect("g", "p1", None, "horizon_cut"),
                    _collect("g", "p2", None, "horizon_cut"))     # None/None (bootstrap)


def test_zero_sum_raises_on_same_role():
    with pytest.raises(AssertionError, match="role"):
        assert_zero_sum(_collect("g", "p1", True, "win"),
                        _collect("g", "p1", False, "loss"))


def test_zero_sum_raises_on_battle_id_mismatch():
    with pytest.raises(AssertionError, match="battle_id"):
        assert_zero_sum(_collect("g1", "p1", True, "win"),
                        _collect("g2", "p2", False, "loss"))


def test_zero_sum_raises_on_turn_clock_mismatch():
    with pytest.raises(AssertionError, match="turn-clock"):
        assert_zero_sum(_collect("g", "p1", True, "win", n_steps=2),
                        _collect("g", "p2", False, "loss", n_steps=3))


def test_zero_sum_raises_when_both_win():
    with pytest.raises(AssertionError, match="zero-sum"):
        assert_zero_sum(_collect("g", "p1", True, "win"),
                        _collect("g", "p2", True, "win"))         # +1 + +1 = 2


def test_zero_sum_raises_on_outcome_presence_mismatch():
    with pytest.raises(AssertionError, match="presence mismatch"):
        assert_zero_sum(_collect("g", "p1", True, "win"),         # +1
                        _collect("g", "p2", None, "horizon_cut"))  # None
