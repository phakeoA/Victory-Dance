"""Task 3a.1: PPO self-play trajectory schema — round-trip + terminal-reward logic."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")
import numpy as np

_REPO = Path(__file__).resolve().parents[3]
for _p in (str(_REPO / "local_battle"), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from self_play.schema import (  # noqa: E402
    Transition, EpisodeMeta, Trajectory, TerminalType, PASS_ACTION,
)


def _traj(terminal_type="win", won=True):
    meta = EpisodeMeta(
        battle_id="b-1", own_role="p1",
        own_team=["zoroark", "ditto", "charizard", "whimsicott", "sneasler", "basculegion"],
        opp_team=["a", "b", "c", "d", "e", "f"],
        tp_bring=[0, 1, 2, 3], tp_leads=[0, 1],
        won=won, terminal_type=terminal_type, n_turns=2,
    )
    t0 = Transition(state=np.arange(8, dtype=np.float32), action_s0=3, action_s1=PASS_ACTION,
                    gimmick_s0=1, logprob=-0.7, value=0.55, reward=0.0, turn=1)
    t1 = Transition(state=np.full(8, 0.5, dtype=np.float32), action_s0=12, action_s1=4,
                    logprob=-1.2, value=0.8, reward=1.0, done=True,
                    decision_type="replacement", turn=2)
    return Trajectory(meta=meta, transitions=[t0, t1])


def test_trajectory_round_trips_through_obj():
    tr = _traj()
    back = Trajectory.from_obj(tr.to_obj())
    assert back.meta == tr.meta                       # dataclass eq (no ndarray in meta)
    assert len(back.transitions) == 2
    for a, b in zip(tr.transitions, back.transitions):
        assert np.allclose(a.state, b.state)
        assert a.state.dtype == np.float32
        assert (a.action_s0, a.action_s1, a.gimmick_s0, a.gimmick_s1) == \
               (b.action_s0, b.action_s1, b.gimmick_s0, b.gimmick_s1)
        assert (a.logprob, a.value, a.reward, a.done, a.decision_type, a.turn) == \
               (b.logprob, b.value, b.reward, b.done, b.decision_type, b.turn)


def test_pass_sentinel_preserved():
    assert PASS_ACTION == -1
    assert Trajectory.from_obj(_traj().to_obj()).transitions[0].action_s1 == PASS_ACTION


def test_outcome_reward_by_terminal_type():
    assert _traj("win", True).meta.outcome_reward() == 1.0
    assert _traj("loss", False).meta.outcome_reward() == -1.0
    assert _traj("adjudicated", True).meta.outcome_reward() == 1.0    # server picked us
    assert _traj("adjudicated", False).meta.outcome_reward() == -1.0  # server picked opp
    assert _traj("draw", None).meta.outcome_reward() == 0.0           # real terminal 0, no bootstrap
    assert _traj("horizon_cut", None).meta.outcome_reward() is None   # bootstrap instead
    assert _traj("fallback", None).meta.outcome_reward() is None      # discarded


def test_is_trainable_drops_only_fallback():
    assert _traj("fallback", None).meta.is_trainable is False
    for tt in ("win", "loss", "draw", "adjudicated", "horizon_cut"):
        assert _traj(tt, None).meta.is_trainable is True


def test_bootstraps_only_on_horizon_cut():
    assert _traj("horizon_cut", None).meta.bootstraps is True
    for tt in ("win", "loss", "draw", "adjudicated", "fallback"):
        assert _traj(tt, None).meta.bootstraps is False


def test_terminal_type_enum_values():
    assert {t.value for t in TerminalType} == {
        "win", "loss", "draw", "adjudicated", "horizon_cut", "fallback"}
