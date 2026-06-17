"""Task 3a.4: Type-C self-play store (Trajectory <-> jsonl) + reward-integrity guard."""
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

from self_play.collector import TrajectoryCollector  # noqa: E402
from self_play.reward import place_terminal_reward  # noqa: E402
from self_play.store import (  # noqa: E402
    TrajectoryStore, write_trajectories, read_trajectories, iter_trajectories,
    assert_terminal_rewards_clean,
)

STATE_DIM = 5


def _traj(battle_id, role, won, terminal_type, n=3, place=True):
    c = TrajectoryCollector(battle_id, role)
    for t in range(1, n + 1):
        c.add_step(state=np.full(STATE_DIM, float(t), dtype=np.float32),
                   action_s0=t, action_s1=-1, gimmick_s0=1, logprob=-0.3 * t,
                   value=0.4 + 0.1 * t, turn=t)
    tr = c.finish(own_team=["s%d" % i for i in range(6)], opp_team=["o"] * 6,
                  tp_bring=[0, 1, 2, 3], tp_leads=[0, 1], won=won, terminal_type=terminal_type)
    return place_terminal_reward(tr) if place else tr


def test_round_trip_write_read(tmp_path):
    p = tmp_path / "buf.jsonl"
    src = [_traj("g", "p1", True, "win"), _traj("g", "p2", False, "loss")]
    assert write_trajectories(p, src) == 2
    back = read_trajectories(p)
    assert len(back) == 2
    for a, b in zip(src, back):
        assert a.meta == b.meta
        assert len(a.transitions) == len(b.transitions)
        for x, y in zip(a.transitions, b.transitions):
            assert np.allclose(x.state, y.state) and y.state.dtype == np.float32
            assert (x.action_s0, x.gimmick_s0, x.logprob, x.value, x.reward, x.done) == \
                   (y.action_s0, y.gimmick_s0, y.logprob, y.value, y.reward, y.done)


def test_streaming_append_and_iter(tmp_path):
    p = tmp_path / "buf.jsonl"
    with TrajectoryStore(p) as s:
        s.append(_traj("g1", "p1", True, "win"))
        s.append(_traj("g2", "p1", False, "loss"))
        assert s.n_written == 2
    ids = [t.meta.battle_id for t in iter_trajectories(p)]
    assert ids == ["g1", "g2"]


def test_missing_file_reads_empty(tmp_path):
    assert read_trajectories(tmp_path / "nope.jsonl") == []


def test_state_dim_validation_raises(tmp_path):
    p = tmp_path / "buf.jsonl"
    write_trajectories(p, [_traj("g", "p1", True, "win")])
    assert len(read_trajectories(p, expected_state_dim=STATE_DIM)) == 1
    with pytest.raises(AssertionError, match="state dim"):
        read_trajectories(p, expected_state_dim=STATE_DIM + 1)


def test_terminal_rewards_clean_passes_on_placed(tmp_path):
    assert_terminal_rewards_clean([_traj("g", "p1", True, "win"),
                                   _traj("g", "p2", None, "draw"),
                                   _traj("g", "p1", None, "horizon_cut")])


def test_terminal_rewards_clean_catches_nonterminal_reward():
    tr = _traj("g", "p1", True, "win")
    tr.transitions[0].reward = 0.5            # simulate a stray/normalized reward
    with pytest.raises(AssertionError, match="non-terminal reward"):
        assert_terminal_rewards_clean([tr])


def test_terminal_rewards_clean_catches_out_of_range_terminal():
    tr = _traj("g", "p1", True, "win")
    tr.transitions[-1].reward = 5.0           # simulate reward blow-up / bad scale
    with pytest.raises(AssertionError, match="outside"):
        assert_terminal_rewards_clean([tr])
