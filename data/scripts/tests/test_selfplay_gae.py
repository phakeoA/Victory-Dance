"""Task 3b.2: GAE advantage/return computation + advantage standardization."""
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
from self_play.gae import compute_gae, standardize, compute_batch_gae, DEFAULT_GAMMA, DEFAULT_LAM  # noqa: E402


def _traj(terminal_type, won, values, rewards=None, n=None):
    n = n or len(values)
    c = TrajectoryCollector("g", "p1")
    for t in range(n):
        c.add_step(state=np.zeros(3, np.float32), action_s0=0, action_s1=-1,
                   value=float(values[t]), turn=t + 1)
    tr = c.finish(own_team=["a"] * 6, opp_team=["b"] * 6, tp_bring=[0, 1, 2, 3],
                  tp_leads=[0, 1], won=won, terminal_type=terminal_type)
    place_terminal_reward(tr)
    if rewards is not None:                       # override for hand-calc cases
        for i, r in enumerate(rewards):
            tr.transitions[i].reward = float(r)
    return tr


def test_default_gamma_is_floor():
    assert DEFAULT_GAMMA == 0.997 and DEFAULT_LAM == 0.95


def test_single_step_terminal():
    tr = _traj("win", True, values=[0.5])         # reward +1 placed on the one step
    adv, ret = compute_gae(tr, gamma=1.0, lam=1.0)
    assert adv == pytest.approx([0.5])             # 1 - V = 0.5, no bootstrap (terminal)
    assert ret == pytest.approx([1.0])             # adv + V


def test_two_step_win_undiscounted_handcalc():
    # rewards [0, 1], values [0.5, 0.5], gamma=lam=1 -> both steps lead to the +1 win
    tr = _traj("win", True, values=[0.5, 0.5])     # reward auto: [0, +1]
    adv, ret = compute_gae(tr, gamma=1.0, lam=1.0)
    assert adv == pytest.approx([0.5, 0.5])
    assert ret == pytest.approx([1.0, 1.0])        # return == adv + value


def test_returns_equal_adv_plus_values():
    tr = _traj("loss", False, values=[0.3, 0.6, 0.2])
    adv, ret = compute_gae(tr)
    vals = np.array([t.value for t in tr.transitions], dtype=np.float32)
    assert np.allclose(ret, adv + vals)


def test_horizon_cut_bootstraps_with_given_value():
    tr = _traj("horizon_cut", None, values=[0.5])  # reward 0 (no terminal), bootstraps
    adv, ret = compute_gae(tr, gamma=1.0, lam=1.0, bootstrap_value=0.8)
    assert adv == pytest.approx([0.3])             # 0 + 1*0.8 - 0.5
    assert ret == pytest.approx([0.8])             # continuation value


def test_horizon_cut_proxy_when_no_bootstrap_value():
    tr = _traj("horizon_cut", None, values=[0.5])
    adv, _ = compute_gae(tr, gamma=1.0, lam=1.0)   # proxy = last value 0.5
    assert adv == pytest.approx([0.0])             # 0 + 1*0.5 - 0.5


def test_terminal_vs_bootstrap_differ():
    a_term, _ = compute_gae(_traj("win", True, values=[0.5]), gamma=1.0)
    a_boot, _ = compute_gae(_traj("horizon_cut", None, values=[0.5]),
                            gamma=1.0, bootstrap_value=0.8)
    assert a_term[0] != a_boot[0]


def test_standardize_zero_mean_unit_std():
    out = standardize(np.array([1.0, 2.0, 3.0, 4.0], np.float32))
    assert out.mean() == pytest.approx(0.0, abs=1e-5)
    assert out.std() == pytest.approx(1.0, abs=1e-3)
    assert standardize(np.zeros(0, np.float32)).size == 0


def test_empty_trajectory_yields_empty():
    c = TrajectoryCollector("g", "p1")
    tr = c.finish(own_team=["a"] * 6, opp_team=["b"] * 6, tp_bring=[], tp_leads=[],
                  won=True, terminal_type="win")
    adv, ret = compute_gae(tr)
    assert adv.size == 0 and ret.size == 0


def test_compute_batch_gae_concatenates_and_standardizes():
    t1 = _traj("win", True, values=[0.5, 0.5])
    t2 = _traj("loss", False, values=[0.4, 0.4, 0.4])
    adv, ret = compute_batch_gae([t1, t2])
    assert adv.shape == (5,) and ret.shape == (5,)   # 2 + 3 steps concatenated
    assert adv.mean() == pytest.approx(0.0, abs=1e-5)  # standardized across the batch
    # without standardization, returns are unchanged scale
    adv_raw, _ = compute_batch_gae([t1, t2], standardize_adv=False)
    assert adv_raw.shape == (5,)
