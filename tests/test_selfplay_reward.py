"""Task 3a.3: terminal reward placement + MODEL-DRIVEN% hard-fail."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
from v_dance.selfplay.collector import TrajectoryCollector  # noqa: E402
from v_dance.selfplay.reward import (  # noqa: E402
    place_terminal_reward, model_driven_fraction, assert_model_driven, prepare_batch,
    drop_fallback_pairs,
)


def _traj_tag(tag, terminal_type, won, role):
    c = TrajectoryCollector(tag, role)
    c.add_step(state=np.zeros(4, np.float32), action_s0=1, action_s1=-1, turn=1, value=0.5)
    return c.finish(own_team=["a"] * 6, opp_team=["b"] * 6, tp_bring=[0, 1, 2, 3],
                    tp_leads=[0, 1], won=won, terminal_type=terminal_type)


def _traj(terminal_type, won, role="p1", n=3):
    c = TrajectoryCollector("g", role)
    for t in range(1, n + 1):
        c.add_step(state=np.zeros(4, np.float32), action_s0=t, action_s1=-1, turn=t, value=0.5)
    return c.finish(own_team=["a"] * 6, opp_team=["b"] * 6, tp_bring=[0, 1, 2, 3],
                    tp_leads=[0, 1], won=won, terminal_type=terminal_type)


def test_terminal_reward_on_last_step_only():
    for tt, won, exp in [("win", True, 1.0), ("loss", False, -1.0),
                         ("draw", None, 0.0), ("adjudicated", True, 1.0),
                         ("adjudicated", False, -1.0)]:
        tr = place_terminal_reward(_traj(tt, won))
        assert tr.transitions[-1].reward == exp, (tt, won)
        assert all(t.reward == 0.0 for t in tr.transitions[:-1])   # earlier steps sparse


def test_horizon_cut_leaves_zero_and_flags_bootstrap():
    tr = place_terminal_reward(_traj("horizon_cut", None))
    assert tr.transitions[-1].reward == 0.0      # no +-1 terminal
    assert tr.meta.bootstraps is True            # GAE bootstraps gamma*V(s_cut)


def test_place_reward_rejects_fallback():
    with pytest.raises(AssertionError, match="FALLBACK"):
        place_terminal_reward(_traj("fallback", None))


def test_place_reward_empty_is_noop():
    c = TrajectoryCollector("g", "p1")
    tr = c.finish(own_team=["a"] * 6, opp_team=["b"] * 6, tp_bring=[], tp_leads=[],
                  won=True, terminal_type="win")
    assert place_terminal_reward(tr).transitions == []   # no crash


def test_drop_fallback_pairs_drops_both_perspectives():
    # T3.1: battle B1 — p1 FORFEITED as a backstop (fallback); p2 sees a mislabeled +1 win. BOTH
    # perspectives must be dropped by battle_id. Battle B2 — a normal game — both kept.
    batch = [
        _traj_tag("B1", "fallback", None, "p1"),
        _traj_tag("B1", "win", True, "p2"),
        _traj_tag("B2", "win", True, "p1"),
        _traj_tag("B2", "loss", False, "p2"),
    ]
    kept = drop_fallback_pairs(batch)
    assert {t.meta.battle_id for t in kept} == {"B2"}
    assert len(kept) == 2


def test_drop_fallback_pairs_noop_without_fallback_returns_same_list():
    trs = [_traj_tag("X", "win", True, "p1"), _traj_tag("X", "loss", False, "p2")]
    assert drop_fallback_pairs(trs) is trs        # early return when nothing to drop


def test_drop_fallback_pairs_keeps_unclassifiable_stubs():
    stub = object()                                # no .meta → conservatively KEPT
    assert drop_fallback_pairs([stub]) == [stub]


def test_model_driven_fraction_excludes_tp_and_handles_empty():
    assert model_driven_fraction({}) == 1.0
    sc = {"model": 90, "forced_switch_model": 9, "retry": 1, "tp_model": 30}
    assert model_driven_fraction(sc) == pytest.approx(0.99)   # tp_model excluded; 99/100


def test_assert_model_driven_threshold():
    assert assert_model_driven({"model": 100}, 0.99) == 1.0
    assert assert_model_driven({"model": 99, "retry": 1}, 0.99) == pytest.approx(0.99)
    with pytest.raises(AssertionError, match="MODEL-DRIVEN"):
        assert_model_driven({"model": 95, "forced_switch": 5}, 0.99)   # 0.95 < 0.99


def test_prepare_batch_drops_fallback_and_places_rewards():
    batch = [_traj("win", True), _traj("loss", False, role="p2"),
             _traj("fallback", None)]
    kept = prepare_batch(batch, source_counts={"model": 100})
    assert len(kept) == 2                                  # fallback dropped
    assert {t.transitions[-1].reward for t in kept} == {1.0, -1.0}


def test_prepare_batch_hard_fails_on_low_model_driven():
    with pytest.raises(AssertionError, match="MODEL-DRIVEN"):
        prepare_batch([_traj("win", True)],
                      source_counts={"model": 90, "retry": 10})   # 0.90 < 0.99
