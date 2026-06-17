"""Task 3b.6: the single value-space invariant (everything is value_pm ∈ [-1,1]).

Pure-numpy checks of the range backstop, the GAE identity, the sparse return bound,
and the win-prob distribution heuristic — plus a torch integration test that the
trainer's per-update backstop catches a corrupted (non-pm) value.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
from v_dance.selfplay.schema import Transition, PASS_ACTION  # noqa: E402
from v_dance.selfplay import value_space as vs  # noqa: E402


def _tx(value):
    return Transition(state=np.zeros(3, np.float32), action_s0=0, action_s1=PASS_ACTION,
                      value=float(value))


# ── range backstop ────────────────────────────────────────────────────────────
def test_in_pm_range_and_assert():
    assert vs.in_pm_range([-1.0, 0.0, 1.0])
    assert not vs.in_pm_range([0.0, 1.5])
    vs.assert_value_pm([-1.0, 0.3, 0.999])                 # no raise
    with pytest.raises(AssertionError, match="value_pm range"):
        vs.assert_value_pm([0.0, 1.5])
    with pytest.raises(AssertionError, match="value_pm range"):
        vs.assert_value_pm([-2.0])                          # raw-logit-like


def test_assert_transitions_value_space():
    vs.assert_transitions_value_space([_tx(-0.5), _tx(0.0), _tx(0.9)])
    with pytest.raises(AssertionError, match="Transition.value"):
        vs.assert_transitions_value_space([_tx(0.5), _tx(2.0)])
    vs.assert_transitions_value_space([])                   # empty ok


# ── GAE identity + sparse bound ───────────────────────────────────────────────
def test_assert_gae_identity():
    adv = np.array([0.5, -0.2]); val = np.array([0.3, 0.4]); ret = adv + val
    vs.assert_gae_identity(adv, ret, val)                   # holds
    with pytest.raises(AssertionError, match="returns == advantages"):
        vs.assert_gae_identity(adv, ret + 0.1, val)         # broken identity
    with pytest.raises(AssertionError, match="shape mismatch"):
        vs.assert_gae_identity(adv, np.array([0.0]), val)


def test_assert_gae_value_space_sparse_bound():
    adv = np.array([0.5, -0.5]); val = np.array([0.4, -0.4]); ret = adv + val
    vs.assert_gae_value_space(adv, ret, val)                # |ret| <= 1
    big_val = np.array([0.9, 0.9]); big_ret = adv + big_val  # ret[0] = 1.4 > 1
    with pytest.raises(AssertionError, match="GAE returns"):
        vs.assert_gae_value_space(adv, big_ret, big_val)
    vs.assert_gae_value_space(adv, big_ret, big_val, allow_shaping=True)  # PBRS skips bound


# ── win-prob distribution heuristic ───────────────────────────────────────────
def test_looks_like_winprob():
    rng = np.random.default_rng(0)
    winprob_batch = rng.uniform(0.0, 1.0, size=200)         # all >= 0 -> suspicious
    assert vs.looks_like_winprob(winprob_batch) is True
    pm_batch = rng.uniform(-1.0, 1.0, size=200)             # has negatives -> fine
    assert vs.looks_like_winprob(pm_batch) is False
    assert vs.looks_like_winprob(np.array([0.1, 0.2, 0.3])) is False   # too small to judge


def test_value_space_report():
    rep = vs.value_space_report([_tx(-0.4), _tx(0.2), _tx(0.6)])
    assert rep["n"] == 3
    assert rep["value_min"] == pytest.approx(-0.4)
    assert rep["frac_negative"] == pytest.approx(1 / 3)
    assert rep["looks_like_winprob"] is False               # tiny batch never flagged


# ── trainer integration (torch) ───────────────────────────────────────────────
def test_trainer_backstop_catches_corrupted_value(tmp_path):
    pytest.importorskip("torch")
    import torch
    from v_dance.encoders.state_encoder import (get_state_dim, get_action_dim, get_gimmick_dim,
                               get_state_layout_version)
    from v_dance.models.bc_model import BCPolicy
    from v_dance.selfplay.actor_critic import ActorCritic
    from v_dance.selfplay.collector import TrajectoryCollector
    from v_dance.selfplay.reward import place_terminal_reward
    from v_dance.selfplay.trainer import PPOTrainer, TrainConfig

    sd, ad = get_state_dim(), get_action_dim()
    torch.manual_seed(0)
    m = BCPolicy(state_dim=sd, action_dim=ad, hidden_dims=(8, 4), dropout=0.0,
                 gimmick_dim=get_gimmick_dim())
    p = tmp_path / "bc.pt"
    torch.save({"model_state": m.state_dict(), "config": {
        "state_dim": sd, "action_dim": ad, "hidden_dims": (8, 4), "dropout": 0.0,
        "heads": ("our_a", "our_b"), "gimmick_dim": get_gimmick_dim(),
        "gimmick_trained": True, "value_trained": True,
        "state_layout_version": get_state_layout_version()}}, p)
    ac = ActorCritic.from_bc_checkpoint(p)

    c = TrajectoryCollector("g", "p1")
    for t in range(3):
        c.add_step(state=np.random.default_rng(t).standard_normal(sd).astype(np.float32),
                   action_s0=0, action_s1=PASS_ACTION, mask_s0=[1] * ad,
                   logprob=-1.0, value=0.1, turn=t + 1)
    tr = c.finish(own_team=["a"] * 6, opp_team=["b"] * 6, tp_bring=[0, 1, 2, 3],
                  tp_leads=[0, 1], won=True, terminal_type="win")
    place_terminal_reward(tr)
    tr.transitions[1].value = 5.0                           # corrupt: out of value_pm range

    trainer = PPOTrainer(ac, train_cfg=TrainConfig(minibatch_size=0, ppo_epochs=1))
    with pytest.raises(AssertionError, match="value_pm"):
        trainer.ppo_update([tr])
