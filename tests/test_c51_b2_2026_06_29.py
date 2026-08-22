"""B2 tests for the C51 distributional value head — END-TO-END wiring (docs/c51_value_head_design.md).

B1 built the math/arch (off by default). B2 wires it: from_bc_checkpoint builds the c51 critic
(atoms head + support + scalar-tied warm-start), ppo_forward/evaluate_actions surface the distribution
mean + atoms_logits, ppo_losses/_value_loss + warmup_critic consume them. These tests drive a REAL
c51 ActorCritic through ppo_forward / warm-up / ppo_update on synthetic trajectories.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("numpy")
import numpy as np
import torch

from conftest import write_attn_ckpt
from v_dance.encoders.state_encoder import get_state_dim, get_action_dim
from v_dance.selfplay.actor_critic import ActorCritic, init_value_atoms_from_scalar
from v_dance.selfplay.schema import Transition, PASS_ACTION
from v_dance.selfplay.collector import TrajectoryCollector
from v_dance.selfplay.reward import place_terminal_reward
from v_dance.selfplay import policy_eval as pe
from v_dance.selfplay.ppo import PPOConfig
from v_dance.selfplay.trainer import PPOTrainer, TrainConfig

STATE_DIM, ACTION_DIM = get_state_dim(), get_action_dim()


def _c51_ac(tmp_path, n_atoms=51, name="bc.pt"):
    return ActorCritic.from_bc_checkpoint(write_attn_ckpt(tmp_path / name, seed=3),
                                          n_value_atoms=n_atoms)


def _scalar_ac(tmp_path, name="bc_s.pt"):
    return ActorCritic.from_bc_checkpoint(write_attn_ckpt(tmp_path / name, seed=3))


def _traj(ac, won=True, n=4, seed=0):
    c = TrajectoryCollector(f"g{seed}", "p1")
    rng = np.random.default_rng(seed)
    for t in range(n):
        st = Transition(state=rng.standard_normal(STATE_DIM).astype(np.float32),
                        action_s0=t % ACTION_DIM, action_s1=PASS_ACTION, mask_s0=[1] * ACTION_DIM)
        lp, _, v = pe.evaluate_actions(ac, [st])
        c.add_step(state=st.state, action_s0=t % ACTION_DIM, action_s1=PASS_ACTION,
                   mask_s0=[1] * ACTION_DIM, logprob=float(lp[0].detach()),
                   value=float(v[0].detach()), turn=t + 1)
    tr = c.finish(own_team=["a"] * 6, opp_team=["b"] * 6, tp_bring=[0, 1, 2, 3],
                  tp_leads=[0, 1], won=won, terminal_type="win" if won else "loss")
    place_terminal_reward(tr)
    return tr


# ── critic construction ───────────────────────────────────────────────────────
def test_from_bc_checkpoint_builds_c51_critic(tmp_path):
    ac = _c51_ac(tmp_path, 51)
    assert ac.critic.is_c51 and ac.critic.support.numel() == 51
    x = torch.randn(8, STATE_DIM)
    with torch.no_grad():
        dist, vpm = ac.critic.value_dist(x), ac.value_pm(x)
    assert dist.shape == (8, 51)
    assert torch.allclose(dist.sum(-1), torch.ones(8), atol=1e-5)
    assert (vpm.abs() <= 1.0 + 1e-5).all()
    # the ACTOR stays scalar (its value path is vestigial) — only the critic got the atoms head
    assert getattr(ac.policy, "value_atoms_head", None) is None
    assert ac.critic.net.value_atoms_head is not None


def test_scalar_critic_default_unchanged(tmp_path):
    ac = _scalar_ac(tmp_path)
    assert not ac.critic.is_c51 and ac.critic.support is None
    x = torch.randn(5, STATE_DIM)
    with torch.no_grad():
        assert torch.allclose(ac.value_pm(x), 2.0 * torch.sigmoid(ac.critic.forward(x)) - 1.0, atol=1e-6)


def test_scalar_tied_init_aligns_dist_mean_with_scalar(tmp_path):
    ac = _c51_ac(tmp_path, 51)
    x = torch.randn(128, STATE_DIM)
    with torch.no_grad():
        scalar = 2.0 * torch.sigmoid(ac.critic.forward(x)) - 1.0      # scalar value_pm
        c51mean = ac.value_pm(x)                                       # distribution mean
    corr = torch.corrcoef(torch.stack([scalar, c51mean]))[0, 1]
    assert float(corr) > 0.8                                          # warm-start ties them
    assert float(c51mean[scalar.argmax()]) > 0                       # most-positive scalar -> +mean
    assert float(c51mean[scalar.argmin()]) < 0                       # most-negative scalar -> -mean


# ── forward wiring ─────────────────────────────────────────────────────────────
def test_ppo_forward_populates_atoms_logits_for_c51(tmp_path):
    ac = _c51_ac(tmp_path, 51)
    tr = _traj(ac, won=True, n=4, seed=1)
    ev = pe.ppo_forward(ac, tr.transitions, tau=1.0)
    assert ev.atoms_logits is not None and ev.atoms_logits.shape[-1] == 51
    assert (ev.value_pm.abs() <= 1.0 + 1e-5).all()


def test_ppo_forward_scalar_has_no_atoms(tmp_path):
    ac = _scalar_ac(tmp_path)
    tr = _traj(ac, won=True, n=4, seed=1)
    ev = pe.ppo_forward(ac, tr.transitions, tau=1.0)
    assert ev.atoms_logits is None


def test_evaluate_actions_records_dist_mean_for_c51(tmp_path):
    ac = _c51_ac(tmp_path, 51)
    tr = _traj(ac, won=True, n=4, seed=1)
    _lp, _ent, v = pe.evaluate_actions(ac, tr.transitions)
    with torch.no_grad():
        ref = ac.critic.value_pm(pe._states_tensor(tr.transitions, "cpu"))
    assert torch.allclose(v, ref, atol=1e-5)
    assert (v.abs() <= 1.0 + 1e-5).all()


# ── end-to-end train loop ──────────────────────────────────────────────────────
def test_c51_warmup_and_update_run_and_move_critic(tmp_path):
    ac = _c51_ac(tmp_path, 51)
    trajs = [_traj(ac, won=True, seed=1), _traj(ac, won=False, seed=2)]
    cfg = PPOConfig(value_loss_mode="c51", n_atoms=51)
    tr = PPOTrainer(ac, ppo_cfg=cfg,
                    train_cfg=TrainConfig(minibatch_size=0, ppo_epochs=1, target_kl_from_bc=None))
    before = [p.detach().clone() for p in ac.critic_parameters()]
    w = tr.warmup_critic(trajs, n_updates=3)                  # regresses the c51 CE (no crash)
    assert math.isfinite(w["value_loss"])
    n_rebased = tr.rebase_values(trajs)                       # value_pm = dist mean, still in [-1,1]
    assert n_rebased == sum(len(t.transitions) for t in trajs)
    st = tr.ppo_update(trajs)                                 # full update through the c51 loss path
    assert st.get("nonfinite_skips", 0) == 0
    after = [p.detach().clone() for p in ac.critic_parameters()]
    assert any(not torch.equal(b, a) for b, a in zip(before, after))   # critic actually moved


def test_c51_warmup_reduces_loss(tmp_path):
    ac = _c51_ac(tmp_path, 51)
    trajs = [_traj(ac, won=True, seed=1), _traj(ac, won=False, seed=2)]
    cfg = PPOConfig(value_loss_mode="c51", n_atoms=51)
    tr = PPOTrainer(ac, ppo_cfg=cfg, train_cfg=TrainConfig(minibatch_size=0))
    first = tr.warmup_critic(trajs, n_updates=1)["value_loss"]
    last = tr.warmup_critic(trajs, n_updates=80)["value_loss"]
    assert last < first                                       # the distributional critic learns
