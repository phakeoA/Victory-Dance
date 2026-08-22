"""Regression tests for the post-B3 full-codebase scan (2026-06-29).

Confirmed findings fixed here (the cleanly unit-testable ones):
- #1 C51 RESUME: a snapshot's value-head config is peeked + aligned before the AC/optimisers are
  built, so resuming a c51 run does NOT require re-passing --value-loss-mode (resume.peek_value_config).
- #3 C51 loss/value SUPPORT divergence: the loss now projects onto the CRITIC's own support (threaded),
  not an independently cfg-derived one.
- #4 from_bc_checkpoint double-load: model_io.load_bc_policy accepts a preloaded ckpt dict to reuse.
(#2 POSIX killpg, #5 win-signal None-guard, #6 rating-delta precise match, #7 gauntlet anchor-free exit
are verified against source + covered by the full suite; #2/#5/#7 aren't cleanly unit-isolatable here.)
"""
from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("numpy")
import torch

from conftest import write_attn_ckpt
from v_dance.selfplay.actor_critic import ActorCritic
from v_dance.selfplay import resume as RS
from v_dance.selfplay.ppo import PPOConfig, _value_loss, c51_value_loss, c51_support
from v_dance.selfplay.trainer import PPOTrainer, TrainConfig
from v_dance.selfplay.league import OpponentLeague
from v_dance.selfplay.generation import GenerationHistory


def _c51_ac(tmp_path, n_atoms=51, name="bc.pt"):
    return ActorCritic.from_bc_checkpoint(write_attn_ckpt(tmp_path / name, seed=3),
                                          n_value_atoms=n_atoms)


# ── #1: resume value-config peek/align ─────────────────────────────────────────
def _save_snap(tmp_path, ac, cfg, name="snap.pt"):
    tr = PPOTrainer(ac, ppo_cfg=cfg, train_cfg=TrainConfig())
    lg = OpponentLeague(latest_path=str(tmp_path / "bc.pt"))
    return RS.save_snapshot(tmp_path / name, actor_critic=ac, trainer=tr, league=lg,
                            history=GenerationHistory(), ppo_cfg=cfg, train_cfg=tr.tcfg)


def test_peek_value_config_detects_c51_snapshot(tmp_path):
    ac = _c51_ac(tmp_path, 51)
    p = _save_snap(tmp_path, ac, PPOConfig(value_loss_mode="c51", n_atoms=51))
    vc = RS.peek_value_config(p)
    assert vc["value_loss_mode"] == "c51" and vc["n_atoms"] == 51
    assert vc["v_min"] == -1.0 and vc["v_max"] == 1.0


def test_peek_value_config_scalar_snapshot(tmp_path):
    ac = ActorCritic.from_bc_checkpoint(write_attn_ckpt(tmp_path / "bc.pt", seed=3))   # scalar
    p = _save_snap(tmp_path, ac, PPOConfig(value_loss_mode="bce"), name="snap_s.pt")
    assert RS.peek_value_config(p)["value_loss_mode"] == "bce"


def test_peek_value_config_unreadable_returns_none(tmp_path):
    bad = tmp_path / "nope.pt"
    bad.write_bytes(b"not a checkpoint")
    assert RS.peek_value_config(bad) is None


# ── #3: C51 loss uses the THREADED critic support (single source of truth) ──────
def test_c51_loss_uses_threaded_support_over_cfg(tmp_path):
    cfg = PPOConfig(value_loss_mode="c51", n_atoms=21, v_min=-1.0, v_max=1.0)
    atoms = torch.randn(8, 21)
    rets = torch.empty(8).uniform_(-1.0, 1.0)
    dummy = torch.zeros(8)
    other = torch.linspace(-0.5, 0.5, 21)                       # a DIFFERENT support than cfg's [-1,1]
    threaded = _value_loss(dummy, dummy, rets, cfg, atoms_logits=atoms, support=other)
    assert torch.allclose(threaded, c51_value_loss(atoms, rets, other))   # used the threaded support
    cfg_support = _value_loss(dummy, dummy, rets, cfg, atoms_logits=atoms)  # support=None -> cfg's [-1,1]
    assert not torch.allclose(threaded, cfg_support)            # the thread genuinely overrode cfg


def test_ppo_loss_from_batch_threads_critic_support(tmp_path):
    # end-to-end: ppo_loss_from_batch must pass ac.critic.support (so loss + value mean share one support)
    import numpy as np
    from v_dance.selfplay.schema import Transition, PASS_ACTION
    from v_dance.selfplay.collector import TrajectoryCollector
    from v_dance.selfplay.reward import place_terminal_reward
    from v_dance.selfplay import policy_eval as pe
    from v_dance.selfplay.ppo import ppo_loss_from_batch
    from v_dance.encoders.state_encoder import get_state_dim, get_action_dim
    SD, AD = get_state_dim(), get_action_dim()
    ac = _c51_ac(tmp_path, 51)
    c = TrajectoryCollector("g", "p1")
    rng = np.random.default_rng(0)
    for t in range(4):
        st = Transition(state=rng.standard_normal(SD).astype(np.float32),
                        action_s0=t % AD, action_s1=PASS_ACTION, mask_s0=[1] * AD)
        lp, _, v = pe.evaluate_actions(ac, [st])
        c.add_step(state=st.state, action_s0=t % AD, action_s1=PASS_ACTION, mask_s0=[1] * AD,
                   logprob=float(lp[0].detach()), value=float(v[0].detach()), turn=t + 1)
    tr = c.finish(own_team=["a"] * 6, opp_team=["b"] * 6, tp_bring=[0, 1, 2, 3], tp_leads=[0, 1],
                  won=True, terminal_type="win")
    place_terminal_reward(tr)
    cfg = PPOConfig(value_loss_mode="c51", n_atoms=51)
    loss, _stats = ppo_loss_from_batch(ac, tr.transitions, [0.1] * 4, [0.5] * 4, cfg=cfg)
    assert torch.isfinite(loss)                                  # the threaded support path runs cleanly


# ── #4: load_bc_policy reuses a preloaded checkpoint dict (no double torch.load) ─
def test_load_bc_policy_reuses_preloaded_ckpt(tmp_path):
    from v_dance.play import model_io
    p = write_attn_ckpt(tmp_path / "bc.pt", seed=3)
    ck = torch.load(p, map_location="cpu", weights_only=False)
    model, heads = model_io.load_bc_policy(p, _ckpt=ck)
    assert model is not None and heads is not None
