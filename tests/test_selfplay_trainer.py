"""Task 3b.4: PPO training loop — critic-only warm-up + collapse guards.

Covers explained_variance, the warm-up freezing the actor while improving the critic,
a full PPO update moving both nets, and the two collapse guards (KL-from-BC and
explained-variance auto-halt).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("numpy")
import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
from v_dance.encoders.state_encoder import (get_state_dim, get_action_dim, get_gimmick_dim,  # noqa: E402
                           get_state_layout_version)
from conftest import write_attn_ckpt  # noqa: E402
from v_dance.selfplay.actor_critic import ActorCritic  # noqa: E402
from v_dance.selfplay.schema import Transition, PASS_ACTION  # noqa: E402
from v_dance.selfplay.collector import TrajectoryCollector  # noqa: E402
from v_dance.selfplay.reward import place_terminal_reward  # noqa: E402
from v_dance.selfplay import policy_eval as pe  # noqa: E402
from v_dance.selfplay.ppo import PPOConfig  # noqa: E402
from v_dance.selfplay.trainer import PPOTrainer, TrainConfig, explained_variance  # noqa: E402
from v_dance.selfplay.generation import build_train_configs  # noqa: E402

STATE_DIM, ACTION_DIM, GIMMICK_DIM = get_state_dim(), get_action_dim(), get_gimmick_dim()


def _write_ckpt(path):
    return write_attn_ckpt(path, seed=3)


def _fresh_ac(tmp_path, name="bc.pt"):
    return ActorCritic.from_bc_checkpoint(_write_ckpt(tmp_path / name))


def _traj(ac, won=True, n=4, seed=0):
    """An on-policy trajectory: old logprob/value are this ac's own current outputs."""
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


def _params(it):
    return [p.detach().clone() for p in it]


# ── explained variance ────────────────────────────────────────────────────────
def test_explained_variance():
    r = np.array([1.0, -1.0, 0.5, -0.5])
    assert explained_variance(r, r) == pytest.approx(1.0)               # perfect
    assert explained_variance(np.full_like(r, r.mean()), r) == pytest.approx(0.0, abs=1e-9)
    assert explained_variance(np.array([5.0]), np.array([5.0])) == 0.0  # constant returns
    assert math.isnan(explained_variance(np.array([]), np.array([])))


# ── collapse-revert: optimiser reset ──────────────────────────────────────────
def test_reset_optimizers_clears_adam_moments(tmp_path):
    """On a collapse REVERT the champion WEIGHTS are restored but the Adam first/second-
    moment estimates that drove the collapse must NOT persist (they'd re-apply large steps
    and re-collapse). reset_optimizers() rebuilds fresh optimisers with empty state and the
    configured LRs."""
    ac = _fresh_ac(tmp_path)
    trajs = [_traj(ac, won=True, seed=1), _traj(ac, won=False, seed=2)]
    tr = PPOTrainer(ac, train_cfg=TrainConfig(minibatch_size=0))
    tr.ppo_update(trajs)                                   # populates Adam moment buffers
    assert len(tr.actor_opt.state) > 0 or len(tr.critic_opt.state) > 0
    old_actor_opt = tr.actor_opt
    tr.reset_optimizers()
    assert tr.actor_opt is not old_actor_opt              # fresh optimiser objects
    assert len(tr.actor_opt.state) == 0 and len(tr.critic_opt.state) == 0   # no stale moments
    assert tr.actor_opt.param_groups[0]["lr"] == pytest.approx(tr.tcfg.actor_lr)
    assert tr.critic_opt.param_groups[0]["lr"] == pytest.approx(tr.tcfg.critic_lr)


def test_ppo_update_skips_nonfinite_loss(tmp_path, monkeypatch):
    """NaN guard: a minibatch whose loss is non-finite is SKIPPED (no backward/step), the weights are left
    UNCHANGED, and the skip count is surfaced — so a degenerate batch can't silently poison the weights,
    the Adam moments, or the per-gen snapshots on an unattended run."""
    import v_dance.selfplay.ppo as P
    ac = _fresh_ac(tmp_path)
    trajs = [_traj(ac, won=True, seed=1), _traj(ac, won=False, seed=2)]
    tr = PPOTrainer(ac, train_cfg=TrainConfig(minibatch_size=0, ppo_epochs=1, target_kl_from_bc=None))
    before = [p.detach().clone() for p in ac.actor_parameters()]
    before_c = [p.detach().clone() for p in ac.critic_parameters()]

    monkeypatch.setattr(P, "ppo_loss_from_batch",
                        lambda *a, **k: (torch.tensor(float("nan")), {"loss": float("nan")}))
    stats = tr.ppo_update(trajs)

    assert stats["nonfinite_skips"] >= 1
    assert len(tr.actor_opt.state) == 0 and len(tr.critic_opt.state) == 0   # optimizer never stepped
    assert all(torch.equal(b, p) for b, p in zip(before, ac.actor_parameters())), "actor weights unchanged"
    assert all(torch.equal(b, p) for b, p in zip(before_c, ac.critic_parameters())), "critic weights unchanged"


# ── warm-up ───────────────────────────────────────────────────────────────────
def test_warmup_freezes_actor_moves_critic(tmp_path):
    ac = _fresh_ac(tmp_path)
    trajs = [_traj(ac, won=True, seed=1), _traj(ac, won=False, seed=2)]
    tr = PPOTrainer(ac, ppo_cfg=PPOConfig(value_loss_mode="huber"),
                    train_cfg=TrainConfig(minibatch_size=0))
    actor_before = _params(ac.actor_parameters())
    critic_before = _params(ac.critic_parameters())
    tr.warmup_critic(trajs, n_updates=3)
    assert all(torch.equal(b, p) for b, p in zip(actor_before, ac.actor_parameters()))  # frozen
    assert any(not torch.equal(b, p) for b, p in zip(critic_before, ac.critic_parameters()))


def test_warmup_improves_explained_variance(tmp_path):
    ac = _fresh_ac(tmp_path)
    trajs = [_traj(ac, won=True, seed=1), _traj(ac, won=False, seed=2)]
    tr = PPOTrainer(ac, ppo_cfg=PPOConfig(value_loss_mode="huber"),
                    train_cfg=TrainConfig(minibatch_size=0, critic_lr=1e-2))
    txns, _, ret = tr._flatten(trajs)
    pre_ev = tr._critic_ev(txns, ret)
    stats = tr.warmup_critic(trajs, n_updates=60)
    assert math.isfinite(stats["value_loss"])
    assert stats["explained_variance"] > pre_ev                        # critic learned the returns


def test_warmup_migrates_critic_to_return_scale(tmp_path):
    """Regression: the warm-up must move V ONTO the return scale (it runs value-clip-OFF).
    A decisively-won game has returns ~+1 while the cold critic sits near value_pm 0; after
    warm-up the critic's mean value must land CLOSE to the mean return. (With the value-clip
    bug it throttles ~0.13 short; the clip-free warm-up lands within ~0.03.)"""
    ac = _fresh_ac(tmp_path)
    trajs = [_traj(ac, won=True, n=5, seed=1)]                          # returns ~ +1
    tr = PPOTrainer(ac, ppo_cfg=PPOConfig(value_clip=0.2),
                    train_cfg=TrainConfig(minibatch_size=0, critic_lr=1e-2))
    txns, _, ret = tr._flatten(trajs)
    tr.warmup_critic(trajs, n_updates=300)
    with torch.no_grad():
        v = ac.value_pm(tr._states(txns)).cpu().numpy()
    assert abs(float(v.mean()) - float(ret.mean())) < 0.1, (
        f"critic did not reach the return scale: mean v={v.mean():.3f} vs return {ret.mean():.3f}")


def test_rebase_values_overwrites_with_post_warmup_critic(tmp_path):
    """#23: at gen 0 the warm-up migrates the critic off the BC value scale, so each step's stored
    collection-time value (the value-clip anchor AND the GAE baseline for ppo_update) is stale.
    rebase_values() overwrites every transition's value with the CURRENT critic V(s) so gen-0
    behaves like gen-N (old_value == the critic at the start of the update)."""
    ac = _fresh_ac(tmp_path)
    trajs = [_traj(ac, won=True, n=5, seed=1)]                          # returns ~ +1
    tr = PPOTrainer(ac, ppo_cfg=PPOConfig(value_clip=0.2),
                    train_cfg=TrainConfig(minibatch_size=0, critic_lr=1e-2))
    txns = [t for trj in trajs for t in trj.transitions]
    pre = [t.value for t in txns]
    tr.warmup_critic(trajs, n_updates=300)                             # migrates the critic
    with torch.no_grad():
        v_now = ac.value_pm(tr._states(txns)).cpu().numpy()
    # the warm-up genuinely moved the critic away from the stored collection-time value
    assert any(abs(p - float(vn)) > 0.05 for p, vn in zip(pre, v_now)), "warm-up did not drift V"
    n = tr.rebase_values(trajs)
    assert n == len(txns)
    for t, vn in zip(txns, v_now):                                     # now == the post-warmup critic
        assert t.value == pytest.approx(float(vn), abs=1e-5)


def test_rebase_values_empty_is_noop(tmp_path):
    ac = _fresh_ac(tmp_path)
    assert PPOTrainer(ac).rebase_values([]) == 0


# ── full PPO update ───────────────────────────────────────────────────────────
def test_ppo_update_moves_both_nets(tmp_path):
    ac = _fresh_ac(tmp_path)
    trajs = [_traj(ac, won=True, seed=1), _traj(ac, won=False, seed=2)]
    tr = PPOTrainer(ac, train_cfg=TrainConfig(minibatch_size=0, ppo_epochs=2))
    a_before, c_before = _params(ac.actor_parameters()), _params(ac.critic_parameters())
    st = tr.ppo_update(trajs)
    assert any(not torch.equal(b, p) for b, p in zip(a_before, ac.actor_parameters()))
    assert any(not torch.equal(b, p) for b, p in zip(c_before, ac.critic_parameters()))
    assert st["halted"] is False
    assert st["n_steps"] == 8 and tr.updates == 1
    for k in ("loss", "policy_loss", "value_loss", "entropy", "explained_variance"):
        assert math.isfinite(st[k])


def test_ppo_update_empty_batch(tmp_path):
    ac = _fresh_ac(tmp_path)
    st = PPOTrainer(ac).ppo_update([])
    assert st["halted"] is False and st["n_steps"] == 0


# ── collapse guards ───────────────────────────────────────────────────────────
def test_kl_from_bc_autohalt(tmp_path):
    ac = _fresh_ac(tmp_path)
    trajs = [_traj(ac, won=True, seed=1), _traj(ac, won=False, seed=2)]
    tr = PPOTrainer(ac, train_cfg=TrainConfig(
        minibatch_size=0, ppo_epochs=3, actor_lr=1e-2, target_kl_from_bc=1e-9))
    st = tr.ppo_update(trajs)
    assert st["halted"] is True
    assert "kl_from_bc" in st["halt_reason"]


def test_explained_variance_autohalt(tmp_path):
    ac = _fresh_ac(tmp_path)
    trajs = [_traj(ac, won=True, seed=1), _traj(ac, won=False, seed=2)]
    tr = PPOTrainer(ac, train_cfg=TrainConfig(
        minibatch_size=0, ppo_epochs=1, min_explained_variance=2.0))  # impossible floor
    st = tr.ppo_update(trajs)
    assert st["halted"] is True
    assert "explained_variance" in st["halt_reason"]


# ── optimiser state (3c.4 resumability foundation) ────────────────────────────
def test_optimizer_state_populated_after_update(tmp_path):
    ac = _fresh_ac(tmp_path)
    trajs = [_traj(ac, won=True, seed=1), _traj(ac, won=False, seed=2)]
    tr = PPOTrainer(ac, train_cfg=TrainConfig(minibatch_size=0, ppo_epochs=1))
    tr.ppo_update(trajs)
    assert tr.actor_opt.state_dict()["state"]      # Adam moments now populated
    assert tr.critic_opt.state_dict()["state"]


# ── 3c.7a: BC-prior preservation is ON in the live config builder ──────────────
def test_build_train_configs_defaults_enable_kl_to_bc():
    """Regression: the LIVE run must turn KL-to-BC ON by default (sec 12 item 1) — guards
    against silently reverting kl_coef to 0 (which lets PPO crush the BC prior)."""
    ppo, train = build_train_configs()
    assert ppo.kl_coef > 0.0                                  # BC prior preserved
    assert train.target_kl_from_bc is not None and train.target_kl_from_bc > 0  # collapse guard on


def test_build_train_configs_tau_forced_into_ppo():
    """The PPO log-prob recompute temperature MUST equal the collection temperature."""
    ppo, _ = build_train_configs(tau=1.3)
    assert ppo.tau == pytest.approx(1.3)


def test_build_train_configs_guards_off_when_nonpositive():
    ppo, train = build_train_configs(kl_coef=0.0, target_kl_bc=0.0, min_ev=None)
    assert ppo.kl_coef == 0.0
    assert train.target_kl_from_bc is None                   # <=0 disables the guard
    assert train.min_explained_variance is None


# ── 3c.8b: CPU inference-copy for the hybrid (collection on CPU, update on GPU) ─
def test_inference_copy_independent_cpu_eval(tmp_path):
    """The collection copy must have its OWN weights (mutating it can't touch the persistent
    actor-critic / optimiser graph), be on the requested device, and be in eval mode."""
    ac = _fresh_ac(tmp_path)
    cp = ac.inference_copy("cpu")
    a0 = next(ac.policy.parameters())
    c0 = next(cp.policy.parameters())
    assert torch.equal(a0.detach().cpu(), c0.detach().cpu())     # starts identical
    assert not cp.training and not cp.policy.training            # eval mode
    with torch.no_grad():
        c0.add_(1.0)                                             # mutate the COPY
    assert not torch.equal(a0.detach().cpu(), c0.detach().cpu())  # original untouched


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_hybrid_ppo_update_on_cuda_with_cpu_collection_copy(tmp_path):
    """3c.8b hybrid END-TO-END on a real GPU: build the actor-critic + trainer on cuda (the
    GPU PPO update), collect through a CPU inference-copy (sec 20), then run a real PPO
    update on the GPU. Catches any cpu<->cuda tensor-mismatch in the device threading and
    confirms the AC stays on the GPU + the optimisers actually step."""
    ckpt = _write_ckpt(tmp_path / "bc.pt")
    ac = ActorCritic.from_bc_checkpoint(ckpt, device="cuda")
    assert next(ac.policy.parameters()).is_cuda
    coll = ac.inference_copy("cpu")                          # collection runs on CPU
    assert not next(coll.policy.parameters()).is_cuda
    trajs = [_traj(coll, won=True, seed=1), _traj(coll, won=False, seed=2)]
    tr = PPOTrainer(ac, train_cfg=TrainConfig(minibatch_size=0, ppo_epochs=2), device="cuda")
    before = next(ac.policy.parameters()).detach().clone()
    st = tr.ppo_update(trajs)
    assert st["halted"] is False and st["n_steps"] == 8
    assert next(ac.policy.parameters()).is_cuda             # stayed on GPU
    assert not torch.equal(before, next(ac.policy.parameters()))  # actually updated


def test_build_train_configs_drive_a_real_kl_autohalt(tmp_path):
    """End-to-end: a trainer built from the LIVE config builder actually early-halts on
    BC-drift — proves the collapse guard is wired through the path the CLI uses, not just
    settable in isolation."""
    ac = _fresh_ac(tmp_path)
    trajs = [_traj(ac, won=True, seed=1), _traj(ac, won=False, seed=2)]
    ppo, train = build_train_configs(target_kl_bc=1e-9)       # impossibly tight -> must halt
    train.minibatch_size, train.ppo_epochs, train.actor_lr = 0, 3, 1e-2
    st = PPOTrainer(ac, ppo_cfg=ppo, train_cfg=train).ppo_update(trajs)
    assert st["halted"] is True and "kl_from_bc" in st["halt_reason"]
