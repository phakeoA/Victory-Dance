"""Task 3b.3: PPO losses — clip + value(+clip) + entropy + KL-to-BC.

Pure-tensor hand-checks of each loss term + the value-clip mechanic, then the
from-batch orchestration (ratio==1 at init, KL==0 at init, differentiable to actor
AND critic, GAE end-to-end).
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
from v_dance.models.bc_model import BCPolicy  # noqa: E402
from v_dance.selfplay.actor_critic import ActorCritic  # noqa: E402
from v_dance.selfplay.schema import Transition, PASS_ACTION  # noqa: E402
from v_dance.selfplay.collector import TrajectoryCollector  # noqa: E402
from v_dance.selfplay.reward import place_terminal_reward  # noqa: E402
from v_dance.selfplay.gae import compute_batch_gae  # noqa: E402
from v_dance.selfplay import policy_eval as pe  # noqa: E402
from v_dance.selfplay import ppo as P  # noqa: E402

STATE_DIM, ACTION_DIM, GIMMICK_DIM = get_state_dim(), get_action_dim(), get_gimmick_dim()


def _write_ckpt(path):
    torch.manual_seed(2)
    model = BCPolicy(state_dim=STATE_DIM, action_dim=ACTION_DIM,
                     hidden_dims=(8, 4), dropout=0.0, gimmick_dim=GIMMICK_DIM)
    torch.save({"model_state": model.state_dict(), "config": {
        "state_dim": STATE_DIM, "action_dim": ACTION_DIM, "hidden_dims": (8, 4),
        "dropout": 0.0, "heads": ("our_a", "our_b"), "gimmick_dim": GIMMICK_DIM,
        "gimmick_trained": True, "value_trained": True,
        "state_layout_version": get_state_layout_version()}}, path)
    return path


@pytest.fixture(scope="module")
def ac(tmp_path_factory):
    return ActorCritic.from_bc_checkpoint(_write_ckpt(tmp_path_factory.mktemp("ppo") / "bc.pt"))


def _txn(a0, a1, *, g0=0, g1=0, m0=None, m1=None, gm0=None, gm1=None, seed=0):
    g = np.random.default_rng(seed)
    return Transition(state=g.standard_normal(STATE_DIM).astype(np.float32),
                      action_s0=a0, action_s1=a1, gimmick_s0=g0, gimmick_s1=g1,
                      mask_s0=m0, mask_s1=m1, gmask_s0=gm0, gmask_s1=gm1)


# ── pure clip loss ────────────────────────────────────────────────────────────
def _z(n=2):
    return torch.zeros(n)


def test_clip_loss_ratio_one():
    cfg = P.PPOConfig(value_coef=0, entropy_coef=0, standardize_adv=False)
    adv = torch.tensor([2.0, -1.0])
    loss, st = P.ppo_losses(new_logprob=_z(), old_logprob=_z(), advantages=adv,
                            value_pm=_z(), old_value_pm=_z(), returns=_z(),
                            entropy=_z(), cfg=cfg)
    assert st["policy_loss"] == pytest.approx(-0.5, abs=1e-6)   # -mean([2,-1])
    assert float(loss) == pytest.approx(-0.5, abs=1e-6)
    assert st["ratio_mean"] == pytest.approx(1.0, abs=1e-6)
    assert st["clip_fraction"] == pytest.approx(0.0)


def test_clip_engages_on_large_ratio():
    cfg = P.PPOConfig(clip_eps=0.2, value_coef=0, entropy_coef=0, standardize_adv=False)
    new = torch.log(torch.tensor([2.0, 2.0]))                   # ratio = 2
    adv = torch.tensor([1.0, 1.0])
    loss, st = P.ppo_losses(new_logprob=new, old_logprob=_z(), advantages=adv,
                            value_pm=_z(), old_value_pm=_z(), returns=_z(),
                            entropy=_z(), cfg=cfg)
    assert st["policy_loss"] == pytest.approx(-1.2, abs=1e-5)   # clipped to 1+eps
    assert st["clip_fraction"] == pytest.approx(1.0)


# ── value loss + clip ─────────────────────────────────────────────────────────
def test_value_huber_zero_at_perfect_prediction():
    cfg = P.PPOConfig(value_loss_mode="huber")
    v = torch.tensor([0.3, -0.4])
    assert float(P._value_loss(v, v, v, cfg)) == pytest.approx(0.0, abs=1e-7)


def test_value_clip_caps_huber():
    cfg = P.PPOConfig(value_loss_mode="huber", value_clip=0.1, huber_delta=1.0)
    v_new = torch.tensor([0.5]); old = torch.tensor([0.0]); ret = torch.tensor([0.0])
    # max(huber(0.5), huber(0.1)) = max(0.125, 0.005) = 0.125
    assert float(P._value_loss(v_new, old, ret, cfg)) == pytest.approx(0.125, abs=1e-6)


def test_value_bce_minimised_at_target():
    cfg = P.PPOConfig(value_loss_mode="bce", value_clip=10.0)
    old = torch.tensor([0.0]); ret = torch.tensor([0.4])
    close = float(P._value_loss(torch.tensor([0.4]), old, ret, cfg))
    far = float(P._value_loss(torch.tensor([-0.8]), old, ret, cfg))
    assert close < far


def test_value_clip_off_keeps_gradient_past_boundary():
    """The warm-up value-clip bug, at the mechanism level: when v_new has moved PAST the
    clip boundary toward the target, the pessimistic max selects the clipped (boundary-
    frozen) loss whose gradient w.r.t. v_new is 0 -> the critic can't migrate. clip=False
    restores a real gradient toward the target (the 3b.4 fix)."""
    cfg = P.PPOConfig(value_loss_mode="huber", value_clip=0.2)
    old, target = torch.tensor([0.0]), torch.tensor([1.0])
    v1 = torch.tensor([0.5], requires_grad=True)               # 0.5 > old + 0.2
    P._value_loss(v1, old, target, cfg, clip=True).backward()
    v2 = torch.tensor([0.5], requires_grad=True)
    P._value_loss(v2, old, target, cfg, clip=False).backward()
    assert v1.grad.abs().item() == pytest.approx(0.0, abs=1e-8)   # clipped: gradient vanishes
    assert v2.grad.abs().item() > 0.1                            # unclipped: still pulling to target


# ── entropy + KL terms ────────────────────────────────────────────────────────
def test_entropy_bonus_lowers_loss():
    base = dict(new_logprob=_z(), old_logprob=_z(), advantages=_z(),
                value_pm=_z(), old_value_pm=_z(), returns=_z(), cfg=P.PPOConfig(
                    value_coef=0, entropy_coef=0.1, standardize_adv=False))
    lo, _ = P.ppo_losses(entropy=torch.tensor([0.1, 0.1]), **base)
    hi, _ = P.ppo_losses(entropy=torch.tensor([1.0, 1.0]), **base)
    assert float(hi) < float(lo)                                # more entropy -> lower loss


def test_kl_penalty_only_when_coef_positive():
    kl = torch.tensor([0.5, 0.5])
    common = dict(new_logprob=_z(), old_logprob=_z(), advantages=_z(),
                  value_pm=_z(), old_value_pm=_z(), returns=_z(), entropy=_z(), kl_to_ref=kl)
    off, st_off = P.ppo_losses(cfg=P.PPOConfig(value_coef=0, entropy_coef=0,
                                               standardize_adv=False, kl_coef=0.0), **common)
    on, st_on = P.ppo_losses(cfg=P.PPOConfig(value_coef=0, entropy_coef=0,
                                             standardize_adv=False, kl_coef=1.0), **common)
    assert st_off["kl_to_bc"] == pytest.approx(0.5)             # always logged
    assert float(on) - float(off) == pytest.approx(0.5, abs=1e-6)  # penalised only when on


def test_empty_minibatch_raises():
    with pytest.raises(ValueError, match="empty minibatch"):
        P.ppo_losses(new_logprob=_z(0), old_logprob=_z(0), advantages=_z(0),
                     value_pm=_z(0), old_value_pm=_z(0), returns=_z(0), entropy=_z(0))


# ── reference policy + KL-to-BC integration ───────────────────────────────────
def test_make_reference_frozen_and_matches_bc(ac):
    ref = P.make_reference_policy(ac)
    assert all(not p.requires_grad for p in ref.parameters())
    x = torch.randn(3, STATE_DIM)
    with torch.no_grad():
        a_ref, _, _ = ref(x)
        a_ac, _, _ = ac.policy(x)
    assert torch.allclose(a_ref["our_a"], a_ac["our_a"], atol=1e-6)


def test_kl_zero_at_init(ac):
    ref = P.make_reference_policy(ac)
    txns = [_txn(i % ACTION_DIM, PASS_ACTION, m0=[1] * ACTION_DIM, seed=i) for i in range(4)]
    ev = pe.ppo_forward(ac, txns, ref_policy=ref)
    assert ev.kl_to_ref is not None
    assert torch.all(ev.kl_to_ref.abs() < 1e-5)                 # new == BC -> KL 0


def test_kl_positive_after_actor_drift(tmp_path):
    ac2 = ActorCritic.from_bc_checkpoint(_write_ckpt(tmp_path / "bc.pt"))
    ref = P.make_reference_policy(ac2)                          # snapshot BEFORE drift
    with torch.no_grad():
        ac2.policy.heads["our_a"].bias[0].add_(5.0)            # boost ONE action's logit
    txns = [_txn(3, PASS_ACTION, m0=[1] * ACTION_DIM, seed=i) for i in range(4)]
    ev = pe.ppo_forward(ac2, txns, ref_policy=ref)
    assert float(ev.kl_to_ref.mean().detach()) > 1e-3


# ── from-batch orchestration ──────────────────────────────────────────────────
def _onpolicy_txns(ac, n=6):
    """Transitions whose stored old logprob/value are the policy's OWN current outputs
    (so the PPO ratio is exactly 1 at the first epoch)."""
    txns = [_txn(i % ACTION_DIM, PASS_ACTION, g0=0, m0=[1] * ACTION_DIM, gm0=[1, 0], seed=i)
            for i in range(n)]
    lp, _, v = pe.evaluate_actions(ac, txns)
    for t, l, vv in zip(txns, lp.detach().tolist(), v.detach().tolist()):
        t.logprob, t.value = float(l), float(vv)
    return txns


def test_from_batch_ratio_one_at_init(ac):
    txns = _onpolicy_txns(ac)
    adv = np.ones(len(txns), np.float32)
    ret = np.zeros(len(txns), np.float32)
    _, st = P.ppo_loss_from_batch(ac, txns, adv, ret, cfg=P.PPOConfig())
    assert st["ratio_mean"] == pytest.approx(1.0, abs=1e-4)
    assert st["clip_fraction"] == pytest.approx(0.0, abs=1e-6)
    assert st["approx_kl_old_new"] == pytest.approx(0.0, abs=1e-5)


def test_from_batch_backprops_to_actor_and_critic(tmp_path):
    ac2 = ActorCritic.from_bc_checkpoint(_write_ckpt(tmp_path / "bc.pt"))
    ref = P.make_reference_policy(ac2)
    txns = _onpolicy_txns(ac2)
    adv = np.array([1.0, -1.0, 0.5, -0.5, 0.2, -0.2], np.float32)
    ret = np.zeros(len(txns), np.float32)
    cfg = P.PPOConfig(kl_coef=0.1)
    loss, st = P.ppo_loss_from_batch(ac2, txns, adv, ret, cfg=cfg, ref_policy=ref)
    loss.backward()
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in ac2.actor_parameters())
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in ac2.critic_parameters())
    assert all(math.isfinite(v) for v in st.values())


def test_gae_end_to_end(ac):
    """compute_batch_gae -> ppo_loss_from_batch (the real advantage/return path)."""
    c = TrajectoryCollector("g", "p1")
    for t in range(3):
        step = _txn(t % ACTION_DIM, PASS_ACTION, m0=[1] * ACTION_DIM, seed=t)
        lp, _, v = pe.evaluate_actions(ac, [step])
        c.add_step(state=step.state, action_s0=t % ACTION_DIM, action_s1=PASS_ACTION,
                   mask_s0=[1] * ACTION_DIM,
                   logprob=float(lp[0].detach()), value=float(v[0].detach()), turn=t + 1)
    tr = c.finish(own_team=["a"] * 6, opp_team=["b"] * 6, tp_bring=[0, 1, 2, 3],
                  tp_leads=[0, 1], won=True, terminal_type="win")
    place_terminal_reward(tr)
    adv, ret = compute_batch_gae([tr], standardize_adv=False)
    loss, st = P.ppo_loss_from_batch(ac, tr.transitions, adv, ret, cfg=P.PPOConfig())
    assert math.isfinite(st["loss"]) and math.isfinite(st["value_loss"])
