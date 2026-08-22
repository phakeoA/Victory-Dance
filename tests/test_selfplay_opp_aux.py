"""Piece 2 (Level-A): auxiliary opponent-prediction CE term in the PPO update.

Covers the pure CE helper (value, PASS-masking, gradient, opp-head gating), its
plumbing through ``ppo_losses`` (added only when ``opp_aux_coef>0`` and present), and
``ppo_forward`` end-to-end on a real attn model that carries opp_a/opp_b heads.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("numpy")
import numpy as np
import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[1]
from conftest import write_attn_ckpt  # noqa: E402
from v_dance.encoders.state_encoder import get_state_dim, get_action_dim  # noqa: E402
from v_dance.selfplay.actor_critic import ActorCritic  # noqa: E402
from v_dance.selfplay.schema import Transition, PASS_ACTION  # noqa: E402
from v_dance.selfplay import policy_eval as pe  # noqa: E402
from v_dance.selfplay import ppo as P  # noqa: E402

STATE_DIM, ACTION_DIM = get_state_dim(), get_action_dim()


def _logits(B, A, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.randn(B, A, generator=g)
    return t.requires_grad_(True)


def _txn(a0=0, a1=0, opp_a=PASS_ACTION, opp_b=PASS_ACTION):
    return Transition(state=np.zeros(STATE_DIM, np.float32), action_s0=a0, action_s1=a1,
                      opp_a_action=opp_a, opp_b_action=opp_b)


# ── pure helper: opp_aux_ce ────────────────────────────────────────────────────
def test_none_without_opp_heads():
    al = {"our_a": _logits(3, ACTION_DIM), "our_b": _logits(3, ACTION_DIM)}
    txns = [_txn(opp_a=1, opp_b=2) for _ in range(3)]
    ce, n = pe.opp_aux_ce(al, txns, head_names=("our_a", "our_b"))
    assert ce is None and n == 0


def test_matches_manual_cross_entropy():
    A = ACTION_DIM
    la, lb = _logits(4, A, 1), _logits(4, A, 2)
    al = {"our_a": _logits(4, A, 3), "our_b": _logits(4, A, 4), "opp_a": la, "opp_b": lb}
    txns = [_txn(opp_a=i % A, opp_b=(i + 1) % A) for i in range(4)]
    ce, n = pe.opp_aux_ce(al, txns, head_names=tuple(al))
    assert n == 8                                          # 4 rows x 2 heads
    ta = torch.tensor([i % A for i in range(4)])
    tb = torch.tensor([(i + 1) % A for i in range(4)])
    man = (F.cross_entropy(la, ta, reduction="sum")
           + F.cross_entropy(lb, tb, reduction="sum")) / 8
    assert torch.allclose(ce, man, atol=1e-6)


def test_masks_pass_targets():
    A = ACTION_DIM
    la, lb = _logits(3, A, 1), _logits(3, A, 2)
    al = {"our_a": _logits(3, A), "our_b": _logits(3, A), "opp_a": la, "opp_b": lb}
    # only row 0's opp_a is a real target; opp_b is all-PASS
    txns = [_txn(opp_a=5, opp_b=PASS_ACTION),
            _txn(opp_a=PASS_ACTION, opp_b=PASS_ACTION),
            _txn(opp_a=PASS_ACTION, opp_b=PASS_ACTION)]
    ce, n = pe.opp_aux_ce(al, txns, head_names=tuple(al))
    assert n == 1
    assert torch.allclose(ce, F.cross_entropy(la[0:1], torch.tensor([5])), atol=1e-6)


def test_none_when_all_pass():
    A = ACTION_DIM
    al = {"our_a": _logits(2, A), "our_b": _logits(2, A),
          "opp_a": _logits(2, A), "opp_b": _logits(2, A)}
    ce, n = pe.opp_aux_ce(al, [_txn(), _txn()], head_names=tuple(al))
    assert ce is None and n == 0


def test_grad_flows_to_opp_logits():
    A = ACTION_DIM
    la, lb = _logits(2, A, 1), _logits(2, A, 2)
    al = {"our_a": _logits(2, A), "our_b": _logits(2, A), "opp_a": la, "opp_b": lb}
    ce, _ = pe.opp_aux_ce(al, [_txn(opp_a=1, opp_b=2), _txn(opp_a=3, opp_b=4)],
                          head_names=tuple(al))
    ce.backward()
    assert la.grad is not None and la.grad.abs().sum() > 0
    assert lb.grad is not None and lb.grad.abs().sum() > 0


# ── ppo_losses plumbing ─────────────────────────────────────────────────────────
def _base_inputs(B=4):
    return dict(
        new_logprob=torch.zeros(B, requires_grad=True),
        old_logprob=torch.zeros(B),
        advantages=torch.ones(B),
        value_pm=torch.zeros(B, requires_grad=True),
        old_value_pm=torch.zeros(B),
        returns=torch.zeros(B),
        entropy=torch.zeros(B),
    )


def test_ppo_losses_adds_opp_term_when_coef_positive():
    opp_ce = torch.tensor(2.0)
    loss_off, st_off = P.ppo_losses(**_base_inputs(), opp_ce=opp_ce,
                                    cfg=P.PPOConfig(opp_aux_coef=0.0))
    loss_on, st_on = P.ppo_losses(**_base_inputs(), opp_ce=opp_ce,
                                  cfg=P.PPOConfig(opp_aux_coef=0.5))
    assert st_on["opp_ce"] == pytest.approx(2.0)
    assert float(loss_on.detach()) - float(loss_off.detach()) == pytest.approx(0.5 * 2.0, abs=1e-5)


def test_ppo_losses_noop_when_opp_ce_none():
    loss_a, st = P.ppo_losses(**_base_inputs(), opp_ce=None,
                              cfg=P.PPOConfig(opp_aux_coef=0.5))
    loss_b, _ = P.ppo_losses(**_base_inputs(), cfg=P.PPOConfig(opp_aux_coef=0.5))
    assert math.isnan(st["opp_ce"])
    assert float(loss_a.detach()) == pytest.approx(float(loss_b.detach()))


# ── ppo_forward end-to-end (real attn model) ────────────────────────────────────
def test_ppo_forward_opp_ce_present_with_opp_heads(tmp_path):
    ck = write_attn_ckpt(tmp_path / "opp.pt",
                         heads=("our_a", "our_b", "opp_a", "opp_b"),
                         gimmick_heads=("our_a", "our_b"))
    ac = ActorCritic.from_bc_checkpoint(ck)
    assert "opp_a" in ac.head_names and "opp_b" in ac.head_names
    ev = pe.ppo_forward(ac, [_txn(a0=0, a1=PASS_ACTION, opp_a=1, opp_b=2)])
    assert ev.opp_ce is not None and float(ev.opp_ce.detach()) >= 0.0


def test_ppo_forward_opp_ce_none_without_opp_heads(tmp_path):
    ac = ActorCritic.from_bc_checkpoint(write_attn_ckpt(tmp_path / "noopp.pt"))
    ev = pe.ppo_forward(ac, [_txn(a0=0, a1=PASS_ACTION, opp_a=1, opp_b=2)])
    assert ev.opp_ce is None


def test_build_train_configs_opp_aux_and_gimmick_precedence():
    """CLI value > --config file value > PPOConfig default, for both Phase-1 knobs."""
    from v_dance.selfplay.generation import build_train_configs
    # default (no CLI, no file) -> PPOConfig defaults
    d, _ = build_train_configs()
    assert d.opp_aux_coef == 0.0 and d.gimmick_kl_weight == 1.0
    # explicit (CLI-derived) kwarg wins
    e, _ = build_train_configs(opp_aux_coef=0.3, gimmick_kl_weight=0.5)
    assert e.opp_aux_coef == pytest.approx(0.3) and e.gimmick_kl_weight == pytest.approx(0.5)
    # --config file value (ppo_overrides) is KEPT when no CLI value is given (the gap-fix)
    f, _ = build_train_configs(ppo_overrides={"opp_aux_coef": 0.25, "gimmick_kl_weight": 0.7})
    assert f.opp_aux_coef == pytest.approx(0.25) and f.gimmick_kl_weight == pytest.approx(0.7)
    # CLI value BEATS the file value
    c, _ = build_train_configs(opp_aux_coef=0.4, ppo_overrides={"opp_aux_coef": 0.25})
    assert c.opp_aux_coef == pytest.approx(0.4)
