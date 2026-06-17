"""Task 3b.5: per-head masked joint log-prob + entropy (PPO ratio recompute).

Covers the pure masked-categorical core, the transition-batch wiring (head->slot
mapping, PASS handling, gimmick gating), the collection<->recompute consistency
that makes the PPO ratio meaningful, differentiability, and the schema/collector
mask plumbing.
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

_REPO = Path(__file__).resolve().parents[3]
for _p in (str(_REPO / "local_battle"), str(_REPO / "data" / "scripts"),
           str(_REPO / "ai_train_scripts" / "BC_model"), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from state_encoder import (get_state_dim, get_action_dim, get_gimmick_dim,  # noqa: E402
                           get_state_layout_version)
from bc_model import BCPolicy  # noqa: E402
from self_play.actor_critic import ActorCritic  # noqa: E402
from self_play.schema import Transition, PASS_ACTION  # noqa: E402
from self_play.collector import TrajectoryCollector  # noqa: E402
from self_play import policy_eval as pe  # noqa: E402

STATE_DIM, ACTION_DIM, GIMMICK_DIM = get_state_dim(), get_action_dim(), get_gimmick_dim()


def _write_ckpt(path):
    torch.manual_seed(1)
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
    p = tmp_path_factory.mktemp("pe") / "bc.pt"
    return ActorCritic.from_bc_checkpoint(_write_ckpt(p))


def _txn(a0, a1, *, g0=0, g1=0, m0=None, m1=None, gm0=None, gm1=None, seed=0):
    g = np.random.default_rng(seed)
    return Transition(state=g.standard_normal(STATE_DIM).astype(np.float32),
                      action_s0=a0, action_s1=a1, gimmick_s0=g0, gimmick_s1=g1,
                      mask_s0=m0, mask_s1=m1, gmask_s0=gm0, gmask_s1=gm1)


# ── pure core ─────────────────────────────────────────────────────────────────
def test_masked_log_softmax_zeros_illegal():
    logits = torch.tensor([[1.0, 1.0, 5.0, 5.0]])
    legal = torch.tensor([[True, True, False, False]])
    log_p = pe.masked_log_softmax(logits, legal)
    p = log_p.exp()
    assert p[0, 2] < 1e-10 and p[0, 3] < 1e-10        # illegal ~ 0 prob
    assert p[0, :2].sum().item() == pytest.approx(1.0, abs=1e-5)


def test_categorical_two_uniform_legal():
    logits = torch.zeros(1, 4)                          # equal logits
    legal = torch.tensor([[True, True, False, False]])
    lp, ent = pe.categorical_logprob_entropy(logits, legal, torch.tensor([0]))
    assert lp.item() == pytest.approx(-math.log(2), abs=1e-5)
    assert ent.item() == pytest.approx(math.log(2), abs=1e-5)   # max entropy over 2


def test_categorical_single_legal_is_deterministic():
    logits = torch.tensor([[3.0, -2.0]])
    legal = torch.tensor([[True, False]])
    lp, ent = pe.categorical_logprob_entropy(logits, legal, torch.tensor([0]))
    assert lp.item() == pytest.approx(0.0, abs=1e-6)    # only option -> prob 1
    assert ent.item() == pytest.approx(0.0, abs=1e-6)


def test_temperature_flattens_distribution():
    logits = torch.tensor([[2.0, 0.0]])
    legal = torch.tensor([[True, True]])
    _, ent_cold = pe.categorical_logprob_entropy(logits, legal, torch.tensor([0]), tau=0.5)
    _, ent_hot = pe.categorical_logprob_entropy(logits, legal, torch.tensor([0]), tau=4.0)
    assert ent_hot.item() > ent_cold.item()            # hotter -> higher entropy


# ── transition-batch wiring ───────────────────────────────────────────────────
def test_evaluate_shapes_and_value_range(ac):
    txns = [_txn(0, 1, seed=i) for i in range(5)]
    lp, ent, v = pe.evaluate_actions(ac, txns)
    assert lp.shape == (5,) and ent.shape == (5,) and v.shape == (5,)
    assert torch.all(v >= -1) and torch.all(v <= 1)
    assert torch.all(ent >= -1e-6)


def test_empty_batch():
    # no model call needed; just the empty guard
    class _Dummy:  # not used
        pass
    lp, ent, v = pe.evaluate_actions(None, [])
    assert lp.numel() == 0 and ent.numel() == 0 and v.numel() == 0


def test_both_pass_is_zero(ac):
    lp, ent, _ = pe.evaluate_actions(ac, [_txn(PASS_ACTION, PASS_ACTION)])
    assert lp.item() == pytest.approx(0.0, abs=1e-6)
    assert ent.item() == pytest.approx(0.0, abs=1e-6)


def test_matches_manual_core_recompute(ac):
    """Integration check: evaluate_actions == hand-applying the core to the actor's
    own logits, with slot->head mapping + PASS + gimmick gating."""
    m0 = [1] * ACTION_DIM
    gm0 = [1, 1]
    txn = _txn(3, PASS_ACTION, g0=1, m0=m0, gm0=gm0, seed=42)
    lp, ent, _ = pe.evaluate_actions(ac, [txn])

    states = torch.as_tensor(txn.state[None], dtype=torch.float32)
    a_logits, g_logits, _ = ac(states)
    legal = torch.tensor([[bool(x) for x in m0]])
    a_lp, a_ent = pe.categorical_logprob_entropy(a_logits["our_a"], legal, torch.tensor([3]))
    g_legal = torch.tensor([[True, True]])
    g_lp, g_ent = pe.categorical_logprob_entropy(g_logits["our_a"], g_legal, torch.tensor([1]))
    assert lp.item() == pytest.approx((a_lp + g_lp).item(), abs=1e-5)
    assert ent.item() == pytest.approx((a_ent + g_ent).item(), abs=1e-5)


def test_gimmick_term_skipped_without_mask(ac):
    """A present gimmick mask adds the gimmick log-prob; None skips it."""
    base = dict(g0=1, m0=[1] * ACTION_DIM, seed=7)
    with_g = pe.evaluate_actions(ac, [_txn(3, PASS_ACTION, gm0=[1, 1], **base)])[0]
    without_g = pe.evaluate_actions(ac, [_txn(3, PASS_ACTION, gm0=None, **base)])[0]
    assert with_g.item() != pytest.approx(without_g.item(), abs=1e-6)


def test_consistency_ratio_one(ac):
    """Re-evaluating the SAME steps with the SAME weights reproduces logprob exactly
    -> the PPO ratio at epoch 0 is 1 (the importance-sampling sanity invariant)."""
    txns = [_txn(i % ACTION_DIM, PASS_ACTION, m0=[1] * ACTION_DIM, seed=i) for i in range(6)]
    lp1, _, _ = pe.evaluate_actions(ac, txns)
    lp2, _, _ = pe.evaluate_actions(ac, txns)
    assert torch.allclose(lp1, lp2, atol=1e-6)
    assert torch.allclose((lp2 - lp1).exp(), torch.ones_like(lp1), atol=1e-6)


def test_differentiable_to_actor_and_critic(tmp_path):
    ac = ActorCritic.from_bc_checkpoint(_write_ckpt(tmp_path / "bc.pt"))
    txns = [_txn(2, 5, g0=0, g1=0, m0=[1] * ACTION_DIM, m1=[1] * ACTION_DIM,
                 gm0=[1, 0], gm1=[1, 0], seed=i) for i in range(4)]
    lp, ent, v = pe.evaluate_actions(ac, txns)
    (lp.sum() + 0.01 * ent.sum()).backward(retain_graph=True)
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in ac.actor_parameters())
    v.sum().backward()
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in ac.critic_parameters())


# ── schema / collector / integrity plumbing ───────────────────────────────────
def test_schema_roundtrips_masks():
    t = _txn(3, 7, g0=1, m0=[1, 0, 1] + [0] * (ACTION_DIM - 3),
             m1=[1] * ACTION_DIM, gm0=[1, 1], gm1=[1, 0])
    back = Transition.from_obj(t.to_obj())
    assert back.mask_s0 == t.mask_s0 and back.mask_s1 == t.mask_s1
    assert back.gmask_s0 == t.gmask_s0 and back.gmask_s1 == t.gmask_s1


def test_collector_records_masks():
    c = TrajectoryCollector("g", "p1")
    c.add_step(state=np.zeros(STATE_DIM, np.float32), action_s0=3, action_s1=PASS_ACTION,
               gimmick_s0=1, mask_s0=[1] * ACTION_DIM, gmask_s0=[1, 1], turn=1)
    tr = c.finish(own_team=["a"] * 6, opp_team=["b"] * 6, tp_bring=[0, 1, 2, 3],
                  tp_leads=[0, 1], won=True, terminal_type="win")
    t = tr.transitions[0]
    assert t.mask_s0 == [1] * ACTION_DIM and t.gmask_s0 == [1, 1]
    assert t.mask_s1 is None and t.gmask_s1 is None


def test_assert_actions_legal_passes_and_raises():
    ok = [_txn(2, PASS_ACTION, m0=[0, 0, 1] + [0] * (ACTION_DIM - 3))]
    pe.assert_actions_legal(ok)                                    # legal -> no raise
    bad_a = [_txn(0, PASS_ACTION, m0=[0, 1] + [0] * (ACTION_DIM - 2))]  # action 0 illegal
    with pytest.raises(AssertionError, match="illegal under mask_s0"):
        pe.assert_actions_legal(bad_a)
    bad_g = [_txn(2, PASS_ACTION, g0=1, m0=[0, 0, 1] + [0] * (ACTION_DIM - 3), gm0=[1, 0])]
    with pytest.raises(AssertionError, match="illegal under gmask_s0"):
        pe.assert_actions_legal(bad_g)
