"""W3b-1b (2026-09-02) — pair-conditional policy evaluation (docs/w3b_ladder_ppo_design.md §3).

The served era-4 2b decode is SEQUENTIAL (first slot from the zero-cond logits, the other from a
partner-conditioned forward); the RL evaluator used to score both slots as independent softmaxes,
so a ladder step's PPO ratio was never 1 even before the first update. ``policy_eval`` now has a
pair mode that reproduces the serve decode over the stored EFFECTIVE masks, plus the parity
switches for the heads the serve argmaxes (gimmick, forced replacement). Everything here drives the
REAL serve sampler (model_io.bc_action_indices under a pair_cond tiny policy) and checks the
evaluator recomputes exactly what it recorded.
"""
from __future__ import annotations

import copy
import math

import pytest

pytest.importorskip("torch")
import numpy as np
import torch

from v_dance.encoders.state_encoder import get_action_dim, get_gimmick_dim, get_state_dim
from v_dance.models.bc_model_attn import AttnBCPolicy
from v_dance.play import model_io as M
from v_dance.selfplay import policy_eval as pe
from v_dance.selfplay.actor_critic import ActorCritic, AttnCritic
from v_dance.selfplay.collector import TrajectoryCollector
from v_dance.selfplay.ppo import PPOConfig, make_reference_policy, ppo_loss_from_batch
from v_dance.selfplay.schema import PASS_ACTION, Transition

A, S, G = get_action_dim(), get_state_dim(), get_gimmick_dim()
HEADS = ("our_a", "our_b", "opp_a", "opp_b")
OUR = ("our_a", "our_b")


def _pair_ac(seed: int = 3) -> ActorCritic:
    torch.manual_seed(seed)
    pol = AttnBCPolicy(d_model=32, n_heads=4, n_layers=1, dropout=0.0, heads=HEADS,
                       gimmick_heads=OUR, pair_cond=True).eval()
    pol._pair_decode = True                     # what load_bc_policy stamps for a pair checkpoint
    critic = AttnCritic(copy.deepcopy(pol)).eval()
    return ActorCritic(pol, critic, pol.head_names, pol.gimmick_head_names, True)


def _masks(rng, n_legal=(4, 12)):
    m = [False] * A
    for i in rng.choice(A, size=int(rng.integers(*n_legal)), replace=False):
        m[int(i)] = True
    return m


def _serve_steps(ac, n: int, tau: float, seed: int = 0, drop=None, gmask=None):
    """Record ``n`` decisions exactly as the online bot does: the real serve sampler under the
    pair decode → the decode record → a Transition with the effective masks, the summed
    behaviour log-prob and the decode order."""
    rng = np.random.default_rng(seed)
    txns = []
    for i in range(n):
        x = rng.standard_normal(S).astype(np.float32)
        m0, m1 = _masks(rng), _masks(rng)
        M.LAST_DECODE.clear()
        a0, a1 = M.bc_action_indices(ac.policy, ac.head_names, x, m0, m1, temperature=tau,
                                     rng=rng, pair_futility=drop)
        rec = M.decode_record()
        assert rec["pair"] is True and rec["picks"] == (a0, a1)
        lp = sum(float(t) for t in rec["logp"] if t is not None)
        txns.append(Transition(state=x, action_s0=(PASS_ACTION if a0 is None else a0),
                               action_s1=(PASS_ACTION if a1 is None else a1),
                               logprob=lp, mask_s0=list(rec["masks"][0]), mask_s1=list(rec["masks"][1]),
                               gmask_s0=gmask, gmask_s1=gmask, pair_first=rec["first"]))
    return txns


# ── parity with the serve decode ─────────────────────────────────────────────
def test_pair_mode_reproduces_the_recorded_behaviour_logprob_exactly():
    ac = _pair_ac()
    txns = _serve_steps(ac, 14, tau=0.7, seed=1)
    recorded = torch.tensor([t.logprob for t in txns])
    with torch.no_grad():
        lp_pair, ent, _ = pe.evaluate_actions(ac, txns, tau=0.7, pair=True)
        lp_indep, _, _ = pe.evaluate_actions(ac, txns, tau=0.7, pair=False)
    assert torch.allclose(lp_pair, recorded, atol=1e-4)               # the ratio is exactly 1
    assert not torch.allclose(lp_indep, recorded, atol=1e-3)          # the old evaluator was not
    assert bool((ent > 0).all())
    assert all(t.pair_first in (0, 1) for t in txns)


def test_recompute_order_matches_the_recorded_order_at_the_warm_start():
    ac = _pair_ac(seed=5)
    txns = _serve_steps(ac, 12, tau=1.0, seed=2)
    with torch.no_grad():
        ev_rec = pe.ppo_forward(ac, txns, tau=1.0, pair=True, order="recorded")
        ev_re = pe.ppo_forward(ac, txns, tau=1.0, pair=True, order="recompute")
    assert ev_rec.pair_flips == 0.0 and ev_re.pair_flips == 0.0
    assert torch.allclose(ev_rec.logprob, ev_re.logprob, atol=1e-5)
    # a legacy step without pair_first falls back to the recompute → same result here
    for t in txns:
        t.pair_first = None
    with torch.no_grad():
        ev_legacy = pe.ppo_forward(ac, txns, tau=1.0, pair=True, order="recorded")
    assert torch.allclose(ev_legacy.logprob, ev_rec.logprob, atol=1e-5)


def test_pair_futility_drop_is_honoured_through_the_stored_mask():
    ac = _pair_ac(seed=8)

    def drop(second, first, a_first):           # drop the lowest legal action of the second slot
        return {0, 1, 2}
    txns = _serve_steps(ac, 10, tau=0.8, seed=3, drop=drop)
    assert any(sum(t.mask_s0) + sum(t.mask_s1) < 2 * A for t in txns)
    with torch.no_grad():
        lp, _, _ = pe.evaluate_actions(ac, txns, tau=0.8, pair=True)
    assert torch.allclose(lp, torch.tensor([t.logprob for t in txns]), atol=1e-4)


def test_pair_mode_with_a_single_pick_is_the_zero_cond_term():
    ac = _pair_ac(seed=9)
    rng = np.random.default_rng(4)
    x = rng.standard_normal(S).astype(np.float32)
    m1 = _masks(rng)
    M.LAST_DECODE.clear()
    a0, a1 = M.bc_action_indices(ac.policy, ac.head_names, x, [False] * A, m1, temperature=0.9, rng=rng)
    rec = M.decode_record()
    assert a0 is None and rec["logp"][0] is None
    t = Transition(state=x, action_s0=PASS_ACTION, action_s1=a1, logprob=float(rec["logp"][1]),
                   mask_s0=[0] * A, mask_s1=list(rec["masks"][1]), pair_first=rec["first"])
    with torch.no_grad():
        lp, _, _ = pe.evaluate_actions(ac, [t], tau=0.9, pair=True)
    assert lp.item() == pytest.approx(t.logprob, abs=1e-4)


# ── the argmax-served heads ──────────────────────────────────────────────────
def test_gimmick_terms_switch_drops_the_gimmick_head():
    ac = _pair_ac(seed=2)
    txns = _serve_steps(ac, 6, tau=0.7, seed=5, gmask=[1, 1, 0][:G] + [0] * max(0, G - 3))
    recorded = torch.tensor([t.logprob for t in txns])
    with torch.no_grad():
        with_g, _, _ = pe.evaluate_actions(ac, txns, tau=0.7, pair=True, gimmick_terms=True)
        without_g, _, _ = pe.evaluate_actions(ac, txns, tau=0.7, pair=True, gimmick_terms=False)
    assert torch.allclose(without_g, recorded, atol=1e-4)             # actions only = the sampler's
    assert not torch.allclose(with_g, recorded, atol=1e-3)            # the gimmick term would bias it


def test_policy_mask_zeroes_replacement_steps_and_keeps_their_value():
    ac = _pair_ac(seed=4)
    txns = _serve_steps(ac, 5, tau=0.7, seed=6)
    txns[1].decision_type = "replacement"
    txns[3].decision_type = "replacement"
    pm = pe.replacement_policy_mask(txns)
    assert pm.tolist() == [True, False, True, False, True]
    with torch.no_grad():
        ev = pe.ppo_forward(ac, txns, tau=0.7, pair=True, policy_mask=pm,
                            ref_policy=make_reference_policy(ac))
    assert ev.logprob[1].item() == 0.0 and ev.entropy[3].item() == 0.0 and ev.kl_to_ref[1].item() == 0.0
    assert ev.logprob[0].item() != 0.0 and ev.value_pm.shape == (5,)


# ── through the PPO loss ─────────────────────────────────────────────────────
def test_ppo_loss_under_the_ladder_parity_config_has_ratio_one_at_init():
    ac = _pair_ac(seed=6)
    txns = _serve_steps(ac, 16, tau=0.5, seed=7)
    txns[2].decision_type = "replacement"
    txns[2].logprob = 0.0                                             # argmax in serve → 0 by convention
    cfg = PPOConfig(tau=0.5, pair_decode=True, gimmick_terms=False, replacement_policy=False,
                    kl_coef=0.5, clip_eps=0.1)
    rng = np.random.default_rng(0)
    adv, ret = rng.standard_normal(16), rng.uniform(-1, 1, 16)
    loss, st = ppo_loss_from_batch(ac, txns, adv, ret, cfg=cfg, ref_policy=make_reference_policy(ac))
    assert bool(torch.isfinite(loss))
    assert st["ratio_mean"] == pytest.approx(1.0, abs=1e-3)
    assert abs(st["approx_kl_old_new"]) < 1e-4                      # old == new before any step
    assert st["pair_flips"] == 0.0 and abs(st["kl_to_bc"]) < 1e-5
    loss.backward()                                                  # differentiable through the pair forward
    assert any(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in ac.actor_parameters())


# ── schema plumbing ──────────────────────────────────────────────────────────
def test_pair_first_round_trips_and_stays_absent_for_legacy_steps():
    c = TrajectoryCollector("b", "p1")
    c.add_step(state=np.zeros(4, np.float32), action_s0=1, action_s1=2, pair_first=1)
    c.add_step(state=np.zeros(4, np.float32), action_s0=1, action_s1=2)
    t1, t0 = c.last_step(), c._steps[0]
    assert t0.pair_first == 1 and t1.pair_first is None
    assert Transition.from_obj(t0.to_obj()).pair_first == 1
    assert "pair_first" not in t1.to_obj() and Transition.from_obj(t1.to_obj()).pair_first is None
