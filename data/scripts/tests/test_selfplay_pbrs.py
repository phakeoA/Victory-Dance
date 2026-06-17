"""Task 3b.7: gated PBRS — frozen Phi, Phi(terminal)=0, telescoping invariance,
shaping-fraction cap + lambda anneal, default OFF.
"""
from __future__ import annotations

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
from self_play.reward import place_terminal_reward  # noqa: E402
from self_play.gae import compute_batch_gae, DEFAULT_GAMMA  # noqa: E402
from self_play import pbrs  # noqa: E402

STATE_DIM, ACTION_DIM = get_state_dim(), get_action_dim()


def _write_ckpt(path):
    torch.manual_seed(4)
    m = BCPolicy(state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_dims=(8, 4),
                 dropout=0.0, gimmick_dim=get_gimmick_dim())
    torch.save({"model_state": m.state_dict(), "config": {
        "state_dim": STATE_DIM, "action_dim": ACTION_DIM, "hidden_dims": (8, 4),
        "dropout": 0.0, "heads": ("our_a", "our_b"), "gimmick_dim": get_gimmick_dim(),
        "gimmick_trained": True, "value_trained": True,
        "state_layout_version": get_state_layout_version()}}, path)
    return path


@pytest.fixture(scope="module")
def ac(tmp_path_factory):
    return ActorCritic.from_bc_checkpoint(_write_ckpt(tmp_path_factory.mktemp("pbrs") / "bc.pt"))


@pytest.fixture(scope="module")
def shaper(ac):
    return pbrs.PotentialShaper.from_critic(ac.critic, coef=0.2)


def _traj(won=True, n=4, seed=0):
    c = TrajectoryCollector(f"g{seed}", "p1")
    rng = np.random.default_rng(seed)
    for t in range(n):
        c.add_step(state=rng.standard_normal(STATE_DIM).astype(np.float32),
                   action_s0=t % ACTION_DIM, action_s1=PASS_ACTION, mask_s0=[1] * ACTION_DIM,
                   logprob=-1.0, value=0.05, turn=t + 1)
    tr = c.finish(own_team=["a"] * 6, opp_team=["b"] * 6, tp_bring=[0, 1, 2, 3],
                  tp_leads=[0, 1], won=won, terminal_type="win" if won else "loss")
    place_terminal_reward(tr)
    return tr


# ── frozen Phi ────────────────────────────────────────────────────────────────
def test_phi_separate_and_frozen(ac, shaper):
    assert id(shaper) != id(ac.critic)
    assert all(not p.requires_grad for p in shaper.parameters())
    ac_store = {p.data_ptr() for p in ac.critic.parameters()}
    sh_store = {p.data_ptr() for p in shaper.parameters()}
    assert ac_store.isdisjoint(sh_store)               # deep-copied, no shared tensors


def test_phi_terminal_zero(shaper):
    assert pbrs.PotentialShaper.phi_terminal() == 0.0
    pbrs.assert_phi_terminal_zero(shaper)


def test_phi_in_potential_range(shaper):
    p = shaper.phi_values(np.random.default_rng(0).standard_normal((20, STATE_DIM)).astype(np.float32))
    assert np.all(p >= -0.2 - 1e-6) and np.all(p <= 0.2 + 1e-6)   # coef-scaled 2sigma-1


# ── F formula + telescoping invariance ────────────────────────────────────────
def test_F_formula_and_terminal(shaper):
    tr = _traj(won=True, n=4)
    phi, F = pbrs._potential_and_F(tr, shaper, gamma=DEFAULT_GAMMA)
    # interior: F_t = gamma*phi[t+1] - phi[t]
    for t in range(len(F) - 1):
        assert F[t] == pytest.approx(DEFAULT_GAMMA * phi[t + 1] - phi[t], abs=1e-6)
    # terminal: Phi(s_T)=0 -> F = -phi[n-1] (NO gamma*Phi(terminal) inflation)
    assert F[-1] == pytest.approx(-phi[-1], abs=1e-6)


def test_telescoping_residual_zero(shaper):
    for won in (True, False):
        tr = _traj(won=won, n=6, seed=1)
        assert abs(pbrs.telescoping_residual(tr, shaper, gamma=DEFAULT_GAMMA)) < 1e-4


# ── shaping application ───────────────────────────────────────────────────────
def test_shape_trajectory_adds_lambda_F(shaper):
    tr = _traj(won=True, n=4)
    base = [t.reward for t in tr.transitions]
    F = pbrs.shape_trajectory(tr, shaper, gamma=DEFAULT_GAMMA, lam=0.5)
    for t, transition in enumerate(tr.transitions):
        assert transition.reward == pytest.approx(base[t] + 0.5 * F[t], abs=1e-6)
    # terminal reward changed only by -lambda*Phi(s_{n-1}) (no win-correlated +gamma*Phi(s_T))
    phi, _ = pbrs._potential_and_F(_traj(won=True, n=4), shaper, gamma=DEFAULT_GAMMA)
    assert tr.transitions[-1].reward == pytest.approx(base[-1] + 0.5 * (-phi[-1]), abs=1e-6)


def test_lambda_zero_is_noop(shaper):
    tr = _traj(won=True, n=4)
    base = [t.reward for t in tr.transitions]
    pbrs.shape_trajectory(tr, shaper, gamma=DEFAULT_GAMMA, lam=0.0)
    assert [t.reward for t in tr.transitions] == base


# ── shaping fraction + cap + anneal ───────────────────────────────────────────
def test_shaping_fraction_and_anneal(shaper):
    trajs = [_traj(won=True, n=5, seed=1), _traj(won=False, n=5, seed=2)]
    f_full = pbrs.shaping_fraction(trajs, shaper, gamma=DEFAULT_GAMMA, lam=1.0)
    f_half = pbrs.shaping_fraction(trajs, shaper, gamma=DEFAULT_GAMMA, lam=0.5)
    assert f_full >= 0.0
    assert f_half == pytest.approx(0.5 * f_full, abs=1e-6)        # linear in lambda
    assert pbrs.shaping_fraction(trajs, shaper, lam=0.0) == pytest.approx(0.0)  # anneal -> 0


def test_assert_shaping_fraction(shaper):
    trajs = [_traj(won=True, n=4, seed=1)]
    pbrs.assert_shaping_fraction(trajs, shaper, lam=0.01, cap=0.3)   # tiny lambda -> passes
    with pytest.raises(AssertionError, match="shaping fraction"):
        pbrs.assert_shaping_fraction(trajs, shaper, lam=1.0, cap=1e-6)  # impossible cap


# ── default OFF + GAE-on-shaped ───────────────────────────────────────────────
def test_default_off_is_noop(shaper):
    tr = _traj(won=True, n=4)
    base = [t.reward for t in tr.transitions]
    out = pbrs.maybe_shape_batch([tr], shaper, pbrs.PBRSConfig(enabled=False))
    assert out["applied"] is False
    assert [t.reward for t in tr.transitions] == base               # untouched


def test_enabled_shapes_and_gae_runs(shaper):
    trajs = [_traj(won=True, n=5, seed=1), _traj(won=False, n=5, seed=2)]
    base = [[t.reward for t in tr.transitions] for tr in trajs]
    out = pbrs.maybe_shape_batch(trajs, shaper, pbrs.PBRSConfig(enabled=True, coef=0.2, lam=0.2))
    assert out["applied"] is True
    assert any([t.reward for t in tr.transitions] != b for tr, b in zip(trajs, base))
    adv, ret = compute_batch_gae(trajs, standardize_adv=False)      # GAE on shaped rewards
    assert np.all(np.isfinite(adv)) and np.all(np.isfinite(ret))
