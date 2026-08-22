"""B1 unit tests for the C51 distributional value head (docs/c51_value_head_design.md).

Covers the PURE math + arch pieces built in B1 (the projection, the CE loss, the atoms head,
value_dist / value_pm, and the scalar-critic / forward-parity back-compat). The end-to-end
wiring (ppo_forward -> ppo_losses -> _value_loss, warm-up two-hot init, checkpoint) is B2/B3.
"""
import pytest
import torch

from v_dance.selfplay.ppo import (
    PPOConfig, c51_support, project_returns_to_support, c51_value_loss, _value_loss,
)
from v_dance.selfplay.actor_critic import AttnCritic
from v_dance.models.bc_model_attn import AttnBCPolicy


def _cfg(n_atoms=51):
    return PPOConfig(value_loss_mode="c51", n_atoms=n_atoms, v_min=-1.0, v_max=1.0)


def _small_policy(n_value_atoms=0):
    return AttnBCPolicy(d_model=32, n_heads=2, n_layers=1, n_value_atoms=n_value_atoms).eval()


# ── support + projection ──────────────────────────────────────────────────────
def test_support_is_uniform_linspace():
    z = c51_support(_cfg(51))
    assert z.numel() == 51
    assert torch.isclose(z[0], torch.tensor(-1.0)) and torch.isclose(z[-1], torch.tensor(1.0))
    d = z[1:] - z[:-1]
    assert torch.allclose(d, d[0].expand_as(d), atol=1e-6)


def test_projection_dirac_at_atom_is_onehot():
    z = c51_support(_cfg(51))
    for k in (0, 13, 25, 50):
        m = project_returns_to_support(z[k:k + 1].clone(), z)
        assert torch.isclose(m.sum(), torch.tensor(1.0), atol=1e-5)
        assert torch.argmax(m, dim=1).item() == k
        assert torch.isclose(m[0, k], torch.tensor(1.0), atol=1e-5)


def test_projection_rows_sum_to_one_and_mean_matches_target():
    z = c51_support(_cfg(51))
    r = torch.tensor([-0.97, -0.5, -0.02, 0.0, 0.03, 0.5, 0.97, 0.41])
    m = project_returns_to_support(r, z)
    assert torch.allclose(m.sum(dim=1), torch.ones(r.shape[0]), atol=1e-5)
    assert (m >= 0).all()
    assert torch.allclose((m * z).sum(dim=1), r, atol=1e-4)   # two-hot mean == target


def test_projection_clamps_out_of_range():
    z = c51_support(_cfg(51))
    m = project_returns_to_support(torch.tensor([-5.0, 5.0]), z)
    assert torch.allclose(m.sum(dim=1), torch.ones(2), atol=1e-5)
    assert torch.allclose((m * z).sum(dim=1), torch.tensor([-1.0, 1.0]), atol=1e-4)
    assert torch.isclose(m[0, 0], torch.tensor(1.0), atol=1e-5)     # all mass at v_min
    assert torch.isclose(m[1, -1], torch.tensor(1.0), atol=1e-5)    # all mass at v_max


# ── CE loss ───────────────────────────────────────────────────────────────────
def test_c51_loss_minimized_at_target():
    z = c51_support(_cfg(21))
    r = torch.tensor([0.3, -0.4, 0.0])
    target = project_returns_to_support(r, z)
    perfect_logits = torch.log(target.clamp_min(1e-9))             # softmax ~= target
    loss_perfect = c51_value_loss(perfect_logits, r, z)
    ent = -(target * torch.log(target.clamp_min(1e-9))).sum(dim=1).mean()
    assert torch.isclose(loss_perfect, ent, atol=1e-3)            # CE(p,p) == H(p)
    loss_uniform = c51_value_loss(torch.zeros_like(target), r, z)  # zeros -> uniform softmax
    assert float(loss_uniform) > float(loss_perfect) + 1e-2


def test_c51_loss_finite_and_differentiable():
    z = c51_support(_cfg(51))
    logits = torch.randn(8, 51, requires_grad=True)
    r = torch.empty(8).uniform_(-1, 1)
    loss = c51_value_loss(logits, r, z)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


# ── critic / model integration ────────────────────────────────────────────────
def test_value_pm_in_range_and_equals_dist_mean():
    torch.manual_seed(0)
    pol = _small_policy(n_value_atoms=51)
    crit = AttnCritic(pol, support=c51_support(_cfg(51))).eval()
    x = torch.randn(16, pol.state_dim)
    with torch.no_grad():
        dist, vpm = crit.value_dist(x), crit.value_pm(x)
    assert crit.is_c51
    assert dist.shape == (16, 51)
    assert torch.allclose(dist.sum(dim=1), torch.ones(16), atol=1e-5)
    assert (vpm <= 1.0 + 1e-5).all() and (vpm >= -1.0 - 1e-5).all()
    assert torch.allclose(vpm, (dist * crit.support).sum(dim=1), atol=1e-5)


def test_value_atoms_logits_shape_batched_and_unbatched():
    pol = _small_policy(n_value_atoms=51)
    x = torch.randn(4, pol.state_dim)
    assert pol.value_atoms_logits(x).shape == (4, 51)
    assert pol.value_atoms_logits(x[0]).shape == (51,)


def test_scalar_critic_unchanged_when_no_support():
    pol = _small_policy(n_value_atoms=0)
    crit = AttnCritic(pol)                       # scalar critic (back-compat)
    assert crit.support is None and not crit.is_c51
    x = torch.randn(5, pol.state_dim)
    with torch.no_grad():
        assert torch.allclose(crit.value_pm(x), 2.0 * torch.sigmoid(crit.forward(x)) - 1.0, atol=1e-6)
    with pytest.raises(RuntimeError):
        crit.value_dist(x)


def test_no_atoms_head_when_disabled():
    pol = _small_policy(n_value_atoms=0)
    assert pol.value_atoms_head is None
    with pytest.raises(RuntimeError):
        pol.value_atoms_logits(torch.randn(2, pol.state_dim))


def test_forward_unchanged_three_tuple_with_atoms_head():
    pol = _small_policy(n_value_atoms=51)        # atoms head present, forward shape unchanged
    actions, gimmicks, value = pol(torch.randn(3, pol.state_dim))
    assert value.shape == (3,)
    assert isinstance(actions, dict) and len(actions) > 0


def test_value_loss_c51_branch():
    cfg = _cfg(51)
    r = torch.empty(8).uniform_(-1, 1)
    dummy = torch.zeros(8)
    with pytest.raises(ValueError):                # c51 requires atoms_logits
        _value_loss(dummy, dummy, r, cfg)
    loss = _value_loss(dummy, dummy, r, cfg, atoms_logits=torch.randn(8, 51))
    assert torch.isfinite(loss)
