"""15b-arch.1: the self-attention block on TeamPreviewModel.

The attention layer is what makes the synergy TAGS interact (setter<->abuser, spread<->immunity,
reverser<->debuff). It MUST (a) preserve permutation-equivariance over our roster, (b) leave legacy
mean-pool checkpoints byte-identical (no attn params when off), and (c) actually be in the gradient graph.
"""
from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

from v_dance.models.teampreview_model import TeamPreviewModel  # noqa: E402

_PERM = torch.tensor([3, 0, 5, 1, 4, 2])


def _io(B=2, V=50, F=20):
    torch.manual_seed(0)
    return (torch.randint(1, V, (B, 6)), torch.randint(1, V, (B, 6)),
            torch.randn(B, 6, F), torch.randn(B, 6, F))


def test_legacy_has_no_attn_params_and_loads_clean():
    leg = TeamPreviewModel(50, 20, hidden=16)
    assert not hasattr(leg, "self_attn") and not hasattr(leg, "cross_attn")
    # use_self_attn=False is identical to legacy (no attn module created)
    off = TeamPreviewModel(50, 20, hidden=16, use_self_attn=False)
    assert not hasattr(off, "self_attn")
    # a legacy state_dict loads into a legacy model with no missing/unexpected keys
    TeamPreviewModel(50, 20, hidden=16).load_state_dict(leg.state_dict())
    # and the self-attn model has MORE params than legacy (the block is real)
    assert TeamPreviewModel(50, 20, hidden=16, use_self_attn=True).count_parameters() > leg.count_parameters()


def test_self_attn_preserves_perm_equivariance():
    m = TeamPreviewModel(50, 20, hidden=16, use_self_attn=True).eval()
    oi, pi, of, pf = _io()
    bl, ll = m(oi, pi, of, pf)
    bl2, ll2 = m(oi[:, _PERM], pi, of[:, _PERM], pf)
    assert torch.allclose(bl[:, _PERM], bl2, atol=1e-5)
    assert torch.allclose(ll[:, _PERM], ll2, atol=1e-5)


def test_opp_permutation_invariance_with_attn():
    m = TeamPreviewModel(50, 20, hidden=16, use_self_attn=True).eval()
    oi, pi, of, pf = _io()
    bl, _ = m(oi, pi, of, pf)
    bl2, _ = m(oi, pi[:, _PERM], of, pf[:, _PERM])          # permuting opp leaves our logits unchanged
    assert torch.allclose(bl, bl2, atol=1e-5)


def test_self_attn_is_in_the_gradient_graph():
    torch.manual_seed(0)
    m = TeamPreviewModel(50, 20, hidden=16, use_self_attn=True)
    oi, pi, of, pf = _io()
    of = of.clone().requires_grad_(True)
    m(oi, pi, of, pf)[0].sum().backward()
    assert m.self_attn.in_proj_weight.grad is not None
    assert m.self_attn.in_proj_weight.grad.abs().sum() > 0          # attention actually trains
    assert of.grad is not None


def test_cross_attn_optional_runs():
    m = TeamPreviewModel(50, 20, hidden=16, use_self_attn=True, use_cross_attn=True).eval()
    oi, pi, of, pf = _io()
    bl, ll = m(oi, pi, of, pf)
    assert bl.shape == (2, 6) and ll.shape == (2, 6)
    assert hasattr(m, "cross_attn")


def test_cross_attn_preserves_perm_invariants():
    m = TeamPreviewModel(50, 20, hidden=16, use_self_attn=True, use_cross_attn=True).eval()
    oi, pi, of, pf = _io()
    bl, ll = m(oi, pi, of, pf)
    bl2, ll2 = m(oi[:, _PERM], pi, of[:, _PERM], pf)               # our-perm equivariance
    assert torch.allclose(bl[:, _PERM], bl2, atol=1e-5) and torch.allclose(ll[:, _PERM], ll2, atol=1e-5)
    bl3, _ = m(oi, pi[:, _PERM], of, pf[:, _PERM])                 # opp-perm invariance with cross-attn on
    assert torch.allclose(bl, bl3, atol=1e-5)


def test_pad_rows_isolated_from_real_mons():
    # 15b-arch.1 review fix: pad slots (all-zero feature rows) must not affect the genuine mons'
    # logits even with self-attention on — they are masked as keys and dropped from the context mean.
    m = TeamPreviewModel(50, 20, hidden=16, use_self_attn=True).eval()
    torch.manual_seed(1)
    oi = torch.randint(1, 50, (1, 6)); pi = torch.randint(1, 50, (1, 6))
    of = torch.randn(1, 6, 20); pf = torch.randn(1, 6, 20)
    of[:, 4:] = 0.0; oi[:, 4:] = 0                                  # slots 4,5 = PAD (zero feat, idx 0)
    bl, ll = m(oi, pi, of, pf)
    oi2 = oi.clone(); oi2[0, 4], oi2[0, 5] = 7, 13                  # change pad-row idx (still zero feat)
    bl2, ll2 = m(oi2, pi, of, pf)
    assert torch.allclose(bl[:, :4], bl2[:, :4], atol=1e-6)         # real mons (0-3) unaffected
    assert torch.allclose(ll[:, :4], ll2[:, :4], atol=1e-6)


def test_structural_equivariance_without_dropout_in_train():
    # equivariance is structural (not an eval-only artifact): with dropout=0 it holds in train mode too.
    m = TeamPreviewModel(50, 20, hidden=16, dropout=0.0, use_self_attn=True).train()
    oi, pi, of, pf = _io()
    bl, ll = m(oi, pi, of, pf)
    bl2, ll2 = m(oi[:, _PERM], pi, of[:, _PERM], pf)
    assert torch.allclose(bl[:, _PERM], bl2, atol=1e-5) and torch.allclose(ll[:, _PERM], ll2, atol=1e-5)


def test_teammate_bias_applies_and_stays_perm_equivariant():
    m = TeamPreviewModel(50, 20, hidden=16, use_self_attn=True, use_teammate_bias=True).eval()
    assert hasattr(m, "teammate_scale")
    oi, pi, of, pf = _io()
    aff = torch.rand(2, 6, 6); aff = (aff + aff.transpose(1, 2)) / 2          # symmetric affinity
    bl, ll = m(oi, pi, of, pf, our_affinity=aff)
    bl0, _ = m(oi, pi, of, pf, our_affinity=None)
    assert (bl - bl0).abs().max() > 1e-4                                       # the prior changes the output
    # permute roster AND affinity rows+cols consistently -> equivariant
    aff_p = aff[:, _PERM][:, :, _PERM]
    bl2, ll2 = m(oi[:, _PERM], pi, of[:, _PERM], pf, our_affinity=aff_p)
    assert torch.allclose(bl[:, _PERM], bl2, atol=1e-5) and torch.allclose(ll[:, _PERM], ll2, atol=1e-5)


def test_teammate_scale_trains():
    torch.manual_seed(0)
    m = TeamPreviewModel(50, 20, hidden=16, use_self_attn=True, use_teammate_bias=True)
    oi, pi, of, pf = _io()
    m(oi, pi, of, pf, our_affinity=torch.rand(2, 6, 6))[0].sum().backward()
    assert m.teammate_scale.grad is not None and m.teammate_scale.grad.abs() > 0
