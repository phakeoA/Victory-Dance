"""#23: the per-mon SET-ATTENTION battle-net candidate (AttnBCPolicy).

It must (a) honour the exact BCPolicy forward contract so it is a drop-in,
(b) reshape the frozen v4 flat layout correctly (12x149 + 78 global),
(c) be PAD-SAFE on empty slots (no NaN), and (d) actually COUPLE the mons via
attention (perturbing a bench mon changes an active slot's logits) — the whole
reason the architecture exists.
"""
from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

from v_dance.encoders.state_encoder import (  # noqa: E402
    get_state_dim,
    get_action_dim,
    get_gimmick_dim,
    POKEMON_FEATURES,
)
from v_dance.models.bc_model_attn import AttnBCPolicy, build_attn_model  # noqa: E402

SD = get_state_dim()
AD = get_action_dim()
GD = get_gimmick_dim()
PF = POKEMON_FEATURES


def _slot(i: int) -> slice:
    """Feature span of mon-slot i in the flat vector."""
    return slice(i * PF, (i + 1) * PF)


def _rand(B: int = 4) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(B, SD)


def test_forward_contract():
    """The (actions, gimmicks, value) forward contract: head names, shapes, finite outputs."""
    m = AttnBCPolicy(d_model=32, n_layers=1).eval()
    x = _rand()
    actions, gimmicks, value = m(x)
    assert set(actions) == {"our_a", "our_b"}
    assert set(gimmicks) == {"our_a", "our_b"}
    for k in actions:
        assert actions[k].shape == (4, AD)
    for k in gimmicks:
        assert gimmicks[k].shape == (4, GD)
    assert value.shape == (4,)
    for t in (*actions.values(), *gimmicks.values(), value):
        assert torch.isfinite(t).all()


def test_state_dim_guard():
    """A wrong layout width must fail LOUD, never silently zero-pad."""
    with pytest.raises(ValueError):
        AttnBCPolicy(state_dim=SD + 1)
    m = AttnBCPolicy(d_model=16, n_layers=1).eval()
    with pytest.raises(ValueError):
        m(torch.randn(2, SD - 1))


def test_pad_safe_only_actives_present():
    """Only the 2 own-active mons present (bench + opp all empty) -> no NaN."""
    m = AttnBCPolicy(d_model=32, n_layers=2).eval()
    x = torch.zeros(3, SD)
    # own actives = slots 0,1; leave 2..11 empty (no mon). global = slots end.
    x[:, _slot(0)] = torch.randn(3, PF)
    x[:, _slot(1)] = torch.randn(3, PF)
    x[:, 12 * PF:] = torch.randn(3, SD - 12 * PF)
    actions, gimmicks, value = m(x)
    for t in (*actions.values(), *gimmicks.values(), value):
        assert torch.isfinite(t).all(), "pad-safety: empty slots produced NaN/inf"


def test_fully_empty_row_does_not_nan():
    """Degenerate all-zero mon batch row (guard against fully-masked attention)."""
    m = AttnBCPolicy(d_model=16, n_layers=1).eval()
    x = torch.zeros(2, SD)
    actions, gimmicks, value = m(x)
    for t in (*actions.values(), *gimmicks.values(), value):
        assert torch.isfinite(t).all()


def test_deterministic_in_eval():
    m = AttnBCPolicy(d_model=32, n_layers=2).eval()
    x = _rand()
    a1, g1, v1 = m(x)
    a2, g2, v2 = m(x)
    assert torch.equal(v1, v2)
    for k in a1:
        assert torch.equal(a1[k], a2[k])


def test_attention_couples_mons():
    """Perturbing a BENCH mon (slot 5) must change an ACTIVE slot's logits —
    proof the self-attention actually wires the 12 tokens together (a flat
    per-mon encoder reading only token 0 would be unaffected)."""
    m = AttnBCPolicy(d_model=32, n_layers=2).eval()
    x = _rand(2)
    base, _, base_v = m(x)
    x2 = x.clone()
    x2[:, _slot(5)] = x2[:, _slot(5)] + 3.0  # nudge own bench mon
    pert, _, pert_v = m(x2)
    assert not torch.allclose(base["our_a"], pert["our_a"]), "attention not coupling mons"
    assert not torch.allclose(base_v, pert_v)


def test_attention_couples_under_mask():
    """Coupling must hold on a REALISTICALLY SPARSE board (most slots empty, so
    key_padding_mask is active) — not just the all-12-present case. Perturbing
    the one present bench mon changes our_a logits, finitely."""
    m = AttnBCPolicy(d_model=32, n_layers=2).eval()
    x = torch.zeros(2, SD)
    for i in (0, 1, 2, 3, 4):           # own+opp actives + one own bench present
        x[:, _slot(i)] = torch.randn(2, PF)
    x[:, 12 * PF:] = torch.randn(2, SD - 12 * PF)
    base, _, _ = m(x)
    x2 = x.clone()
    x2[:, _slot(4)] = x2[:, _slot(4)] + 3.0   # nudge the lone present bench mon
    pert, _, _ = m(x2)
    assert torch.isfinite(pert["our_a"]).all()
    assert not torch.allclose(base["our_a"], pert["our_a"]), "no coupling under mask"


def test_empty_own_active_slot_finite():
    """Forced-replacement state: own_a (slot 0) is empty. The our_a head still
    reads its (masked) token and must produce FINITE logits — legality masking is
    the downstream authority, but the net must never emit NaN."""
    m = AttnBCPolicy(d_model=32, n_layers=2).eval()
    x = torch.zeros(2, SD)
    for i in (1, 2, 3):                  # own_b + both opp actives present; own_a empty
        x[:, _slot(i)] = torch.randn(2, PF)
    x[:, 12 * PF:] = torch.randn(2, SD - 12 * PF)
    actions, gimmicks, value = m(x)
    for t in (*actions.values(), *gimmicks.values(), value):
        assert torch.isfinite(t).all()


def test_auxiliary_opp_heads():
    """Extra opp action heads read the opp active tokens; gimmick stays own-only."""
    m = AttnBCPolicy(
        d_model=16, n_layers=1, heads=("our_a", "our_b", "opp_a", "opp_b")
    ).eval()
    actions, gimmicks, _ = m(_rand(2))
    assert set(actions) == {"our_a", "our_b", "opp_a", "opp_b"}
    assert set(gimmicks) == {"our_a", "our_b"}  # gimmick default = own actives only
    for k in actions:
        assert actions[k].shape == (2, AD)


def test_builder_and_param_count():
    m = build_attn_model(d_model=32, n_layers=2, device="cpu")
    assert m.count_parameters() > 10_000
    # a compact net: the small-width attn stays well under a million params
    assert m.count_parameters() < 1_000_000
