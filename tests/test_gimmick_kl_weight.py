"""Lock the per-head gimmick-KL slack (mega-bias fix): `gimmick_kl_weight` in `_joint_kl` scales the MEGA
(gimmick) head's KL-to-BC contribution ONLY, linearly, leaving the move-policy KL untouched. This is what
lets RL un-learn the BC "always mega turn 1" data-bias while keeping the move policy anchored.
"""
from __future__ import annotations

import torch

from v_dance.selfplay.policy_eval import _joint_kl, _SlotBatch


def _slot(gimmick: bool):
    B, A, G = 2, 4, 3
    return _SlotBatch(
        head="h",
        acts=torch.tensor([0, 1]),
        valid=torch.tensor([True, True]),
        amask=torch.ones(B, A, dtype=torch.bool),
        gims=torch.tensor([1, 0]),
        gmask=torch.ones(B, G, dtype=torch.bool),
        has_g=torch.tensor([gimmick, gimmick]),
        gimmick=gimmick,
    )


# Deterministic logits with new != ref on BOTH heads → both KLs are strictly positive.
_NEW_A = {"h": torch.tensor([[2., 0., 0., 0.], [0., 1., 0., 0.]])}
_REF_A = {"h": torch.tensor([[0., 0., 0., 2.], [0., 0., 1., 0.]])}
_NEW_G = {"h": torch.tensor([[1., 0., 0.], [0., 2., 0.]])}
_REF_G = {"h": torch.tensor([[0., 0., 1.], [2., 0., 0.]])}


def _kl(w, slot):
    return _joint_kl(_NEW_A, _NEW_G, _REF_A, _REF_G, [slot], tau=1.0, gimmick_kl_weight=w)


def test_gimmick_weight_scales_only_the_mega_head_linearly():
    sb = _slot(gimmick=True)
    full = _kl(1.0, sb)      # action_kl + gimmick_kl
    none = _kl(0.0, sb)      # action_kl only (mega head fully unanchored)
    half = _kl(0.5, sb)

    gimmick_contrib = full - none
    assert (gimmick_contrib > 1e-6).all(), "the gimmick head should contribute positive KL at weight 1"
    # weight scales ONLY the gimmick term, linearly: kl(w) == action_kl + w*gimmick_kl
    assert torch.allclose(half, none + 0.5 * gimmick_contrib, atol=1e-6)
    assert torch.allclose(_kl(1.0, sb), none + gimmick_contrib, atol=1e-6)


def test_default_weight_unchanged_and_no_gimmick_slot_unaffected():
    sb = _slot(gimmick=True)
    # default 1.0 must equal an explicit 1.0 (byte-identical to the old joint KL)
    assert torch.allclose(_joint_kl(_NEW_A, _NEW_G, _REF_A, _REF_G, [sb], tau=1.0), _kl(1.0, sb))
    # a slot with NO gimmick head: the weight is a no-op (move-policy KL only)
    ng = _slot(gimmick=False)
    assert torch.allclose(_kl(0.0, ng), _kl(1.0, ng), atol=1e-7)
