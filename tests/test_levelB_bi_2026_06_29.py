"""Level B (B-i) unit tests — opponent-conditioned OUR heads (docs/levelB_opp_conditioning_design.md).

B-i adds the ARCH (off by default): the OUR action heads (our_a/our_b) additionally read the DETACHED
predicted opp-action distribution [softmax(opp_a) || softmax(opp_b)] so the policy best-responds to its own
opp prediction. The BC retrain (Level-B anchor) + adaptability probe are B-ii; the self-play A/B is B-iii.
"""
from __future__ import annotations

import copy

import pytest

pytest.importorskip("torch")
import torch

from v_dance.models.bc_model_attn import AttnBCPolicy
from v_dance.encoders.state_encoder import get_action_dim
from v_dance.play import model_io
from v_dance.selfplay.actor_critic import AttnCritic, ActorCritic

A = get_action_dim()
FOUR = ("our_a", "our_b", "opp_a", "opp_b")


def _pol(opp_cond, heads=FOUR):
    return AttnBCPolicy(d_model=32, n_heads=2, n_layers=1, heads=heads, opp_cond=opp_cond).eval()


# ── arch ────────────────────────────────────────────────────────────────────
def test_opp_cond_grows_only_the_our_heads():
    p = _pol(True)
    head_in = p.heads["opp_a"].in_features          # opp heads unchanged = 2*d_model
    assert p.heads["our_a"].in_features == head_in + 2 * A   # + softmax(opp_a) ++ softmax(opp_b)
    assert p.heads["our_b"].in_features == head_in + 2 * A
    assert p.heads["opp_b"].in_features == head_in
    assert p.opp_cond is True


def test_opp_cond_off_is_unchanged_arch():
    p = _pol(False)
    head_in = p.heads["opp_a"].in_features
    assert p.heads["our_a"].in_features == head_in   # no opp_feat appended when off
    assert p.opp_cond is False


def test_opp_cond_requires_opp_heads():
    with pytest.raises(ValueError):
        AttnBCPolicy(d_model=32, n_heads=2, n_layers=1, heads=("our_a", "our_b"), opp_cond=True)


# ── forward ───────────────────────────────────────────────────────────────────
def test_forward_shapes_all_heads():
    p = _pol(True)
    acts, gim, val = p(torch.randn(5, p.state_dim))
    for h in FOUR:
        assert acts[h].shape == (5, A)
    assert val.shape == (5,)


def test_opp_prediction_is_detached_from_our_loss():
    p = _pol(True)
    acts, _, _ = p(torch.randn(4, p.state_dim))
    acts["our_a"].sum().backward()
    # the our-head loss must NOT reach the opp head (opp_feat is detached) — keeps the predictor honest
    assert p.heads["opp_a"].weight.grad is None
    assert p.heads["opp_b"].weight.grad is None
    # ...but the our head itself (and, via enc/g, the trunk) DID get gradient
    assert p.heads["our_a"].weight.grad is not None
    assert p.mon_enc[0].weight.grad is not None


def test_our_heads_actually_use_the_opp_prediction():
    # changing the opp heads (-> a different predicted opp dist) must change the OUR logits when opp_cond,
    # and must NOT when off (proves the conditioning is wired + gated).
    x = torch.randn(4, 0 + AttnBCPolicy(d_model=32, n_heads=2, n_layers=1).state_dim)
    for cond, expect_change in ((True, True), (False, False)):
        p = _pol(cond)
        with torch.no_grad():
            base = p(x)[0]["our_a"].clone()
            p.heads["opp_a"].bias.add_(8.0 * torch.randn_like(p.heads["opp_a"].bias))  # shift opp prediction
            after = p(x)[0]["our_a"]
        changed = not torch.allclose(base, after, atol=1e-6)
        assert changed is expect_change


# ── checkpoint stamp + model_io rebuild ────────────────────────────────────────
def test_opp_cond_stamps_and_reloads(tmp_path):
    p = _pol(True)
    ac = ActorCritic(p, AttnCritic(copy.deepcopy(p)), p.head_names, p.gimmick_head_names, True)
    ck = ac.state_checkpoint()
    assert ck["config"]["opp_cond"] is True
    path = tmp_path / "opp_cond.pt"
    torch.save(ck, path)
    m, _heads = model_io.load_bc_policy(path)            # rebuild must honor the stamp (else shape mismatch)
    assert m.opp_cond is True
    assert m.heads["our_a"].in_features == m.heads["opp_a"].in_features + 2 * A
