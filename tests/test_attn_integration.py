"""#23 integration — the AttnBCPolicy plugs into the serve/load + actor-critic + PBRS
infra as a drop-in for the flat BCPolicy:

  * model_io.load_bc_policy arch-dispatches on config model_type=='attn'
  * all 3 value-readout arms (mean / concat_active / cls_query) load + serve
  * the shared-trunk AttnCritic is LEAK-FREE: a critic step leaves the actor
    bit-identical and an actor step leaves the critic bit-identical
  * ActorCritic.save(verify=True) round-trips an attn checkpoint through the loader
  * PBRS (default-OFF) does not crash with an attn critic
The flat BCPolicy was retired in the attn-only refactor (#27); a legacy flat checkpoint is rejected.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

import v_dance.play.model_io as M  # noqa: E402
from v_dance.models.bc_model_attn import AttnBCPolicy  # noqa: E402
from v_dance.encoders.state_encoder import (  # noqa: E402
    get_state_dim, get_state_layout_version, get_gimmick_dim,
)

SD = get_state_dim()


def _save_attn_ckpt(tmp_path, value_readout="mean", value_trained=True):
    m = AttnBCPolicy(d_model=32, n_heads=4, n_layers=1, value_readout=value_readout)
    cfg = {
        "model_type": "attn", "state_dim": SD,
        "state_layout_version": get_state_layout_version(),
        "action_dim": 16, "gimmick_dim": get_gimmick_dim(),
        "d_model": 32, "n_heads": 4, "n_layers": 1, "ff_mult": 2, "dropout": 0.0,
        "heads": ["our_a", "our_b"], "gimmick_heads": ["our_a", "our_b"],
        "value_readout": value_readout,
        "value_trained": value_trained, "gimmick_trained": True,
    }
    p = tmp_path / f"attn_{value_readout}.pt"
    torch.save({"model_state": m.state_dict(), "config": cfg}, p)
    return p


# ── load dispatch + readout arms ──────────────────────────────────────────────
@pytest.mark.parametrize("readout", ["mean", "concat_active", "cls_query"])
def test_attn_loads_and_serves(tmp_path, readout):
    model, heads = M.load_bc_policy(_save_attn_ckpt(tmp_path, readout))
    assert isinstance(model, AttnBCPolicy)
    assert model.value_readout == readout and model.d_model == 32 and model.n_layers == 1
    sv = np.zeros(SD, dtype=np.float32)
    v = M.value_logit(model, sv)
    assert v is not None and 0.0 <= v <= 1.0 and M.value_trained(model)
    a0, a1 = M.bc_action_indices(model, heads, sv, [True] * 16, [True] * 16)
    assert a0 is not None and a1 is not None
    g0, g1 = M.gimmick_logits(model, heads, sv)
    assert g0.shape == (get_gimmick_dim(),)


def test_flat_checkpoint_rejected(tmp_path):
    # Attn-only refactor (#27): a legacy flat checkpoint (no model_type=="attn") is no longer
    # loadable — the loader rejects it loudly rather than silently mis-building a model.
    cfg = {"state_dim": SD, "state_layout_version": get_state_layout_version(),
           "action_dim": 16, "gimmick_dim": 2, "hidden_dims": [16], "dropout": 0.0,
           "heads": ["our_a", "our_b"]}
    p = tmp_path / "flat.pt"
    torch.save({"model_state": {}, "config": cfg}, p)
    with pytest.raises(ValueError, match="not an attn checkpoint"):
        M.load_bc_policy(p)


def test_attn_1d_2d_parity():
    m = AttnBCPolicy(d_model=32, n_heads=4, n_layers=1).eval()
    with torch.no_grad():
        o1 = m(torch.zeros(SD))
        o2 = m(torch.zeros(1, SD))
    assert torch.equal(o1[0]["our_a"], o2[0]["our_a"][0])
    assert torch.equal(o1[2], o2[2][0])              # value: scalar vs (1,)


# ── shared-trunk AttnCritic: leak-free invariant ──────────────────────────────
def test_attn_actor_critic_is_leak_free(tmp_path):
    from v_dance.selfplay.actor_critic import ActorCritic, AttnCritic
    ac = ActorCritic.from_bc_checkpoint(_save_attn_ckpt(tmp_path), require_value_trained=False)
    assert isinstance(ac.critic, AttnCritic)
    x = torch.randn(4, SD)

    # (1) a CRITIC-only step must leave EVERY actor param bit-identical.
    actor_before = [p.detach().clone() for p in ac.actor_parameters()]
    copt = torch.optim.SGD(ac.critic_parameters(), lr=0.5)
    copt.zero_grad()
    ac.value_logit(x).pow(2).mean().backward()
    copt.step()
    assert all(torch.equal(b, p) for b, p in zip(actor_before, ac.actor_parameters()))
    assert any(p.grad is not None for p in ac.critic_parameters())  # critic actually got grad

    # (2) an ACTOR-only step (action loss) must leave EVERY critic param bit-identical.
    critic_before = [p.detach().clone() for p in ac.critic_parameters()]
    aopt = torch.optim.SGD(ac.actor_parameters(), lr=0.5)
    aopt.zero_grad()
    actions, _g, _v = ac.policy(x)
    sum(t.pow(2).mean() for t in actions.values()).backward()
    aopt.step()
    assert all(torch.equal(b, p) for b, p in zip(critic_before, ac.critic_parameters()))


def test_attn_state_checkpoint_roundtrips(tmp_path):
    from v_dance.selfplay.actor_critic import ActorCritic
    ac = ActorCritic.from_bc_checkpoint(
        _save_attn_ckpt(tmp_path, "concat_active"), require_value_trained=False)
    out = tmp_path / "ac.pt"
    ac.save(out, generation=7, verify=True)          # verify=True reloads via model_io -> must pass
    pol, _heads = M.load_bc_policy(out)
    assert isinstance(pol, AttnBCPolicy) and pol.value_readout == "concat_active"
    ac.restore_from(out)                              # policy + critic restore (no raise)


# ── PBRS (default-OFF) must not crash with an attn critic ─────────────────────
def test_pbrs_from_attn_critic_no_crash(tmp_path):
    from v_dance.selfplay.actor_critic import ActorCritic
    from v_dance.selfplay.pbrs import PotentialShaper
    ac = ActorCritic.from_bc_checkpoint(_save_attn_ckpt(tmp_path), require_value_trained=False)
    shaper = PotentialShaper.from_critic(ac.critic, coef=0.2)
    phi = shaper.phi_values(np.zeros((3, SD), dtype=np.float32))
    assert phi.shape == (3,) and np.isfinite(phi).all() and (np.abs(phi) <= 0.2 + 1e-6).all()
    assert all(not p.requires_grad for p in shaper.parameters())   # frozen
