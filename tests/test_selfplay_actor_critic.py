"""Task 3b.1: actor-critic init FROM the BC checkpoint (cloned separate critic).

Verifies the sec-2 contract: the critic is a *separate* clone of the BC
trunk+value_head (identical at init, but its own tensors), policy and critic
optimise independently, and the value-space helpers (logit / winprob / value_pm)
are consistent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("numpy")
import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
from v_dance.encoders.state_encoder import get_state_dim, get_action_dim, get_gimmick_dim, get_state_layout_version  # noqa: E402
from conftest import write_attn_ckpt  # noqa: E402
import v_dance.play.model_io as model_io  # noqa: E402
from v_dance.selfplay.actor_critic import ActorCritic, AttnCritic  # noqa: E402

STATE_DIM = get_state_dim()
ACTION_DIM = get_action_dim()
GIMMICK_DIM = get_gimmick_dim()


def test_optional_annotation_resolves():
    """Regression (pre-refactor scan): ``actor_critic`` annotated ``generation:
    Optional[int]`` on ``state_checkpoint``/``save`` without importing ``Optional``
    (masked by ``from __future__ import annotations``).  Resolving the hints exercises
    the module namespace and would NameError if the import were missing again."""
    import typing
    typing.get_type_hints(ActorCritic.state_checkpoint)
    typing.get_type_hints(ActorCritic.save)


def _write_ckpt(tmp_path, value_trained=True, gimmick_trained=True, name="bc.pt") -> Path:
    """A tiny but layout-CURRENT attn BC checkpoint (small net for speed). state_dim /
    layout_version must match the live encoder or model_io's stale-layout guard rejects it."""
    return write_attn_ckpt(tmp_path / name, value_trained=value_trained,
                           gimmick_trained=gimmick_trained)


def _x(b=3):
    g = torch.Generator().manual_seed(7)
    return torch.randn(b, STATE_DIM, generator=g)


def test_loads_and_reports_heads(tmp_path):
    ac = ActorCritic.from_bc_checkpoint(_write_ckpt(tmp_path))
    assert ac.head_names == ("our_a", "our_b")
    assert ac.gimmick_head_names == ("our_a", "our_b")
    assert ac.value_trained is True
    assert isinstance(ac.critic, AttnCritic)


def test_requires_value_trained_by_default(tmp_path):
    p = _write_ckpt(tmp_path, value_trained=False)
    with pytest.raises(ValueError, match="UNTRAINED value head"):
        ActorCritic.from_bc_checkpoint(p)
    # explicit override builds a (cold) critic without raising
    ac = ActorCritic.from_bc_checkpoint(p, require_value_trained=False)
    assert ac.value_trained is False


def test_critic_is_separate_object_no_shared_params(tmp_path):
    ac = ActorCritic.from_bc_checkpoint(_write_ckpt(tmp_path))
    assert id(ac.critic) != id(ac.policy)
    actor_ids = {id(p) for p in ac.policy.parameters()}
    critic_ids = {id(p) for p in ac.critic.parameters()}
    assert actor_ids.isdisjoint(critic_ids)
    # storage pointers disjoint too (deepcopy allocates fresh tensors)
    actor_store = {p.data_ptr() for p in ac.policy.parameters()}
    critic_store = {p.data_ptr() for p in ac.critic.parameters()}
    assert actor_store.isdisjoint(critic_store)


def test_critic_init_equals_bc_value_head(tmp_path):
    """At init the cloned critic logit == the BC policy's value logit (exact copy)."""
    ac = ActorCritic.from_bc_checkpoint(_write_ckpt(tmp_path))
    ac.policy.eval(); ac.critic.eval()
    x = _x()
    with torch.no_grad():
        _, _, bc_value = ac.policy(x)          # actor's own value head
        critic_logit = ac.critic(x)
    assert torch.allclose(bc_value, critic_logit, atol=1e-6)


def test_value_space_helpers(tmp_path):
    ac = ActorCritic.from_bc_checkpoint(_write_ckpt(tmp_path))
    x = _x()
    with torch.no_grad():
        logit = ac.value_logit(x)
        p = ac.winprob(x)
        vpm = ac.value_pm(x)
    assert torch.all(p >= 0) and torch.all(p <= 1)            # win-prob in [0,1]
    assert torch.allclose(vpm, 2.0 * p - 1.0, atol=1e-6)      # pm == 2p-1
    assert torch.all(vpm >= -1) and torch.all(vpm <= 1)       # pm in [-1,1]
    assert torch.allclose(p, torch.sigmoid(logit), atol=1e-6)


def test_forward_shapes(tmp_path):
    ac = ActorCritic.from_bc_checkpoint(_write_ckpt(tmp_path))
    actions, gimmicks, value = ac(_x(b=5))
    assert set(actions) == {"our_a", "our_b"}
    assert actions["our_a"].shape == (5, ACTION_DIM)
    assert set(gimmicks) == {"our_a", "our_b"}
    assert gimmicks["our_a"].shape == (5, GIMMICK_DIM)
    assert value.shape == (5,)


def test_critic_update_leaves_actor_untouched(tmp_path):
    """A gradient step on the critic must not change ANY actor parameter (decoupled)."""
    ac = ActorCritic.from_bc_checkpoint(_write_ckpt(tmp_path))
    actor_before = [p.detach().clone() for p in ac.actor_parameters()]
    critic_before = [p.detach().clone() for p in ac.critic_parameters()]

    opt = torch.optim.SGD(ac.critic_parameters(), lr=0.1)
    opt.zero_grad()
    loss = ac.critic(_x()).pow(2).mean()       # arbitrary critic-only loss
    loss.backward()
    # the actor params must have NO grad from a critic-only loss
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0
               for p in ac.actor_parameters())
    opt.step()

    for b, p in zip(actor_before, ac.actor_parameters()):
        assert torch.equal(b, p)               # actor bit-identical
    assert any(not torch.equal(b, p)
               for b, p in zip(critic_before, ac.critic_parameters()))  # critic moved


def test_actor_parameters_exclude_value_head(tmp_path):
    ac = ActorCritic.from_bc_checkpoint(_write_ckpt(tmp_path))
    actor_ids = {id(p) for p in ac.actor_parameters()}
    vh_ids = {id(p) for p in ac.policy.value_head.parameters()}
    assert actor_ids.isdisjoint(vh_ids)        # vestigial value head excluded
    # the per-mon encoder + both action heads + both gimmick heads ARE included
    enc_ids = {id(p) for p in ac.policy.mon_enc.parameters()}
    head_ids = {id(p) for n, p in ac.policy.named_parameters()
                if n.startswith("heads.") or n.startswith("gimmick_heads.")}
    assert enc_ids.issubset(actor_ids)
    assert head_ids.issubset(actor_ids)


def test_save_is_bc_loadable_and_restores_both_nets(tmp_path):
    """save() writes a gauntlet-loadable BC checkpoint that ALSO carries the critic, so
    restore_from() reverts BOTH policy and critic (3c.3b revert / 3c.4 resume)."""
    ac = ActorCritic.from_bc_checkpoint(_write_ckpt(tmp_path))
    x = _x()
    with torch.no_grad():
        a_logits = ac.policy(x)[0]["our_a"].clone()
        c_value = ac.critic(x).clone()
    p = tmp_path / "gen0.pt"
    ac.save(p, generation=0)

    # the saved checkpoint loads as a plain BC policy (the gauntlet path) — no raise
    pol, heads = model_io.load_bc_policy(p)
    assert heads == ("our_a", "our_b") and model_io.value_trained(pol)

    # perturb the live actor-critic, then restore -> both nets recover exactly
    with torch.no_grad():
        for prm in ac.actor_parameters():
            prm.add_(1.0)
        for prm in ac.critic_parameters():
            prm.add_(1.0)
    ac.restore_from(p)
    with torch.no_grad():
        assert torch.allclose(ac.policy(x)[0]["our_a"], a_logits, atol=1e-6)
        assert torch.allclose(ac.critic(x), c_value, atol=1e-6)


def test_save_verify_raises_on_broken_checkpoint(tmp_path, monkeypatch):
    """save(verify=True) must FAIL LOUD when the written checkpoint won't reload (so a
    broken save can't slip through and be silently evaluated as a no-model fallback)."""
    ac = ActorCritic.from_bc_checkpoint(_write_ckpt(tmp_path))
    # corrupt the config so the reload mismatches (wrong hidden dims)
    monkeypatch.setattr(ac, "state_checkpoint",
                        lambda generation=None: {"model_state": ac.policy.state_dict(),
                                                 "config": {"state_dim": STATE_DIM,
                                                            "action_dim": ACTION_DIM,
                                                            "hidden_dims": (999,),
                                                            "state_layout_version":
                                                                get_state_layout_version()}})
    with pytest.raises(Exception):
        ac.save(tmp_path / "broken.pt")


def test_state_dict_round_trips_both_nets(tmp_path):
    """Resumability foundation (3c.4): one state_dict carries policy.* + critic.*."""
    ac = ActorCritic.from_bc_checkpoint(_write_ckpt(tmp_path))
    # clone the snapshot: state_dict() returns tensors that SHARE storage with the
    # live params, so an in-place perturbation below would otherwise corrupt it too.
    sd = {k: v.detach().clone() for k, v in ac.state_dict().items()}
    assert any(k.startswith("policy.") for k in sd)
    assert any(k.startswith("critic.") for k in sd)
    # perturb the critic, then restore from the snapshot -> exact recovery
    with torch.no_grad():
        for p in ac.critic_parameters():
            p.add_(1.0)
    ac.load_state_dict(sd)
    x = _x()
    with torch.no_grad():
        restored = ac.critic(x)
    ac2 = ActorCritic.from_bc_checkpoint(_write_ckpt(tmp_path))
    with torch.no_grad():
        fresh = ac2.critic(x)
    assert torch.allclose(restored, fresh, atol=1e-6)
