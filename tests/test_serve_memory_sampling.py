"""Serve-side Phase-3 plumbing tests:

  * model_io._policy_forward — single-frame vs frame-stack dispatch (memory
    models route through forward_with_memory; stateless models degrade to the
    last frame, so callers may stack unconditionally).
  * live_vgc_base._memory_stack — per-battle frame ring buffer (append, same-
    turn replace, max_mem_len trim, stateless no-op).
  * game_runner._seat_sampling — the serve-sampling A/B seat temperatures.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from v_dance.encoders.state_encoder import get_state_dim
from v_dance.models.bc_model_attn import AttnBCPolicy
from v_dance.play import model_io as M


def _models():
    torch.manual_seed(5)
    stateless = AttnBCPolicy(d_model=32, n_heads=4, n_layers=1, dropout=0.0)
    memory = AttnBCPolicy(d_model=32, n_heads=4, n_layers=1, dropout=0.0,
                          memory_dim=16, mem_heads=2)
    stateless.eval(), memory.eval()
    return stateless, memory


def test_policy_forward_dispatch():
    stateless, memory = _models()
    x = np.random.RandomState(0).rand(3, get_state_dim()).astype(np.float32)

    # 1-D on either model == plain forward (reference under no_grad too — the
    # transformer's fused inference kernel differs numerically from the grad path)
    a, _, v = M._policy_forward(stateless, x[-1])
    with torch.no_grad():
        a_ref, _, v_ref = stateless(torch.as_tensor(x[-1]))
    assert torch.equal(a["our_a"], a_ref["our_a"]) and torch.equal(v, v_ref)

    # 2-D on a STATELESS model → last frame only
    a2, _, v2 = M._policy_forward(stateless, x)
    assert torch.equal(a2["our_a"], a_ref["our_a"]) and torch.equal(v2, v_ref)

    # 2-D on a MEMORY model → forward_with_memory (not the single-frame path)
    a3, _, v3 = M._policy_forward(memory, x)
    with torch.no_grad():
        a3_ref, _, v3_ref = memory.forward_with_memory(torch.as_tensor(x))
    assert torch.equal(a3["our_a"], a3_ref["our_a"]) and torch.equal(v3, v3_ref)
    with torch.no_grad():
        a1_mem, _, _ = memory(torch.as_tensor(x[-1]))
    assert not torch.equal(a3["our_a"], a1_mem["our_a"]), \
        "memory model ignored its history"


def test_head_and_value_helpers_accept_stack():
    _, memory = _models()
    x = np.random.RandomState(1).rand(4, get_state_dim()).astype(np.float32)
    l0, l1 = M.head_logits(memory, ("our_a", "our_b"), x, "cpu")
    assert l0.shape == l1.shape and l0.ndim == 1
    memory._value_trained = True
    wp = M.value_logit(memory, x, "cpu")
    assert wp is not None and 0.0 <= wp <= 1.0


# ── live player frame buffer ────────────────────────────────────────────────

class _FakeBattle:
    def __init__(self, tag="battle-x-1", turn=1):
        self.battle_tag = tag
        self.turn = turn


def _bare_player(model):
    from v_dance.play.player import VGCPlayer     # concrete subclass (base is abstract)
    p = VGCPlayer.__new__(VGCPlayer)              # __new__ test pattern used repo-wide
    p._model = model
    p._mem_frames = {}
    return p


def test_memory_stack_stateless_noop():
    stateless, _ = _models()
    p = _bare_player(stateless)
    vec = np.ones(get_state_dim(), dtype=np.float32)
    out = p._memory_stack(_FakeBattle(), vec, "turn")
    assert out is vec and p._mem_frames == {}


def test_memory_stack_grows_replaces_and_trims():
    _, memory = _models()
    memory.max_mem_len = 3
    p = _bare_player(memory)
    b = _FakeBattle()
    v = lambda k: np.full(get_state_dim(), k, dtype=np.float32)

    out = p._memory_stack(b, v(1), "turn")
    assert out.shape == (1, get_state_dim())
    # same-decision re-call (a rejected order re-encodes the IDENTICAL state)
    # replaces the frame, never stacks it
    out = p._memory_stack(b, v(1), "turn")
    assert out.shape == (1, get_state_dim()) and out[0, 0] == 1
    # same (turn, type) with a CHANGED state is a DISTINCT decision — e.g. a
    # SECOND forced replacement after the first resolved. Sub-indexed append:
    # it must never overwrite the first decision's frame (scan 2026-07-02).
    out = p._memory_stack(b, v(2), "turn")
    assert out.shape == (2, get_state_dim())
    assert out[0, 0] == 1 and out[1, 0] == 2
    # replacement on the same turn is a DISTINCT decision type → appended
    b2 = _FakeBattle(turn=1)
    out = p._memory_stack(b2, v(3), "replacement")
    assert out.shape == (3, get_state_dim())
    # max_mem_len=3 → oldest frame trimmed
    b.turn = 2
    out = p._memory_stack(b, v(12), "turn")
    assert out.shape == (3, get_state_dim())
    assert out[0, 0] == 2 and out[-1, 0] == 12


def test_memory_stack_isolated_per_battle():
    _, memory = _models()
    p = _bare_player(memory)
    va = np.zeros(get_state_dim(), dtype=np.float32)
    p._memory_stack(_FakeBattle("battle-a"), va, "turn")
    out_b = p._memory_stack(_FakeBattle("battle-b"), va, "turn")
    assert out_b.shape[0] == 1
    assert set(p._mem_frames) == {"battle-a", "battle-b"}


# ── serve-sampling A/B seat config ──────────────────────────────────────────

def test_seat_sampling_modes():
    from v_dance.selfplay.game_runner import _seat_sampling
    # off → the collection tau on both seats (byte-identical self-play default)
    assert _seat_sampling("off", 1.0, 0.45, 0.9) == (1.0, 1.0, 1.0, 1.0)
    # ab → p1 samples at serve_tau/top-p, p2 pure argmax
    assert _seat_sampling("ab", 1.0, 0.45, 0.9) == (0.45, 0.0, 0.9, 1.0)
    # both → smoke, both seats at serve settings
    assert _seat_sampling("both", 1.0, 0.5, 0.95) == (0.5, 0.5, 0.95, 0.95)
