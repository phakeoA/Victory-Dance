"""Phase 1a — match-memory core on AttnBCPolicy (frame-stacked causal
time-axis transformer; docs/memory_core_phase1a_design.md).

Exit criteria covered here (offline, no retrain):
  1. memory_dim=0 (default) is UNCHANGED: state_dict key-set == the real anchor
     checkpoint's, param count identical, anchor loads strict.
  2. Warm-start surgery: a widened memory model fed the stateless checkpoint
     reproduces its single-turn logits BIT-EXACTLY (zeros in the new columns).
  3. forward_with_memory: shape contract, causality (earlier frames influence
     the last-frame output), max_mem_len window, padding mask.
  4. Gradients flow into the memory modules from a last-frame loss.
  5. model_io round-trip for a memory checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from v_dance.encoders.state_encoder import (
    get_action_dim,
    get_gimmick_dim,
    get_state_dim,
    get_state_layout_version,
)
from v_dance.models.bc_model_attn import (
    AttnBCPolicy,
    init_memory_model_from_stateless,
)

_ANCHOR = (Path(__file__).resolve().parents[1] / "ai_train_scripts" / "BC_model"
           / "checkpoints_attn_pre_gen141" / "battle_base.pt")

_TINY = dict(d_model=32, n_heads=4, n_layers=1, dropout=0.0,
             heads=("our_a", "our_b", "opp_a", "opp_b"),
             gimmick_heads=("our_a", "our_b"))


def _tiny(memory_dim=0, **over):
    kw = dict(_TINY)
    kw.update(over)
    return AttnBCPolicy(memory_dim=memory_dim, mem_heads=2, **kw)


def _rand_x(*shape):
    g = torch.Generator().manual_seed(7)
    return torch.rand(*shape, get_state_dim(), generator=g)


# ── 1. stateless default is unchanged ──────────────────────────────────────

def test_stateless_keyset_matches_anchor_checkpoint():
    if not _ANCHOR.exists():
        pytest.skip("anchor checkpoint not present")
    ck = torch.load(_ANCHOR, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    if cfg.get("state_dim") != get_state_dim():
        pytest.skip("anchor predates the current layout")
    model = AttnBCPolicy(
        state_dim=cfg["state_dim"], action_dim=cfg["action_dim"],
        gimmick_dim=cfg.get("gimmick_dim", get_gimmick_dim()),
        d_model=cfg["d_model"], n_heads=cfg["n_heads"], n_layers=cfg["n_layers"],
        ff_mult=cfg.get("ff_mult", 2), dropout=cfg.get("dropout", 0.0),
        heads=tuple(cfg["heads"]), gimmick_heads=cfg.get("gimmick_heads"),
        value_readout=cfg.get("value_readout", "mean"),
    )
    assert set(model.state_dict()) == set(ck["model_state"])
    model.load_state_dict(ck["model_state"], strict=True)   # raises on any drift


def test_stateless_has_no_memory_modules():
    m = _tiny(memory_dim=0)
    assert not hasattr(m, "mem_proj") and not hasattr(m, "mem_attn")
    assert m.memory_dim == 0
    with pytest.raises(RuntimeError, match="memory_dim"):
        m.forward_with_memory(_rand_x(2, 3))


# ── 2. warm-start surgery byte-identity ────────────────────────────────────

def test_surgery_reproduces_stateless_logits_exactly():
    torch.manual_seed(11)
    stateless = _tiny(memory_dim=0)
    mem_model = _tiny(memory_dim=16)
    init_memory_model_from_stateless(mem_model, stateless.state_dict())
    stateless.eval(), mem_model.eval()
    x = _rand_x(5)
    with torch.no_grad():
        a0, g0, v0 = stateless(x)
        a1, g1, v1 = mem_model(x)
    for k in a0:
        assert torch.equal(a0[k], a1[k]), f"action head {k} diverged"
    for k in g0:
        assert torch.equal(g0[k], g1[k]), f"gimmick head {k} diverged"
    # value readout allclose (not equal): reduction order differs across BLAS backends (CI Linux)
    assert torch.allclose(v0, v1, rtol=0, atol=1e-6), "value head diverged"


def test_surgery_rejects_stateless_target():
    with pytest.raises(ValueError, match="memory_dim=0"):
        init_memory_model_from_stateless(_tiny(0), _tiny(0).state_dict())


# ── 3. forward_with_memory contract ────────────────────────────────────────

def test_sequence_forward_shapes_and_finiteness():
    m = _tiny(memory_dim=16)
    m.eval()
    B, T = 3, 6
    with torch.no_grad():
        actions, gimmicks, value = m.forward_with_memory(_rand_x(B, T))
    assert set(actions) == set(_TINY["heads"])
    for v in actions.values():
        assert v.shape == (B, get_action_dim()) and torch.isfinite(v).all()
    for v in gimmicks.values():
        assert v.shape == (B, get_gimmick_dim()) and torch.isfinite(v).all()
    assert value.shape == (B,) and torch.isfinite(value).all()
    # unbatched (T, state_dim) -> unbatched outputs (forward parity)
    with torch.no_grad():
        a1, _, v1 = m.forward_with_memory(_rand_x(T))
    assert a1["our_a"].shape == (get_action_dim(),) and v1.shape == ()


def test_memory_is_causal_and_live():
    """Perturbing an EARLIER frame must change the last-frame logits (the
    memory actually reads history), while the last frame's own tokens pin the
    non-memory inputs."""
    m = _tiny(memory_dim=16)
    m.eval()
    x = _rand_x(1, 5)
    x2 = x.clone()
    x2[0, 1] = torch.rand_like(x2[0, 1])          # change turn 2 of 5
    with torch.no_grad():
        a, _, _ = m.forward_with_memory(x)
        b, _, _ = m.forward_with_memory(x2)
    assert not torch.equal(a["our_a"], b["our_a"]), \
        "earlier-frame perturbation did not reach the last-frame output"


def test_window_keeps_most_recent_frames():
    m = _tiny(memory_dim=16, max_mem_len=4)
    m.eval()
    x = _rand_x(2, 9)
    with torch.no_grad():
        a_full, _, _ = m.forward_with_memory(x)
        a_win, _, _ = m.forward_with_memory(x[:, -4:])
    for k in a_full:
        assert torch.allclose(a_full[k], a_win[k], atol=1e-6), \
            "truncation must keep exactly the trailing max_mem_len frames"


def test_frame_padding_equals_unpadded():
    """THE train/serve consistency guarantee (audit 2026-07-02): positions are
    RELATIVE TO THE PRESENT, so a left-padded fixed-T window (the training
    regime) and the same history unpadded (the serve regime) produce the SAME
    outputs — real frames carry identical age embeddings either way and the
    padding is masked out."""
    m = _tiny(memory_dim=16)
    m.eval()
    real = _rand_x(1, 3)
    pad = torch.zeros(1, 2, get_state_dim())
    padded = torch.cat([pad, real], dim=1)                    # left-padded to T=5
    mask = torch.tensor([[True, True, False, False, False]])
    with torch.no_grad():
        a_ref, _, v_ref = m.forward_with_memory(real)
        a_pad, _, v_pad = m.forward_with_memory(padded, frame_padding_mask=mask)
    for k in a_ref:
        assert torch.allclose(a_pad[k], a_ref[k], atol=1e-5), \
            f"head {k}: padded window diverged from the unpadded history"
    assert torch.allclose(v_pad, v_ref, atol=1e-5)


# ── 4. gradients reach the memory modules ──────────────────────────────────

def test_memory_modules_receive_gradients():
    m = _tiny(memory_dim=16)
    actions, _, value = m.forward_with_memory(_rand_x(2, 4))
    loss = actions["our_a"].sum() + value.sum()
    loss.backward()
    assert m.mem_proj.weight.grad is not None and m.mem_proj.weight.grad.abs().sum() > 0
    got_attn_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                        for p in m.mem_attn.parameters())
    assert got_attn_grad, "no gradient reached the causal memory transformer"
    assert m.mem_pos_emb.grad is not None


# ── 5. model_io round-trip for a memory checkpoint ─────────────────────────

def test_model_io_roundtrip_memory_checkpoint(tmp_path):
    from v_dance.play.model_io import load_bc_policy
    m = _tiny(memory_dim=16)
    m.eval()
    cfg = {"model_type": "attn", "state_dim": get_state_dim(),
           "action_dim": get_action_dim(), "gimmick_dim": get_gimmick_dim(),
           "state_layout_version": get_state_layout_version(),
           "d_model": 32, "n_heads": 4, "n_layers": 1, "ff_mult": 2,
           "dropout": 0.0, "value_readout": "mean",
           "heads": list(_TINY["heads"]), "gimmick_heads": list(_TINY["gimmick_heads"]),
           "value_trained": True, "gimmick_trained": True,
           "memory_dim": 16, "mem_layers": 2, "mem_heads": 2, "max_mem_len": 64}
    p = tmp_path / "mem_ckpt.pt"
    torch.save({"model_state": m.state_dict(), "config": cfg}, p)
    loaded, _head_names = load_bc_policy(p, device="cpu")
    assert loaded.memory_dim == 16
    x = _rand_x(3)
    with torch.no_grad():
        a0, _, v0 = m(x)
        a1, _, v1 = loaded(x)
    for k in a0:
        assert torch.equal(a0[k], a1[k])
    assert torch.equal(v0, v1)
    xs = _rand_x(2, 5)
    with torch.no_grad():
        s0, _, sv0 = m.forward_with_memory(xs)
        s1, _, sv1 = loaded.forward_with_memory(xs)
    for k in s0:
        assert torch.equal(s0[k], s1[k])
    assert torch.equal(sv0, sv1)
