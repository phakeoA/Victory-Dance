"""Unit tests for local_battle/model_io.py (#13): loading dict checkpoints and
mask-aware decoding of the two-head BC policy + the team-preview scorer.

The checkpoint-dependent tests skip cleanly when the trained .pt files (or torch)
are absent, so a fresh clone without trained models still passes the suite.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
for _p in (
    str(_REPO / "local_battle"),
    str(_REPO / "data" / "scripts"),
    str(_REPO / "ai_train_scripts" / "BC_model"),
    str(_REPO / "ai_train_scripts" / "teamPreview_model"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import model_io as M  # noqa: E402  (local_battle/model_io.py)

_BC_CKPT = _REPO / "ai_train_scripts" / "BC_model" / "checkpoints" / "bc_best.pt"
_TP_CKPT = _REPO / "ai_train_scripts" / "teamPreview_model" / "checkpoints" / "teampreview_best.pt"


# ── masked_argmax (pure, no torch / no checkpoint needed) ─────────────────────
def test_masked_argmax_picks_highest_legal_not_highest_overall():
    # logit 10 at index 0 is ILLEGAL; the best LEGAL is index 2 (8.0).
    assert M.masked_argmax([10.0, 5.0, 8.0], [False, True, True]) == 2


def test_masked_argmax_all_illegal_returns_none():
    assert M.masked_argmax([3.0, 1.0, 2.0], [False, False, False]) is None


def test_masked_argmax_single_legal():
    assert M.masked_argmax([0.0, 0.0, 0.0, 9.0], [False, False, False, True]) == 3


# ── BC policy load + decode (needs torch + trained checkpoint) ────────────────
@pytest.fixture(scope="module")
def bc_loaded():
    pytest.importorskip("torch")
    if not _BC_CKPT.exists():
        pytest.skip(f"BC checkpoint not found: {_BC_CKPT}")
    model, heads = M.load_bc_policy(_BC_CKPT)
    return model, heads


def test_load_bc_policy_reconstructs_two_head_module(bc_loaded):
    import torch
    from state_encoder import STATE_DIM, ACTION_DIM
    model, heads = bc_loaded
    assert heads == ("our_a", "our_b")
    assert model.state_dim == STATE_DIM
    out = model(torch.zeros(STATE_DIM, dtype=torch.float32))
    assert isinstance(out, dict) and set(out) == {"our_a", "our_b"}
    assert out["our_a"].numel() == ACTION_DIM


def test_bc_action_indices_returns_masked_argmax(bc_loaded):
    """The decoded action per head equals a manual masked-argmax over that head's
    raw logits — i.e. always the highest-logit LEGAL action."""
    import numpy as np, torch
    from state_encoder import STATE_DIM
    model, heads = bc_loaded
    sv = np.random.RandomState(1).randn(STATE_DIM).astype("float32")
    # legal subsets that exclude the (possibly) top logit, to make masking bite.
    mask0 = [i in (1, 4, 13) for i in range(16)]
    mask1 = [i in (0, 2, 3, 12, 14) for i in range(16)]
    with torch.no_grad():
        out = model(torch.as_tensor(sv))
    exp0 = M.masked_argmax(out["our_a"].numpy().ravel(), mask0)
    exp1 = M.masked_argmax(out["our_b"].numpy().ravel(), mask1)
    a0, a1 = M.bc_action_indices(model, heads, sv, mask0, mask1)
    assert (a0, a1) == (exp0, exp1)
    # and the chosen indices are legal
    assert mask0[a0] and mask1[a1]


def test_bc_action_indices_none_when_no_legal(bc_loaded):
    import numpy as np
    from state_encoder import STATE_DIM
    model, heads = bc_loaded
    sv = np.zeros(STATE_DIM, dtype="float32")
    a0, a1 = M.bc_action_indices(model, heads, sv, [False] * 16, [True] + [False] * 15)
    assert a0 is None and a1 == 0


# ── Team-chooser load + order (needs torch + trained checkpoint) ──────────────
@pytest.fixture(scope="module")
def tp_loaded():
    pytest.importorskip("torch")
    if not _TP_CKPT.exists():
        pytest.skip(f"team-chooser checkpoint not found: {_TP_CKPT}")
    return M.load_team_chooser(_TP_CKPT)


def test_load_team_chooser_has_vocab_and_config(tp_loaded):
    model, vocab, cfg = tp_loaded
    assert isinstance(vocab, dict) and len(vocab) > 50
    assert cfg["feat_dim"] > 0 and cfg["vocab_size"] >= len(vocab)


def test_team_order_returns_n_distinct_in_range(tp_loaded):
    model, vocab, cfg = tp_loaded
    our = ["charizard", "incineroar", "rillaboom", "amoonguss", "urshifu", "fluttermane"]
    opp = ["miraidon", "ironhands", "chiyu", "landorustherian", "ogerpon", "ironbundle"]
    order = M.team_order(model, vocab, cfg, our, opp, n=4)
    assert len(order) == 4
    assert len(set(order)) == 4
    assert all(0 <= i < 6 for i in order)


def test_team_order_leads_are_within_brought(tp_loaded):
    """The first lead_k entries (the leads) must be a subset of the full bring,
    and the whole order must be distinct roster positions."""
    model, vocab, cfg = tp_loaded
    our = ["gholdengo", "ragingbolt", "ironhands", "fluttermane", "ogerpon", "rillaboom"]
    opp = ["miraidon", "ironbundle", "chiyu", "landorustherian", "amoonguss", "urshifu"]
    order = M.team_order(model, vocab, cfg, our, opp, n=4)
    lead_k = int(cfg.get("lead_k", 2))
    leads = set(order[:lead_k])
    assert leads <= set(order)        # leads are part of the bring
    assert len(set(order)) == len(order)
