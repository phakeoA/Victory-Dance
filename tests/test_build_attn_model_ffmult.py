"""Regression: build_attn_model must FORWARD ff_mult.

The trainer stamps ``args.ff_mult`` into the checkpoint config and model_io rebuilds
with it; if build_attn_model drops ff_mult it builds ``ff_mult=2`` while the config
claims another value, so any ``--ff-mult != 2`` checkpoint fails to reload with a
silent SHAPE MISMATCH. These tests lock the forward + the end-to-end round-trip.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")
import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parents[1]
from v_dance.models.bc_model_attn import build_attn_model  # noqa: E402
from v_dance.encoders.state_encoder import (  # noqa: E402
    get_state_dim, get_action_dim, get_gimmick_dim, get_state_layout_version)
import v_dance.play.model_io as model_io  # noqa: E402


def test_build_attn_model_forwards_ff_mult():
    m = build_attn_model(d_model=32, n_heads=4, n_layers=1, ff_mult=4, heads=("our_a", "our_b"))
    assert m.ff_mult == 4
    # the transformer's feed-forward dim actually reflects ff_mult (= ff_mult * d_model)
    ff_dims = [mod.linear1.out_features for mod in m.modules()
               if isinstance(mod, nn.TransformerEncoderLayer)]
    assert ff_dims and all(d == 4 * 32 for d in ff_dims)


def test_ff_mult_checkpoint_round_trips_through_model_io(tmp_path):
    """A ff_mult!=2 checkpoint built the trainer's way must RELOAD (the pre-fix bug
    raised a strict load_state_dict size mismatch here)."""
    m = build_attn_model(d_model=32, n_heads=4, n_layers=1, ff_mult=4,
                         heads=("our_a", "our_b"))
    cfg = {
        "model_type": "attn", "state_dim": get_state_dim(), "action_dim": get_action_dim(),
        "gimmick_dim": get_gimmick_dim(), "state_layout_version": get_state_layout_version(),
        "d_model": 32, "n_heads": 4, "n_layers": 1, "ff_mult": 4, "dropout": 0.0,
        "value_readout": "mean", "heads": ["our_a", "our_b"],
        "gimmick_heads": ["our_a", "our_b"], "value_trained": True, "gimmick_trained": True,
    }
    p = tmp_path / "ff4.pt"
    torch.save({"model_state": m.state_dict(), "config": cfg}, p)
    policy, heads = model_io.load_bc_policy(p)   # pre-fix: RuntimeError (size mismatch)
    assert policy.ff_mult == 4
    assert tuple(heads) == ("our_a", "our_b")
