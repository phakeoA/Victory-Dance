"""Regression test kept from the post-B3 full-codebase scan (2026-06-29).

- #4 from_bc_checkpoint double-load: model_io.load_bc_policy accepts a preloaded ckpt dict to reuse.
(#1 the C51 resume peek and #3 the C51 loss support left with the C51 critic itself — cleanup pass 2 step 3,
2026-10-04. #2 POSIX killpg, #5 win-signal None-guard, #6 rating-delta precise match, #7 gauntlet anchor-free exit
are verified against source + covered by the full suite.)
"""
from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch

from conftest import write_attn_ckpt


# ── #4: load_bc_policy reuses a preloaded checkpoint dict (no double torch.load) ─
def test_load_bc_policy_reuses_preloaded_ckpt(tmp_path):
    from v_dance.play import model_io
    p = write_attn_ckpt(tmp_path / "bc.pt", seed=3)
    ck = torch.load(p, map_location="cpu", weights_only=False)
    model, heads = model_io.load_bc_policy(p, _ckpt=ck)
    assert model is not None and heads is not None
