"""Smoke tests for v_dance.eval.bc_val_report — the fine-grained checkpoint
comparison ruler (per-turn-bucket / per-head / decision-type / value Brier)."""

from __future__ import annotations

import numpy as np
import torch

from v_dance.encoders.state_encoder import ACTIONS_PER_SLOT, get_state_dim
from v_dance.eval.bc_val_report import _bucket_of, evaluate_checkpoint, print_report
from v_dance.training.bc_dataset import HEADS

from conftest import write_attn_ckpt


def _val_examples(n=12):
    rng = np.random.RandomState(3)
    out = []
    for i in range(n):
        mask = np.zeros(ACTIONS_PER_SLOT, dtype=np.float32)
        mask[:4] = 1.0
        out.append({
            "x": rng.rand(get_state_dim()).astype(np.float32),
            "targets": {HEADS[i % 2]: int(rng.randint(0, 4))},
            "masks": {HEADS[i % 2]: mask},
            "gimmick_targets": {}, "gimmick_masks": {},
            "replay_id": f"r{i // 4}", "perspective": "p1" if i % 2 == 0 else "p2",
            "rating": None, "rating_delta": 0.0, "won": bool(i % 2),
            "turn": (i % 6) + 1, "decision_type": "turn",
        })
    return out


def test_bucket_boundaries():
    assert _bucket_of(1) == "t1"
    assert _bucket_of(3) == "t2-3"
    assert _bucket_of(6) == "t4-6"
    assert _bucket_of(10) == "t7-10"
    assert _bucket_of(42) == "t11+"
    assert _bucket_of(None) == "t1"


def test_evaluate_and_print_stateless(tmp_path, capsys):
    ckpt = write_attn_ckpt(tmp_path / "tiny.pt",
                           heads=("our_a", "our_b"), gimmick_heads=("our_a", "our_b"))
    val_ex = _val_examples()
    r = evaluate_checkpoint(str(ckpt), val_ex, device="cpu", batch_size=4)
    assert r["pooled"].n == len(val_ex)          # one valid head per example
    assert r["value"]["n"] == len(val_ex)
    assert 0.0 <= r["pooled"].rate() <= 1.0
    # buckets partition the decisions
    assert sum(a.n for a in r["by_bucket"].values()) == r["pooled"].n
    print_report([r, r])                          # self-comparison: deltas all zero
    out = capsys.readouterr().out
    assert "per turn bucket" in out and "(+0.0000)" in out
