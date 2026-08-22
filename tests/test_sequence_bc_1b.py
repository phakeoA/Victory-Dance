"""Phase 1b — sequence BC dataset (--sequence-len) over the match-memory core.

Covers:
  * sequence_len=1 items are byte-identical to the stateless dataset (no x_seq).
  * x_seq assembly: left zero-padding + mask, correct history slice, trajectory
    isolation by (replay_id, perspective), last frame == the item's own x.
  * augmentation: only the LAST frame is augmented; history stays raw.
  * end-to-end: a DataLoader batch through AttnBCPolicy.forward_with_memory
    trains (loss backward) — the exact train_bc sequence path.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from v_dance.encoders.state_encoder import ACTIONS_PER_SLOT, get_state_dim
from v_dance.training.bc_dataset import BCDataset, HEADS


def _ex(rid, persp, i):
    """Minimal usable example; x is a recognisable constant row."""
    x = np.full(get_state_dim(), 0.001 * (i + 1), dtype=np.float32)
    mask = np.zeros(ACTIONS_PER_SLOT, dtype=np.float32)
    mask[:3] = 1.0
    return {
        "x": x, "targets": {HEADS[0]: 1}, "masks": {HEADS[0]: mask},
        "gimmick_targets": {}, "gimmick_masks": {},
        "replay_id": rid, "perspective": persp,
        "rating": None, "rating_delta": 0.0, "won": True,
        "turn": i, "decision_type": "turn",
    }


def _mixed_examples():
    """Two interleaved trajectories of one replay (p1: 3 frames, p2: 2), the
    interleaving mirroring real file order (p1, p2, p1, p2, p1)."""
    return [
        _ex("r1", "p1", 0), _ex("r1", "p2", 1),
        _ex("r1", "p1", 2), _ex("r1", "p2", 3),
        _ex("r1", "p1", 4),
    ]


def test_sequence_len_1_has_no_seq_fields():
    ds = BCDataset(_mixed_examples(), sequence_len=1)
    item = ds[0]
    assert "x_seq" not in item and "frame_padding_mask" not in item


def test_sequence_assembly_and_trajectory_isolation():
    ds = BCDataset(_mixed_examples(), sequence_len=3)
    # last p1 item (index 4) — history must be the two EARLIER p1 frames only
    item = ds[4]
    assert item["x_seq"].shape == (3, get_state_dim())
    assert not item["frame_padding_mask"].any()
    assert torch.equal(item["x_seq"][0], ds.X_t[0])   # p1 frame 0
    assert torch.equal(item["x_seq"][1], ds.X_t[2])   # p1 frame 1 (index 2!)
    assert torch.equal(item["x_seq"][2], ds.X_t[4])   # itself
    # first p2 item (index 1) — no history: two left-padding frames
    item = ds[1]
    assert item["frame_padding_mask"].tolist() == [True, True, False]
    assert torch.equal(item["x_seq"][2], ds.X_t[1])
    assert item["x_seq"][0].abs().sum() == 0


def test_sequence_last_frame_matches_targets_under_augmentation():
    ds = BCDataset(_mixed_examples(), sequence_len=3, augment_move_order=True)
    item = ds[4]
    # augmented fetch: x_seq's LAST frame is the augmented x (same object
    # semantics), history frames equal the RAW stored frames
    assert torch.equal(item["x_seq"][2], item["x"])
    assert torch.equal(item["x_seq"][0], ds.X_t[0])
    assert torch.equal(item["x_seq"][1], ds.X_t[2])


def test_sequence_batch_trains_through_memory_model():
    from v_dance.models.bc_model_attn import AttnBCPolicy
    ds = BCDataset(_mixed_examples() * 4, sequence_len=3)
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    model = AttnBCPolicy(d_model=32, n_heads=4, n_layers=1, dropout=0.0,
                         memory_dim=16, mem_heads=2)
    batch = next(iter(loader))
    actions, gimmicks, value = model.forward_with_memory(
        batch["x_seq"], frame_padding_mask=batch["frame_padding_mask"])
    assert actions["our_a"].shape == (4, ACTIONS_PER_SLOT)
    loss = actions["our_a"].sum() + value.sum()
    loss.backward()
    assert model.mem_proj.weight.grad is not None


def test_sequence_len_validation():
    with pytest.raises(ValueError, match="sequence_len"):
        BCDataset(_mixed_examples(), sequence_len=0)
