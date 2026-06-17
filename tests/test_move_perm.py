"""Task #7 (#22): move-slot permutation augmentation.

7a — the permutation helpers in state_encoder (state move blocks + action label
+ action mask stay consistent under a move-slot permutation).
"""

from __future__ import annotations

import numpy as np

import v_dance.encoders.state_encoder as se


def test_permute_action_index_moves_and_passthrough():
    perm = [2, 0, 1, 3]                       # new pos i holds old move perm[i]
    # old move 0 (idx 0*3+1=1) → new position perm.index(0)=1 → 1*3+1=4
    assert se.permute_action_index(1, perm) == 4
    # old move 2 (idx 2*3+0=6) → new position perm.index(2)=0 → 0
    assert se.permute_action_index(6, perm) == 0
    # switches / None / out-of-range pass through
    assert se.permute_action_index(12, perm) == 12
    assert se.permute_action_index(15, perm) == 15
    assert se.permute_action_index(None, perm) is None


def test_permute_action_mask_row_consistency():
    perm = [2, 0, 1, 3]
    row = list(range(16))                     # distinct values to track
    out = se.permute_action_mask_row(row, perm)
    for j in range(se.NUM_MOVES):
        for b in range(3):
            assert out[j * 3 + b] == row[perm[j] * 3 + b]
    assert out[12:] == row[12:]               # switch entries unchanged


def test_permute_move_slots_reorders_blocks():
    base = se.own_active_move_base(0)
    vec = np.zeros(se.get_state_dim(), np.float32)
    for m in range(se.NUM_MOVES):
        vec[base + m * se.MOVE_FEATURES] = m + 1     # mark each move block
    perm = [2, 0, 1, 3]
    se.permute_move_slots(vec, 0, perm)
    for i in range(se.NUM_MOVES):
        assert vec[base + i * se.MOVE_FEATURES] == perm[i] + 1   # new pos i = old perm[i]


def test_permute_move_slots_only_touches_that_slot():
    vec = np.random.RandomState(3).randn(se.get_state_dim()).astype(np.float32)
    orig = vec.copy()
    se.permute_move_slots(vec, 0, [1, 0, 2, 3])
    b0 = se.own_active_move_base(0)
    b1 = se.own_active_move_base(1)
    # our_b's move block and everything outside our_a's move block are untouched.
    assert np.allclose(vec[b1:b1 + se.NUM_MOVES * se.MOVE_FEATURES],
                       orig[b1:b1 + se.NUM_MOVES * se.MOVE_FEATURES])
    assert np.allclose(vec[:b0], orig[:b0])


def test_permute_then_inverse_is_identity():
    vec = np.random.RandomState(0).randn(se.get_state_dim()).astype(np.float32)
    orig = vec.copy()
    perm = [2, 0, 3, 1]
    inv = [perm.index(i) for i in range(se.NUM_MOVES)]
    se.permute_move_slots(vec, 1, perm)
    se.permute_move_slots(vec, 1, inv)
    assert np.allclose(vec, orig)


def test_label_tracks_features_under_permutation():
    """The (move features ↔ action label) coupling is invariant: after permuting,
    the target points at the SAME move's features and stays mask-legal."""
    base = se.own_active_move_base(0)
    vec = np.zeros(se.get_state_dim(), np.float32)
    for m in range(se.NUM_MOVES):
        vec[base + m * se.MOVE_FEATURES] = m + 1
    target = 1 * 3 + 0                         # chose old move 1, bucket 0
    mask = [0] * 16
    for m in range(se.NUM_MOVES):
        mask[m * 3] = 1                         # bucket 0 legal for all moves
    perm = [3, 1, 0, 2]
    se.permute_move_slots(vec, 0, perm)
    new_t = se.permute_action_index(target, perm)
    new_mask = se.permute_action_mask_row(mask, perm)
    new_pos = new_t // 3
    assert vec[base + new_pos * se.MOVE_FEATURES] == 1 + 1     # old move 1's marker
    assert new_mask[new_t] == 1                                # still legal
