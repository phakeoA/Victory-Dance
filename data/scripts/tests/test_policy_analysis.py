"""Unit tests for policy_analysis (ally mis-target classifier + move-slot
permutation helpers used by the order-invariance probe)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from policy_analysis import (  # noqa: E402
    MOVE_BLOCK_START, move_block_start, is_ally_mistarget,
    permute_move_slots, permute_mask_row, unpermute_action, flatten_move_known,
)
from state_encoder import (  # noqa: E402
    VodStateEncoder, POKEMON_FEATURES, NUM_MOVES, MOVE_FEATURES,
)


def test_move_block_offset_matches_encoder():
    """The move-block offset must point at where the encoder actually writes move
    features.  Encode a mon whose first move is Close Combat (base_power 120) and
    assert the first move feature there is 120/150."""
    assert MOVE_BLOCK_START == 70  # frozen layout
    enc = VodStateEncoder()
    snap = {"our_active": {"our_a": {"species": "Sneasler",
                                     "revealed_moves": ["Close Combat"]}}}
    vec = enc.encode_snapshot(snap, turn=1)
    # first move feature = base_power / 150
    assert vec[move_block_start(0)] == pytest.approx(120 / 150.0, abs=1e-6)


def test_permute_move_slots_identity_and_reversal():
    vec = np.zeros(POKEMON_FEATURES * 2, dtype=np.float32)
    base = move_block_start(0)
    for m in range(NUM_MOVES):                 # tag each move block with m+1
        vec[base + m * MOVE_FEATURES: base + (m + 1) * MOVE_FEATURES] = m + 1

    # identity perm → unchanged
    assert np.array_equal(permute_move_slots(vec, 0, (0, 1, 2, 3)), vec)

    # reversal: NEW slot 0 holds OLD slot 3, etc.
    rev = permute_move_slots(vec, 0, (3, 2, 1, 0))
    for m in range(NUM_MOVES):
        block = rev[base + m * MOVE_FEATURES: base + (m + 1) * MOVE_FEATURES]
        assert np.all(block == (3 - m) + 1)

    # other slots / non-move features untouched
    assert np.array_equal(rev[move_block_start(1):], vec[move_block_start(1):])


def test_permute_move_slots_rejects_non_permutation():
    vec = np.zeros(POKEMON_FEATURES * 2, dtype=np.float32)
    with pytest.raises(ValueError):
        permute_move_slots(vec, 0, (0, 0, 1, 2))


def test_permute_mask_row_and_unpermute_roundtrip():
    # move slot 1 legal at buckets 0 and 1; a switch legal too.
    row = [0] * 16
    row[1 * 3 + 0] = 1
    row[1 * 3 + 1] = 1
    row[12] = 1
    perm = (1, 0, 2, 3)                 # swap move slots 0 and 1
    prow = permute_mask_row(row, perm)
    # the legal buckets moved from OLD slot 1 to NEW slot 0
    assert prow[0 * 3 + 0] == 1 and prow[0 * 3 + 1] == 1
    assert prow[1 * 3 + 0] == 0
    assert prow[12] == 1               # switch unchanged

    # unpermute maps a permuted-space action back to the original move slot
    assert unpermute_action(0 * 3 + 0, perm) == 1 * 3 + 0   # new slot0 ← old slot1
    assert unpermute_action(12, perm) == 12                 # switch unchanged
    assert unpermute_action(None, perm) is None


def test_flatten_move_known():
    from state_encoder import MOVE_FEATURES
    vec = np.zeros(POKEMON_FEATURES * 2, dtype=np.float32)
    base = move_block_start(0)
    # set a confidence gradient: is_known = 1.0, 0.5, 0.0, 0.0 across the 4 slots
    for m, k in enumerate((1.0, 0.5, 0.0, 0.0)):
        vec[base + m * MOVE_FEATURES + (MOVE_FEATURES - 1)] = k
    flat = flatten_move_known(vec, 0)
    for m in range(NUM_MOVES):
        assert flat[base + m * MOVE_FEATURES + (MOVE_FEATURES - 1)] == 1.0
    # other slot untouched
    assert np.array_equal(flat[move_block_start(1):], vec[move_block_start(1):])
    # only the is_known byte changed (the rest of each move block is unchanged)
    assert flat[base] == vec[base]


def test_is_ally_mistarget():
    # Sneasler: Close Combat (damaging, 'normal') in slot 0, Helping Hand... use a
    # mon with a damaging move and an ally-support move.
    mon = {"species": "Talonflame",
           "revealed_moves": ["Acrobatics", "Tailwind"]}
    # action 0*3+2 = move slot 0 (Acrobatics, damaging) at the ALLY → mis-target
    assert is_ally_mistarget(mon, 0 * 3 + 2) is True
    # same move at a FOE bucket → not a mis-target
    assert is_ally_mistarget(mon, 0 * 3 + 0) is False
    # a switch → never a mis-target
    assert is_ally_mistarget(mon, 12) is False
    assert is_ally_mistarget(mon, None) is False

    # an ally-KIND move at the ally is NOT a mis-target (it belongs there)
    ally_mon = {"species": "Whimsicott", "revealed_moves": ["Helping Hand"]}
    assert is_ally_mistarget(ally_mon, 0 * 3 + 2) is False
