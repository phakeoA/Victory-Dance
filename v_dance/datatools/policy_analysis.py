"""
policy_analysis.py — torch-free, unit-testable helpers for analysing the BC policy
offline.  Two diagnostics live on top of these:

  • ALLY MIS-TARGET (#12): does the policy's masked-argmax aim a DAMAGING (non
    ally-kind) move at our own ally (bucket 2)?  — the "attacked its teammate" signal.

  • MOVE-ORDER SENSITIVITY (#13): training orders a mon's 4 move slots by reveal
    order (move_slots_for_mon), the LIVE encoder by mon.moves.values() (request)
    order.  If the net is order-INVARIANT this residual is harmless.  Permuting a
    mon's 4 move-feature blocks (and the matching action-mask buckets) and checking
    whether the policy still picks the SAME PHYSICAL move quantifies that.

The helpers here are pure numpy / dict logic so they unit-test without torch; the
corpus scan that needs the model lives in tests/_probe_policy.py.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
from v_dance.encoders.state_encoder import (  # noqa: E402
    POKEMON_FEATURES, NUM_TYPES, NUM_STATUS, NUM_BOOSTS, NUM_MOVES, MOVE_FEATURES,
    ITEM_FEATURES, ABILITY_FEATURES,
    SWITCH_OFFSET, move_slots_for_mon, _move_target_kind, _ALLY_KINDS,
)

# Offset of the first MOVE feature block within a POKEMON slot — derived from the
# frozen layout (hp + 2×types + base6 + est6 + known1 + mega1 + tera1 + status +
# boosts), NOT hardcoded, so it can't silently drift from POKEMON_FEATURES.
MOVE_BLOCK_START = 1 + 2 * NUM_TYPES + 6 + 6 + 1 + 1 + 1 + NUM_STATUS + NUM_BOOSTS
# Sanity: the 4 move blocks are followed by the item + ability effect blocks
# (gap #5) and then exactly 4 tail flags (is_active/is_revealed/is_fainted/
# is_transformed).
assert (MOVE_BLOCK_START + NUM_MOVES * MOVE_FEATURES
        + ITEM_FEATURES + ABILITY_FEATURES + 4 == POKEMON_FEATURES), (
    f"move-block offset {MOVE_BLOCK_START} inconsistent with POKEMON_FEATURES "
    f"{POKEMON_FEATURES}"
)

# Active own slots in the state layout: our_a = slot 0, our_b = slot 1.
HEAD_SLOT = {"our_a": 0, "our_b": 1}


def move_block_start(slot_index: int) -> int:
    """Byte offset of move slot 0's feature block for active ``slot_index``."""
    return slot_index * POKEMON_FEATURES + MOVE_BLOCK_START


# ── #12: ally mis-target classifier ───────────────────────────────────────────
def is_ally_mistarget(mon: Optional[dict], action_index: Optional[int]) -> bool:
    """True iff ``action_index`` aims at the ally (target bucket 2) with a move that
    is NOT an ally-kind move — i.e. a damaging / foe-targeting move pointed at our
    own teammate.  Ally-support moves (Helping Hand / Coaching = adjacentAlly) are
    NOT counted (their natural target is the ally)."""
    if action_index is None or action_index >= SWITCH_OFFSET:
        return False
    m_idx, bucket = divmod(action_index, 3)
    if bucket != 2:
        return False
    slots = move_slots_for_mon(mon)
    if m_idx >= len(slots):
        return False
    return _move_target_kind(slots[m_idx][0]) not in _ALLY_KINDS


# ── #13: move-slot permutation (order-invariance probe) ────────────────────────
def permute_move_slots(vec: np.ndarray, slot_index: int, perm: Sequence[int]) -> np.ndarray:
    """Return a COPY of ``vec`` with the 4 MOVE feature blocks of active
    ``slot_index`` reordered so NEW move slot m holds OLD move slot ``perm[m]``."""
    if sorted(perm) != list(range(NUM_MOVES)):
        raise ValueError(f"perm must be a permutation of 0..{NUM_MOVES-1}, got {perm}")
    out = vec.copy()
    base = move_block_start(slot_index)
    for m in range(NUM_MOVES):
        src = base + perm[m] * MOVE_FEATURES
        dst = base + m * MOVE_FEATURES
        out[dst:dst + MOVE_FEATURES] = vec[src:src + MOVE_FEATURES]
    return out


def permute_mask_row(row: Sequence[int], perm: Sequence[int]) -> List[int]:
    """Permute the 12 move-target buckets of a 16-action mask row so NEW move slot
    m ← OLD move slot ``perm[m]``; the 4 switch indices (12-15) are unchanged."""
    out = list(row)
    for m in range(NUM_MOVES):
        for t in range(3):
            out[m * 3 + t] = row[perm[m] * 3 + t]
    return out


def flatten_move_known(vec: np.ndarray, slot_index: int) -> np.ndarray:
    """Return a COPY of ``vec`` with the ``is_known`` feature (last of each
    MOVE_FEATURES block) set to 1.0 for all 4 move slots of active ``slot_index``.

    Training Type-B own mons carry a confidence GRADIENT (revealed moves is_known
    1.0, belief-padded moves p<1.0), but the LIVE encoder emits is_known=1.0 for
    every own move (all 4 come from the request).  Flattening reproduces the serve
    distribution, so an order-sensitivity measured on a flattened board isolates the
    FEATURE-driven residual a live bot would actually see."""
    out = vec.copy()
    base = move_block_start(slot_index)
    for m in range(NUM_MOVES):
        out[base + m * MOVE_FEATURES + (MOVE_FEATURES - 1)] = 1.0
    return out


def unpermute_action(action: Optional[int], perm: Sequence[int]) -> Optional[int]:
    """Map an action chosen in PERMUTED space back to the original space: a move
    action m*3+t in the permuted board refers to OLD move slot perm[m].  Switch
    actions (>=12) and None are returned unchanged."""
    if action is None or action >= SWITCH_OFFSET:
        return action
    m, t = divmod(action, 3)
    return perm[m] * 3 + t
