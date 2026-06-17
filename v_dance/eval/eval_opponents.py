"""
local_battle/eval_opponents.py  —  scripted baseline opponents for the gauntlet
===============================================================================
Two non-trained reference opponents stronger than the random player, so the
win-rate gauntlet (#3) measures the BC model against a SKILL LADDER instead of a
single floor:

  * ``MaxDamageVGCPlayer``  — each slot uses its highest-base-power legal move
    (never wastes a turn on a status move when a damaging one is up, never
    attacks its own ally), switching only when no move is legal.
  * ``HeuristicVGCPlayer``  — scores each legal (move, target) by
    ``base_power x type-effectiveness x STAB`` against the ACTUAL foe and breaks
    ties toward KO-ing the faster opponent.  A simple type/speed player.

Both reuse the shared VGC codec (``build_legal_action_mask`` / ``action_to_order``
in vgc_base) so every order they emit is legal in this doubles format, and both
subclass the ROOT ``VGCPlayerBase`` (no gap-#6 splice — only OUR model needs the
opponent reconstruction; an opponent just needs to play legally).

Action index layout (matches state_encoder): ``i < 12`` → move ``i//3`` at target
``i%3`` (0=opp0, 1=opp1, 2=ally); ``i >= 12`` → switch to bench slot ``i-12``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np

# ── Path bootstrap (local_battle FIRST, then repo root + data/scripts) ────────
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
from poke_env.battle import DoubleBattle  # noqa: E402

from v_dance.play.vgc_base import VGCPlayerBase, build_legal_action_mask  # noqa: E402
from v_dance.encoders.state_encoder import SWITCH_OFFSET, NUM_MOVES  # noqa: E402

log = logging.getLogger(__name__)

_TARGET_ALLY = 2          # action target bucket 2 == own ally
_ALLY_PENALTY = -1e6      # never pick a damaging move aimed at our own ally
_SWITCH_SCORE = -1.0      # prefer ANY legal move over a switch
_NO_MOVE = -2.0           # undecodable move slot


def _active_mon(battle: DoubleBattle, slot: int):
    try:
        active = battle.active_pokemon
    except ValueError:
        return None
    return active[slot] if slot < len(active) else None


def _move_for(battle: DoubleBattle, slot: int, move_idx: int):
    """The Move object behind move-index ``move_idx`` for ``slot`` — same ordering
    (``list(mon.moves.values())[:NUM_MOVES]``) the codec / action mask use."""
    mon = _active_mon(battle, slot)
    if mon is None:
        return None
    moves = list(mon.moves.values())[:NUM_MOVES]
    return moves[move_idx] if move_idx < len(moves) else None


def _foe(battle: DoubleBattle, bucket: int):
    """The opponent mon in target bucket 0/1 (positional), or None."""
    try:
        opp = battle.opponent_active_pokemon
    except (ValueError, AttributeError):
        return None
    return opp[bucket] if (opp and len(opp) > bucket) else None


class _ScoringVGCPlayer(VGCPlayerBase):
    """Shared base: pick, per slot, the legal action with the highest ``_score``.

    Subclasses implement ``_score(battle, slot, action) -> float``.  The cross-slot
    switch dedup (both slots choosing the same bench mon — "can only switch in
    once") mirrors the model player's serve-side dedup."""

    _source = "scripted"

    def __init__(self, replay_path: Optional[Path] = None, **kwargs):
        super().__init__(replay_path=replay_path, **kwargs)

    def _best_action(self, battle: DoubleBattle, slot: int,
                     exclude: Iterable[int] = ()) -> int:
        mask = build_legal_action_mask(battle, slot)
        excl = set(exclude)
        legal = [i for i, ok in enumerate(mask) if ok and i not in excl]
        if not legal:
            return 0
        return max(legal, key=lambda i: self._score(battle, slot, i))

    def _select_actions(self, battle: DoubleBattle,
                        state_vec: np.ndarray) -> Tuple[int, int, str]:
        a0 = self._best_action(battle, 0)
        a1 = self._best_action(battle, 1)
        # Both slots can't switch the SAME bench mon in — re-pick slot 1.
        if a0 == a1 and a0 >= SWITCH_OFFSET:
            a1 = self._best_action(battle, 1, exclude={a0})
        return a0, a1, self._source

    def _score(self, battle: DoubleBattle, slot: int, action: int) -> float:
        raise NotImplementedError


class MaxDamageVGCPlayer(_ScoringVGCPlayer):
    """Highest-base-power legal move per slot (foe-targeted)."""

    _source = "max_damage"

    def _score(self, battle: DoubleBattle, slot: int, action: int) -> float:
        if action >= SWITCH_OFFSET:
            return _SWITCH_SCORE
        move = _move_for(battle, slot, action // 3)
        if move is None:
            return _NO_MOVE
        if action % 3 == _TARGET_ALLY:
            return _ALLY_PENALTY
        return float(getattr(move, "base_power", 0) or 0)


class HeuristicVGCPlayer(_ScoringVGCPlayer):
    """base_power x type-effectiveness x STAB against the actual foe; ties broken
    toward the faster foe (a simple type/speed player)."""

    _source = "heuristic"
    _STAB = 1.5
    _SPEED_TIEBREAK = 1e-4

    def _score(self, battle: DoubleBattle, slot: int, action: int) -> float:
        if action >= SWITCH_OFFSET:
            return _SWITCH_SCORE
        bucket = action % 3
        move = _move_for(battle, slot, action // 3)
        if move is None:
            return _NO_MOVE
        if bucket == _TARGET_ALLY:
            return _ALLY_PENALTY
        bp = float(getattr(move, "base_power", 0) or 0)
        if bp <= 0:
            return 0.0                              # status move: low priority
        foe = _foe(battle, bucket)
        eff, spe = 1.0, 0.0
        if foe is not None:
            try:
                eff = float(foe.damage_multiplier(move))
            except Exception:                       # pragma: no cover - defensive
                eff = 1.0
            spe = float((getattr(foe, "base_stats", None) or {}).get("spe", 0) or 0)
        mon = _active_mon(battle, slot)
        stab = self._STAB if (mon is not None and move.type in (mon.types or [])) else 1.0
        return bp * eff * stab + spe * self._SPEED_TIEBREAK
