"""B-L1 serve-time pattern tilt — Phase-3 design lever L1 (docs/phase3_adaptation_loop_design.md).

Case study #1: the USER spammed Wide Guard and the net kept clicking spread moves (it SEES the
streak via the encoder's protect_counter but cloned ~1600-elo demonstrators who don't respond).
The trained net stays FROZEN; when the opponent's side has used a spread-blocking move (Wide
Guard / Mat Block) on ≥ STREAK_MIN consecutive completed turns, we add a NEGATIVE logit bias to
the spread-move actions so the model's own single-target/status/reposition preferences win
unless spread is overwhelmingly preferred. A tilt, not a mask — the model still chooses.

Default OFF everywhere (``adapt_rules=False`` / ``--adapt-rules``); prod serving byte-identical.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from v_dance.encoders.encoder_layout import ACTIONS_PER_SLOT

log = logging.getLogger(__name__)

# Spread-BLOCKING moves (the streak trigger). Deliberately NOT the whole protect family —
# plain Protect blocks single-target too, so tilting away from spread is only correct for the
# wide blockers (keep the rule list SHORT; every rule is a new habit to exploit).
WIDE_GUARD_MOVES = {"wideguard", "matblock"}
# Move DEX target kinds whose codec action is the spread bucket (m_idx*3 + 0).
SPREAD_KINDS = {"allAdjacentFoes", "allAdjacent"}

STREAK_MIN = 2          # consecutive opp turns with a wide blocker before the tilt fires
SPREAD_BIAS = -2.5      # logit bias on spread actions while the streak holds


def _norm(s: str) -> str:
    return "".join(c for c in (s or "").lower() if c.isalnum())


def wide_guard_streak(proto_lines: Optional[List[str]], opp_role: str) -> int:
    """Consecutive COMPLETED turns (ending at the most recent) on which the opponent's side
    used a Wide Guard-family move, parsed from the raw protocol log.

    ``|move|p2a: Nick|Wide Guard|...`` lines attribute the move; ``|turn|N`` lines delimit
    turns. The current (still-deciding) turn has no move lines yet and is excluded.
    """
    if not proto_lines:
        return 0
    used_by_turn = {}
    cur = 0
    for ln in proto_lines:
        parts = ln.split("|")
        if len(parts) >= 3 and parts[1] == "turn":
            try:
                cur = int(parts[2])
            except ValueError:
                continue
            used_by_turn.setdefault(cur, False)
        elif len(parts) >= 4 and parts[1] == "move" and cur >= 1:
            if parts[2].strip().lower().startswith(opp_role) and _norm(parts[3]) in WIDE_GUARD_MOVES:
                used_by_turn[cur] = True
    if not used_by_turn:
        return 0
    last_completed = max(used_by_turn) - 1     # the max |turn| N is the one being decided NOW
    streak = 0
    t = last_completed
    while t >= 1 and used_by_turn.get(t, False):
        streak += 1
        t -= 1
    return streak


def spread_bias_for_kinds(move_kinds: List[Optional[str]],
                          bias: float = SPREAD_BIAS) -> Optional[np.ndarray]:
    """(ACTIONS_PER_SLOT,) float32 bias vector penalising the spread bucket (m*3) of every
    spread move in ``move_kinds`` (index-aligned with the slot's available-move list), or
    None when the slot has no spread move (no tilt to apply)."""
    arr = np.zeros(ACTIONS_PER_SLOT, dtype=np.float32)
    hit = False
    for m_idx, kind in enumerate(move_kinds[:4]):
        if kind in SPREAD_KINDS:
            arr[m_idx * 3] = bias
            hit = True
    return arr if hit else None


def action_biases(player, battle) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Per-slot logit-bias vectors for this decision (None, None when no rule fires).
    Reads the player's raw per-battle protocol log — serve-only, no encoder coupling."""
    from v_dance.play.live_vgc_base import _norm_tag                    # no cycle at import time
    from v_dance.play.vgc_base import _move_target_kind, own_active_move_list

    role = getattr(battle, "player_role", None) or "p1"
    opp_role = "p2" if role == "p1" else "p1"
    lines = (getattr(player, "_proto_log", None) or {}).get(_norm_tag(battle.battle_tag))
    streak = wide_guard_streak(lines, opp_role)
    if streak < STREAK_MIN:
        return None, None

    biases: List[Optional[np.ndarray]] = [None, None]
    try:
        active = battle.active_pokemon
    except ValueError:
        return None, None
    for slot in (0, 1):
        mon = active[slot] if slot < len(active) else None
        if mon is None or getattr(mon, "fainted", False):
            continue
        kinds = [_move_target_kind(mv.id) for mv in own_active_move_list(battle, slot, mon)]
        biases[slot] = spread_bias_for_kinds(kinds)
    if any(b is not None for b in biases):
        log.info("Turn %s [%s] adapt-rules: opp Wide Guard streak=%d → spread tilt %.1f ON",
                 getattr(battle, "turn", "?"), battle.battle_tag, streak, SPREAD_BIAS)
    return biases[0], biases[1]
