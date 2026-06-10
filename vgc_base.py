"""
vgc_base.py  —  Shared foundation for VGC Pokémon Showdown players
==================================================================
Contains:
  • ReplayBuffer          — JSON-lines per-turn transition recorder
  • Pure action helpers   — legal mask, action→order, random action
  • Teampreview heuristic — first-N roster order (TODO: type-advantage)
  • VGCPlayerBase         — abstract poke-env Player subclass

Concrete subclasses must implement _select_actions().

  RandomVGCPlayer  (random_player.py) — uniform random legal action
  VGCPlayer        (player.py)        — neural-network driven

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Action encoding (matches state_encoder.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    0–11  →  move_idx (0–3)  ×  target (0=opp0, 1=opp1, 2=ally)
    12–15 →  switch to bench slot (0–3)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Replay buffer format  (one JSON object per line)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{
  "battle_id"  : str,
  "turn"       : int,
  "state"      : [float, ...],   # STATE_DIM floats
  "action_s0"  : int,            # action taken for slot 0
  "action_s1"  : int,            # action taken for slot 1
  "source"     : "model"|"random",
  "outcome"    : 1|0|-1|null     # 1=win 0=loss -1=draw; null until battle ends
}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import json
import logging
import random
from abc import abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from poke_env.player import Player
from poke_env.player.battle_order import DoubleBattleOrder, PassBattleOrder, SingleBattleOrder
from poke_env.battle import DoubleBattle, Move, Pokemon

from state_encoder import (
    StateEncoder,
    MOVE_TARGET_PAIRS,
    SWITCH_OFFSET,
    ACTIONS_PER_SLOT,
    STATE_DIM,
)

log = logging.getLogger(__name__)

# ── Target constants (mirror state_encoder action space) ──────────────────────
_TARGET_OPP0 = 0
_TARGET_OPP1 = 1
_TARGET_ALLY = 2

# VGC: choose 4 leads from a 6-mon roster
VGC_TEAM_SIZE = 4


# ══════════════════════════════════════════════════════════════════════════════
# Replay buffer
# ══════════════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    """
    Appends transitions to a JSON-lines file.
    Each line is one turn.  On battle end, outcome is written back to all
    turns from that battle by rewriting only the relevant lines.
    """

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("a", encoding="utf-8")
        self._battle_line_indices: dict[str, list[int]] = {}
        self._line_count = _count_lines(path)

    def record(
        self,
        battle_id: str,
        turn: int,
        state: np.ndarray,
        action_s0: int,
        action_s1: int,
        source: str,
    ) -> None:
        entry = {
            "battle_id": battle_id,
            "turn":      turn,
            "state":     state.tolist(),
            "action_s0": action_s0,
            "action_s1": action_s1,
            "source":    source,
            "outcome":   None,
        }
        self._handle.write(json.dumps(entry) + "\n")
        self._handle.flush()
        self._battle_line_indices.setdefault(battle_id, []).append(self._line_count)
        self._line_count += 1

    def finalise(self, battle_id: str, outcome: int) -> None:
        """Back-fill outcome (1=win, 0=loss, -1=draw) for all turns of a battle."""
        indices = self._battle_line_indices.pop(battle_id, [])
        if not indices:
            return
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
            for idx in indices:
                if idx < len(lines):
                    obj = json.loads(lines[idx])
                    obj["outcome"] = outcome
                    lines[idx] = json.dumps(obj)
            self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception as exc:
            log.warning("ReplayBuffer.finalise failed for %s: %s", battle_id, exc)

    def close(self) -> None:
        self._handle.close()


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


# ══════════════════════════════════════════════════════════════════════════════
# Teampreview heuristic
# ══════════════════════════════════════════════════════════════════════════════

def _heuristic_team_order(battle: DoubleBattle) -> List[int]:
    """
    Return a list of 0-based indices into battle.teampreview_team.
    Currently picks the first VGC_TEAM_SIZE mons in roster order.
    TODO: replace with type-advantage scoring once confirmed working.
    """
    team = list(battle.teampreview_team)
    n    = min(VGC_TEAM_SIZE, len(team))
    log.debug("Teampreview: first-%d roster order → %s", n, [team[i].species for i in range(n)])
    return list(range(n))


# ══════════════════════════════════════════════════════════════════════════════
# Action helpers  (pure functions, no player state)
# ══════════════════════════════════════════════════════════════════════════════

def build_legal_action_mask(battle: DoubleBattle, slot: int) -> list[bool]:
    """
    Return a 16-element bool list marking which actions are legal for `slot`.
    slot 0 = own active mon 0,  slot 1 = own active mon 1.
    """
    mask = [False] * ACTIONS_PER_SLOT

    try:
        active = battle.active_pokemon
    except ValueError:
        return mask

    mon: Optional[Pokemon] = active[slot] if slot < len(active) else None
    if mon is None or mon.fainted:
        return mask

    # ── Move-target actions (0–11) ────────────────────────────────────────────
    available_moves = list(mon.moves.values())
    opp_active  = battle.opponent_active_pokemon
    opp0_alive  = len(opp_active) > 0 and opp_active[0] is not None
    opp1_alive  = len(opp_active) > 1 and opp_active[1] is not None
    ally_slot   = 1 - slot
    ally_alive  = (
        len(active) > ally_slot
        and active[ally_slot] is not None
        and not active[ally_slot].fainted
    )

    for action_idx, (move_idx, target) in enumerate(MOVE_TARGET_PAIRS):
        if move_idx >= len(available_moves):
            continue
        move: Move = available_moves[move_idx]
        if move.current_pp == 0:
            continue
        if target == _TARGET_OPP0 and not opp0_alive:
            continue
        if target == _TARGET_OPP1 and not opp1_alive:
            continue
        if target == _TARGET_ALLY and not ally_alive:
            continue
        mask[action_idx] = True

    # ── Switch actions (12–15) ────────────────────────────────────────────────
    bench = [
        p for p in battle.team.values()
        if p not in set(filter(None, active)) and not p.fainted
    ]
    for bench_idx, _ in enumerate(bench[:4]):
        mask[SWITCH_OFFSET + bench_idx] = True

    return mask


def action_to_order(action: int, battle: DoubleBattle, slot: int) -> Optional[SingleBattleOrder]:
    """
    Convert an integer action (0–15) into a SingleBattleOrder for the given slot.
    Returns None if the action cannot be executed (caller should fall back).

    Uses Player.create_order (static) with an integer move_target from
    DoubleBattle.to_showdown_target(), as required by poke-env's doubles API.
    """
    try:
        active = battle.active_pokemon
    except ValueError:
        return None

    mon: Optional[Pokemon] = active[slot] if slot < len(active) else None
    if mon is None:
        return None

    opp_active = battle.opponent_active_pokemon

    if action < SWITCH_OFFSET:
        move_idx, target_code = MOVE_TARGET_PAIRS[action]
        moves = list(mon.moves.values())
        if move_idx >= len(moves):
            return None
        move = moves[move_idx]

        if target_code == _TARGET_OPP0:
            targets = [p for p in opp_active if p is not None]
            target_mon = targets[0] if targets else None
        elif target_code == _TARGET_OPP1:
            targets = [p for p in opp_active if p is not None]
            target_mon = targets[1] if len(targets) > 1 else (targets[0] if targets else None)
        else:  # ally
            ally_slot = 1 - slot
            target_mon = active[ally_slot] if ally_slot < len(active) else None

        if target_mon is None:
            return None

        showdown_target = battle.to_showdown_target(move, target_mon)
        return Player.create_order(move, move_target=showdown_target)

    else:
        bench_idx = action - SWITCH_OFFSET
        bench = [
            p for p in battle.team.values()
            if p not in set(filter(None, active)) and not p.fainted
        ]
        if bench_idx >= len(bench):
            return None
        return Player.create_order(bench[bench_idx])


def random_legal_action(battle: DoubleBattle, slot: int) -> int:
    """Pick a uniformly random legal action for the slot."""
    mask = build_legal_action_mask(battle, slot)
    legal = [i for i, ok in enumerate(mask) if ok]
    return random.choice(legal) if legal else 0


# ══════════════════════════════════════════════════════════════════════════════
# Abstract base player
# ══════════════════════════════════════════════════════════════════════════════

class VGCPlayerBase(Player):
    """
    Abstract poke-env Player for VGC doubles.

    Handles teampreview, forceSwitch, turn routing, replay recording, and
    order construction.  Subclasses only need to implement _select_actions().

    Parameters
    ----------
    replay_path : Path or None
        Where to write the JSON-lines replay buffer.
        Defaults to replay_buffer/replay.jsonl
    **kwargs
        Forwarded to poke_env.player.Player.
    """

    def __init__(self, replay_path: Optional[Path] = None, **kwargs):
        super().__init__(**kwargs)
        self._encoder = StateEncoder()
        _rp = replay_path or Path("replay_buffer/replay.jsonl")
        self._replay  = ReplayBuffer(_rp)

    # ── Subclass contract ─────────────────────────────────────────────────────

    @abstractmethod
    def _select_actions(
        self,
        battle: DoubleBattle,
        state_vec: np.ndarray,
    ) -> Tuple[int, int, str]:
        """
        Given the encoded battle state, return (action_s0, action_s1, source).
        source is a short string label, e.g. "model" or "random".
        Both actions must be in [0, ACTIONS_PER_SLOT).
        """

    # ── poke-env entry points ─────────────────────────────────────────────────

    def teampreview(self, battle: DoubleBattle) -> str:
        """
        Called once per battle during teampreview.
        Returns a Showdown /team command, e.g. '/team 1234'.

        This format doesn't send |poke| messages so teampreview_team is
        always empty — falls back to battle.team directly.
        """
        team = list(battle.teampreview_team) or list(battle.team.values())
        max_size = battle.max_team_size if battle.max_team_size else VGC_TEAM_SIZE
        n = min(VGC_TEAM_SIZE, len(team), max_size)

        order = self._choose_team_order(battle, team, n)

        # Pad with unchosen mons if needed (edge-case guard)
        used = set(order)
        for i in range(len(team)):
            if len(order) >= n:
                break
            if i not in used:
                order.append(i)

        showdown_order = "".join(str(i + 1) for i in order)
        species_names  = [team[i].species for i in order]
        log.info(
            "Teampreview [%s] → /team %s  (%s)",
            battle.battle_tag, showdown_order, ", ".join(species_names),
        )
        return f"/team {showdown_order}"

    def choose_move(self, battle: DoubleBattle):
        """
        Called by poke-env every turn.  Handles three distinct request types:

        1. forceSwitch — one or both active slots fainted; send bench replacements.
                         Slots that did NOT faint receive PassBattleOrder.
        2. Normal turn — encode state, call _select_actions, record to replay.
        3. Non-doubles — shouldn't happen, falls back to choose_random_move.
        """
        if not isinstance(battle, DoubleBattle):
            return self.choose_random_move(battle)

        if any(battle.force_switch):
            return self._handle_force_switch(battle)

        state_vec = self._encoder.encode(battle)
        action_s0, action_s1, source = self._select_actions(battle, state_vec)

        log.debug(
            "Turn %d [%s] a0=%d a1=%d src=%s",
            battle.turn, battle.battle_tag, action_s0, action_s1, source,
        )

        self._replay.record(
            battle_id = battle.battle_tag,
            turn      = battle.turn,
            state     = state_vec,
            action_s0 = action_s0,
            action_s1 = action_s1,
            source    = source,
        )

        order_s0 = self._safe_order(action_s0, battle, slot=0)
        order_s1 = self._safe_order(action_s1, battle, slot=1)
        return DoubleBattleOrder(order_s0, order_s1)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _choose_team_order(self, battle: DoubleBattle, team: list, n: int) -> List[int]:
        """
        Return up to n 0-based team indices for teampreview.
        Base implementation uses the first-N heuristic.
        Override in subclasses that have a team-chooser model.
        """
        return _heuristic_team_order(battle)[:n]

    def _handle_force_switch(self, battle: DoubleBattle) -> DoubleBattleOrder:
        """
        Handle a forceSwitch request.

        force_switch[i] == True  → slot fainted; pick a bench replacement.
        force_switch[i] == False → slot still alive; send PassBattleOrder.

        battle.available_switches is List[List[Pokemon]], one list per slot,
        already filtered by poke-env to exclude fainted / already-active mons.
        """
        force    = battle.force_switch
        switches = battle.available_switches

        orders: list        = []
        used_switches: set  = set()

        for slot in range(2):
            if slot >= len(force) or not force[slot]:
                orders.append(PassBattleOrder())
            else:
                candidates = [
                    p for p in (switches[slot] if slot < len(switches) else [])
                    if id(p) not in used_switches
                ]
                if candidates:
                    chosen = random.choice(candidates)
                    used_switches.add(id(chosen))
                    orders.append(Player.create_order(chosen))
                else:
                    log.warning("forceSwitch slot %d: no available switches — sending Pass.", slot)
                    orders.append(PassBattleOrder())

        order_s0 = orders[0] if len(orders) > 0 else PassBattleOrder()
        order_s1 = orders[1] if len(orders) > 1 else PassBattleOrder()

        log.debug(
            "forceSwitch [%s] force=%s → %s / %s",
            battle.battle_tag, force, order_s0.message, order_s1.message,
        )
        return DoubleBattleOrder(order_s0, order_s1)

    def _safe_order(self, action: int, battle: DoubleBattle, slot: int) -> SingleBattleOrder:
        """
        Convert action int → SingleBattleOrder, with two fallback levels:
          1. Try the given action.
          2. Try a fresh random legal action.
          3. PassBattleOrder (slot has nothing to do this turn).
        DoubleBattleOrder never accepts None, so Pass is the correct no-op.
        """
        order = action_to_order(action, battle, slot)
        if order is not None:
            return order
        log.debug("_safe_order: action %d slot %d → None, trying random fallback.", action, slot)
        order = action_to_order(random_legal_action(battle, slot), battle, slot)
        if order is not None:
            return order
        log.debug("_safe_order: slot %d has no legal actions — sending Pass.", slot)
        return PassBattleOrder()

    def _battle_finished_callback(self, battle: DoubleBattle) -> None:
        """Called by poke-env when a battle ends — back-fills outcomes in replay."""
        if battle.won:
            outcome = 1
        elif battle.lost:
            outcome = 0
        else:
            outcome = -1
        self._replay.finalise(battle.battle_tag, outcome)
        log.info(
            "Battle %s finished — outcome: %s  (W/L/D so far: %d/%d/%d)",
            battle.battle_tag,
            {1: "WIN", 0: "LOSS", -1: "DRAW"}[outcome],
            self.n_won_battles,
            self.n_lost_battles,
            self.n_tied_battles,
        )

    def close(self) -> None:
        """Flush and close the replay buffer."""
        self._replay.close()
