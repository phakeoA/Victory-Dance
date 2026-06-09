"""
player.py  —  VGC neural-network player for poke-env DoubleBattle
==================================================================
VGCPlayer wraps a PyTorch model that maps an encoded battle state to
two actions (one per active slot).  When the model is unavailable or
produces an illegal action it falls back to a random legal move.

A separate *team-chooser* model handles the teampreview lead selection
(which 4 Pokémon to send, and in what order).  Until that model is
available the player defaults to a heuristic (best average type
advantage against the visible opponent team).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Network contract  (battle model)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input  : float32 tensor  shape (STATE_DIM,)  or  (batch, STATE_DIM)
Output : int64   tensor  shape (2,)           or  (batch, 2)
           output[0] → action for active slot 0   (0–15)
           output[1] → action for active slot 1   (0–15)

Action encoding (matches state_encoder.py):
    0–11  →  move_idx (0–3)  ×  target (0=opp0, 1=opp1, 2=ally)
    12–15 →  switch to bench slot (0–3)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Network contract  (team-chooser model)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input  : float32 tensor  shape (TEAM_STATE_DIM,)   (TBD — your design)
Output : int64   tensor  shape (4,)
           indices into battle.teampreview_team (0-based, length-6 list)
           first two entries are the *leads* sent to active slots.
If model=None the heuristic fallback is used instead.

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
from pathlib import Path
from typing import List, Optional

import numpy as np

from poke_env.player import Player
from poke_env.battle import DoubleBattle, Move, Pokemon

from state_encoder import (
    StateEncoder,
    MOVE_TARGET_PAIRS,
    SWITCH_OFFSET,
    ACTIONS_PER_SLOT,
    STATE_DIM,
)

log = logging.getLogger(__name__)

# ── Optional torch import (graceful if not installed yet) ─────────────────────
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    log.warning("PyTorch not found — VGCPlayer will always use random fallback.")


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
        # track line offsets per battle_id so we can back-fill outcomes
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
        """
        Back-fill outcome (1=win, 0=loss, -1=draw) for all turns of a battle.
        Rewrites only the relevant lines in-place.
        """
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
    Currently just picks the first VGC_TEAM_SIZE mons in roster order.
    TODO: replace with type-advantage scoring once confirmed working.
    """
    team = list(battle.teampreview_team)
    n    = min(VGC_TEAM_SIZE, len(team))
    log.debug(
        "Teampreview: first-%d roster order → %s",
        n, [team[i].species for i in range(n)],
    )
    return list(range(n))


# ══════════════════════════════════════════════════════════════════════════════
# Action decoder
# ══════════════════════════════════════════════════════════════════════════════

def _build_legal_action_mask(
    battle: DoubleBattle,
    slot: int,
) -> list[bool]:
    """
    Return a 16-element bool list marking which actions are legal for `slot`.
    slot 0 = own active mon 0,  slot 1 = own active mon 1.
    """
    mask = [False] * ACTIONS_PER_SLOT

    try:
        active = battle.active_pokemon  # List[Optional[Pokemon]]
    except ValueError:
        return mask

    mon: Optional[Pokemon] = active[slot] if slot < len(active) else None
    if mon is None or mon.fainted:
        return mask

    # ── Move-target actions (0–11) ────────────────────────────────────────────
    available_moves = list(mon.moves.values())
    opp_active = battle.opponent_active_pokemon  # List[Optional[Pokemon]]
    opp0_alive = len(opp_active) > 0 and opp_active[0] is not None
    opp1_alive = len(opp_active) > 1 and opp_active[1] is not None
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
        # Target legality
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


def _action_to_poke_env(
    action: int,
    battle: DoubleBattle,
    slot: int,
):
    """
    Convert an integer action (0–15) into the poke-env order object
    for the given slot.  Returns None if the action cannot be executed
    (caller should fall back to random).
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
        # Move action
        move_idx, target_code = MOVE_TARGET_PAIRS[action]
        moves = list(mon.moves.values())
        if move_idx >= len(moves):
            return None
        move = moves[move_idx]

        # Resolve poke-env target
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

        return battle.create_order(move, move_target=target_mon)

    else:
        # Switch action
        bench_idx = action - SWITCH_OFFSET
        bench = [
            p for p in battle.team.values()
            if p not in set(filter(None, active)) and not p.fainted
        ]
        if bench_idx >= len(bench):
            return None
        return battle.create_order(bench[bench_idx])


def _random_legal_action(battle: DoubleBattle, slot: int) -> int:
    """Pick a uniformly random legal action for the slot."""
    mask = _build_legal_action_mask(battle, slot)
    legal = [i for i, ok in enumerate(mask) if ok]
    return random.choice(legal) if legal else 0


# ══════════════════════════════════════════════════════════════════════════════
# VGCPlayer
# ══════════════════════════════════════════════════════════════════════════════

class VGCPlayer(Player):
    """
    poke-env Player that uses a neural network to pick actions and,
    optionally, a separate model to choose the team during teampreview.

    Parameters
    ----------
    model : nn.Module or None
        Battle-action model.  Input shape (STATE_DIM,), output shape (2,).
        Pass None to always use random-legal fallback.
    team_chooser : nn.Module or None
        Teampreview model.  Input/output contract is project-specific
        (TBD when the network is designed).  Pass None to use the built-in
        type-advantage heuristic.
    replay_path : Path or None
        Where to write the JSON-lines replay buffer.
        Defaults to  replay_buffer/replay.jsonl
    device : str
        'cpu' or 'cuda' — where to run both models.
    **kwargs
        Forwarded to poke_env.player.Player (account_configuration,
        battle_format, team, max_concurrent_battles, …)
    """

    def __init__(
        self,
        model=None,
        team_chooser=None,
        replay_path: Optional[Path] = None,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._encoder      = StateEncoder()
        self._model        = model
        self._team_chooser = team_chooser
        self._device       = device

        if model is not None and _TORCH_AVAILABLE:
            self._model = model.to(device)
            self._model.eval()

        if team_chooser is not None and _TORCH_AVAILABLE:
            self._team_chooser = team_chooser.to(device)
            self._team_chooser.eval()

        _rp = replay_path or Path("replay_buffer/replay.jsonl")
        self._replay = ReplayBuffer(_rp)
        log.info(
            "VGCPlayer ready | battle_model=%s | team_chooser=%s | replay=%s",
            "loaded" if model        else "none (random fallback)",
            "loaded" if team_chooser else "none (heuristic fallback)",
            _rp,
        )

    # ── poke-env entry points ─────────────────────────────────────────────────

    def teampreview(self, battle: DoubleBattle) -> str:
        """
        Called once per battle during teampreview.
        Must return a Showdown /team command string, e.g. '/team 1324'.
        Indices are 1-based positions in battle.teampreview_team.

        Strategy:
          1. If team_chooser model is loaded → use it.
          2. Otherwise → first-N roster order.
        """
        # teampreview_team is a set populated from |poke| messages.
        # Fall back to battle.team if it's empty (shouldn't happen, but guard).
        team = list(battle.teampreview_team)
        if not team:
            log.warning("teampreview_team empty — falling back to battle.team")
            team = list(battle.team.values())

        max_size = battle.max_team_size if battle.max_team_size else VGC_TEAM_SIZE
        n = min(VGC_TEAM_SIZE, len(team), max_size)

        order: List[int]  # 0-based indices into `team`

        if self._team_chooser is not None and _TORCH_AVAILABLE:
            order = self._run_team_chooser(battle, team, n)
        else:
            order = _heuristic_team_order(battle)[:n]

        # Pad with any remaining mons not already selected (shouldn't be needed
        # for VGC, but guards against edge cases with smaller rosters).
        used = set(order)
        for i in range(len(team)):
            if len(order) >= n:
                break
            if i not in used:
                order.append(i)

        # poke-env expects 1-based indices
        showdown_order = "".join(str(i + 1) for i in order)
        species_names  = [team[i].species for i in order]
        log.info(
            "Teampreview [%s] → /team %s  (%s)",
            battle.battle_tag,
            showdown_order,
            ", ".join(species_names),
        )
        return f"/team {showdown_order}"

    def choose_move(self, battle):
        """Required by poke-env Player ABC — not used in doubles."""
        return self.choose_random_move(battle)

    def choose_doubles_move(self, battle: DoubleBattle):
        """Called by poke-env each turn to get the player's orders."""
        state_vec = self._encoder.encode(battle)

        action_s0, action_s1, source = self._select_actions(battle, state_vec)

        log.debug(
            "Turn %d [%s] a0=%d a1=%d src=%s",
            battle.turn, battle.battle_tag, action_s0, action_s1, source,
        )

        # Record transition (outcome filled in later)
        self._replay.record(
            battle_id  = battle.battle_tag,
            turn       = battle.turn,
            state      = state_vec,
            action_s0  = action_s0,
            action_s1  = action_s1,
            source     = source,
        )

        order_s0 = self._safe_order(action_s0, battle, slot=0)
        order_s1 = self._safe_order(action_s1, battle, slot=1)

        return self.create_doubles_order(order_s0, order_s1)

    # ── Team chooser (neural model path) ─────────────────────────────────────

    def _run_team_chooser(
        self,
        battle: DoubleBattle,
        team: list,
        n: int,
    ) -> List[int]:
        """
        Run the team_chooser model and return up to `n` 0-based team indices.
        Falls back to the heuristic on any error or illegal output.

        The input tensor format is intentionally left open — encode whatever
        teampreview features you need before passing the model in.
        This stub passes a zero vector of length 6*101 as a placeholder;
        replace the encoding block once your TEAM_STATE_DIM is defined.
        """
        try:
            # ── TODO: replace with real teampreview state encoding ────────────
            TEAM_STATE_DIM = len(team) * 101   # placeholder
            t = torch.zeros(TEAM_STATE_DIM, dtype=torch.float32, device=self._device)
            # ──────────────────────────────────────────────────────────────────

            with torch.no_grad():
                out = self._team_chooser(t)   # expected shape: (n,)
                indices = [int(x.item()) for x in out[:n]]

            # Validate: must be unique valid indices
            valid = [i for i in indices if 0 <= i < len(team)]
            if len(set(valid)) == n:
                return valid

            log.warning(
                "team_chooser produced invalid/duplicate indices %s — "
                "falling back to heuristic.", indices
            )
        except Exception as exc:
            log.warning("team_chooser inference failed (%s) — using heuristic.", exc)

        return _heuristic_team_order(battle)[:n]

    # ── Action selection ──────────────────────────────────────────────────────

    def _select_actions(
        self,
        battle: DoubleBattle,
        state_vec: np.ndarray,
    ) -> tuple[int, int, str]:
        """
        Run the model and return (action_s0, action_s1, source).
        Falls back to random if model is absent or produces illegal actions.
        """
        if self._model is None or not _TORCH_AVAILABLE:
            return (
                _random_legal_action(battle, 0),
                _random_legal_action(battle, 1),
                "random",
            )

        try:
            with torch.no_grad():
                t = torch.tensor(state_vec, dtype=torch.float32, device=self._device)
                out = self._model(t)          # shape (2,)
                a0  = int(out[0].item())
                a1  = int(out[1].item())

            mask0 = _build_legal_action_mask(battle, 0)
            mask1 = _build_legal_action_mask(battle, 1)

            used_random = False
            if not (0 <= a0 < ACTIONS_PER_SLOT and mask0[a0]):
                log.debug("Slot 0 model action %d illegal — random fallback.", a0)
                a0 = _random_legal_action(battle, 0)
                used_random = True
            if not (0 <= a1 < ACTIONS_PER_SLOT and mask1[a1]):
                log.debug("Slot 1 model action %d illegal — random fallback.", a1)
                a1 = _random_legal_action(battle, 1)
                used_random = True

            source = "random" if used_random else "model"
            return a0, a1, source

        except Exception as exc:
            log.warning("Model inference failed (%s) — using random.", exc)
            return (
                _random_legal_action(battle, 0),
                _random_legal_action(battle, 1),
                "random",
            )

    def _safe_order(self, action: int, battle: DoubleBattle, slot: int):
        """Convert action int → poke-env order, falling back to random if needed."""
        order = _action_to_poke_env(action, battle, slot)
        if order is not None:
            return order
        log.debug("_safe_order: action %d slot %d produced None — last-resort random.", action, slot)
        fallback_action = _random_legal_action(battle, slot)
        return _action_to_poke_env(fallback_action, battle, slot)

    # ── Battle result hook ────────────────────────────────────────────────────

    def _battle_finished_callback(self, battle: DoubleBattle) -> None:
        """Called by poke-env when a battle ends — back-fill outcomes."""
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
        """Call when done to flush and close the replay buffer."""
        self._replay.close()
