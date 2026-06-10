"""
player.py  —  Neural-network VGC player
========================================
VGCPlayer loads a trained PyTorch model from disk and uses it to pick
actions each turn.  When no model path is supplied it delegates entirely
to RandomVGCPlayer, so the two are always drop-in replacements for each
other.

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
           indices into the teampreview roster (0-based)
           first two entries are the *leads* sent to active slots.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Usage
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    from player import VGCPlayer
    from poke_env import AccountConfiguration

    # With a trained model:
    bot = VGCPlayer(
        model_path=Path("checkpoints/battle_model.pt"),
        replay_path=Path("replay_buffer/TrainerRed.jsonl"),
        account_configuration=AccountConfiguration("TrainerRed", None),
        battle_format="gen9championsvgc2026regma",
        team=team_str,
    )

    # No model yet — behaves identically to RandomVGCPlayer:
    bot = VGCPlayer(
        replay_path=Path("replay_buffer/TrainerRed.jsonl"),
        account_configuration=AccountConfiguration("TrainerRed", None),
        battle_format="gen9championsvgc2026regma",
        team=team_str,
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from poke_env.battle import DoubleBattle

from vgc_base import (
    VGCPlayerBase,
    _heuristic_team_order,
    build_legal_action_mask,
    random_legal_action,
    VGC_TEAM_SIZE,
)
from state_encoder import ACTIONS_PER_SLOT, STATE_DIM

log = logging.getLogger(__name__)

# ── Optional torch import ─────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    log.warning("PyTorch not found — VGCPlayer will always use random fallback.")


class VGCPlayer(VGCPlayerBase):
    """
    VGC player driven by a trained PyTorch model.

    If model_path is None (or PyTorch is unavailable), action selection falls
    back to uniform random — identical behaviour to RandomVGCPlayer.

    Parameters
    ----------
    model_path : Path or str or None
        Path to a saved nn.Module (torch.save / torch.load).
        Pass None to run without a model (random fallback).
    team_chooser_path : Path or str or None
        Path to a saved team-chooser nn.Module.
        Pass None to use the first-N heuristic.
    replay_path : Path or None
        Where to write the JSON-lines replay buffer.
    device : str
        'cpu' or 'cuda'.
    **kwargs
        Forwarded to poke_env.player.Player.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        team_chooser_path: Optional[Path] = None,
        replay_path: Optional[Path] = None,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(replay_path=replay_path, **kwargs)

        self._device       = device
        self._model        = None
        self._team_chooser = None

        # ── Load battle model ─────────────────────────────────────────────────
        if model_path is not None and _TORCH_AVAILABLE:
            try:
                self._model = torch.load(
                    model_path, map_location=device, weights_only=False
                )
                self._model.eval()
                log.info("VGCPlayer: loaded battle model from %s", model_path)
            except Exception as exc:
                log.error(
                    "VGCPlayer: failed to load battle model from %s (%s) "
                    "— falling back to random.",
                    model_path, exc,
                )

        # ── Load team-chooser model ───────────────────────────────────────────
        if team_chooser_path is not None and _TORCH_AVAILABLE:
            try:
                self._team_chooser = torch.load(
                    team_chooser_path, map_location=device, weights_only=False
                )
                self._team_chooser.eval()
                log.info("VGCPlayer: loaded team-chooser from %s", team_chooser_path)
            except Exception as exc:
                log.error(
                    "VGCPlayer: failed to load team-chooser from %s (%s) "
                    "— using heuristic.",
                    team_chooser_path, exc,
                )

        log.info(
            "VGCPlayer ready | battle_model=%s | team_chooser=%s | replay=%s",
            f"loaded ({model_path})" if self._model else "none (random fallback)",
            f"loaded ({team_chooser_path})" if self._team_chooser else "none (heuristic)",
            replay_path or Path("replay_buffer/replay.jsonl"),
        )

    # ── Action selection ──────────────────────────────────────────────────────

    def _select_actions(
        self,
        battle: DoubleBattle,
        state_vec: np.ndarray,
    ) -> Tuple[int, int, str]:
        """
        Run the battle model and return (action_s0, action_s1, source).
        Any illegal output is corrected with a random legal action.
        Falls back to fully random if model is absent or raises.
        """
        if self._model is None or not _TORCH_AVAILABLE:
            return (
                random_legal_action(battle, 0),
                random_legal_action(battle, 1),
                "random",
            )

        try:
            with torch.no_grad():
                t   = torch.tensor(state_vec, dtype=torch.float32, device=self._device)
                out = self._model(t)    # expected shape: (2,)
                a0  = int(out[0].item())
                a1  = int(out[1].item())

            mask0 = build_legal_action_mask(battle, 0)
            mask1 = build_legal_action_mask(battle, 1)

            used_random = False
            if not (0 <= a0 < ACTIONS_PER_SLOT and mask0[a0]):
                log.debug("Slot 0 model action %d illegal — random fallback.", a0)
                a0 = random_legal_action(battle, 0)
                used_random = True
            if not (0 <= a1 < ACTIONS_PER_SLOT and mask1[a1]):
                log.debug("Slot 1 model action %d illegal — random fallback.", a1)
                a1 = random_legal_action(battle, 1)
                used_random = True

            return a0, a1, "random" if used_random else "model"

        except Exception as exc:
            log.warning("Model inference failed (%s) — using random.", exc)
            return (
                random_legal_action(battle, 0),
                random_legal_action(battle, 1),
                "random",
            )

    # ── Team chooser ──────────────────────────────────────────────────────────

    def _choose_team_order(self, battle: DoubleBattle, team: list, n: int) -> List[int]:
        """Use the team-chooser model if loaded, otherwise the first-N heuristic."""
        if self._team_chooser is None or not _TORCH_AVAILABLE:
            return _heuristic_team_order(battle)[:n]

        try:
            TEAM_STATE_DIM = len(team) * 101   # TODO: replace with real encoding
            t = torch.zeros(TEAM_STATE_DIM, dtype=torch.float32, device=self._device)

            with torch.no_grad():
                out     = self._team_chooser(t)
                indices = [int(x.item()) for x in out[:n]]

            valid = [i for i in indices if 0 <= i < len(team)]
            if len(set(valid)) == n:
                return valid

            log.warning(
                "team_chooser produced invalid/duplicate indices %s — using heuristic.", indices
            )
        except Exception as exc:
            log.warning("team_chooser inference failed (%s) — using heuristic.", exc)

        return _heuristic_team_order(battle)[:n]
