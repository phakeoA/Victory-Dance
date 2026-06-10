"""
random_player.py  —  Random-action VGC player
==============================================
RandomVGCPlayer plays fully randomly: every turn it picks a uniformly
random legal action for each active slot.  It still handles teampreview,
forceSwitch, and replay recording via VGCPlayerBase.

Use this as the opponent during early self-play, or as a benchmark
baseline to verify that trained models are actually improving.

Usage
-----
    from random_player import RandomVGCPlayer
    from poke_env import AccountConfiguration

    bot = RandomVGCPlayer(
        replay_path=Path("replay_buffer/random.jsonl"),
        account_configuration=AccountConfiguration("RandomBot", None),
        battle_format="gen9championsvgc2026regma",
        team=team_str,
    )
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from poke_env.battle import DoubleBattle

from vgc_base import VGCPlayerBase, random_legal_action, VGC_TEAM_SIZE

log = logging.getLogger(__name__)


class RandomVGCPlayer(VGCPlayerBase):
    """
    VGC player that makes every decision uniformly at random:
      • Teampreview — shuffles the roster and picks the first N slots.
      • In-battle   — picks a uniformly random legal action each turn.

    Parameters
    ----------
    replay_path : Path or None
        Where to write the JSON-lines replay buffer.
        Defaults to replay_buffer/replay.jsonl
    **kwargs
        Forwarded to poke_env.player.Player (account_configuration,
        battle_format, team, max_concurrent_battles, …)
    """

    def __init__(self, replay_path: Optional[Path] = None, **kwargs):
        super().__init__(replay_path=replay_path, **kwargs)
        log.info(
            "RandomVGCPlayer ready | replay=%s",
            replay_path or Path("replay_buffer/replay.jsonl"),
        )

    def _choose_team_order(self, battle: DoubleBattle, team: list, n: int) -> List[int]:
        """Shuffle the roster and pick n slots at random."""
        indices = list(range(len(team)))
        random.shuffle(indices)
        return indices[:n]

    def _select_actions(
        self,
        battle: DoubleBattle,
        state_vec: np.ndarray,
    ) -> Tuple[int, int, str]:
        """Pick a uniformly random legal action for each active slot."""
        a0 = random_legal_action(battle, 0)
        a1 = random_legal_action(battle, 1)
        return a0, a1, "random"
