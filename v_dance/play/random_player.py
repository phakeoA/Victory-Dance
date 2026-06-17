"""
local_battle/random_player.py  —  Random-action VGC player (gap-#6 splice)
==========================================================================
Identical to the repo-root ``random_player.py`` RandomVGCPlayer, except it
inherits the gap-#6 splicing base.  Actions are still uniformly random, but the
state vector it records each turn now carries the vod_parser-reconstructed
opponent side — useful as a control that exercises the splice end-to-end
(and as a self-play opponent for the model player).

    from local_battle.random_player import RandomVGCPlayer
"""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# ── Path bootstrap (local_battle FIRST; see live_vgc_base for rationale) ──────
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
from poke_env.battle import DoubleBattle

from v_dance.play.live_vgc_base import SplicingVGCPlayerBase as VGCPlayerBase
from v_dance.play.vgc_base import random_legal_action, VGC_TEAM_SIZE

log = logging.getLogger(__name__)


class RandomVGCPlayer(VGCPlayerBase):
    """VGC player that makes every decision uniformly at random, with the
    gap-#6 opponent splice inherited from SplicingVGCPlayerBase."""

    def __init__(self, replay_path: Optional[Path] = None, **kwargs):
        super().__init__(replay_path=replay_path, **kwargs)
        log.info(
            "RandomVGCPlayer ready (gap-#6 splice) | replay=%s",
            replay_path or Path("artifacts/replay_buffer/replay.jsonl"),
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
