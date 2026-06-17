"""
Two-head Behaviour-Cloning policy for Victory-Dance (VGC Reg M-A).

A plain MLP trunk over the frozen 938-dim state vector, feeding two
INDEPENDENT 16-way action heads — one per active slot (our_a, our_b).  This is
the BC v0 baseline; an auxiliary opponent head (predicting opp actions) is
A/B'd against it later by passing extra head names.

    logits = model(x)            # dict {"our_a": (B,16), "our_b": (B,16)}

The heads are independent Linear layers on a shared trunk, matching how the
policy is used at inference: one board state → both slots' action logits at
once.  Action legality masking is applied by the trainer (train_bc.py), not the
model, so the raw logits stay reusable for MCTS / value bootstrapping later.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

# ── Bootstrap: locate data/scripts by walking up (folder-depth independent) ───
def _find_scripts_dir() -> Path:
    for parent in Path(__file__).resolve().parents:
        cand = parent / "data" / "scripts"
        if cand.is_dir():
            return cand
    raise RuntimeError(f"could not locate data/scripts above {__file__}")

_SCRIPTS_DIR = _find_scripts_dir()
from v_dance.encoders.state_encoder import get_state_dim, get_action_dim, get_gimmick_dim  # noqa: E402

DEFAULT_HEADS: Tuple[str, str] = ("our_a", "our_b")


class BCPolicy(nn.Module):
    """
    MLP trunk -> one independent action head per slot + one independent gimmick
    (mega) head per slot.

    Args:
        state_dim:   input width (frozen STATE_DIM, 1806 as of gap #5 item/ability)
        action_dim:  move/switch logits per head (default: frozen ACTION_DIM == 16)
        gimmick_dim: gimmick logits per head (default: GIMMICK_DIM == 2, {none, mega})
        hidden_dims: trunk layer widths (default: (512, 256))
        dropout:     dropout after each hidden layer (default: 0.1)
        heads:       head names (default: our_a, our_b). Pass a longer tuple to
                     add the auxiliary opponent heads for the A/B experiment.

    The gimmick heads are SEPARATE from the action heads (orthogonal decision: a
    mega is a checkbox on the chosen move, not a competing action), sharing the
    same trunk.  forward returns (actions, gimmicks).
    """

    def __init__(
        self,
        state_dim: int = get_state_dim(),
        action_dim: int = get_action_dim(),
        hidden_dims: Sequence[int] = (512, 256),
        dropout: float = 0.1,
        heads: Sequence[str] = DEFAULT_HEADS,
        gimmick_dim: int = get_gimmick_dim(),
        gimmick_heads: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gimmick_dim = gimmick_dim
        self.head_names: Tuple[str, ...] = tuple(heads)
        # Gimmick heads default to the OWN active slots only (our_a, our_b) — the
        # only slots whose gimmick is a real decision.  Auxiliary opponent heads
        # (opp_a/opp_b, task #9) predict ACTIONS only, no opp gimmick head.
        if gimmick_heads is None:
            gimmick_heads = tuple(h for h in self.head_names if h in DEFAULT_HEADS)
        self.gimmick_head_names: Tuple[str, ...] = tuple(gimmick_heads)

        layers: list[nn.Module] = []
        prev = state_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.trunk_out = prev

        self.heads = nn.ModuleDict(
            {name: nn.Linear(prev, action_dim) for name in self.head_names}
        )
        # Parallel per-slot gimmick (mega) heads — same trunk, separate logits.
        self.gimmick_heads = nn.ModuleDict(
            {name: nn.Linear(prev, gimmick_dim) for name in self.gimmick_head_names}
        )
        # Scalar position-VALUE head (win probability): ONE number per board,
        # trained on the game OUTCOME (BCE over sigmoid).  Shares the trunk; the
        # policy/gimmick heads are unchanged.  This is the strength unlock — it
        # enables a 1-ply value lookahead / search on top of the BC prior (#2).
        self.value_head = nn.Linear(prev, 1)

        self._init_weights()

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
        """x: (B, state_dim) -> (actions, gimmicks, value).

        ``actions``  = {head_name: (B, action_dim)} raw move/switch logits.
        ``gimmicks`` = {head_name: (B, gimmick_dim)} raw gimmick logits.
        ``value``    = (B,) raw win-LOGIT (apply sigmoid for win probability).
        """
        z = self.trunk(x)
        actions = {name: head(z) for name, head in self.heads.items()}
        gimmicks = {name: head(z) for name, head in self.gimmick_heads.items()}
        value = self.value_head(z).squeeze(-1)          # (B,) raw win-logit
        return actions, gimmicks, value

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(
    hidden_dims: Sequence[int] = (512, 256),
    dropout: float = 0.1,
    heads: Sequence[str] = DEFAULT_HEADS,
    device: str = "cpu",
    gimmick_heads: Optional[Sequence[str]] = None,
) -> BCPolicy:
    """Build a BCPolicy on ``device`` and print a one-line summary."""
    model = BCPolicy(
        state_dim=get_state_dim(),
        action_dim=get_action_dim(),
        hidden_dims=hidden_dims,
        dropout=dropout,
        heads=heads,
        gimmick_heads=gimmick_heads,
    ).to(device)
    print(
        f"[BCPolicy] {model.count_parameters():,} params | "
        f"state_dim={model.state_dim} action_dim={model.action_dim} "
        f"gimmick_dim={model.gimmick_dim} "
        f"heads={model.head_names} gimmick_heads={model.gimmick_head_names} device={device}"
    )
    return model
