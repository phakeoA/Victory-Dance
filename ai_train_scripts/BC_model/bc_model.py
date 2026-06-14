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
from typing import Dict, Sequence, Tuple

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
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from state_encoder import get_state_dim, get_action_dim  # noqa: E402

DEFAULT_HEADS: Tuple[str, str] = ("our_a", "our_b")


class BCPolicy(nn.Module):
    """
    MLP trunk -> one independent linear head per slot.

    Args:
        state_dim:   input width (default: frozen STATE_DIM == 938)
        action_dim:  logits per head (default: frozen ACTION_DIM == 16)
        hidden_dims: trunk layer widths (default: (512, 256))
        dropout:     dropout after each hidden layer (default: 0.1)
        heads:       head names (default: our_a, our_b). Pass a longer tuple to
                     add the auxiliary opponent heads for the A/B experiment.
    """

    def __init__(
        self,
        state_dim: int = get_state_dim(),
        action_dim: int = get_action_dim(),
        hidden_dims: Sequence[int] = (512, 256),
        dropout: float = 0.1,
        heads: Sequence[str] = DEFAULT_HEADS,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.head_names: Tuple[str, ...] = tuple(heads)

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

        self._init_weights()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """x: (B, state_dim) -> {head_name: (B, action_dim) raw logits}."""
        z = self.trunk(x)
        return {name: head(z) for name, head in self.heads.items()}

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
) -> BCPolicy:
    """Build a BCPolicy on ``device`` and print a one-line summary."""
    model = BCPolicy(
        state_dim=get_state_dim(),
        action_dim=get_action_dim(),
        hidden_dims=hidden_dims,
        dropout=dropout,
        heads=heads,
    ).to(device)
    print(
        f"[BCPolicy] {model.count_parameters():,} params | "
        f"state_dim={model.state_dim} action_dim={model.action_dim} "
        f"hidden={tuple(hidden_dims)} heads={model.head_names} device={device}"
    )
    return model
