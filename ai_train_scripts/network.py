"""
AlphaZero-style neural network for VGC Reg M-A.

Architecture:
  - Shared residual tower (processes the flat state vector)
  - Policy head: outputs logits over ACTION_DIM per slot
  - Value head: outputs scalar win probability in (-1, 1)

The network receives the encoded battle state and returns:
  - policy_logits: (batch, ACTION_DIM)  — softmax gives action priors for MCTS
  - value: (batch, 1)                   — tanh output, +1 = win, -1 = loss

Design notes:
  - Fully connected residual blocks (no CNN; state is a flat vector, not a grid)
  - Batch norm for training stability
  - Designed to be small enough to evaluate thousands of times per second
    during MCTS on a single GPU
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# state_encoder lives at data/scripts/ — add it to sys.path (same pattern as the
# sibling BC_model/teamPreview_model scripts).  The old `envs.state_encoder`
# import path never existed in this project.
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


STATE_DIM = get_state_dim()
ACTION_DIM = get_action_dim()


class ResidualBlock(nn.Module):
    """
    A fully-connected residual block:
        x → Linear → BN → ReLU → Linear → BN → (+x) → ReLU
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        return F.relu(out + residual)


class VGCNet(nn.Module):
    """
    Dual-head network for VGC AlphaZero.

    Args:
        state_dim:   Input state dimension (default: STATE_DIM from encoder)
        action_dim:  Policy output dimension per slot (default: ACTION_DIM)
        hidden_dim:  Width of the residual tower (default: 256)
        num_blocks:  Number of residual blocks (default: 6)
        dropout:     Dropout rate in residual blocks (default: 0.1)
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = 256,
        num_blocks: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        # Residual tower (shared body)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)
        ])

        # Policy head — outputs logits for slot 0 and slot 1 separately
        # (MCTS will handle joint action selection)
        policy_hidden = hidden_dim // 2
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, policy_hidden),
            nn.ReLU(),
            nn.Linear(policy_hidden, action_dim),  # raw logits
        )

        # Value head — outputs scalar in (-1, 1)
        value_hidden = hidden_dim // 2
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, value_hidden),
            nn.ReLU(),
            nn.Linear(value_hidden, 1),
            nn.Tanh(),
        )

        self._init_weights()

    def forward(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: (batch, state_dim) float tensor

        Returns:
            policy_logits: (batch, action_dim) — apply softmax for MCTS priors
            value:         (batch, 1)          — tanh, +1=win, -1=loss
        """
        x = self.input_proj(state)
        for block in self.res_blocks:
            x = block(x)
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value

    def predict(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference-mode forward pass. Returns softmax policy and value.
        Handles single states (1D input) by adding batch dim.
        """
        single = state.dim() == 1
        if single:
            state = state.unsqueeze(0)

        self.eval()
        with torch.no_grad():
            logits, value = self.forward(state)
            policy = F.softmax(logits, dim=-1)

        if single:
            return policy.squeeze(0), value.squeeze(0)
        return policy, value

    def _init_weights(self):
        """Xavier initialization for faster convergence from scratch."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Loss computation ───────────────────────────────────────────────────────────

class AlphaZeroLoss(nn.Module):
    """
    AlphaZero loss = value MSE + policy cross-entropy + L2 regularization.

        L = (z - v)^2 - π^T log(p) + λ ||θ||^2

    where:
        z = actual game outcome (+1 win, -1 loss, 0 draw)
        v = network value prediction
        π = MCTS visit count distribution (target policy)
        p = network policy (softmax of logits)
    """

    def __init__(self, l2_coef: float = 1e-4):
        super().__init__()
        self.l2_coef = l2_coef

    def forward(
        self,
        policy_logits: torch.Tensor,  # (batch, action_dim)
        value: torch.Tensor,           # (batch, 1)
        target_policy: torch.Tensor,   # (batch, action_dim)  — MCTS visit counts, normalized
        target_value: torch.Tensor,    # (batch, 1)           — game outcome
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            total_loss, value_loss, policy_loss
        """
        # Value loss: MSE
        value_loss = F.mse_loss(value, target_value)

        # Policy loss: cross-entropy with MCTS target distribution
        # Use log_softmax for numerical stability
        log_probs = F.log_softmax(policy_logits, dim=-1)
        policy_loss = -(target_policy * log_probs).sum(dim=-1).mean()

        total_loss = value_loss + policy_loss
        return total_loss, value_loss, policy_loss


# ── Convenience factory ────────────────────────────────────────────────────────

def build_network(
    hidden_dim: int = 256,
    num_blocks: int = 6,
    device: str = "cpu",
) -> VGCNet:
    """Build and return a VGCNet on the specified device."""
    net = VGCNet(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
    ).to(device)
    print(f"[VGCNet] {net.count_parameters():,} trainable parameters")
    print(f"[VGCNet] state_dim={STATE_DIM}, action_dim={ACTION_DIM}, device={device}")
    return net
