"""
Team-preview ("bring") model for Victory-Dance (VGC Reg M-A).

A matchup-aware, permutation-equivariant scorer:

  1. Each Pokémon (ours and the opponent's) becomes a vector from a learned
     species embedding concatenated with its dex features, passed through a
     shared per-mon MLP.
  2. Symmetric context = mean-pool of our 6 mon vectors and of the opponent's
     6 — order-invariant, so the bring decision sees the WHOLE matchup.
  3. Each of our 6 mons is scored from (its own vector | our context | opp
     context) into two logits: "bring this mon" and "lead with this mon".

Because the per-mon scorer shares weights and the context is a mean, the bring
/ lead logits are exactly permutation-equivariant over our roster order — the
right inductive bias for a set-selection problem.

    bring_logits, lead_logits = model(our_idx, opp_idx, our_feat, opp_feat)
        # each (B, 6) — take top-4 / top-2 (the trainer enforces the counts).
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn


def _mlp(sizes: Sequence[int], dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:           # no activation after the last linear
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class TeamPreviewModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        feat_dim: int,
        emb_dim: int = 32,
        hidden: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.feat_dim = feat_dim
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.mon_mlp = _mlp([emb_dim + feat_dim, hidden, hidden], dropout)
        # per-mon score sees: own vector | our context | opp context
        self.score_mlp = _mlp([hidden * 3, hidden, 2], dropout)
        self._init_weights()

    def encode_team(self, idx: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """(B, 6) ids + (B, 6, F) feats -> (B, 6, hidden) per-mon vectors."""
        emb = self.emb(idx)                       # (B, 6, emb)
        x = torch.cat([emb, feat], dim=-1)        # (B, 6, emb+F)
        return self.mon_mlp(x)                     # (B, 6, hidden)

    def forward(
        self,
        our_idx: torch.Tensor,
        opp_idx: torch.Tensor,
        our_feat: torch.Tensor,
        opp_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        our_h = self.encode_team(our_idx, our_feat)   # (B, 6, H)
        opp_h = self.encode_team(opp_idx, opp_feat)   # (B, 6, H)
        our_ctx = our_h.mean(dim=1)                    # (B, H)
        opp_ctx = opp_h.mean(dim=1)                    # (B, H)
        ctx = torch.cat([our_ctx, opp_ctx], dim=-1)    # (B, 2H)
        ctx_exp = ctx.unsqueeze(1).expand(-1, our_h.shape[1], -1)  # (B, 6, 2H)
        per = torch.cat([our_h, ctx_exp], dim=-1)      # (B, 6, 3H)
        logits = self.score_mlp(per)                   # (B, 6, 2)
        return logits[..., 0], logits[..., 1]          # bring_logits, lead_logits

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.emb.weight, std=0.1)
        with torch.no_grad():
            self.emb.weight[0].zero_()             # padding row

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(
    vocab_size: int,
    feat_dim: int,
    emb_dim: int = 32,
    hidden: int = 128,
    dropout: float = 0.1,
    device: str = "cpu",
) -> TeamPreviewModel:
    model = TeamPreviewModel(vocab_size, feat_dim, emb_dim, hidden, dropout).to(device)
    print(
        f"[TeamPreviewModel] {model.count_parameters():,} params | "
        f"vocab={vocab_size} feat_dim={feat_dim} emb={emb_dim} hidden={hidden} "
        f"device={device}"
    )
    return model
