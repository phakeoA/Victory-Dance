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

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn


def _masked_mean(h: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
    """Mean of h (B,N,H) over NON-pad rows; pad (B,N) bool True=pad. 0 if a row is all-pad."""
    valid = (~pad).unsqueeze(-1).to(h.dtype)             # (B, N, 1)
    return (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)


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
        use_self_attn: bool = False,
        use_cross_attn: bool = False,
        attn_heads: int = 4,
        use_teammate_bias: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.feat_dim = feat_dim
        self.use_self_attn = use_self_attn
        self.use_cross_attn = use_cross_attn
        self.use_teammate_bias = use_teammate_bias
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.mon_mlp = _mlp([emb_dim + feat_dim, hidden, hidden], dropout)
        # 15b-arch.1: positionless self-attention over OUR 6 mons makes the synergy TAGS interact
        # (weather/terrain setter<->abuser, spread<->immunity, reverser<->debuff) — the mean-pool +
        # independent scoring could not express "A specifically enables B". Perm-equivariant (no
        # positional encoding); created ONLY when enabled so legacy mean-pool checkpoints load with
        # NO missing/unexpected keys. NO key_padding_mask: OOV mons share PAD idx 0, so idx-based
        # masking would wrongly drop real OOV mons — VGC preview always has exactly 6, so all attend.
        if use_self_attn:
            self.self_attn = nn.MultiheadAttention(hidden, attn_heads, dropout=dropout, batch_first=True)
            self.self_ln = nn.LayerNorm(hidden)
            if use_teammate_bias:   # 15b-feat.1b: learnable weight on the Pikalytics co-occurrence prior
                self.teammate_scale = nn.Parameter(torch.tensor(1.0))
        if use_cross_attn:    # optional later ablation: our mons attend to the opponent's 6
            self.cross_attn = nn.MultiheadAttention(hidden, attn_heads, dropout=dropout, batch_first=True)
            self.cross_ln = nn.LayerNorm(hidden)
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
        our_affinity: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        our_h = self.encode_team(our_idx, our_feat)   # (B, 6, H)
        opp_h = self.encode_team(opp_idx, opp_feat)   # (B, 6, H)
        if self.use_self_attn or self.use_cross_attn:
            # Pad-safe attention (15b-arch.1 review): a PAD slot is an ALL-ZERO FEATURE row (what
            # _pack_side writes) — which, unlike idx==0, does NOT collide with OOV mons (those still
            # carry pokedex types/stats). Mask pad rows as attention KEYS and drop them from the
            # context mean so genuine mons can't attend to / be diluted by padding. (For a full 6-mon
            # roster the mask is all-False -> identical to a plain mean over the 6.)
            our_pad = our_feat.abs().sum(dim=-1) == 0   # (B, 6) True = pad
            opp_pad = opp_feat.abs().sum(dim=-1) == 0
            if self.use_self_attn:
                # 15b-feat.1b: add the Pikalytics co-occurrence prior as a learnable-scaled, additive
                # attention-score bias so attention starts focused on commonly-teamed pairs. When the
                # bias is active we fold pad-KEY masking (-inf) INTO the float attn_mask, so we never mix
                # a bool key_padding_mask with a float attn_mask (deprecated). Permutes with the roster.
                self_mask, self_kpm = None, our_pad
                if self.use_teammate_bias and our_affinity is not None:
                    bsz, n = our_affinity.shape[0], our_affinity.shape[1]
                    heads = self.self_attn.num_heads
                    bias = self.teammate_scale * our_affinity                      # (B, N, N)
                    bias = bias.masked_fill(our_pad.unsqueeze(1), float("-inf"))    # mask pad KEYS
                    self_mask = bias.unsqueeze(1).expand(bsz, heads, n, n).reshape(bsz * heads, n, n)
                    self_kpm = None
                our_h = self.self_ln(our_h + self.self_attn(
                    our_h, our_h, our_h, key_padding_mask=self_kpm, attn_mask=self_mask,
                    need_weights=False)[0])
            if self.use_cross_attn:                    # OUR mons attend to OPP 6
                our_h = self.cross_ln(our_h + self.cross_attn(
                    our_h, opp_h, opp_h, key_padding_mask=opp_pad, need_weights=False)[0])
            our_h = our_h.masked_fill(our_pad.unsqueeze(-1), 0.0)   # isolate pad rows (finite, excluded)
            our_ctx = _masked_mean(our_h, our_pad)     # (B, H)
            opp_ctx = _masked_mean(opp_h, opp_pad)
        else:
            our_ctx = our_h.mean(dim=1)                # legacy mean-pool (byte-identical when attn off)
            opp_ctx = opp_h.mean(dim=1)
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
    use_self_attn: bool = False,
    use_cross_attn: bool = False,
    attn_heads: int = 4,
    use_teammate_bias: bool = False,
) -> TeamPreviewModel:
    model = TeamPreviewModel(vocab_size, feat_dim, emb_dim, hidden, dropout,
                             use_self_attn=use_self_attn, use_cross_attn=use_cross_attn,
                             attn_heads=attn_heads, use_teammate_bias=use_teammate_bias).to(device)
    print(
        f"[TeamPreviewModel] {model.count_parameters():,} params | "
        f"vocab={vocab_size} feat_dim={feat_dim} emb={emb_dim} hidden={hidden} "
        f"self_attn={use_self_attn} cross_attn={use_cross_attn} device={device}"
    )
    return model
