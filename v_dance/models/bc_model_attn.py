"""
Per-mon SET-ATTENTION Behaviour-Cloning policy for Victory-Dance (VGC Reg M-A).

This is the candidate battle-net architecture (#23) — an alternative to the flat
``BCPolicy`` trunk in :mod:`v_dance.models.bc_model`.  It is a DROP-IN: same
``forward`` contract ``(actions, gimmicks, value)`` and the same head names, so
the trainer / actor-critic can swap it in without touching their call sites.  It
is NOT wired into training yet — it is a gated candidate (needs a fresh BC
pretrain + a successor self-play run; see the to-do list).

WHY this exists
---------------
``BCPolicy`` consumes the frozen STATE_DIM vector as one unstructured blob: a
512-wide Linear sees all STATE_DIM numbers at once, so the mon in bench-slot 3 and the
mon in bench-slot 4 are processed by ENTIRELY SEPARATE weights, and any notion of
"these twelve things are all Pokémon, compare them" must be re-learned from
scratch for every slot.  The CS230 paper (Tse, 2022) and our own 15b TP redesign
both point at the same structural prior: process each Pokémon with a SHARED
per-mon encoder, then let the mons INTERACT.

The frozen layout makes this a MODEL-ONLY change — no encoder change, no data
re-export (numbers below = current v18; the model auto-sizes from get_state_dim()):

    STATE_DIM = 12 mon-slots x POKEMON_FEATURES (405) + GLOBAL_FEATURES (101) = 4961

    slot 0  own active a   (our_a)        slots 4-7   own bench
    slot 1  own active b   (our_b)        slots 8-11  opp bench
    slot 2  opp active a   (opp_a)
    slot 3  opp active b   (opp_b)

So we reshape the existing flat vector to (B, 12, 149) IN-MODEL, run a shared
per-mon encoder + learned per-slot positional embedding + a self-attention block
(the mons attend to one another), fuse with an encoded global block, and read the
SAME heads off the relevant tokens.

DESIGN NOTES
------------
* ONE shared per-mon encoder for all 12 mons (not the paper's split own/opp
  "Model A"/"Model B").  Our encoding already carries observability via the
  ``*_known`` flags, and a learned per-slot positional embedding tells the model
  which side/role each token is, so a single encoder + attention is both more
  parameter-efficient and lets own<->opp interactions flow through attention.
* The 12 slots are SEMANTICALLY ORDERED (the encoder writes own-active first,
  bench deterministically seen-alive -> fainted -> unseen), and the policy/gimmick
  heads are PER-SLOT (our_a reads token 0, our_b token 1).  So this is NOT meant
  to be permutation-invariant over slots — the learned positional embedding is
  correct, and the *attention* (not invariance) is what buys the synergy /
  threat-assessment inductive bias.
* PAD-SAFE: empty slots are all-zero in the encoding (no mon).  Absent mons are
  masked out of attention (``key_padding_mask``) and excluded from the value
  pool.  Own active slots CAN be empty too — in a forced-replacement / post-faint
  state the encoder writes an all-zero own-active block (state_encoder.py:789-790
  returns early on a None mon) — so that slot's per-slot head logits are then
  meaningless but finite; downstream LEGALITY MASKING is the real authority.  A
  guard covers the degenerate fully-empty row so attention never sees an
  all-masked query.
* Smaller than ``BCPolicy`` at the default width, so MCTS / self-play inference
  stays fast.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from v_dance.encoders.state_encoder import (
    get_state_dim,
    get_action_dim,
    get_gimmick_dim,
    POKEMON_FEATURES,
    # v9 (B1-mechanics): identity-embedding vocab sizes + within-row index offsets
    ABILITY_VOCAB_SIZE, MOVE_VOCAB_SIZE, ITEM_VOCAB_SIZE,
    ABILITY_ID_REL, ITEM_ID_REL, ABILITY_KNOWN_REL, MOVE_ID_RELS, NUM_MOVES,
    ACTIVE_SLOTS,
    BENCH_SLOTS,
    OPP_BENCH_SLOTS,
    GLOBAL_FEATURES,
)

DEFAULT_HEADS: Tuple[str, str] = ("our_a", "our_b")

NUM_MON_SLOTS: int = ACTIVE_SLOTS + BENCH_SLOTS + OPP_BENCH_SLOTS  # 12

# Which mon-token each head reads.  Active slots only — these are the slots whose
# move/switch (and, for own, gimmick) is an actual decision.  Auxiliary opponent
# action heads (task #9 A/B) read the opp active tokens.
HEAD_SLOT: Dict[str, int] = {"our_a": 0, "our_b": 1, "opp_a": 2, "opp_b": 3}


class AttnBCPolicy(nn.Module):
    """
    Shared per-mon encoder -> self-attention over the 12 mon tokens -> per-slot
    action + gimmick heads + a pooled value head.

    Args:
        state_dim:    input width (frozen STATE_DIM == 4961 at layout v18).
        action_dim:   move/switch logits per head (frozen ACTION_DIM == 16).
        gimmick_dim:  gimmick logits per head (GIMMICK_DIM == 3, {none, mega, tera}; v11 Phase D).
        d_model:      per-mon token width (default 128).
        n_heads:      attention heads (default 4).
        n_layers:     stacked self-attention layers (default 2).
        ff_mult:      feed-forward expansion inside each attn layer (default 2).
        dropout:      dropout in encoder + attention (default 0.1).
        heads:        action head names (default our_a, our_b).  Pass a longer
                      tuple to add the auxiliary opponent heads.
        gimmick_heads:gimmick head names (default = OWN active heads only).

    forward(x) -> (actions, gimmicks, value), byte-compatible with BCPolicy:
        actions  = {head_name: (B, action_dim)}   raw move/switch logits
        gimmicks = {head_name: (B, gimmick_dim)}   raw gimmick logits
        value    = (B,)                            raw win-LOGIT (sigmoid -> prob)
    """

    def __init__(
        self,
        state_dim: int = get_state_dim(),
        action_dim: int = get_action_dim(),
        gimmick_dim: int = get_gimmick_dim(),
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_mult: int = 2,
        dropout: float = 0.1,
        heads: Sequence[str] = DEFAULT_HEADS,
        gimmick_heads: Optional[Sequence[str]] = None,
        value_readout: str = "mean",
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gimmick_dim = gimmick_dim
        self.d_model = d_model
        # 23-load: store the attn hyperparams as plain attributes so the checkpoint config
        # stamp and the model_io loader stay in lockstep WITHOUT fragile module introspection.
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.ff_mult = ff_mult
        self.dropout = dropout
        self.n_mon_slots = NUM_MON_SLOTS
        self.mon_features = POKEMON_FEATURES
        self.global_features = GLOBAL_FEATURES
        if value_readout not in ("mean", "concat_active", "cls_query"):
            raise ValueError(f"unknown value_readout {value_readout!r}; "
                             "expected 'mean' | 'concat_active' | 'cls_query'")
        self.value_readout = value_readout

        expected = self.n_mon_slots * self.mon_features + self.global_features
        if state_dim != expected:
            raise ValueError(
                f"AttnBCPolicy expects state_dim == {expected} "
                f"({self.n_mon_slots}x{self.mon_features} + {self.global_features}), got {state_dim}. "
                "This model reshapes the flat v4 layout in-place; a layout change needs this updated."
            )

        self.head_names: Tuple[str, ...] = tuple(heads)
        for name in self.head_names:
            if name not in HEAD_SLOT:
                raise ValueError(f"unknown head {name!r}; known: {sorted(HEAD_SLOT)}")
        if gimmick_heads is None:
            gimmick_heads = tuple(h for h in self.head_names if h in DEFAULT_HEADS)
        self.gimmick_head_names: Tuple[str, ...] = tuple(gimmick_heads)

        # v9 (B1-mechanics): per-mon ability/item identity + per-move identity embeddings. The encoder
        # writes raw vocab INDICES into the flat row; the model extracts them (forward) and looks them up
        # here, so identity is a LEARNED vector — not a raw ordinal Linear input. dims << d_model.
        self.ability_emb_dim, self.item_emb_dim, self.move_emb_dim = 24, 16, 16
        self.ability_emb = nn.Embedding(ABILITY_VOCAB_SIZE, self.ability_emb_dim, padding_idx=0)
        self.item_emb = nn.Embedding(ITEM_VOCAB_SIZE, self.item_emb_dim, padding_idx=0)
        self.move_emb = nn.Embedding(MOVE_VOCAB_SIZE, self.move_emb_dim, padding_idx=0)
        _emb_extra = self.ability_emb_dim + self.item_emb_dim + NUM_MOVES * self.move_emb_dim

        # Shared per-mon encoder (applied identically to all 12 slots). Input = the flat per-mon row
        # (identity-index columns zeroed in forward) concatenated with the looked-up identity embeddings.
        self.mon_enc = nn.Sequential(
            nn.Linear(self.mon_features + _emb_extra, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        # Learned per-slot positional/role embedding (12 semantic slots).
        self.pos_emb = nn.Parameter(torch.zeros(self.n_mon_slots, d_model))

        # Self-attention block: the mons attend to one another.  norm_first for
        # training stability; batch_first for (B, slots, d) tensors.
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # enable_nested_tensor=False: the nested-tensor fast path is disabled
        # anyway under norm_first=True, and setting it explicitly silences the
        # PyTorch UserWarning.
        self.attn = nn.TransformerEncoder(
            enc_layer, num_layers=n_layers, enable_nested_tensor=False
        )

        # Global (field) encoder.
        self.glob_enc = nn.Sequential(
            nn.Linear(self.global_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Heads read [mon token (d) || global context (d)] = 2*d_model.
        head_in = 2 * d_model
        self.heads = nn.ModuleDict(
            {name: nn.Linear(head_in, action_dim) for name in self.head_names}
        )
        self.gimmick_heads = nn.ModuleDict(
            {name: nn.Linear(head_in, gimmick_dim) for name in self.gimmick_head_names}
        )
        # 23-valhead: the value readout is a MEASURED choice (the masked-mean default mixes
        # own/opp/fainted/unseen token roles + averages away the per-slot pos_emb; the RL critic
        # warm-starts off it). 'concat_active' reads the two own-active tokens; 'cls_query' lets a
        # learned query attend over present tokens. Default 'mean' creates NO extra modules ->
        # legacy checkpoints + the reviewed net stay byte-identical.
        if value_readout == "concat_active":
            val_in = 3 * d_model                       # own_a token || own_b token || global
        else:
            val_in = 2 * d_model                       # readout vector || global
        if value_readout == "cls_query":
            self.value_query = nn.Parameter(torch.zeros(1, 1, d_model))
            self.value_attn = nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout, batch_first=True)
        self.value_head = nn.Linear(val_in, 1)

        self._init_weights()

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
        """x: (B, state_dim) -> (actions, gimmicks, value).

        Accepts an unbatched (state_dim,) input too — the serve helpers
        (model_io.value_logit / head_logits / bc_action_indices) pass one state
        vector — matching BCPolicy's nn.Linear broadcasting (1-D in -> unbatched out)
        so AttnBCPolicy is a true drop-in."""
        single = x.dim() == 1
        if single:
            x = x.unsqueeze(0)
        if x.dim() != 2 or x.shape[1] != self.state_dim:
            raise ValueError(
                f"expected (B, {self.state_dim}) or ({self.state_dim},), got {tuple(x.shape)}")
        B = x.shape[0]
        n_mon_feats = self.n_mon_slots * self.mon_features

        mons = x[:, :n_mon_feats].reshape(B, self.n_mon_slots, self.mon_features)
        glob = x[:, n_mon_feats:]                                   # (B, global)

        # Presence: an all-zero mon block == an empty slot.  abs-sum>0 is the
        # robust UNION over heterogeneous presence flags (actives carry
        # is_active=1; seen mons carry hp_frac/is_revealed; unseen-alive opp
        # stubs carry hp_frac=1) — no single feature is set for every role, so we
        # test "any nonzero".  Holds exactly on the frozen v4 layout.
        present = mons.abs().sum(dim=-1) > 0                         # (B, slots) bool — BEFORE zeroing indices

        # v9: extract the identity-INDEX columns, look them up (ability conf-scaled by ability_known), and
        # zero them out of the raw row so the large ordinal values never reach the Linear.
        ab_idx = mons[:, :, ABILITY_ID_REL].round().long().clamp_(0, self.ability_emb.num_embeddings - 1)
        it_idx = mons[:, :, ITEM_ID_REL].round().long().clamp_(0, self.item_emb.num_embeddings - 1)
        ab_known = mons[:, :, ABILITY_KNOWN_REL:ABILITY_KNOWN_REL + 1]               # (B, slots, 1)
        mv_idx = torch.stack([mons[:, :, r] for r in MOVE_ID_RELS], dim=-1) \
            .round().long().clamp_(0, self.move_emb.num_embeddings - 1)              # (B, slots, NUM_MOVES)
        ab_e = self.ability_emb(ab_idx) * ab_known                                   # (B, slots, d_a)
        it_e = self.item_emb(it_idx)                                                # (B, slots, d_i)
        mv_e = self.move_emb(mv_idx).reshape(B, self.n_mon_slots, -1)               # (B, slots, NUM_MOVES*d_m)

        mons = mons.clone()
        mons[:, :, ABILITY_ID_REL] = 0.0
        mons[:, :, ITEM_ID_REL] = 0.0
        for _r in MOVE_ID_RELS:
            mons[:, :, _r] = 0.0
        mon_in = torch.cat([mons, ab_e, it_e, mv_e], dim=-1)        # (B, slots, mon_features + emb_extra)

        tok = self.mon_enc(mon_in) + self.pos_emb.unsqueeze(0)      # (B, slots, d)

        # Mask absent mons as KEYS.  Own actives are USUALLY present, but a
        # forced-replacement state can leave an active slot empty, so guard the
        # degenerate fully-empty row to avoid an all-masked attention query.
        key_padding_mask = ~present                                 # True == ignore
        all_absent = key_padding_mask.all(dim=1)
        if bool(all_absent.any()):
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_absent] = False

        enc = self.attn(tok, src_key_padding_mask=key_padding_mask)  # (B, slots, d)

        g = self.glob_enc(glob)                                     # (B, d)

        actions: Dict[str, torch.Tensor] = {}
        for name, head in self.heads.items():
            ctx = torch.cat([enc[:, HEAD_SLOT[name], :], g], dim=-1)
            actions[name] = head(ctx)

        gimmicks: Dict[str, torch.Tensor] = {}
        for name, head in self.gimmick_heads.items():
            ctx = torch.cat([enc[:, HEAD_SLOT[name], :], g], dim=-1)
            gimmicks[name] = head(ctx)

        value = self.value_head(self._value_feature(enc, present, g)).squeeze(-1)  # (B,)

        if single:                                          # 1-D in -> unbatched out (BCPolicy parity)
            actions = {k: v.squeeze(0) for k, v in actions.items()}
            gimmicks = {k: v.squeeze(0) for k, v in gimmicks.items()}
            value = value.squeeze(0)
        return actions, gimmicks, value

    def _value_feature(
        self, enc: torch.Tensor, present: torch.Tensor, g: torch.Tensor
    ) -> torch.Tensor:
        """Readout vector fed to the value head, per ``value_readout`` (fused with global g)."""
        if self.value_readout == "concat_active":
            # the two own-active tokens carry pos_emb identity the mean averages away.
            return torch.cat([enc[:, 0, :], enc[:, 1, :], g], dim=-1)
        if self.value_readout == "cls_query":
            kpm = ~present                                          # True == ignore
            all_absent = kpm.all(dim=1)
            if bool(all_absent.any()):                             # guard all-masked query
                kpm = kpm.clone()
                kpm[all_absent] = False
            q = self.value_query.expand(enc.shape[0], -1, -1)      # (B, 1, d)
            out, _ = self.value_attn(q, enc, enc, key_padding_mask=kpm, need_weights=False)
            return torch.cat([out[:, 0, :], g], dim=-1)            # (B, 2d)
        # 'mean' (default): masked mean over PRESENT tokens (masked_fill so absent/NaN tokens
        # never poison the pool), fused with the global context. Byte-identical to the original.
        pmask = present.unsqueeze(-1)                               # (B, slots, 1)
        enc_present = enc.masked_fill(~pmask, 0.0)
        denom = present.sum(dim=1, keepdim=True).clamp(min=1).to(enc.dtype)  # (B,1)
        pooled = enc_present.sum(dim=1) / denom                     # (B, d)
        return torch.cat([pooled, g], dim=-1)                       # (B, 2d)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        if hasattr(self, "value_query"):               # cls_query readout only
            nn.init.normal_(self.value_query, mean=0.0, std=0.02)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_attn_model(
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 2,
    dropout: float = 0.1,
    heads: Sequence[str] = DEFAULT_HEADS,
    device: str = "cpu",
    gimmick_heads: Optional[Sequence[str]] = None,
    value_readout: str = "mean",
) -> AttnBCPolicy:
    """Build an AttnBCPolicy on ``device`` and print a one-line summary."""
    model = AttnBCPolicy(
        state_dim=get_state_dim(),
        action_dim=get_action_dim(),
        gimmick_dim=get_gimmick_dim(),
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
        heads=heads,
        gimmick_heads=gimmick_heads,
        value_readout=value_readout,
    ).to(device)
    print(
        f"[AttnBCPolicy] {model.count_parameters():,} params | "
        f"state_dim={model.state_dim} action_dim={model.action_dim} "
        f"gimmick_dim={model.gimmick_dim} d_model={d_model} n_layers={n_layers} "
        f"value_readout={value_readout} heads={model.head_names} "
        f"gimmick_heads={model.gimmick_head_names} device={device}"
    )
    return model
