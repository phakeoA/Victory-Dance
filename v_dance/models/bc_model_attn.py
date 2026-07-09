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
re-export (the model auto-sizes from get_state_dim() — see encoder_layout.py for the
live constants; do NOT hardcode them here, they rot at every layout bump):

    STATE_DIM = NUM_MON_SLOTS x POKEMON_FEATURES + GLOBAL_FEATURES  (5057 at layout v19)

    slot 0  own active a   (our_a)        slots 4-7   own bench
    slot 1  own active b   (our_b)        slots 8-11  opp bench
    slot 2  opp active a   (opp_a)
    slot 3  opp active b   (opp_b)

So we reshape the existing flat vector to (B, NUM_MON_SLOTS, POKEMON_FEATURES) IN-MODEL, run a shared
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
  state the offline encoder writes an all-zero own-active block (returns early on a
  None mon) — so that slot's per-slot head logits are then
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
        state_dim:    input width (frozen STATE_DIM from get_state_dim(); 5057 at layout v19).
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
        n_value_atoms: int = 0,
        opp_cond: bool = False,
        memory_dim: int = 0,
        mem_layers: int = 2,
        mem_heads: int = 4,
        max_mem_len: int = 64,
        n_archetypes: int = 0,
        z_dim: int = 0,
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
                "This model reshapes the flat encoder layout in-place; a layout change needs this updated."
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

        # ── Phase 1a: MATCH-MEMORY core (frame-stacked causal time-axis transformer,
        # per VGC-Bench arXiv:2506.10326 — NOT an LSTM: no hidden-state carry, no
        # BPTT; serve keeps the last n state vectors and recomputes each turn).
        # memory_dim=0 (default) builds NO new modules → state_dict keys, param
        # count and forward output stay byte-identical to the stateless model, so
        # every existing checkpoint loads unchanged (n_value_atoms precedent).
        # With memory_dim>0 every head reads an extra ``mem`` feature appended
        # LAST in its input (zeros in single-turn forward; the causal summary of
        # the match so far in forward_with_memory) — appended last so a stateless
        # checkpoint warm-starts by zero-padding the new columns
        # (init_memory_model_from_stateless) and reproduces its logits exactly.
        self.memory_dim = int(memory_dim)
        self.mem_layers = int(mem_layers)
        self.mem_heads = int(mem_heads)
        self.max_mem_len = int(max_mem_len)
        if self.memory_dim:
            # Turn summary = [masked-mean(present mon tokens) || global] → d_mem.
            self.mem_proj = nn.Linear(2 * d_model, self.memory_dim)
            self.mem_pos_emb = nn.Parameter(torch.zeros(self.max_mem_len, self.memory_dim))
            mem_layer = nn.TransformerEncoderLayer(
                d_model=self.memory_dim,
                nhead=self.mem_heads,
                dim_feedforward=ff_mult * self.memory_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.mem_attn = nn.TransformerEncoder(
                mem_layer, num_layers=self.mem_layers, enable_nested_tensor=False
            )

        # ── Phase 2 (z v1): discrete own-team ARCHETYPE conditioning.  z = OUR
        # game plan (constant per battle, from the own 6-mon roster — see
        # datatools/team_archetypes.py); memory = THIS match — deliberately
        # disjoint roles (design §1).  n_archetypes=0 (default) builds NO new
        # modules → byte-identical to the pre-z model, every checkpoint loads
        # unchanged.  With z on, every head reads a z_dim embedding appended
        # LAST (after the memory feature), so a narrower checkpoint warm-starts
        # by zero-padding the trailing columns (init_extended_model_from_ckpt)
        # and reproduces its logits exactly.  Embedding index n_archetypes is
        # the UNKNOWN archetype (unlabelled examples / missing serve lookup).
        if (n_archetypes > 0) != (z_dim > 0):
            raise ValueError(
                f"n_archetypes ({n_archetypes}) and z_dim ({z_dim}) must be "
                f"both 0 (z off) or both > 0 (z on)")
        self.n_archetypes = int(n_archetypes)
        self.z_dim = int(z_dim)
        # Explicit per-forward archetype ids win; else this per-instance default
        # (the serve player stamps it at teampreview — one team per player);
        # else UNKNOWN.  None = unset.
        self._default_archetype_id: Optional[int] = None
        if self.z_dim:
            self.z_emb = nn.Embedding(self.n_archetypes + 1, self.z_dim)

        # Heads read [mon token (d) || global context (d)] = 2*d_model
        # (+ memory_dim, then z_dim, appended LAST when those cores are on).
        head_in = 2 * d_model
        # Level B (opp-conditioning): the OUR action heads (DEFAULT_HEADS) ALSO read the DETACHED predicted
        # opp-action distribution [softmax(opp_a) || softmax(opp_b)] (one softmax per aux opp head, each
        # action_dim wide) so the policy best-responds to its own opp prediction. Requires the aux opp heads
        # to be present. opp_cond=False (default) -> the our heads keep head_in (byte-identical arch).
        self.opp_cond = bool(opp_cond)
        self._opp_head_names = tuple(h for h in self.head_names if h not in DEFAULT_HEADS)
        if self.opp_cond and not self._opp_head_names:
            raise ValueError("opp_cond=True requires aux opp heads (e.g. opp_a/opp_b) in `heads`; none present")
        opp_feat_dim = len(self._opp_head_names) * action_dim if self.opp_cond else 0
        self.heads = nn.ModuleDict(
            {name: nn.Linear(head_in + (opp_feat_dim if name in DEFAULT_HEADS else 0)
                             + self.memory_dim + self.z_dim, action_dim)
             for name in self.head_names}
        )
        self.gimmick_heads = nn.ModuleDict(
            {name: nn.Linear(head_in + self.memory_dim + self.z_dim, gimmick_dim)
             for name in self.gimmick_head_names}
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
        val_in += self.memory_dim + self.z_dim  # memory then z appended LAST (zeros single-turn / z off)
        if value_readout == "cls_query":
            self.value_query = nn.Parameter(torch.zeros(1, 1, d_model))
            self.value_attn = nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout, batch_first=True)
        self.value_head = nn.Linear(val_in, 1)
        # C51 distributional value head (opt-in): a per-atom head over the value support,
        # reading the SAME readout vector as the scalar value_head. n_value_atoms=0 (default)
        # builds NO extra module -> existing checkpoints / the scalar critic stay byte-identical.
        self.n_value_atoms = int(n_value_atoms)
        self.value_atoms_head = (nn.Linear(val_in, self.n_value_atoms)
                                 if self.n_value_atoms else None)

        self._init_weights()

    def _encode_single_turn(self, x: torch.Tensor):
        """PER-TURN encoder body: (B, state_dim) | (state_dim,) -> (enc, present, g, single).

        Factored out of ``forward`` so the C51 value-atoms head and the Phase-1a
        match-memory path (``forward_with_memory``) reuse the IDENTICAL mon-encoder
        + self-attention + global stack without duplicating it."""
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
        return enc, present, g, single

    def set_default_archetype(self, archetype_id: Optional[int]) -> None:
        """Per-instance default archetype id for forwards that pass none — the
        serve path (one team per player instance, so the id is constant).  None
        clears it (forwards fall back to the UNKNOWN embedding)."""
        if archetype_id is not None and not (0 <= int(archetype_id) <= self.n_archetypes):
            raise ValueError(
                f"archetype_id {archetype_id} out of range [0, {self.n_archetypes}] "
                f"(index {self.n_archetypes} = UNKNOWN)")
        self._default_archetype_id = None if archetype_id is None else int(archetype_id)

    def _resolve_z(self, B: int, device, archetype_id=None) -> Optional[torch.Tensor]:
        """(B, z_dim) archetype embedding, or None when z is off.  Resolution:
        explicit ``archetype_id`` (int or (B,) tensor) > the per-instance
        default (serve) > the UNKNOWN embedding."""
        if not self.z_dim:
            return None
        if archetype_id is None:
            archetype_id = (self._default_archetype_id
                            if self._default_archetype_id is not None
                            else self.n_archetypes)                     # UNKNOWN
        ids = torch.as_tensor(archetype_id, dtype=torch.long, device=device)
        if ids.dim() == 0:
            ids = ids.expand(B)
        ids = ids.clamp(0, self.n_archetypes)
        return self.z_emb(ids)                                          # (B, z_dim)

    def forward(
        self, x: torch.Tensor, archetype_id=None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
        """x: (B, state_dim) -> (actions, gimmicks, value).

        Accepts an unbatched (state_dim,) input too — the serve helpers
        (model_io.value_logit / head_logits / bc_action_indices) pass one state
        vector — matching BCPolicy's nn.Linear broadcasting (1-D in -> unbatched out)
        so AttnBCPolicy is a true drop-in.

        Memory-core note: with memory_dim>0 the per-head memory feature is ZEROS
        here (no match history in a single-turn call) — with the zero-padded
        warm-start (init_memory_model_from_stateless) the logits equal the
        stateless model's exactly.

        ``archetype_id`` (Phase-2 z): optional int or (B,) tensor of own-team
        archetype ids; omitted → the per-instance default (set_default_archetype)
        or the UNKNOWN embedding.  Ignored when z is off."""
        enc, present, g, single = self._encode_single_turn(x)
        mem = (enc.new_zeros(enc.shape[0], self.memory_dim)
               if self.memory_dim else None)
        z = self._resolve_z(enc.shape[0], enc.device, archetype_id)
        return self._heads_from(enc, present, g, mem, single, z=z)

    def forward_with_memory(
        self, x_seq: torch.Tensor, frame_padding_mask: Optional[torch.Tensor] = None,
        archetype_id=None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
        """Frame-stacked sequence forward: heads answer for the LAST frame,
        conditioned on a causal memory over the whole (in-window) match so far.

        x_seq: (B, T, state_dim) or (T, state_dim) — one replay's consecutive
            turn states in order (the caller must never cross replay
            boundaries; the 1b dataset groups frames by replay_id).  Only the
            most recent ``max_mem_len`` frames are kept.
        frame_padding_mask: optional (B, T) bool, True = PADDING frame to be
            ignored by the memory attention (for left-padded batches of
            unequal-length histories).  The LAST frame must never be padding.

        Returns the same ``(actions, gimmicks, value)`` contract as ``forward``,
        for the final frame of each sequence.
        """
        if not self.memory_dim:
            raise RuntimeError("forward_with_memory requires memory_dim>0 "
                               "(this model was built stateless)")
        single = x_seq.dim() == 2
        if single:
            x_seq = x_seq.unsqueeze(0)
        if x_seq.dim() != 3 or x_seq.shape[-1] != self.state_dim:
            raise ValueError(
                f"expected (B, T, {self.state_dim}) or (T, {self.state_dim}), "
                f"got {tuple(x_seq.shape)}")
        if x_seq.shape[1] > self.max_mem_len:               # keep the most recent window
            x_seq = x_seq[:, -self.max_mem_len:, :]
            if frame_padding_mask is not None:
                frame_padding_mask = frame_padding_mask[:, -self.max_mem_len:]
        B, T, _ = x_seq.shape

        enc, present, g, _ = self._encode_single_turn(x_seq.reshape(B * T, self.state_dim))

        # Turn-summary tokens: [masked-mean(present mon tokens) || global] -> d_mem.
        summ = torch.cat([self._masked_mean(enc, present), g], dim=-1)   # (B*T, 2d)
        # Positional indexing is RELATIVE TO THE PRESENT (flip): the LAST frame
        # always reads mem_pos_emb[0], the one before it [1], … — so training's
        # fixed-T LEFT-padded windows and serve's growing UNPADDED stacks are
        # positionally IDENTICAL (audit 2026-07-02: absolute/left-aligned
        # indexing put the supervised frame at T-1 in training but at 0,1,2,…
        # live, feeding never-trained pos rows into every serve decision).
        # Left-padding lands at the LARGEST ages — masked out anyway.
        tok = (self.mem_proj(summ.reshape(B, T, -1))
               + self.mem_pos_emb[:T].flip(0).unsqueeze(0))
        causal = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=tok.device), diagonal=1)
        if frame_padding_mask is None:
            mem_seq = self.mem_attn(tok, mask=causal)                     # (B, T, d_mem)
        else:
            # Merge causality + padding into one per-batch attn mask instead of
            # src_key_padding_mask: a LEFT-padding query would otherwise have
            # zero attendable keys → softmax NaN → 0·NaN poisons every later
            # layer.  Each padding query attends to ITSELF instead — its output
            # is meaningless but finite, and it stays key-masked for all real
            # queries, so the garbage never propagates.
            blocked = causal.unsqueeze(0) | frame_padding_mask.unsqueeze(1)  # (B, T, T)
            self_diag = (torch.eye(T, dtype=torch.bool, device=tok.device)
                         .unsqueeze(0) & frame_padding_mask.unsqueeze(-1))
            blocked = blocked & ~self_diag
            attn_mask = blocked.repeat_interleave(self.mem_heads, dim=0)     # (B·nhead, T, T)
            mem_seq = self.mem_attn(tok, mask=attn_mask)
        mem = mem_seq[:, -1, :]                                           # (B, d_mem)

        # Final frame's per-mon tokens / presence / global feed the heads.
        enc_last = enc.reshape(B, T, self.n_mon_slots, self.d_model)[:, -1]
        present_last = present.reshape(B, T, self.n_mon_slots)[:, -1]
        g_last = g.reshape(B, T, self.d_model)[:, -1]
        z = self._resolve_z(B, enc_last.device, archetype_id)
        return self._heads_from(enc_last, present_last, g_last, mem, single, z=z)

    def _heads_from(
        self,
        enc: torch.Tensor,
        present: torch.Tensor,
        g: torch.Tensor,
        mem: Optional[torch.Tensor],
        single: bool,
        z: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
        """Shared head stack over one turn's tokens (+ optional match memory
        and archetype embedding).

        Feature order per head is [mon || global || (opp_feat) || mem || z] —
        the widening features LAST so a narrower checkpoint's weights occupy
        the leading columns verbatim under the zero-pad warm-start."""
        actions: Dict[str, torch.Tensor] = {}
        # Opp-PREDICTION heads first, so the OUR heads can (Level B) condition on the predicted opp action.
        for name in self._opp_head_names:
            parts = [enc[:, HEAD_SLOT[name], :], g]
            if mem is not None:
                parts.append(mem)
            if z is not None:
                parts.append(z)
            actions[name] = self.heads[name](torch.cat(parts, dim=-1))
        # Level B: the DETACHED predicted opp-action distribution(s) fed into the OUR heads. Detached so the
        # our-policy gradient never corrupts the predictor (the opp heads learn only from the aux CE).
        opp_feat = None
        if self.opp_cond and self._opp_head_names:
            opp_feat = torch.cat([torch.softmax(actions[n], dim=-1)
                                  for n in self._opp_head_names], dim=-1).detach()
        for name in self.head_names:
            if name in self._opp_head_names:
                continue                                    # opp heads already computed above
            parts = [enc[:, HEAD_SLOT[name], :], g]
            if opp_feat is not None and name in DEFAULT_HEADS:
                parts.append(opp_feat)
            if mem is not None:
                parts.append(mem)
            if z is not None:
                parts.append(z)
            actions[name] = self.heads[name](torch.cat(parts, dim=-1))

        gimmicks: Dict[str, torch.Tensor] = {}
        for name, head in self.gimmick_heads.items():
            parts = [enc[:, HEAD_SLOT[name], :], g]
            if mem is not None:
                parts.append(mem)
            if z is not None:
                parts.append(z)
            gimmicks[name] = head(torch.cat(parts, dim=-1))

        val_feat = self._value_feature(enc, present, g)
        if mem is not None:
            val_feat = torch.cat([val_feat, mem], dim=-1)
        if z is not None:
            val_feat = torch.cat([val_feat, z], dim=-1)
        value = self.value_head(val_feat).squeeze(-1)       # (B,)

        if single:                                          # 1-D in -> unbatched out (BCPolicy parity)
            actions = {k: v.squeeze(0) for k, v in actions.items()}
            gimmicks = {k: v.squeeze(0) for k, v in gimmicks.items()}
            value = value.squeeze(0)
        return actions, gimmicks, value

    def value_atoms_logits(self, x: torch.Tensor) -> torch.Tensor:
        """C51 per-atom value logits — (B, n_value_atoms) | (n_value_atoms,). Uses the SAME
        readout vector as the scalar value head. Raises if no atoms head was built."""
        if self.value_atoms_head is None:
            raise RuntimeError(
                "value_atoms_logits: no atoms head (construct with n_value_atoms>0)")
        enc, present, g, single = self._encode_single_turn(x)
        feat = self._value_feature(enc, present, g)
        if self.memory_dim:                        # single-turn call — no match history
            feat = torch.cat([feat, feat.new_zeros(feat.shape[0], self.memory_dim)], dim=-1)
        z = self._resolve_z(feat.shape[0], feat.device)
        if z is not None:
            feat = torch.cat([feat, z], dim=-1)
        logits = self.value_atoms_head(feat)                                   # (B, n_atoms)
        return logits.squeeze(0) if single else logits

    def add_value_atoms_head(self, n_atoms: int) -> None:
        """Attach a C51 per-atom value head AFTER construction (e.g. onto a deep-copy of a scalar BC
        policy when building a distributional critic). Sized to the scalar value head's input so it
        reads the SAME value readout. Cold-initialised — the caller warm-starts it
        (init_value_atoms_from_scalar) and the critic warm-up sharpens it."""
        self.n_value_atoms = int(n_atoms)
        self.value_atoms_head = nn.Linear(self.value_head.in_features, int(n_atoms))

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
        return torch.cat([self._masked_mean(enc, present), g], dim=-1)  # (B, 2d)

    @staticmethod
    def _masked_mean(enc: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        """Masked mean over PRESENT mon tokens: (B, slots, d) -> (B, d).

        Shared by the 'mean' value readout and the memory core's turn-summary
        tokens (identical semantics: absent slots never poison the pool)."""
        pmask = present.unsqueeze(-1)                               # (B, slots, 1)
        enc_present = enc.masked_fill(~pmask, 0.0)
        denom = present.sum(dim=1, keepdim=True).clamp(min=1).to(enc.dtype)  # (B,1)
        return enc_present.sum(dim=1) / denom                       # (B, d)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        if hasattr(self, "value_query"):               # cls_query readout only
            nn.init.normal_(self.value_query, mean=0.0, std=0.02)
        if hasattr(self, "mem_pos_emb"):               # memory core only
            nn.init.normal_(self.mem_pos_emb, mean=0.0, std=0.02)
        if hasattr(self, "z_emb"):                     # Phase-2 z only — fresh init;
            # the warm-start zeroes the heads' z COLUMNS instead (exactly one
            # side non-zero, so gradients flow and init stays byte-compatible).
            nn.init.normal_(self.z_emb.weight, mean=0.0, std=0.02)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def init_extended_model_from_ckpt(
    model: AttnBCPolicy, ckpt_state: Dict[str, torch.Tensor], extra_cols: int
) -> AttnBCPolicy:
    """Warm-start a model whose head Linears grew by exactly ``extra_cols``
    TRAILING input columns (memory and/or z — both append LAST) from a
    narrower checkpoint.

    Every shared tensor is copied verbatim; each widened head Linear
    (action/gimmick/value) gets the checkpoint's weights in its LEADING
    columns and ZEROS in the new trailing columns, so a forward whose new
    features are zeros — or enter only through zeroed columns — reproduces the
    checkpoint's logits bit-exactly at init.  New-module keys the checkpoint
    lacks (mem_proj / mem_attn / mem_pos_emb / z_emb) keep their fresh init:
    their output reaches the heads through the zeroed columns, so they are
    invisible until trained.

    Returns ``model`` (mutated in place) for chaining.
    """
    if extra_cols <= 0:
        raise ValueError(f"init_extended_model_from_ckpt: extra_cols must be > 0, "
                         f"got {extra_cols} (a same-width warm start is a plain "
                         f"load_state_dict)")
    own = model.state_dict()
    unmatched: list = []
    for k, v in ckpt_state.items():
        if k not in own:
            unmatched.append(k)
            continue
        if own[k].shape == v.shape:
            own[k] = v.clone()
        elif (own[k].dim() == 2 and v.dim() == 2
              and own[k].shape[0] == v.shape[0]
              and own[k].shape[1] == v.shape[1] + extra_cols):
            w = torch.zeros_like(own[k])
            w[:, : v.shape[1]] = v
            own[k] = w
        else:
            raise ValueError(
                f"init_extended_model_from_ckpt: incompatible shape for {k}: "
                f"checkpoint {tuple(v.shape)} vs model {tuple(own[k].shape)} "
                f"(expected growth of exactly {extra_cols} trailing columns)")
    if unmatched:
        raise ValueError(
            f"init_extended_model_from_ckpt: checkpoint keys missing from the "
            f"model: {unmatched[:8]}{'…' if len(unmatched) > 8 else ''}")
    model.load_state_dict(own)
    return model


def init_memory_model_from_stateless(
    mem_model: AttnBCPolicy, ckpt_state: Dict[str, torch.Tensor]
) -> AttnBCPolicy:
    """Warm-start a memory model (memory_dim>0, z off) from a STATELESS
    checkpoint — the Phase-1b wrapper over ``init_extended_model_from_ckpt``
    (the memory columns are the only growth)."""
    if not mem_model.memory_dim:
        raise ValueError("init_memory_model_from_stateless: model has memory_dim=0")
    return init_extended_model_from_ckpt(
        mem_model, ckpt_state, extra_cols=mem_model.memory_dim + mem_model.z_dim)


def build_attn_model(
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 2,
    ff_mult: int = 2,
    dropout: float = 0.1,
    heads: Sequence[str] = DEFAULT_HEADS,
    device: str = "cpu",
    gimmick_heads: Optional[Sequence[str]] = None,
    value_readout: str = "mean",
    opp_cond: bool = False,
    memory_dim: int = 0,
    mem_layers: int = 2,
    mem_heads: int = 4,
    max_mem_len: int = 64,
    n_archetypes: int = 0,
    z_dim: int = 0,
) -> AttnBCPolicy:
    """Build an AttnBCPolicy on ``device`` and print a one-line summary.

    ``ff_mult`` MUST be forwarded (not left at the AttnBCPolicy default): the trainer
    stamps ``args.ff_mult`` into the checkpoint config and ``model_io`` rebuilds with it,
    so dropping it here would build ``ff_mult=2`` while the config claims another value →
    a silent reload SHAPE MISMATCH for any ``--ff-mult != 2``."""
    model = AttnBCPolicy(
        state_dim=get_state_dim(),
        action_dim=get_action_dim(),
        gimmick_dim=get_gimmick_dim(),
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        ff_mult=ff_mult,
        dropout=dropout,
        heads=heads,
        gimmick_heads=gimmick_heads,
        value_readout=value_readout,
        opp_cond=opp_cond,
        memory_dim=memory_dim,
        mem_layers=mem_layers,
        mem_heads=mem_heads,
        max_mem_len=max_mem_len,
        n_archetypes=n_archetypes,
        z_dim=z_dim,
    ).to(device)
    print(
        f"[AttnBCPolicy] {model.count_parameters():,} params | "
        f"state_dim={model.state_dim} action_dim={model.action_dim} "
        f"gimmick_dim={model.gimmick_dim} d_model={d_model} n_layers={n_layers} "
        f"ff_mult={ff_mult} value_readout={value_readout} heads={model.head_names} "
        f"gimmick_heads={model.gimmick_head_names} memory_dim={memory_dim} "
        f"n_archetypes={n_archetypes} z_dim={z_dim} device={device}"
    )
    return model
