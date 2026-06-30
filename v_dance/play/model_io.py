"""
local_battle/model_io.py  —  load trained checkpoints + decode them for serving
================================================================================
The training scripts save checkpoints as DICTS — ``{"model_state", "config",
...}`` — not pickled ``nn.Module`` objects.  The live player therefore cannot
``torch.load(path)`` and call the result; it must reconstruct the architecture
from ``config`` and ``load_state_dict``.  This module centralises that, plus the
mask-aware decoding of the two-head battle policy and the team-preview scorer, so
the logic is small, shared, and unit-testable in isolation (task #13).

  battle:  load_bc_policy(path)         -> (AttnBCPolicy, head_names)
           bc_action_indices(model, heads, state_vec, mask0, mask1)
               -> (a0|None, a1|None)    # masked argmax per head, None if no legal

  preview: load_team_chooser(path)      -> (TeamPreviewModel, vocab, config)
           team_order(model, vocab, cfg, our_species, opp_species, n)
               -> [roster indices]      # leads first, then the rest of the bring
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

# ── Canonical PRODUCTION checkpoints — the single source of truth so generation.py / gauntlet.py /
#    run_local_battle.py (and any future caller) never drift apart again.  #27 deleted the flat
#    Models use a ``<type>_<variant>[_genN].pt`` naming scheme (2026-06-25).
#    ⚠ ENCODER-v18 INTERIM (2026-06-26): the self-play champion ``checkpoints_attn/battle_selfplay_gen141.pt``
#    (gen141) is a v17 model and is UNLOADABLE on the current v18 encoder — the stale-layout guard in
#    load_bc_policy() rejects it (STATE_DIM / layout-version mismatch). Until a v18 self-play champion exists,
#    PRODUCTION serves the v18 BC anchor ``checkpoints_attn_pre_gen141/battle_base.pt`` (retrained on
#    M-A + M-B + Bo3, layout v18, STATE_DIM 4961, val top1 0.536). ⟵ once a v18 champion exists, restore the
#    champion path on the DEFAULT_BC_CHECKPOINT line below.
#    Team-preview: ``checkpoints/teampreview_sbda.pt`` (SBDA tpfeat-v6, feat_dim 253) is UNAFFECTED by the v18
#    bump and still serves — the gen141+SBDA pair beat the old prod (BC base + legacy 46-dim TP) 61.8% over
#    1480 M-B battles. Variant kept alongside: ``checkpoints_pre_sbda/teampreview_base.pt`` (legacy 46-dim).
#    ⚠ Self-play TRAINING points its base/KL-anchor ckpt at ``battle_base.pt`` (the BC anchor) — which is now
#    ALSO the interim served file; this is fine (the anchor IS the best v18 model we have). See the configs.
_AI_TRAIN = Path(__file__).resolve().parents[2] / "ai_train_scripts"
# INTERIM v18 serve (see note above): v17 gen141 is unloadable on v18 → serve the v18 BC anchor battle_base.pt.
DEFAULT_BC_CHECKPOINT = _AI_TRAIN / "BC_model" / "checkpoints_attn_pre_gen141" / "battle_base.pt"
DEFAULT_TP_CHECKPOINT = _AI_TRAIN / "teamPreview_model" / "checkpoints" / "teampreview_sbda.pt"

try:
    import torch
    _TORCH = True
except ImportError:  # pragma: no cover
    _TORCH = False


# ── Mask-aware argmax ─────────────────────────────────────────────────────────
def masked_argmax(logits: Sequence[float], mask: Sequence[bool]) -> Optional[int]:
    """Index of the highest-logit LEGAL action, or None if nothing is legal."""
    best_i, best_v = None, float("-inf")
    for i, ok in enumerate(mask):
        if ok and i < len(logits) and float(logits[i]) > best_v:
            best_v, best_i = float(logits[i]), i
    return best_i


def masked_sample(
    logits: Sequence[float], mask: Sequence[bool],
    temperature: float = 0.0, top_p: float = 1.0, rng=None,
) -> Optional[int]:
    """Pick a LEGAL action index from ``logits`` under ``mask``.

    ``temperature <= 0`` → deterministic argmax (byte-identical to
    ``masked_argmax``, including first-wins tie-breaking).  ``temperature > 0`` →
    softmax-sample over the legal logits at that temperature, optionally
    restricted to the top-``p`` nucleus (the smallest set of highest-probability
    actions whose cumulative mass ≥ top_p; the top-1 is always kept).  Returns
    None when no action is legal.

    ``rng`` may be a numpy Generator / RandomState (or None → module default).
    Pure-argmax serving is brittle on OOD boards; a small temperature trades a
    little top-1 fidelity for exploration / less exploitability (TIER-4)."""
    legal = [i for i, ok in enumerate(mask) if ok and i < len(logits)]
    if not legal:
        return None
    if temperature is None or temperature <= 0.0:
        return max(legal, key=lambda i: float(logits[i]))

    z = np.array([float(logits[i]) for i in legal], dtype=np.float64) / float(temperature)
    z -= z.max()                                   # stabilise before exp
    p = np.exp(z)
    p /= p.sum()

    if top_p < 1.0:
        order = np.argsort(-p)                      # high→low probability
        csum = np.cumsum(p[order])
        # keep the smallest prefix whose cumulative mass reaches top_p (inclusive
        # of the crossing action); always keep at least the most likely one.
        cut = min(max(int(np.searchsorted(csum, top_p)) + 1, 1), len(order))
        masked = np.zeros_like(p)
        masked[order[:cut]] = p[order[:cut]]
        total = masked.sum()
        if total > 0:
            p = masked / total

    draw = (rng if rng is not None else np.random).choice(len(legal), p=p)
    return legal[int(draw)]


# ── Battle policy (two-head BC) ───────────────────────────────────────────────
def load_bc_policy(path, device: str = "cpu", _ckpt=None):
    """Rebuild the AttnBCPolicy from a dict checkpoint and load its weights.

    Returns ``(model, head_names)``.  Back-compat: if ``path`` is a pickled
    nn.Module it is returned as-is with head_names=None. ``_ckpt`` (internal): an
    already-loaded checkpoint dict to reuse instead of re-reading ``path`` from disk
    (avoids a double torch.load when the caller has the dict, e.g. C51 auto-detect)."""
    if not _TORCH:
        raise RuntimeError("PyTorch unavailable")
    ckpt = _ckpt if _ckpt is not None else torch.load(path, map_location=device, weights_only=False)
    if not (isinstance(ckpt, dict) and "model_state" in ckpt):
        # legacy: a directly-pickled module
        if callable(ckpt):
            ckpt.eval()
            return ckpt, None
        raise ValueError(f"unrecognised BC checkpoint at {path}: {type(ckpt)}")
    cfg = ckpt.get("config", {})
    from v_dance.encoders.state_encoder import get_gimmick_dim, get_state_dim, get_state_layout_version

    # Stale-layout guard (#5): a checkpoint trained on an older tensor layout has a
    # different STATE_DIM, which would otherwise fail deep inside the first matmul
    # with a cryptic shape error (the 938→1386→1398→1806 churn did exactly that).
    # Reject it loudly here so the only fix — re-export + retrain on the current
    # layout — is unmistakable.  The layout VERSION is checked first (it survives
    # even a future same-dim reorder); the dim is the hard backstop.
    ckpt_dim = cfg.get("state_dim")
    ckpt_ver = cfg.get("state_layout_version")
    cur_dim, cur_ver = get_state_dim(), get_state_layout_version()
    if ckpt_dim is not None and ckpt_dim != cur_dim:
        raise ValueError(
            f"BC checkpoint state_dim={ckpt_dim} (layout v{ckpt_ver}) does not match "
            f"the current encoder STATE_DIM={cur_dim} (layout v{cur_ver}) at {path}. "
            f"This checkpoint predates a state-layout change — re-export + retrain "
            f"on the current layout."
        )
    if ckpt_ver is not None and ckpt_ver != cur_ver:
        raise ValueError(
            f"BC checkpoint state_layout_version={ckpt_ver} != current {cur_ver} "
            f"at {path} (same STATE_DIM but a different layout) — retrain required."
        )
    # Attn-only (refactor #27): the per-mon set-attention AttnBCPolicy is THE production battle
    # net; the flat BCPolicy was retired. A legacy flat checkpoint (no model_type=="attn") is no
    # longer loadable — reject it loudly so the only fix (retrain on the attn arch) is unmistakable.
    if cfg.get("model_type") != "attn":
        raise ValueError(
            f"BC checkpoint at {path} is not an attn checkpoint (config model_type="
            f"{cfg.get('model_type')!r}). The flat BCPolicy was retired in the attn-only refactor "
            f"(#27) — retrain on the attn architecture."
        )
    from v_dance.models.bc_model_attn import AttnBCPolicy
    model = AttnBCPolicy(
        state_dim=cfg["state_dim"],
        action_dim=cfg["action_dim"],
        gimmick_dim=cfg.get("gimmick_dim", get_gimmick_dim()),
        d_model=cfg.get("d_model", 128),
        n_heads=cfg.get("n_heads", 4),
        n_layers=cfg.get("n_layers", 2),
        ff_mult=cfg.get("ff_mult", 2),
        dropout=cfg.get("dropout", 0.0),
        heads=tuple(cfg.get("heads", ("our_a", "our_b"))),
        # Aux-opp checkpoints carry extra action heads but gimmick heads only for
        # the own slots — pass the saved set so the strict load matches.
        gimmick_heads=cfg.get("gimmick_heads"),
        value_readout=cfg.get("value_readout", "mean"),
        opp_cond=cfg.get("opp_cond", False),    # Level B: rebuild the opp-conditioned our heads if stamped
    )
    # A PRE-gimmick checkpoint has no ``gimmick_heads.*`` weights.  The model now
    # always carries gimmick heads, so load non-strictly for those old checkpoints
    # (the untrained gimmick head stays at init) and FLAG it so the player never
    # acts on an untrained gimmick head — it megas only with a real gimmick-trained
    # checkpoint (post re-export+retrain), exactly as the handoff requires.
    state = ckpt["model_state"]
    has_gimmick = any(k.startswith("gimmick_heads.") for k in state)
    has_value = any(k.startswith("value_head.") for k in state)
    # Load non-strictly when EITHER newer head is absent (a pre-gimmick or
    # pre-value checkpoint) so the missing head stays at init; the *_trained flags
    # below gate whether the serve player is allowed to act on it.
    model.load_state_dict(state, strict=has_gimmick and has_value)
    # The gimmick head is usable only if the checkpoint both CONTAINS it AND was
    # trained on gimmick-labelled data (config flag, default True for forward
    # compat when the weights are present but the flag predates this field).
    model._gimmick_trained = bool(has_gimmick and cfg.get("gimmick_trained", True))
    # Value head usable only if present AND trained on outcome-labelled data.
    model._value_trained = bool(has_value and cfg.get("value_trained", False))
    model.to(device).eval()
    return model, tuple(cfg.get("heads", ("our_a", "our_b")))


def _head_logits(out, head_names) -> Tuple[np.ndarray, np.ndarray]:
    """Pull the two slot logit vectors out of a forward() result, in slot order
    (head 0 → our_a → slot 0, head 1 → our_b → slot 1)."""
    if isinstance(out, dict):
        names = head_names or tuple(out.keys())
        l0, l1 = out[names[0]], out[names[1]]
    else:  # tensor (2, A) or list
        l0, l1 = out[0], out[1]
    return (np.asarray(l0.detach().cpu()).ravel(),
            np.asarray(l1.detach().cpu()).ravel())


def bc_action_indices(
    model, head_names, state_vec: np.ndarray,
    mask0: Sequence[bool], mask1: Sequence[bool], device: str = "cpu",
    temperature: float = 0.0, top_p: float = 1.0, rng=None,
) -> Tuple[Optional[int], Optional[int]]:
    """Run the policy on one state vector and return a LEGAL action index for each
    active slot (None where no legal action exists).

    Defaults (``temperature=0``) → masked argmax, byte-identical to the original.
    Pass ``temperature > 0`` (and optionally ``top_p`` < 1) for serve-side
    temperature / nucleus sampling (TIER-4)."""
    l0, l1 = head_logits(model, head_names, state_vec, device)
    return (masked_sample(l0, mask0, temperature, top_p, rng),
            masked_sample(l1, mask1, temperature, top_p, rng))


def head_logits(
    model, head_names, state_vec: np.ndarray, device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the policy once and return the two per-head raw ACTION logit vectors
    (slot 0 = our_a, slot 1 = our_b).  Used when the caller needs the logits
    directly — e.g. the forced-replacement path applies a per-slot switch-only
    mask with cross-slot dedup, which a single bc_action_indices call can't
    express."""
    with torch.no_grad():
        t = torch.as_tensor(np.asarray(state_vec, dtype=np.float32), device=device)
        out = model(t)
    # forward now returns (actions, gimmicks); back-compat with an action-only dict.
    actions = out[0] if isinstance(out, tuple) else out
    return _head_logits(actions, head_names)


def gimmick_logits(
    model, head_names, state_vec: np.ndarray, device: str = "cpu",
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Run the policy once and return the two per-head raw GIMMICK logit vectors
    (slot 0 = our_a, slot 1 = our_b), or None for a legacy action-only model.
    The caller masked-argmaxes these over the per-slot gimmick legal mask."""
    with torch.no_grad():
        t = torch.as_tensor(np.asarray(state_vec, dtype=np.float32), device=device)
        out = model(t)
    if not (isinstance(out, tuple) and len(out) >= 2):
        return None
    return _head_logits(out[1], head_names)


def gimmick_trained(model) -> bool:
    """Whether ``model`` was loaded from a checkpoint that actually trained the
    gimmick head (load_bc_policy sets this).  The player must not act on an
    untrained gimmick head (a pre-gimmick checkpoint loaded non-strictly)."""
    return bool(getattr(model, "_gimmick_trained", False))


def value_logit(model, state_vec: np.ndarray, device: str = "cpu") -> Optional[float]:
    """Win PROBABILITY in [0,1] for one state from the value head, or None for a
    legacy (pre-value) model.  Runs the policy once and applies sigmoid to the
    scalar value logit — the basis for a 1-ply value lookahead at serve (#2)."""
    import math
    with torch.no_grad():
        t = torch.as_tensor(np.asarray(state_vec, dtype=np.float32), device=device)
        out = model(t)
    if not (isinstance(out, tuple) and len(out) >= 3):
        return None
    v = float(np.asarray(out[2].detach().cpu()).ravel()[0])
    return 1.0 / (1.0 + math.exp(-v))


def value_trained(model) -> bool:
    """Whether ``model`` was loaded from a checkpoint that actually trained the
    value head (load_bc_policy sets this); the serve player must not trust an
    untrained value head."""
    return bool(getattr(model, "_value_trained", False))


def _masked_softmax(logits, mask) -> np.ndarray:
    """Softmax of ``logits`` over the LEGAL actions (``mask`` truthy), zero elsewhere; an all-zero
    vector when no action is legal. Output length = ``len(logits)``."""
    z = np.asarray(logits, dtype=np.float64).ravel()
    n = z.shape[0]
    out = np.zeros(n, dtype=np.float64)
    m = np.asarray(list(mask), dtype=bool)
    if m.shape[0] < n:
        m = np.concatenate([m, np.zeros(n - m.shape[0], dtype=bool)])
    idx = np.nonzero(m[:n])[0]
    if idx.size == 0:
        return out
    zz = z[idx] - z[idx].max()
    p = np.exp(zz)
    out[idx] = p / p.sum()
    return out


def opp_action_prior(model, state_vec: np.ndarray, snap: dict, *, device: str = "cpu",
                     opp_heads=("opp_a", "opp_b")):
    """The net's predicted OPPONENT action distribution: ``{"opp_a": [A], "opp_b": [A]}`` =
    softmax(opp-head logits) masked by ``build_opp_action_mask(snap)`` (decision-time opp legality from
    the flipped snapshot).  This is the opponent model the Level-C search plans against (belief-weighted
    expectimax) — NOT used by A3 itself.  Returns ``None`` for a model with no opponent heads (a plain
    our-only BC net).  Runs the policy ONCE; each value is an ``np.ndarray`` summing to 1 over the legal
    opp actions (all-zero when the opp slot has no legal action)."""
    with torch.no_grad():
        t = torch.as_tensor(np.asarray(state_vec, dtype=np.float32), device=device)
        out = model(t)
    actions = out[0] if isinstance(out, tuple) else out
    if not isinstance(actions, dict) or any(h not in actions for h in opp_heads):
        return None
    l0, l1 = _head_logits(actions, opp_heads)
    from v_dance.encoders.action_codec import build_opp_action_mask
    masks = build_opp_action_mask(snap or {})
    return {"opp_a": _masked_softmax(l0, masks.get("opp_a") or []),
            "opp_b": _masked_softmax(l1, masks.get("opp_b") or [])}


# ── Team-preview scorer ───────────────────────────────────────────────────────
def load_team_chooser(path, device: str = "cpu"):
    """Rebuild the TeamPreviewModel from a dict checkpoint and load its weights.
    Returns ``(model, vocab, config)``."""
    if not _TORCH:
        raise RuntimeError("PyTorch unavailable")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if not (isinstance(ckpt, dict) and "model_state" in ckpt):
        if callable(ckpt):
            ckpt.eval()
            return ckpt, ckpt.__dict__.get("vocab", {}), {}
        raise ValueError(f"unrecognised team-chooser checkpoint at {path}")
    cfg = ckpt.get("config", {})
    vocab = ckpt.get("vocab", {})
    # 15b-io.1 lockstep guard: an SBDA checkpoint declares feature_schema=tpfeat-vN; its
    # feat_dim MUST match the current extractor so serving builds the SAME channels it was
    # trained on (else the synergy tags silently zero-pad).  Fail at LOAD, like the BC
    # stale-layout guard.  Legacy nets carry no schema → skipped → byte-identical.
    if uses_tp_features(cfg):
        from v_dance.training.tp_features import FEAT_DIM, FEATURE_SCHEMA_VERSION
        schema = cfg.get("feature_schema")
        if int(cfg.get("feat_dim", 0)) != FEAT_DIM or schema != FEATURE_SCHEMA_VERSION:
            raise ValueError(
                f"team-chooser checkpoint feature_schema={schema} (feat_dim={cfg.get('feat_dim')}) "
                f"is out of lockstep with the current tp_features {FEATURE_SCHEMA_VERSION} "
                f"(FEAT_DIM={FEAT_DIM}) at {path} — re-export + retrain the SBDA TP net on the "
                f"current schema."
            )
    from v_dance.models.teampreview_model import TeamPreviewModel
    model = TeamPreviewModel(
        vocab_size=cfg["vocab_size"],
        feat_dim=cfg["feat_dim"],
        emb_dim=cfg.get("emb_dim", 32),
        hidden=cfg.get("hidden", 128),
        dropout=cfg.get("dropout", 0.0),
        # 15b-arch.1: absent in legacy configs -> False -> byte-identical mean-pool model loads unchanged.
        use_self_attn=cfg.get("use_self_attn", False),
        use_cross_attn=cfg.get("use_cross_attn", False),
        attn_heads=cfg.get("attn_heads", 4),
        use_teammate_bias=cfg.get("use_teammate_bias", False),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, vocab, cfg


# ── Team-preview feature recipe: legacy dex-only vs SBDA tp_features (15b-io.1) ─
def uses_tp_features(cfg: Optional[dict]) -> bool:
    """True when a team-chooser checkpoint was trained on the SBDA per-mon feature
    schema (``tp_features.own/opp_mon_features`` — the mechanic-tag synergy block)
    rather than the legacy 46-dim ``mon_dex_features``.

    Detected by the explicit ``feature_schema`` stamp the SBDA trainer writes;
    legacy checkpoints have no such stamp and stay on the old path (byte-identical).
    The feat_dim is the hard backstop the load-time lockstep guard checks."""
    schema = (cfg or {}).get("feature_schema")
    return bool(schema) and str(schema).startswith("tpfeat")


class _NullBelief:
    """A belief that knows nothing — lets the SBDA extractor still produce a valid
    typing-grounded base (dex types/stats + type immunity/effectiveness, no
    ability/move/usage/teammate tags) when no real ``BeliefState`` is available,
    instead of crashing.  The serve player passes the real Pikalytics belief; this
    is only the safety floor so a missing belief degrades gracefully (and loudly —
    the caller warns), never silently corrupts the synergy channels."""
    def known(self, species):  # noqa: D401
        return False

    def ability_distribution(self, species, top_k: int = 4):
        return []

    def move_distribution(self, species, top_k: int = 8):
        return []

    def usage(self, species):
        return 0.0

    def teammates(self, species, top_k: int = 16):
        return []


_NULL_BELIEF = _NullBelief()


def _pack_side(species: Sequence[str], vocab: dict, feat_dim: int, *,
               belief=None, own_known: Optional[dict] = None,
               use_tp_features: bool = False):
    """6-slot (idx[6], feat[6,F]) tensors for one side's roster, padding to 6.

    Two feature recipes, selected by the loaded checkpoint's schema so train==serve:
      * legacy (``use_tp_features=False``): dex features only
        (``teampreview_dataset.mon_dex_features``) — byte-identical to the original.
      * SBDA (``use_tp_features=True``): the SHARED ``tp_features`` extractor (the same
        call the SBDA dataset uses), needing a ``BeliefState`` (or the null floor).
        ``own_known`` (a ``{norm_species: OwnKnown}``) turns on the sharp OWN overlay;
        ``None`` = symmetric belief-only base — parity-correct for a Type-B BC-pretrained
        net, whose ``has_own_detail`` bit is always 0.
    """
    from v_dance.parser.vod_parser.pokedex import norm_species
    idx = [0] * 6
    if use_tp_features:
        from v_dance.training.tp_features import (
            FEAT_DIM, own_mon_features, opp_mon_features,
        )
        # LOCKSTEP guard (15b-io.1): the checkpoint's feat_dim MUST equal the current
        # extractor's FEAT_DIM, else a schema drift would silently zero-pad the synergy
        # channels (the exact failure the handoff warned about).  Fail loud instead.
        if int(feat_dim) != FEAT_DIM:
            raise ValueError(
                f"team-chooser feat_dim={feat_dim} != tp_features.FEAT_DIM={FEAT_DIM} — the "
                f"checkpoint's SBDA feature schema is out of lockstep with the code; re-export "
                f"+ retrain the TP net on the current schema."
            )
        b = belief if belief is not None else _NULL_BELIEF
        feat = np.zeros((6, FEAT_DIM), dtype=np.float32)
        for i, sp in enumerate(list(species)[:6]):
            ns = norm_species(sp)
            idx[i] = int(vocab.get(ns, 0))     # 0 = PAD / unseen
            if own_known is not None:
                feat[i] = own_mon_features(ns, b, own_known.get(ns))
            else:
                feat[i] = opp_mon_features(ns, b)
        return idx, feat

    # legacy dex-only path (unchanged) ─────────────────────────────────────────
    from v_dance.training.teampreview_dataset import mon_dex_features
    feat = np.zeros((6, feat_dim), dtype=np.float32)
    for i, sp in enumerate(list(species)[:6]):
        ns = norm_species(sp)
        idx[i] = int(vocab.get(ns, 0))     # 0 = PAD / unseen
        feat[i] = mon_dex_features(ns)
    return idx, feat


def team_order(
    model, vocab: dict, cfg: dict,
    our_species: Sequence[str], opp_species: Sequence[str],
    n: int, device: str = "cpu", *,
    belief=None, own_known: Optional[dict] = None,
) -> List[int]:
    """Return roster indices to bring, LEADS FIRST (matching how the trainer
    labels — the first two brought are the leads), capped at ``n``.

    ``our_species`` / ``opp_species`` are the teampreview rosters (any species
    string form; normalised internally).  Falls back to first-n on any issue.

    ``belief`` (a Pikalytics ``BeliefState``) is REQUIRED for an SBDA checkpoint
    (``feature_schema=tpfeat-*``) so the per-mon synergy features + teammate-bias
    prior match what the net was trained on; legacy 46-dim nets ignore it.
    ``own_known`` optionally turns on the sharp OWN overlay (gated to the self-play
    fine-tune; the BC-pretrained net serves overlay-off for train/serve parity).
    """
    valid = min(len(our_species), 6)
    if valid == 0:
        return list(range(n))
    feat_dim = cfg.get("feat_dim", 46)
    bring_k = int(cfg.get("bring_k", 4))
    lead_k = int(cfg.get("lead_k", 2))
    use_tp = uses_tp_features(cfg)

    oi, of = _pack_side(our_species, vocab, feat_dim,
                        belief=belief, own_known=own_known, use_tp_features=use_tp)
    pi, pf = _pack_side(opp_species, vocab, feat_dim,
                        belief=belief, own_known=None, use_tp_features=use_tp)

    # 15b-feat.1b: feed the Pikalytics co-occurrence prior as the self-attention
    # bias when the SBDA net was built with ``use_teammate_bias``.  Needs a belief;
    # absent → ``our_affinity=None`` and the model runs without the prior (its
    # forward treats None as "no bias"), so serving stays correct, just un-primed.
    aff_t = None
    if use_tp and getattr(model, "use_teammate_bias", False) and belief is not None:
        from v_dance.training.tp_features import teammate_affinity_matrix
        aff = teammate_affinity_matrix(our_species, belief, n=6)
        aff_t = torch.as_tensor(aff[None], device=device)

    with torch.no_grad():
        bring_logits, lead_logits = model(
            torch.as_tensor([oi], device=device),
            torch.as_tensor([pi], device=device),
            torch.as_tensor(of[None], device=device),
            torch.as_tensor(pf[None], device=device),
            aff_t,
        )
    bring = np.asarray(bring_logits[0].detach().cpu()).ravel()
    lead = np.asarray(lead_logits[0].detach().cpu()).ravel()

    # Top bring_k roster positions (over valid slots), then leads = top lead_k of
    # those by the lead head; emit leads first then the remaining brought.
    by_bring = sorted(range(valid), key=lambda i: -bring[i])
    brought = by_bring[: min(bring_k, valid, n)]
    leads = sorted(brought, key=lambda i: -lead[i])[:lead_k]
    order = leads + [b for b in brought if b not in leads]
    return order[:n]
