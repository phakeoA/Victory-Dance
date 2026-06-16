"""
local_battle/model_io.py  —  load trained checkpoints + decode them for serving
================================================================================
The training scripts save checkpoints as DICTS — ``{"model_state", "config",
...}`` — not pickled ``nn.Module`` objects.  The live player therefore cannot
``torch.load(path)`` and call the result; it must reconstruct the architecture
from ``config`` and ``load_state_dict``.  This module centralises that, plus the
mask-aware decoding of the two-head battle policy and the team-preview scorer, so
the logic is small, shared, and unit-testable in isolation (task #13).

  battle:  load_bc_policy(path)         -> (BCPolicy, head_names)
           bc_action_indices(model, heads, state_vec, mask0, mask1)
               -> (a0|None, a1|None)    # masked argmax per head, None if no legal

  preview: load_team_chooser(path)      -> (TeamPreviewModel, vocab, config)
           team_order(model, vocab, cfg, our_species, opp_species, n)
               -> [roster indices]      # leads first, then the rest of the bring
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

# ── Path bootstrap: the model packages + data/scripts (encoders/pokedex) ──────
_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (
    str(_REPO_ROOT / "data" / "scripts"),
    str(_REPO_ROOT / "ai_train_scripts" / "BC_model"),
    str(_REPO_ROOT / "ai_train_scripts" / "teamPreview_model"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

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
def load_bc_policy(path, device: str = "cpu"):
    """Rebuild the BCPolicy from a dict checkpoint and load its weights.

    Returns ``(model, head_names)``.  Back-compat: if ``path`` is a pickled
    nn.Module it is returned as-is with head_names=None.
    """
    if not _TORCH:
        raise RuntimeError("PyTorch unavailable")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if not (isinstance(ckpt, dict) and "model_state" in ckpt):
        # legacy: a directly-pickled module
        if callable(ckpt):
            ckpt.eval()
            return ckpt, None
        raise ValueError(f"unrecognised BC checkpoint at {path}: {type(ckpt)}")
    cfg = ckpt.get("config", {})
    from bc_model import BCPolicy
    from state_encoder import get_gimmick_dim
    model = BCPolicy(
        state_dim=cfg["state_dim"],
        action_dim=cfg["action_dim"],
        hidden_dims=tuple(cfg.get("hidden_dims", (512, 256))),
        dropout=cfg.get("dropout", 0.0),
        heads=tuple(cfg.get("heads", ("our_a", "our_b"))),
        gimmick_dim=cfg.get("gimmick_dim", get_gimmick_dim()),
        # Aux-opp checkpoints carry extra action heads but gimmick heads only for
        # the own slots — pass the saved set so the strict load matches.
        gimmick_heads=cfg.get("gimmick_heads"),
    )
    # A PRE-gimmick checkpoint has no ``gimmick_heads.*`` weights.  The model now
    # always carries gimmick heads, so load non-strictly for those old checkpoints
    # (the untrained gimmick head stays at init) and FLAG it so the player never
    # acts on an untrained gimmick head — it megas only with a real gimmick-trained
    # checkpoint (post re-export+retrain), exactly as the handoff requires.
    state = ckpt["model_state"]
    has_gimmick = any(k.startswith("gimmick_heads.") for k in state)
    model.load_state_dict(state, strict=has_gimmick)
    # The gimmick head is usable only if the checkpoint both CONTAINS it AND was
    # trained on gimmick-labelled data (config flag, default True for forward
    # compat when the weights are present but the flag predates this field).
    model._gimmick_trained = bool(has_gimmick and cfg.get("gimmick_trained", True))
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
    from teampreview_model import TeamPreviewModel
    model = TeamPreviewModel(
        vocab_size=cfg["vocab_size"],
        feat_dim=cfg["feat_dim"],
        emb_dim=cfg.get("emb_dim", 32),
        hidden=cfg.get("hidden", 128),
        dropout=cfg.get("dropout", 0.0),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, vocab, cfg


def _pack_side(species: Sequence[str], vocab: dict, feat_dim: int):
    """6-slot (idx[6], feat[6,F]) tensors for one side's roster, padding to 6 and
    matching the training encoding (vocab id by normalised species; dex features
    from teampreview_dataset.mon_dex_features)."""
    from teampreview_dataset import mon_dex_features
    from vod_parser.pokedex import norm_species
    idx = [0] * 6
    feat = np.zeros((6, feat_dim), dtype=np.float32)
    for i, sp in enumerate(list(species)[:6]):
        ns = norm_species(sp)
        idx[i] = int(vocab.get(ns, 0))     # 0 = PAD / unseen
        feat[i] = mon_dex_features(ns)
    return idx, feat


def team_order(
    model, vocab: dict, cfg: dict,
    our_species: Sequence[str], opp_species: Sequence[str],
    n: int, device: str = "cpu",
) -> List[int]:
    """Return roster indices to bring, LEADS FIRST (matching how the trainer
    labels — the first two brought are the leads), capped at ``n``.

    ``our_species`` / ``opp_species`` are the teampreview rosters (any species
    string form; normalised internally).  Falls back to first-n on any issue.
    """
    valid = min(len(our_species), 6)
    if valid == 0:
        return list(range(n))
    feat_dim = cfg.get("feat_dim", 46)
    bring_k = int(cfg.get("bring_k", 4))
    lead_k = int(cfg.get("lead_k", 2))

    oi, of = _pack_side(our_species, vocab, feat_dim)
    pi, pf = _pack_side(opp_species, vocab, feat_dim)
    with torch.no_grad():
        bring_logits, lead_logits = model(
            torch.as_tensor([oi], device=device),
            torch.as_tensor([pi], device=device),
            torch.as_tensor(of[None], device=device),
            torch.as_tensor(pf[None], device=device),
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
