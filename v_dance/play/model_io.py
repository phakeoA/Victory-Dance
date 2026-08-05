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

import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

# ── Canonical PRODUCTION checkpoints — the single source of truth so every caller
#    (harnesses / gauntlet / run_local_battle) never drifts. ``<type>_<variant>[_genN].pt``
#    naming scheme (2026-06-25).
#    ⚠ STATE (D6 promotion, 2026-07-12 — USER call on the era-2/set-head evidence): the
#    defaults now POINT AT the measured winners, so a flag-less/env-less caller serves the
#    same stack as the ``.env`` deploy.
#    * BC = ``checkpoints_attn_era2`` (ep5 val 0.592; step-4 gates green; online block
#      performance rating 1271 vs era-1's 1212 at equal opposition).
#    * TP = ``checkpoints_set/teampreview_sbda.pt`` (contrastive set head; --set-ab gate
#      +3.8pp bring-set exact over v6 serve-faithful; online block performance 1425 at
#      mean-1393 opposition — artifacts/logs/tp_set_ab_gate.log).
#    Era-chain rule: warm-start + adv-value base for era N+1 = the gated era-N net; the
#    pre_gen141 anchor FILE (val 0.5853) stays untouched as the fixed historical reference.
#    Era-3 promoted 2026-07-20; same-day upgraded to ARM B (M2 closed-strip: ruler sweep-win
#    0.6111 pooled, t1 recovered + late-game kept; gates green). Rollback chain:
#    ``checkpoints_attn_era3_cand`` (50g-laddered) → ``checkpoints_attn_era2/battle_base.pt`` ·
#    ``checkpoints/teampreview_sbda.pt`` (v6). Legacy 46-dim TP at ``checkpoints_pre_sbda/``.
_AI_TRAIN = Path(__file__).resolve().parents[2] / "ai_train_scripts"
DEFAULT_BC_CHECKPOINT = _AI_TRAIN / "BC_model" / "checkpoints_attn_era3_armB" / "battle_base.pt"
DEFAULT_TP_CHECKPOINT = _AI_TRAIN / "teamPreview_model" / "checkpoints_set" / "teampreview_sbda.pt"

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
_PAIR_ECHO_DONE = False    # 2b: the pair-decode load echo prints once per process


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
        # Phase 1a match-memory core — absent keys (every pre-memory checkpoint)
        # default to 0/stateless, so old checkpoints rebuild byte-identically.
        memory_dim=cfg.get("memory_dim", 0),
        mem_layers=cfg.get("mem_layers", 2),
        mem_heads=cfg.get("mem_heads", 4),
        max_mem_len=cfg.get("max_mem_len", 64),
        # Phase 2 archetype-z — absent keys (every pre-z checkpoint) default to
        # 0/off, rebuilding byte-identically.
        n_archetypes=cfg.get("n_archetypes", 0),
        z_dim=cfg.get("z_dim", 0),
        # Era-4 2b — absent key (every pre-2b checkpoint) defaults to False,
        # rebuilding byte-identically.
        pair_cond=cfg.get("pair_cond", False),
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
    # Memory models: the TRAINED window length (train_bc --sequence-len stamp).
    # Serve/eval must not exceed it — frames at ages >= the trained window would
    # read never-trained mem_pos_emb rows (audit 2026-07-02). 1 = stateless.
    model._sequence_len = int(cfg.get("sequence_len", 1) or 1)
    # Phase-2 z checkpoints embed the archetype centroids (centroids/mu/sd/
    # mon_feature_dim) so serve-time nearest-centroid assignment needs no
    # sidecar file — the checkpoint is self-contained. None for pre-z ckpts.
    model._z_artifact = ckpt.get("z_artifact")
    # Era-4 2b: sequential pair decode rides the checkpoint stamp; VD_PAIR_DECODE=0
    # is the serve kill switch (zero-cond forwards reproduce the unconditioned donor
    # bit-exactly by the zero-pad warm-start construction). Echo per the standing
    # verify-flag-uptake rule — ONCE per process (self-play meters reload the target
    # every iteration; a per-load echo floods the log).
    if cfg.get("pair_cond", False):
        import os as _os
        enabled = _os.environ.get("VD_PAIR_DECODE", "1").strip().lower() not in ("0", "false", "off")
        model._pair_decode = enabled
        global _PAIR_ECHO_DONE
        if not _PAIR_ECHO_DONE:
            _PAIR_ECHO_DONE = True
            print(f"[battle] 2b pair decode "
                  f"{'ACTIVE (sequential, confidence-first)' if enabled else 'DISABLED by VD_PAIR_DECODE'}"
                  f" — pair_cond checkpoint")
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


def _policy_forward(model, state_input, device: str = "cpu", partner_actions=None):
    """One policy forward for ONE decision, single-frame or frame-stacked.

    ``state_input``: a 1-D ``(STATE_DIM,)`` vector → the plain single-turn
    forward (unchanged behaviour), or a 2-D ``(T, STATE_DIM)`` MATCH FRAME-STACK
    → ``forward_with_memory`` on a memory model (Phase-3 serve-time memory
    carry; the outputs answer for the LAST frame).  A 2-D input handed to a
    STATELESS model degrades gracefully to the last frame alone, so callers may
    stack unconditionally.  ``partner_actions`` (2b pair decode): only passed on
    when set, so legacy pickled modules never see the kwarg."""
    t = torch.as_tensor(np.asarray(state_input, dtype=np.float32), device=device)
    kw = {"partner_actions": partner_actions} if partner_actions is not None else {}
    with torch.no_grad():
        if t.dim() == 2:
            if int(getattr(model, "memory_dim", 0) or 0):
                return model.forward_with_memory(t, **kw)
            return model(t[-1], **kw)
        return model(t, **kw)


def bc_action_indices(
    model, head_names, state_vec: np.ndarray,
    mask0: Sequence[bool], mask1: Sequence[bool], device: str = "cpu",
    temperature: float = 0.0, top_p: float = 1.0, rng=None,
    bias0=None, bias1=None, pair_futility=None,
) -> Tuple[Optional[int], Optional[int]]:
    """Run the policy on one state (vector OR frame-stack — see _policy_forward)
    and return a LEGAL action index for each active slot (None where no legal
    action exists).

    Defaults (``temperature=0``) → masked argmax, byte-identical to the original.
    Pass ``temperature > 0`` (and optionally ``top_p`` < 1) for serve-side
    temperature / nucleus sampling (TIER-4). ``bias0``/``bias1`` (optional
    per-slot (ACTION_DIM,) arrays) are ADDED to the raw logits before the masked
    pick — the B-L1 adapt-rules tilt; None (the default) is byte-identical.

    Era-4 2b: a pair_cond checkpoint (unless VD_PAIR_DECODE=0) decodes
    SEQUENTIALLY — see _pair_decode_indices — turning the joint pick into
    p(a)·p(b|a); non-pair checkpoints keep the original independent path.
    ``pair_futility`` (optional, pair decode only): callable
    ``(second_slot, first_slot, first_action) -> action indices to DROP`` from
    the second slot's mask — the JOINT futility rules only a sequential decode
    can express (e.g. Helping Hand when the partner already chose a status
    move). Built by the caller from game knowledge; None = no joint rules."""
    if getattr(model, "_pair_decode", False):
        return _pair_decode_indices(model, head_names, state_vec, mask0, mask1,
                                    device, temperature, top_p, rng, bias0, bias1,
                                    pair_futility=pair_futility)
    l0, l1 = head_logits(model, head_names, state_vec, device)
    if bias0 is not None:
        l0 = l0 + np.asarray(bias0, dtype=l0.dtype)
    if bias1 is not None:
        l1 = l1 + np.asarray(bias1, dtype=l1.dtype)
    return (masked_sample(l0, mask0, temperature, top_p, rng),
            masked_sample(l1, mask1, temperature, top_p, rng))


def _pair_decode_indices(
    model, head_names, state_vec: np.ndarray,
    mask0: Sequence[bool], mask1: Sequence[bool], device: str = "cpu",
    temperature: float = 0.0, top_p: float = 1.0, rng=None,
    bias0=None, bias1=None, pair_futility=None,
) -> Tuple[Optional[int], Optional[int]]:
    """2b sequential decode: pick the HIGHER-CONFIDENCE slot first from the
    zero-cond logits, then re-forward with that pick as the partner one-hot so
    the second slot answers p(b|a).  Deterministic under argmax (ties → slot 0);
    the temperature/top_p/bias knobs apply per slot exactly as in the
    independent path.  A slot with no legal action conditions the other with
    zeros — identical to the zero-cond path, which the cond-dropout trained."""
    l0, l1 = head_logits(model, head_names, state_vec, device)
    if bias0 is not None:
        l0 = l0 + np.asarray(bias0, dtype=l0.dtype)
    if bias1 is not None:
        l1 = l1 + np.asarray(bias1, dtype=l1.dtype)
    conf0 = float(_masked_softmax(l0, mask0).max())
    conf1 = float(_masked_softmax(l1, mask1).max())
    first = 0 if conf0 >= conf1 else 1
    logits = (l0, l1)
    masks = (mask0, mask1)
    biases = (bias0, bias1)
    a_first = masked_sample(logits[first], masks[first], temperature, top_p, rng)
    second = 1 - first
    if a_first is None:                       # nothing to condition on → zero-cond second
        return ((None, masked_sample(l1, mask1, temperature, top_p, rng)) if first == 0
                else (masked_sample(l0, mask0, temperature, top_p, rng), None))
    onehot = torch.zeros(int(getattr(model, "action_dim")), dtype=torch.float32)
    onehot[int(a_first)] = 1.0
    out = _policy_forward(model, state_vec, device,
                          partner_actions={head_names[second]: onehot})
    actions = out[0] if isinstance(out, tuple) else out
    l_pair = _head_logits(actions, head_names)[second]
    if biases[second] is not None:
        l_pair = l_pair + np.asarray(biases[second], dtype=l_pair.dtype)
    mask_second = masks[second]
    if pair_futility is not None:
        # Joint futility (2026-07-24 batch): the caller's game-knowledge rule drops
        # second-slot actions PROVABLY wasted given the first slot's pick (e.g.
        # Helping Hand when the partner chose a status move). Never empties the
        # mask — if every action would drop, keep the original (fail open).
        try:
            dropped = set(pair_futility(second, first, int(a_first)) or ())
        except Exception:
            dropped = set()
        if dropped:
            trimmed = [bool(ok) and (i not in dropped) for i, ok in enumerate(mask_second)]
            if any(trimmed):
                mask_second = trimmed
    a_second = masked_sample(l_pair, mask_second, temperature, top_p, rng)
    return (a_first, a_second) if first == 0 else (a_second, a_first)


def head_logits(
    model, head_names, state_vec: np.ndarray, device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the policy once and return the two per-head raw ACTION logit vectors
    (slot 0 = our_a, slot 1 = our_b).  Used when the caller needs the logits
    directly — e.g. the forced-replacement path applies a per-slot switch-only
    mask with cross-slot dedup, which a single bc_action_indices call can't
    express.  Accepts a (T, STATE_DIM) frame-stack for memory models."""
    out = _policy_forward(model, state_vec, device)
    # forward now returns (actions, gimmicks); back-compat with an action-only dict.
    actions = out[0] if isinstance(out, tuple) else out
    return _head_logits(actions, head_names)


def gimmick_logits(
    model, head_names, state_vec: np.ndarray, device: str = "cpu",
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Run the policy once and return the two per-head raw GIMMICK logit vectors
    (slot 0 = our_a, slot 1 = our_b), or None for a legacy action-only model.
    The caller masked-argmaxes these over the per-slot gimmick legal mask.
    Accepts a (T, STATE_DIM) frame-stack for memory models."""
    out = _policy_forward(model, state_vec, device)
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
    scalar value logit — the basis for a 1-ply value lookahead at serve (#2).
    Accepts a (T, STATE_DIM) frame-stack for memory models (NOT a batch — batch
    scoring is value_logit_batch)."""
    import math
    out = _policy_forward(model, state_vec, device)
    if not (isinstance(out, tuple) and len(out) >= 3):
        return None
    v = float(np.asarray(out[2].detach().cpu()).ravel()[0])
    return 1.0 / (1.0 + math.exp(-v))


def value_logit_batch(model, state_vecs, device: str = "cpu"):
    """Win-prob ∈ [0,1] for a BATCH of states — ONE model forward over an (N, STATE_DIM) stack (the Level-C
    search's leaf evaluator; uses the GPU when ``device='cuda'``). Returns an np.ndarray (N,) or None for a
    legacy (pre-value) model."""
    arr = np.asarray(state_vecs, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    with torch.no_grad():
        out = model(torch.as_tensor(arr, device=device))
    if not (isinstance(out, tuple) and len(out) >= 3):
        return None
    v = np.asarray(out[2].detach().cpu(), dtype=np.float64).ravel()
    return 1.0 / (1.0 + np.exp(-v))


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
        # v8 changed DIMS AND OFFSETS. v6/v7 checkpoints stay servable through the FROZEN
        # v7 extractor (tp_features_v7 — the exact code they trained on, dead tags and all;
        # serving them through the fixed v8 channels would be train/serve drift, not a fix).
        # _pack_side dispatches on the checkpoint schema.
        if schema in _TP_LEGACY_SCHEMAS:
            from v_dance.training.tp_features_v7 import FEAT_DIM as expected_dim
        elif schema == FEATURE_SCHEMA_VERSION:
            expected_dim = FEAT_DIM
        else:
            expected_dim = None
        if expected_dim is None or int(cfg.get("feat_dim", 0)) != expected_dim:
            raise ValueError(
                f"team-chooser checkpoint feature_schema={schema} (feat_dim={cfg.get('feat_dim')}) "
                f"is out of lockstep with the current tp_features {FEATURE_SCHEMA_VERSION} "
                f"(FEAT_DIM={FEAT_DIM}, legacy-servable: {sorted(_TP_LEGACY_SCHEMAS)}) at {path} — "
                f"re-export + retrain the SBDA TP net on the current schema."
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
        # 2026-07-11: contrastive set head — absent in older configs -> False -> byte-identical.
        use_set_head=cfg.get("use_set_head", False),
        # DS-4c stage 3: Bo3 set-context input — same absent-defaults-False compat rule.
        use_set_ctx=cfg.get("use_set_ctx", False),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, vocab, cfg


# tpfeat schemas served through the FROZEN v7 extractor (v7 = v6 + opp overlay, same dims).
_TP_LEGACY_SCHEMAS = frozenset({"tpfeat-v6", "tpfeat-v7"})


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


class OwnBuildBelief:
    """A belief view whose marginals are CRISP (p=1.0) for OUR OWN mons' true builds,
    delegating everything else to the wrapped real belief.

    v8 serve-side own-knowledge fix (2026-07-23): the own-OVERLAY pathway has almost no
    training data (27 OTS files in ~5.6k), so serving overlay-on would feed untrained
    weights. Instead the own side's BASE channels — the ones every training row exercises —
    are computed from the true build: ability/moves/item at p=1.0. That is distribution
    SHARPENING, not schema drift (the meta already trains crisp values: Ninetales Drought
    0.96, Incineroar Intimidate 0.9997) and it fixes the two-mega smear for our own mon
    (our Charizardite-Y holder reaches sun=1.0 via the stone augmentation instead of the
    belief's X/Y-mixed marginal). Also covers OOV own mons (any USER team): a species the
    belief has never seen still gets crisp mechanic tags from its actual build."""

    def __init__(self, belief, builds: dict):
        self._b = belief if belief is not None else _NULL_BELIEF
        self._builds = builds or {}

    def _bd(self, species):
        return self._builds.get(species)

    def known(self, species):
        return True if self._bd(species) else self._b.known(species)

    def ability_distribution(self, species, top_k: int = 4):
        bd = self._bd(species)
        if bd and bd.get("ability"):
            return [{"name": bd["ability"], "p": 1.0}]
        return self._b.ability_distribution(species, top_k=top_k)

    def move_distribution(self, species, top_k: int = 8):
        bd = self._bd(species)
        if bd and bd.get("moves"):
            return [{"name": m, "p": 1.0} for m in list(bd["moves"])[:top_k]]
        return self._b.move_distribution(species, top_k=top_k)

    def item_distribution(self, species, top_k: int = 5):
        bd = self._bd(species)
        if bd and bd.get("item"):
            return [{"name": bd["item"], "p": 1.0}]
        fn = getattr(self._b, "item_distribution", None)
        return fn(species, top_k=top_k) if callable(fn) else []

    def usage(self, species):
        return self._b.usage(species)

    def teammates(self, species, top_k: int = 16):
        fn = getattr(self._b, "teammates", None)
        return fn(species, top_k=top_k) if callable(fn) else []

    def expected_stats_weighted(self, species, base_stats, **kw):
        fn = getattr(self._b, "expected_stats_weighted", None)
        return fn(species, base_stats, **kw) if callable(fn) else None


def _pack_side(species: Sequence[str], vocab: dict, feat_dim: int, *,
               belief=None, own_known: Optional[dict] = None,
               use_tp_features: bool = False, tp_schema: Optional[str] = None):
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
        # Schema dispatch (v8): a v6/v7 checkpoint is served through the FROZEN v7 extractor —
        # the exact channels it trained on — while v8+ uses the current tp_features.
        if tp_schema in _TP_LEGACY_SCHEMAS:
            from v_dance.training.tp_features_v7 import (
                FEAT_DIM, own_mon_features, opp_mon_features,
            )
        else:
            from v_dance.training.tp_features import (
                FEAT_DIM, own_mon_features, opp_mon_features,
            )
        # LOCKSTEP guard (15b-io.1): the checkpoint's feat_dim MUST equal its extractor's
        # FEAT_DIM, else a schema drift would silently zero-pad the synergy channels (the
        # exact failure the handoff warned about).  Fail loud instead.
        if int(feat_dim) != FEAT_DIM:
            raise ValueError(
                f"team-chooser feat_dim={feat_dim} != tp_features.FEAT_DIM={FEAT_DIM} "
                f"(schema {tp_schema!r}) — the checkpoint's SBDA feature schema is out of "
                f"lockstep with the code; re-export + retrain the TP net on the current schema."
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


# ── TP joint conditional re-decode (2026-07-10) ────────────────────────────────
# The greedy decode takes the top-bring_k mons by INDEPENDENT per-mon logits — it cannot express
# teammate-conditional structure ("Mawile's value depends on who else comes"), which mode-mixed
# brings on mode teams (both mega-stone holders in 3 of 4 online losses). Joint decode re-scores
# every C(valid, k) bring-subset with the SAME net: the left-behind mons are pad-masked (idx 0 +
# zero feature rows — the exact _pack_side pad convention whose rows the model's attention keys
# and context mean already exclude), so each candidate set is scored by its members' bring logits
# IN THE CONTEXT OF exactly its teammates. No rules, no retrain; exceptions (a both-megas bring
# that genuinely fits a matchup) emerge from the net per opponent. Gate: the S3 ruler --joint-ab
# must show joint ≥ greedy on human bring-set agreement before TP_JOINT_BRING flips to True.
TP_JOINT_BRING = False     # serve default; flip only after the offline gate passes

# ── TP contrastive set-head decode (2026-07-11) ────────────────────────────────
# docs/tp_contrastive_set_head_design.md: a checkpoint trained with --set-head carries a
# use_set_head stamp and a head that scores complete 4-subsets AS UNITS (marginal-sum +
# pairwise + set-level terms, trained with a 15-way listwise CE against the human's set).
# Unlike the null-#8 joint decode there is NO masking: the trunk always forwards the FULL
# 6-mon preview (train == serve, nothing OOD) and subset scores are assembled from tables.
# Dispatch is BY CKPT STAMP; this constant is only the kill-switch (TP_JOINT_BRING pattern).
TP_SET_HEAD = True

# ── TP near-tie sampling (era-4 Phase 1a, 2026-07-20) ─────────────────────────
# The preview is ONE joint decision, so sampling here cannot break pair coherence (the
# N1 serve-tau refutation was about INDEPENDENT battle-slot heads). VD_TP_TIE_EPS > 0
# samples among subset/lead candidates within eps LOGITS of the top score (prob-ratio
# >= exp(-eps) of the best — checkpoint-scale-free), softmax-weighted. 0/unset = exact
# argmax, original code path. Audited habit this attacks: one lead pair 41/50 games at
# 19W-22L while the near-tie alternate went 6W-1L.
_TP_TIE_ECHOED = False


def _tp_tie_eps() -> float:
    global _TP_TIE_ECHOED
    try:
        eps = float(os.environ.get("VD_TP_TIE_EPS", "0") or 0.0)
    except ValueError:
        eps = 0.0
    if eps > 0 and not _TP_TIE_ECHOED:
        print(f"[tp] near-tie sampling ACTIVE: eps={eps}")   # flag-uptake echo (session log)
        _TP_TIE_ECHOED = True
    return eps


def _near_tie_sample(scores: np.ndarray, eps: float):
    """(index, deviated): argmax when eps<=0 or no tie; else a softmax-weighted draw
    among entries within eps logits of the max."""
    best = int(np.argmax(scores))
    if eps <= 0:
        return best, False
    cands = np.flatnonzero(scores >= scores[best] - eps)
    if len(cands) <= 1:
        return best, False
    w = np.exp(scores[cands] - scores[best])
    pick = int(np.random.choice(cands, p=w / w.sum()))
    return pick, pick != best


def _set_head_order(model, oi, of, pi, pf, aff_t, valid: int, bring_k: int, lead_k: int,
                    n: int, device: str, ctx_kw: Optional[dict] = None):
    """(order, True) via the contrastive set head, or (None, False) when not applicable
    (model has no set head / nothing to choose)."""
    from itertools import combinations
    if not getattr(model, "use_set_head", False):
        return None, False
    k = min(bring_k, valid, n)
    if k <= 0 or valid <= k:
        return None, False                       # ≤ one candidate subset → greedy is already exact
    subsets = list(combinations(range(valid), k))             # ≤ C(6,4)=15
    with torch.no_grad():
        scores, _bl, ll = model.score_subsets(
            torch.as_tensor([list(oi)], device=device),
            torch.as_tensor([list(pi)], device=device),
            torch.as_tensor(of[None], device=device),
            torch.as_tensor(pf[None], device=device),
            aff_t, subsets=subsets, **(ctx_kw or {}))
    scores = np.asarray(scores[0].detach().cpu())
    ll = np.asarray(ll[0].detach().cpu())
    eps = _tp_tie_eps()
    if eps <= 0:                                   # default: exact argmax, original path
        brought = list(subsets[int(np.argmax(scores))])
        leads = sorted(brought, key=lambda i: -ll[i])[:lead_k]  # lead decode unchanged (within bring)
        return (leads + [b for b in brought if b not in leads])[:n], True
    si, dev_s = _near_tie_sample(scores, eps)
    brought = list(subsets[si])
    pairs = list(combinations(brought, min(lead_k, len(brought)))) or [tuple(brought)]
    pair_scores = np.asarray([float(ll[list(p)].sum()) for p in pairs])
    pj, dev_l = _near_tie_sample(pair_scores, eps)
    leads = sorted(pairs[pj], key=lambda i: -ll[i])
    if dev_s or dev_l:
        print(f"[tp] near-tie deviation (eps={eps}): set={'+'.join(map(str, brought))} "
              f"leads={'+'.join(map(str, leads))} (set_dev={dev_s} lead_dev={dev_l})")
    return (list(leads) + [b for b in brought if b not in leads])[:n], True


def _joint_bring_order(model, oi, of, pi, pf, aff_t, valid: int, bring_k: int, lead_k: int,
                       n: int, device: str):
    """(order, True) via joint subset re-scoring, or (None, False) when not applicable
    (nothing to choose / a mean-pool legacy net whose context is not pad-safe)."""
    from itertools import combinations
    k = min(bring_k, valid, n)
    if k <= 0 or valid <= k:
        return None, False                       # ≤ one candidate subset → greedy is already exact
    if not (getattr(model, "use_self_attn", False) or getattr(model, "use_cross_attn", False)):
        return None, False                       # pad-safety needs the attention path (v6+ SBDA)
    subsets = list(combinations(range(valid), k))            # ≤ C(6,4)=15
    B = len(subsets)
    oi_b = np.tile(np.asarray(oi, dtype=np.int64), (B, 1))   # (B, 6)
    of_b = np.tile(of[None], (B, 1, 1))                      # (B, 6, F)
    for b, sub in enumerate(subsets):
        excl = [j for j in range(oi_b.shape[1]) if j not in sub]
        oi_b[b, excl] = 0
        of_b[b, excl] = 0.0                      # pad convention: idx 0 + all-zero feature row
    pi_b = np.tile(np.asarray(pi, dtype=np.int64), (B, 1))
    pf_b = np.tile(pf[None], (B, 1, 1))
    with torch.no_grad():
        bl, ll = model(torch.as_tensor(oi_b, device=device),
                       torch.as_tensor(pi_b, device=device),
                       torch.as_tensor(of_b, device=device),
                       torch.as_tensor(pf_b, device=device),
                       aff_t.expand(B, -1, -1) if aff_t is not None else None)
    bl = np.asarray(bl.detach().cpu())
    ll = np.asarray(ll.detach().cpu())
    scores = [float(bl[b, list(sub)].mean()) for b, sub in enumerate(subsets)]
    best = int(np.argmax(scores))
    brought = list(subsets[best])
    leads = sorted(brought, key=lambda i: -ll[best, i])[:lead_k]
    return (leads + [b for b in brought if b not in leads])[:n], True


def team_order(
    model, vocab: dict, cfg: dict,
    our_species: Sequence[str], opp_species: Sequence[str],
    n: int, device: str = "cpu", *,
    belief=None, own_known: Optional[dict] = None,
    opp_known: Optional[dict] = None,
    own_build: Optional[dict] = None,
    joint_bring: Optional[bool] = None,
    our_set_ctx: Optional[np.ndarray] = None,
    opp_set_ctx: Optional[np.ndarray] = None,
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
    ``opp_known`` (``{norm_species: OwnKnown}``) is the tpfeat-v7 OTS pathway: in
    open-team-sheet play the opponent's revealed builds ride the opp overlay —
    pass it only when sheets are actually visible; ``None`` = closed regime.
    ``own_build`` (``{norm_species: {"ability","item","moves"}}``) is the v8
    crisp-own-side pathway: our own mons' BASE channels are computed from the true
    build via ``OwnBuildBelief`` (p=1.0 marginals). IGNORED for v6/v7 checkpoints —
    they serve byte-identically through the frozen extractor.
    """
    valid = min(len(our_species), 6)
    if valid == 0:
        return list(range(n))
    feat_dim = cfg.get("feat_dim", 46)
    bring_k = int(cfg.get("bring_k", 4))
    lead_k = int(cfg.get("lead_k", 2))
    use_tp = uses_tp_features(cfg)

    own_side_belief = belief
    if (own_build and use_tp
            and (cfg or {}).get("feature_schema") not in _TP_LEGACY_SCHEMAS):
        own_side_belief = OwnBuildBelief(belief, own_build)

    oi, of = _pack_side(our_species, vocab, feat_dim,
                        belief=own_side_belief, own_known=own_known, use_tp_features=use_tp,
                        tp_schema=(cfg or {}).get("feature_schema"))
    pi, pf = _pack_side(opp_species, vocab, feat_dim,
                        belief=belief, own_known=opp_known, use_tp_features=use_tp,
                        tp_schema=(cfg or {}).get("feature_schema"))

    # 15b-feat.1b: feed the Pikalytics co-occurrence prior as the self-attention
    # bias when the SBDA net was built with ``use_teammate_bias``.  Needs a belief;
    # absent → ``our_affinity=None`` and the model runs without the prior (its
    # forward treats None as "no bias"), so serving stays correct, just un-primed.
    aff_t = None
    if use_tp and getattr(model, "use_teammate_bias", False) and belief is not None:
        from v_dance.training.tp_features import teammate_affinity_matrix
        aff = teammate_affinity_matrix(our_species, belief, n=6)
        aff_t = torch.as_tensor(aff[None], device=device)

    # DS-4c stage 3: Bo3 previous-game context (None off-set / for non-ctx ckpts —
    # zeros or absence are the zero-init identity either way).
    ctx_kw = {}
    if getattr(model, "use_set_ctx", False) and our_set_ctx is not None:
        ctx_kw = {
            "our_set_ctx": torch.as_tensor(np.asarray(our_set_ctx, dtype=np.float32)[None],
                                           device=device),
            "opp_set_ctx": torch.as_tensor(
                np.asarray(opp_set_ctx if opp_set_ctx is not None
                           else np.zeros_like(our_set_ctx), dtype=np.float32)[None],
                device=device),
        }

    # Contrastive set-head decode (2026-07-11): a --set-head checkpoint scores bring-SETS
    # with a head TRAINED on exactly that question — dispatched by its config stamp.
    if TP_SET_HEAD and getattr(model, "use_set_head", False):
        order, ok = _set_head_order(model, oi, of, pi, pf, aff_t,
                                    valid, bring_k, lead_k, n, device, ctx_kw=ctx_kw)
        if ok:
            return order

    # Joint conditional re-decode (2026-07-10, flag-gated): score bring-SETS, not mons.
    # Falls through to the greedy decode when disabled or not applicable.
    joint = TP_JOINT_BRING if joint_bring is None else bool(joint_bring)
    if joint and use_tp:
        order, ok = _joint_bring_order(model, oi, of, pi, pf, aff_t,
                                       valid, bring_k, lead_k, n, device)
        if ok:
            return order

    with torch.no_grad():
        bring_logits, lead_logits = model(
            torch.as_tensor([oi], device=device),
            torch.as_tensor([pi], device=device),
            torch.as_tensor(of[None], device=device),
            torch.as_tensor(pf[None], device=device),
            aff_t, **ctx_kw,
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
