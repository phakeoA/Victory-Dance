"""PPO losses — clip + value + entropy + KL-to-BC — task 3b.3.

The PPO objective for the actor-critic, kept as a PURE-tensor loss (``ppo_losses``)
so it is hand-checkable, plus a from-batch orchestration (``ppo_loss_from_batch``)
that runs the policy forwards and assembles the inputs. The optimiser / epoch loop /
critic-only warm-up is task 3b.4.

Pieces (docs/ppo_reward_design.md sec 3-5):
  * **Policy clip** — ``ratio = exp(new_logprob - old_logprob)``,
    ``-mean(min(ratio·A, clip(ratio,1±eps)·A))`` over PER-MINIBATCH-standardised
    advantages A (sec 3).
  * **Value** — Phase 1: BCE in win-prob space on the bootstrapped GAE return
    ``(return_pm+1)/2``; Phase 3: Huber on the shaped return. **Value-clip ON** (sec 3):
    the new value is clipped to within ``value_clip`` of the OLD value in the single
    asserted ``value_pm`` space, taking the pessimistic ``max`` of clipped/unclipped.
  * **Entropy bonus** — ``-entropy_coef·mean(entropy)`` (per-head entropy from
    policy_eval) to keep exploration alive.
  * **KL-to-BC prior** — ``+kl_coef·mean(KL(BC||new))`` against a FROZEN BC reference
    (``make_reference_policy``), preserving the BC prior so PPO can't crush the rare
    but correct BC behaviours early (sec 2/12). Always logged; penalised only when
    ``kl_coef > 0``.

ONE value space everywhere — ``value_pm`` ∈ [-1,1], same as the ±1 terminal reward and
the GAE baseline (the §3 / 3b.6 invariant). ``returns`` and ``old_value`` arrive in
that space; the BCE form maps to win-prob purely for the pointwise loss.
"""
from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
import torch
import torch.nn.functional as F

from v_dance.selfplay import policy_eval


@dataclass
class PPOConfig:
    clip_eps: float = 0.2          # PPO policy-ratio clip
    value_clip: float = 0.2        # value-clip range in value_pm space (sec 3)
    value_coef: float = 0.5        # value-loss weight
    entropy_coef: float = 0.01     # entropy bonus weight
    kl_coef: float = 0.0           # KL-to-BC penalty weight (0 = log-only; Phase 1 sets >0)
    gimmick_kl_weight: float = 1.0 # per-head KL scale for the MEGA (gimmick) head ONLY (1.0 = same anchor
                                   # as the move policy). <1.0 loosens the mega anchor so RL can un-learn
                                   # the BC "always mega turn 1" data-bias (see policy_eval._joint_kl).
    opp_aux_coef: float = 0.0      # Level-A aux opponent-prediction CE weight (0 = OFF). When >0 AND the
                                   # model carries opp_a/opp_b heads (Piece-1 anchor), adds a supervised CE
                                   # of those heads vs the recorded opp action (Piece 3) to the loss, so the
                                   # shared trunk becomes opponent-predictive. Phase-1 sets ~0.3.
    value_loss_mode: str = "bce"   # "bce" (Phase 1, win-prob) | "huber" (Phase 3, shaped return)
    huber_delta: float = 1.0
    standardize_adv: bool = True   # per-minibatch advantage standardisation (sec 3)
    adv_eps: float = 1e-8
    tau: float = 1.0               # policy temperature (MUST match collection)


def make_reference_policy(ac, device: str = "cpu"):
    """A FROZEN deep-copy of the actor's BC policy, for the KL-to-BC term. Built once
    at the start of training (the actor still == BC then, so this captures the
    original BC distribution). ``eval`` + ``requires_grad_(False)`` so it never trains
    and never contributes gradient."""
    ref = copy.deepcopy(ac.policy).to(device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


def _standardize_adv(adv: torch.Tensor, eps: float) -> torch.Tensor:
    """Zero-mean / unit-std over the minibatch (affine → symmetry-safe; sec 3).
    Advantages are constants (no grad)."""
    adv = adv.detach()
    if adv.numel() <= 1:
        return adv - adv.mean() if adv.numel() else adv
    return (adv - adv.mean()) / (adv.std(unbiased=False) + eps)


def _value_loss(value_pm: torch.Tensor, old_value_pm: torch.Tensor,
                returns: torch.Tensor, cfg: PPOConfig, clip: bool = True) -> torch.Tensor:
    """Critic loss in the single ``value_pm`` space; pointwise loss is BCE (win-prob,
    Phase 1) or Huber (shaped return, Phase 3).

    ``clip=True`` (the PPO update): value-clip ON — the new value is clipped to within
    ``value_clip`` of the OLD value (pessimistic ``max``), a per-update trust region.

    ``clip=False`` (the critic-only WARM-UP, 3b.4): value-clip OFF. The clip is a trust
    region anchored on the CURRENT-rollout value, but the warm-up regresses repeatedly on
    a FIXED cold-start anchor (the immutable collection-time value), so clipping there
    would freeze V within ±value_clip of the BC value and DEFEAT the migration to the
    gamma-discounted ±1 return scale the warm-up exists to perform (sec 2)."""
    target = returns.detach()
    old = old_value_pm.detach()

    def _pointwise(v: torch.Tensor) -> torch.Tensor:
        if cfg.value_loss_mode == "huber":
            return F.huber_loss(v, target, delta=cfg.huber_delta, reduction="none")
        if cfg.value_loss_mode == "bce":
            e = 1e-6
            p = ((v + 1.0) * 0.5).clamp(e, 1 - e)
            p_tgt = ((target + 1.0) * 0.5).clamp(e, 1 - e)
            return F.binary_cross_entropy(p, p_tgt, reduction="none")
        raise ValueError(f"unknown value_loss_mode {cfg.value_loss_mode!r}")

    l_un = _pointwise(value_pm)
    if not clip:
        return l_un.mean()
    v_clip = old + (value_pm - old).clamp(-cfg.value_clip, cfg.value_clip)
    return torch.max(l_un, _pointwise(v_clip)).mean()


# Public alias for the critic-only warm-up (3b.4), which optimises this loss alone.
value_loss = _value_loss


def ppo_losses(
    *, new_logprob: torch.Tensor, old_logprob: torch.Tensor, advantages: torch.Tensor,
    value_pm: torch.Tensor, old_value_pm: torch.Tensor, returns: torch.Tensor,
    entropy: torch.Tensor, kl_to_ref: Optional[torch.Tensor] = None,
    opp_ce: Optional[torch.Tensor] = None,
    cfg: PPOConfig = PPOConfig(),
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """The pure PPO loss. All tensors are (B,). ``new_logprob``/``entropy``/``kl`` carry
    actor gradient; ``value_pm`` carries critic gradient; ``old_*``/``advantages``/
    ``returns`` are constants. Returns ``(total_loss, stats)``."""
    if new_logprob.numel() == 0:
        raise ValueError("ppo_losses: empty minibatch")
    adv = _standardize_adv(advantages, cfg.adv_eps) if cfg.standardize_adv else advantages.detach()

    ratio = (new_logprob - old_logprob.detach()).exp()
    surr1 = ratio * adv
    surr2 = ratio.clamp(1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv
    policy_loss = -torch.min(surr1, surr2).mean()

    value_loss = _value_loss(value_pm, old_value_pm, returns, cfg)
    entropy_mean = entropy.mean()
    entropy_loss = -entropy_mean

    loss = policy_loss + cfg.value_coef * value_loss + cfg.entropy_coef * entropy_loss
    kl_mean = None
    if kl_to_ref is not None:
        kl_mean = kl_to_ref.mean()
        if cfg.kl_coef > 0.0:
            loss = loss + cfg.kl_coef * kl_mean

    # Level-A aux opponent-prediction CE (already a 0-dim mean over valid opp head-slots).
    # Carries gradient to the opp heads + shared trunk; penalised only when opp_aux_coef > 0
    # AND the model has opp heads (opp_ce None otherwise). Logged always.
    if opp_ce is not None and cfg.opp_aux_coef > 0.0:
        loss = loss + cfg.opp_aux_coef * opp_ce

    with torch.no_grad():
        clip_frac = ((ratio - 1.0).abs() > cfg.clip_eps).float().mean()
        approx_kl_old_new = (old_logprob - new_logprob).mean()   # ~KL(old||new)
        stats = {
            "loss": float(loss),
            "policy_loss": float(policy_loss),
            "value_loss": float(value_loss),
            "entropy": float(entropy_mean),
            "kl_to_bc": float(kl_mean) if kl_mean is not None else float("nan"),
            "opp_ce": float(opp_ce) if opp_ce is not None else float("nan"),
            "approx_kl_old_new": float(approx_kl_old_new),
            "clip_fraction": float(clip_frac),
            "ratio_mean": float(ratio.mean()),
            "adv_mean": float(adv.mean()),
            "adv_std": float(adv.std(unbiased=False)) if adv.numel() > 1 else 0.0,
        }
    return loss, stats


def _col(transitions, attr, device):
    return torch.as_tensor(np.array([getattr(t, attr) for t in transitions], np.float32),
                           device=device)


def ppo_loss_from_batch(
    ac, transitions, advantages, returns, *,
    cfg: PPOConfig = PPOConfig(), ref_policy=None, device: str = "cpu",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Orchestrate one minibatch: run ``ppo_forward`` (new logprob/entropy/value + KL),
    pull the stored old logprob/value, and assemble the PPO loss. ``advantages`` and
    ``returns`` come from ``gae.compute_batch_gae`` (numpy, value_pm space — pass them
    UN-standardised; standardisation happens here per-minibatch)."""
    ev = policy_eval.ppo_forward(ac, list(transitions), cfg.tau, device, ref_policy=ref_policy,
                                 gimmick_kl_weight=cfg.gimmick_kl_weight)
    old_logprob = _col(transitions, "logprob", device)
    old_value_pm = _col(transitions, "value", device)
    adv = torch.as_tensor(np.asarray(advantages, np.float32), device=device)
    ret = torch.as_tensor(np.asarray(returns, np.float32), device=device)
    return ppo_losses(
        new_logprob=ev.logprob, old_logprob=old_logprob, advantages=adv,
        value_pm=ev.value_pm, old_value_pm=old_value_pm, returns=ret,
        entropy=ev.entropy, kl_to_ref=ev.kl_to_ref, opp_ce=ev.opp_ce, cfg=cfg,
    )
