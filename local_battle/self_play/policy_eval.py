"""Per-head masked joint log-prob + entropy — task 3b.5.

THE canonical policy-distribution definition, shared by BOTH ends of PPO so the
importance ratio is exact:
  * collection (3c.1) records ``logprob`` = this function under the behaviour policy;
  * the PPO update (3b.3) recomputes ``logprob`` under the current policy.
Same masking, same temperature -> ``ratio = exp(new - old)`` is meaningful.

The policy factors over FOUR independent heads — slot-0 action, slot-1 action,
slot-0 gimmick, slot-1 gimmick — each a *masked* categorical (illegal actions get
zero probability via build_action_mask / build_gimmick_mask, recorded per step in
3b.5). The joint log-prob is the sum of the per-head terms over the slots that
actually decided:
  * a slot whose action is ``PASS_ACTION`` contributes nothing (no decision);
  * the gimmick term is included only for a slot that has a gimmick head AND a
    non-empty gimmick mask (an acting own slot). A forced-replacement step carries
    a switch-only action mask and a none-only / empty gimmick mask, so it degenerates
    correctly: the switch term is a normal masked categorical, the gimmick term is
    either ~0 (none-only) or skipped (empty) — covering normal turns AND replacements.

Value comes from the SEPARATE critic in ``value_pm`` space ([-1,1]) — the same space
as the terminal reward and the GAE baseline (see actor_critic / sec 3). Everything is
differentiable (no ``no_grad``): PPO backprops the policy loss through the actor heads
and the value loss through the critic.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT / "local_battle"), str(_REPO_ROOT / "data" / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn.functional as F

from self_play.schema import PASS_ACTION, Transition


# ── pure-tensor core (model-free, fully unit-testable) ────────────────────────
def masked_log_softmax(logits: torch.Tensor, legal: torch.Tensor,
                       tau: float = 1.0) -> torch.Tensor:
    """Log-probabilities of a masked categorical: ``log_softmax(logits/tau)`` with
    illegal entries driven to ~zero probability.

    ``logits`` (..., A) float, ``legal`` (..., A) bool. Illegal entries are filled
    with ``finfo.min`` (a large finite negative — NOT -inf) so an all-illegal row
    yields a finite uniform distribution instead of NaN (such rows only occur for a
    PASS / empty slot, whose term is masked out by the caller anyway)."""
    neg = torch.finfo(logits.dtype).min
    z = (logits / tau).masked_fill(~legal, neg)
    return F.log_softmax(z, dim=-1)


def categorical_logprob_entropy(
    logits: torch.Tensor, legal: torch.Tensor, actions: torch.Tensor, tau: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(logprob, entropy) of a batched masked categorical.

    ``logits`` (B,A), ``legal`` (B,A) bool, ``actions`` (B,) long. Returns logprob
    (B,) = log pi(a) and entropy (B,) = -sum_legal p*log p. ``actions`` is clamped to
    >=0 before the gather (a PASS sentinel -1 gathers a harmless index 0; the caller
    zeroes that slot's contribution)."""
    log_p = masked_log_softmax(logits, legal, tau)                 # (B,A)
    lp = log_p.gather(1, actions.clamp_min(0).unsqueeze(1)).squeeze(1)  # (B,)
    p = log_p.exp()
    ent = -torch.where(legal, p * log_p, torch.zeros_like(p)).sum(-1)   # (B,)
    return lp, ent


def masked_kl(logits_p: torch.Tensor, logits_q: torch.Tensor,
              legal: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """Forward KL ``KL(p || q)`` of two masked categoricals over the SAME legal set,
    per row (B,). ``p`` = reference (BC), ``q`` = current policy. Mode-covering: it
    heavily penalises ``q`` driving toward 0 where ``p`` is large (i.e. crushing a BC
    mode — the §12 'don't crush the BC behaviours early' goal). Gradient flows through
    ``logits_q`` only (``p`` is the frozen reference); illegal entries contribute 0."""
    log_p = masked_log_softmax(logits_p, legal, tau)
    log_q = masked_log_softmax(logits_q, legal, tau)
    p = log_p.exp()
    return torch.where(legal, p * (log_p - log_q), torch.zeros_like(p)).sum(-1)


# ── transition-batch evaluation (the PPO / collection entry point) ────────────
_SLOTS = (
    ("action_s0", "gimmick_s0", "mask_s0", "gmask_s0"),   # slot 0 -> head_names[0]
    ("action_s1", "gimmick_s1", "mask_s1", "gmask_s1"),   # slot 1 -> head_names[1]
)


def _stack_action_masks(transitions, attr, dim, device) -> torch.Tensor:
    """(B,dim) bool. A missing (None) action mask -> all-legal (full softmax); real
    RL trajectories always carry the mask, so this fallback only affects maskless
    test / legacy steps."""
    rows = [np.ones(dim, np.float32) if getattr(t, attr) is None
            else np.asarray(getattr(t, attr), np.float32) for t in transitions]
    return torch.as_tensor(np.stack(rows), device=device) > 0.5


def _stack_gimmick_masks(transitions, attr, dim, device):
    """(mask (B,dim) bool, present (B,) bool). A missing or all-zero gimmick mask ->
    present=False, so that slot's gimmick term is skipped (no gimmick decision)."""
    rows, present = [], []
    for t in transitions:
        m = getattr(t, attr)
        if m is None:
            rows.append(np.zeros(dim, np.float32)); present.append(False)
        else:
            a = np.asarray(m, np.float32); rows.append(a); present.append(bool(a.sum() > 0))
    mask = torch.as_tensor(np.stack(rows), device=device) > 0.5
    return mask, torch.as_tensor(present, device=device)


@dataclass
class _SlotBatch:
    """Per-slot decoded tensors for a transition batch (built once, reused by the
    log-prob / entropy / KL accumulators)."""
    head: str
    acts: torch.Tensor       # (B,) long action indices (PASS_ACTION = -1)
    valid: torch.Tensor      # (B,) bool — slot decided (not PASS)
    amask: torch.Tensor      # (B,A) bool legal action mask
    gims: torch.Tensor       # (B,) long gimmick indices
    gmask: torch.Tensor      # (B,G) bool legal gimmick mask
    has_g: torch.Tensor      # (B,) bool — decided AND a real gimmick decision exists
    gimmick: bool            # this slot has a gimmick head


def _slot_batches(transitions, head_names, gimmick_head_names, A, G, device):
    out = []
    for slot, (a_attr, g_attr, m_attr, gm_attr) in enumerate(_SLOTS):
        head = head_names[slot]
        acts = torch.as_tensor(np.array([getattr(t, a_attr) for t in transitions], np.int64),
                               device=device)
        valid = acts != PASS_ACTION
        amask = _stack_action_masks(transitions, m_attr, A, device)
        has_gimmick = head in gimmick_head_names and G > 0
        if has_gimmick:
            gims = torch.as_tensor(np.array([getattr(t, g_attr) for t in transitions], np.int64),
                                   device=device)
            gmask, gpresent = _stack_gimmick_masks(transitions, gm_attr, G, device)
            has_g = valid & gpresent
        else:
            gims = torch.zeros(len(transitions), dtype=torch.int64, device=device)
            gmask = torch.zeros(len(transitions), max(G, 1), dtype=torch.bool, device=device)
            has_g = torch.zeros(len(transitions), dtype=torch.bool, device=device)
        out.append(_SlotBatch(head, acts, valid, amask, gims, gmask, has_g, has_gimmick))
    return out


def _joint_logprob_entropy(action_logits, gimmick_logits, slot_batches, tau):
    """Sum the per-head masked log-prob + entropy over the slots that decided (PASS
    slots and absent gimmick decisions contribute 0)."""
    B = slot_batches[0].acts.shape[0]
    total_lp = torch.zeros(B, device=slot_batches[0].acts.device)
    total_ent = torch.zeros(B, device=total_lp.device)
    z = torch.zeros(B, device=total_lp.device)
    for sb in slot_batches:
        lp, ent = categorical_logprob_entropy(action_logits[sb.head], sb.amask, sb.acts, tau)
        total_lp = total_lp + torch.where(sb.valid, lp, z)
        total_ent = total_ent + torch.where(sb.valid, ent, z)
        if sb.gimmick:
            glp, gent = categorical_logprob_entropy(gimmick_logits[sb.head], sb.gmask, sb.gims, tau)
            total_lp = total_lp + torch.where(sb.has_g, glp, z)
            total_ent = total_ent + torch.where(sb.has_g, gent, z)
    return total_lp, total_ent


def _joint_kl(new_a, new_g, ref_a, ref_g, slot_batches, tau):
    """Sum the per-head forward KL(ref||new) over the slots that decided."""
    B = slot_batches[0].acts.shape[0]
    total = torch.zeros(B, device=slot_batches[0].acts.device)
    z = torch.zeros(B, device=total.device)
    for sb in slot_batches:
        total = total + torch.where(sb.valid, masked_kl(ref_a[sb.head], new_a[sb.head], sb.amask, tau), z)
        if sb.gimmick:
            total = total + torch.where(sb.has_g, masked_kl(ref_g[sb.head], new_g[sb.head], sb.gmask, tau), z)
    return total


def _states_tensor(transitions, device):
    return torch.as_tensor(np.stack([np.asarray(t.state, np.float32) for t in transitions]),
                           device=device)


def evaluate_actions(
    ac, transitions: List[Transition], tau: float = 1.0, device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the actor-critic over a batch of transitions and return
    ``(logprob, entropy, value_pm)``, each (B,). This is the COLLECTION-side entry
    point (3c.1 records ``logprob`` from it); the PPO trainer uses ``ppo_forward``.

      * ``logprob``  — joint masked log-prob of each step's recorded (a0,g0,a1,g1);
      * ``entropy``  — summed per-head entropy (for the PPO entropy bonus);
      * ``value_pm`` — the SEPARATE critic's value in [-1,1].

    Differentiable: policy terms carry gradient to the actor heads, ``value_pm`` to
    the critic. ``tau`` is the policy temperature and MUST match what collection used."""
    if not transitions:
        z = torch.zeros(0, device=device)
        return z, z, z
    action_logits, gimmick_logits, value_logit = ac(_states_tensor(transitions, device))
    A = next(iter(action_logits.values())).shape[-1]
    G = next(iter(gimmick_logits.values())).shape[-1] if gimmick_logits else 0
    sb = _slot_batches(transitions, ac.head_names, ac.gimmick_head_names, A, G, device)
    lp, ent = _joint_logprob_entropy(action_logits, gimmick_logits, sb, tau)
    return lp, ent, 2.0 * torch.sigmoid(value_logit) - 1.0


@dataclass
class PPOEval:
    """Per-step quantities the PPO loss needs from one policy forward."""
    logprob: torch.Tensor      # (B,) new joint log-prob
    entropy: torch.Tensor      # (B,) summed per-head entropy
    value_pm: torch.Tensor     # (B,) critic value in [-1,1]
    kl_to_ref: Optional[torch.Tensor]  # (B,) KL(BC||new), or None if no reference given


def ppo_forward(
    ac, transitions: List[Transition], tau: float = 1.0, device: str = "cpu",
    ref_policy=None,
) -> PPOEval:
    """One actor-critic forward (+ one frozen-reference forward if given) → the
    PPOEval bundle. Returned ``kl_to_ref`` is the exact per-step forward KL(BC||new)
    summed over decided heads — a logged diagnostic always, and the KL-to-BC penalty
    when ``cfg.kl_coef > 0`` (3b.3). At init (new == BC) it is exactly 0."""
    if not transitions:
        z = torch.zeros(0, device=device)
        return PPOEval(z, z, z, None if ref_policy is None else z)
    states = _states_tensor(transitions, device)
    action_logits, gimmick_logits, value_logit = ac(states)
    A = next(iter(action_logits.values())).shape[-1]
    G = next(iter(gimmick_logits.values())).shape[-1] if gimmick_logits else 0
    sb = _slot_batches(transitions, ac.head_names, ac.gimmick_head_names, A, G, device)
    lp, ent = _joint_logprob_entropy(action_logits, gimmick_logits, sb, tau)
    value_pm = 2.0 * torch.sigmoid(value_logit) - 1.0

    kl = None
    if ref_policy is not None:
        ref_a, ref_g, _ = ref_policy(states)        # frozen BC reference logits
        kl = _joint_kl(action_logits, gimmick_logits, ref_a, ref_g, sb, tau)
    return PPOEval(lp, ent, value_pm, kl)


# ── data-integrity guard (mirrors corpus_qa's "illegal under mask") ───────────
def assert_actions_legal(transitions: List[Transition]) -> None:
    """Every recorded non-PASS action / present gimmick must be LEGAL under its
    stored mask. A violation means a masked log-prob of -inf (ratio blows up) — a
    corruption alarm, not a recoverable state. No-op for steps without stored masks."""
    for i, t in enumerate(transitions):
        for a_attr, g_attr, m_attr, gm_attr in _SLOTS:
            a = getattr(t, a_attr)
            m = getattr(t, m_attr)
            if a != PASS_ACTION and m is not None:
                assert 0 <= a < len(m) and m[a] == 1, \
                    f"transition {i}: {a_attr}={a} illegal under {m_attr}={list(m)}"
                gm = getattr(t, gm_attr)
                g = getattr(t, g_attr)
                if gm is not None and any(gm):
                    assert 0 <= g < len(gm) and gm[g] == 1, \
                        f"transition {i}: {g_attr}={g} illegal under {gm_attr}={list(gm)}"
