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
import torch
import torch.nn.functional as F

from v_dance.encoders.action_codec import OPP_HEADS
from v_dance.selfplay.schema import PASS_ACTION, Transition


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


def _joint_kl(new_a, new_g, ref_a, ref_g, slot_batches, tau, gimmick_kl_weight: float = 1.0):
    """Sum the per-head forward KL(ref||new) over the slots that decided. ``gimmick_kl_weight`` scales the
    GIMMICK (mega) head's KL contribution ONLY (default 1.0 = byte-identical to the old joint KL). Set it
    < 1.0 to anchor the mega head to BC more LOOSELY than the move policy, so RL can un-learn the BC
    "always mega turn 1" data-bias while the move policy stays anchored. The heads are independent, so this
    is a clean per-head knob; it down-weights the mega head in BOTH the KL penalty AND the target_kl_bc
    drift-halt (so a mega-head drift won't trip the early stop — exactly what we want)."""
    B = slot_batches[0].acts.shape[0]
    total = torch.zeros(B, device=slot_batches[0].acts.device)
    z = torch.zeros(B, device=total.device)
    for sb in slot_batches:
        total = total + torch.where(sb.valid, masked_kl(ref_a[sb.head], new_a[sb.head], sb.amask, tau), z)
        if sb.gimmick:
            g_kl = gimmick_kl_weight * masked_kl(ref_g[sb.head], new_g[sb.head], sb.gmask, tau)
            total = total + torch.where(sb.has_g, g_kl, z)
    return total


# ── Level-A auxiliary opponent-action prediction (Piece 2) ────────────────────
def opp_aux_ce(action_logits, transitions, head_names, device: str = "cpu"):
    """UNMASKED cross-entropy of the auxiliary opponent-action heads (``opp_a``/``opp_b``)
    vs the recorded opp-action targets (Piece 3 ``opp_a_action`` / ``opp_b_action``).

    Only heads PRESENT in the model (i.e. after the Piece-1 anchor retrain with
    ``--aux-opp-head``) and slots with a VALID (non-PASS) target contribute. The recorded
    opp action always EXECUTED, so it is always legal → no legal mask is needed, and the
    aux heads are representation-only (never used to pick actions). Returns
    ``(opp_ce, n_valid)``: ``opp_ce`` is a 0-dim mean CE over the valid head-slot terms
    (carries gradient to the opp heads AND the shared trunk — the Level-A goal), or
    ``None`` when no opp head is present / the batch has no valid opp target."""
    present = [h for h in OPP_HEADS if h in action_logits and h in head_names]
    if not present:
        return None, 0
    total_ce, total_valid = None, 0
    for h in present:
        logits = action_logits[h]                                       # (B,A) with grad
        tgt = torch.as_tensor(
            np.array([getattr(t, h + "_action") for t in transitions], np.int64),
            device=device)
        valid = tgt != PASS_ACTION
        nv = int(valid.sum().item())
        if nv == 0:
            continue
        ce = F.cross_entropy(logits, tgt.clamp_min(0), reduction="none")  # (B,)
        s = torch.where(valid, ce, torch.zeros_like(ce)).sum()
        total_ce = s if total_ce is None else total_ce + s
        total_valid += nv
    if total_valid == 0:
        return None, 0
    return total_ce / total_valid, total_valid


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
    states = _states_tensor(transitions, device)
    action_logits, gimmick_logits, value_logit = ac(states)
    A = next(iter(action_logits.values())).shape[-1]
    G = next(iter(gimmick_logits.values())).shape[-1] if gimmick_logits else 0
    sb = _slot_batches(transitions, ac.head_names, ac.gimmick_head_names, A, G, device)
    lp, ent = _joint_logprob_entropy(action_logits, gimmick_logits, sb, tau)
    # C51: collection records the distribution MEAN as value_pm (still in [-1,1]); scalar keeps win-prob.
    if getattr(ac, "critic", None) is not None and ac.critic.is_c51:
        return lp, ent, ac.critic.value_pm(states)
    return lp, ent, 2.0 * torch.sigmoid(value_logit) - 1.0


@dataclass
class PPOEval:
    """Per-step quantities the PPO loss needs from one policy forward."""
    logprob: torch.Tensor      # (B,) new joint log-prob
    entropy: torch.Tensor      # (B,) summed per-head entropy
    value_pm: torch.Tensor     # (B,) critic value in [-1,1]
    kl_to_ref: Optional[torch.Tensor]  # (B,) KL(BC||new), or None if no reference given
    opp_ce: Optional[torch.Tensor] = None  # 0-dim aux opp-prediction CE, or None (no opp head / no target)
    atoms_logits: Optional[torch.Tensor] = None  # (B, n_atoms) C51 per-atom value logits, or None (scalar critic)


def ppo_forward(
    ac, transitions: List[Transition], tau: float = 1.0, device: str = "cpu",
    ref_policy=None, gimmick_kl_weight: float = 1.0,
) -> PPOEval:
    """One actor-critic forward (+ one frozen-reference forward if given) → the
    PPOEval bundle. Returned ``kl_to_ref`` is the exact per-step forward KL(BC||new)
    summed over decided heads — a logged diagnostic always, and the KL-to-BC penalty
    when ``cfg.kl_coef > 0`` (3b.3). At init (new == BC) it is exactly 0."""
    if not transitions:
        z = torch.zeros(0, device=device)
        return PPOEval(z, z, z, None if ref_policy is None else z)
    states = _states_tensor(transitions, device)
    # C51: a full ac(states) runs the critic trunk to produce a SCALAR value_logit that the C51 branch
    # below DISCARDS, then value_atoms_logits(states) runs that same trunk AGAIN — two critic passes for one
    # value. Run the ACTOR alone for the policy heads (action_logits are IDENTICAL — ac.forward delegates
    # them to self.policy) and the critic ONCE for the atoms. The scalar path keeps the single ac() call, so
    # it is byte-identical (audit 2026-06-30 — perf, C51-only).
    is_c51 = getattr(ac, "critic", None) is not None and ac.critic.is_c51
    if is_c51:
        action_logits, gimmick_logits, _ = ac.policy(states)   # actor only; the critic runs once below
    else:
        action_logits, gimmick_logits, value_logit = ac(states)
    A = next(iter(action_logits.values())).shape[-1]
    G = next(iter(gimmick_logits.values())).shape[-1] if gimmick_logits else 0
    sb = _slot_batches(transitions, ac.head_names, ac.gimmick_head_names, A, G, device)
    lp, ent = _joint_logprob_entropy(action_logits, gimmick_logits, sb, tau)
    # C51: the value baseline is the distribution MEAN and the loss needs the raw per-atom logits;
    # the scalar critic keeps the win-prob path. value_pm stays in [-1,1] either way.
    atoms_logits = None
    if is_c51:
        atoms_logits = ac.critic.value_atoms_logits(states)
        value_pm = (atoms_logits.softmax(dim=-1) * ac.critic.support).sum(dim=-1)
    else:
        value_pm = 2.0 * torch.sigmoid(value_logit) - 1.0

    kl = None
    if ref_policy is not None:
        ref_a, ref_g, _ = ref_policy(states)        # frozen BC reference logits
        kl = _joint_kl(action_logits, gimmick_logits, ref_a, ref_g, sb, tau, gimmick_kl_weight)
    opp_ce, _ = opp_aux_ce(action_logits, transitions, ac.head_names, device)
    return PPOEval(lp, ent, value_pm, kl, opp_ce=opp_ce, atoms_logits=atoms_logits)


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
