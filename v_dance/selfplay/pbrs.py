"""Gated potential-based reward shaping (PBRS) — task 3b.7. DEFAULT OFF.

Phase-3-only, added ONLY on a measured stall (docs/ppo_reward_design.md sec 4/8). When
on, shaping is PBRS exclusively: ``F_t = gamma*Phi(s_{t+1}) - Phi(s_t)`` with Phi a
function of STATE ONLY. By Ng/Harada/Russell (1999) Thm 1 the discounted shaping return
telescopes to the policy-independent constant ``-Phi(s_0)`` (with ``Phi(terminal)=0``),
so the optimum is provably unchanged — shaping changes learning SPEED, not the optimum.

Phi = a FROZEN snapshot of the calibrated value head: ``Phi(s) = c*(2*sigmoid(logit)-1)``,
c ∈ [0.1,0.3]. It is a SEPARATE nn.Module (``id != critic``), ``.eval()`` +
``requires_grad_(False)`` — a policy-gradient step can never move it, and it contributes
no gradient.

Invariants enforced/tested (sec 5/13):
  * **Phi(terminal) := 0** hard rule. This kills the outcome-correlated ``gamma*Phi(s_T)``
    term on the terminal transition (otherwise ~±1.25·c of dense, win-correlated reward
    leaks in). NOTE: the terminal transition STILL gets its ``-lambda*Phi(s_{n-1})`` PBRS
    term — that term is REQUIRED for the telescoping invariance; dropping it (to make the
    terminal reward literally unchanged) would BREAK invariance. So "shaped_r == terminal_r"
    is realised as "Phi(terminal)=0 ⇒ no gamma*Phi(s_T) inflation", verified by
    ``test``s here + the telescoping residual ≈ 0 check.
  * **Telescoping residual** ``sum_t gamma^t F_t + Phi(s_0) ≈ 0`` per episode.
  * **Shaping-fraction cap** ``mean(|sum lambda*F|)/mean(|terminal R|) < 0.3`` and ``-> 0``
    as lambda anneals; ``lambda -> 0`` over the final ~30-40% of training so the shipped
    policy converges to and is judged on the pure ±1 objective.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
import copy

import torch
import torch.nn as nn

from v_dance.selfplay.gae import DEFAULT_GAMMA


@dataclass
class PBRSConfig:
    enabled: bool = False            # GATED: default OFF (Phase 3 only, on measured stall)
    coef: float = 0.2                # c in Phi = c*(2*sigmoid-1), 0.1-0.3
    lam: float = 1.0                 # shaping weight lambda (anneal -> 0)
    shaping_fraction_cap: float = 0.3


class PotentialShaper(nn.Module):
    """Frozen Phi: a separate snapshot of ``trunk + value_head`` scaled by ``coef``.
    ``Phi(s) = coef*(2*sigmoid(value_logit) - 1)`` ∈ [-coef, coef]; ``Phi(terminal) = 0``."""

    def __init__(self, trunk: nn.Module, value_head: nn.Module, coef: float = 0.2,
                 device: str = "cpu"):
        super().__init__()
        self.trunk = trunk
        self.value_head = value_head
        self.coef = float(coef)
        self.device = device
        self.to(device).eval()
        for p in self.parameters():
            p.requires_grad_(False)

    @classmethod
    def from_critic(cls, critic, coef: float = 0.2, device: str = "cpu") -> "PotentialShaper":
        """Snapshot Phi from the (BC-calibrated) critic — a DEEP COPY, so Phi is frozen
        and independent of the critic that keeps training (sec 4)."""
        shaper = cls(copy.deepcopy(critic.trunk), copy.deepcopy(critic.value_head), coef, device)
        assert id(shaper) != id(critic), "Phi must be a SEPARATE module from the critic (sec 5)"
        return shaper

    @torch.no_grad()
    def phi(self, states) -> torch.Tensor:
        t = states if torch.is_tensor(states) else torch.as_tensor(
            np.asarray(states, np.float32), device=self.device)
        z = self.value_head(self.trunk(t)).squeeze(-1)
        return self.coef * (2.0 * torch.sigmoid(z) - 1.0)

    def phi_values(self, states) -> np.ndarray:
        return self.phi(states).cpu().numpy().astype(np.float32)

    @staticmethod
    def phi_terminal() -> float:
        """Hard rule: the potential of any terminal state is 0 (sec 5)."""
        return 0.0


def _potential_and_F(traj, shaper: PotentialShaper, gamma: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(phi, F)`` for a trajectory: ``phi[t] = Phi(s_t)`` and
    ``F[t] = gamma*Phi(s_{t+1}) - Phi(s_t)`` with ``Phi(s_terminal) = 0`` for the last step.
    Read-only (does NOT modify rewards)."""
    ts = traj.transitions
    n = len(ts)
    if n == 0:
        e = np.zeros(0, np.float32)
        return e, e.copy()
    phi = shaper.phi_values(np.stack([np.asarray(t.state, np.float32) for t in ts]))
    F = np.zeros(n, np.float32)
    for t in range(n):
        phi_next = phi[t + 1] if t < n - 1 else PotentialShaper.phi_terminal()  # Phi(terminal)=0
        F[t] = gamma * phi_next - phi[t]
    return phi, F


def shape_trajectory(traj, shaper: PotentialShaper, gamma: float = DEFAULT_GAMMA,
                     lam: float = 1.0) -> np.ndarray:
    """Add ``lambda*F_t`` to each transition's reward IN PLACE; return the ``F`` array.
    Apply ONCE per trajectory (it mutates rewards)."""
    _phi, F = _potential_and_F(traj, shaper, gamma)
    for t, transition in enumerate(traj.transitions):
        transition.reward = float(transition.reward + lam * F[t])
    return F


def telescoping_residual(traj, shaper: PotentialShaper, gamma: float = DEFAULT_GAMMA) -> float:
    """``sum_t gamma^t F_t + Phi(s_0)`` — must be ≈ 0 (PBRS telescopes to -Phi(s_0))."""
    phi, F = _potential_and_F(traj, shaper, gamma)
    if F.size == 0:
        return 0.0
    disc = float(np.sum([(gamma ** t) * F[t] for t in range(len(F))]))
    return disc + float(phi[0])


def shaping_fraction(trajectories, shaper: PotentialShaper, gamma: float = DEFAULT_GAMMA,
                     lam: float = 1.0) -> float:
    """``mean_episode(|sum_t lambda*F_t|) / mean_episode(|terminal R|)`` (sec 5). Decided
    episodes have |terminal R| = 1. Returns 0 when there is no terminal magnitude."""
    nums, dens = [], []
    for tr in trajectories:
        _phi, F = _potential_and_F(tr, shaper, gamma)
        if F.size == 0:
            continue
        nums.append(abs(lam * float(F.sum())))
        term = tr.meta.outcome_reward()
        dens.append(abs(term) if term is not None else 0.0)
    if not nums:
        return 0.0
    mean_den = float(np.mean(dens))
    return float(np.mean(nums) / mean_den) if mean_den > 1e-9 else 0.0


def assert_phi_terminal_zero(shaper: PotentialShaper) -> None:
    assert PotentialShaper.phi_terminal() == 0.0, "Phi(terminal) must be hard-zero (sec 5)"


def assert_shaping_fraction(trajectories, shaper: PotentialShaper, gamma: float = DEFAULT_GAMMA,
                            lam: float = 1.0, cap: float = 0.3) -> float:
    frac = shaping_fraction(trajectories, shaper, gamma, lam)
    assert frac < cap, (
        f"shaping fraction {frac:.3f} >= cap {cap} — PBRS would dominate the ±1 terminal "
        f"objective (sec 5). Lower coef or lambda.")
    return frac


def maybe_shape_batch(trajectories, shaper: PotentialShaper, cfg: PBRSConfig,
                      gamma: float = DEFAULT_GAMMA) -> dict:
    """Apply gated PBRS to a batch. No-op (and reports ``applied=False``) when
    ``cfg.enabled`` is False — the default. When on, it enforces the shaping-fraction
    cap, then adds ``lambda*F`` to every trajectory's rewards (before GAE)."""
    if not cfg.enabled:
        return {"applied": False}
    frac = assert_shaping_fraction(trajectories, shaper, gamma, cfg.lam, cfg.shaping_fraction_cap)
    for tr in trajectories:
        shape_trajectory(tr, shaper, gamma, cfg.lam)
    return {"applied": True, "shaping_fraction": frac, "lam": cfg.lam}
