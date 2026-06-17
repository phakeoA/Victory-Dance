"""GAE advantage + return computation for PPO (task 3b.2 — the core of 3b).

Turns a finished ``Trajectory`` (rewards + collection-time values + the terminal/
bootstrap flag from 3a) into per-step advantages and value targets. Pure numpy — no
torch, no live model — so the PPO objective math is fully unit-testable offline.

Invariants (docs/ppo_reward_design.md sec 3 / sec 13):
  - gamma = 0.997 FLOOR (never lower — it silently suppresses Perish/TR/long-stall
    wins). lambda = 0.95.
  - A real terminal (win/loss/draw) bootstraps with V(s_T)=0. A HORIZON_CUT is a
    TRUNCATION, not a termination -> bootstrap gamma*V(s_cut) (NEVER a +-1 terminal).
  - Advantages are standardized to zero-mean/unit-std PER BATCH (affine, sign-safe),
    with BOTH perspectives of a game in the same batch (the caller concatenates).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from self_play.schema import Trajectory

DEFAULT_GAMMA = 0.997   # sec 13 FLOOR — do not lower
DEFAULT_LAM = 0.95


def compute_gae(
    traj: Trajectory,
    gamma: float = DEFAULT_GAMMA,
    lam: float = DEFAULT_LAM,
    bootstrap_value: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(advantages, returns)`` for one trajectory (one perspective).

    ``bootstrap_value`` = V(s_cut), the critic value of the state AFTER the last action,
    used ONLY for a HORIZON_CUT truncation. If None on a truncated episode we fall back
    to the last recorded value as a proxy (off-by-one but fine — the cut is set above
    the 99th-pct game length so it almost never fires). Ignored for real terminals
    (they bootstrap with 0). ``returns = advantages + values`` (the critic target)."""
    ts = traj.transitions
    n = len(ts)
    if n == 0:
        empty = np.zeros(0, dtype=np.float32)
        return empty, empty.copy()

    rewards = np.array([t.reward for t in ts], dtype=np.float64)
    values = np.array([t.value for t in ts], dtype=np.float64)

    if traj.meta.bootstraps:                       # HORIZON_CUT truncation -> bootstrap
        last_next_value = float(bootstrap_value) if bootstrap_value is not None else float(values[-1])
        last_nonterminal = 1.0
    else:                                          # real terminal -> V(terminal) = 0
        last_next_value = 0.0
        last_nonterminal = 0.0

    adv = np.zeros(n, dtype=np.float64)
    gae = 0.0
    for t in range(n - 1, -1, -1):
        if t == n - 1:
            next_value, nonterminal = last_next_value, last_nonterminal
        else:
            next_value, nonterminal = values[t + 1], 1.0   # mid-episode steps never terminal
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae

    returns = adv + values
    return adv.astype(np.float32), returns.astype(np.float32)


def standardize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Zero-mean / unit-std normalization (the standard PPO advantage trick). Affine in
    the advantage, so it does not change the policy-gradient sign. ``eps`` floors the
    std so a near-constant batch (e.g. no near-terminal steps) doesn't blow up."""
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    return (x - x.mean()) / (x.std() + eps)


def compute_batch_gae(
    trajectories: List[Trajectory],
    gamma: float = DEFAULT_GAMMA,
    lam: float = DEFAULT_LAM,
    standardize_adv: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """GAE over a batch: per-trajectory advantages/returns, concatenated in order, with
    advantages standardized ACROSS THE WHOLE BATCH (sec 3 — pass both perspectives of
    each game in the same batch so the normalization scales both sides identically).
    Returns ``(advantages, returns)`` as flat arrays aligned to the concatenated steps.
    Empty/zero-length trajectories contribute nothing."""
    advs: List[np.ndarray] = []
    rets: List[np.ndarray] = []
    for tr in trajectories:
        a, r = compute_gae(tr, gamma, lam)
        if a.size:
            advs.append(a)
            rets.append(r)
    if not advs:
        empty = np.zeros(0, dtype=np.float32)
        return empty, empty.copy()
    advantages = np.concatenate(advs)
    returns = np.concatenate(rets)
    if standardize_adv:
        advantages = standardize(advantages)
    return advantages, returns
