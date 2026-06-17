"""Single value-space invariant — task 3b.6.

The #1 silent PPO bug is a value-space mismatch (docs/ppo_reward_design.md sec 3). The
ONE space used everywhere is ``value_pm = 2*sigmoid(logit) - 1`` ∈ [-1,1] — the same
axis as the ±1 terminal reward, so ``adv = r + gamma*V' - V`` is consistent. The
critic's win-prob [0,1] parameterisation is an INTERNAL detail of the BCE value loss;
the value STORED in ``Transition.value``, the GAE baseline, the GAE returns, the
horizon-cut bootstrap, and the critic regression target are ALL ``value_pm``.

THE INSIDIOUS MISTAKE is storing the win-prob ([0,1]) instead of ``value_pm`` ([-1,1])
in ``Transition.value``: a [0,1] number PASSES a [-1,1] range check (it's a subset!).
So a pure range assert cannot catch it. ``looks_like_winprob`` — a distribution
heuristic over a balanced self-play batch, which MUST contain losing (negative-value)
states — is the real catch, and belongs in the Phase-0 harness (3a.6). Collection
(3c.1) must take its value from ``ac.value_pm`` / ``evaluate_actions`` (value_pm), never
``model_io.value_logit`` (win-prob).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

VALUE_PM_LO, VALUE_PM_HI = -1.0, 1.0


def in_pm_range(x, tol: float = 1e-4) -> bool:
    a = np.asarray(x, dtype=np.float64)
    if a.size == 0:
        return True
    return bool(a.min() >= VALUE_PM_LO - tol and a.max() <= VALUE_PM_HI + tol)


def assert_value_pm(x, name: str = "value", tol: float = 1e-4) -> None:
    """Assert a scalar/array lies in the value_pm range [-1,1]. A gross-error backstop
    (raw logits, win-prob*2, blown-up sums) — NOT a win-prob catch."""
    a = np.asarray(x, dtype=np.float64)
    if a.size == 0:
        return
    lo, hi = float(a.min()), float(a.max())
    assert lo >= VALUE_PM_LO - tol and hi <= VALUE_PM_HI + tol, (
        f"{name} outside value_pm range [-1,1]: min={lo:.5f} max={hi:.5f} — value-space "
        f"mismatch (sec 3). Is a raw logit / win-prob being stored instead of 2*sigmoid-1?"
    )


def assert_transitions_value_space(transitions, tol: float = 1e-4) -> None:
    """Range backstop over a batch: every ``Transition.value`` is value_pm-ranged."""
    if not transitions:
        return
    assert_value_pm([t.value for t in transitions], "Transition.value", tol)


def assert_gae_identity(advantages, returns, values, tol: float = 1e-3) -> None:
    """The single-space identity ``returns == advantages + values`` (gae.py construction).
    Asserting it confirms the three arrays are aligned AND in one space — a mismatch
    (e.g. returns built in win-prob while values are pm) breaks it."""
    adv = np.asarray(advantages, np.float64)
    ret = np.asarray(returns, np.float64)
    val = np.asarray(values, np.float64)
    assert adv.shape == ret.shape == val.shape, (
        f"GAE array shape mismatch: adv{adv.shape} ret{ret.shape} val{val.shape}")
    if adv.size == 0:
        return
    assert np.allclose(ret, adv + val, atol=tol), (
        "GAE identity returns == advantages + values broken — array misalignment or a "
        "value-space mismatch between returns and values (sec 3).")


def assert_gae_value_space(advantages, returns, values, tol: float = 1e-3,
                           allow_shaping: bool = False) -> None:
    """Phase-0 / sparse-phase check: the GAE identity PLUS the sparse-phase bound
    ``|return| <= 1`` (a ±1 terminal with gamma<=1 and pm values can't discount past
    ±1). ``allow_shaping=True`` skips the bound (PBRS adds the shaping term, 3b.7)."""
    assert_gae_identity(advantages, returns, values, tol)
    if not allow_shaping:
        assert_value_pm(returns, "GAE returns", tol=tol)


def looks_like_winprob(values, n_min: int = 50, neg_tol: float = 1e-3) -> bool:
    """Heuristic suspicion flag for the Phase-0 harness (3a.6). A balanced both-
    perspective self-play batch in value_pm MUST contain meaningfully NEGATIVE values
    (losing positions). If a large batch (>= ``n_min``) is entirely >= 0 and <= 1, it
    was almost certainly stored as win-prob [0,1] instead of value_pm [-1,1]. Returns
    True = suspicious. Small batches can legitimately be all-positive -> never flagged."""
    a = np.asarray(values, np.float64)
    if a.size < n_min:
        return False
    return bool(a.min() >= -neg_tol and a.max() <= 1.0 + neg_tol)


def value_space_report(transitions) -> Dict[str, float]:
    """Diagnostic summary of the value distribution for a batch (Phase-0 / logging)."""
    if not transitions:
        return {"n": 0}
    vals = np.asarray([t.value for t in transitions], np.float64)
    return {
        "n": int(vals.size),
        "value_min": float(vals.min()), "value_max": float(vals.max()),
        "value_mean": float(vals.mean()), "frac_negative": float((vals < 0).mean()),
        "looks_like_winprob": looks_like_winprob(vals),
    }
