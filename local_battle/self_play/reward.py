"""Terminal reward placement + collection-time integrity checks (task 3a.3).

Implements docs/ppo_reward_design.md sec 1: the terminal reward goes on the LAST step
only (sparse), driven by the terminal type; FALLBACK trajectories are discarded; and
the collector HARD-FAILS if MODEL-DRIVEN% drops below threshold (a self-play corpus
with fallbacks is corrupted — fallbacks reward the wrong action and break on-policy
assumptions). PBRS shaping (sec 4) is NOT added here — it is a gated 3b.7 concern.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from self_play.schema import Trajectory

# Sources that count as MODEL-DRIVEN (mirror gauntlet.py's report: model + replacement).
_MODEL_DRIVEN_SOURCES = ("model", "forced_switch_model")


def place_terminal_reward(traj: Trajectory) -> Trajectory:
    """Put the sec 1 terminal reward on the LAST step (in place):
      win / loss / adjudicated -> +-1   (via meta.won)
      draw                     ->  0    (real terminal, NO bootstrap)
      horizon_cut              ->  0 on the last step; GAE BOOTSTRAPS gamma*V(s_cut)
                                   from the recorded value + meta.bootstraps (NOT +-1)
      fallback                 -> must have been discarded first (asserted)
    Earlier steps keep reward 0 (sparse)."""
    if not traj.transitions:
        return traj
    assert traj.meta.is_trainable, \
        "place_terminal_reward got a FALLBACK trajectory — discard it first (sec 1)"
    r = traj.meta.outcome_reward()          # None for horizon_cut (bootstrap, not +-1)
    traj.transitions[-1].reward = 0.0 if r is None else float(r)
    return traj


def model_driven_fraction(source_counts: Dict[str, int]) -> float:
    """Fraction of per-turn decisions driven by the model (model + forced_switch_model),
    excluding the per-battle team-preview tp_* counts. 1.0 when there are no decisions.
    Mirrors gauntlet.py's MODEL-DRIVEN computation."""
    turn = {k: v for k, v in source_counts.items() if not k.startswith("tp_")}
    total = sum(turn.values())
    if total <= 0:
        return 1.0
    model = sum(turn.get(s, 0) for s in _MODEL_DRIVEN_SOURCES)
    return model / total


def assert_model_driven(source_counts: Dict[str, int], threshold: float = 0.99) -> float:
    """HARD-FAIL (sec 1/sec 13) if MODEL-DRIVEN% < threshold. Returns the fraction on
    success. The 100%-model-driven live work makes this ~always pass; a drop means a
    desync re-opened (e.g. illusion switch rejections) and the corpus is corrupted."""
    frac = model_driven_fraction(source_counts)
    assert frac >= threshold, (
        f"MODEL-DRIVEN {frac:.4f} < {threshold} — self-play corpus corrupted by "
        f"fallbacks; source_counts={dict(source_counts)}")
    return frac


def prepare_batch(
    trajectories: List[Trajectory], *,
    source_counts: Optional[Dict[str, int]] = None,
    min_model_driven: float = 0.99,
) -> List[Trajectory]:
    """Finalise a collected batch for PPO: optionally HARD-FAIL on low MODEL-DRIVEN%,
    DROP fallback trajectories (sec 1 — never let an engineering desync become a learned
    target), and place the terminal reward on each kept trajectory. Returns the trainable
    trajectories with rewards placed."""
    if source_counts is not None:
        assert_model_driven(source_counts, min_model_driven)
    kept = [t for t in trajectories if t.meta.is_trainable]
    for t in kept:
        place_terminal_reward(t)
    return kept
