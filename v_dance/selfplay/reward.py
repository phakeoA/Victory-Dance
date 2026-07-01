"""Terminal reward placement + collection-time integrity checks (task 3a.3).

Implements docs/ppo_reward_design.md sec 1: the terminal reward goes on the LAST step
only (sparse), driven by the terminal type; FALLBACK trajectories are discarded; and
the collector HARD-FAILS if MODEL-DRIVEN% drops below threshold (a self-play corpus
with fallbacks is corrupted — fallbacks reward the wrong action and break on-policy
assumptions). PBRS shaping (sec 4) is NOT added here — it is a gated 3b.7 concern.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from v_dance.selfplay.schema import Trajectory

# Sources that count as MODEL-DRIVEN (mirror gauntlet.py's report: model + replacement).
# B1 (Level-C): a belief-weighted-SEARCH pick is tagged source="search" — it IS a model-grounded
# decision (the search ranks candidates with the policy + value head), so it counts as model-driven.
# A search FALLBACK runs the raw policy and returns source="model", so that path is already covered.
_MODEL_DRIVEN_SOURCES = ("model", "forced_switch_model", "search")
# BOOKKEEPING-only counters excluded from the MODEL-DRIVEN denominator (#21). `rejected_resample` is
# incremented ALONGSIDE `model` when a Showdown-rejected MODEL order is re-sampled to a fresh legal
# MODEL action (the executed pick is still model-driven), so leaving it in the "all non-tp" denominator
# wrongly deflated MODEL-DRIVEN% toward the 0.95 warn / 0.75 abort on a legitimate Choice-lock/Encore/
# Taunt resample burst. `abandon_forfeit` is a watchdog count, not a per-turn decision. `finalize_failed`
# is a post-game bookkeeping count (a trajectory that failed to finalize), also not a per-turn decision.
# Everything else non-tp (model, forced_switch_model, retry_default, forced_default, forfeit,
# forced_switch_escape, forced_switch, …) IS a real executed decision and stays in the denominator — a
# blocklist keeps the guard's coverage intact (a whitelist would silently drop any source it forgot to list).
_NON_DECISION_COUNTERS = ("rejected_resample", "abandon_forfeit", "finalize_failed")


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
    """Fraction of EXECUTED per-turn decisions driven by the model (model + forced_switch_model +
    B1 search) vs executed non-model escapes (retry / default / forced-switch escape). Bookkeeping
    AND DIAGNOSTIC counters (rejected_resample, abandon_forfeit, tp_*, belief_* feed-fire, search_*
    fire/fallback) are excluded from the denominator — they are NOT per-turn decision sources — so a
    legitimate resample burst or a belief/search-diagnostic burst can't trip the guard. 1.0 when
    there are no decisions. Matches game_runner.phase0_report's de-duped MODEL-DRIVEN measure."""
    turn = {k: v for k, v in source_counts.items()
            if not k.startswith("tp_") and not k.startswith("belief_")
            and not k.startswith("search_") and k not in _NON_DECISION_COUNTERS}
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


def drop_fallback_pairs(trajectories: List[Trajectory]) -> List[Trajectory]:
    """T3.1: drop FALLBACK trajectories (loop-guard backstop-forfeits) AND their zero-sum mirror —
    the SAME battle_id recorded from the OTHER perspective, which sees a (mislabeled) +1 win. A
    forfeited battle is an engineering escape, not a game; rewarding EITHER side corrupts credit
    assignment (sec 1). The live collection path flattens both perspectives into one flat list, so we
    drop by battle_id rather than by pairing. ``is_trainable`` is False only for FALLBACK (HORIZON_CUT
    bootstraps and is kept). Defensive: anything without a readable ``meta`` is conservatively KEPT —
    never silently drop data we cannot classify."""
    def _meta(t):
        return getattr(t, "meta", None)

    bad = {m.battle_id for t in trajectories
           if (m := _meta(t)) is not None and not m.is_trainable}
    if not bad:
        return trajectories
    return [t for t in trajectories
            if (m := _meta(t)) is None or m.battle_id not in bad]


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
