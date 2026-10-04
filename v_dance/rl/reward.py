"""Terminal reward placement + collection-time integrity checks (task 3a.3).

Implements docs/ppo_reward_design.md sec 1: the terminal reward goes on the LAST step
only (sparse), driven by the terminal type; FALLBACK trajectories are discarded; and
the collector HARD-FAILS if MODEL-DRIVEN% drops below threshold (a self-play corpus
with fallbacks is corrupted — fallbacks reward the wrong action and break on-policy
assumptions). The default (v1) reward is terminal-only ±1 (the gated PBRS option, sec 4, was removed 2026-10-04).

REWARD v2 (2026-10-04, USER: "better than +1 win -1 losing while not being harmful to pokemon playstyles … make sure
that winning and losing is overall the biggest punishment and reward"; memory 14 'REWARD v2: DESIGN DECIDED'), only for
a trajectory collected under ``--reward-v2`` (``meta.reward_mode == "v2"``; anything else is v1, byte-identical):
  · the LOSS MARGIN — a win is +1 however it was won; a loss is -1 + 0.25 × (their brought mons we fainted) / 4, so a
    close loss beats a wipe-out but every loss stays ≥ 1.75 below every win; our own faints / sacrifices never count;
  · the FIELD POTENTIAL — each step pays the CHANGE of Φ (``Transition.phi``, 0.1 × weather owner + 0.1 × terrain
    owner, v_dance/selfplay/field_shaping.py): F = κ·(γ·Φ(s') − Φ(s)), Φ(terminal) = 0 (potential-based shaping —
    it telescopes, so it cannot be farmed and cannot change which policy is best); κ fades to 0 over the run's last
    third (``field_fade``);
  · every v2 reward is × 1/(1 + 0.2) so a return stays inside the critic's [-1, 1].
The terminal value goes on the last step at collection (``place_terminal_reward``); the shaping and the scale are
applied when the trainer computes GAE (``shaped_rewards``), so a stored v2 trajectory still holds sparse rewards.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from v_dance.rl.schema import Trajectory

REWARD_V2 = "v2"
LOSS_MARGIN = 0.25           # the most a loss earns back (all four of their brought mons fainted — impossible on a loss)
BRING = 4                    # VGC brings four
FIELD_STRENGTH = 0.1         # Φ per domain (weather, terrain): a full theirs → ours swing in one domain = +0.2
V2_SCALE = 1.0 / (1.0 + 2.0 * FIELD_STRENGTH)   # |r_T| ≤ 1 and |Φ| ≤ 0.2 → every v2 return within [-1, 1]


def is_v2(meta) -> bool:
    return getattr(meta, "reward_mode", None) == REWARD_V2


def terminal_reward(meta) -> Optional[float]:
    """The reward placed on a trajectory's LAST step (unscaled). v1 = ``outcome_reward()`` (±1 / draw 0 / None). v2 =
    the same, except a loss (outright or adjudicated) earns back ``LOSS_MARGIN × opp_fainted / BRING`` — a win is +1
    however it was won (our sacrifices never count). ``opp_fainted`` unknown on a v2 loss counts as 0."""
    r = meta.outcome_reward()
    if r is None or r >= 0.0 or not is_v2(meta):          # v1 · a win · a draw · a horizon cut
        return r
    k = min(max(int(getattr(meta, "opp_fainted", None) or 0), 0), BRING)
    return -1.0 + LOSS_MARGIN * k / BRING


def field_fade(gen: int, total_generations: Optional[int]) -> float:
    """κ for generation ``gen`` of a reward-v2 run whose generations are 0 .. ``total_generations`` − 1: full strength
    through the first two thirds, then linear to 0 at the last generation (40 gens → 1.0 through gen 26, 12/13 at gen
    27, … 0 at gen 39). Counted from gen 0, so a resume whose launch ends at the same generation (``--generations`` =
    the generations LEFT) continues the original schedule. ``total_generations`` None / ≤ 0 (an open-ended run) →
    never fades."""
    if not total_generations or int(total_generations) <= 0:
        return 1.0
    total = int(total_generations)
    end = total - 1                                       # the last generation: κ = 0
    start = int(round(2.0 * total / 3.0)) - 1             # the last full-strength generation
    if end <= start or gen <= start:
        return 1.0
    if gen >= end:
        return 0.0
    return float(end - gen) / float(end - start)


def shaped_rewards(traj: Trajectory, gamma: float, field_kappa: float = 1.0) -> np.ndarray:
    """The per-step rewards GAE uses. v1: the stored rewards, untouched. v2: ``V2_SCALE × (r_t + κ·(γ·Φ_{t+1} − Φ_t))``
    with Φ_T = 0 after a real terminal (a horizon cut — never produced today — keeps its last Φ, like its value
    bootstrap). A step with no recorded Φ counts as 0 (neutral)."""
    ts = traj.transitions
    r = np.array([t.reward for t in ts], dtype=np.float64)
    if not ts or not is_v2(traj.meta):
        return r
    phi = np.array([float(t.phi) if getattr(t, "phi", None) is not None else 0.0 for t in ts], dtype=np.float64)
    nxt = np.append(phi[1:], phi[-1] if traj.meta.bootstraps else 0.0)
    return V2_SCALE * (r + float(field_kappa) * (float(gamma) * nxt - phi))

# Sources that count as MODEL-DRIVEN (mirror gauntlet.py's report: model + replacement).
_MODEL_DRIVEN_SOURCES = ("model", "forced_switch_model")
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
      win / loss / adjudicated -> +-1   (via meta.won; v2: a loss -1 + the margin, ``terminal_reward``)
      draw                     ->  0    (real terminal, NO bootstrap)
      horizon_cut              ->  0 on the last step; GAE BOOTSTRAPS gamma*V(s_cut)
                                   from the recorded value + meta.bootstraps (NOT +-1)
      fallback                 -> must have been discarded first (asserted)
    Earlier steps keep reward 0 (sparse)."""
    if not traj.transitions:
        return traj
    assert traj.meta.is_trainable, \
        "place_terminal_reward got a FALLBACK trajectory — discard it first (sec 1)"
    r = terminal_reward(traj.meta)          # None for horizon_cut (bootstrap, not +-1)
    traj.transitions[-1].reward = 0.0 if r is None else float(r)
    return traj


def model_driven_fraction(source_counts: Dict[str, int]) -> float:
    """Fraction of EXECUTED per-turn decisions driven by the model (model + forced_switch_model +
    B1 search) vs executed non-model escapes (retry / default / forced-switch escape). Bookkeeping
    AND DIAGNOSTIC counters (rejected_resample, abandon_forfeit, tp_*, belief_* feed-fire, search_*
    fire/fallback) are excluded from the denominator — they are NOT per-turn decision sources — so a
    legitimate resample burst or a belief/search-diagnostic burst can't trip the guard. 1.0 when
    there are no decisions. Matches game_runner.phase0_report's de-duped MODEL-DRIVEN measure."""
    # 2026-10-02 (drill review RL-3, pre-existing since 09-03): spawn_* are the server-side spawner's BOOKKEEPING
    # (pairs / decisions / elapsed_s …), stripped exactly like game_runner.phase0_report strips them — counted as
    # non-model steps they dragged the fraction under 0.5 and the 0.75 guard aborted every --spawn-rooms run.
    turn = {k: v for k, v in source_counts.items()
            if not k.startswith("tp_") and not k.startswith("belief_")
            and not k.startswith("search_") and not k.startswith("spawn_")
            and k not in _NON_DECISION_COUNTERS}
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
