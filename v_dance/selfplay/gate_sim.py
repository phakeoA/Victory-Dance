"""Promotion-gate calibration simulator (pure — numpy only, no torch / poke-env).

The red-team's blocker: in symmetric self-play the candidate-vs-champion head-to-head
converges to ~50% by construction, so a significance-AND-effect-size PROGRESS bar can make
HOLD the mathematical attractor and the champion FREEZES. The fix (lenient AND + a forced
re-anchor release valve) needs its thresholds chosen from an EMPIRICAL promote-rate curve,
not guessed. This module Monte-Carlos the gate's head-to-head test so we can SEE, before
wiring anything live:

  * ``h2h_promote_rate``  — P(the bar fires) as a function of the TRUE edge + n_games + the
    (z, min_delta) operating point. The power curve (true edge > 0.5) AND the false-positive
    rate (true edge == 0.5) in one function.
  * ``simulate_sawtooth_run`` — a full multi-gen run where the policy keeps improving but the
    edge vs the FROZEN champion only grows while it isn't promoted (a sawtooth that RESETS on
    each promotion). Reports promotes + the gap between them, with an optional ``reanchor_every``
    forced re-anchor that caps the gap — so the freeze and the release valve are both visible.

The per-trial decision is IDENTICAL to ``promotion_gate``'s head-to-head branch when the
scripted ladder is flat (saturated): ``margin_lo = (wins/n - 0.5) - z*sqrt(0.25/n)`` and
promote iff ``margin_lo > min_delta`` (null-anchored SE, the fixed 0.5 reference). A test
pins this equivalence to the real gate.

Run it:  ``python -m v_dance.selfplay.gate_sim --demo``
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

# Canonical plateau detector lives WITH the gate (gate.py) so the sim and the real gate share
# ONE implementation — re-exported here so GS.is_plateau keeps working.
from v_dance.selfplay.gate import (  # noqa: F401,E402
    is_plateau, wilson_lower_bound, wilson_upper_bound)


def _se(n_games: int) -> float:
    """Null-anchored SE of a win-rate tested against the fixed 0.5 coin-flip (sqrt(0.25/n))."""
    return (0.25 / max(1, int(n_games))) ** 0.5


def h2h_promote_rate(true_p: float, n_games: int, *, z: float = 1.0, min_delta: float = 0.0,
                     trials: int = 20000, seed: int = 0) -> float:
    """Monte-Carlo P(the head-to-head PROGRESS bar fires) when the candidate's TRUE win-rate
    vs the champion is ``true_p``, measured over ``n_games`` games. Mirrors the gate exactly:
    promote iff ``(wins/n - 0.5) - z*sqrt(0.25/n) > min_delta``. At ``true_p == 0.5`` this is
    the gate's FALSE-POSITIVE rate; above 0.5 it is the POWER (true-positive rate)."""
    rng = np.random.default_rng(seed)
    wins = rng.binomial(int(n_games), float(true_p), size=int(trials))
    margin_lo = (wins / n_games - 0.5) - z * _se(n_games)
    return float(np.mean(margin_lo > min_delta))


def min_detectable_edge(n_games: int, *, z: float = 1.0, min_delta: float = 0.0) -> float:
    """The OBSERVED win-rate a candidate must hit for the bar to fire (the realised bar):
    ``0.5 + min_delta + z*sqrt(0.25/n)``. Shows how strict an operating point really is."""
    return 0.5 + min_delta + z * _se(n_games)


@dataclass
class GateOp:
    """A named (z, min_delta) operating point for the gate's head-to-head bar."""
    label: str
    z: float
    min_delta: float


def simulate_sawtooth_run(n_gens: int, edge_growth_per_gen: float, n_games: int, *,
                          z: float = 1.0, min_delta: float = 0.0,
                          reanchor_every: Optional[int] = None, seed: int = 0) -> dict:
    """Simulate ``n_gens`` generations where the policy keeps improving at
    ``edge_growth_per_gen`` (win-rate vs the CURRENT champion accrued each gen it ISN'T
    promoted) — a sawtooth that resets to ~0.5 on every promotion (the champion advances).
    Each gen draws ``wins ~ Binomial(n_games, true_p)`` and applies the head-to-head bar. If
    ``reanchor_every`` is set, a FORCED re-anchor fires after that many held gens (the v2
    release valve: competence holds + improving ⇒ advance the champion anyway). Returns the
    promote/forced counts, the gaps between champion advances, and the max edge built up —
    so a freeze (few advances, large gaps/edges) vs a healthy cadence is visible."""
    rng = np.random.default_rng(seed)
    gens_since = 0
    sig_promotes = 0          # advanced by the significance bar
    forced = 0                # advanced by the forced re-anchor valve
    gaps: List[int] = []
    max_edge = 0.0
    for _ in range(int(n_gens)):
        gens_since += 1
        true_p = min(0.95, 0.5 + edge_growth_per_gen * gens_since)
        max_edge = max(max_edge, true_p - 0.5)
        wins = rng.binomial(int(n_games), true_p)
        bar_fires = (wins / n_games - 0.5) - z * _se(n_games) > min_delta
        force = (reanchor_every is not None and gens_since >= int(reanchor_every)
                 and not bar_fires)
        if bar_fires or force:
            sig_promotes += int(bar_fires)
            forced += int(force and not bar_fires)
            gaps.append(gens_since)
            gens_since = 0
    advances = sig_promotes + forced
    mean_gap = float(np.mean(gaps)) if gaps else None
    return {
        "advances": advances, "sig_promotes": sig_promotes, "forced": forced,
        "mean_gap": mean_gap,
        "max_gap": (max(gaps) if gaps else int(n_gens)),
        "max_edge_at_decision": round(max_edge, 4),
        # FROZEN = the champion advances RARELY (a large real edge has to build up before the
        # bar fires): mean gap > 10 gens, or it never advanced. The sawtooth eventually clears
        # any bar given enough accumulation, so "rarely" — not "never" — is the freeze signal.
        "frozen": (mean_gap is None) or (mean_gap > 10.0),
    }


def simulate_frozen_ladder(n_gens: int, true_edge_fn, n_games: int, *,
                           promote_threshold: float = 0.70, plateau_window: int = 5,
                           plateau_margin: float = 0.01, seed: int = 0) -> dict:
    """The 'static until proven' ladder: the champion stays FROZEN until the candidate clears a
    HIGH observed bar (``promote_threshold``, e.g. 0.70) — OR a PLATEAU backstop fires because
    the climb has genuinely stalled below the bar. ``true_edge_fn(t)`` returns the candidate's
    TRUE win-rate-minus-0.5 vs the CURRENT frozen champion, ``t`` gens after it was last frozen
    (``t`` resets to 0 on every champion advance — the edge starts over against the new, stronger
    champion). Each gen draws a noisy ~``n_games`` mirror. Returns the advance counts (split by
    cause) + a per-gen event trace, so 'still climbing → waits' vs 'plateaued → backstop' is
    visible. This is the v2 design: high bar primary, plateau detector as the only backstop."""
    rng = np.random.default_rng(seed)
    t = 0
    obs_history: List[float] = []
    promotes_by_bar = 0
    advances_by_plateau = 0
    gaps: List[int] = []
    events: List[Tuple[int, str, float, int]] = []
    for gen in range(int(n_gens)):
        t += 1
        edge = max(0.0, min(0.45, float(true_edge_fn(t))))
        wins = rng.binomial(int(n_games), 0.5 + edge)
        obs = wins / n_games
        obs_history.append(obs)
        if obs >= promote_threshold:
            promotes_by_bar += 1
            events.append((gen, "PROMOTE(bar)", round(obs, 3), t))
            gaps.append(t); t = 0; obs_history = []
        elif is_plateau(obs_history, plateau_window, plateau_margin):
            advances_by_plateau += 1
            events.append((gen, "BACKSTOP(plateau)", round(obs, 3), t))
            gaps.append(t); t = 0; obs_history = []
        else:
            events.append((gen, "hold", round(obs, 3), t))
    return {
        "promotes_by_bar": promotes_by_bar,
        "advances_by_plateau": advances_by_plateau,
        "advances": promotes_by_bar + advances_by_plateau,
        "mean_gap": (float(np.mean(gaps)) if gaps else None),
        "events": events,
    }


# ══ P2.1 — HoF veto + 0.55 mirror + mirror-collapse calibration ════════════════
# Vectorised Wilson bounds (the scalar gate.wilson_*_bound applied to a numpy array of win
# counts) — kept here, and a test pins them == the scalar impls so the sim can NEVER drift
# from the real gate's decision rule.
def _wilson_upper_vec(wins, n: int, z: float):
    p = wins / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = p + z2 / (2.0 * n)
    margin = z * np.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return np.minimum(1.0, (centre + margin) / denom)


def _wilson_lower_vec(wins, n: int, z: float):
    p = wins / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = p + z2 / (2.0 * n)
    margin = z * np.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return np.maximum(0.0, (centre - margin) / denom)


def _draw_wins(rng, true_p: float, n: int, trials: int, rho: float = 0.0):
    """Per-suspect win counts over ``n`` games. ``rho==0`` → i.i.d. Binomial. ``rho>0`` →
    BETA-BINOMIAL with intra-cluster correlation ``rho`` (overdispersion): real model-vs-model
    games share a latent matchup strength, so the games within one suspect are CORRELATED and the
    EFFECTIVE n is < the nominal n (``n_eff ≈ n / (1 + (n-1)·rho)``). This is the same effect that
    made the mirror need 240 not 200 — the correlated sweep re-prices every threshold under it."""
    tp = min(1.0 - 1e-9, max(1e-9, float(true_p)))
    if rho <= 0.0:
        return rng.binomial(n, tp, size=trials)
    a = tp * (1.0 - rho) / rho
    b = (1.0 - tp) * (1.0 - rho) / rho
    p_eff = rng.beta(a, b, size=trials)
    return rng.binomial(n, p_eff)


def hof_snapshot_veto_rate(true_p: float, n_games: int, *, z: float = 1.96,
                           trials: int = 20000, seed: int = 0, rho: float = 0.0) -> float:
    """P(a SINGLE HoF suspect at true win-rate ``true_p`` is VETOED) = P(wilson_upper(wins,n,z)
    < 0.50). For ``true_p < 0.5`` this is the per-snapshot CATCH power (detecting a real weak
    suspect / RPS counter); for ``true_p >= 0.5`` it is the per-snapshot FALSE-reject rate."""
    rng = np.random.default_rng(seed)
    wins = _draw_wins(rng, true_p, n_games, trials, rho)
    return float(np.mean(_wilson_upper_vec(wins, n_games, z) < 0.5))


def hof_reject_rate(true_ps, n_games: int, *, z: float = 1.96, min_games: int = 40,
                    trials: int = 20000, seed: int = 0, rho: float = 0.0) -> float:
    """P(the HoF gate REJECTS) for a suspect set with true win-rates ``true_ps``, each played
    ``n_games``. The gate rejects iff ANY eligible (``n_games >= min_games``) suspect has
    wilson_upper < 0.50 — EXACTLY ``hall_of_fame_gate``'s worst-of-snapshots rule (P2.2). Joint
    Monte-Carlo (so the correlated variant composes). A planted set ``[0.20, 0.55, 0.55, ...]``
    gives the CATCH rate; an all-honest set ``[0.55]*M`` gives the FAMILY-WISE false-reject rate."""
    rng = np.random.default_rng(seed)
    rejected = np.zeros(int(trials), dtype=bool)
    if n_games < min_games:                      # no eligible suspect can veto → never rejects
        return 0.0
    for tp in true_ps:
        wins = _draw_wins(rng, tp, n_games, trials, rho)
        rejected |= (_wilson_upper_vec(wins, n_games, z) < 0.5)
    return float(np.mean(rejected))


def hof_bandmean_veto_rate(true_ps_in_band, n_per_snap: int, *, z: float = 1.96,
                           trials: int = 20000, seed: int = 0, rho: float = 0.0) -> float:
    """The OLD per-BAND-MEAN rule the design REJECTED: pool a band's snapshots into ONE
    (wins, games) and veto iff the band MEAN's wilson_upper < 0.50. This exists ONLY as the
    regression-guard for the core design decision — it shows P(catch) ≈ 0 for a single hidden
    counter (one 0.20 among 0.55 fodder), which is why the gate MUST be per-snapshot."""
    rng = np.random.default_rng(seed)
    total = np.zeros(int(trials), dtype=np.int64)
    for tp in true_ps_in_band:
        total += _draw_wins(rng, tp, n_per_snap, trials, rho).astype(np.int64)
    games = len(list(true_ps_in_band)) * n_per_snap
    return float(np.mean(_wilson_upper_vec(total, games, z) < 0.5))


def mirror_promote_rate(true_p: float, n_games: int, *, threshold: float = 0.55,
                        z: float = 1.645, trials: int = 20000, seed: int = 0,
                        rho: float = 0.0) -> float:
    """P(the v2 ``beat_champion`` bar fires) at true mirror win-rate ``true_p`` over ``n_games``:
    observed >= ``threshold`` AND the normal-approx lower CI ``p - z*sqrt(p(1-p)/n) > 0.5`` —
    IDENTICAL to ``promotion_gate_v2``'s mirror branch (a fidelity test pins it). ``true_p == 0.5``
    ⇒ FALSE-promote rate; ``> 0.5`` ⇒ power. The 0.70→0.55 recalibration reads its n off this."""
    rng = np.random.default_rng(seed)
    wins = _draw_wins(rng, true_p, n_games, trials, rho)
    p = wins / n_games
    lower = p - z * np.sqrt(p * (1.0 - p) / max(1, n_games))
    return float(np.mean((p >= threshold) & (lower > 0.5)))


def mirror_collapse_rate(true_p: float, n_games: int, *, z: float = 1.645, margin: float = 0.05,
                         trials: int = 20000, seed: int = 0, rho: float = 0.0) -> float:
    """P(the mirror-collapse REVERT fires) = P(wilson_upper(wins,n,z) < 0.50 - ``margin``) at true
    mirror win-rate ``true_p`` (the learner's win-rate vs its own FROZEN champion). ``true_p`` near
    0.5 (a freshly re-anchored, healthy learner restarting ~even vs the new champion) ⇒ the
    FALSE-revert rate, which MUST be ~0; ``true_p`` well below 0.5 (a genuinely eroded learner) ⇒
    the catch rate. Picks ``mirror_collapse_margin`` so a healthy learner is never reverted."""
    rng = np.random.default_rng(seed)
    wins = _draw_wins(rng, true_p, n_games, trials, rho)
    return float(np.mean(_wilson_upper_vec(wins, n_games, z) < (0.5 - margin)))


def force_valve_rate(suspect_true_p: float, n_per_snap: int, *, force_limit: int = 4,
                     z: float = 1.96, marginal_cut: float = 0.40, marginal_stat: str = "point",
                     n_gens: int = 40, trials: int = 2000, seed: int = 0) -> dict:
    """Calibrate the plateau-reanchor FORCE valve under a PERSISTENT weak suspect. Each gen the
    HoF tests one suspect held at ``suspect_true_p``; count CONSECUTIVE vetoes (wilson_upper<0.5);
    after ``force_limit`` consecutive vetoes, FORCE the re-anchor ONLY if the worst suspect is
    MARGINAL — ``marginal_stat`` ∈ {'point' (observed p), 'lower' (wilson_lower)} ``>= marginal_cut``.
    A CATASTROPHIC suspect (below the cut) must NEVER force (freeze + alert per the user). Returns
    mean force events/run + the fraction of runs that ever forced — so 'marginal mild hole advances
    within the limit' vs 'catastrophic hole never advances' is visible, and the marginal statistic
    can be chosen (the originally-specified wilson_lower>=0.45 is degenerate — a vetoed suspect's
    lower CI is almost always < 0.45, so the split must sit lower / on the point estimate)."""
    rng = np.random.default_rng(seed)
    forces = 0
    runs_forced = 0
    for _ in range(int(trials)):
        streak = 0
        forced_this_run = False
        for _g in range(int(n_gens)):
            wins = int(rng.binomial(n_per_snap, max(1e-9, min(1 - 1e-9, suspect_true_p))))
            vetoed = wilson_upper_bound(wins, n_per_snap, z) < 0.5
            stat = (wins / n_per_snap) if marginal_stat == "point" \
                else wilson_lower_bound(wins, n_per_snap, z)
            streak = streak + 1 if vetoed else 0
            if streak >= force_limit:
                if stat >= marginal_cut:            # marginal hole → force-advance
                    forces += 1
                    forced_this_run = True
                streak = 0                          # reset the window after the decision point
        runs_forced += int(forced_this_run)
    return {"mean_forces": forces / trials, "frac_runs_forced": runs_forced / trials}


# ── demo (no server / torch): print the calibration tables ────────────────────
def _demo(trials: int = 30000, seed: int = 0) -> None:
    ops = [
        GateOp("current  (z=1.0,  δ=0.00)", 1.0, 0.00),     # post-SE-fix current gate
        GateOp("v2-lenient(z=1.0, δ=0.02)", 1.0, 0.02),     # proposed lenient progress bar
        GateOp("v2-mid   (z=1.28, δ=0.02)", 1.28, 0.02),    # 90% one-sided
        GateOp("strict   (z=1.645,δ=0.05)", 1.645, 0.05),   # the v1 bar that FREEZES
    ]
    true_ps = [0.50, 0.52, 0.55, 0.58, 0.60, 0.65]

    print("== Promotion-gate calibration (Monte-Carlo, no server) ====================")
    # n=60 is the REAL mirror size (eval_battles auto-sizes to ~60/opp); n=120 = a "bump the
    # mirror" option that buys more power at ~2x the mirror's eval cost.
    for n_games in (60, 120):
        print(f"\n  PROMOTE-RATE vs TRUE head-to-head edge   (n_games={n_games}, "
              f"trials={trials})")
        print(f"    realised bar (observed WR needed to fire):")
        for op in ops:
            print(f"      {op.label}: >{min_detectable_edge(n_games, z=op.z, min_delta=op.min_delta)*100:4.1f}%")
        header = "    true edge |" + "".join(f" {op.label[:18]:>18} |" for op in ops)
        print(header)
        print("    " + "-" * (len(header) - 4))
        for tp in true_ps:
            row = f"    {tp*100:5.1f}%   |"
            for op in ops:
                r = h2h_promote_rate(tp, n_games, z=op.z, min_delta=op.min_delta,
                                     trials=trials, seed=seed)
                tag = "  <- FP" if tp == 0.50 else ""
                row += f" {r*100:16.1f}% |" if not tag else f" {r*100:11.1f}%{tag} |"
            print(row)
        print("    (row 50.0% = false-positive rate; higher rows = power / true-positive rate)")

    print("\n  FREEZE / RELEASE-VALVE — sawtooth run (policy improves vs a frozen champion)")
    print("    50 gens, n_games=100; edge_growth = win-rate gained per held gen.")
    print(f"    {'config':28} {'edge/gen':>9} {'reanchor':>9} {'advances':>9} "
          f"{'mean_gap':>9} {'max_edge':>9} {'frozen?':>8}")
    for op in (ops[1], ops[3]):                  # v2-lenient vs the strict (freezing) bar
        for growth in (0.005, 0.010, 0.020):
            for reanchor in (None, 8):
                r = simulate_sawtooth_run(50, growth, 100, z=op.z, min_delta=op.min_delta,
                                          reanchor_every=reanchor, seed=seed)
                print(f"    {op.label:28} {growth:9.3f} {str(reanchor):>9} "
                      f"{r['advances']:9d} {str(r['mean_gap']):>9} "
                      f"{r['max_edge_at_decision']:9.3f} {str(r['frozen']):>8}")

    # ── frozen-champion ladder: high bar (70%) + plateau backstop ──────────────
    print("\n  FROZEN-CHAMPION LADDER — high 70% bar + plateau backstop (window=5)")
    print("    The mirror's NOISE decides whether the bar/detector even work. Compare n_games:")
    trajectories = [
        ("improving  (edge +1.2%/gen)", lambda t: 0.012 * t),                       # keeps climbing
        ("plateaued  (saturates ~60%)", lambda t: 0.10 * (1.0 - 2.71828 ** (-t / 4.0))),
    ]
    for n_games in (60, 240):
        print(f"\n    --- n_games={n_games} ---")
        for name, fn in trajectories:
            r = simulate_frozen_ladder(60, fn, n_games, promote_threshold=0.70,
                                       plateau_window=5, plateau_margin=0.01, seed=seed)
            ev = [e for e in r["events"] if e[1] != "hold"][:6]
            seq = "  ".join(f"g{g}:{k.split('(')[0]}@{o*100:.0f}%" for g, k, o, _ in ev)
            print(f"      [{name}]  by-bar={r['promotes_by_bar']}  by-plateau="
                  f"{r['advances_by_plateau']}")
            print(f"        first advances: {seq or '(none)'}")
    print("    → at n=60 the bar FALSE-promotes the plateaued-60% policy (noise hits 70%) AND")
    print("      the detector FALSE-fires on the climbing one. At n=240 the bar only fires for")
    print("      the genuine climber and the backstop only fires on the real plateau.")
    print("===========================================================================")
    _demo_p21(trials, seed)


def _demo_p21(trials: int = 20000, seed: int = 0) -> None:
    """P2.1 — HoF veto + 0.55 mirror + mirror-collapse + force-valve calibration tables. These
    pick the LOCKED numbers wired in P2.2: HoF z=1.96/n=60/<=6 suspects; mirror thr=0.55/n=360;
    mirror-collapse margin=0.05/n=360; force valve simplified to freeze+alert (the marginal
    auto-advance window is empty at the HoF's n)."""
    t = min(int(trials), 20000)
    print("\n== P2.1 — HoF / mirror-0.55 / collapse calibration (Monte-Carlo) ==========")
    tps = [0.20, 0.30, 0.40, 0.50, 0.55, 0.60]
    print("  (1) HoF PER-SNAPSHOT veto rate  z=1.96  (true<0.5 = CATCH, >=0.5 = false-reject):")
    print("      n   |" + "".join(f"{p:>7.2f}" for p in tps))
    for n in (40, 60, 80, 120):
        print(f"      {n:<4}|" + "".join(
            f"{hof_snapshot_veto_rate(p, n, z=1.96, trials=t, seed=seed)*100:6.1f}%" for p in tps))
    catch = hof_reject_rate([0.20, 0.60, 0.60, 0.60, 0.60, 0.60], 60, z=1.96, trials=t, seed=seed)
    band = hof_bandmean_veto_rate([0.20, 0.60, 0.60, 0.60, 0.60, 0.60], 60, z=1.96, trials=t, seed=seed)
    print(f"\n  (2) REGRESSION GUARD (one 0.20 counter among five 0.60, n=60): per-snapshot catch "
          f"{catch*100:.1f}%  vs  band-mean {band*100:.1f}%")
    print("      → band-mean is BLIND to a hidden counter; the gate MUST be per-snapshot.")
    print("\n  (3) FAMILY-WISE false-reject (M=6) + CATCH under correlation rho:")
    print(f"      {'rho':>6} {'catch@0.30':>11} {'FWER@0.55':>10} {'FWER@0.50':>10}")
    for rho in (0.0, 0.002, 0.005, 0.01):
        print(f"      {rho:6.3f} "
              f"{hof_snapshot_veto_rate(0.30,60,z=1.96,trials=t,seed=seed,rho=rho)*100:10.1f}% "
              f"{hof_reject_rate([0.55]*6,60,z=1.96,trials=t,seed=seed,rho=rho)*100:9.1f}% "
              f"{hof_reject_rate([0.50]*6,60,z=1.96,trials=t,seed=seed,rho=rho)*100:9.1f}%")
    print("      → LOCKED: z=1.96, n=60/suspect, <=6 suspects — robust to rho<=~0.005.")
    mtps = [0.50, 0.55, 0.58, 0.60, 0.65]
    print("\n  (4) MIRROR 0.55 bar promote-rate  thr=0.55 z=1.645  (0.50 col = false-promote):")
    print("      n   |" + "".join(f"{p:>7.2f}" for p in mtps))
    for n in (240, 300, 360, 500):
        print(f"      {n:<4}|" + "".join(
            f"{mirror_promote_rate(p, n, threshold=0.55, z=1.645, trials=t, seed=seed)*100:6.1f}%"
            for p in mtps))
    print("      → LOCKED: threshold=0.55, n=360 (FP ~3%, pow@0.58 ~89%); true-0.55 ~50%/gen (~2 gens).")
    print("      → correlation-sensitive (FP rises with rho) — laterals are absorbed by the HoF + collapse-revert.")
    ctps = [0.50, 0.48, 0.45, 0.42, 0.40, 0.35, 0.30]
    print("\n  (5) MIRROR-COLLAPSE revert  z=1.645 bar wilson_upper<0.45  (~0.5 = false-revert MUST be ~0):")
    print("      n   |" + "".join(f"{p:>7.2f}" for p in ctps))
    for n in (240, 360):
        print(f"      {n:<4}|" + "".join(
            f"{mirror_collapse_rate(p, n, z=1.645, margin=0.05, trials=t, seed=seed)*100:6.1f}%"
            for p in ctps))
    print("      → LOCKED: margin=0.05 (bar 0.45), n=360 (false-revert@0.50 ~0%, catch@0.35 ~99%).")
    fzero = force_valve_rate(0.30, 60, force_limit=4, marginal_cut=0.0, n_gens=40, trials=2000, seed=seed)
    fcut = force_valve_rate(0.30, 60, force_limit=4, marginal_cut=0.40, n_gens=40, trials=2000, seed=seed)
    print(f"\n  (6) FORCE-VALVE (weak 0.30 suspect, n=60): cut=0.0 forces "
          f"{fzero['frac_runs_forced']*100:.0f}% of runs, cut=0.40 forces {fcut['frac_runs_forced']*100:.0f}%.")
    print("      → FINDING: a persistently-vetoed suspect is ALWAYS point<0.40 (catastrophic); the")
    print("        marginal auto-advance window is EMPTY at n=60 → SIMPLIFY to freeze+alert+--hof-override.")
    print("===========================================================================")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Promotion-gate calibration simulator (pure)")
    ap.add_argument("--demo", action="store_true", help="print the promote-rate + freeze tables")
    ap.add_argument("--trials", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.demo:
        _demo(args.trials, args.seed)
    else:
        ap.print_help()
