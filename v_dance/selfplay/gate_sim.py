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
from v_dance.selfplay.gate import is_plateau  # noqa: F401,E402


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
