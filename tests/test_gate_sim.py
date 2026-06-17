"""Gate calibration simulator (v_dance/selfplay/gate_sim.py) — the Monte-Carlo harness that
prices the promotion gate's head-to-head bar BEFORE wiring v2 live. Tests the power/FP math,
the freeze + release-valve dynamics, and (crucially) that the simulator's per-trial decision
is IDENTICAL to the real ``promotion_gate``'s head-to-head branch."""
from __future__ import annotations

import pytest

pytest.importorskip("numpy")

from v_dance.selfplay import gate_sim as GS
from v_dance.selfplay.generation import promotion_gate, GateConfig


# ── power / false-positive curve ──────────────────────────────────────────────
def test_promote_rate_is_monotone_in_true_edge():
    rates = [GS.h2h_promote_rate(p, 100, z=1.0, min_delta=0.0, trials=8000, seed=1)
             for p in (0.50, 0.55, 0.60, 0.70)]
    assert all(a <= b + 1e-9 for a, b in zip(rates, rates[1:]))   # more true edge ⇒ more promotes
    assert rates[0] < 0.25                                        # FP at a coin-flip is modest
    assert rates[-1] > 0.9                                        # a big true edge nearly always fires


def test_false_positive_rate_tracks_z_band():
    # at true 0.5, the promote-rate is the one-sided tail at z — looser z ⇒ higher FP.
    fp_loose = GS.h2h_promote_rate(0.50, 100, z=1.0, min_delta=0.0, trials=40000, seed=2)
    fp_tight = GS.h2h_promote_rate(0.50, 100, z=1.645, min_delta=0.05, trials=40000, seed=2)
    assert fp_loose > fp_tight
    assert 0.10 < fp_loose < 0.22          # ~16% one-sided tail at z=1.0
    assert fp_tight < 0.03                 # strict bar barely false-positives


def test_min_detectable_edge_matches_realised_bar():
    # the bar fires iff observed WR > 0.5 + min_delta + z*sqrt(0.25/n)
    assert GS.min_detectable_edge(100, z=1.645, min_delta=0.05) == pytest.approx(0.5 + 0.05 + 1.645 * 0.05)
    assert GS.min_detectable_edge(100, z=1.0, min_delta=0.02) == pytest.approx(0.5 + 0.02 + 1.0 * 0.05)


# ── fidelity: the sim's decision == the real gate's head-to-head branch ────────
def test_sim_decision_matches_real_promotion_gate():
    """For a saturated-flat scripted ladder, promotion_gate promotes iff its beats_best bar
    fires — which must be EXACTLY the sim's per-trial rule. Check a sweep of win counts."""
    n, z, md = 100, 1.0, 0.02
    se = GS._se(n)
    for wins in range(45, 75):
        sim_promotes = (wins / n - 0.5) - z * se > md
        verdict, st = promotion_gate(196, 200, 196, 200, GateConfig(z=z, min_delta=md),
                                     prevbest_wins=wins, prevbest_games=n)
        gate_promotes = (verdict == "promote" and st["verdict_reason"] == "beats_prev_best")
        assert sim_promotes == gate_promotes, f"mismatch at wins={wins}"


# ── freeze + release valve ────────────────────────────────────────────────────
def test_strict_bar_freezes_low_edge_run():
    """A strict bar over a slowly-improving policy advances the champion RARELY — a big real
    edge has to build up before the bar fires (the quantitative freeze)."""
    strict = GS.simulate_sawtooth_run(50, 0.005, 100, z=1.645, min_delta=0.05, seed=3)
    lenient = GS.simulate_sawtooth_run(50, 0.005, 100, z=1.0, min_delta=0.02, seed=3)
    assert strict["frozen"] is True                  # mean gap between advances > 10 gens
    assert strict["max_edge_at_decision"] > 0.08     # a large real edge built up unrewarded
    assert strict["advances"] < lenient["advances"]  # the lenient bar advances far more often


def test_lenient_bar_advances_more_than_strict():
    lenient = GS.simulate_sawtooth_run(50, 0.010, 100, z=1.0, min_delta=0.02, seed=4)
    strict = GS.simulate_sawtooth_run(50, 0.010, 100, z=1.645, min_delta=0.05, seed=4)
    assert lenient["advances"] > strict["advances"]


# ── frozen-champion ladder + plateau detector ─────────────────────────────────
def test_is_plateau_detects_flat_not_rising():
    rising = [0.50, 0.52, 0.54, 0.56, 0.58, 0.62, 0.64, 0.66, 0.68, 0.70]
    assert GS.is_plateau(rising, window=5, margin=0.01) is False        # recent >> prior
    flat = [0.60, 0.59, 0.61, 0.60, 0.60, 0.60, 0.61, 0.59, 0.60, 0.60]
    assert GS.is_plateau(flat, window=5, margin=0.01) is True           # recent ≈ prior
    declining = [0.65, 0.64, 0.63, 0.62, 0.61, 0.58, 0.57, 0.56, 0.55, 0.54]
    assert GS.is_plateau(declining, window=5, margin=0.01) is True      # recent < prior
    assert GS.is_plateau([0.6] * 9, window=5) is False                  # < 2*window → keep waiting


def test_frozen_ladder_improving_promotes_by_bar_with_enough_games():
    """At a clean mirror size (n=400) a genuinely-climbing policy promotes via the 70% BAR and
    is NEVER falsely flagged as a plateau."""
    r = GS.simulate_frozen_ladder(60, lambda t: 0.012 * t, 400,
                                  promote_threshold=0.70, plateau_window=5, seed=0)
    assert r["promotes_by_bar"] >= 2
    assert r["advances_by_plateau"] == 0


def test_frozen_ladder_plateaued_fires_backstop_not_bar_with_enough_games():
    """A policy that saturates ~60% never clears the 70% bar at n=400 (no lucky false-promote);
    the PLATEAU backstop is what advances the champion."""
    r = GS.simulate_frozen_ladder(60, lambda t: 0.10 * (1.0 - 2.71828 ** (-t / 4.0)), 400,
                                  promote_threshold=0.70, plateau_window=5, seed=0)
    assert r["promotes_by_bar"] == 0
    assert r["advances_by_plateau"] >= 2


def test_frozen_ladder_n60_is_too_noisy_for_either_mechanism():
    """The load-bearing finding: at the CURRENT ~60-game mirror the noise breaks BOTH — the
    70% bar leaks (a plateaued-60% policy false-promotes) AND the detector false-fires on a
    climber. This is why the design needs more mirror games, not a logic change."""
    plateaued = GS.simulate_frozen_ladder(60, lambda t: 0.10 * (1.0 - 2.71828 ** (-t / 4.0)),
                                          60, promote_threshold=0.70, plateau_window=5, seed=0)
    improving = GS.simulate_frozen_ladder(60, lambda t: 0.012 * t, 60,
                                          promote_threshold=0.70, plateau_window=5, seed=0)
    assert plateaued["promotes_by_bar"] >= 1        # 60%-true LEAKS through the 70% bar on noise
    assert improving["advances_by_plateau"] >= 1    # climber FALSELY flagged as plateaued


def test_reanchor_valve_caps_the_gap():
    """The forced re-anchor bounds how long the champion can stall, even when the
    significance bar never fires (the freeze case)."""
    no_valve = GS.simulate_sawtooth_run(50, 0.0, 100, z=1.645, min_delta=0.05,
                                        reanchor_every=None, seed=5)
    valve = GS.simulate_sawtooth_run(50, 0.0, 100, z=1.645, min_delta=0.05,
                                     reanchor_every=8, seed=5)
    assert no_valve["advances"] <= 1 and no_valve["frozen"]   # true edge 0 ⇒ bar ~never fires
    assert valve["forced"] >= 5 and valve["max_gap"] <= 8     # valve advances every ~8 gens
