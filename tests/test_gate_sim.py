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


# ── P2.1: HoF veto + 0.55 mirror recalibration + mirror-collapse + force valve ──
def test_wilson_vecs_match_scalar_gate_impl():
    """The sim's vectorised Wilson bounds MUST equal gate.py's scalar ones, so the calibration
    can never drift from the real gate's decision rule (the fidelity invariant)."""
    import numpy as np
    from v_dance.selfplay.gate import wilson_lower_bound, wilson_upper_bound
    wins = np.array([0, 10, 30, 45, 60, 90, 120])
    for n, z in [(120, 1.96), (240, 1.645)]:
        up = GS._wilson_upper_vec(wins, n, z)
        lo = GS._wilson_lower_vec(wins, n, z)
        for i, w in enumerate(wins):
            assert up[i] == pytest.approx(wilson_upper_bound(int(w), n, z), abs=1e-9)
            assert lo[i] == pytest.approx(wilson_lower_bound(int(w), n, z), abs=1e-9)


def test_hof_snapshot_catch_monotone_and_calibrated():
    catch = [GS.hof_snapshot_veto_rate(p, 60, z=1.96, trials=8000, seed=1)
             for p in (0.20, 0.30, 0.40, 0.50)]
    assert catch[0] > catch[1] > catch[2] > catch[3]              # weaker suspect ⇒ more catch
    assert catch[0] > 0.95                                        # a 0.20 counter is almost always caught
    assert GS.hof_snapshot_veto_rate(0.30, 60, z=1.96, trials=8000, seed=1) > 0.85    # ≥85% @0.30
    assert GS.hof_snapshot_veto_rate(0.50, 60, z=1.96, trials=20000, seed=1) < 0.05   # low false-reject
    assert GS.hof_snapshot_veto_rate(0.60, 60, z=1.96, trials=8000, seed=1) < 0.02


def test_hof_per_snapshot_beats_bandmean_on_hidden_counter():
    """THE design-decision regression guard: one 0.20 counter hidden among 0.60 fodder is caught
    ~always per-snapshot but is INVISIBLE to the rejected band-mean rule."""
    suspects = [0.20, 0.60, 0.60, 0.60, 0.60, 0.60]
    assert GS.hof_reject_rate(suspects, 60, z=1.96, trials=8000, seed=2) > 0.95
    assert GS.hof_bandmean_veto_rate(suspects, 60, z=1.96, trials=8000, seed=2) < 0.05


def test_hof_min_games_floor_blocks_veto():
    # below the min-games floor an underpowered suspect cannot veto (no noise-freeze)
    assert GS.hof_reject_rate([0.20], 30, z=1.96, min_games=40, trials=4000, seed=3) == 0.0
    assert GS.hof_reject_rate([0.20], 60, z=1.96, min_games=40, trials=4000, seed=3) > 0.9


def test_hof_family_false_reject_bounded_and_correlation_inflates():
    iid = GS.hof_reject_rate([0.55] * 6, 60, z=1.96, trials=20000, seed=4, rho=0.0)
    corr = GS.hof_reject_rate([0.55] * 6, 60, z=1.96, trials=20000, seed=4, rho=0.005)
    assert iid < 0.05                       # a realistic 0.55-vs-ancestors candidate rarely false-rejects
    assert corr > iid                       # correlated games lower effective n ⇒ more false-reject


def test_mirror_promote_rate_matches_v2_gate_and_recalibrates():
    from v_dance.selfplay.gate import promotion_gate_v2, GateConfigV2
    cfg = GateConfigV2(promote_threshold=0.55, promote_z=1.645, min_h2h_games=360)
    n = 360
    for wins in (190, 200, 210, 230):
        p = wins / n
        sim = (p >= 0.55) and (p - 1.645 * (p * (1 - p) / n) ** 0.5 > 0.5)
        _, st = promotion_gate_v2(scripted_wins=140, scripted_games=180, high_water=0.55,
                                  mirror_wins=wins, mirror_games=n, h2h_history=[0.5],
                                  have_champion=True, cfg=cfg)
        assert sim == st["mirror"]["beats_bar"], f"mismatch at wins={wins}"
    assert GS.mirror_promote_rate(0.50, 360, threshold=0.55, trials=20000, seed=5) < 0.06   # FP modest
    assert GS.mirror_promote_rate(0.60, 360, threshold=0.55, trials=8000, seed=5) > 0.9      # strong power


def test_mirror_collapse_never_false_reverts_but_catches_erosion():
    assert GS.mirror_collapse_rate(0.50, 360, z=1.645, margin=0.05, trials=20000, seed=6) < 0.01
    assert GS.mirror_collapse_rate(0.48, 360, z=1.645, margin=0.05, trials=20000, seed=6) < 0.02
    assert GS.mirror_collapse_rate(0.35, 360, z=1.645, margin=0.05, trials=8000, seed=6) > 0.9
    assert GS.mirror_collapse_rate(0.30, 360, z=1.645, margin=0.05, trials=8000, seed=6) > 0.98


def test_force_valve_marginal_window_empty_at_n60():
    """A persistently-vetoed suspect is always point<0.40 (catastrophic), so the marginal auto-
    advance never fires at the HoF's n — the sim that justified simplifying the valve to
    freeze+alert+--hof-override."""
    catastrophic = GS.force_valve_rate(0.30, 60, force_limit=4, marginal_cut=0.40,
                                       marginal_stat="point", n_gens=40, trials=1500, seed=7)
    allmarginal = GS.force_valve_rate(0.30, 60, force_limit=4, marginal_cut=0.0,
                                      marginal_stat="point", n_gens=40, trials=1500, seed=7)
    assert catastrophic["frac_runs_forced"] == 0.0      # never force-advances a real counter
    assert allmarginal["frac_runs_forced"] > 0.9        # mechanism works when everything is 'marginal'


# ── 19b.1 — staged short→full mirror calibration ──────────────────────────────
def test_mirror_gate_verdict_matches_the_rate_functions():
    """The per-trial mirror verdict must equal the gate's own promote / collapse conditions (the
    same ones mirror_promote_rate / mirror_collapse_rate — already pinned to promotion_gate_v2 —
    use), so the staged sim can never drift from the real gate."""
    import numpy as np
    n = 360
    rng = np.random.default_rng(11)
    for tp in (0.50, 0.55, 0.40):
        wins = GS._draw_wins(rng, tp, n, 40000)
        v = GS.mirror_gate_verdict(wins, n)
        p = wins / n
        promote = (p >= 0.55) & (p - 1.645 * np.sqrt(p * (1 - p) / n) > 0.5)
        revert = GS._wilson_upper_vec(wins, n, 1.645) < 0.45
        assert np.array_equal(v == 1, promote)              # PROMOTE branch
        assert np.array_equal(v == -1, revert)              # REVERT branch
        assert not np.any(promote & revert)                 # mutually exclusive
    short = GS._draw_wins(np.random.default_rng(1), 0.70, 90, 2000)   # a STRONG short result …
    assert np.all(GS.mirror_gate_verdict(short, 90) == 0)            # … still only HOLD (n < 360)


def test_mirror_escalate_mask_escalates_borderline_skips_interior():
    import numpy as np
    n = 300
    assert bool(GS.mirror_escalate_mask(np.array([int(0.62 * n)]), n)[0])   # clear promoter → escalate
    assert bool(GS.mirror_escalate_mask(np.array([int(0.38 * n)]), n)[0])   # clear collapse  → escalate
    assert not bool(GS.mirror_escalate_mask(np.array([int(0.50 * n)]), n)[0])  # tight interior → skip
    # asymmetric z: a lenient promote-side z lets a ~0.50 short skip at a smaller n (revert side held)
    n2 = 150
    assert bool(GS.mirror_escalate_mask(np.array([int(0.50 * n2)]), n2, z_promote=1.645)[0])     # wide → escalate
    assert not bool(GS.mirror_escalate_mask(np.array([int(0.50 * n2)]), n2,
                                            z_promote=1.0, revert_band=0.40)[0])                  # lenient → skip


def test_staged_escalated_matches_full_and_skip_is_decision_equivalent():
    # a clear promoter (0.65) almost always escalates → staged == the full mirror, no false-skip
    s = GS.staged_mirror_stats(0.65, 120, 360, trials=8000, seed=3)
    assert s["escalate_rate"] > 0.95 and s["false_skip"] < 0.01
    # at the danger edges (just-promote / just-revert / dead-centre) a skip must never miss a verdict
    for tp in (0.55, 0.45, 0.50):
        r = GS.staged_mirror_stats(tp, 120, 360, trials=8000, seed=3)
        assert r["false_skip"] < 0.01
    # avg_games is bounded by [n_short, full_n] and frac is consistent
    assert 120 <= s["avg_games"] <= 360
    assert abs(s["avg_games_frac"] - s["avg_games"] / 360) < 1e-9


def test_staged_strict_equivalence_yields_negligible_savings():
    """19b.1's headline finding (regression-guarded): under STRICT decision-equivalence (the
    symmetric 0.45/0.55 band) the short mirror almost never skips — the promote bar sits only ~0.05
    above the typical ~0.50 mirror rate, so a short mirror can't rule a promote in/out cheaply. The
    eval saving is negligible, which is WHY 19b.2-.4 were shelved."""
    tooth = [0.50, 0.51, 0.52, 0.53, 0.54, 0.55]
    agg = GS.staged_mirror_aggregate(tooth, 180, 360, z_esc=1.645, trials=8000, seed=4)
    assert agg["mean_cost_frac"] > 0.95         # < 5% of the full-mirror eval saved
    assert agg["worst_false_skip"] < 0.01       # and it IS decision-equivalent (just not worth it)
