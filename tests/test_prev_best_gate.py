"""Gate upgrade: a NON-SATURATING, cycle-safe promotion bar.

The scripted-ladder gate goes blind once the policy crushes the scripted opponents
(gen N ≈ gen N+1 ≈ 98% → delta ≈ 0 → "hold" forever).  A head-to-head vs ``prev_best``
(the current accepted BEST, NOT gen N-1) keeps the gate making progress: beating your
strongest past self by a significant margin is a bar that only rises.  Reverting on a
real scripted COLLAPSE still takes precedence (safety).
"""

from __future__ import annotations

import pytest

from v_dance.selfplay.generation import (
    promotion_gate, aggregate_prev_best, aggregate_scripted, GateConfig,
)


# ── backward compatibility: no prev_best ⟹ original scripted-only gate ───────────
def test_no_prevbest_reduces_to_scripted_gate():
    assert promotion_gate(50, 100, 0, 0)[0] == "promote"               # no baseline
    assert promotion_gate(260, 400, 200, 400, GateConfig(z=1.0))[0] == "promote"
    assert promotion_gate(220, 400, 260, 400, GateConfig(z=1.0))[0] == "revert"   # 0.55<0.65
    assert promotion_gate(241, 400, 240, 400, GateConfig(z=1.0))[0] == "hold"


# ── the saturation fix: scripted flat, beat prev_best ⟹ promote ─────────────────
def test_saturated_scripted_but_beats_prevbest_promotes():
    # scripted 0.98 vs 0.98 → delta 0 → scripted "hold"; prev_best 70/100 → beats it.
    v, st = promotion_gate(196, 200, 196, 200, GateConfig(z=1.0),
                           prevbest_wins=70, prevbest_games=100)
    assert v == "promote"
    assert st["scripted_verdict"] == "hold"          # scripted alone would NOT promote
    assert st["verdict_reason"] == "beats_prev_best"
    assert st["prevbest"]["beats_best"] is True


def test_saturated_scripted_and_prevbest_coinflip_holds():
    v, st = promotion_gate(196, 200, 196, 200, GateConfig(z=1.0),
                           prevbest_wins=50, prevbest_games=100)
    assert v == "hold"
    assert st["prevbest"]["beats_best"] is False


def test_marginal_prevbest_win_does_not_promote_cycle_safety():
    # a within-noise 52/100 head-to-head must NOT promote — else non-transitive
    # RPS cycles would promote forever.  Significance (z-band) gates it out.
    v, _ = promotion_gate(196, 200, 196, 200, GateConfig(z=1.0),
                          prevbest_wins=52, prevbest_games=100)
    assert v == "hold"


# ── collapse safety: scripted regression reverts even if it edges prev_best ──────
def test_scripted_collapse_reverts_despite_prevbest_edge():
    v, st = promotion_gate(200, 400, 260, 400, GateConfig(z=1.0),
                           prevbest_wins=70, prevbest_games=100)
    assert v == "revert"
    assert st["verdict_reason"] == "scripted_collapse"


# ── scripted improvement still promotes (reason recorded) ───────────────────────
def test_scripted_improvement_promotes_with_reason():
    v, st = promotion_gate(300, 400, 200, 400, GateConfig(z=1.0),
                           prevbest_wins=50, prevbest_games=100)
    assert v == "promote"
    assert st["verdict_reason"] == "scripted"


# ── aggregate helpers ───────────────────────────────────────────────────────────
def test_aggregate_prev_best_extracts_mirror_and_scripted_excludes_it():
    results = {"random": (20, 30), "max_damage": (18, 30), "heuristic": (15, 30),
               "prev_best": (7, 10)}
    assert aggregate_prev_best(results) == (7, 10)
    # scripted aggregation must NOT include the prev_best mirror
    assert aggregate_scripted(results) == (20 + 18 + 15, 90)


def test_aggregate_prev_best_absent_is_zero():
    assert aggregate_prev_best({"random": (1, 2)}) == (0, 0)


# ── wiring: run_generation feeds the BEST (not gen N-1) to eval + the gate ───────
class _FakeLeague:
    def __init__(self):
        self.snapshots = []
        self.latest_path = None
        self.pfsp_resets = 0
    def admit(self, sid, path, gen, elo, is_champion=False):
        self.snapshots.append((sid, path, gen, elo, is_champion))
    def reset_pfsp(self):
        self.pfsp_resets += 1
    def prune(self, cap, keep_recent=6):
        return []


class _FakeTrainer:
    def warmup_critic(self, *a, **k):
        pass
    def ppo_update(self, *a, **k):
        return {"halted": False}


def test_run_generation_passes_champion_path_and_promotes_on_bar_clear():
    """run_generation (v2 gate) passes the CHAMPION (best_path) to the eval as the mirror anchor,
    and promotes (beat_champion) when the candidate clears the 0.55 bar over >= min_h2h_games (360)
    mirror games — advancing the champion and resetting the head-to-head history."""
    from v_dance.selfplay.generation import run_generation, GenConfig, GenerationHistory
    league = _FakeLeague()
    history = GenerationHistory()
    history.best_path = "gen0.pt"          # the current accepted CHAMPION (mirror anchor)
    history.best_scripted = (140, 200)

    seen = {}

    def eval_fn(path, prev_best_path):
        seen["prev_best_path"] = prev_best_path
        # healthy scripted + the candidate beats the champion 270/360 (0.75 >= 0.55, >= min_h2h_games)
        return ({"random": (60, 67), "max_damage": (60, 67), "heuristic": (60, 66),
                 "prev_best": (270, 360)}, 1500.0)

    rep = run_generation(object(), _FakeTrainer(), league, history,
                         collect_fn=lambda ac, lg, gen: ([], {}),
                         eval_fn=eval_fn, save_fn=lambda ac, gen: "candidate.pt",
                         cfg=GenConfig())

    assert seen["prev_best_path"] == "gen0.pt"          # the CHAMPION is the mirror anchor
    assert rep["verdict"] == "promote" and rep["reason"] == "beat_champion"
    assert league.pfsp_resets == 1                       # latest changed → PFSP reset
    assert history.best_path == "candidate.pt"          # champion advanced to the candidate
    assert history.h2h_history == []                    # reset on champion advance


def test_h2h_se_is_null_anchored_not_two_proportion():
    """The head-to-head bar tests the candidate's mirror win-rate against the FIXED 0.5
    null, so SE = sqrt(0.25/n) — NOT the inflated two-proportion sqrt(p(1-p)/n + 0.25/n).
    A 60/100 mirror at z=1.645 clears the corrected bar (margin_lo>0) but FAILED under the
    old sqrt(2)-inflated SE — i.e. the fix loosens the head-to-head to what z/min_delta mean."""
    import math
    v, st = promotion_gate(196, 200, 196, 200, GateConfig(z=1.645),
                           prevbest_wins=60, prevbest_games=100)
    assert st["prevbest"]["se"] == pytest.approx(math.sqrt(0.25 / 100))   # 0.05, null-anchored
    assert st["prevbest"]["beats_best"] is True
    assert v == "promote" and st["verdict_reason"] == "beats_prev_best"
    # the OLD inflated SE (~0.07) would have given margin_lo<0 → a (wrong) HOLD here:
    old_se = math.sqrt(0.6 * 0.4 / 100 + 0.25 / 100)
    assert (0.60 - 0.5) - 1.645 * old_se < 0


def test_run_generation_use_prev_best_false_skips_mirror_and_holds():
    """--no-prev-best (GateConfig.use_prev_best=False): run_generation passes prev_best_path=None
    so the head-to-head mirror NEVER runs. With no bar (no mirror games), no plateau history, and
    no scripted collapse, the v2 gate HOLDS — the champion stays frozen ('freeze past gen 0')."""
    from v_dance.selfplay.generation import (
        run_generation, GenConfig, GenerationHistory,
    )
    league = _FakeLeague()
    history = GenerationHistory()
    history.best_path = "gen0.pt"          # a champion EXISTS, but the mirror is disabled
    history.best_scripted = (196, 200)

    seen = {}

    def eval_fn(path, prev_best_path):
        seen["prev_best_path"] = prev_best_path
        results = {"random": (65, 66), "max_damage": (65, 66), "heuristic": (66, 68)}
        if prev_best_path is not None:      # the mirror only runs when the bar is ON
            results["prev_best"] = (180, 240)
        return results, 1500.0

    rep = run_generation(object(), _FakeTrainer(), league, history,
                         collect_fn=lambda ac, lg, gen: ([], {}),
                         eval_fn=eval_fn, save_fn=lambda ac, gen: f"gen{gen}.pt",
                         cfg=GenConfig(gate=GateConfig(use_prev_best=False)))

    assert seen["prev_best_path"] is None               # mirror skipped → no head-to-head games
    assert rep["verdict"] == "hold" and rep["reason"] == "hold"   # no bar, no collapse → hold
    assert league.pfsp_resets == 0                      # no champion change → no PFSP reset


def test_run_generation_plateau_backstop_fires_and_resets_history():
    """Multi-gen v2 wiring: a flat ~58% mirror (below the 70% bar, not losing) eventually trips
    the PLATEAU backstop — validating that run_generation records each gen's h2h BEFORE the gate
    (so the window fills) and RESETS the history when the champion re-anchors."""
    from v_dance.selfplay.generation import run_generation, GenConfig, GenerationHistory
    league = _FakeLeague()
    history = GenerationHistory()
    history.best_path = "champ.pt"
    history.best_scripted = (180, 200)
    history.scripted_high_water = 0.60      # floor set; a flat 58% mirror won't collapse

    def eval_fn(path, prev_best_path):
        return ({"random": (60, 67), "max_damage": (60, 67), "heuristic": (60, 66),
                 "prev_best": (round(0.58 * 240), 240)}, 1500.0)   # flat ~58%, >= min_h2h_games

    reasons = []
    for _ in range(12):
        rep = run_generation(object(), _FakeTrainer(), league, history,
                             collect_fn=lambda ac, lg, gen: ([], {}), eval_fn=eval_fn,
                             save_fn=lambda ac, gen: f"g{gen}.pt", cfg=GenConfig())
        reasons.append(rep["reason"])

    assert "plateau_reanchor" in reasons                 # the backstop fired on the flat climb
    assert reasons[:9] == ["hold"] * 9                   # held while the window filled (< 2*window)
    assert history.h2h_history == [] or len(history.h2h_history) < 10   # reset on the re-anchor
