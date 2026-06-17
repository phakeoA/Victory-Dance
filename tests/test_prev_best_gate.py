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
    def admit(self, sid, path, gen, elo):
        self.snapshots.append((sid, path, gen, elo))


class _FakeTrainer:
    def warmup_critic(self, *a, **k):
        pass
    def ppo_update(self, *a, **k):
        return {"halted": False}


def test_run_generation_passes_best_path_and_promotes_on_saturated_beat():
    from v_dance.selfplay.generation import (
        run_generation, GenConfig, GenerationHistory,
    )
    league = _FakeLeague()
    history = GenerationHistory()
    history.best_path = "gen0.pt"          # an already-accepted best (the prev_best)
    history.best_scripted = (196, 200)     # saturated scripted baseline (0.98)

    seen = {}

    def collect_fn(ac, lg, gen):
        return [], {}

    def save_fn(ac, gen):
        return f"gen{gen}.pt"

    def eval_fn(path, prev_best_path):
        seen["prev_best_path"] = prev_best_path
        # scripted saturated (≈0.98) + candidate beats prev_best 70/100
        return ({"random": (65, 66), "max_damage": (65, 66), "heuristic": (66, 68),
                 "prev_best": (70, 100)}, 1500.0)

    rep = run_generation(object(), _FakeTrainer(), league, history,
                         collect_fn=collect_fn, eval_fn=eval_fn, save_fn=save_fn,
                         cfg=GenConfig(gate=GateConfig(z=1.0)))

    assert seen["prev_best_path"] == "gen0.pt"          # the BEST, not gen N-1
    assert rep["verdict"] == "promote"
    assert rep["gate"]["verdict_reason"] == "beats_prev_best"
