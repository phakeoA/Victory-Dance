"""Level-C B1: MODEL-DRIVEN accounting must understand the search source.

Regression for the smoke (b1_smoke) reporting bugs: a belief-weighted-search pick is tagged
``source="search"`` but was NOT in ``_MODEL_DRIVEN_SOURCES`` (so it counted as non-model), and the
``search_*`` fire/fallback DIAGNOSTIC counters were summed into ``non_model`` (double-counting) —
together dragging MODEL-DRIVEN to ~18% on a clean search run. These tests pin the fix across all
three sites that measure model-driven %: reward.py, generation.py, and game_runner.phase0_report.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("numpy")
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
from v_dance.encoders.state_encoder import get_state_dim, get_action_dim  # noqa: E402
from v_dance.selfplay.collector import TrajectoryCollector  # noqa: E402
from v_dance.selfplay.schema import PASS_ACTION  # noqa: E402
from v_dance.selfplay import game_runner as G  # noqa: E402

SD, AD = get_state_dim(), get_action_dim()


# ── shared trajectory helper (mirrors test_selfplay_game_runner._game) ─────────
def _game(tag, p1_won, n=3, turn=5):
    cp1, cp2 = TrajectoryCollector(tag, "p1"), TrajectoryCollector(tag, "p2")
    for c in (cp1, cp2):
        for i in range(n):
            c.add_step(state=np.zeros(SD, np.float32), action_s0=0, action_s1=PASS_ACTION,
                       mask_s0=[1] * AD, logprob=-1.0, value=0.0, turn=i + 1)
    t1 = G.finalize_trajectory(cp1, won=p1_won, terminal_type="win" if p1_won else "loss",
                               own_team=["a"], n_turns=turn)
    t2 = G.finalize_trajectory(cp2, won=not p1_won, terminal_type="loss" if p1_won else "win",
                               own_team=["b"], n_turns=turn)
    return t1, t2


def _games(pattern, turn=5):
    return [_game(f"b{i}", w, turn=turn) for i, w in enumerate(pattern)]


# ── reward.py: the collector's HARD-FAIL guard ─────────────────────────────────
def test_reward_model_driven_set_includes_search():
    from v_dance.selfplay.reward import _MODEL_DRIVEN_SOURCES
    assert "search" in _MODEL_DRIVEN_SOURCES


def test_recording_gate_mirrors_model_driven_set():
    # The step-RECORDING gate (_record_rl_decision) MUST equal the MODEL-DRIVEN accounting set, else a
    # model-driven source like 'search' is counted but its turns are never recorded -> n_steps=0,
    # MODEL-DRIVEN reads 0% on a legit search run, and the search corpus is empty (audit 2026-06-30).
    from v_dance.selfplay.reward import _MODEL_DRIVEN_SOURCES
    assert G._MODEL_SOURCES == _MODEL_DRIVEN_SOURCES
    assert "search" in G._MODEL_SOURCES
    # the dead, never-updated module-level mirror is gone (was a drift hazard).
    assert not hasattr(G, "MODEL_DRIVEN_SOURCES")


def test_model_driven_fraction_counts_search_and_strips_diagnostics():
    from v_dance.selfplay.reward import model_driven_fraction
    # bare "search" is a model-grounded decision; search_*/belief_* are DIAGNOSTIC, not decisions.
    sc = {"search": 97, "model": 2, "forced_default": 1,
          "search_search": 97, "search_fallback": 5, "belief_dmg_used": 40}
    # numerator = search(97) + model(2) = 99; denom = 99 + forced_default(1) = 100  → 0.99
    assert model_driven_fraction(sc) == pytest.approx(0.99)


def test_assert_model_driven_passes_on_search_heavy_run():
    from v_dance.selfplay.reward import assert_model_driven
    # Pre-fix this raised (search counted as non-model → frac≈0.005). Post-fix it passes.
    frac = assert_model_driven(
        {"search": 199, "model": 1, "search_search": 199, "search_fallback": 0}, 0.99)
    assert frac == pytest.approx(1.0)


# ── generation.py: the per-gen edge-event tally (RL / B3 train-with-search) ─────
def test_generation_model_driven_mirror_includes_search():
    from v_dance.selfplay.generation import _MODEL_DRIVEN_SOURCES
    assert "search" in _MODEL_DRIVEN_SOURCES


def test_fs_monitor_excludes_search_and_diagnostics():
    from v_dance.selfplay.generation import fs_monitor_counts
    fs = fs_monitor_counts({"search": 500, "search_search": 500, "search_fallback": 2,
                            "belief_dmg_used": 9, "model": 10, "forced_default": 3, "retry": 1})
    assert fs["forced_default"] == 3 and fs["retry"] == 1          # real edges tallied
    assert "search" not in fs and "search_search" not in fs        # search source/diag excluded
    assert "belief_dmg_used" not in fs                              # belief diag excluded
    assert fs["total"] == 4                                         # forced_default(3) + retry(1) only


# ── game_runner.phase0_report: the A/B plumbing gate ───────────────────────────
def test_phase0_surfaces_search_diag_and_excludes_from_non_model():
    # 6 games x 2 trajs x 3 steps = 36 executed (de-duped) model steps. source_counts carries the bare
    # "search" decision source + the search_* DIAGNOSTIC counters + one forced_default edge.
    sc = {"search": 500, "search_search": 500, "search_fallback": 3,
          "model": 3, "forced_default": 2}
    rep = G.phase0_report(_games([True, False] * 3), sc, min_games=4)
    assert rep["search_diag"] == {"search": 500, "fallback": 3}    # surfaced (stripped prefix)
    assert rep["non_model_executed"] == 2                          # only forced_default; search excluded
    assert rep["model_driven"] == pytest.approx(36 / 38)          # 36 / (36 + 2)
    assert rep["model_driven_ok"] is False or rep["model_driven"] > 0.9   # well above the spurious ~0.18


def test_phase0_no_search_keys_leaves_empty_diag():
    rep = G.phase0_report(_games([True, False] * 3), {"model": 36}, min_games=4)
    assert rep["search_diag"] == {} and rep["model_driven"] == 1.0


def test_print_phase0_report_includes_search_feed(capsys):
    rep = G.phase0_report(_games([True, False] * 3),
                          {"search": 100, "search_search": 100, "search_fallback": 4}, min_games=4)
    G.print_phase0_report(rep)
    out = capsys.readouterr().out
    assert "search feed" in out and "search decisions" in out and "fallbacks" in out
