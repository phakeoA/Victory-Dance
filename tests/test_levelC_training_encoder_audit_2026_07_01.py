"""
Level C — training+encoder deep-audit fixes (2026-07-01, Workflow wf_bdcfa8e3).

(1) est_stats HP channel clamped to [0,1] (bulky mons like Blissey overshot to ~1.2). (2) the belief
spread-nature heuristic prefers the offence-BOOSTING nature, not merely a non-lowering one (a special spread
was tagged with an Atk-boosting nature). Both are encoder/belief changes; #1 bakes in on a RETRAIN, #2 needs a
RE-EXPORT (stats_estimate is baked into the jsonl at export) — negligible corpus impact.
"""
from __future__ import annotations

import pytest

pytest.importorskip("numpy")
import numpy as np

from v_dance.encoders.state_encoder import StateEncoder


def _mon(species, hp=100.0, est_hp=None):
    m = {"species": species, "base_species": species, "hp_pct": hp, "seen": True,
         "is_fainted": False, "known_moves": ["Fake Out"], "revealed_moves": [],
         "boosts": {}, "status": None}
    if est_hp is not None:
        m["stats_estimate"] = {"mode": "distribution",
                               "stats": {"hp": est_hp, "atk": 100, "def": 100, "spa": 100, "spd": 100, "spe": 100}}
    return m


_EST_HP_CH = 47   # our_a's est-HP channel (base=41..46, then est_stats HP at 47) — verified empirically


def _snap(est_hp):
    return {"our_active": {"our_a": _mon("Blissey", est_hp=est_hp), "our_b": None},
            "opp_active": {"opp_a": None, "opp_b": None},
            "our_bench": [], "opp_bench": [], "field": {}, "side_conditions": {}}


def test_est_stats_hp_clamped_to_unit_range():
    # Blissey-class est HP overshoots _EST_STAT_NORM(300): 362/300=1.21, 600/300=2.0 — both must clamp to 1.0;
    # a sub-300 value is untouched (audit 2026-07-01).
    enc = StateEncoder()
    assert enc.encode_snapshot(_snap(362), turn=1)[_EST_HP_CH] == pytest.approx(1.0)   # was 1.21 before the fix
    assert enc.encode_snapshot(_snap(600), turn=1)[_EST_HP_CH] == pytest.approx(1.0)   # was 2.0
    assert enc.encode_snapshot(_snap(150), turn=1)[_EST_HP_CH] == pytest.approx(0.5)   # unclamped


def test_best_nature_prefers_offence_boost():
    # audit 2026-07-01 (shipped WITH the corpus re-export): a SpA-invested spread must take a SpA-BOOSTING nature
    # (Modest), not merely a non-Atk-lowering one (the old drop-only check picked Brave for a special Torterra).
    from v_dance.parser.belief_state import _best_nature_for_spread
    assert _best_nature_for_spread([11, 0, 0, 28, 0, 27],
                                   [{"nature": "Brave", "pct": 40.0}, {"nature": "Modest", "pct": 30.0}]) == "Modest"
    assert _best_nature_for_spread([0, 28, 0, 0, 0, 27],
                                   [{"nature": "Modest", "pct": 40.0}, {"nature": "Adamant", "pct": 30.0}]) == "Adamant"
    # no boost-matching nature in the distribution → fall back to a non-lowering one (Timid drops atk, not spa).
    assert _best_nature_for_spread([0, 0, 0, 28, 0, 0], [{"nature": "Timid", "pct": 50.0}]) == "Timid"
    # neutral/defensive spread (atk_ev == spa_ev) keeps the modal.
    assert _best_nature_for_spread([0, 0, 4, 0, 0, 0], [{"nature": "Bold", "pct": 60.0}]) == "Bold"
