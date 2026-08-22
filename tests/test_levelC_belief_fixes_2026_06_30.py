"""
Level C — belief audit fixes (2026-06-30): consumed-item not shown as held, observe-damage idempotency +
non-KO (censoring) gate, speed-σ absolute floor, category/damage double-count avoidance.
"""
from __future__ import annotations

from v_dance.parser.belief_state import BeliefState
from v_dance.parser.match_belief import MatchBelief


def _mb():
    return MatchBelief(BeliefState())


def test_consumed_item_not_collapsed_in_belief():
    mb = _mb()
    mb.ingest_mon({"base_species": "Incineroar", "species": "Incineroar",
                   "known_item": "Focus Sash", "item_consumed": True, "revealed_moves": []})
    block = mb.block_for("Incineroar")
    assert block is not None
    items = block.get("items") or []
    # a CONSUMED Sash must NOT collapse the held-item belief to {Focus Sash, p=1.0}
    assert not (len(items) == 1 and items[0].get("revealed") and items[0].get("name") == "Focus Sash")


def test_held_item_still_collapses():
    mb = _mb()
    mb.ingest_mon({"base_species": "Incineroar", "species": "Incineroar",
                   "known_item": "Sitrus Berry", "item_consumed": False, "revealed_moves": []})
    items = mb.block_for("Incineroar").get("items") or []
    assert items and items[0].get("revealed") and items[0].get("name") == "Sitrus Berry"


def _dmg_args():
    return dict(mu_ref=0.3, ref_main=50.0, ref_def_stat=120.0, ref_hp=200.0,
                category="physical", obs_frac_of_max=0.3)


def test_observe_damage_idempotent_on_event_key():
    mb = _mb()
    for _ in range(3):
        mb.observe_damage_taken("Incineroar", event_key="turn5:opp_a", **_dmg_args())
    obs = mb._mons[mb._key("Incineroar")]
    assert len(obs.damage_constraints) == 1            # same event folded once


def test_observe_damage_skips_ko():
    mb = _mb()
    mb.observe_damage_taken("Incineroar", was_ko=True, **_dmg_args())
    assert mb._key("Incineroar") not in mb._mons or not mb._mons[mb._key("Incineroar")].damage_constraints


def test_speed_sigma_has_absolute_floor():
    mb = _mb()
    mb.observe_speed_bound("Incineroar", threshold_base_spe=100.0, faster=True, sigma_spe=0.1)
    sig = mb._mons[mb._key("Incineroar")].speed_constraints[0]["sigma"]
    assert sig >= 2.0                                   # tiny analyzer σ floored
