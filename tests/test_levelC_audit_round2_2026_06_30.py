"""
Level C — round-2 codebase-audit fixes (2026-06-30, find->verify Workflow wf_000370a6).

One file collecting the regression tests for the audit findings that don't belong to an existing
suite: the white_box faint miscount, the direction-tagged damage loglik (+ both narrower guards), the
live belief-feed species/consumed-item guards, and the de-duplicated belief floor constant. (The
self-play recording-gate fix is pinned in test_levelC_phase0_search_accounting_2026_06_30.py.)
"""
from __future__ import annotations

import math

import pytest

pytest.importorskip("numpy")


# ── #2 white_box_sim: unrevealed bench (hp_pct=None) is NOT fainted ─────────────
def test_is_fainted_and_count_fainted_side():
    from v_dance.encoders.white_box_sim import is_fainted, _count_fainted_side
    assert is_fainted(None) is False
    assert is_fainted({"hp_pct": None}) is False                      # unrevealed bench = full HP, not faint
    assert is_fainted({"hp_pct": None, "is_fainted": False}) is False
    assert is_fainted({"hp_pct": 100.0}) is False
    assert is_fainted({"hp_pct": 0.0}) is True                        # real KO
    assert is_fainted({"is_fainted": True, "hp_pct": 50.0}) is True   # flagged KO
    snap = {
        "opp_active": {"opp_a": {"hp_pct": 100.0}, "opp_b": {"hp_pct": 40.0}},
        "opp_bench": [{"hp_pct": None}, {"hp_pct": None}, {"hp_pct": 0.0, "is_fainted": True}],
    }
    # only the one genuinely-KO'd bench mon counts; the two unrevealed (None) bench mons do NOT.
    assert _count_fainted_side(snap, "opp") == 1


# ── #4/#5 direction-tagged damage loglik + direction-aware narrower guards ───────
from v_dance.parser.belief_state import BeliefState, NATURE_BOOSTS  # noqa: E402
from v_dance.parser.match_belief import MatchBelief, _spread_key     # noqa: E402


@pytest.fixture(scope="module")
def belief() -> "BeliefState":
    try:
        return BeliefState()
    except Exception as e:  # pragma: no cover
        pytest.skip(f"BeliefState unavailable: {e}")


def _drops(nature, stat):
    return NATURE_BOOSTS.get(nature, ("", ""))[1] == stat


def _drop_mass(block, stat):
    return sum(s["p"] for s in (block.get("spreads") or []) if _drops(s["nature"], stat))


def _find_mixed_for_stat(belief, stat):
    for sp in belief.all_pokemon():
        spreads = belief.spread_distribution(sp, top_k=5)
        if len(spreads) < 2:
            continue
        nats = {s["nature"] for s in spreads}
        if any(_drops(n, stat) for n in nats) and any(not _drops(n, stat) for n in nats):
            return sp
    return None


def _uniform_loglik(block):
    """A neutral per-spread loglik (all 0.0) covering every spread → the damage narrower is a no-op, so
    any change is attributable solely to the category/item nature prior (isolates the guard behaviour)."""
    return {_spread_key(s): 0.0 for s in (block.get("spreads") or [])}


def test_observe_damage_loglik_tags_direction_and_stat(belief):
    mb = MatchBelief(belief)
    sp = next(iter(belief.all_pokemon()))
    mb.observe_damage_loglik(sp, {("Jolly", ()): 0.0}, direction="off", stat="spa")
    c = mb._mons[mb._key(sp)].damage_constraints[0]
    assert c["mode"] == "loglik" and c["direction"] == "off" and c["stat"] == "spa"


def test_defensive_loglik_does_not_cancel_category_narrowing(belief):
    sp = _find_mixed_for_stat(belief, "atk")
    if sp is None:
        pytest.skip("no mixed -atk species")
    base = belief.belief_block(sp, top_k=5)
    mb = MatchBelief(belief)
    mb.observe_move(sp, "Earthquake")                                 # standard physical → offense={atk}
    mb.observe_damage_loglik(sp, _uniform_loglik(base), direction="def")   # opp bulk, says nothing about atk
    narrowed = mb.block_for(sp)
    # the DEFENSIVE loglik must NOT trip the guard → the -Atk category penalty still fires.
    assert _drop_mass(narrowed, "atk") < _drop_mass(base, "atk") - 1e-9


def test_offensive_loglik_on_stat_still_guards(belief):
    sp = _find_mixed_for_stat(belief, "atk")
    if sp is None:
        pytest.skip("no mixed -atk species")
    base = belief.belief_block(sp, top_k=5)
    mb = MatchBelief(belief)
    mb.observe_move(sp, "Earthquake")
    mb.observe_damage_loglik(sp, _uniform_loglik(base), direction="off", stat="atk")  # already pins atk
    got = mb.block_for(sp)
    # an OFFENSIVE loglik on atk DOES cover the stat → category prior defers → spreads == prior (loglik uniform).
    # abs=1e-3: block_for() renormalizes the distribution to sum-1, while belief_block() returns 4-decimal-rounded
    # p's that sum to ~0.9999; that ~1e-4 normalization gap lands on the largest (Timid, -Atk) spread and is NOT
    # a narrowing leak (verified: identical natures/EVs on every spread — only one p shifts by 1e-4). A genuine
    # category-narrowing leak would move drop-mass by the multiplicative -Atk penalty (≫1e-3).
    assert _drop_mass(got, "atk") == pytest.approx(_drop_mass(base, "atk"), abs=1e-3)


def test_defensive_loglik_does_not_cancel_item_narrowing(belief):
    sp = _find_mixed_for_stat(belief, "atk")
    if sp is None:
        pytest.skip("no mixed -atk species")
    base = belief.belief_block(sp, top_k=5)
    mb = MatchBelief(belief)
    mb.observe_item(sp, "Choice Band")                               # boosts atk → -Atk natures incoherent
    mb.observe_damage_loglik(sp, _uniform_loglik(base), direction="def")
    narrowed = mb.block_for(sp)
    assert _drop_mass(narrowed, "atk") < _drop_mass(base, "atk") - 1e-9


# ── #8 item-classification frozensets must not drift between the two modules ─────
def test_item_classification_sets_do_not_drift():
    # mechanic_tags (encoder feature bits) and battle_mechanics (damage-calc primitives) each keep a COPY
    # of the item-classification frozensets — a deliberate replication, because the mechanic_tags ->
    # battle_mechanics -> encoder_layout -> mechanic_tags import cycle blocks a shared import. They MUST
    # stay identical (else the encoded item-tag bit and the damage band disagree for the same mon). This
    # guard fails the instant a one-sided edit drifts them (audit 2026-06-30).
    import v_dance.encoders.mechanic_tags as mt
    import v_dance.encoders.battle_mechanics as bm
    shared = [n for n in vars(mt)
              if isinstance(getattr(mt, n), frozenset) and isinstance(getattr(bm, n, None), frozenset)]
    for req in ("_STATUS_CURE_BERRIES", "_CONTACT_PUNISH_ITEMS", "_EFFECT_SHIELD_ITEMS",
                "_DROP_GUARD_ITEMS", "_LUCK_ITEMS", "_TYPE_BOOST_ITEMS"):
        assert req in shared, f"{req} no longer shared between the modules — audit assumption broke"
    mismatched = {n: sorted(getattr(mt, n) ^ getattr(bm, n)) for n in shared
                  if getattr(mt, n) != getattr(bm, n)}
    assert not mismatched, f"item-classification frozensets DRIFTED between modules: {mismatched}"


# ── round-3 audit fixes ─────────────────────────────────────────────────────────
def test_item_narrowing_fires_for_mixed_choice_attacker(belief):
    # round-3: a Choice-item holder that revealed BOTH a physical AND a special move → category narrower
    # BAILS (mixed offense), so the item narrower must NOT defer — it's the only channel left to shave the
    # incoherent -Atk natures off a Choice Band. (Guard was `boosted in offense`; now `offense == {boosted}`.)
    sp = _find_mixed_for_stat(belief, "atk")
    if sp is None:
        pytest.skip("no mixed -atk species")
    base = belief.belief_block(sp, top_k=5)
    mb = MatchBelief(belief)
    mb.observe_move(sp, "Earthquake")            # physical
    mb.observe_move(sp, "Ice Beam")              # special → offensive_categories == {'atk','spa'} (mixed)
    mb.observe_item(sp, "Choice Band")           # boosts atk → item coherence still applies to -Atk natures
    narrowed = mb.block_for(sp)
    assert _drop_mass(narrowed, "atk") < _drop_mass(base, "atk") - 1e-9


def test_fs_monitor_excludes_finalize_failed():
    # round-3: finalize_failed is post-game bookkeeping (reward._NON_DECISION_COUNTERS), NOT an fs-monitor edge
    # event — it must be excluded from the RL edge tally too (the two classifications had drifted).
    from v_dance.selfplay.generation import fs_monitor_counts, _NON_EDGE_SOURCE_KEYS
    assert "finalize_failed" in _NON_EDGE_SOURCE_KEYS
    fs = fs_monitor_counts({"finalize_failed": 3, "forced_default": 2, "model": 10})
    assert "finalize_failed" not in fs and fs["forced_default"] == 2 and fs["total"] == 2
