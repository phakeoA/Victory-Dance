"""
Level C — revealed-item → spread COHERENCE reconditioning (the 1 deferred B1-audit item, 2026-06-30).

Covers MatchBelief._narrow_spreads_by_item: a revealed Choice Scarf/Band/Specs down-weights spreads whose
nature LOWERS the item's boosted stat (competitively incoherent), needing no joint usage data. Conservative —
never zeroes a spread; a no-op for non-role items; and DEFERS (double-count guard) when the same stat is
already constrained by an observed move (category), damage (A2) or speed (A3) event.

Uses the real (active-format) BeliefState so narrowing runs against real Pikalytics spreads; the test species
is discovered dynamically so the suite does not rot when the meta data changes.
"""
from __future__ import annotations

import pytest

from v_dance.parser.belief_state import BeliefState, NATURE_BOOSTS
from v_dance.parser.match_belief import MatchBelief, _ITEM_BOOST_STAT, _ITEM_NATURE_PENALTY


@pytest.fixture(scope="module")
def belief() -> BeliefState:
    try:
        return BeliefState()
    except Exception as e:  # pragma: no cover - data must be present in-repo
        pytest.skip(f"BeliefState unavailable: {e}")


# ── helpers ──────────────────────────────────────────────────────────────────
def _drops(nature: str, stat: str) -> bool:
    return NATURE_BOOSTS.get(nature, ("", ""))[1] == stat


def _drop_mass(block: dict, stat: str) -> float:
    return sum(s["p"] for s in (block.get("spreads") or []) if _drops(s["nature"], stat))


def _find_mixed_for_stat(belief: BeliefState, stat: str):
    """A species with >=2 spreads where at least one nature DROPS `stat` and at least one does NOT — so an
    item that boosts `stat` has an incoherent spread to re-weight."""
    for sp in belief.all_pokemon():
        spreads = belief.spread_distribution(sp, top_k=5)
        if len(spreads) < 2:
            continue
        nats = {s["nature"] for s in spreads}
        if any(_drops(n, stat) for n in nats) and any(not _drops(n, stat) for n in nats):
            return sp
    return None


def _known_species(belief: BeliefState) -> str:
    sp = next((s for s in belief.all_pokemon() if belief.spread_distribution(s, top_k=5)), None)
    assert sp is not None, "no usable species in belief data"
    return sp


# ── the item→stat map ──────────────────────────────────────────────────────────
def test_item_boost_stat_map():
    assert _ITEM_BOOST_STAT == {"choicescarf": "spe", "choiceband": "atk", "choicespecs": "spa"}
    assert 0.0 < _ITEM_NATURE_PENALTY < 1.0


# ── the core narrowing, per Choice item ─────────────────────────────────────────
@pytest.mark.parametrize("item,stat", [
    ("Choice Scarf", "spe"), ("Choice Band", "atk"), ("Choice Specs", "spa"),
])
def test_choice_item_down_weights_incoherent_nature(belief, item, stat):
    sp = _find_mixed_for_stat(belief, stat)
    if sp is None:
        pytest.skip(f"no species with mixed -{stat} spreads in current data")
    base = belief.belief_block(sp, top_k=5)
    mb = MatchBelief(belief)
    mb.observe_item(sp, item)
    narrowed = mb.block_for(sp)
    # the stat-lowering spreads lose relative mass; the boosted stat's expected value does not drop.
    assert _drop_mass(narrowed, stat) < _drop_mass(base, stat) - 1e-9
    assert narrowed["expected_stats"][stat] >= base["expected_stats"][stat] - 1e-6
    # conservative: nothing zeroed, still a valid distribution of the same cardinality.
    assert all(s["p"] > 0 for s in narrowed["spreads"])
    assert abs(sum(s["p"] for s in narrowed["spreads"]) - 1.0) < 1e-2
    assert len(narrowed["spreads"]) == len(base["spreads"])
    # the item distribution still collapses to the reveal (unchanged behaviour).
    assert narrowed["items"] == [{"name": item, "p": 1.0, "revealed": True}]


# ── non-role items are a pure no-op on spreads ──────────────────────────────────
@pytest.mark.parametrize("item", ["Leftovers", "Life Orb", "Assault Vest"])
def test_non_role_item_is_noop_on_spreads(belief, item):
    sp = _find_mixed_for_stat(belief, "spe") or _known_species(belief)
    base = belief.belief_block(sp, top_k=5)
    mb = MatchBelief(belief)
    mb.observe_item(sp, item)
    got = mb.block_for(sp)
    assert got["spreads"] == base["spreads"]                 # spreads untouched
    assert got["expected_stats"] == base["expected_stats"]
    assert got["items"][0]["name"] == item                   # item still collapses


# ── double-count guards: defer when another channel already constrained the stat ─
def test_guard_observed_move_suppresses_band(belief):
    """A revealed Choice Band adds NOTHING once a physical move already drove the Atk category narrowing."""
    sp = _find_mixed_for_stat(belief, "atk")
    if sp is None:
        pytest.skip("no species with mixed -atk spreads in current data")
    mb_move = MatchBelief(belief)
    mb_move.observe_move(sp, "Earthquake")                    # category narrowing fires on atk
    only_move = mb_move.block_for(sp)
    mb_both = MatchBelief(belief)
    mb_both.observe_move(sp, "Earthquake")
    mb_both.observe_item(sp, "Choice Band")                   # item guard → no second atk penalty
    both = mb_both.block_for(sp)
    assert both["spreads"] == only_move["spreads"]
    assert both["expected_stats"] == only_move["expected_stats"]


def test_guard_speed_constraint_suppresses_scarf(belief):
    """A revealed Choice Scarf adds NOTHING once a speed-order observation already constrained Spe."""
    sp = _find_mixed_for_stat(belief, "spe")
    if sp is None:
        pytest.skip("no species with mixed -spe spreads in current data")
    mb_spd = MatchBelief(belief)
    mb_spd.observe_speed_bound(sp, threshold_base_spe=100.0, faster=True)
    only_spd = mb_spd.block_for(sp)
    mb_both = MatchBelief(belief)
    mb_both.observe_speed_bound(sp, threshold_base_spe=100.0, faster=True)
    mb_both.observe_item(sp, "Choice Scarf")                  # item guard → no extra spe penalty
    both = mb_both.block_for(sp)
    assert both["spreads"] == only_spd["spreads"]
    assert both["expected_stats"] == only_spd["expected_stats"]


# ── consumed item never narrows (defence-in-depth via ingest_mon) ───────────────
def test_consumed_item_does_not_narrow(belief):
    sp = _find_mixed_for_stat(belief, "spe") or _known_species(belief)
    base = belief.belief_block(sp, top_k=5)
    mb = MatchBelief(belief)
    mb.ingest_mon({"species": sp, "base_species": sp,
                   "known_item": "Choice Scarf", "item_consumed": True})
    got = mb.block_for(sp)
    assert got["spreads"] == base["spreads"]                  # consumed → known_item stays None → no narrowing
