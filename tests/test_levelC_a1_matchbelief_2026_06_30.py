"""
Level C / Component A1 — MatchBelief unit tests (2026-06-30).

Covers the within-game belief tracker:
  * accumulation of move/item/ability reveals (+ dedup, persistence),
  * parity with BeliefState.belief_block when nothing is observed,
  * the NEW move-category spread narrowing (fires on single-category standard
    offense; conservative — never zeroes a spread; correctly EXCLUDES cross-stat
    moves like Body Press / Foul Play / fixed-damage),
  * the drop-in enrich_mon path.

Uses the real (active-format) BeliefState so the narrowing is exercised against
real Pikalytics spreads; the test species is discovered dynamically so the suite
does not rot when the meta data changes.
"""
from __future__ import annotations

import pytest

from v_dance.parser.belief_state import BeliefState, NATURE_BOOSTS
from v_dance.parser.match_belief import (
    MatchBelief,
    _offensive_stat_for_move,
    _move_category,
)


@pytest.fixture(scope="module")
def belief() -> BeliefState:
    try:
        return BeliefState()
    except Exception as e:  # pragma: no cover - data must be present in-repo
        pytest.skip(f"BeliefState unavailable: {e}")


# ── helpers ────────────────────────────────────────────────────────────────────
def _drops(nature: str, stat: str) -> bool:
    return NATURE_BOOSTS.get(nature, ("", ""))[1] == stat


def _find_mixed_nature_species(belief: BeliefState) -> str | None:
    """A species with >=2 spreads where at least one nature drops Atk and at least
    one does not — so Atk-category narrowing has something to re-weight."""
    for sp in belief.all_pokemon():
        spreads = belief.spread_distribution(sp, top_k=5)
        if len(spreads) < 2:
            continue
        natset = {s["nature"] for s in spreads}
        if any(_drops(n, "atk") for n in natset) and any(not _drops(n, "atk") for n in natset):
            return sp
    return None


def _known_species(belief: BeliefState) -> str:
    sp = next((s for s in belief.all_pokemon() if belief.spread_distribution(s, top_k=5)), None)
    assert sp is not None, "no usable species in belief data"
    return sp


def _atk_drop_mass(block: dict) -> float:
    return sum(s["p"] for s in (block.get("spreads") or []) if _drops(s["nature"], "atk"))


# ── move-category helper classification (the conservative core) ─────────────────
def test_offensive_stat_classification():
    assert _move_category("Earthquake") == "physical"
    assert _move_category("Ice Beam") == "special"
    assert _offensive_stat_for_move("Earthquake") == "atk"
    assert _offensive_stat_for_move("Ice Beam") == "spa"
    # Special move that hits Def still scales with the user's SpA -> kept.
    assert _offensive_stat_for_move("Psyshock") == "spa"
    # Stat-independent moves reveal nothing about the spread.
    assert _offensive_stat_for_move("Body Press") is None      # uses Defense
    assert _offensive_stat_for_move("Foul Play") is None       # uses target's Atk
    assert _offensive_stat_for_move("Seismic Toss") is None    # fixed = level
    assert _offensive_stat_for_move("Protect") is None         # status


# ── accumulation ────────────────────────────────────────────────────────────────
def test_move_accumulation_and_dedup(belief):
    mb = MatchBelief(belief)
    sp = _known_species(belief)
    mb.observe_move(sp, "Protect")
    mb.observe_move(sp, "Protect")   # duplicate
    mb.observe_move(sp, "Earthquake")
    block = mb.block_for(sp)
    known = [m.lower() for m in block["moves_known"]]
    assert known.count("protect") == 1
    assert "earthquake" in known
    # a revealed move is never re-offered as a prediction
    pred = {m["name"].lower() for m in block["moves_predicted"]}
    assert "protect" not in pred and "earthquake" not in pred


def test_item_and_ability_collapse(belief):
    mb = MatchBelief(belief)
    sp = _known_species(belief)
    mb.observe_item(sp, "Choice Scarf")
    mb.observe_ability(sp, "Intimidate")
    block = mb.block_for(sp)
    assert block["items"] == [{"name": "Choice Scarf", "p": 1.0, "revealed": True}]
    assert block["abilities"] and block["abilities"][0]["name"] == "Intimidate"
    assert block["abilities"][0]["p"] == 1.0


def test_persistence_across_calls(belief):
    mb = MatchBelief(belief)
    sp = _known_species(belief)
    mb.observe_move(sp, "Earthquake")
    _ = mb.block_for(sp)
    # a later, unrelated observation must not lose the earlier one
    mb.observe_item(sp, "Leftovers")
    later = mb.block_for(sp)
    assert any(m.lower() == "earthquake" for m in later["moves_known"])
    assert later["items"][0]["name"] == "Leftovers"


def test_reset_clears(belief):
    mb = MatchBelief(belief)
    sp = _known_species(belief)
    mb.observe_move(sp, "Earthquake")
    mb.reset()
    assert mb.offensive_categories(sp) == set()


def test_ingest_mon(belief):
    mb = MatchBelief(belief)
    sp = _known_species(belief)
    mon = {
        "species": sp, "base_species": sp,
        "revealed_moves": ["Earthquake", "Protect"],
        "known_item": "Assault Vest", "known_ability": None,
        "can_have_choice_item": False,
    }
    mb.ingest_mon(mon)
    assert mb.offensive_categories(sp) == {"atk"}
    block = mb.block_for(sp)
    assert block["items"][0]["name"] == "Assault Vest"


# ── parity (no observation ⇒ identical to belief_block) ─────────────────────────
def test_parity_with_belief_block_no_obs(belief):
    mb = MatchBelief(belief)
    sp = _known_species(belief)
    got = mb.block_for(sp)
    ref = belief.belief_block(sp, top_k=5)
    assert got["spreads"] == ref["spreads"]
    assert got["expected_stats"] == ref["expected_stats"]
    assert got["items"] == ref["items"]
    assert got["abilities"] == ref["abilities"]


# ── move-category narrowing ─────────────────────────────────────────────────────
def test_category_narrowing_physical(belief):
    sp = _find_mixed_nature_species(belief)
    if sp is None:
        pytest.skip("no species with mixed-nature spreads in current data")
    base = belief.belief_block(sp, top_k=5)
    mb = MatchBelief(belief)
    mb.observe_move(sp, "Earthquake")          # standard physical -> invests Atk
    narrowed = mb.block_for(sp)

    # -Atk-nature spreads lose relative mass; Atk expected-stat does not drop.
    assert _atk_drop_mass(narrowed) < _atk_drop_mass(base) - 1e-9
    assert narrowed["expected_stats"]["atk"] >= base["expected_stats"]["atk"] - 1e-6
    # conservative: no spread eliminated; still a valid distribution.
    assert all(s["p"] > 0 for s in narrowed["spreads"])
    assert abs(sum(s["p"] for s in narrowed["spreads"]) - 1.0) < 1e-2
    assert len(narrowed["spreads"]) == len(base["spreads"])


def test_category_narrowing_excludes_cross_stat(belief):
    sp = _find_mixed_nature_species(belief)
    if sp is None:
        pytest.skip("no species with mixed-nature spreads in current data")
    base = belief.belief_block(sp, top_k=5)
    mb = MatchBelief(belief)
    mb.observe_move(sp, "Body Press")          # Physical CATEGORY but uses Defense
    got = mb.block_for(sp)
    assert mb.offensive_categories(sp) == set()
    assert got["spreads"] == base["spreads"]   # untouched
    assert got["expected_stats"] == base["expected_stats"]


def test_mixed_offense_no_narrowing(belief):
    sp = _find_mixed_nature_species(belief)
    if sp is None:
        pytest.skip("no species with mixed-nature spreads in current data")
    base = belief.belief_block(sp, top_k=5)
    mb = MatchBelief(belief)
    mb.observe_move(sp, "Earthquake")          # physical
    mb.observe_move(sp, "Ice Beam")            # special -> mixed, ambiguous
    got = mb.block_for(sp)
    assert mb.offensive_categories(sp) == {"atk", "spa"}
    assert got["spreads"] == base["spreads"]


def test_special_only_narrows_spa(belief):
    """Symmetric check: special-only offense down-weights -SpA natures."""
    sp = None
    for cand in belief.all_pokemon():
        spreads = belief.spread_distribution(cand, top_k=5)
        if len(spreads) < 2:
            continue
        natset = {s["nature"] for s in spreads}
        if any(_drops(n, "spa") for n in natset) and any(not _drops(n, "spa") for n in natset):
            sp = cand
            break
    if sp is None:
        pytest.skip("no species with mixed -SpA spreads in current data")
    base = belief.belief_block(sp, top_k=5)
    mb = MatchBelief(belief)
    mb.observe_move(sp, "Ice Beam")
    narrowed = mb.block_for(sp)
    base_spa = sum(s["p"] for s in base["spreads"] if _drops(s["nature"], "spa"))
    new_spa = sum(s["p"] for s in narrowed["spreads"] if _drops(s["nature"], "spa"))
    assert new_spa < base_spa - 1e-9
    assert all(s["p"] > 0 for s in narrowed["spreads"])


def test_enrich_mon_drop_in(belief):
    mb = MatchBelief(belief)
    sp = _known_species(belief)
    mon = {"species": sp, "base_species": sp, "revealed_moves": ["Earthquake"],
           "is_mega": False}
    ok = mb.enrich_mon(mon)
    assert ok is True
    assert mon["belief"]["source"] == "pikalytics"
    assert mon["stats_estimate"]["mode"] == "distribution"
    assert set(mon["stats_estimate"]["stats"].keys()) == {"hp", "atk", "def", "spa", "spd", "spe"}


def test_unknown_species_returns_none(belief):
    mb = MatchBelief(belief)
    assert mb.block_for("NotARealPokemonXYZ") is None
    assert mb.enrich_mon({"species": "Notarealpokemonxyz"}) is False
