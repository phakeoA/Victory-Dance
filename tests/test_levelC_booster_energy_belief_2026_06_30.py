"""
Level C — Protosynthesis / Quark-Drive (Booster Energy) highest-stat belief signal (2026-06-30).

When a paradox mon's boost activates, the boosted stat IS its HIGHEST non-HP stat — a strong EV-spread
constraint. Covers: the parser volatile → boosted-stat extraction (byte-parity offline/live), MatchBelief
accumulation via ingest_mon (volatiles.paradox_boosted_stat), and the _narrow_spreads_by_boost_stat rank
narrowing (down-weights spreads whose argmax stat differs; never zeroed; no-op without a reveal).
"""
from __future__ import annotations

import pytest

pytest.importorskip("numpy")

from v_dance.parser.vod_parser.battle_models import volatile_flags
from v_dance.parser.belief_state import BeliefState, dex_base_stats, calc_full_stats
from v_dance.parser.match_belief import MatchBelief, _STAT_TIEBREAK


# ── parser: the volatile id → boosted stat ──────────────────────────────────────
def test_volatile_flags_extracts_boosted_stat():
    assert volatile_flags({"protosynthesisatk"})["paradox_boosted_stat"] == "atk"
    assert volatile_flags({"quarkdrivespa"})["paradox_boosted_stat"] == "spa"
    assert volatile_flags({"protosynthesisspe"})["paradox_boosted_stat"] == "spe"
    # spe case also still lights the dedicated bool the speed calc consumes
    assert volatile_flags({"quarkdrivespe"})["paradox_speed"] is True
    # no paradox volatile → None, and an unrelated volatile doesn't false-trip
    assert volatile_flags({"substitute"})["paradox_boosted_stat"] is None
    assert volatile_flags(set())["paradox_boosted_stat"] is None


# ── belief fixture + helpers ────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def belief() -> "BeliefState":
    try:
        return BeliefState()
    except Exception as e:  # pragma: no cover
        pytest.skip(f"BeliefState unavailable: {e}")


def _argmax(base, spread):
    full = calc_full_stats(base, spread["evs_actual"], spread["nature"], level=50)
    return max(_STAT_TIEBREAK, key=lambda st: (full.get(st, 0), -_STAT_TIEBREAK.index(st)))


def _find_mixed_argmax_species(belief):
    """A species whose top spreads do NOT all share one highest stat — so the boost narrowing can move mass."""
    for sp in belief.all_pokemon():
        spreads = belief.spread_distribution(sp, top_k=5)
        if len(spreads) < 2:
            continue
        base = dex_base_stats(sp)
        if not base:
            continue
        argmaxes = {_argmax(base, s) for s in spreads}
        if len(argmaxes) >= 2:
            return sp, base
    return None, None


def _mismatch_mass(block, base, revealed):
    return sum(s["p"] for s in block["spreads"] if _argmax(base, s) != revealed)


# ── belief: ingest + narrowing ──────────────────────────────────────────────────
def test_ingest_mon_reads_boosted_stat_from_volatiles(belief):
    mb = MatchBelief(belief)
    sp = next(iter(belief.all_pokemon()))
    mb.ingest_mon({"species": sp, "base_species": sp,
                   "volatiles": {"paradox_boosted_stat": "spa"}})
    assert mb._mons[mb._key(sp)].known_boosted_stat == "spa"


def test_boost_stat_narrowing_downweights_mismatched_argmax(belief):
    sp, base = _find_mixed_argmax_species(belief)
    if sp is None:
        pytest.skip("no species with mixed-argmax spreads in current data")
    ref = belief.belief_block(sp, top_k=5)
    revealed = _argmax(base, ref["spreads"][0])          # reveal the top spread's highest stat
    mb = MatchBelief(belief)
    mb.observe_boosted_stat(sp, revealed)
    narrowed = mb.block_for(sp)
    # spreads whose argmax != the revealed stat lose relative mass; nothing is zeroed.
    assert _mismatch_mass(narrowed, base, revealed) < _mismatch_mass(ref, base, revealed) - 1e-9
    assert all(s["p"] > 0 for s in narrowed["spreads"])
    assert abs(sum(s["p"] for s in narrowed["spreads"]) - 1.0) < 1e-2
    assert len(narrowed["spreads"]) == len(ref["spreads"])
    # NB: this is a RANK reveal (which stat is HIGHEST), not a magnitude one — concentrating mass on
    # argmax==revealed spreads does NOT monotonically raise that stat's expected value (a rank-matched
    # spread can be a balanced one), so we assert only the rank-mass shift + a valid, non-zeroed distribution.


def test_no_reveal_is_noop(belief):
    sp, _ = _find_mixed_argmax_species(belief)
    if sp is None:
        pytest.skip("no species with mixed-argmax spreads in current data")
    ref = belief.belief_block(sp, top_k=5)
    got = MatchBelief(belief).block_for(sp)               # no observe_boosted_stat
    assert got["spreads"] == ref["spreads"]
    assert got["expected_stats"] == ref["expected_stats"]


def test_invalid_stat_ignored(belief):
    mb = MatchBelief(belief)
    sp = next(iter(belief.all_pokemon()))
    mb.observe_boosted_stat(sp, "hp")                     # HP is never boosted → ignored
    mb.observe_boosted_stat(sp, None)
    assert mb._mons.get(mb._key(sp)) is None or mb._mons[mb._key(sp)].known_boosted_stat is None
