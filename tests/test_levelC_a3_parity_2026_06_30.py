"""
Level C / A3 — belief-OFF byte-parity smoke (2026-06-30).

Proves "belief-OFF == current bytes": with no within-game evidence the within-game MatchBelief is a
faithful drop-in for the static prior, and the single-parse refactor (``reconstruct_for_decision``)
did NOT change the encoded opponent bytes. Since ALL the A3 wiring is gated behind the off-by-default
``use_match_belief`` flag, these establish that prod serving is unchanged.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from v_dance.parser.belief_state import BeliefState, dex_base_stats
from v_dance.parser.match_belief import MatchBelief
from v_dance.encoders.state_encoder import VodStateEncoder
from v_dance.encoders.live_state_encoder import (
    reconstruct_for_decision, opp_snapshot_from_log_prefix,
)


@pytest.fixture(scope="module")
def belief() -> BeliefState:
    try:
        return BeliefState()
    except Exception as e:  # pragma: no cover
        pytest.skip(f"BeliefState unavailable: {e}")


def _opp_species(belief: BeliefState) -> str:
    for sp in belief.all_pokemon():
        if dex_base_stats(sp) and belief.spread_distribution(sp, top_k=5):
            return sp
    pytest.skip("no species with spread + base stats")


def _header(our: str, opp: str) -> str:
    return (
        "|player|p1|alice|101|1500\n|player|p2|bob|102|1500\n|gen|9\n|tier|[Gen 9] Test Doubles\n"
        f"|poke|p1|{our}, L50|\n|poke|p1|Pikachu, L50|\n"
        f"|poke|p2|{opp}, L50|\n|poke|p2|Pikachu, L50|\n"
        "|teamsize|p1|2\n|teamsize|p2|2\n|start\n"
        f"|switch|p1a: {our}|{our}, L50|100/100\n|switch|p1b: Pikachu|Pikachu, L50|100/100\n"
        f"|switch|p2a: {opp}|{opp}, L50|100/100\n|switch|p2b: Pikachu|Pikachu, L50|100/100\n"
    )


# ── no within-game evidence → MatchBelief == static prior ─────────────────────
def test_block_for_no_evidence_equals_static(belief):
    for sp in [_opp_species(belief)] + list(belief.all_pokemon())[:25]:
        if not (dex_base_stats(sp) and belief.spread_distribution(sp, top_k=5)):
            continue
        static = belief.belief_block(sp, top_k=5)
        narrowed = MatchBelief(belief).block_for(sp)              # no observations fed
        assert static == narrowed, f"{sp}: block diverged with no evidence"


def test_enrich_mon_no_evidence_matches_static_stats(belief):
    sp = _opp_species(belief)
    mon = {"species": sp, "base_species": sp, "revealed_moves": [], "known_moves": []}
    assert MatchBelief(belief).enrich_mon(mon)
    blk = belief.belief_block(sp, top_k=5)
    assert mon["stats_estimate"]["stats"] == blk["expected_stats"]   # identical est-stats → identical bytes


# ── single-parse refactor → identical opponent bytes ──────────────────────────
def test_reconstruct_byte_parity(belief):
    opp_sp = _opp_species(belief)
    log = _header("Garchomp", opp_sp) + (
        "|turn|1\n"
        f"|move|p1a: Garchomp|Earthquake|p2a: {opp_sp}\n"
        f"|-damage|p2a: {opp_sp}|60/100\n"
        "|upkeep\n|turn|2\n"
    )
    snap_new, prev = reconstruct_for_decision(log, "p1", 2)
    snap_old = opp_snapshot_from_log_prefix(log, "p1", 2)
    assert snap_new is not None and snap_old is not None and prev["turn"] == 1
    enc = VodStateEncoder(belief)
    a = enc.encode_snapshot(copy.deepcopy(snap_new), turn=2)
    b = enc.encode_snapshot(copy.deepcopy(snap_old), turn=2)
    assert np.array_equal(a, b)                                   # one parse → byte-identical to two
    assert np.all(np.isfinite(a))
