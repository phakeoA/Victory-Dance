"""
Level C / A3 — live MatchBelief WIRING unit tests (2026-06-30).

Covers the ``SplicingVGCPlayerBase`` scaffold that puts the within-game ``MatchBelief`` in FRONT of the
served net by PRE-ENRICHING the gap-#6 opponent snapshot before ``encode()``:

  * flag OFF → ``_apply_match_belief`` is a TRUE no-op (the snapshot object is returned unchanged, no
    ``stats_estimate`` written, no per-battle belief created) → prod serving stays byte-identical;
  * flag ON  → each opponent mon (active + bench) gets a ``stats_estimate`` + ``belief`` block from the
    per-battle MatchBelief, so the encoder's static ``_enrich_opp_snapshot`` then NO-OPs and splices the
    sharpened bytes;
  * the per-battle MatchBelief is keyed by battle tag — ISOLATED across battles, REUSED within one — and
    ACCUMULATES reveals across turns (the A1 category signal, observed live through the wiring);
  * identity gating (Zoroark Illusion) flows through the wiring — a disguise's move is NOT attributed;
  * a ``None`` snapshot (reconstruction failed) → returns ``None`` (degrade to poke-env view);
  * an encoder with no ``BeliefState`` → no-op (static fallback).

The numeric narrowing math is covered by the A1/A2/A3 unit tests on ``MatchBelief`` itself; these tests
verify the PLUMBING (lifecycle, gating, no-op-when-off).
"""
from __future__ import annotations

import types

import pytest

from v_dance.parser.belief_state import BeliefState
from v_dance.parser.match_belief import MatchBelief


@pytest.fixture(scope="module")
def belief() -> BeliefState:
    try:
        return BeliefState()
    except Exception as e:  # pragma: no cover
        pytest.skip(f"BeliefState unavailable: {e}")


def _enrichable_species(belief: BeliefState) -> str:
    """A species for which ``MatchBelief.enrich_mon`` writes a belief block (has
    Pikalytics data + base stats), so the wiring tests assert on a real enrichment."""
    for sp in belief.all_pokemon():
        mb = MatchBelief(belief)
        if mb.enrich_mon({"species": sp, "base_species": sp,
                          "revealed_moves": [], "known_moves": []}):
            return sp
    pytest.skip("no enrichable species in belief")


def _player(*, use_match_belief: bool, belief):
    """A SplicingVGCPlayerBase built via __new__ (skip the networked poke-env init),
    with only the attributes ``_apply_match_belief`` touches set."""
    from v_dance.play.live_vgc_base import SplicingVGCPlayerBase

    class _C(SplicingVGCPlayerBase):
        def _select_actions(self, battle, state_vec):  # abstract contract
            return 0, 0, "test"

    p = _C.__new__(_C)
    p._use_match_belief = use_match_belief
    p._match_belief = {}
    p._encoder = types.SimpleNamespace(belief=belief, level=50)
    return p


def _battle(tag: str):
    return types.SimpleNamespace(battle_tag=tag)


def _mon(sp: str, moves=None) -> dict:
    return {"species": sp, "base_species": sp,
            "revealed_moves": list(moves or []), "known_moves": [],
            "known_item": None, "known_ability": None}


def _snapshot(opp_a=None, opp_b=None, bench=None) -> dict:
    return {"opp_active": {"opp_a": opp_a, "opp_b": opp_b}, "opp_bench": list(bench or [])}


# ── flag OFF → true no-op (prod byte-identical) ───────────────────────────────
def test_flag_off_is_noop(belief):
    sp = _enrichable_species(belief)
    p = _player(use_match_belief=False, belief=belief)
    mon = _mon(sp)
    snap = _snapshot(opp_a=mon)
    out = p._apply_match_belief(snap, _battle("battle-1"))
    assert out is snap                       # same object, returned untouched
    assert "stats_estimate" not in mon       # no enrichment happened
    assert "belief" not in mon
    assert p._match_belief == {}             # no per-battle belief created at all


# ── flag ON → each opp mon enriched from the per-battle belief ────────────────
def test_flag_on_enriches_active(belief):
    sp = _enrichable_species(belief)
    p = _player(use_match_belief=True, belief=belief)
    mon = _mon(sp)
    out = p._apply_match_belief(_snapshot(opp_a=mon), _battle("battle-1"))
    assert out is not None
    est = mon.get("stats_estimate")
    assert est and est.get("mode") == "distribution"
    assert isinstance(est.get("stats"), dict) and est["stats"]
    assert mon.get("belief")
    assert set(p._match_belief) == {"battle-1"}   # per-battle belief created + keyed by tag


def test_flag_on_enriches_bench(belief):
    sp = _enrichable_species(belief)
    p = _player(use_match_belief=True, belief=belief)
    bench_mon = _mon(sp)
    p._apply_match_belief(_snapshot(bench=[bench_mon]), _battle("battle-1"))
    assert bench_mon.get("stats_estimate")     # bench mons are enriched too


# ── reveals accumulate across turns (A1 live, through the wiring) ─────────────
def test_reveals_accumulate_across_turns(belief):
    sp = _enrichable_species(belief)
    p = _player(use_match_belief=True, belief=belief)
    # turn 1: nothing revealed yet
    p._apply_match_belief(_snapshot(opp_a=_mon(sp)), _battle("battle-1"))
    mb = p._match_belief["battle-1"]
    assert mb.offensive_categories(sp) == set()
    # turn 2: a STANDARD physical move is now revealed → ingested through the wiring
    p._apply_match_belief(_snapshot(opp_a=_mon(sp, moves=["Earthquake"])), _battle("battle-1"))
    assert mb.offensive_categories(sp) == {"atk"}   # persisted on the SAME per-battle belief


# ── per-battle isolation (no cross-game carryover) ───────────────────────────
def test_belief_isolated_across_battles(belief):
    sp = _enrichable_species(belief)
    p = _player(use_match_belief=True, belief=belief)
    p._apply_match_belief(_snapshot(opp_a=_mon(sp, moves=["Earthquake"])), _battle("battle-1"))
    p._apply_match_belief(_snapshot(opp_a=_mon(sp)), _battle("battle-2"))
    assert p._match_belief["battle-1"].offensive_categories(sp) == {"atk"}
    assert p._match_belief["battle-2"].offensive_categories(sp) == set()   # fresh game, no carryover
    assert p._match_belief["battle-1"] is not p._match_belief["battle-2"]


def test_battle_tag_normalised(belief):
    """The tag key is _norm_tag'd (leading '>' stripped) — matches the _proto_log convention so a
    reset in _battle_finished_callback (which uses _norm_tag) finds the same key."""
    sp = _enrichable_species(belief)
    p = _player(use_match_belief=True, belief=belief)
    p._apply_match_belief(_snapshot(opp_a=_mon(sp)), _battle(">battle-9"))
    assert set(p._match_belief) == {"battle-9"}


# ── identity gating flows through the wiring ─────────────────────────────────
def test_disguised_mon_reveals_not_attributed(belief):
    sp = _enrichable_species(belief)
    p = _player(use_match_belief=True, belief=belief)
    mon = _mon(sp, moves=["Earthquake"])
    mon["illusion_active"] = True                  # a Zoroark disguised as this species
    p._apply_match_belief(_snapshot(opp_a=mon), _battle("battle-1"))
    # the disguise's move must NOT update the apparent species' offensive belief
    assert p._match_belief["battle-1"].offensive_categories(sp) == set()


# ── degrade paths ────────────────────────────────────────────────────────────
def test_none_snapshot_returns_none(belief):
    p = _player(use_match_belief=True, belief=belief)
    assert p._apply_match_belief(None, _battle("battle-1")) is None
    assert p._match_belief == {}                    # nothing created for a missing snapshot


def test_no_belief_on_encoder_is_noop(belief):
    sp = _enrichable_species(belief)
    p = _player(use_match_belief=True, belief=None)   # encoder without a BeliefState
    mon = _mon(sp)
    out = p._apply_match_belief(_snapshot(opp_a=mon), _battle("battle-1"))
    assert out is not None
    assert "stats_estimate" not in mon
    assert p._match_belief == {}
