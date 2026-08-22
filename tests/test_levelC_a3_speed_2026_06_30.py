"""
Level C / Component A3 — speed-tier narrowing + turn-order analyzer unit tests (2026-06-30).

Covers:
  * `MatchBelief.observe_speed_bound` + `_narrow_spreads_by_speed` — a `faster` bound down-weights SLOW spreads,
    `slower` down-weights FAST; σ controls sharpness; never-zero; no-op/degenerate guards.
  * `analyze_speed_order` — the full priority physics: same/different brackets, Prankster (+1 status), Gale Wings
    (+1 Flying at full HP), Trick Room (flips), Quash/After-You (`order_forced` → bail), known-multiplier division,
    σ widening on unknown opp context.
  * `observe_speed_order` glue — records when the order is speed-determined, no-ops otherwise.
"""
from __future__ import annotations

import pytest

from v_dance.parser.belief_state import BeliefState, dex_base_stats, calc_full_stats
from v_dance.parser.match_belief import MatchBelief, analyze_speed_order, identity_reliable

NEUTRAL = "Hardy"
PHYS = {"priority": 0, "category": "physical", "type": "normal"}


@pytest.fixture(scope="module")
def belief() -> BeliefState:
    try:
        return BeliefState()
    except Exception as e:  # pragma: no cover
        pytest.skip(f"BeliefState unavailable: {e}")


def _species_with_base(belief: BeliefState) -> str:
    sp = next((s for s in belief.all_pokemon() if dex_base_stats(s)), None)
    assert sp is not None
    return sp


def _spread(evs_actual, p, nature=NEUTRAL) -> dict:
    return {"nature": nature, "evs": [min(e // 8, 32) for e in evs_actual],
            "evs_actual": list(evs_actual), "p": float(p)}


def _block(species, spreads) -> dict:
    return {"species_key": species, "spreads": [dict(s) for s in spreads],
            "expected_stats": {}, "items": [], "abilities": [], "moves_known": [], "moves_predicted": []}


def _p_of(block, evs) -> float:
    return next(s["p"] for s in block["spreads"] if s["evs_actual"] == list(evs))


def _full(base, evs):
    return calc_full_stats(base, evs, NEUTRAL)


FAST, SLOW = [0, 0, 0, 0, 4, 252], [252, 0, 252, 0, 0, 0]


# ── narrowing ─────────────────────────────────────────────────────────────────
def test_faster_downweights_slow(belief):
    sp = _species_with_base(belief); base = dex_base_stats(sp)
    f, s = _full(base, FAST), _full(base, SLOW)
    T = (f["spe"] + s["spe"]) / 2.0
    mb = MatchBelief(belief)
    mb.observe_speed_bound(sp, threshold_base_spe=T, faster=True, sigma_spe=4.0)
    block = _block(sp, [_spread(FAST, 0.5), _spread(SLOW, 0.5)])
    base_spe = 0.5 * f["spe"] + 0.5 * s["spe"]
    mb._narrow_spreads_by_speed(block, sp, sp)
    assert _p_of(block, FAST) > 0.5         # opp WAS faster ⇒ fast spread consistent
    assert _p_of(block, SLOW) < 0.5
    assert all(x["p"] > 0 for x in block["spreads"])
    assert abs(sum(x["p"] for x in block["spreads"]) - 1.0) < 1e-2
    assert block["expected_stats"]["spe"] > base_spe


def test_slower_downweights_fast(belief):
    sp = _species_with_base(belief); base = dex_base_stats(sp)
    f, s = _full(base, FAST), _full(base, SLOW)
    T = (f["spe"] + s["spe"]) / 2.0
    mb = MatchBelief(belief)
    mb.observe_speed_bound(sp, threshold_base_spe=T, faster=False, sigma_spe=4.0)
    block = _block(sp, [_spread(FAST, 0.5), _spread(SLOW, 0.5)])
    mb._narrow_spreads_by_speed(block, sp, sp)
    assert _p_of(block, SLOW) > 0.5         # opp was SLOWER ⇒ slow spread consistent
    assert _p_of(block, FAST) < 0.5
    assert all(x["p"] > 0 for x in block["spreads"])


def test_speed_sigma_sharper(belief):
    sp = _species_with_base(belief); base = dex_base_stats(sp)
    f, s = _full(base, FAST), _full(base, SLOW)
    T = (f["spe"] + s["spe"]) / 2.0

    def slow_p(sig):
        mb = MatchBelief(belief)
        mb.observe_speed_bound(sp, threshold_base_spe=T, faster=True, sigma_spe=sig)
        block = _block(sp, [_spread(FAST, 0.5), _spread(SLOW, 0.5)])
        mb._narrow_spreads_by_speed(block, sp, sp)
        return _p_of(block, SLOW)
    assert slow_p(2.0) < slow_p(30.0)       # tighter σ ⇒ slow spread driven lower


def test_speed_no_constraints_noop(belief):
    sp = _species_with_base(belief)
    mb = MatchBelief(belief)
    block = _block(sp, [_spread(FAST, 0.5), _spread(SLOW, 0.5)])
    before = [x["p"] for x in block["spreads"]]
    mb._narrow_spreads_by_speed(block, sp, sp)
    assert [x["p"] for x in block["spreads"]] == before


def test_speed_degenerate_ignored(belief):
    sp = _species_with_base(belief)
    mb = MatchBelief(belief)
    mb.observe_speed_bound(sp, threshold_base_spe=0.0, faster=True)
    mb.observe_speed_bound(sp, threshold_base_spe=None, faster=True)
    assert mb._mons.get(mb._key(sp)) is None or not mb._mons[mb._key(sp)].speed_constraints


# ── analyzer: priority physics ──────────────────────────────────────────────────
def test_analyzer_same_bracket_faster():
    r = analyze_speed_order(opp_moved_first=True, our_eff_speed=200.0, our_move=PHYS, opp_move=PHYS)
    assert r is not None and r["faster"] is True
    assert abs(r["threshold_base_spe"] - 200.0) < 1e-6


def test_analyzer_diff_priority_bracket_none():
    r = analyze_speed_order(opp_moved_first=True, our_eff_speed=200.0,
                            our_move=PHYS, opp_move={"priority": 1, "category": "physical", "type": "normal"})
    assert r is None      # opp used a +1 priority move → order was priority, not speed


def test_analyzer_negative_priority_bracket_none():
    r = analyze_speed_order(opp_moved_first=False, our_eff_speed=200.0,
                            our_move=PHYS, opp_move={"priority": -6, "category": "status", "type": "normal"})
    assert r is None      # opp's −6 move (e.g. Roar) is a different bracket


def test_analyzer_prankster_both_same_bracket():
    status = {"priority": 0, "category": "status", "type": "normal"}
    r = analyze_speed_order(opp_moved_first=True, our_eff_speed=150.0, our_move=status, opp_move=status,
                            our_ability="prankster", opp_ability="prankster")   # both +1 → same bracket
    assert r is not None and r["faster"] is True


def test_analyzer_prankster_one_side_none():
    status = {"priority": 0, "category": "status", "type": "normal"}
    r = analyze_speed_order(opp_moved_first=True, our_eff_speed=150.0, our_move=status, opp_move=status,
                            opp_ability="prankster")   # only opp +1 → different bracket
    assert r is None


def test_analyzer_gale_wings_full_hp_vs_chip():
    fly = {"priority": 0, "category": "physical", "type": "flying"}
    r_full = analyze_speed_order(opp_moved_first=True, our_eff_speed=150.0, our_move=PHYS, opp_move=fly,
                                 opp_ability="galewings", opp_hp_frac=1.0)
    assert r_full is None          # +1 from Gale Wings at full HP → different bracket
    r_chip = analyze_speed_order(opp_moved_first=True, our_eff_speed=150.0, our_move=PHYS, opp_move=fly,
                                 opp_ability="galewings", opp_hp_frac=0.5)
    assert r_chip is not None       # below full HP → no bump → same bracket → speed read


def test_analyzer_trick_room_flips():
    r = analyze_speed_order(opp_moved_first=True, trick_room=True, our_eff_speed=200.0,
                            our_move=PHYS, opp_move=PHYS)
    assert r["faster"] is False     # moving first under Trick Room ⇒ SLOWER


def test_analyzer_order_forced_none():
    r = analyze_speed_order(opp_moved_first=True, order_forced=True, our_eff_speed=200.0,
                            our_move=PHYS, opp_move=PHYS)
    assert r is None                # Quash / After You / Quick Claw manipulated the order


def test_analyzer_known_mult_and_sigma_widen():
    r = analyze_speed_order(opp_moved_first=True, our_eff_speed=300.0, opp_speed_mult_known=2.0,  # opp Tailwind
                            our_move=PHYS, opp_move=PHYS, opp_context_known=False)
    assert abs(r["threshold_base_spe"] - 150.0) < 1e-6      # 300 / 2
    assert r["sigma_spe"] > 0.06 * 150.0                    # widened for a possible hidden Scarf


def test_observe_speed_order_glue(belief):
    sp = _species_with_base(belief)
    mb = MatchBelief(belief)
    mb.observe_speed_order(sp, opp_moved_first=True, our_eff_speed=150.0, our_move=PHYS, opp_move=PHYS)
    assert mb._mons[mb._key(sp)].speed_constraints           # recorded a constraint
    mb2 = MatchBelief(belief)
    mb2.observe_speed_order(sp, opp_moved_first=True, our_eff_speed=150.0, our_move=PHYS,
                            opp_move={"priority": 1, "category": "physical", "type": "normal"})
    assert mb2._mons.get(mb2._key(sp)) is None or not mb2._mons[mb2._key(sp)].speed_constraints   # no-op


# ── identity guard: Zoroark Illusion / Ditto Transform ──────────────────────────
def test_identity_reliable():
    assert identity_reliable({"species": "Garchomp"}) is True
    assert identity_reliable(None) is False
    assert identity_reliable({"species": "Garchomp", "illusion_active": True}) is False   # Zoroark disguise
    assert identity_reliable({"species": "Garchomp", "disguise_species": "Zoroark"}) is False
    assert identity_reliable({"species": "Garchomp", "is_transformed": True}) is False     # Ditto/Transform
    assert identity_reliable({"species": "Garchomp", "transformed_into": "Landorus"}) is False


def test_ingest_mon_skips_disguised(belief):
    sp = _species_with_base(belief)
    mb = MatchBelief(belief)
    mb.ingest_mon({"species": sp, "base_species": sp, "revealed_moves": ["Earthquake"],
                   "known_item": "Choice Scarf", "illusion_active": True})     # Zoroark disguise → skip
    mb.ingest_mon({"species": sp, "base_species": sp, "revealed_moves": ["Ice Beam"],
                   "is_transformed": True})                                     # transformed Ditto → skip
    assert mb._mons.get(mb._key(sp)) is None or not mb._mons[mb._key(sp)].revealed_moves
    # a reliable mon still ingests normally
    mb.ingest_mon({"species": sp, "base_species": sp, "revealed_moves": ["Earthquake"]})
    assert mb._mons[mb._key(sp)].revealed_moves == ["Earthquake"]
