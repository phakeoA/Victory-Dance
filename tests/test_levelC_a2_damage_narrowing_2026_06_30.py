"""
Level C / Component A2 — damage-roll spread narrowing unit tests (2026-06-30, fully-fleshed).

Verifies the Bayesian log-normal `MatchBelief.observe_damage_taken` / `observe_damage_dealt` +
`_narrow_spreads_by_damage` (with the additive-+2-correct per-candidate scaling and log-space accumulation):
  * DEFENSIVE: a LOW observed roll down-weights FRAIL spreads (mon is bulkier); HIGH down-weights BULKY;
  * the spread whose predicted damage is CLOSEST to the observed roll gains the most weight;
  * smaller σ → sharper narrowing; multiple constraints accumulate (log-space Bayesian product);
  * OFFENSIVE: a HIGH observed roll down-weights LOW-offense spreads (`observe_damage_dealt`);
  * never zeroes a spread (floored), distribution stays normalised, even over many events;
  * `category` selects the right stat; degenerate inputs ignored; no constraints = no-op.

Predicted means use the SAME +2-correct scaling as the implementation (see _mu_def / _mu_off).
"""
from __future__ import annotations

import pytest

from v_dance.parser.belief_state import BeliefState, dex_base_stats, calc_full_stats
from v_dance.parser.match_belief import MatchBelief, _P_FLOOR

NEUTRAL = "Hardy"   # no stat boost/drop
REF_MAIN = 70.0     # a plausible +2-free main damage term (HP points) for the reference


@pytest.fixture(scope="module")
def belief() -> BeliefState:
    try:
        return BeliefState()
    except Exception as e:  # pragma: no cover
        pytest.skip(f"BeliefState unavailable: {e}")


def _species_with_base(belief: BeliefState) -> str:
    sp = next((s for s in belief.all_pokemon() if dex_base_stats(s)), None)
    assert sp is not None, "no species with dex base stats"
    return sp


def _spread(evs_actual, p, nature=NEUTRAL) -> dict:
    return {"nature": nature, "evs": [min(e // 8, 32) for e in evs_actual],
            "evs_actual": list(evs_actual), "p": float(p)}


def _block(species, spreads) -> dict:
    return {"species_key": species, "spreads": [dict(s) for s in spreads],
            "expected_stats": {}, "items": [], "abilities": [],
            "moves_known": [], "moves_predicted": []}


def _p_of(block, evs) -> float:
    return next(s["p"] for s in block["spreads"] if s["evs_actual"] == list(evs))


def _full(base, evs):
    return calc_full_stats(base, evs, NEUTRAL)


def _mu_def(mu_ref, ref_main, ref_stat, ref_hp, cand_stat, cand_hp) -> float:
    main_cand = ref_main * (ref_stat / cand_stat)            # main ∝ 1/Def, +2 additive
    return mu_ref * (ref_hp / cand_hp) * (main_cand + 2.0) / (ref_main + 2.0)


def _mu_off(mu_ref, ref_main, ref_stat, cand_stat) -> float:
    main_cand = ref_main * (cand_stat / ref_stat)            # main ∝ Atk, +2 additive
    return mu_ref * (main_cand + 2.0) / (ref_main + 2.0)


BULKY, FRAIL = [252, 0, 252, 0, 0, 4], [0, 252, 0, 0, 0, 252]
MID = [252, 0, 0, 0, 0, 4]   # HP-only — bulk between the two


def test_low_damage_downweights_frail(belief):
    sp = _species_with_base(belief); base = dex_base_stats(sp)
    b = _full(base, BULKY)
    mb = MatchBelief(belief)
    # reference = bulky; obs == its predicted mean (mu_ref) ⇒ bulky consistent, frail predicts MORE ⇒ down.
    mb.observe_damage_taken(sp, mu_ref=0.40, ref_main=REF_MAIN, ref_def_stat=b["def"], ref_hp=b["hp"],
                            category="physical", obs_frac_of_max=0.40)
    block = _block(sp, [_spread(BULKY, 0.5), _spread(FRAIL, 0.5)])
    base_def = 0.5 * _full(base, BULKY)["def"] + 0.5 * _full(base, FRAIL)["def"]
    mb._narrow_spreads_by_damage(block, sp, sp)
    assert _p_of(block, FRAIL) < 0.5
    assert _p_of(block, BULKY) > 0.5
    assert all(s["p"] > 0 for s in block["spreads"])
    assert abs(sum(s["p"] for s in block["spreads"]) - 1.0) < 1e-2
    assert block["expected_stats"]["def"] > base_def


def test_high_damage_downweights_bulky(belief):
    sp = _species_with_base(belief); base = dex_base_stats(sp)
    b, f = _full(base, BULKY), _full(base, FRAIL)
    frail_mu = _mu_def(0.40, REF_MAIN, b["def"], b["hp"], f["def"], f["hp"])
    mb = MatchBelief(belief)
    mb.observe_damage_taken(sp, mu_ref=0.40, ref_main=REF_MAIN, ref_def_stat=b["def"], ref_hp=b["hp"],
                            category="physical", obs_frac_of_max=frail_mu)   # obs matches the FRAIL prediction
    block = _block(sp, [_spread(BULKY, 0.5), _spread(FRAIL, 0.5)])
    mb._narrow_spreads_by_damage(block, sp, sp)
    assert _p_of(block, BULKY) < 0.5
    assert _p_of(block, FRAIL) > 0.5
    assert all(s["p"] > 0 for s in block["spreads"])


def test_closest_spread_wins(belief):
    sp = _species_with_base(belief); base = dex_base_stats(sp)
    b, m = _full(base, BULKY), _full(base, MID)
    mid_mu = _mu_def(0.40, REF_MAIN, b["def"], b["hp"], m["def"], m["hp"])
    mb = MatchBelief(belief)
    mb.observe_damage_taken(sp, mu_ref=0.40, ref_main=REF_MAIN, ref_def_stat=b["def"], ref_hp=b["hp"],
                            category="physical", obs_frac_of_max=mid_mu)
    block = _block(sp, [_spread(BULKY, 1 / 3), _spread(MID, 1 / 3), _spread(FRAIL, 1 / 3)])
    mb._narrow_spreads_by_damage(block, sp, sp)
    ps = {tuple(e): _p_of(block, e) for e in (BULKY, MID, FRAIL)}
    assert ps[tuple(MID)] == max(ps.values())   # mid (matches obs) wins
    assert all(v > 0 for v in ps.values())


def test_smaller_sigma_sharper(belief):
    sp = _species_with_base(belief); base = dex_base_stats(sp)
    b, f = _full(base, BULKY), _full(base, FRAIL)
    obs = _mu_def(0.40, REF_MAIN, b["def"], b["hp"], f["def"], f["hp"]) * 1.3   # off both ⇒ σ matters

    def frail_p(sigma):
        mb = MatchBelief(belief)
        mb.observe_damage_taken(sp, mu_ref=0.40, ref_main=REF_MAIN, ref_def_stat=b["def"], ref_hp=b["hp"],
                                category="physical", obs_frac_of_max=0.40, sigma_log=sigma)
        block = _block(sp, [_spread(BULKY, 0.5), _spread(FRAIL, 0.5)])
        mb._narrow_spreads_by_damage(block, sp, sp)
        return _p_of(block, FRAIL)
    assert frail_p(0.10) < frail_p(0.40)


def test_multi_event_sharpens_and_floors(belief):
    sp = _species_with_base(belief); base = dex_base_stats(sp)
    b = _full(base, BULKY)

    def frail_p(n):
        mb = MatchBelief(belief)
        for _ in range(n):
            mb.observe_damage_taken(sp, mu_ref=0.40, ref_main=REF_MAIN, ref_def_stat=b["def"], ref_hp=b["hp"],
                                    category="physical", obs_frac_of_max=0.40)
        block = _block(sp, [_spread(BULKY, 0.5), _spread(FRAIL, 0.5)])
        mb._narrow_spreads_by_damage(block, sp, sp)
        return _p_of(block, FRAIL)
    assert frail_p(3) < frail_p(1)
    assert frail_p(200) >= _P_FLOOR   # never zeroed even over MANY events (log-space, no underflow)


def test_offensive_downweights_low_attack(belief):
    sp = _species_with_base(belief); base = dex_base_stats(sp)
    high_atk, low_atk = [0, 252, 0, 0, 0, 4], [252, 0, 0, 0, 0, 4]
    ha = _full(base, high_atk)
    mb = MatchBelief(belief)
    mb.observe_damage_dealt(sp, mu_ref=0.40, ref_main=REF_MAIN, ref_atk_stat=ha["atk"],
                            category="physical", obs_frac_of_max=0.40)   # obs ⇒ high-Atk consistent
    block = _block(sp, [_spread(high_atk, 0.5), _spread(low_atk, 0.5)])
    mb._narrow_spreads_by_damage(block, sp, sp)
    assert _p_of(block, low_atk) < 0.5
    assert _p_of(block, high_atk) > 0.5
    assert all(s["p"] > 0 for s in block["spreads"])


def test_category_selects_spd(belief):
    sp = _species_with_base(belief); base = dex_base_stats(sp)
    high_spd, low_spd = [252, 0, 0, 0, 252, 4], [0, 252, 0, 0, 0, 252]
    hs = _full(base, high_spd)
    mb = MatchBelief(belief)
    mb.observe_damage_taken(sp, mu_ref=0.40, ref_main=REF_MAIN, ref_def_stat=hs["spd"], ref_hp=hs["hp"],
                            category="special", obs_frac_of_max=0.40)
    block = _block(sp, [_spread(high_spd, 0.5), _spread(low_spd, 0.5)])
    mb._narrow_spreads_by_damage(block, sp, sp)
    assert _p_of(block, low_spd) < 0.5
    assert _p_of(block, high_spd) > 0.5


def test_additive_plus2_correct(belief):
    """The per-candidate μ must match the +2-correct formula (not pure proportional scaling)."""
    sp = _species_with_base(belief); base = dex_base_stats(sp)
    b, f = _full(base, BULKY), _full(base, FRAIL)
    # obs set to the CORRECT frail prediction ⇒ frail beats bulky; a +2-wrong impl would mis-rank near the margin.
    obs = _mu_def(0.40, REF_MAIN, b["def"], b["hp"], f["def"], f["hp"])
    mb = MatchBelief(belief)
    mb.observe_damage_taken(sp, mu_ref=0.40, ref_main=REF_MAIN, ref_def_stat=b["def"], ref_hp=b["hp"],
                            category="physical", obs_frac_of_max=obs, sigma_log=0.06)
    block = _block(sp, [_spread(BULKY, 0.5), _spread(FRAIL, 0.5)])
    mb._narrow_spreads_by_damage(block, sp, sp)
    assert _p_of(block, FRAIL) > _p_of(block, BULKY)


def test_no_constraints_noop(belief):
    sp = _species_with_base(belief)
    mb = MatchBelief(belief)
    block = _block(sp, [_spread(BULKY, 0.5), _spread(FRAIL, 0.5)])
    before = [s["p"] for s in block["spreads"]]
    mb._narrow_spreads_by_damage(block, sp, sp)
    assert [s["p"] for s in block["spreads"]] == before


def test_degenerate_inputs_ignored(belief):
    sp = _species_with_base(belief)
    mb = MatchBelief(belief)
    mb.observe_damage_taken(sp, mu_ref=0.0, ref_main=70, ref_def_stat=100, ref_hp=180, category="physical", obs_frac_of_max=0.3)
    mb.observe_damage_taken(sp, mu_ref=0.4, ref_main=None, ref_def_stat=100, ref_hp=180, category="physical", obs_frac_of_max=0.3)
    mb.observe_damage_taken(sp, mu_ref=0.4, ref_main=70, ref_def_stat=None, ref_hp=180, category="physical", obs_frac_of_max=0.3)
    mb.observe_damage_taken(sp, mu_ref=0.4, ref_main=70, ref_def_stat=100, ref_hp=180, category="status", obs_frac_of_max=0.3)
    mb.observe_damage_dealt(sp, mu_ref=0.4, ref_main=None, ref_atk_stat=120, category="physical", obs_frac_of_max=0.3)
    assert mb._mons.get(mb._key(sp)) is None or not mb._mons[mb._key(sp)].damage_constraints


def test_block_for_integration_runs(belief):
    sp = next((s for s in belief.all_pokemon()
               if len(belief.spread_distribution(s, top_k=5)) >= 2 and dex_base_stats(s)), None)
    if sp is None:
        pytest.skip("no multi-spread species with base stats")
    mb = MatchBelief(belief)
    mb.observe_damage_taken(sp, mu_ref=0.30, ref_main=70, ref_def_stat=120, ref_hp=180,
                            category="physical", obs_frac_of_max=0.15)
    block = mb.block_for(sp)
    assert block is not None
    assert all(s["p"] > 0 for s in block["spreads"])
    assert abs(sum(s["p"] for s in block["spreads"]) - 1.0) < 1e-2
