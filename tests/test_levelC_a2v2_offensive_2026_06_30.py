"""
Level C / A2 v2 — offensive poke-env per-candidate damage narrowing (2026-06-30).

Covers the three new pieces:
  * ``MatchBelief.observe_damage_loglik`` — a precomputed per-spread log-likelihood narrows the spread
    distribution (calculator-agnostic);
  * ``pokeenv_damage.offensive_loglik`` — injects each candidate spread's stats + an item hypothesis into
    the opp mon, calls (monkeypatched) ``calculate_damage``, marginalises, and RESTORES the live mon;
  * the feeder routes an offensive event to the poke-env callable when provided, else the analytic path;
  * the player ``_offensive_loglik`` glue degrades cleanly.
"""
from __future__ import annotations

import types

import pytest

from v_dance.parser.belief_state import BeliefState, dex_base_stats, calc_full_stats
from v_dance.parser.match_belief import MatchBelief, _spread_key
from v_dance.parser.vod_parser.pokedex import norm_species
from v_dance.play.live_belief_feed import feed_damage_constraints


@pytest.fixture(scope="module")
def belief() -> BeliefState:
    try:
        return BeliefState()
    except Exception as e:  # pragma: no cover
        pytest.skip(f"BeliefState unavailable: {e}")


def _species_with_atk_spread(belief: BeliefState) -> str:
    """A species whose top-5 spreads carry ≥2 distinct Atk values (so a damage obs can disambiguate)."""
    for sp in belief.all_pokemon():
        base = dex_base_stats(sp)
        block = belief.belief_block(sp, top_k=5) if base else None
        spreads = (block or {}).get("spreads") or []
        if base and len(spreads) >= 2:
            atks = {calc_full_stats(base, s["evs_actual"], s["nature"])["atk"] for s in spreads}
            if len(atks) >= 2:
                return sp
    pytest.skip("no species with a varied Atk spread")


def _max_atk_spread(belief, sp):
    base = dex_base_stats(sp)
    spreads = belief.belief_block(sp, top_k=5)["spreads"]
    return max(spreads, key=lambda s: calc_full_stats(base, s["evs_actual"], s["nature"])["atk"]), base


# ── MatchBelief.observe_damage_loglik (pure narrowing) ────────────────────────
def test_observe_damage_loglik_narrows_toward_favoured_spread(belief):
    sp = _species_with_atk_spread(belief)
    base = dex_base_stats(sp)
    static = belief.belief_block(sp, top_k=5)
    smax = max(static["spreads"],   # the max-Atk spread WITHIN this list (identity-comparable below)
               key=lambda s: calc_full_stats(base, s["evs_actual"], s["nature"])["atk"])
    mb = MatchBelief(belief)
    ll = {_spread_key(s): (0.0 if s is smax else -5.0) for s in static["spreads"]}
    mb.observe_damage_loglik(sp, ll)
    narrowed = mb.block_for(sp)
    assert static["spreads"] != narrowed["spreads"]
    assert narrowed["expected_stats"]["atk"] >= static["expected_stats"]["atk"]   # up-weight high-Atk
    ps = [s["p"] for s in narrowed["spreads"]]
    assert all(p > 0 for p in ps) and abs(sum(ps) - 1.0) <= 0.02                  # never-zero, normalised


def test_observe_damage_loglik_empty_is_noop(belief):
    sp = _species_with_atk_spread(belief)
    mb = MatchBelief(belief)
    mb.observe_damage_loglik(sp, {})
    assert belief.belief_block(sp, top_k=5)["spreads"] == mb.block_for(sp)["spreads"]


# ── poke-env per-candidate calc (monkeypatched) ───────────────────────────────
class _FakeMon:
    def __init__(self, name, base_species, max_hp):
        self.name = name
        self.base_species = base_species
        self.species = norm_species(base_species)
        self._stats = {k: 1 for k in ("hp", "atk", "def", "spa", "spd", "spe")}
        self._item = "saveditem"
        self._ability = "savedability"
        self.max_hp = max_hp

    def identifier(self, role):
        return f"{role}: {self.name}"

    @property
    def stats(self):
        return self._stats


class _FakeBattle:
    def __init__(self, opp, our):
        self.opponent_role, self.player_role = "p2", "p1"
        self._m = {opp.identifier("p2"): opp, our.identifier("p1"): our}

    def get_pokemon(self, ident):
        return self._m[ident]


def _fake_calc(att_id, def_id, move, battle, is_critical=False):
    """Damage ∝ the (injected) attacker Atk × the Choice-Band multiplier — so the test can verify both
    injection and item marginalisation."""
    att = battle.get_pokemon(att_id)
    atk = att._stats.get("atk") or 0
    mult = 1.5 if att._item == "choiceband" else 1.0
    d = atk * mult * 0.5
    return d, d


def test_offensive_loglik_injects_marginalises_restores(belief, monkeypatch):
    sp = _species_with_atk_spread(belief)
    smax, base = _max_atk_spread(belief, sp)
    amax = calc_full_stats(base, smax["evs_actual"], smax["nature"])["atk"]
    max_hp = 200
    obs = (amax * 1.5 * 0.5) / max_hp                       # the max-Atk spread under a Choice Band
    opp = _FakeMon("Opp", sp, max_hp=999)
    our = _FakeMon("Our", "Pikachu", max_hp=max_hp)
    battle = _FakeBattle(opp, our)
    monkeypatch.setattr("v_dance.play.pokeenv_damage.calculate_damage", _fake_calc)
    from v_dance.play.pokeenv_damage import offensive_loglik
    ll = offensive_loglik(battle, opp, our, "Close Combat", is_critical=False, obs_frac=obs,
                          belief=belief, opp_snapshot_mon={"known_item": "Choice Band"})
    assert ll is not None and len(ll) >= 2
    assert max(ll, key=ll.get) == _spread_key(smax)         # best match = the max-Atk spread
    # the live mon is RESTORED exactly (the critical safety invariant)
    assert opp._item == "saveditem" and opp._ability == "savedability"
    assert all(v == 1 for v in opp._stats.values())


def test_offensive_loglik_none_without_pokeenv(belief, monkeypatch):
    monkeypatch.setattr("v_dance.play.pokeenv_damage._HAS_POKE_ENV", False)
    from v_dance.play.pokeenv_damage import offensive_loglik
    opp = _FakeMon("Opp", "Garchomp", 999)
    our = _FakeMon("Our", "Pikachu", 200)
    assert offensive_loglik(_FakeBattle(opp, our), opp, our, "Earthquake",
                            is_critical=False, obs_frac=0.4, belief=belief) is None


# ── DEFENSIVE direction (opp is the defender; narrow its bulk) ─────────────────
def _species_with_def_spread(belief):
    for sp in belief.all_pokemon():
        base = dex_base_stats(sp)
        block = belief.belief_block(sp, top_k=5) if base else None
        spreads = (block or {}).get("spreads") or []
        if base and len(spreads) >= 2:
            bulks = set()
            for s in spreads:
                f = calc_full_stats(base, s["evs_actual"], s["nature"])
                bulks.add(round(f["def"] * f["hp"]))
            if len(bulks) >= 2:
                return sp
    pytest.skip("no species with a varied Def×HP spread")


def _fake_calc_def(att_id, def_id, move, battle, is_critical=False):
    """Damage ∝ 1/defender_Def (Eviolite ×0.5) — disambiguates Def, and HP via the per-candidate denom."""
    defn = battle.get_pokemon(def_id)
    dfn = defn._stats.get("def") or 1
    mult = 0.5 if defn._item == "eviolite" else 1.0
    d = 8000.0 * mult / dfn
    return d, d


def test_defensive_loglik_narrows_bulk(belief, monkeypatch):
    sp = _species_with_def_spread(belief)
    base = dex_base_stats(sp)
    spreads = belief.belief_block(sp, top_k=5)["spreads"]

    def bulk(s):
        f = calc_full_stats(base, s["evs_actual"], s["nature"])
        return f["def"] * f["hp"]

    smax = max(spreads, key=bulk)
    fmax = calc_full_stats(base, smax["evs_actual"], smax["nature"])
    obs = (8000.0 / fmax["def"]) / fmax["hp"]              # the bulkiest spread, no Eviolite
    opp = _FakeMon("Opp", sp, max_hp=999)
    our = _FakeMon("Our", "Pikachu", max_hp=999)
    battle = _FakeBattle(opp, our)
    monkeypatch.setattr("v_dance.play.pokeenv_damage.calculate_damage", _fake_calc_def)
    from v_dance.play.pokeenv_damage import damage_loglik
    ll = damage_loglik(battle, opp_mon=opp, our_mon=our, direction="def", move_name="Earthquake",
                       is_critical=False, obs_frac=obs, belief=belief, opp_snapshot_mon={})
    assert ll is not None and len(ll) >= 2
    assert max(ll, key=ll.get) == _spread_key(smax)        # the bulkiest spread best matches the small obs
    assert opp._item == "saveditem" and opp._ability == "savedability"     # restored


# ── item + ability marginalisation plumbing ───────────────────────────────────
def test_marginalises_over_items_and_abilities(belief, monkeypatch):
    sp = _species_with_atk_spread(belief)
    seen = []

    def rec_calc(att_id, def_id, move, battle, is_critical=False):
        att = battle.get_pokemon(att_id)
        seen.append((att._item, att._ability))
        return 10.0, 10.0

    opp = _FakeMon("Opp", sp, 999)
    our = _FakeMon("Our", "Pikachu", 200)
    battle = _FakeBattle(opp, our)
    monkeypatch.setattr("v_dance.play.pokeenv_damage.calculate_damage", rec_calc)
    from v_dance.play.pokeenv_damage import damage_loglik
    damage_loglik(battle, opp_mon=opp, our_mon=our, direction="off", move_name="Close Combat",
                  is_critical=False, obs_frac=0.05, belief=belief, opp_snapshot_mon={})
    assert None in {i for i, a in seen}                    # the residual "no item" hypothesis is present
    assert len(seen) >= 2                                  # marginalised over >1 (item × ability) combo
    # a REVEALED item collapses item marginalisation to that single item
    seen.clear()
    damage_loglik(battle, opp_mon=opp, our_mon=our, direction="off", move_name="Close Combat",
                  is_critical=False, obs_frac=0.05, belief=belief,
                  opp_snapshot_mon={"known_item": "Choice Band"})
    assert {i for i, a in seen} == {norm_species("Choice Band")}


def _defensive_prev(opp_sp):
    return {
        "state_before_actions": {"p1": {
            "our_active": {"our_a": dict(OUR), "our_b": None},
            "opp_active": {"opp_a": {"species": opp_sp, "base_species": opp_sp}, "opp_b": None},
        }},
        "damage_events": [{
            "event": "damage", "slot": "p2a", "species": opp_sp,         # OUR p1a hit the OPP at p2a
            "source_slot": "p1a", "source_species": "Garchomp", "source_move": "Earthquake",
            "hp_pct_after": 60.0, "hp_pct_delta": -40.0, "crit": False,
        }],
    }


def test_feeder_routes_defensive_to_pokeenv(belief):
    sp = _species_with_atk_spread(belief)
    mb = MatchBelief(belief)
    seen = {}

    def fake_fn(**kw):
        seen.update(kw)
        return {("Adamant", (0, 252, 0, 0, 4, 252)): 0.0}

    stats = feed_damage_constraints(mb, belief, _defensive_prev(sp), "p1", OUR_STATS,
                                    damage_loglik_fn=fake_fn)
    assert stats["used_def"] == 1
    assert seen["direction"] == "def" and seen["opp_key"] == "opp_a" and seen["our_key"] == "our_a"
    cons = mb._mons[norm_species(sp)].damage_constraints
    assert len(cons) == 1 and cons[0]["mode"] == "loglik"


# ── feeder routing ────────────────────────────────────────────────────────────
OUR = {"species": "Garchomp", "base_species": "Garchomp"}
OUR_STATS = {norm_species("Garchomp"): {"hp": 183, "atk": 180, "def": 115, "spa": 90, "spd": 105, "spe": 169}}


def _offensive_prev(opp_sp):
    return {
        "state_before_actions": {"p1": {
            "our_active": {"our_a": dict(OUR), "our_b": None},
            "opp_active": {"opp_a": {"species": opp_sp, "base_species": opp_sp}, "opp_b": None},
        }},
        "damage_events": [{
            "event": "damage", "slot": "p1a", "species": "Garchomp",
            "source_slot": "p2a", "source_species": opp_sp, "source_move": "Earthquake",
            "hp_pct_after": 60.0, "hp_pct_delta": -40.0, "crit": False,
        }],
    }


def test_feeder_routes_offensive_to_pokeenv(belief):
    sp = _species_with_atk_spread(belief)
    mb = MatchBelief(belief)
    seen = {}

    def fake_fn(**kw):
        seen.update(kw)
        return {("Adamant", (0, 252, 0, 0, 4, 252)): 0.0}

    stats = feed_damage_constraints(mb, belief, _offensive_prev(sp), "p1", OUR_STATS,
                                    damage_loglik_fn=fake_fn)
    assert stats["used_off"] == 1
    assert (seen["opp_species"] == sp and seen["direction"] == "off"
            and seen["opp_key"] == "opp_a" and seen["our_key"] == "our_a")
    cons = mb._mons[norm_species(sp)].damage_constraints
    assert len(cons) == 1 and cons[0]["mode"] == "loglik"


def test_feeder_falls_back_to_analytic_on_none(belief):
    sp = _species_with_atk_spread(belief)
    mb = MatchBelief(belief)
    stats = feed_damage_constraints(mb, belief, _offensive_prev(sp), "p1", OUR_STATS,
                                    damage_loglik_fn=lambda **kw: None)   # poke-env couldn't run
    assert stats["used_off"] == 1
    cons = mb._mons[norm_species(sp)].damage_constraints
    assert len(cons) == 1 and cons[0]["mode"] == "off"      # analytic constraint recorded instead


# ── player glue ───────────────────────────────────────────────────────────────
def test_player_offensive_loglik_degrades_on_missing_mon():
    from v_dance.play.live_vgc_base import SplicingVGCPlayerBase

    class _C(SplicingVGCPlayerBase):
        def _select_actions(self, battle, state_vec):
            return 0, 0, "test"

    p = _C.__new__(_C)
    p._encoder = types.SimpleNamespace(belief=None, level=50)
    battle = types.SimpleNamespace(opponent_active_pokemon=[None, None], active_pokemon=[None, None])
    out = p._damage_loglik(battle, direction="off", opp_key="opp_a", our_key="our_a",
                           opp_species="Garchomp", our_species="Garchomp", opp_snapshot_mon={},
                           move="Earthquake", crit=False, obs_frac=0.4)
    assert out is None


def test_player_loglik_guards_our_mon_species_mismatch():
    # #3 (audit 2026-06-30): if OUR live-board mon at the slot isn't the one in the T-1 event (we switched
    # / fainted since the hit), the poke-env calc would use the wrong defender/attacker + max-HP denominator.
    # The guard must degrade to None (→ analytic fallback), NOT compute a wrong-reference narrowing.
    from v_dance.play.live_vgc_base import SplicingVGCPlayerBase

    class _C(SplicingVGCPlayerBase):
        def _select_actions(self, battle, state_vec):
            return 0, 0, "test"

    p = _C.__new__(_C)
    p._encoder = types.SimpleNamespace(belief=None, level=50)
    opp = types.SimpleNamespace(base_species="Garchomp", species="Garchomp")
    our = types.SimpleNamespace(base_species="Flutter Mane", species="Flutter Mane")   # live mon at our slot
    battle = types.SimpleNamespace(opponent_active_pokemon=[opp, None], active_pokemon=[our, None])
    out = p._damage_loglik(battle, direction="off", opp_key="opp_a", our_key="our_a",
                           opp_species="Garchomp", our_species="Landorus",   # the event's our mon (differs)
                           opp_snapshot_mon={}, move="Earthquake", crit=False, obs_frac=0.4)
    assert out is None
