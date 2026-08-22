"""
Level C — deep search-forward-model audit fixes (2026-07-01, Workflow wf_48c8ac49).

Forward-vs-leaf contradictions in white_box_sim / search.py that corrupted the B1 rollout's damage/KO
estimates: raw (un-canonicalised) weather/terrain tokens, the missing -ate move retype, Choice Scarf wrongly
granting the Band/Specs ×1.5, Scope-Lens crit, Intimidate-on-entry, and belief-scenario base-forme stats for a
mega'd opp. All behind the default-OFF use_search flag (prod byte-identical).
"""
from __future__ import annotations

import pytest

pytest.importorskip("numpy")

from v_dance.encoders import white_box_sim as W
from v_dance.encoders.battle_mechanics import _gen9_moves
from v_dance.parser.vod_parser.pokedex import norm_species


def _mon(species, hp=100.0, atk=200, spa=200, spe=100, hp_stat=200, defn=120, item=None, ability=None):
    return {
        "species": species, "base_species": species, "hp_pct": hp, "status": None, "is_fainted": False,
        "boosts": {}, "known_item": item, "known_ability": ability, "mega_ability": None, "is_mega": False,
        "volatiles": {"has_substitute": False, "perish_norm": 0.0, "residual_damage": False},
        "revealed_moves": [], "times_attacked": 0,
        "stats_estimate": {"mode": "distribution",
                           "stats": {"hp": hp_stat, "atk": atk, "def": defn, "spa": spa, "spd": defn, "spe": spe}},
    }


def _state(*, weather=None, terrain=None, our_a=None, our_b=None, opp_a=None, opp_b=None):
    return {
        "field": {"weather": weather, "terrain": terrain, "trick_room_turns_remaining": 0},
        "side_conditions": {"our_side": {"tailwind_turns_remaining": 0, "screens": {}},
                            "opp_side": {"tailwind_turns_remaining": 0, "screens": {}}},
        "our_active": {"our_a": our_a, "our_b": our_b}, "opp_active": {"opp_a": opp_a, "opp_b": opp_b},
        "our_bench": [], "opp_bench": [],
    }


def _dmg(state, att, move, defn):
    """Mean % damage of one single-target hit (isolates the damage mechanic)."""
    mid = norm_species(move)
    return W._move_damage_pct(state, att, mid, _gen9_moves().get(mid), defn, "opp_a", is_spread=False)[0]


# ── #1 raw weather/terrain tokens must be canonicalised ─────────────────────────
def test_raw_weather_token_applies_the_damage_mult():
    att = _mon("Kingdra", spa=200)                       # Water/Dragon special attacker
    defn = _mon("Snorlax", defn=100, hp_stat=200)        # Normal, neutral to Water
    no_w = _dmg(_state(opp_a=defn, our_a=att), att, "Scald", defn)
    rain = _dmg(_state(weather="RainDance", opp_a=defn, our_a=att), att, "Scald", defn)   # RAW token
    assert rain > no_w * 1.4                             # ~1.5x rain boost applied (was a no-op before the fix)


def test_raw_terrain_token_applies_the_damage_mult():
    att = _mon("Rampardos", atk=200)
    grounded = _mon("Snorlax", defn=100, hp_stat=200)
    base = _dmg(_state(opp_a=grounded, our_a=att), att, "Earthquake", grounded)
    grassy = _dmg(_state(terrain="grassy", opp_a=grounded, our_a=att), att, "Earthquake", grounded)  # raw lowercase
    assert grassy < base * 0.9                           # Grassy Terrain halves EQ vs grounded (grassy_eq fix)


# ── #2 -ate ability move retype (Pixilate) ──────────────────────────────────────
def test_ate_ability_retypes_move_for_type_effectiveness():
    att = _mon("Sylveon", spa=200, ability="Pixilate")   # Fairy; Pixilate → Normal moves become Fairy
    ghost = _mon("Dragapult", defn=100, hp_stat=200)     # Dragon/Ghost: Normal→0 (immune), Fairy→2x
    d = _dmg(_state(opp_a=ghost, our_a=att), att, "Hyper Voice", ghost)
    assert d > 0                                         # before the fix: Normal vs Ghost → _ZERO_DMG


# ── #3 Choice Scarf must NOT grant the Band/Specs x1.5 ──────────────────────────
def test_choice_scarf_gives_no_damage_boost():
    defn = _mon("Snorlax", defn=100, hp_stat=200)
    st = _state(opp_a=defn)
    d_none = _dmg(st, _mon("Rampardos", atk=200, item=None), "Rock Slide", defn)
    d_scarf = _dmg(st, _mon("Rampardos", atk=200, item="Choice Scarf"), "Rock Slide", defn)
    d_band = _dmg(st, _mon("Rampardos", atk=200, item="Choice Band"), "Rock Slide", defn)
    assert d_scarf == pytest.approx(d_none)              # Scarf: NO boost (was x1.5 before the fix)
    assert d_band > d_none * 1.4                         # Band: x1.5 (still correct)


# ── #7 Scope Lens raises the expected-crit fold ─────────────────────────────────
def test_scope_lens_raises_expected_crit():
    defn = _mon("Snorlax", defn=100, hp_stat=200)
    st = _state(opp_a=defn)
    d_none = _dmg(st, _mon("Rampardos", atk=200, item=None), "Rock Slide", defn)
    d_scope = _dmg(st, _mon("Rampardos", atk=200, item="Scope Lens"), "Rock Slide", defn)
    assert d_scope > d_none                              # +1 crit stage → higher mean (was hardcoded None)


# ── #4 Intimidate on switch-in lowers the opposing actives' Atk ─────────────────
def test_intimidate_on_switch_in_lowers_foe_atk():
    incoming = _mon("Incineroar", ability="Intimidate")
    st = _state(our_a=_mon("Garchomp"), opp_a=_mon("Snorlax"))
    st["our_bench"] = [incoming]
    W.switch_in(st, "our_a", 0)                          # switch Incineroar into our_a
    assert st["opp_active"]["opp_a"]["boosts"].get("atk") == -1


# ── #6 belief scenarios use the CURRENT (mega) forme's base stats ───────────────
def test_spread_full_stats_uses_mega_forme_base():
    from v_dance.play.search import _spread_full_stats
    spread = {"nature": "Modest", "evs_actual": [0, 0, 0, 252, 4, 252], "p": 1.0}
    mega = {"species": "charizardmegay", "base_species": "charizard", "is_mega": True,
            "belief": {"spreads": [spread]}}
    base = {"species": "charizard", "base_species": "charizard", "belief": {"spreads": [spread]}}
    spa_mega = _spread_full_stats(mega, 0)[0]["spa"]
    spa_base = _spread_full_stats(base, 0)[0]["spa"]
    assert spa_mega > spa_base                           # mega base SpA 159 > base 109 → uses the current forme
