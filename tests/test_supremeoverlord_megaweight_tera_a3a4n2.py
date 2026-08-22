"""v11 A3 (Supreme Overlord band) + A4 (live mega weight) + N2 (live tera-typing fix).

A3: Kingambit's Supreme Overlord boosts damage ×powMod[min(fainted_allies,5)]/4096 (exact chainModify
    table); the band ignored it. value-only, both encoders (shared `_ability_damage_mult`). The sim latches
    `fallen` at switch-in; we use the CURRENT side-faint count (`fainted_allies`) because poke-env exposes no
    latched value — a documented, parity-forced approximation.
A4: live encoder read `species_weight(mon.species)`, but poke-env keeps `mon.species` at the BASE forme
    post-mega (and forme-change/transform) while updating `_weightkg`. So megas read the base weight, drifting
    from the offline parser (which mutates species to the mega forme). Fix: read poke-env's `mon.weight`.
N2 (D7): live `_live_eff_types` read the NONEXISTENT attr `terastallized` (always False) instead of
    `is_terastallized`, so the live type-eff/damage cross used PRE-tera typing on every post-tera turn.

Re-audit: scratch/v11_reaudit_synthesis.md.
"""
from __future__ import annotations

import numpy as np
from poke_env.battle import Pokemon

from v_dance.encoders.state_encoder import (
    StateEncoder, MOVE_FEATURES, POKEMON_FEATURES, _MOVE_BLOCK_REL, _WEIGHT_REL,
    move_slots_for_mon, norm_species, _ability_damage_mult, _SUPREME_OVERLORD_MULT, _DMG_BOOST_AB,
)
from v_dance.encoders import damage_mechanics as _DMG
from v_dance.encoders.live_state_encoder import LiveStateEncoder, _live_eff_types


# ════════════════════════════ A3 unit: Supreme Overlord multiplier ════════════════════════════
def test_supremeoverlord_in_boost_set():
    assert "supremeoverlord" in _DMG_BOOST_AB


def test_supremeoverlord_exact_table():
    # dex abilities.ts:4734 powMod=[4096,4506,4915,5325,5734,6144]/4096
    expect = (1.0, 4506 / 4096, 4915 / 4096, 5325 / 4096, 5734 / 4096, 1.5)
    assert _SUPREME_OVERLORD_MULT == expect
    # the index-4 value is 5734/4096 = 1.39990… (NOT 1.4 — exactly why the integer table beats 1+0.1n)
    assert abs(_SUPREME_OVERLORD_MULT[4] - 1.3999023) < 1e-6


def _so(fainted):
    return _ability_damage_mult("ironhead", "supremeoverlord", None, eff_move_type="STEEL", is_stab=True,
                                is_physical=True, type_mult=1.0, hp_frac=1.0, att_burned=False,
                                att_statused=False, is_spread=False, fainted_allies=fainted)


def test_supremeoverlord_scales_with_faints():
    assert _so(0) == 1.0
    assert abs(_so(1) - 4506 / 4096) < 1e-9
    assert abs(_so(3) - 5325 / 4096) < 1e-9
    assert _so(5) == 1.5


def test_supremeoverlord_caps_at_five():
    assert _so(6) == _so(5) == 1.5            # min(faints,5) → no IndexError, capped at +50%


def test_supremeoverlord_only_for_that_ability():
    # a non-Supreme-Overlord attacker is unaffected by the faint count
    m = _ability_damage_mult("ironhead", "ironfist", None, eff_move_type="STEEL", is_stab=True,
                             is_physical=True, type_mult=1.0, hp_frac=1.0, att_burned=False,
                             att_statused=False, is_spread=False, fainted_allies=5)
    assert m == 1.0                           # Iron Fist doesn't boost Iron Head (not a punch), no SO leak


# ════════════════════════════ A3 e2e: band scales via the snapshot faint count ════════════════════════════
def _mon(species, ability, *, moves=(), fainted=False):
    return {"species": species, "base_species": species, "hp_pct": 100.0, "seen": True,
            "is_fainted": fainted, "known_moves": list(moves), "revealed_moves": [], "boosts": {},
            "status": None, "known_ability": ability, "volatiles": {},
            "stats_estimate": {"mode": "exact",
                               "stats": {"atk": 135, "spa": 60, "def": 120, "spd": 80, "hp": 280, "spe": 50}}}


def _band_so(att_move, n_fainted):
    att = _mon("Kingambit", "Supreme Overlord", moves=[att_move])
    bench = [_mon("Pawniard", "Defiant", fainted=True) for _ in range(n_fainted)]
    snap = {"our_active": {"our_a": att, "our_b": None},
            "opp_active": {"opp_a": _mon("Blissey", "Natural Cure"), "opp_b": None},
            "our_bench": bench, "opp_bench": [], "field": {}, "side_conditions": {}}
    vec = StateEncoder().encode_snapshot(snap, turn=1)
    for m_idx, (mv, _c) in enumerate(move_slots_for_mon(att)):
        if norm_species(mv) == norm_species(att_move):
            b = _MOVE_BLOCK_REL + m_idx * MOVE_FEATURES
            return float(vec[b + 13]), float(vec[b + 13 + 1])
    raise AssertionError(att_move)


def test_a3_band_scales_with_faints_e2e():
    none = _band_so("Iron Head", 0)
    three = _band_so("Iron Head", 3)
    assert none[1] > 0.0
    assert abs(three[1] / none[1] - 5325 / 4096) < 5e-3      # +3 faints → ×1.30005


def test_a3_zero_faints_unchanged():
    # with no fallen allies the band is identical to a vanilla attacker (SO multiplier == 1.0)
    so = _band_so("Iron Head", 0)
    assert 0.0 < so[0] < so[1]
    assert abs(so[0] / so[1] - 0.85) < 1e-3                  # just the 0.85–1.0 roll, no SO scaling


# ════════════════════════════ A4: live mega WEIGHT (forme-aware) ════════════════════════════
def _wt_channel(mon):
    enc = LiveStateEncoder()
    vec = np.zeros(POKEMON_FEATURES, dtype=np.float32)
    enc._write_pokemon(vec, 0, mon, is_active=True, is_own=False)
    return float(vec[_WEIGHT_REL])


def test_a4_mega_weight_premise():
    # poke-env keeps mon.species at the BASE forme post-mega but updates mon.weight → the exact desync A4 fixes
    ch = Pokemon(gen=9, species="charizard")
    ch.mega_evolve("charizarditey")
    assert ch.species == "charizard"                          # base forme retained (store_species=False)
    assert ch.weight == 100.5                                 # but weight is mega-aware
    assert _DMG.species_weight(ch.species) == 90.5            # species_weight(species) = WRONG (base)
    assert _DMG.species_weight("charizardmegay") == 100.5     # offline reference (parser mutates species)


def test_a4_live_weight_channel_uses_mega_weight():
    base = Pokemon(gen=9, species="charizard")
    mega = Pokemon(gen=9, species="charizard")
    mega.mega_evolve("charizarditey")
    assert abs(_wt_channel(base) - 90.5 / 500.0) < 1e-6
    # the FIX: mega weight channel == 100.5/500 (a revert to species_weight(species) would read 90.5/500)
    assert abs(_wt_channel(mega) - 100.5 / 500.0) < 1e-6
    # and it matches the offline reference (byte-parity restored)
    assert abs(_wt_channel(mega) - _DMG.species_weight("charizardmegay") / 500.0) < 1e-6


def test_a4_nonmega_weight_unchanged():
    # the fallback keeps non-mega values identical to the old species_weight path (no-op blast radius)
    chomp = Pokemon(gen=9, species="garchomp")
    assert abs(_wt_channel(chomp) - _DMG.species_weight("garchomp") / 500.0) < 1e-6


# ════════════════════════════ N2 (D7): live tera typing ════════════════════════════
def test_n2_tera_typing_active():
    mon = Pokemon(gen=9, species="garchomp")
    assert _live_eff_types(mon) == ["DRAGON", "GROUND"]       # pre-tera dex types
    mon.terastallize("Steel")
    assert mon.is_terastallized
    assert _live_eff_types(mon) == ["STEEL"]                  # post-tera typing (the fix reads is_terastallized)


def test_n2_old_attr_was_nonexistent():
    # documents the bug: poke-env has `is_terastallized`, never a bare `terastallized` → the old getattr
    # default-False made the branch dead code (live used pre-tera typing on every post-tera turn).
    mon = Pokemon(gen=9, species="garchomp")
    mon.terastallize("Steel")
    assert hasattr(mon, "is_terastallized")
    assert not hasattr(mon, "terastallized")
