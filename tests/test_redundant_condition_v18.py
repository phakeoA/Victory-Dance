"""v18 (Option 1): the per-move REDUNDANT-CONDITION feature — re-casting an already-active
screen/weather/terrain. Locks the shared helper's semantics (the offline + live encoders both call it,
so parity is covered by test_encoder_parity; here we pin the logic + the layout bump)."""
from __future__ import annotations

import pytest

pytest.importorskip("numpy")

from v_dance.encoders.battle_mechanics import move_redundant_condition, _REDUNDANT_OWN_SIDE_NAMES
from v_dance.encoders.encoder_layout import (
    MOVE_FEATURES, POKEMON_FEATURES, get_state_dim, get_state_layout_version,
)


def test_layout_v18():
    assert MOVE_FEATURES == 59            # +1 redundant-condition bit over v17's 56
    assert POKEMON_FEATURES == 413        # +1 per-move × 4 moves
    assert get_state_dim() == 5057
    assert get_state_layout_version() == 19


# ── own-side conditions (screens / tailwind / safeguard / mist / lucky chant) ──
def test_screen_redundant_only_when_active():
    assert move_redundant_condition("lightscreen", frozenset({"LIGHT_SCREEN"}), None, None) == 1.0
    assert move_redundant_condition("lightscreen", frozenset(), None, None) == 0.0
    assert move_redundant_condition("reflect", frozenset({"REFLECT"}), None, None) == 1.0
    assert move_redundant_condition("auroraveil", frozenset({"AURORA_VEIL"}), None, None) == 1.0
    assert move_redundant_condition("tailwind", frozenset({"TAILWIND"}), None, None) == 1.0


def test_screen_uses_the_relevant_side():
    # the caller passes the CASTER's own side; a screen up on the OTHER side is not redundant for me
    assert move_redundant_condition("lightscreen", frozenset(), None, None) == 0.0


# ── weather (the user's Rain Dance case) — alias-proof normalisation ──
def test_weather_redundant_matches_active_weather():
    assert move_redundant_condition("raindance", frozenset(), "RAINDANCE", None) == 1.0
    assert move_redundant_condition("raindance", frozenset(), "RainDance", None) == 1.0   # offline raw token
    assert move_redundant_condition("raindance", frozenset(), "SUNNYDAY", None) == 0.0
    assert move_redundant_condition("raindance", frozenset(), None, None) == 0.0
    assert move_redundant_condition("sunnyday", frozenset(), "SUNNYDAY", None) == 1.0
    # SNOW alias: the Snow move is 'snowscape'; active token may be SNOW / SNOWSCAPE
    assert move_redundant_condition("snowscape", frozenset(), "SNOWSCAPE", None) == 1.0
    assert move_redundant_condition("snowscape", frozenset(), "SNOW", None) == 1.0


def test_terrain_redundant():
    assert move_redundant_condition("electricterrain", frozenset(), None, "ELECTRIC_TERRAIN") == 1.0
    assert move_redundant_condition("grassyterrain", frozenset(), None, "electric") == 0.0
    assert move_redundant_condition("psychicterrain", frozenset(), None, "PSYCHIC_TERRAIN") == 1.0


def test_hail_move_maps_to_snow():
    # gen9 Hail sets snow; redundant under SNOW or SNOWSCAPE (alias-proof)
    assert move_redundant_condition("hail", frozenset(), "SNOW", None) == 1.0
    assert move_redundant_condition("hail", frozenset(), "SNOWSCAPE", None) == 1.0
    assert move_redundant_condition("hail", frozenset(), "RAINDANCE", None) == 0.0


def test_same_effect_screen_moves():
    # Glitzy Glow == Light Screen, Baddy Bad == Reflect (defensively mapped)
    assert move_redundant_condition("glitzyglow", frozenset({"LIGHT_SCREEN"}), None, None) == 1.0
    assert move_redundant_condition("baddybad", frozenset({"REFLECT"}), None, None) == 1.0
    assert move_redundant_condition("glitzyglow", frozenset(), None, None) == 0.0


# ── exclusions + non-setters ──
def test_non_setter_and_excluded_moves_are_never_redundant():
    assert move_redundant_condition("tackle", frozenset({"LIGHT_SCREEN"}), "RAINDANCE", None) == 0.0
    assert move_redundant_condition("protect", frozenset({"LIGHT_SCREEN"}), None, None) == 0.0
    # Trick Room EXCLUDED — re-casting toggles it OFF (not a wasted turn), so never flagged redundant
    assert move_redundant_condition("trickroom", frozenset({"TRICK_ROOM"}), None, None) == 0.0
    # entry hazards EXCLUDED (opp-side + stackable)
    assert move_redundant_condition("stealthrock", frozenset({"STEALTH_ROCK"}), None, None) == 0.0
    assert move_redundant_condition(None, frozenset(), None, None) == 0.0
    assert move_redundant_condition("", frozenset(), None, None) == 0.0


def test_redundant_names_cover_the_own_side_moves():
    # the canonical names the live filter keys on must include every own-side move's target condition
    assert _REDUNDANT_OWN_SIDE_NAMES == frozenset(
        {"LIGHT_SCREEN", "REFLECT", "AURORA_VEIL", "TAILWIND", "SAFEGUARD", "MIST", "LUCKY_CHANT"})


# ── encoding-level acceptance: the bit lands at the right offset and flips with the board state ──
from v_dance.encoders.state_encoder import StateEncoder
from v_dance.encoders.action_codec import _MOVE_BLOCK_REL

_REDUNDANT_REL = _MOVE_BLOCK_REL + 25   # move-0 redundancy bit = 25 core feats into the move block


def _mon(species, moves):
    return {"species": species, "base_species": species, "hp_pct": 100.0, "seen": True,
            "is_fainted": False, "known_moves": list(moves), "revealed_moves": [],
            "boosts": {}, "status": None}


def _encode(field, side_conditions, moves=("Light Screen", "Spirit Break")):
    snap = {"our_active": {"our_a": _mon("Grimmsnarl", moves), "our_b": None},
            "opp_active": {}, "our_bench": [], "opp_bench": [],
            "field": field, "side_conditions": side_conditions}
    return StateEncoder().encode_snapshot(snap, turn=3)


def test_encode_light_screen_redundancy_bit_flips():
    up   = _encode({}, {"our_side": {"screens": {"light_screen": 1}}})
    down = _encode({}, {})
    assert up[_REDUNDANT_REL] == 1.0, "Light Screen while it's up must set the redundancy bit"
    assert down[_REDUNDANT_REL] == 0.0, "Light Screen while it's down must NOT set the bit"


def test_encode_rain_dance_redundancy_bit_flips():
    moves = ("Rain Dance", "Thunder")
    rain = _encode({"weather": "RainDance"}, {}, moves)
    none = _encode({}, {}, moves)
    sun  = _encode({"weather": "SunnyDay"}, {}, moves)
    assert rain[_REDUNDANT_REL] == 1.0, "Rain Dance under rain must set the bit"
    assert none[_REDUNDANT_REL] == 0.0, "Rain Dance with no weather must NOT set the bit"
    assert sun[_REDUNDANT_REL] == 0.0, "Rain Dance under sun must NOT set the bit"


# ── opp + bench mons use the CORRECT side's conditions (scan coverage gap) ──
from v_dance.encoders.encoder_layout import POKEMON_FEATURES


def _slot_redundant(vec, slot):
    return vec[slot * POKEMON_FEATURES + _REDUNDANT_REL]


def test_encode_opp_mon_uses_opp_side_conditions():
    """An OPP mon's Light Screen is redundant w.r.t. the OPP side, not ours (slot 2 = opp_a)."""
    ls = ("Light Screen", "Spirit Break")
    snap = lambda sc: {"our_active": {"our_a": _mon("Grimmsnarl", ("Tackle",)), "our_b": None},
                       "opp_active": {"opp_a": _mon("Grimmsnarl", ls), "opp_b": None},
                       "our_bench": [], "opp_bench": [], "field": {}, "side_conditions": sc}
    opp_up = StateEncoder().encode_snapshot(snap({"opp_side": {"screens": {"light_screen": 1}}}), turn=3)
    our_up = StateEncoder().encode_snapshot(snap({"our_side": {"screens": {"light_screen": 1}}}), turn=3)
    assert _slot_redundant(opp_up, 2) == 1.0, "opp Light Screen redundant when OPP side has it up"
    assert _slot_redundant(our_up, 2) == 0.0, "opp Light Screen NOT redundant when only OUR side has it"


def test_encode_bench_mon_redundancy_bit():
    """Own bench mon (slot 4) reads OUR side conditions for its move redundancy."""
    ls = ("Light Screen", "Spirit Break")
    snap = {"our_active": {"our_a": _mon("Incineroar", ("Fake Out",)), "our_b": None},
            "opp_active": {}, "our_bench": [_mon("Grimmsnarl", ls)], "opp_bench": [],
            "field": {}, "side_conditions": {"our_side": {"screens": {"light_screen": 1}}}}
    vec = StateEncoder().encode_snapshot(snap, turn=3)
    assert _slot_redundant(vec, 4) == 1.0, "bench mon's Light Screen redundant when our screen is up"


def test_encode_unknown_move_leaves_redundancy_bit_zero():
    """An unknown move (absent from moves.json) hits the offline early-return → the whole move block,
    incl. the redundancy bit, stays 0 (conservative) even with the screen UP. Locks the scan's
    'unknown-move' edge (a no-op for valid data, but pinned so it can't silently diverge)."""
    vec = _encode({}, {"our_side": {"screens": {"light_screen": 1}}}, moves=("Notarealmove12345",))
    assert vec[_REDUNDANT_REL] == 0.0


# ── #4: offline↔live SIDE-CONDITION EXTRACTION parity (the only redundancy code the move-block-excluded
#     parity harness doesn't already cover; both extractors must yield the SAME canonical name set). ──
def test_side_condition_extraction_parity():
    pytest.importorskip("poke_env")
    from poke_env.battle import SideCondition
    from v_dance.encoders.state_encoder import _active_side_conditions_offline
    from v_dance.encoders.live_state_encoder import LiveStateEncoder
    # the SAME side state in the two input formats (offline snapshot dict vs poke-env enum→int):
    offline_in = {"screens": {"light_screen": 1, "reflect": 1}, "tailwind_turns_remaining": 3,
                  "safeguard": True, "stealth_rock": True, "spikes": 2}   # hazards present...
    live_in = {SideCondition.LIGHT_SCREEN: 5, SideCondition.REFLECT: 5, SideCondition.TAILWIND: 4,
               SideCondition.SAFEGUARD: 3, SideCondition.STEALTH_ROCK: 1, SideCondition.SPIKES: 2}
    want = frozenset({"LIGHT_SCREEN", "REFLECT", "TAILWIND", "SAFEGUARD"})   # ...but hazards are NOT redundancy-relevant
    assert _active_side_conditions_offline(offline_in) == want
    assert LiveStateEncoder._active_side_conditions_live(live_in) == want
    assert _active_side_conditions_offline({}) == frozenset()
    assert LiveStateEncoder._active_side_conditions_live({}) == frozenset()


# ── #6: the helper tolerates a None side_active (defensive — both call sites already pass `or frozenset()`) ──
def test_redundant_condition_tolerates_none_side_active():
    assert move_redundant_condition("lightscreen", None, None, None) == 0.0
    assert move_redundant_condition("raindance", None, "RAINDANCE", None) == 1.0
