"""v11 Phase B.3 — ability-trapping into can_switch (audit #5).

can_switch was NOT(move-trap OR rooted); a Shadow-Tag'd / Arena-Trap'd / Magnet-Pull'd mon read as
free-to-switch. Now an opposing active's trapping ability zeroes can_switch, with Ghost-type + Shed Shell
+ Shadow-Tag-mutual immunities. Value-only (no layout). Verified vs abilities.ts + pokemon.ts tryTrap.
"""
from __future__ import annotations

from v_dance.encoders.state_encoder import (
    StateEncoder, POKEMON_FEATURES, VOLATILE_FEATURES, NUM_TYPES, _ability_trapped,
)

# can_switch is the 5th slot (+4) of the 7-feature volatile block, which sits just before the 4 trailing
# slot flags → index within a mon block = POKEMON_FEATURES - 4 - VOLATILE_FEATURES + 4.
CAN_SWITCH = POKEMON_FEATURES - 4 - NUM_TYPES - VOLATILE_FEATURES + 4   # v11 Phase D D9: tera one-hot before flags


def _f(*a, **k):
    return _ability_trapped(*a, **k)


# ── the helper ────────────────────────────────────────────────────────────────
def test_trap_set():
    # the recognised trap abilities are inlined in _ability_trapped (no shared constant);
    # all three trap a steel+grounded normal foe, a non-trap ability never does.
    for ab in ("shadowtag", "arenatrap", "magnetpull"):
        assert _f(False, True, True, False, False, [ab]) is True
    assert _f(False, True, True, False, False, ["intimidate"]) is False


def test_shadow_tag_rules():
    assert _f(False, False, True, False, False, ["shadowtag"]) is True       # traps a normal foe
    assert _f(True, False, True, False, False, ["shadowtag"]) is False        # Ghost immune
    assert _f(False, False, True, True, False, ["shadowtag"]) is False        # mutual (foe has Shadow Tag)
    assert _f(False, False, True, False, True, ["shadowtag"]) is False        # Shed Shell


def test_arena_trap_rules():
    assert _f(False, False, True, False, False, ["arenatrap"]) is True        # grounded → trapped
    assert _f(False, False, False, False, False, ["arenatrap"]) is False      # ungrounded (Flyer/Levitate)
    assert _f(True, False, True, False, False, ["arenatrap"]) is False        # Ghost immune


def test_magnet_pull_rules():
    assert _f(False, True, True, False, False, ["magnetpull"]) is True         # Steel → trapped
    assert _f(False, False, True, False, False, ["magnetpull"]) is False       # non-Steel
    assert _f(True, True, True, False, False, ["magnetpull"]) is False         # Ghost/Steel → Ghost immune wins


def test_no_trapper_or_irrelevant_ability():
    assert _f(False, True, True, False, False, ["intimidate", None]) is False
    assert _f(False, False, True, False, False, []) is False


# ── end-to-end through encode_snapshot ────────────────────────────────────────
def _mon(species, ability, *, item=None, moves=()):
    return {
        "species": species, "base_species": species, "hp_pct": 100.0,
        "seen": True, "is_fainted": False, "known_moves": list(moves), "revealed_moves": [],
        "boosts": {}, "status": None, "known_ability": ability, "known_item": item,
        "stats_estimate": {"mode": "exact",
                           "stats": {"atk": 120, "spa": 90, "def": 95, "spd": 90, "hp": 190, "spe": 100}},
    }


def _can_switch_our_a(our_a, opp_a):
    snap = {"our_active": {"our_a": our_a, "our_b": None},
            "opp_active": {"opp_a": opp_a, "opp_b": None},
            "our_bench": [], "opp_bench": [], "field": {}, "side_conditions": {}}
    vec = StateEncoder().encode_snapshot(snap, turn=1)
    return float(vec[CAN_SWITCH])      # our_a is slot 0 → mon block starts at index 0


def test_e2e_shadow_tag_zeroes_can_switch():
    free = _can_switch_our_a(_mon("Garchomp", "Rough Skin"), _mon("Snorlax", "Thick Fat"))
    trapped = _can_switch_our_a(_mon("Garchomp", "Rough Skin"), _mon("Gengar", "Shadow Tag"))
    assert free == 1.0, "no trapper → free to switch"
    assert trapped == 0.0, "opposing Shadow Tag → can_switch 0"


def test_e2e_ghost_immune_to_shadow_tag():
    # a Ghost mon (Dragapult: Dragon/Ghost) is immune to trapping → still free vs Shadow Tag.
    assert _can_switch_our_a(_mon("Dragapult", "Clear Body"), _mon("Gengar", "Shadow Tag")) == 1.0


def test_e2e_shed_shell_escapes():
    assert _can_switch_our_a(_mon("Garchomp", "Rough Skin", item="Shed Shell"),
                             _mon("Gengar", "Shadow Tag")) == 1.0


def test_e2e_magnet_pull_only_traps_steel():
    steel = _can_switch_our_a(_mon("Metagross", "Clear Body"), _mon("Magnezone", "Magnet Pull"))
    nonsteel = _can_switch_our_a(_mon("Garchomp", "Rough Skin"), _mon("Magnezone", "Magnet Pull"))
    assert steel == 0.0 and nonsteel == 1.0
