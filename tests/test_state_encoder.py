"""Tests for the state_encoder OFFLINE path, focused on the layout-v2 additions:
opponent-bench slots, the per-mon is_fainted flag, and the team-count globals.
"""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")
import numpy as np

from v_dance.encoders.state_encoder import (
    StateEncoder, get_state_dim,
    POKEMON_FEATURES, ACTIVE_SLOTS, BENCH_SLOTS, OPP_BENCH_SLOTS, GLOBAL_FEATURES,
)

# Per-mon feature offsets (within a 144-float slot, layout-v2 + gap #5)
_HP = 0
_TYPE1 = 1                             # type1 one-hot starts here (20 wide)
_BASE = 1 + 20 + 20                    # base stats start (6 wide)
_IS_ACTIVE = POKEMON_FEATURES - 4      # 140
_IS_REVEALED = POKEMON_FEATURES - 3    # 141
_IS_FAINTED = POKEMON_FEATURES - 2     # 142
_IS_TRANSFORMED = POKEMON_FEATURES - 1 # 143
_OPP_BENCH0 = (ACTIVE_SLOTS + BENCH_SLOTS) * POKEMON_FEATURES   # slot 8 base


def _mon(species, *, hp=100.0, seen=True, fainted=False, moves=("Fake Out",)):
    return {
        "species": species, "base_species": species,
        "hp_pct": (None if (not seen and not fainted) else (0.0 if fainted else hp)),
        "seen": seen, "is_fainted": fainted,
        "known_moves": list(moves), "revealed_moves": [],
        "boosts": {}, "status": None,
    }


def _snap():
    return {
        "our_active": {"our_a": _mon("Incineroar"), "our_b": _mon("Kingambit")},
        "opp_active": {"opp_a": _mon("Calyrex-Shadow"), "opp_b": _mon("Miraidon")},
        "our_bench": [_mon("Rillaboom"), _mon("Urshifu", fainted=True)],
        "opp_bench": [
            _mon("Chien-Pao", seen=True),                  # seen, alive
            _mon("Ogerpon", seen=True, fainted=True),      # seen, fainted
            _mon("Landorus", seen=False),                  # unseen stub
        ],
        "field": {}, "side_conditions": {},
    }


# ── Dimension / structure ───────────────────────────────────────────────────
def test_state_dim_is_layout_v2():
    assert POKEMON_FEATURES == 148      # 110 + item(17) + ability(17) + 4×(MOVE 9→10)
    assert (ACTIVE_SLOTS, BENCH_SLOTS, OPP_BENCH_SLOTS) == (4, 4, 4)
    assert GLOBAL_FEATURES == 78
    assert get_state_dim() == 1854      # 12*148 + 78  (gap #5 item/ability + #6 is_spread)


def test_encode_shape_and_finite():
    vec = StateEncoder().encode_snapshot(_snap(), turn=3)
    assert vec.shape == (get_state_dim(),)
    assert np.isfinite(vec).all()


# ── Opponent bench slots ─────────────────────────────────────────────────────
def test_opp_bench_slots_populated_in_priority_order():
    vec = StateEncoder().encode_snapshot(_snap(), turn=3)

    def slot(j):  # opp bench slot j → absolute base
        return _OPP_BENCH0 + j * POKEMON_FEATURES

    s0, s1, s2, s3 = (slot(0), slot(1), slot(2), slot(3))
    # slot 0 = seen alive (Chien-Pao): revealed, not fainted, hp>0
    assert vec[s0 + _IS_REVEALED] == 1.0
    assert vec[s0 + _IS_FAINTED] == 0.0
    assert vec[s0 + _HP] > 0.0
    # slot 1 = seen fainted (Ogerpon): revealed AND fainted flag set
    assert vec[s1 + _IS_REVEALED] == 1.0
    assert vec[s1 + _IS_FAINTED] == 1.0
    # slot 2 = unseen stub (Landorus): not revealed, not fainted
    assert vec[s2 + _IS_REVEALED] == 0.0
    assert vec[s2 + _IS_FAINTED] == 0.0
    # slot 3 = empty → all zeros
    assert np.all(vec[s3:s3 + POKEMON_FEATURES] == 0.0)


def test_active_and_own_bench_are_never_flagged_fainted():
    vec = StateEncoder().encode_snapshot(_snap(), turn=3)
    # active slots 0..3 and own-bench slots 4..7 all carry is_fainted = 0
    for slot_idx in range(ACTIVE_SLOTS + BENCH_SLOTS):
        base = slot_idx * POKEMON_FEATURES
        assert vec[base + _IS_FAINTED] == 0.0


# ── Team-count globals (last 4 features) ────────────────────────────────────
def test_team_count_globals():
    vec = StateEncoder().encode_snapshot(_snap(), turn=3)
    own_live, opp_live, own_fnt, opp_fnt = vec[-4:]
    # own bench: Rillaboom alive, Urshifu fainted
    assert own_live == pytest.approx(1 / 4)
    assert own_fnt == pytest.approx(1 / 4)
    # opp: 1 seen-alive (Chien-Pao), 1 seen-fainted (Ogerpon); unseen not counted
    assert opp_live == pytest.approx(1 / 4)
    assert opp_fnt == pytest.approx(1 / 4)


def test_unseen_opp_bench_not_counted_as_alive():
    snap = _snap()
    # make ALL opp bench unseen stubs → opp_live and opp_fnt both 0
    snap["opp_bench"] = [_mon("Chien-Pao", seen=False),
                         _mon("Ogerpon", seen=False)]
    vec = StateEncoder().encode_snapshot(snap, turn=1)
    _, opp_live, _, opp_fnt = vec[-4:]
    assert opp_live == 0.0 and opp_fnt == 0.0


# ── Transform / Ditto (Solution A): encode the copied forme + is_transformed ──
def test_transformed_mon_encodes_as_copied_forme():
    ditto = _mon("Ditto")
    ditto.update({"is_transformed": True, "transformed_into": "Garchomp"})
    snap = {
        "our_active": {"our_a": ditto, "our_b": _mon("Garchomp")},
        "opp_active": {}, "our_bench": [], "opp_bench": [],
        "field": {}, "side_conditions": {},
    }
    vec = StateEncoder().encode_snapshot(snap, turn=1)
    a, b = 0, POKEMON_FEATURES          # our_a (Ditto→Garchomp) vs our_b (real Garchomp)
    # types one-hot (40 wide) and base stats (6 wide) match the copied forme
    assert np.array_equal(vec[a + _TYPE1:a + _TYPE1 + 40], vec[b + _TYPE1:b + _TYPE1 + 40])
    assert np.allclose(vec[a + _BASE:a + _BASE + 6], vec[b + _BASE:b + _BASE + 6])
    assert vec[a + _BASE:a + _BASE + 6].sum() > 0     # Garchomp dex present
    # flag set on the transformed Ditto, not on the genuine Garchomp
    assert vec[a + _IS_TRANSFORMED] == 1.0
    assert vec[b + _IS_TRANSFORMED] == 0.0


def test_untransformed_ditto_encodes_as_ditto():
    snap = {
        "our_active": {"our_a": _mon("Ditto"), "our_b": _mon("Garchomp")},
        "opp_active": {}, "our_bench": [], "opp_bench": [],
        "field": {}, "side_conditions": {},
    }
    vec = StateEncoder().encode_snapshot(snap, turn=1)
    a, b = 0, POKEMON_FEATURES
    # a plain Ditto's base stats differ from Garchomp's, and no transform flag
    assert not np.allclose(vec[a + _BASE:a + _BASE + 6], vec[b + _BASE:b + _BASE + 6])
    assert vec[a + _IS_TRANSFORMED] == 0.0


# ── Gap #5: item + ability EFFECT-CATEGORY features ──────────────────────────
from v_dance.encoders.state_encoder import (  # noqa: E402
    ITEM_EFFECT_NAMES, ABILITY_EFFECT_NAMES, NUM_ITEM_EFFECTS, NUM_ABILITY_EFFECTS,
    ITEM_FEATURES, MOVE_FEATURES, NUM_MOVES, item_effect_indices, ability_effect_indices,
    resolve_item_json, resolve_ability_json, dex_unique_ability,
)

# item block base = hp + 2×types + base6 + est6 + known + mega + tera + status7 +
# boosts7 + the 4 move blocks
_ITEM0 = 1 + 20 + 20 + 6 + 6 + 1 + 1 + 1 + 7 + 7 + NUM_MOVES * MOVE_FEATURES
_ABIL0 = _ITEM0 + ITEM_FEATURES                            # ability block base


def _names(idxs, table):
    return {table[i] for i in idxs}


def test_item_effect_indices_categories():
    assert _names(item_effect_indices("focussash"), ITEM_EFFECT_NAMES) == {"has_item", "focus_sash"}
    assert _names(item_effect_indices("choicescarf"), ITEM_EFFECT_NAMES) == {"has_item", "choice", "choice_speed"}
    assert _names(item_effect_indices("choiceband"), ITEM_EFFECT_NAMES) == {"has_item", "choice"}
    assert _names(item_effect_indices("leftovers"), ITEM_EFFECT_NAMES) == {"has_item", "passive_recovery"}
    assert _names(item_effect_indices("occaberry"), ITEM_EFFECT_NAMES) == {"has_item", "resist_berry"}
    assert _names(item_effect_indices("splashplate"), ITEM_EFFECT_NAMES) == {"has_item", "type_boost"}
    # itemless / unknown / mega-stone placeholder
    assert item_effect_indices("") == []
    assert item_effect_indices("nothing") == []
    assert _names(item_effect_indices("megastone"), ITEM_EFFECT_NAMES) == {"has_item"}


def test_ability_effect_indices_categories():
    assert _names(ability_effect_indices("intimidate"), ABILITY_EFFECT_NAMES) == {"intimidate"}
    assert _names(ability_effect_indices("protosynthesis"), ABILITY_EFFECT_NAMES) == {"booster_ability"}
    assert _names(ability_effect_indices("drizzle"), ABILITY_EFFECT_NAMES) == {"weather_setter"}
    assert _names(ability_effect_indices("guts"), ABILITY_EFFECT_NAMES) == {"damage_boost", "guts_boost"}
    assert ability_effect_indices("") == []
    assert ability_effect_indices("illusion") == []   # tracked elsewhere, no stat effect


def test_resolve_item_json_confidence():
    assert resolve_item_json({"known_item": "Assault Vest"}) == ("assaultvest", 1.0)
    # consumed item is no longer held → unknown (matches poke-env nulled item)
    assert resolve_item_json({"known_item": "White Herb", "item_consumed": True}) == ("", 0.0)
    # exact (team-sheet) mon with no item = confirmed itemless at 1.0
    assert resolve_item_json({"exact": {"source": "team_sheet"}}) == ("", 1.0)
    # belief top item at 0.5
    bel = {"belief": {"items": [{"name": "Focus Sash", "p": 0.4}]}}
    assert resolve_item_json(bel) == ("focussash", 0.5)
    assert resolve_item_json({}) == ("", 0.0)


def test_resolve_ability_json_bug8_and_unique():
    # revealed ability on a non-mega mon
    assert resolve_ability_json({"species": "Incineroar", "known_ability": "Intimidate"}) == ("intimidate", 1.0)
    # mega'd mon: pre_mega (user-choosable) ability wins, mega ability ignored
    assert resolve_ability_json(
        {"species": "Gardevoir-Mega", "is_mega": True,
         "known_ability": "Pixilate", "pre_mega_ability": "Trace"}
    ) == ("trace", 1.0)
    # single-ability species is publicly known even with nothing revealed
    assert dex_unique_ability("Zoroark-Hisui") == "Illusion"
    assert resolve_ability_json(
        {"species": "Zoroark-Hisui", "base_species": "Zoroark-Hisui"}
    ) == ("illusion", 1.0)
    # belief fallback at 0.5
    bel = {"species": "Charizard", "base_species": "Charizard",
           "belief": {"abilities": [{"name": "Blaze", "p": 0.7}]}}
    assert resolve_ability_json(bel) == ("blaze", 0.5)


# ── Gap #6: move spread/target-shape flag ────────────────────────────────────
_MOVE0 = 1 + 20 + 20 + 6 + 6 + 1 + 1 + 1 + 7 + 7        # first move block base (70)
_SPREAD_REL = MOVE_FEATURES - 2                          # is_spread position in a move


def test_is_spread_target_both_forms():
    from v_dance.encoders.state_encoder import is_spread_target
    assert is_spread_target("allAdjacentFoes") is True   # offline camelCase
    assert is_spread_target("allAdjacent") is True
    assert is_spread_target("normal") is False
    assert is_spread_target("self") is False
    assert is_spread_target(None) is False

    class _T:                                            # mimic poke-env Target enum
        def __init__(self, n): self.name = n
    assert is_spread_target(_T("ALL_ADJACENT_FOES")) is True
    assert is_spread_target(_T("ALL_ADJACENT")) is True
    assert is_spread_target(_T("NORMAL")) is False


def test_spread_flag_set_for_spread_move_only():
    mon = _mon("Charizard", moves=("Heat Wave", "Close Combat"))
    snap = {"our_active": {"our_a": mon, "our_b": None}, "opp_active": {},
            "our_bench": [], "opp_bench": [], "field": {}, "side_conditions": {}}
    vec = StateEncoder().encode_snapshot(snap, turn=1)
    spread0 = vec[_MOVE0 + 0 * MOVE_FEATURES + _SPREAD_REL]   # Heat Wave (allAdjacentFoes)
    spread1 = vec[_MOVE0 + 1 * MOVE_FEATURES + _SPREAD_REL]   # Close Combat (normal)
    assert spread0 == 1.0
    assert spread1 == 0.0


# ── Gap #7: PP fraction (offline derives from move_pp_used) ──────────────────
_PP_REL = 5                                             # pp_fraction position in a move


def test_pp_fraction_offline_from_move_uses():
    """The OFFLINE path derives pp_fraction from move_pp_used with the SAME max as
    poke-env (base·8//5).  Thunderbolt base pp 15 → max 24; 5 uses → 19/24.  An
    unused move stays 1.0."""
    mon = _mon("Pikachu", moves=("Thunderbolt", "Volt Switch"))
    mon["move_pp_used"] = {"thunderbolt": 5}            # volt switch unused
    snap = {"our_active": {"our_a": mon, "our_b": None}, "opp_active": {},
            "our_bench": [], "opp_bench": [], "field": {}, "side_conditions": {}}
    vec = StateEncoder().encode_snapshot(snap, turn=1)
    pp0 = vec[_MOVE0 + 0 * MOVE_FEATURES + _PP_REL]     # Thunderbolt (used 5)
    pp1 = vec[_MOVE0 + 1 * MOVE_FEATURES + _PP_REL]     # Volt Switch (unused)
    assert pp0 == pytest.approx((24 - 5) / 24)
    assert pp1 == pytest.approx(1.0)


def test_item_ability_block_written_in_snapshot():
    mon = _mon("Incineroar")
    mon.update({"known_item": "Assault Vest", "known_ability": "Intimidate"})
    snap = {
        "our_active": {"our_a": mon, "our_b": None}, "opp_active": {},
        "our_bench": [], "opp_bench": [], "field": {}, "side_conditions": {},
    }
    vec = StateEncoder().encode_snapshot(snap, turn=1)
    item_on = {ITEM_EFFECT_NAMES[k] for k in range(NUM_ITEM_EFFECTS) if vec[_ITEM0 + k] == 1.0}
    assert item_on == {"has_item", "assault_vest"}
    assert vec[_ITEM0 + NUM_ITEM_EFFECTS] == 1.0          # item_known
    abil_on = {ABILITY_EFFECT_NAMES[k] for k in range(NUM_ABILITY_EFFECTS) if vec[_ABIL0 + k] == 1.0}
    assert abil_on == {"intimidate"}
    assert vec[_ABIL0 + NUM_ABILITY_EFFECTS] == 1.0       # ability_known
