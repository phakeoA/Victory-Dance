"""v11 P5 — Magic Room + Wonder Room (pseudo-weather field conditions).

Parser: FieldConditions.magic_room / wonder_room (turns remaining; base 5, 7 if [persistent]) via
-fieldstart/-fieldend/upkeep, mirroring Gravity C.2e. Encoder (value-only, channels already exist):
  · WONDER ROOM swaps Def<->SpD in the damage band (one XOR on the defensive-stat selector, both encoders).
  · MAGIC ROOM suppresses ALL held-item effects (Option C, full ignoringItem() parity) via the shared
    _item_active gate at every item-resolution boundary — damage-band mults, Choice Scarf speed, the per-mon
    effect channels (identity + known KEPT), and the AV/Choice action-mask. Both rooms light their existing
    FIELD_NAMES channels. Verified vs poke-env field.py:19/27, pokemon.ts:568-575 (WR read-swap) +
    abstract_battle.py:898-1061. Byte-parity proven by the parity suites. STATE_DIM unchanged.
"""
from __future__ import annotations

from v_dance.parser.vod_parser.replay_parser import ShowdownReplayParser
from v_dance.encoders.state_encoder import (
    StateEncoder, MOVE_FEATURES, _MOVE_BLOCK_REL, NUM_MOVES, NUM_ITEM_TAGS,
    move_slots_for_mon, norm_species, _certainly_illegal_move_slots,
    ACTIVE_SLOTS, BENCH_SLOTS, OPP_BENCH_SLOTS, POKEMON_FEATURES, NUM_WEATHER, _FIELD_IDX,
    get_state_dim,
)

_TT = 2.0 / 3.0
_BAND_MULT = 4915.0 / 4096.0   # Life Orb / Expert Belt ≈ 1.2

HEADER = """|player|p1|A|1|
|player|p2|B|1|
|poke|p1|Garchomp, L50, M|
|poke|p2|Blissey, L50, F|
|teamsize|p1|1
|teamsize|p2|1
|start
|switch|p1a: Garchomp|Garchomp, L50, M|100/100
|switch|p2a: Blissey|Blissey, L50, F|100/100
"""


def _parse(body):
    return ShowdownReplayParser(HEADER + body + "|turn|999\n", our_player="p1").parse()


def _field(result, n):
    for t in result["turns"]:
        if t["turn"] == n:
            return t["state_before_actions"]["p1"]["field"]
    raise AssertionError(f"turn {n} not found")


# ════════════════════════════ parser tracking (mirror Gravity C.2e) ════════════════════════════
def test_parser_room_start_end_upkeep():
    r = _parse("|turn|1\n|-fieldstart|move: Wonder Room|[of] p1a: Garchomp\n|turn|2\n"
               "|upkeep\n|turn|3\n|-fieldend|move: Wonder Room\n|turn|4\n")
    assert _field(r, 2)["wonder_room_turns_remaining"] == 5
    assert _field(r, 3)["wonder_room_turns_remaining"] == 4    # upkeep decremented
    assert _field(r, 4)["wonder_room_turns_remaining"] == 0


def test_parser_magic_room_persistent():
    r = _parse("|turn|1\n|-fieldstart|move: Magic Room|[of] p1a: Garchomp|[persistent]\n|turn|2\n")
    assert _field(r, 2)["magic_room_turns_remaining"] == 7      # Persistent -> 7
    assert _field(r, 2)["wonder_room_turns_remaining"] == 0


# ════════════════════════════ snapshot helpers (model: test_items_b3) ════════════════════════════
def _mon(species, ability="Pressure", *, moves=(), item=None, defn=120, spd=120, spe=80):
    m = {"species": species, "base_species": species, "hp_pct": 100.0, "seen": True, "is_fainted": False,
         "known_moves": list(moves), "revealed_moves": [], "boosts": {}, "status": None,
         "known_ability": ability, "volatiles": {},
         "stats_estimate": {"mode": "exact",
                            "stats": {"atk": 130, "spa": 80, "def": defn, "spd": spd, "hp": 300, "spe": spe}}}
    if item is not None:
        m["known_item"] = item
    return m


def _vec(att, defender, field):
    snap = {"our_active": {"our_a": att, "our_b": None},
            "opp_active": {"opp_a": defender, "opp_b": None},
            "our_bench": [], "opp_bench": [], "field": field, "side_conditions": {}}
    return StateEncoder().encode_snapshot(snap, turn=1)


def _band(att, move, defender, field):
    vec = _vec(att, defender, field)
    for m_idx, (mv, _c) in enumerate(move_slots_for_mon(att)):
        if norm_species(mv) == norm_species(move):
            b = _MOVE_BLOCK_REL + m_idx * MOVE_FEATURES
            return float(vec[b + 14])   # damage MAX vs enemy0
    raise AssertionError(move)


# ════════════════════════════ field channels light ════════════════════════════
def test_field_channels_light():
    fb = (ACTIVE_SLOTS + BENCH_SLOTS + OPP_BENCH_SLOTS) * POKEMON_FEATURES + NUM_WEATHER
    att, deff = _mon("Garchomp", moves=["Earthquake"]), _mon("Arcanine")
    on = _vec(att, deff, {"magic_room_turns_remaining": 5, "wonder_room_turns_remaining": 5})
    off = _vec(att, deff, {})
    assert on[fb + _FIELD_IDX["MAGIC_ROOM"]] == 1.0 and on[fb + _FIELD_IDX["WONDER_ROOM"]] == 1.0
    assert off[fb + _FIELD_IDX["MAGIC_ROOM"]] == 0.0 and off[fb + _FIELD_IDX["WONDER_ROOM"]] == 0.0


# ════════════════════════════ Wonder Room: Def<->SpD band swap ════════════════════════════
def test_wonder_room_swaps_defensive_stat():
    att = _mon("Garchomp", "Rough Skin", moves=["Earthquake"])      # Earthquake = physical (hits Def)
    wall = _mon("Blissey", defn=200, spd=20)                        # high Def, low SpD
    no_wr = _band(att, "Earthquake", wall, {})                      # physical uses Def(200) → low dmg
    wr = _band(att, "Earthquake", wall, {"wonder_room_turns_remaining": 5})  # → uses SpD(20) → high dmg
    assert wr > no_wr * 1.5


def test_wonder_room_off_is_legacy():
    # with no room active the band must be byte-identical to pre-P5 (guards selector leakage)
    att, wall = _mon("Garchomp", "Rough Skin", moves=["Earthquake"]), _mon("Blissey", defn=200, spd=20)
    assert _band(att, "Earthquake", wall, {}) == _band(att, "Earthquake", wall, {"gravity_turns_remaining": 0})


# ════════════════════════════ Magic Room: item suppression ════════════════════════════
_MR = {"magic_room_turns_remaining": 5}


def test_magic_room_suppresses_expert_belt():
    eb = _mon("Garchomp", "Rough Skin", moves=["Earthquake"], item="Expert Belt")
    plain = _mon("Garchomp", "Rough Skin", moves=["Earthquake"])
    arc = _mon("Arcanine")
    assert abs(_band(eb, "Earthquake", arc, {}) / _band(plain, "Earthquake", arc, {}) - _BAND_MULT) < 5e-3
    # under Magic Room the Expert Belt boost is gone → band == the no-item band
    assert abs(_band(eb, "Earthquake", arc, _MR) - _band(plain, "Earthquake", arc, _MR)) < 1e-4


def test_magic_room_suppresses_resist_berry():
    char = _mon("Charizard", "Blaze", moves=["Flamethrower"])
    glac_no = _mon("Glaceon")
    glac_berry = _mon("Glaceon", item="Occa Berry")
    assert _band(char, "Flamethrower", glac_berry, {}) < _band(char, "Flamethrower", glac_no, {})   # berry resists
    # under Magic Room the berry no longer resists → bands equal
    assert abs(_band(char, "Flamethrower", glac_berry, _MR) - _band(char, "Flamethrower", glac_no, _MR)) < 1e-4


def test_magic_room_zeros_item_effect_channels_keeps_identity():
    ib = _MOVE_BLOCK_REL + NUM_MOVES * MOVE_FEATURES          # item block start (after the move block)
    lo = _mon("Garchomp", "Rough Skin", moves=["Earthquake"], item="Life Orb")
    arc = _mon("Arcanine")
    off = _vec(lo, arc, {})
    on = _vec(lo, arc, _MR)
    tags_off = sum(off[ib + 1: ib + 1 + NUM_ITEM_TAGS])
    tags_on = sum(on[ib + 1: ib + 1 + NUM_ITEM_TAGS])
    assert tags_off > 0 and tags_on == 0                     # effect tags suppressed under Magic Room
    assert on[ib] == off[ib] and on[ib] != 0                 # identity index KEPT
    assert on[ib + 1 + NUM_ITEM_TAGS] == off[ib + 1 + NUM_ITEM_TAGS]   # item_known KEPT


# ════════════════════════════ Magic Room: action-mask lift (site H) ════════════════════════════
def test_magic_room_lifts_assault_vest_ban():
    mon = _mon("Arcanine", item="Assault Vest", moves=["Flamethrower", "Will-O-Wisp"])
    slots = [norm_species(n) for n, _ in move_slots_for_mon(mon)]
    wow = slots.index("willowisp")
    assert wow in _certainly_illegal_move_slots(mon, {})          # AV bans the status move
    assert wow not in _certainly_illegal_move_slots(mon, _MR)     # Magic Room lifts the ban


def test_magic_room_lifts_choice_lock():
    mon = _mon("Garchomp", moves=["Earthquake", "Dragon Claw"], item="Choice Scarf")
    mon["stint_moves"] = ["Earthquake"]                          # locked into Earthquake
    dc = [norm_species(n) for n, _ in move_slots_for_mon(mon)].index("dragonclaw")
    assert dc in _certainly_illegal_move_slots(mon, {})           # Choice locks out Dragon Claw
    assert dc not in _certainly_illegal_move_slots(mon, _MR)      # Magic Room lifts the lock


# ════════════════════════════ layout (value-only) ════════════════════════════
def test_layout_unchanged():
    assert get_state_dim() == 5057
