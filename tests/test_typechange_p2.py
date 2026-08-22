"""v11 P2 — runtime TYPECHANGE (Protean/Libero/Color Change/Soak/Conversion/Burn Up/Double Shock).

A |-start|MON|typechange| line REPLACES a mon's types until it switches / teras / transforms. The OFFLINE
parser previously dropped the type argument, so _effective_types desynced from poke-env (whose type_1/type_2
properties fold _temporary_types) every turn a type had been changed. P2 captures the result types into a
dedicated PokemonSlot.runtime_types field, read by _effective_types at 2nd precedence (tera > runtime > dex).

Parity: offline reads runtime_types; live reads mon.type_1/type_2 (poke-env folds _temporary_types at the
SAME precedence) — NO functional live change. Both come from the SAME typechange protocol line. Verified vs
poke-env pokemon.py:557-560 (start_effect), :1360-1394 (type_1/type_2), :602/:613/:622 (switch/tera/transform
clears), abstract_battle.py:799-806 (the [of] source-copy). VALUE-ONLY: STATE_DIM unchanged (no new channel).
"""
from __future__ import annotations

from types import SimpleNamespace

from v_dance.parser.vod_parser.replay_parser import ShowdownReplayParser
from v_dance.encoders.state_encoder import (
    _effective_types, _canon, get_state_dim, get_state_layout_version,
)
from v_dance.encoders.live_state_encoder import _live_eff_types

HEADER = """|player|p1|A|1|
|player|p2|B|1|
|poke|p1|Meowscarada, L50, M|
|poke|p1|Dragonite, L50, M|
|poke|p2|Garchomp, L50, M|
|teamsize|p1|2
|teamsize|p2|1
|start
|switch|p1a: Meowscarada|Meowscarada, L50, M|100/100
|switch|p2a: Garchomp|Garchomp, L50, M|100/100
"""


def _parse(body):
    return ShowdownReplayParser(HEADER + body + "|turn|999\n", our_player="p1").parse()


def _rt(result, n, rel="our_a"):
    for t in result["turns"]:
        if t["turn"] == n:
            return t["state_before_actions"]["p1"]["our_active"][rel]["runtime_types"]
    raise AssertionError(f"turn {n} not found")


def _mon_snap(result, n, rel="our_a"):
    for t in result["turns"]:
        if t["turn"] == n:
            return t["state_before_actions"]["p1"]["our_active"][rel]
    raise AssertionError(f"turn {n} not found")


# ════════════════════════════ parser capture ════════════════════════════
def test_typechange_single_type():
    # Protean fires on move use: |-start|MON|typechange|Dark|[from] ability: Protean
    r = _parse("|turn|1\n|move|p1a: Meowscarada|Knock Off|p2a: Garchomp\n"
               "|-start|p1a: Meowscarada|typechange|Dark|[from] ability: Protean\n|turn|2\n")
    assert _rt(r, 1) is None
    assert _rt(r, 2) == ["Dark"]


def test_typechange_dual_type():
    r = _parse("|turn|1\n|-start|p1a: Meowscarada|typechange|Normal/Water\n|turn|2\n")
    assert _rt(r, 2) == ["Normal", "Water"]


def test_typechange_end_clears():
    r = _parse("|turn|1\n|-start|p1a: Meowscarada|typechange|Water\n|turn|2\n"
               "|-end|p1a: Meowscarada|typechange\n|turn|3\n")
    assert _rt(r, 2) == ["Water"]
    assert _rt(r, 3) is None


def test_typechange_switch_resets():
    r = _parse("|turn|1\n|-start|p1a: Meowscarada|typechange|Water\n"
               "|switch|p1a: Dragonite|Dragonite, L50, M|100/100\n|turn|2\n"
               "|switch|p1a: Meowscarada|Meowscarada, L50, M|100/100\n|turn|3\n")
    assert _rt(r, 3) is None


def test_typechange_tera_clears():
    r = _parse("|turn|1\n|-start|p1a: Meowscarada|typechange|Water\n|turn|2\n"
               "|-terastallize|p1a: Meowscarada|Fire\n|turn|3\n")
    assert _rt(r, 2) == ["Water"]
    snap3 = _mon_snap(r, 3)
    assert snap3["runtime_types"] is None
    assert snap3["is_terastallized"] is True
    # tera wins in the encoder even though the parser already nulled runtime_types
    assert _effective_types(snap3) == ["FIRE"]


def test_typechange_transform_clears():
    r = _parse("|turn|1\n|-start|p1a: Meowscarada|typechange|Water\n|turn|2\n"
               "|-transform|p1a: Meowscarada|p2a: Garchomp\n|turn|3\n")
    assert _rt(r, 2) == ["Water"]
    assert _rt(r, 3) is None


def test_reflect_type_of_copy():
    # |-start|USER|typechange|<tag>|[of] SOURCE  → copy the source's CURRENT effective types (Garchomp dex)
    r = _parse("|turn|1\n"
               "|-start|p1a: Meowscarada|typechange|[from] move: Reflect Type|[of] p2a: Garchomp\n|turn|2\n")
    rt = _rt(r, 2)
    assert rt is not None
    assert [t.upper() for t in rt] == ["DRAGON", "GROUND"]   # Garchomp dex order


def test_typeadd_ignored():
    # Forest's Curse / Trick-or-Treat emit |-start|MON|typeadd| — poke-env never sets _temporary_types for
    # these, so offline must NOT capture them (capturing would add a type live cannot see → drift).
    r = _parse("|turn|1\n|-start|p1a: Meowscarada|typeadd|Grass\n|turn|2\n")
    assert _rt(r, 2) is None


# ════════════════════════════ offline encoder precedence ════════════════════════════
def test_effective_types_runtime_branch():
    assert _effective_types({"species": "Meowscarada", "runtime_types": ["Water"]}) == ["WATER"]
    assert _effective_types({"species": "Meowscarada", "runtime_types": ["Normal", "Water"]}) == ["NORMAL", "WATER"]


def test_effective_types_tera_beats_runtime():
    assert _effective_types({"species": "Meowscarada", "is_terastallized": True,
                             "known_tera_type": "Fire", "runtime_types": ["Water"]}) == ["FIRE"]


def test_effective_types_runtime_beats_transform():
    assert _effective_types({"species": "Meowscarada", "is_transformed": True,
                             "transformed_into": "Garchomp", "runtime_types": ["Water"]}) == ["WATER"]


def test_effective_types_none_falls_to_dex():
    assert _effective_types({"species": "Garchomp"}) == ["DRAGON", "GROUND"]   # dex order


# ════════════════════════════ live ↔ offline contract ════════════════════════════
class _Ty:
    def __init__(self, name):
        self.name = name


def test_live_eff_types_folds_typechange():
    # poke-env's type_1/type_2 already fold _temporary_types → the live read mirrors the offline runtime branch
    mono = SimpleNamespace(is_terastallized=False, tera_type=None, type_1=_Ty("WATER"), type_2=None)
    assert _live_eff_types(mono) == ["WATER"]
    dual = SimpleNamespace(is_terastallized=False, tera_type=None, type_1=_Ty("NORMAL"), type_2=_Ty("WATER"))
    assert _live_eff_types(dual) == ["NORMAL", "WATER"]
    tera = SimpleNamespace(is_terastallized=True, tera_type=_Ty("FIRE"), type_1=_Ty("WATER"), type_2=None)
    assert _live_eff_types(tera) == ["FIRE"]


def test_offline_live_agree_on_typechange():
    # the load-bearing parity contract: identical canonical type list from both paths for the same typechange
    off = _effective_types({"species": "Meowscarada", "runtime_types": ["Water"]})
    live = _live_eff_types(SimpleNamespace(is_terastallized=False, tera_type=None,
                                           type_1=_Ty("WATER"), type_2=None))
    assert off == live == ["WATER"]


# ════════════════════════════ layout (value-only) ════════════════════════════
def test_layout_unchanged():
    assert get_state_dim() == 5057
    assert get_state_layout_version() == 19
