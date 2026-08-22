"""v11 Phase C.1 — entry hazards (Stealth Rock / Spikes / Toxic Spikes / Sticky Web).

The parser tracked only tailwind + screens; the 4 hazard SC channels existed but were never populated
(offline OR live, via the _offline_sc_idx restriction). C.1 tracks hazards in SideConditions (Spikes 0-3 /
Toxic Spikes 0-2 stack; SR/Web presence), encodes Spikes/Toxic-Spikes as a normalised layer count + SR/Web
as presence, adds them to _offline_sc_idx (live), and handles Court Change (|-swapsideconditions|).
"""
from __future__ import annotations

from v_dance.parser.vod_parser.replay_parser import ShowdownReplayParser
from v_dance.encoders.state_encoder import (
    StateEncoder, POKEMON_FEATURES, NUM_WEATHER, NUM_FIELDS, NUM_SIDE_CONDS, _SC_IDX,
)
from v_dance.encoders.live_state_encoder import LiveStateEncoder

HEADER = """|player|p1|Alice|1|
|player|p2|Bob|1|
|poke|p1|Garchomp, L50, M|
|poke|p2|Skarmory, L50, M|
|teamsize|p1|1
|teamsize|p2|1
|start
|switch|p1a: Garchomp|Garchomp, L50, M|100/100
|switch|p2a: Skarmory|Skarmory, L50, M|100/100
"""

# own_side SC block base in the global feature region
_OWN_SC = 12 * POKEMON_FEATURES + NUM_WEATHER + NUM_FIELDS
_OPP_SC = _OWN_SC + NUM_SIDE_CONDS


def _parse(body):
    return ShowdownReplayParser(HEADER + body + "|turn|999\n", our_player="p1").parse()


def _sc(result, turn, side="our_side"):
    for t in result["turns"]:
        if t["turn"] == turn:
            return t["state_before_actions"]["p1"]["side_conditions"][side]
    raise AssertionError(f"turn {turn} not found")


# ── parser tracking ───────────────────────────────────────────────────────────
def test_spikes_stack_and_other_hazards_separate():
    r = _parse(
        "|turn|1\n|move|p2a: Skarmory|Spikes|p1a: Garchomp\n|-sidestart|p1: Alice|Spikes\n"
        "|turn|2\n|move|p2a: Skarmory|Spikes|p1a: Garchomp\n|-sidestart|p1: Alice|Spikes\n"
        "|move|p2a: Skarmory|Stealth Rock|p1a: Garchomp\n|-sidestart|p1: Alice|move: Stealth Rock\n"
        "|move|p2a: Skarmory|Toxic Spikes|p1a: Garchomp\n|-sidestart|p1: Alice|move: Toxic Spikes\n"
        "|turn|3\n"
    )
    s = _sc(r, 3)
    assert s["spikes"] == 2            # two layers stacked
    assert s["stealth_rock"] is True
    assert s["toxic_spikes"] == 1      # "Toxic Spikes" not double-counted as "Spikes"
    assert s["sticky_web"] is False


def test_spikes_layers_cap_at_3():
    r = _parse(
        "|turn|1\n" + "".join(f"|-sidestart|p1: Alice|Spikes\n" for _ in range(5)) + "|turn|2\n"
    )
    assert _sc(r, 2)["spikes"] == 3


def test_hazard_removal_clears_all_layers():
    r = _parse(
        "|turn|1\n|-sidestart|p1: Alice|Spikes\n|-sidestart|p1: Alice|Spikes\n"
        "|turn|2\n|move|p1a: Garchomp|Rapid Spin|p2a: Skarmory\n|-sideend|p1: Alice|Spikes\n"
        "|turn|3\n"
    )
    assert _sc(r, 2)["spikes"] == 2
    assert _sc(r, 3)["spikes"] == 0


def test_court_change_swaps_sides():
    # SR on p1 (our side); Court Change swaps it to p2 (opp side).
    r = _parse(
        "|turn|1\n|-sidestart|p1: Alice|move: Stealth Rock\n"
        "|turn|2\n|move|p1a: Garchomp|Court Change|p1a: Garchomp\n|-swapsideconditions\n"
        "|turn|3\n"
    )
    assert _sc(r, 2, "our_side")["stealth_rock"] is True
    assert _sc(r, 3, "our_side")["stealth_rock"] is False
    assert _sc(r, 3, "opp_side")["stealth_rock"] is True


# ── offline encode ────────────────────────────────────────────────────────────
def _encode(our_sc=None, opp_sc=None):
    snap = {"our_active": {"our_a": None, "our_b": None},
            "opp_active": {"opp_a": None, "opp_b": None},
            "our_bench": [], "opp_bench": [], "field": {},
            "side_conditions": {"our_side": our_sc or {}, "opp_side": opp_sc or {}}}
    return StateEncoder().encode_snapshot(snap, turn=1)


def test_encode_hazard_channels():
    v = _encode(our_sc={"spikes": 2, "stealth_rock": True, "toxic_spikes": 1, "sticky_web": True})
    assert v[_OWN_SC + _SC_IDX["SPIKES"]] == 2 / 3                 # normalised layer count
    assert v[_OWN_SC + _SC_IDX["TOXIC_SPIKES"]] == 1 / 2
    assert v[_OWN_SC + _SC_IDX["STEALTH_ROCK"]] == 1.0
    assert v[_OWN_SC + _SC_IDX["STICKY_WEB"]] == 1.0
    # opp side untouched
    assert v[_OPP_SC + _SC_IDX["SPIKES"]] == 0.0


def test_encode_full_spikes_is_one():
    v = _encode(opp_sc={"spikes": 3, "toxic_spikes": 2})
    assert v[_OPP_SC + _SC_IDX["SPIKES"]] == 1.0
    assert v[_OPP_SC + _SC_IDX["TOXIC_SPIKES"]] == 1.0


# ── live parity: hazards now in _offline_sc_idx + layer-aware value ───────────
def test_live_sc_value_layer_aware():
    class _SC:
        def __init__(self, name): self.name = name
    assert LiveStateEncoder._sc_value(_SC("SPIKES"), 2) == 2 / 3
    assert LiveStateEncoder._sc_value(_SC("SPIKES"), 3) == 1.0
    assert LiveStateEncoder._sc_value(_SC("TOXIC_SPIKES"), 1) == 1 / 2
    assert LiveStateEncoder._sc_value(_SC("STEALTH_ROCK"), 5) == 1.0     # start-turn val → presence
    assert LiveStateEncoder._sc_value(_SC("TAILWIND"), 3) == 1.0


def test_live_offline_sc_idx_includes_hazards():
    enc = LiveStateEncoder(own_username="Alice")
    for n in ("STEALTH_ROCK", "SPIKES", "TOXIC_SPIKES", "STICKY_WEB"):
        assert _SC_IDX[n] in enc._offline_sc_idx
