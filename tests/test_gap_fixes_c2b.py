"""v11 Phase C.2b — gap-fixes before C.3:
  · Persistent team-protection SIDE conditions that had a channel but were never populated: Safeguard
    (status immunity), Mist (stat-drop/Intimidate immunity), Lucky Chant (crit immunity).
  · Yawn → a `drowsy` volatile channel (the mon sleeps NEXT turn unless it switches).
Both are pure id/flag derivations → offline↔live byte-parity. Safeguard/Mist/Lucky-Chant reuse existing
SC channels (value-only); Yawn adds a volatile channel (layout v13→v14).
"""
from __future__ import annotations

import numpy as np

from v_dance.parser.vod_parser.replay_parser import ShowdownReplayParser
from v_dance.parser.vod_parser.battle_models import volatile_flags
from v_dance.encoders.state_encoder import (
    StateEncoder, POKEMON_FEATURES, NUM_WEATHER, NUM_FIELDS, _SC_IDX, get_state_dim,
    get_state_layout_version, VOLATILE_FEATURES,
)
from v_dance.encoders.live_state_encoder import LiveStateEncoder

_OWN_SC = 12 * POKEMON_FEATURES + NUM_WEATHER + NUM_FIELDS

HEADER = """|player|p1|Alice|1|
|player|p2|Bob|1|
|poke|p1|Garchomp, L50, M|
|poke|p2|Clefable, L50, F|
|teamsize|p1|1
|teamsize|p2|1
|start
|switch|p1a: Garchomp|Garchomp, L50, M|100/100
|switch|p2a: Clefable|Clefable, L50, F|100/100
"""


def _parse(body):
    return ShowdownReplayParser(HEADER + body + "|turn|999\n", our_player="p1").parse()


def _sc(result, turn, side="our_side"):
    for t in result["turns"]:
        if t["turn"] == turn:
            return t["state_before_actions"]["p1"]["side_conditions"][side]
    raise AssertionError(turn)


# ── Safeguard / Mist / Lucky Chant tracking + encode ──────────────────────────
def test_persistent_protections_tracked():
    r = _parse(
        "|turn|1\n|-sidestart|p1: Alice|Safeguard\n|-sidestart|p1: Alice|Mist\n"
        "|-sidestart|p1: Alice|move: Lucky Chant\n|turn|2\n"
    )
    s = _sc(r, 2)
    assert s["safeguard"] is True and s["mist"] is True and s["lucky_chant"] is True


def test_protections_cleared_on_sideend():
    r = _parse(
        "|turn|1\n|-sidestart|p1: Alice|Safeguard\n"
        "|turn|2\n|-sideend|p1: Alice|Safeguard\n|turn|3\n"
    )
    assert _sc(r, 2)["safeguard"] is True
    assert _sc(r, 3)["safeguard"] is False


def _encode_sc(our_sc):
    snap = {"our_active": {"our_a": None, "our_b": None}, "opp_active": {"opp_a": None, "opp_b": None},
            "our_bench": [], "opp_bench": [], "field": {},
            "side_conditions": {"our_side": our_sc, "opp_side": {}}}
    return StateEncoder().encode_snapshot(snap, turn=1)


def test_encode_persistent_protection_channels():
    v = _encode_sc({"safeguard": True, "mist": True, "lucky_chant": True})
    assert v[_OWN_SC + _SC_IDX["SAFEGUARD"]] == 1.0
    assert v[_OWN_SC + _SC_IDX["MIST"]] == 1.0
    assert v[_OWN_SC + _SC_IDX["LUCKY_CHANT"]] == 1.0


def test_live_offline_sc_idx_includes_protections():
    enc = LiveStateEncoder(own_username="Alice")
    for n in ("SAFEGUARD", "MIST", "LUCKY_CHANT"):
        assert _SC_IDX[n] in enc._offline_sc_idx


# ── Yawn → drowsy volatile channel ────────────────────────────────────────────
def test_drowsy_flag():
    assert volatile_flags({"yawn"})["drowsy"] is True
    assert volatile_flags(set())["drowsy"] is False


def _vol(body, turn):
    r = _parse(body)
    for t in r["turns"]:
        if t["turn"] == turn:
            return t["state_before_actions"]["p1"]["our_active"]["our_a"]["volatiles"]
    raise AssertionError(turn)


def test_yawn_tracked_and_cleared():
    v = _vol("|turn|1\n|-start|p1a: Garchomp|move: Yawn\n|turn|2\n", 2)
    assert v["drowsy"] is True
    v2 = _vol("|turn|1\n|-start|p1a: Garchomp|move: Yawn\n"
              "|turn|2\n|-end|p1a: Garchomp|move: Yawn\n|turn|3\n", 3)
    assert v2["drowsy"] is False


def _encode_vol(volatiles):
    mon = {"species": "Garchomp", "base_species": "Garchomp", "hp_pct": 100.0, "seen": True,
           "is_fainted": False, "known_moves": [], "revealed_moves": [], "boosts": {}, "status": None,
           "known_ability": "Rough Skin", "volatiles": volatiles,
           "stats_estimate": {"mode": "exact",
                              "stats": {"atk": 130, "spa": 80, "def": 95, "spd": 85, "hp": 180, "spe": 102}}}
    snap = {"our_active": {"our_a": mon, "our_b": None}, "opp_active": {"opp_a": None, "opp_b": None},
            "our_bench": [], "opp_bench": [], "field": {}, "side_conditions": {}}
    return StateEncoder().encode_snapshot(snap, turn=1)


def test_drowsy_channel_placement():
    base = _encode_vol({})
    v = _encode_vol({"drowsy": True})
    diff = np.flatnonzero(np.abs(v - base) > 1e-6)
    assert len(diff) == 1 and float(v[diff[0]]) == 1.0


def test_layout_v14():
    assert VOLATILE_FEATURES == 12
    assert get_state_dim() == 5057       # v16 (B2b: +2 per-move hit-chance channels)
    assert get_state_layout_version() == 19
