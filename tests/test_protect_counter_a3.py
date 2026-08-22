"""v11 Phase A.3 — the |-singleturn| handler + consecutive-Protect counter.

The only |-singleturn| signal that survives to the next decision snapshot AND is exposed identically by
poke-env is the protect_counter, so A.3 encodes that and nothing else. The offline counter is driven from
the |move| line to byte-match poke-env Pokemon.moved (increment a self-selected non-failed protect-counter
move, else reset; reset on switch / |cant|). The |-singleturn| dispatch is audit-only and must NOT mutate
volatiles or the counter. Design: scratch + memory v11 note; parity contract verified vs poke-env source.
"""
from __future__ import annotations

import math

import numpy as np

from v_dance.parser.vod_parser.replay_parser import ShowdownReplayParser, _PROTECT_COUNTER_MOVES
from v_dance.encoders.state_encoder import (
    StateEncoder, get_state_dim, get_state_layout_version, POKEMON_FEATURES, VOLATILE_FEATURES,
    _PROTECT_COUNTER_CAP,
)

HEADER = """|player|p1|Alice|1|
|player|p2|Bob|1|
|poke|p1|Kingambit, L50, M|
|poke|p1|Incineroar, L50, M|
|poke|p2|Garchomp, L50, M|
|poke|p2|Rotom-Wash, L50|
|teamsize|p1|2
|teamsize|p2|2
|start
|switch|p1a: Kingambit|Kingambit, L50, M|100/100
|switch|p1b: Incineroar|Incineroar, L50, M|100/100
|switch|p2a: Garchomp|Garchomp, L50, M|100/100
|switch|p2b: Rotom-Wash|Rotom-Wash, L50|100/100
"""


def _parse(body):
    # Append a trailing |turn| so the LAST inspected turn's snapshot gets flushed into result["turns"]
    # (a turn is flushed when the next |turn| arrives).
    return ShowdownReplayParser(HEADER + body + "|turn|999\n", our_player="p1").parse()


def _make(body):
    p = ShowdownReplayParser(HEADER + body, our_player="p1")
    p.parse()
    return p


def _turn(result, n):
    for t in result["turns"]:
        if t["turn"] == n:
            return t
    raise AssertionError(f"turn {n} not found")


def _pc(result, n, rel="our_a", side="p1"):
    """protect_counter of a mon at the START-of-turn-n snapshot (what the encoder sees)."""
    st = _turn(result, n)["state_before_actions"][side]
    grp = "our_active" if rel.startswith("our") else "opp_active"
    return st[grp][rel]["volatiles"]["protect_counter"]


# ── the move set ──────────────────────────────────────────────────────────────
def test_protect_counter_move_set():
    assert _PROTECT_COUNTER_MOVES == {
        "protect", "detect", "endure", "spikyshield", "kingsshield",
        "banefulbunker", "burningbulwark", "obstruct", "maxguard",
        "silktrap", "wideguard", "quickguard",
    }
    assert "matblock" not in _PROTECT_COUNTER_MOVES   # Mat Block must NOT increment


# ── increment / reset ladder (snapshot = poke-env's value at turn start) ───────
def test_increment_ladder_and_non_protect_reset():
    r = _parse(
        "|turn|1\n|move|p1a: Kingambit|Protect|p1a: Kingambit\n|-singleturn|p1a: Kingambit|Protect\n"
        "|turn|2\n|move|p1a: Kingambit|Protect|p1a: Kingambit\n|-singleturn|p1a: Kingambit|Protect\n"
        "|turn|3\n|move|p1a: Kingambit|Sucker Punch|p2a: Garchomp\n"
        "|turn|4\n"
    )
    assert _pc(r, 2) == 1            # after 1 Protect
    assert _pc(r, 3) == 2            # after 2 consecutive Protects
    assert _pc(r, 4) == 0            # turn-3 Sucker Punch reset it


def test_failed_suffix_resets():
    for suffix in ("[still]", "[miss]", "[notarget]"):
        r = _parse(
            "|turn|1\n|move|p1a: Kingambit|Protect|p1a: Kingambit\n|-singleturn|p1a: Kingambit|Protect\n"
            f"|turn|2\n|move|p1a: Kingambit|Protect||{suffix}\n"
            "|turn|3\n"
        )
        assert _pc(r, 2) == 1
        assert _pc(r, 3) == 0, f"a {suffix} move must reset the counter"


def test_fail_line_is_irrelevant_counter_from_move_suffix():
    # poke-env reads failure from the |move| suffix only; a trailing |-fail| changes nothing.
    r = _parse(
        "|turn|1\n|move|p1a: Kingambit|Protect||[still]\n|-fail|p1a: Kingambit\n"
        "|turn|2\n"
    )
    assert _pc(r, 2) == 0


def test_switch_resets_counter_both_sides():
    r = _parse(
        "|turn|1\n|move|p1a: Kingambit|Protect|p1a: Kingambit\n|-singleturn|p1a: Kingambit|Protect\n"
        "|turn|2\n|switch|p1a: Incineroar|Incineroar, L50, M|100/100\n"
        "|turn|3\n"
    )
    assert _pc(r, 2) == 1
    assert _pc(r, 3, rel="our_a") == 0      # Incineroar (incoming) starts at 0


def test_cant_resets_counter():
    r = _parse(
        "|turn|1\n|move|p1a: Kingambit|Protect|p1a: Kingambit\n|-singleturn|p1a: Kingambit|Protect\n"
        "|turn|2\n|cant|p1a: Kingambit|flinch\n"
        "|turn|3\n"
    )
    assert _pc(r, 2) == 1
    assert _pc(r, 3) == 0


def test_from_called_move_does_not_increment():
    # a [from]-called protect-family move is excluded (poke-env use=False path) — documented residual.
    r = _parse(
        "|turn|1\n|move|p1a: Kingambit|Protect|p1a: Kingambit|[from]move: Copycat\n"
        "|turn|2\n"
    )
    assert _pc(r, 2) == 0


def test_endure_increments_but_matblock_does_not():
    r_end = _parse(
        "|turn|1\n|move|p1a: Kingambit|Endure|p1a: Kingambit\n"
        "|turn|2\n"
    )
    assert _pc(r_end, 2) == 1            # Endure IS a protect-counter move (offline is_protect misses it)
    r_mat = _parse(
        "|turn|1\n|move|p1a: Kingambit|Mat Block|p1a: Kingambit\n"
        "|turn|2\n"
    )
    assert _pc(r_mat, 2) == 0            # Mat Block is NOT a protect-counter move


def test_kings_shield_and_obstruct_increment():
    for mv in ("King's Shield", "Obstruct"):
        r = _parse(f"|turn|1\n|move|p1a: Kingambit|{mv}|p1a: Kingambit\n|turn|2\n")
        assert _pc(r, 2) == 1, f"{mv} should increment (in _PROTECT_COUNTER_MOVES, not in is_protect)"


# ── |-singleturn| dispatch is audit-only ──────────────────────────────────────
def test_singleturn_does_not_pollute_volatiles_or_counter():
    p = _make(
        "|turn|1\n|-singleturn|p1a: Kingambit|move: Protect\n"
        "|-singleturn|p1a: Kingambit|move: Follow Me\n|-singleturn|p1a: Kingambit|Helping Hand\n"
        "|turn|2\n"
    )
    mon = p.active_slots["p1a"]
    assert mon.volatiles == set(), "|-singleturn| must NOT add persistent volatiles"
    assert mon.protect_counter == 0, "|-singleturn| must NOT drive the counter (|move| does)"


def test_singlemove_handled_without_crash():
    p = _make("|turn|1\n|-singlemove|p1a: Kingambit|Destiny Bond\n|turn|2\n")
    assert p.active_slots["p1a"].volatiles == set()


# ── encoding: normalized channel, byte-identical clamp ────────────────────────
def _encode_with_pc(pc):
    mon = {
        "species": "Kingambit", "base_species": "Kingambit", "hp_pct": 100.0,
        "seen": True, "is_fainted": False, "known_moves": [], "revealed_moves": [],
        "boosts": {}, "status": None, "known_ability": "Supreme Overlord",
        "stats_estimate": {"mode": "exact",
                           "stats": {"atk": 130, "spa": 60, "def": 120, "spd": 100, "hp": 200, "spe": 50}},
        "volatiles": {"protect_counter": pc},
    }
    snap = {"our_active": {"our_a": mon, "our_b": None},
            "opp_active": {"opp_a": None, "opp_b": None},
            "our_bench": [], "opp_bench": [], "field": {}, "side_conditions": {}}
    return StateEncoder().encode_snapshot(snap, turn=1)


def test_channel_normalization_and_placement():
    base = _encode_with_pc(0)
    for pc, expected in [(1, 1 / 3), (2, 2 / 3), (3, 1.0), (5, 1.0)]:
        v = _encode_with_pc(pc)
        diff = np.flatnonzero(np.abs(v - base) > 1e-6)
        assert len(diff) == 1, f"exactly one channel should change for pc={pc}, got {len(diff)}"
        assert math.isclose(float(v[diff[0]]), expected, rel_tol=1e-4, abs_tol=1e-6), \
            f"pc={pc} -> {float(v[diff[0]])}, expected {expected}"


def test_layout_has_protect_counter_channel():
    assert VOLATILE_FEATURES == 12       # A.3 protect_counter + C.2's 3 + C.2b's drowsy + C.3's first_turn
    assert _PROTECT_COUNTER_CAP == 3.0
    # absolute dims move with later v11 phases; pin them in the dedicated layout test (test_state_encoder).
    assert get_state_dim() == 12 * POKEMON_FEATURES + 101
