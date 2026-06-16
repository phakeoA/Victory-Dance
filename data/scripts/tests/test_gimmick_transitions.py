"""Task #3: the gimmick (mega) flag is threaded through transitions.py and a
gimmick_mask is emitted on BOTH turn and replacement transitions.

Drives the real replay_to_transitions pipeline with synthetic logs (belief=None),
mirroring test_replacement_transitions.py.
"""

from __future__ import annotations

import pytest

from vod_parser.transitions import replay_to_transitions

pytest.importorskip("numpy")


_HEADER = [
    "|player|p1|Alice|1|1500",
    "|player|p2|Bob|2|1500",
    "|teamsize|p1|6",
    "|teamsize|p2|6",
    "|gen|9",
    "|tier|[Gen 9] VGC 2026 Reg MA",
    "|poke|p1|Venusaur, L50, M|",
    "|poke|p1|Incineroar, L50, M|",
    "|poke|p1|Rillaboom, L50, M|",
    "|poke|p1|Urshifu, L50|",
    "|poke|p1|Flutter Mane, L50|",
    "|poke|p1|Amoonguss, L50|",
    "|poke|p2|Calyrex-Shadow, L50|",
    "|poke|p2|Miraidon, L50|",
    "|poke|p2|Chien-Pao, L50|",
    "|poke|p2|Ogerpon, L50|",
    "|poke|p2|Landorus, L50|",
    "|poke|p2|Grimmsnarl, L50, M|",
    "|teampreview",
    "|start",
    "|switch|p1a: Venusaur|Venusaur, L50, M|100/100",
    "|switch|p1b: Incineroar|Incineroar, L50, M|100/100",
    "|switch|p2a: Calyrex|Calyrex-Shadow, L50|100/100",
    "|switch|p2b: Miraidon|Miraidon, L50|100/100",
]


def _wrap(lines):
    log = "\n".join(lines)
    return ('<input type="hidden" name="replayid" value="syn-gimmick-1" />'
            f'<script type="text/plain" class="battle-log-data">{log}</script>')


def _run(tmp_path, lines, players=("p1", "p2")):
    f = tmp_path / "syn.html"
    f.write_text(_wrap(lines), encoding="utf-8")
    return replay_to_transitions(f, belief=None, players=list(players), source_type="B")


def _pick(transitions, turn, persp, dtype="turn"):
    return next(t for t in transitions
               if t["turn"] == turn and t["perspective"] == persp
               and t["decision_type"] == dtype)


# Venusaur reveals Body Press on turn 1, then mega-evolves + uses it again on
# turn 2 (so the turn-2 move is encodable from turn-1's reveal, belief=None).
_MEGA_LOG = _HEADER + [
    "|turn|1",
    "|move|p1a: Venusaur|Body Press|p2a: Calyrex",
    "|move|p1b: Incineroar|Knock Off|p2a: Calyrex",
    "|turn|2",
    "|detailschange|p1a: Venusaur|Venusaur-Mega, L50, M",
    "|-mega|p1a: Venusaur|Venusaur|Venusaurite",
    "|move|p1a: Venusaur|Body Press|p2a: Calyrex",
    "|move|p1b: Incineroar|Knock Off|p2a: Calyrex",
    "|turn|3",
    "|win|Alice",
]


def test_turn_transition_mega_move_has_gimmick_index_and_mask(tmp_path):
    t = _pick(_run(tmp_path, _MEGA_LOG), 2, "p1")
    our = {a["slot"]: a for a in t["our_actions"]}
    # p1a Venusaur mega'd this turn
    assert our["our_a"].get("mega") is True
    assert our["our_a"]["action_index"] is not None
    assert our["our_a"]["gimmick_index"] == 1
    # p1b Incineroar did not mega
    assert "mega" not in our["our_b"]
    assert our["our_b"]["gimmick_index"] == 0
    # gimmick mask: Venusaur is mega-capable -> [1,1]; Incineroar -> [1,0]
    assert t["gimmick_mask"]["our_a"] == [1, 1]
    assert t["gimmick_mask"]["our_b"] == [1, 0]


def test_gimmick_flag_threads_to_opponent_perspective(tmp_path):
    """From p2's view p1a's mega lands in opp_actions_actual with mega=True (the
    flag survives normalize_slots).  Opp entries carry no gimmick_index — only
    our_actions are annotated."""
    t = _pick(_run(tmp_path, _MEGA_LOG), 2, "p2")
    opp = {a["slot"]: a for a in t["opp_actions_actual"]}
    assert opp["opp_a"].get("mega") is True
    assert "gimmick_index" not in opp["opp_a"]


def test_turn_gimmick_mask_present_on_every_turn_transition(tmp_path):
    for t in _run(tmp_path, _MEGA_LOG):
        if t["decision_type"] == "turn":
            assert set(t["gimmick_mask"]) == {"our_a", "our_b"}
            assert all(len(r) == 2 for r in t["gimmick_mask"].values())


# Venusaur faints turn 1 → p1 forced replacement (switch-only, never gimmicks).
_REPL_LOG = _HEADER + [
    "|turn|1",
    "|move|p2b: Miraidon|Electro Drift|p1a: Venusaur",
    "|-damage|p1a: Venusaur|0 fnt",
    "|faint|p1a: Venusaur",
    "|switch|p1a: Rillaboom|Rillaboom, L50, M|100/100",
    "|turn|2",
    "|win|Bob",
]


def test_replacement_transition_gimmick_is_switch_only(tmp_path):
    repl = next(t for t in _run(tmp_path, _REPL_LOG)
                if t["decision_type"] == "replacement")
    assert repl["perspective"] == "p1"
    act = repl["our_actions"][0]
    assert act["action"] == "switch"
    assert act["action_index"] is not None
    assert act["gimmick_index"] == 0                  # a replacement never gimmicks
    # deciding (fainted) slot our_a: only "none" legal; ally our_b not deciding.
    assert repl["gimmick_mask"]["our_a"] == [1, 0]
    assert repl["gimmick_mask"]["our_b"] == [0, 0]
