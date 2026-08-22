"""v11 P4 — |-sethp| (Pain Split etc.).

poke-env applies -sethp via Pokemon.set_hp (sets HP directly); the offline parser dropped it, so a Pain
Split (which emits two -sethp lines) left the offline HP stale vs the live path. P4 adds _handle_sethp,
mirroring the HP-setting half of _handle_damage (parse X/Y, keep the denominator for real-HP replays).
Status comes from the dedicated -status/faint events (Pain Split carries no status). Verified vs poke-env
abstract_battle.py:1033-1035 + pokemon.py:515-542. PARSER-ONLY, VALUE: STATE_DIM unchanged; rides re-export.
"""
from __future__ import annotations

from v_dance.parser.vod_parser.replay_parser import ShowdownReplayParser

HEADER = """|player|p1|A|1|
|player|p2|B|1|
|poke|p1|Meowscarada, L50, M|
|poke|p2|Garchomp, L50, M|
|teamsize|p1|1
|teamsize|p2|1
|start
|switch|p1a: Meowscarada|Meowscarada, L50, M|100/100
|switch|p2a: Garchomp|Garchomp, L50, M|100/100
"""


def _parse(body):
    return ShowdownReplayParser(HEADER + body + "|turn|999\n", our_player="p1").parse()


def _hp(result, n, key="our_a", side="our_active"):
    for t in result["turns"]:
        if t["turn"] == n:
            return t["state_before_actions"]["p1"][side][key]["hp_pct"]
    raise AssertionError(f"turn {n} not found")


def test_sethp_percentage():
    r = _parse("|turn|1\n|-sethp|p1a: Meowscarada|50/100\n|turn|2\n")
    assert _hp(r, 1) == 100.0
    assert _hp(r, 2) == 50.0


def test_sethp_pain_split_tag():
    r = _parse("|turn|1\n|-sethp|p1a: Meowscarada|30/100|[from] move: Pain Split|[of] p2a: Garchomp\n|turn|2\n")
    assert _hp(r, 2) == 30.0


def test_sethp_real_hp_denominator():
    # owner-client real HP (X/Y, Y!=100) must normalise to a true percentage (gap #5): 175/200 -> 87.5
    r = _parse("|turn|1\n|-sethp|p1a: Meowscarada|175/200\n|turn|2\n")
    assert _hp(r, 2) == 87.5


def test_pain_split_both_mons():
    # Pain Split averages HP: p1a 100, p2a 20 -> both to 60. Two -sethp lines, one per mon.
    r = _parse("|turn|1\n|-damage|p2a: Garchomp|20/100\n|turn|2\n"
               "|-sethp|p2a: Garchomp|60/100|[from] move: Pain Split|[of] p1a: Meowscarada\n"
               "|-sethp|p1a: Meowscarada|60/100|[from] move: Pain Split|[of] p1a: Meowscarada\n|turn|3\n")
    assert _hp(r, 3, "our_a") == 60.0
    assert _hp(r, 3, "opp_a", "opp_active") == 60.0


def test_sethp_truncated_no_crash():
    r = _parse("|turn|1\n|-sethp|p1a: Meowscarada\n|turn|2\n")   # no hp arg
    assert _hp(r, 2) == 100.0   # unchanged, no exception
