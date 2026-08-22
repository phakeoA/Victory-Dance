"""v11 P3 — boost-manipulation parser messages.

The offline parser handled only |-boost|/|-unboost|; poke-env ALSO applies 8 more boost-mutation messages
that the parser silently dropped, desyncing the 7 boost channels (and derived effective-speed) from poke-env
every time one fired. P3 adds parser handlers for -clearboost / -clearallboost / -clearnegativeboost /
-clearpositiveboost / -setboost / -copyboost / -swapboost / -invertboost, each mirroring the matching poke-env
Pokemon/Battle method in VALUE, plus a clamp fix on the existing -boost/-unboost (poke-env clamps to ±6).

Parser-only: the LIVE encoder reads poke-env's native mon.boosts (already correct) → NO live/encoder change.
Verified vs poke-env abstract_battle.py:898-911/1030-1061 + pokemon.py boost/clear*/set_boost/copy_boosts/
invert_boosts (clamp at :184-187; copyboost direction target<-source at :911; swap split ', ' at :1057).
VALUE-ONLY: STATE_DIM unchanged. PARSER change → rides the end-of-v11 re-export.
"""
from __future__ import annotations

from v_dance.parser.vod_parser.replay_parser import ShowdownReplayParser

HEADER = """|player|p1|A|1|
|player|p2|B|1|
|poke|p1|Meowscarada, L50, M|
|poke|p1|Klefki, L50, M|
|poke|p2|Garchomp, L50, M|
|poke|p2|Dragonite, L50, M|
|teamsize|p1|2
|teamsize|p2|2
|start
|switch|p1a: Meowscarada|Meowscarada, L50, M|100/100
|switch|p1b: Klefki|Klefki, L50, M|100/100
|switch|p2a: Garchomp|Garchomp, L50, M|100/100
|switch|p2b: Dragonite|Dragonite, L50, M|100/100
"""


def _parse(body):
    return ShowdownReplayParser(HEADER + body + "|turn|999\n", our_player="p1").parse()


def _boosts(result, n, key="our_a", side="our_active"):
    for t in result["turns"]:
        if t["turn"] == n:
            return t["state_before_actions"]["p1"][side][key]["boosts"]
    raise AssertionError(f"turn {n} not found")


def _g(b, stat):
    return b.get(stat, 0)


# ════════════════════════════ clear family ════════════════════════════
def test_clearboost_resets_all():
    r = _parse("|turn|1\n|-boost|p1a: Meowscarada|atk|2\n|-boost|p1a: Meowscarada|spe|1\n"
               "|-unboost|p1a: Meowscarada|def|1\n|turn|2\n|-clearboost|p1a: Meowscarada\n|turn|3\n")
    assert _g(_boosts(r, 2), "atk") == 2
    b3 = _boosts(r, 3)
    assert all(_g(b3, s) == 0 for s in ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion"))


def test_clearallboost_both_sides():
    r = _parse("|turn|1\n|-boost|p1a: Meowscarada|atk|2\n|-unboost|p2a: Garchomp|spe|1\n"
               "|turn|2\n|-clearallboost\n|turn|3\n")
    assert _g(_boosts(r, 3, "our_a"), "atk") == 0          # my side cleared
    assert _g(_boosts(r, 3, "opp_a", "opp_active"), "spe") == 0   # opp side cleared too


def test_clearnegativeboost_only_negatives():
    r = _parse("|turn|1\n|-boost|p1a: Meowscarada|atk|3\n|-unboost|p1a: Meowscarada|def|2\n"
               "|turn|2\n|-clearnegativeboost|p1a: Meowscarada|[silent]\n|turn|3\n")
    b3 = _boosts(r, 3)
    assert _g(b3, "atk") == 3 and _g(b3, "def") == 0


def test_clearpositiveboost_only_positives():
    r = _parse("|turn|1\n|-boost|p1a: Meowscarada|atk|3\n|-unboost|p1a: Meowscarada|def|2\n"
               "|turn|2\n|-clearpositiveboost|p1a: Meowscarada\n|turn|3\n")
    b3 = _boosts(r, 3)
    assert _g(b3, "atk") == 0 and _g(b3, "def") == -2


# ════════════════════════════ set (absolute) ════════════════════════════
def test_setboost_anger_point_absolute():
    # from +2, Anger Point SETS atk to +6 (absolute), not +8 additive
    r = _parse("|turn|1\n|-boost|p1a: Meowscarada|atk|2\n|turn|2\n"
               "|-setboost|p1a: Meowscarada|atk|6|[from] ability: Anger Point\n|turn|3\n")
    assert _g(_boosts(r, 3), "atk") == 6


def test_setboost_clamps_defensively():
    r = _parse("|turn|1\n|-setboost|p1a: Meowscarada|atk|7\n|turn|2\n")
    assert _g(_boosts(r, 2), "atk") == 6


# ════════════════════════════ copy (one-way) ════════════════════════════
def test_copyboost_target_receives_source():
    # Psych Up: |-copyboost|SOURCE|TARGET → TARGET (p1a) copies SOURCE (p1b); SOURCE unchanged
    r = _parse("|turn|1\n|-boost|p1b: Klefki|atk|2\n|-boost|p1b: Klefki|spa|1\n"
               "|-unboost|p1a: Meowscarada|atk|3\n|turn|2\n"
               "|-copyboost|p1b: Klefki|p1a: Meowscarada|[from] move: Psych Up\n|turn|3\n")
    a = _boosts(r, 3, "our_a")    # target Meowscarada
    b = _boosts(r, 3, "our_b")    # source Klefki (unchanged)
    assert _g(a, "atk") == 2 and _g(a, "spa") == 1
    assert _g(b, "atk") == 2 and _g(b, "spa") == 1


# ════════════════════════════ swap (symmetric) ════════════════════════════
def test_swapboost_all_via_from():
    # Heart Swap sends no explicit stat list (a [from] marker) → swap all 7
    r = _parse("|turn|1\n|-boost|p1a: Meowscarada|atk|2\n|-unboost|p1b: Klefki|atk|1\n"
               "|-boost|p1b: Klefki|spe|3\n|turn|2\n"
               "|-swapboost|p1a: Meowscarada|p1b: Klefki|[from] move: Heart Swap\n|turn|3\n")
    a = _boosts(r, 3, "our_a")
    b = _boosts(r, 3, "our_b")
    assert _g(a, "atk") == -1 and _g(a, "spe") == 3       # got Klefki's
    assert _g(b, "atk") == 2 and _g(b, "spe") == 0        # got Meowscarada's


def test_swapboost_named_subset():
    # Guard Swap: only def & spd swapped (', '-separated); atk untouched
    r = _parse("|turn|1\n|-boost|p1a: Meowscarada|def|1\n|-boost|p1a: Meowscarada|atk|2\n"
               "|-unboost|p1b: Klefki|def|2\n|turn|2\n"
               "|-swapboost|p1a: Meowscarada|p1b: Klefki|def, spd|[from] move: Guard Swap\n|turn|3\n")
    a = _boosts(r, 3, "our_a")
    b = _boosts(r, 3, "our_b")
    assert _g(a, "def") == -2 and _g(b, "def") == 1        # def swapped
    assert _g(a, "atk") == 2 and _g(b, "atk") == 0         # atk NOT swapped


# ════════════════════════════ invert ════════════════════════════
def test_invertboost():
    r = _parse("|turn|1\n|-boost|p1a: Meowscarada|atk|3\n|-unboost|p1a: Meowscarada|def|2\n"
               "|turn|2\n|-invertboost|p1a: Meowscarada\n|turn|3\n")
    b3 = _boosts(r, 3)
    assert _g(b3, "atk") == -3 and _g(b3, "def") == 2


# ════════════════════════════ existing -boost clamp fix ════════════════════════════
def test_existing_boost_now_clamps():
    r = _parse("|turn|1\n|-boost|p1a: Meowscarada|atk|3\n|-boost|p1a: Meowscarada|atk|3\n"
               "|-boost|p1a: Meowscarada|atk|3\n|turn|2\n")
    assert _g(_boosts(r, 2), "atk") == 6   # 9 clamped to 6


def test_unboost_clamps_floor():
    r = _parse("|turn|1\n|-unboost|p1a: Meowscarada|spe|3\n|-unboost|p1a: Meowscarada|spe|3\n"
               "|-unboost|p1a: Meowscarada|spe|3\n|turn|2\n")
    assert _g(_boosts(r, 2), "spe") == -6


# ════════════════════════════ robustness ════════════════════════════
def test_unresolvable_ident_no_crash():
    # a copy/swap referencing a switched-out / garbage ident must no-op, not raise
    r = _parse("|turn|1\n|-boost|p1a: Meowscarada|atk|2\n"
               "|-copyboost|p2z: Ghost|p1a: Meowscarada\n"
               "|-swapboost|p1a: Meowscarada|p2z: Ghost|[from] move: Heart Swap\n|turn|2\n")
    assert _g(_boosts(r, 2), "atk") == 2   # unchanged, no exception
