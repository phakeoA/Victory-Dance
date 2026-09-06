"""2026-09-06 — the Round / Instruct ``[from] move:`` guard (USER: "Round strategy, although rare, is still used").

Showdown tags a chained Round as ``|move|p1b: Sylveon|Round|p2a: Floette|[from] move: Round``. poke-env 0.16.1 treats
any ``[from] move: X`` as a Sleep-Talk-style override and indexes the mover's known moves — Round names itself and
Instruct names another mon's move, so for an opponent that is a KeyError, the ps_client swallows it and the rest of
the frame (damage, faints) is lost. ``guard_from_move_tags`` strips the tag when the mover does not know X, so the
line parses as a plain move; real overrides are untouched. The frame below is the one from the 2026-09-05 log."""
from __future__ import annotations

import logging

import pytest
from poke_env.battle import DoubleBattle

from v_dance.play.live_vgc_base import guard_from_move_tags

TAG = "battle-gen9championsvgc2026regmb-1"


def _battle() -> DoubleBattle:
    b = DoubleBattle(TAG, "VictoriousDancing", logging.getLogger("t"), gen=9)
    for line in ("|player|p1|Rival|169|1262", "|player|p2|VictoriousDancing|102|1347",
                 "|switch|p1a: Farigiraf|Farigiraf, L50, M|100/100", "|switch|p1b: Sylveon|Sylveon, L50, F|100/100",
                 "|switch|p2a: Floette|Floette, L50, F|100/100", "|switch|p2b: Whimsicott|Whimsicott, L50, F|135/135"):
        b.parse_message(line.split("|"))
    return b


FRAME = [[">" + TAG],
         "|move|p1a: Farigiraf|Round|p2b: Whimsicott".split("|"),
         "|-damage|p2b: Whimsicott|63/135".split("|"),
         "|move|p1b: Sylveon|Round|p2a: Floette|[from] move: Round".split("|"),
         "|-damage|p2a: Floette|0 fnt".split("|"),
         "|faint|p2a: Floette".split("|")]


def test_the_chained_round_line_crashes_bare_poke_env_and_parses_with_the_guard():
    bare = _battle()
    bare.parse_message(FRAME[1])
    with pytest.raises(KeyError):                                   # the 2026-09-05 traceback, reproduced
        bare.parse_message(FRAME[3])
    b = _battle()
    guarded = guard_from_move_tags(b, FRAME)
    assert guarded is not FRAME and guarded[1] == FRAME[1] and guarded[3] == FRAME[3][:-1]   # only the tag went
    for m in guarded[1:]:
        b.parse_message(m)
    sylveon = b.get_pokemon("p1b: Sylveon")
    assert "round" in sylveon.moves and sylveon.moves["round"].current_pp == sylveon.moves["round"].max_pp - 1
    assert "round" in b.get_pokemon("p1a: Farigiraf").moves
    assert b.get_pokemon("p2a: Floette").fainted is True                # the rest of the frame was parsed
    assert b.get_pokemon("p2b: Whimsicott").current_hp == 63


def test_real_overrides_and_frames_without_tags_are_untouched_and_instruct_is_stripped():
    b = _battle()
    sleep = ["|move|p1a: Farigiraf|Sleep Talk|p1a: Farigiraf".split("|"),
             "|move|p1a: Farigiraf|Tackle|p2a: Floette|[from] move: Sleep Talk".split("|")]
    b.parse_message(sleep[0])                                           # poke-env pre-deduces Sleep Talk into the moveset
    msgs = [[">" + TAG], sleep[1]]
    assert guard_from_move_tags(b, msgs) is msgs                        # known override: the SAME list back
    b.parse_message(sleep[1])                                           # and poke-env's own override path still works
    assert "sleeptalk" in b.get_pokemon("p1a: Farigiraf").moves
    plain = [[">" + TAG], "|move|p1b: Sylveon|Hyper Voice|p2a: Floette|[spread] p2a,p2b".split("|"), "|turn|3".split("|")]
    assert guard_from_move_tags(b, plain) is plain                      # no [from] tag: nothing to do
    instruct = [[">" + TAG], "|move|p1b: Sylveon|Hyper Voice|p2a: Floette|[from] move: Instruct".split("|")]
    out = guard_from_move_tags(b, instruct)
    assert out[1] == instruct[1][:-1]                                   # Sylveon does not know Instruct → stripped
    b.parse_message(out[1])
    assert "hypervoice" in b.get_pokemon("p1b: Sylveon").moves
    # the guard never raises: a battle that cannot resolve the mover leaves the line to poke-env
    class Odd:
        def get_pokemon(self, ident):
            raise ValueError("no such mon")
    assert guard_from_move_tags(Odd(), instruct) is instruct
    assert guard_from_move_tags(None, instruct) is instruct and guard_from_move_tags(b, []) == []
