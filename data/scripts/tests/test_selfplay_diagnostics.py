"""Task 3a.5: per-bring / per-matchup win-rate diagnostics."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
for _p in (str(_REPO / "local_battle"), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from self_play.schema import EpisodeMeta  # noqa: E402
from self_play.diagnostics import (  # noqa: E402
    Tally, bring_winrates, lead_winrates, matchup_winrates, overall, summarize,
)

OWN = ["a", "b", "c", "d", "e", "f"]


def _m(bring, won, own=OWN, opp=None):
    tt = {True: "win", False: "loss", None: "draw"}[won]
    return EpisodeMeta(battle_id="g", own_role="p1", own_team=list(own),
                       opp_team=list(opp or ["x"] * 6), tp_bring=list(bring),
                       tp_leads=list(bring[:2]), won=won, terminal_type=tt, n_turns=1)


def test_tally_win_rate_over_decisive_only():
    t = Tally()
    for w in (True, True, False, None):     # 2 win, 1 loss, 1 draw
        t.add(w)
    assert (t.games, t.wins, t.losses, t.nondecisive) == (4, 2, 1, 1)
    assert t.decisive == 3
    assert t.win_rate == pytest.approx(2 / 3)


def test_bring_winrates_group_and_rate():
    batch = [_m([0, 1, 2, 3], True), _m([0, 1, 2, 3], True), _m([0, 1, 2, 3], False),
             _m([2, 3, 4, 5], False), _m([2, 3, 4, 5], None)]
    wr = bring_winrates(batch)
    assert wr[("a", "b", "c", "d")].win_rate == pytest.approx(2 / 3)   # 2W/1L
    assert wr[("c", "d", "e", "f")].win_rate == 0.0                    # 0W/1L (+1 draw)
    assert wr[("c", "d", "e", "f")].games == 2


def test_bring_key_is_order_independent():
    wr = bring_winrates([_m([3, 2, 1, 0], True), _m([0, 1, 2, 3], False)])
    assert len(wr) == 1 and wr[("a", "b", "c", "d")].decisive == 2     # same bring set


def test_out_of_range_bring_index_excluded_not_crash():
    wr = bring_winrates([_m([0, 1, 2, 99], True)])                     # 99 >= len(own)=6
    assert ("a", "b", "c") in wr                                       # 99 dropped


def test_matchup_winrates_keyed_by_both_rosters():
    opp2 = ["p", "q", "r", "s", "t", "u"]
    wr = matchup_winrates([_m([0, 1, 2, 3], True, opp=opp2),
                           _m([0, 1, 2, 3], False, opp=opp2)])
    key = (tuple(sorted(OWN)), tuple(sorted(opp2)))
    assert wr[key].decisive == 2 and wr[key].win_rate == 0.5


def test_overall_is_half_for_mirrored_outcomes():
    assert overall([_m([0, 1, 2, 3], True), _m([0, 1, 2, 3], False)]).win_rate == 0.5


def test_accepts_episode_meta_directly_and_summarize():
    batch = ([_m([0, 1, 2, 3], True)] * 6 + [_m([0, 1, 2, 3], False)] * 2
             + [_m([2, 3, 4, 5], False)] * 6)
    s = summarize(batch, min_games=5)
    assert s["overall"]["games"] == 14
    assert s["n_brings"] == 2
    # worst bring (cdef, all losses) ranks first; best (abcd, 6W/2L) last
    assert s["worst_brings"][0]["bring"] == ["c", "d", "e", "f"]
    assert s["best_brings"][0]["bring"] == ["a", "b", "c", "d"]
