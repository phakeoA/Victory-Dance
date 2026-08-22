"""Unit tests for game_runner._subdivide_matchups (task #13 worker-saturation split).

Without it, team_matchups(["team1"], N) returns ONE (team,team,N) chunk → the whole --search ab
run is a single sequential pairing no matter how high --workers is. The split turns that one chunk
into up-to-n_workers independent sub-chunks (each given a distinct uid → distinct accounts/seeds in
run_self_play_games) that still sum to N and aggregate into the identical A/B verdict.
"""
from __future__ import annotations

from collections import Counter

from v_dance.selfplay.game_runner import _subdivide_matchups


def _sum(chunks):
    return sum(n for _, _, n in chunks)


def _per_pair(chunks):
    c: Counter = Counter()
    for a, b, n in chunks:
        c[(a, b)] += n
    return c


def test_single_team_splits_to_workers_evenly():
    raw = [("team1", "team1", 2000)]
    out = _subdivide_matchups(raw, 10)
    assert len(out) == 10
    assert all(a == "team1" and b == "team1" for a, b, _ in out)
    assert _sum(out) == 2000
    assert out == [("team1", "team1", 200)] * 10          # 2000/10 divides evenly


def test_workers_one_is_no_op():
    raw = [("team1", "team1", 2000)]
    assert _subdivide_matchups(raw, 1) == raw             # default sequential run unchanged


def test_multiteam_already_enough_is_unchanged():
    raw = [("a", "b", 50), ("b", "a", 50), ("a", "c", 50), ("c", "a", 50)]
    assert _subdivide_matchups(raw, 4) == raw             # target == len -> unchanged
    assert _subdivide_matchups(raw, 3) == raw             # target < len  -> unchanged


def test_multiteam_oversplit_preserves_totals_and_per_pair():
    raw = [("a", "b", 50), ("b", "a", 50), ("a", "c", 30), ("c", "a", 30)]
    out = _subdivide_matchups(raw, 10)
    assert len(out) == 10
    assert _sum(out) == 160
    assert _per_pair(out) == _per_pair(raw)               # every matchup's game total preserved


def test_cannot_exceed_game_count():
    raw = [("team1", "team1", 3)]
    out = _subdivide_matchups(raw, 10)                    # only 3 games -> at most 3 sub-chunks
    assert len(out) == 3
    assert _sum(out) == 3
    assert all(n == 1 for _, _, n in out)


def test_uneven_split_sums_exactly_and_is_near_equal():
    raw = [("team1", "team1", 2003)]
    out = _subdivide_matchups(raw, 10)
    assert len(out) == 10
    assert _sum(out) == 2003                              # remainder distributed, sum exact
    sizes = sorted(n for _, _, n in out)
    assert sizes[-1] - sizes[0] <= 1                      # near-equal chunks


def test_zero_game_chunks_dropped():
    raw = [("a", "b", 0), ("b", "a", 4)]
    out = _subdivide_matchups(raw, 4)
    assert _sum(out) == 4
    assert all(n > 0 for _, _, n in out)
    assert _per_pair(out) == Counter({("b", "a"): 4})
