"""B5 (2026-09-29): data/vods/Type_C mixes regulations, so a per-reg export filters replays by format."""
from pathlib import Path

from v_dance.datatools.bulk_parse_replays import filter_format

_NAMES = [
    "20260710-085624-5976_online_battle-gen9championsvgc2026regmb-2646960772.html",
    "20260929-080543-15680_online_battle-gen9championsvgc2026regmc-2689686840.html",
    "20260801-0000-1_online_battle-gen9championsvgc2026regmbbo3-2650000000.html",
    "Gen9ChampionsVGC2026RegMC-2026-09-24-peermol-onthewavenow.html",
]


def test_filter_keeps_only_the_requested_format():
    files = [Path(n) for n in _NAMES]
    assert [p.name for p in filter_format(files, "gen9championsvgc2026regmc")] == [_NAMES[1], _NAMES[3]]
    # regmb must NOT also pick up the Bo3 format whose id it prefixes
    assert [p.name for p in filter_format(files, "gen9championsvgc2026regmb")] == [_NAMES[0]]
    assert filter_format(files, None) == files
