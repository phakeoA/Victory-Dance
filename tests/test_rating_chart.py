"""2026-10-01: the ladder rating chart (v_dance/ui/rating_chart.py) that Mission Control's Rating tab serves."""
import json
import os

from v_dance.ui import rating_chart as RC


def _bench(path, n=120):
    rows = []
    r = 1200
    for i in range(n):
        tag = f"battle-gen9championsvgc2026regmc-{1000 + i}"
        d = 12 if i % 2 else -10
        rows.append({"battle_tag": tag, "result": "ai", "ai_team": "Baltimore_Sand_Psy" if i < 60 else "The_Big_6_v2"})
        rows.append({"type": "rating_update", "battle_tag": tag, "rating": r, "rating_after": r + d, "ts": f"t{i:04d}"})
        r += d
    rows.append({"type": "site_rating", "format": "gen9championsvgc2026regmc", "elo": 1250, "w": 60, "l": 60,
                 "ts": "t0100"})
    path.write_text("\n".join(json.dumps(x) for x in rows) + "\n{half a line", encoding="utf-8")


def test_page_has_one_chart_per_format_team_ticks_and_the_site_truth(tmp_path):
    b = tmp_path / "bench.jsonl"
    _bench(b)
    page, counts = RC.build_page(b, window=100, theme="dark")
    assert counts == {"gen9championsvgc2026regmc": 120}
    assert page.count("<svg") == 1 and "SITE (truth) 1250, 60W-60L" in page
    assert "Baltimore_Sand_Psy" in page and "The_Big_6_v2" in page and "#1b2733" in page   # MC's dark palette


def test_missing_log_renders_a_notice(tmp_path):
    page, counts = RC.build_page(tmp_path / "nope.jsonl")
    assert counts == {} and "No rated games logged yet" in page


def test_cached_page_reparses_only_when_the_log_changes(tmp_path, monkeypatch):
    b = tmp_path / "bench.jsonl"
    _bench(b)
    calls = []
    real = RC.build_page
    monkeypatch.setattr(RC, "build_page", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    RC._CACHE.clear()
    RC.cached_page(b, 100, "dark")
    RC.cached_page(b, 100, "dark")
    assert len(calls) == 1
    with open(b, "a", encoding="utf-8") as f:
        f.write("\n")
    os.utime(b, ns=(os.stat(b).st_atime_ns, os.stat(b).st_mtime_ns + 10**9))
    RC.cached_page(b, 100, "dark")
    assert len(calls) == 2
