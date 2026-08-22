"""report_text (2026-07-10): the benchmark report as a string — shared by the CLI and the
online session logger (artifacts/logs/online_<session>.log gets it appended on exit)."""
from v_dance.eval.human_benchmark_report import report_text


def _row(i, result, sess="s1", note="online_adv"):
    return {"session_id": sess, "note": note, "game_idx": i, "result": result,
            "ai_team": "maw_zard", "opponent": f"opp{i}", "turns": 8,
            "battle_tag": f"battle-x-{i}"}


def test_report_text_contains_all_sections():
    rows = [_row(1, "ai"), _row(2, "human"), _row(3, "ai")]
    t = report_text(rows)
    assert "== human benchmark" in t
    assert "games                : 3  (AI 2 / human 1 / draw 0)" in t
    assert "AI win% (decisive)   : 66.7%" in t
    assert "-- sessions (sets) --" in t and "maw_zard" in t
    assert "exploitability curve" in t
    assert "game  1: 100.0% AI" in t


def test_report_text_min_games_filters_curve():
    rows = [_row(1, "ai", sess="a"), _row(1, "human", sess="b"), _row(2, "ai", sess="a")]
    t = report_text(rows, min_games=2)
    assert "game  1:" in t and "game  2:" not in t     # idx 2 has only 1 pooled game
