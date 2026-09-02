"""Peak elo per regulation in the control panel / Mission Control (USER 2026-09-01).

Before this, both UIs only ever showed the LAST rating — a peak field never existed (git-verified).
RatingBook keys everything by the battle's FORMAT (ratings are per ladder), seeds all-time peaks
from the bench JSONL, tracks session start/peak from the live rating lines, and reports sorted by
regulation. Also locks the host's forget/abandon split that the reconnect relies on."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

pytest.importorskip("poke_env")

from v_dance.play.bot_control_ui import BotController, RatingBook, _reg_label

MB = "gen9championsvgc2026regmb"
MA = "gen9championsvgc2026regma"


def _tag(fmt, n):
    return f"battle-{fmt}-{n}"


def test_reg_label():
    assert _reg_label(MB) == "M-B"
    assert _reg_label(MA) == "M-A"
    assert _reg_label("gen9championsvgc2026regmc") == "M-C"
    assert _reg_label("gen9championsvgc2026regmbbo3") == "M-B"   # Bo3 variant shares the reg


def test_fmt_of_parses_bare_and_private_suffixed_tags():
    assert RatingBook.fmt_of(_tag(MB, 1)) == MB
    assert RatingBook.fmt_of(_tag(MB, 1) + "-6x833abcpw") == MB
    assert RatingBook.fmt_of("not-a-battle") is None
    assert RatingBook.fmt_of(None) is None


def test_all_time_peaks_seed_from_bench_jsonl_per_format(tmp_path: Path):
    rows = [
        # online rows: game row without rating + rating_update rows carrying OUR rating
        {"battle_tag": _tag(MB, 1), "result": "ai", "rating": None},
        {"type": "rating_update", "battle_tag": _tag(MB, 1), "rating": 1300},
        {"type": "rating_update", "battle_tag": _tag(MB, 1), "opponent_rating": 1500},  # theirs: ignored
        {"battle_tag": _tag(MB, 2), "result": "human", "rating": None},
        {"type": "rating_update", "battle_tag": _tag(MB, 2), "rating": 1437},
        # play_ladder-style rows: rating on the game row itself, other regulation
        {"battle_tag": _tag(MA, 9), "result": "ai", "rating": 1120},
        {"battle_tag": _tag(MA, 10), "result": "ai", "rating": 1090},
        # junk that must be skipped
        {"battle_tag": None, "rating": 9999},
        {"battle_tag": _tag(MB, 3), "rating": True},
    ]
    p = tmp_path / "bench.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n", encoding="utf-8")
    rb = RatingBook(p)
    s = {d["format"]: d for d in rb.summary()}
    assert s[MB]["all_time_peak"] == 1437 and s[MB]["all_time_games"] == 3
    assert s[MA]["all_time_peak"] == 1120 and s[MA]["all_time_games"] == 2
    assert s[MB]["session_peak"] is None                  # nothing live yet
    assert [d["reg"] for d in rb.summary()] == ["M-A", "M-B"]   # sorted by regulation


def test_record_tracks_session_start_peak_and_lifts_all_time():
    rb = RatingBook()
    rb.record(_tag(MB, 1), 1100)
    rb.record(_tag(MB, 2), 1376)
    rb.record(_tag(MB, 3), 1262)
    rb.record(_tag(MB, 4), "not a number")                # ignored
    (d,) = rb.summary()
    assert d["session_start"] == 1100 and d["session_peak"] == 1376 and d["current"] == 1262
    assert d["all_time_peak"] == 1376                     # no seed → the session IS the record


def test_missing_or_broken_bench_file_is_harmless(tmp_path: Path):
    assert RatingBook(tmp_path / "nope.jsonl").summary() == []
    bad = tmp_path / "bad.jsonl"
    bad.write_bytes(b"\xff\xfe garbage")
    assert RatingBook(bad).summary() == []


def test_panel_status_carries_peaks_and_link():
    class _Link:
        def status(self):
            return {"enabled": True, "down": False, "idle_s": 2.0, "reconnects": 1,
                    "probes": 0, "frames": 10}

    async def main():
        c = BotController(page=None, host=None, tally={"ai": 0, "you": 0, "draw": 0}, ai_pool=[],
                          fmt=MB, username="VictoriousDancing", loop=asyncio.get_running_loop(),
                          env_path=Path("unused.env"))
        c.ratings.record(_tag(MB, 1), 1300)
        c.link = _Link()
        return c.status()

    s = asyncio.run(main())
    assert s["peaks"][0]["format"] == MB and s["peaks"][0]["session_peak"] == 1300
    assert s["link"]["reconnects"] == 1
    # the default (no online watchdog) is an explicit null, so the UIs render "—"
    assert BotController(page=None, host=None, tally={}, ai_pool=[], fmt=MB, username="x",
                         loop=None, env_path=Path("u")).status()["link"] is None


# ── host: forget (rejoin) vs abandon (room gone) ─────────────────────────────
def test_host_forget_battle_drops_state_without_ending_and_a_replay_rebuilds_it():
    from v_dance.formats import DEFAULT_FORMAT
    from v_dance.play.browser.battle_host import BattleHost

    tag = f"battle-{DEFAULT_FORMAT}-1"
    init = (f">{tag}\n|init|battle\n|title|Alice vs. Bob\n|gametype|doubles\n"
            "|player|p1|Alice|1|\n|player|p2|Bob|2|\n|turn|1")
    h = BattleHost(team=None, model_path=None, team_chooser_path=None)
    h.feed(init)
    assert tag in h.battles and h.player._proto_log.get(tag)
    h.forget_battle(tag)
    assert tag not in h.battles and tag not in h._ended      # forgotten, NOT ended
    assert tag not in h.player._proto_log                    # splice buffer starts empty again
    produced = h.feed(init)                                  # the server's replayed log → fresh object
    assert tag in h.battles and h.player._proto_log.get(tag)
    assert (tag, "/rejectopenteamsheets") in produced        # poke-env re-answers OTS (consumer drops it)


def test_host_abandon_battle_marks_ended_so_stray_frames_are_gated():
    from v_dance.formats import DEFAULT_FORMAT
    from v_dance.play.browser.battle_host import BattleHost

    tag = f"battle-{DEFAULT_FORMAT}-2"
    init = (f">{tag}\n|init|battle\n|title|A vs. B\n|gametype|doubles\n"
            "|player|p1|A|1|\n|player|p2|B|2|\n|turn|1")
    h = BattleHost(team=None, model_path=None, team_chooser_path=None)
    h.feed(init)
    h.abandon_battle(tag)
    assert tag not in h.battles and tag in h._ended
    assert h.feed(f">{tag}\n|noinit|nonexistent|gone") == []   # gated: never reaches poke-env
