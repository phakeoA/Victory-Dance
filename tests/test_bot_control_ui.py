"""Control panel (2026-07-10, USER request) — bot_control_ui.

The panel INDIRECTLY drives the online bot: ladder runs (N rated games, /utm+/search exactly like
the client's Battle! button), private challenges by username, an auto-accept toggle
(play_vs_human_browser.AUTO_ACCEPT), a team pin the consumer's pick honours, and a format choice
persisted to .env (applies next launch — the stack binds format at import). These tests drive the
controller directly with fakes (no Playwright, no network) + one real-HTTP smoke test.
"""
from __future__ import annotations

import asyncio
import json
import urllib.request
from pathlib import Path

import pytest

import v_dance.play.play_vs_human_browser as _pvhb
from v_dance.play import bot_control_ui as bcu
from v_dance.play.bot_control_ui import BotController, start_control_ui

FMT = "gen9championsvgc2026regmb"
POOL = ["alpha_team", "beta_team"]


class FakePage:
    def __init__(self):
        self.sent = []                     # (js, arg) of every app.socket.send injection

    async def evaluate(self, js, arg=None):
        self.sent.append((js, arg))
        return None


class _FakeTeam:
    def yield_team(self):
        return "PACKED_TEAM"


class FakePlayer:
    def __init__(self):
        self._team = _FakeTeam()
        self._team_name = None
        self.updated = []

    def update_team(self, team):
        self.updated.append(team)


class FakeHost:
    def __init__(self):
        self.player = FakePlayer()


def _ctrl(loop, page=None, **kw):
    c = BotController(page=page or FakePage(), host=FakeHost(), tally={"ai": 0, "you": 0, "draw": 0},
                      ai_pool=POOL, fmt=FMT, username="VictoriousDancing", loop=loop,
                      env_path=kw.pop("env_path", Path("unused.env")), **kw)
    c._load_scoped_team = lambda scoped: f"LOADED:{scoped}"   # no filesystem in tests
    return c


@pytest.fixture(autouse=True)
def _no_team_fs(monkeypatch):
    monkeypatch.setattr(bcu, "discover_teams", lambda reg=None: [])


def _sent_cmds(page):
    """The page injections flattened to searchable strings."""
    return [f"{js}||{arg}" for js, arg in page.sent]


def _rate(c, tag):
    """Feed the two |raw| rating lines (us + them) → Δelo exchanged → battle confirmed done."""
    c._on_rating(tag, "VictoriousDancing", 1200)
    c._on_rating(tag, "SomeOpponent", 1210)


# ── ladder run ────────────────────────────────────────────────────────────────
def test_start_ladder_sends_utm_then_search_with_pinned_team():
    async def main():
        page = FakePage()
        c = _ctrl(asyncio.get_running_loop(), page)
        await c.start_ladder(3, "alpha_team")
        return c, page

    c, page = asyncio.run(main())
    cmds = _sent_cmds(page)
    assert any("/utm" in s and "PACKED_TEAM" in s for s in cmds)
    assert any("/search" in s and FMT in s for s in cmds)
    assert cmds.index(next(s for s in cmds if "/utm" in s)) \
        < cmds.index(next(s for s in cmds if "/search" in s))   # team uploaded BEFORE queueing
    assert c.run_active and c.run_target == 3 and c.run_done == 0
    assert c._search_outstanding
    assert c.host.player._team_name == "alpha_team"             # decision core bound to the pin


def test_run_counts_wins_requeues_and_completes():
    async def main():
        page = FakePage()
        c = _ctrl(asyncio.get_running_loop(), page)
        await c.start_ladder(2, "alpha_team")
        # game 1 starts (clears the outstanding search) and ends
        c.tap_frame(f">battle-{FMT}-111\n|init|battle")
        assert not c._search_outstanding and f"battle-{FMT}-111" in c._live
        c.tap_frame(f">battle-{FMT}-111\n|win|SomeOpponent")
        assert c.run_done == 1 and c.run_active
        # |win| alone is NOT done: the tick must defer until the rating exchange confirms
        c._resume_scheduled = False
        c._resume_tick()
        await asyncio.sleep(0.05)
        assert sum("/search" in s for s in _sent_cmds(page)) == 1
        _rate(c, f"battle-{FMT}-111")                           # Δelo exchanged → confirmed
        c._resume_scheduled = False
        c._resume_tick()
        await asyncio.sleep(0.05)                               # let the created task run
        assert sum("/search" in s for s in _sent_cmds(page)) == 2
        # a re-delivered |win| for the same room must NOT double-count (rejoin re-send)
        c.tap_frame(f">battle-{FMT}-111\n|win|SomeOpponent")
        assert c.run_done == 1
        # game 2 ends → target reached → run stops
        c.tap_frame(f">battle-{FMT}-222\n|init|battle")
        c.tap_frame(f">battle-{FMT}-222\n|win|VictoriousDancing")
        assert c.run_done == 2 and not c.run_active
        return c, page

    asyncio.run(main())


def test_regression_updatesearch_first_does_not_stall_the_run():
    """2026-07-10 live stall: for a FOUND ladder game the server's updatesearch (games list) arrives
    BEFORE the battle room's own frames. The updatesearch-side add used to leave _search_outstanding
    latched True → the +4s resume silently bailed → the run died after game 1. Now both sighting
    paths clear the flag and the single-flight resume keeps re-checking until search 2 goes out."""
    async def main():
        page = FakePage()
        c = _ctrl(asyncio.get_running_loop(), page)
        await c.start_ladder(2, "alpha_team")
        assert c._search_outstanding
        tag = f"battle-{FMT}-777"
        # 1) the server acks the found game via updatesearch FIRST (the real ordering)
        c.tap_frame(f'|updatesearch|{json.dumps({"searching": [], "games": {tag: "title"}})}')
        assert not c._search_outstanding            # ← the latched flag was the bug
        assert tag in c._live
        # 2) then the battle frames flow and the game ends
        c.tap_frame(f">{tag}\n|init|battle")
        c.tap_frame(f">{tag}\n|win|SomeOpponent")
        assert c.run_done == 1 and c.run_active
        assert c._resume_scheduled                   # exactly one resume tick pending
        _rate(c, tag)                                # Δelo exchanged (gating tested elsewhere)
        # 3) fire the tick — the second search must go out
        c._resume_scheduled = False
        c._resume_tick()
        await asyncio.sleep(0.05)
        assert sum("/search" in s for s in _sent_cmds(page)) == 2

    asyncio.run(main())


def test_base_tag_normalises_private_suffix():
    from v_dance.play.bot_control_ui import _base_tag
    assert _base_tag(f"battle-{FMT}-2647021200") == f"battle-{FMT}-2647021200"
    assert _base_tag(f"battle-{FMT}-2647021200-6x833vkm5lopvmjyy3emxtquprxaxv0pw") \
        == f"battle-{FMT}-2647021200"
    assert _base_tag("not-a-battle-room") == "not-a-battle-room"   # unknown shape passes through


def test_regression_private_suffix_alias_does_not_stall_the_run():
    """2026-07-10 live stall #2 (game 4 of the first 10-run): updatesearch announced the found game
    under the BARE roomid while the room's frames carried the private-access suffix. The bare alias
    stayed in _live after the suffixed room's |win| → the resume watcher deferred forever. Both
    paths now normalise via _base_tag."""
    async def main():
        page = FakePage()
        c = _ctrl(asyncio.get_running_loop(), page)
        await c.start_ladder(10, "alpha_team")
        bare = f"battle-{FMT}-2647021200"
        suffixed = bare + "-6x833vkm5lopvmjyy3emxtquprxaxv0pw"
        # the EXACT live ordering from the log: bare via updatesearch, then suffixed frames + win
        c.tap_frame(f'|updatesearch|{json.dumps({"searching": [], "games": {bare: "t"}})}')
        c.tap_frame(f">{suffixed}\n|init|battle")
        assert len(c._live) == 1                     # ONE battle, not two aliases
        c.tap_frame(f">{suffixed}\n|win|CarlosRS9")
        assert not c._live                           # ← the ghost alias was the bug
        assert c.run_done == 1 and c.run_active
        _rate(c, suffixed)                           # rating lines arrive on the SUFFIXED id too
        c._resume_scheduled = False
        c._resume_tick()
        await asyncio.sleep(0.05)
        assert sum("/search" in s for s in _sent_cmds(page)) == 2   # run continues

    asyncio.run(main())


def test_ghost_live_sweep_unsticks_a_run():
    async def main():
        page = FakePage()
        c = _ctrl(asyncio.get_running_loop(), page)
        await c.start_ladder(5, "alpha_team")
        c._search_outstanding = False
        c._live.add(f"battle-{FMT}-ghost")           # debris with no frames flowing
        c._last_battle_frame_at = c.loop.time() - 400.0
        c._resume_scheduled = False
        c._resume_tick()
        await asyncio.sleep(0.05)
        assert not c._live                           # swept
        assert sum("/search" in s for s in _sent_cmds(page)) == 2   # and the run resumed

    asyncio.run(main())


def test_resume_tick_keeps_watching_while_busy_and_recovers_lost_search():
    async def main():
        page = FakePage()
        c = _ctrl(asyncio.get_running_loop(), page)
        await c.start_ladder(3, "alpha_team")
        c.tap_frame(f">battle-{FMT}-1\n|init|battle")
        c.tap_frame(f">battle-{FMT}-1\n|win|X")
        _rate(c, f"battle-{FMT}-1")                  # confirm game 1 (Δelo gating tested elsewhere)
        # a live battle (e.g. accepted challenge) defers the search but NEVER kills the run
        c._live.add(f"battle-{FMT}-challenge")
        c._resume_scheduled = False
        c._resume_tick()
        assert c._resume_scheduled                   # rescheduled, not dead
        assert sum("/search" in s for s in _sent_cmds(page)) == 1
        c._live.discard(f"battle-{FMT}-challenge")
        # a /search that was never acked (no updatesearch) is recovered after the timeout
        c._search_outstanding = True
        c._search_sent_at = c.loop.time() - 60.0
        c._resume_scheduled = False
        c._resume_tick()
        await asyncio.sleep(0.05)
        assert sum("/search" in s for s in _sent_cmds(page)) == 2

    asyncio.run(main())


def test_stop_ladder_cancels_search():
    async def main():
        page = FakePage()
        c = _ctrl(asyncio.get_running_loop(), page)
        await c.start_ladder(5, None)
        await c.stop_ladder()
        assert not c.run_active and not c._search_outstanding
        assert any("/cancelsearch" in s for s in _sent_cmds(page))

    asyncio.run(main())


def test_updatesearch_tracks_queue_state():
    async def main():
        c = _ctrl(asyncio.get_running_loop())
        c._search_outstanding = True
        c.tap_frame(f'|updatesearch|{json.dumps({"searching": [FMT], "games": None})}')
        assert c.searching
        assert not c._search_outstanding             # updatesearch = the authoritative ack
        c.tap_frame(f'|updatesearch|{json.dumps({"searching": [], "games": {f"battle-{FMT}-9": "t"}})}')
        assert not c.searching and f"battle-{FMT}-9" in c._live

    asyncio.run(main())


# ── Δelo battle-done confirmation + auto-close (USER, 2026-07-10 pt-9d) ──────
def test_search_waits_for_rating_exchange():
    """'Done' = someone lost elo and someone gained elo — |win| alone must not trigger the next
    search; one rating line is not enough; two distinct users' lines confirm it."""
    async def main():
        page = FakePage()
        c = _ctrl(asyncio.get_running_loop(), page)
        await c.start_ladder(3, "alpha_team")
        tag = f"battle-{FMT}-42"
        c.tap_frame(f">{tag}\n|init|battle")
        c.tap_frame(f">{tag}\n|win|SomeOpponent")
        assert tag in c._ended_unconfirmed and tag in c.status()["awaiting_confirm"]
        for _ in range(2):                            # win-only, then one-rating-only: both defer
            c._resume_scheduled = False
            c._resume_tick()
            await asyncio.sleep(0.05)
            assert sum("/search" in s for s in _sent_cmds(page)) == 1
            c._on_rating(tag, "VictoriousDancing", 1200)
        c._on_rating(tag, "SomeOpponent", 1210)       # the second user's line = Δelo exchanged
        assert tag not in c._ended_unconfirmed
        c._resume_scheduled = False
        c._resume_tick()
        await asyncio.sleep(0.05)
        assert sum("/search" in s for s in _sent_cmds(page)) == 2

    asyncio.run(main())


def test_confirm_timeout_proceeds_for_unrated_games():
    async def main():
        page = FakePage()
        c = _ctrl(asyncio.get_running_loop(), page)
        await c.start_ladder(3, "alpha_team")
        tag = f"battle-{FMT}-43"
        c.tap_frame(f">{tag}\n|win|SomeOpponent")
        c._confirm(tag, timeout=True)                 # what the +60s timer fires
        assert tag not in c._ended_unconfirmed
        c._resume_scheduled = False
        c._resume_tick()
        await asyncio.sleep(0.05)
        assert sum("/search" in s for s in _sent_cmds(page)) == 2
        c._confirm(tag, timeout=True)                 # idempotent (rating may beat the timer)

    asyncio.run(main())


def test_auto_close_leaves_the_full_suffixed_room_after_confirm():
    async def main():
        page = FakePage()
        c = _ctrl(asyncio.get_running_loop(), page)
        c.set_auto_close(True)
        bare = f"battle-{FMT}-77"
        suffixed = bare + "-s3cr3tsuffix"
        c.tap_frame(f">{suffixed}\n|init|battle")
        c.tap_frame(f">{suffixed}\n|win|SomeOpponent")
        assert not any("leaveRoom" in s for s in _sent_cmds(page))   # not before the confirm
        _rate(c, suffixed)
        await asyncio.sleep(0.05)
        leaves = [(js, arg) for js, arg in page.sent if "leaveRoom" in js]
        assert leaves and leaves[0][1] == suffixed    # closes the FULL roomid, not the base
        # toggle honoured: a second battle with auto-close OFF stays open
        c.set_auto_close(False)
        assert c.status()["auto_close"] is False
        tag2 = f"battle-{FMT}-78"
        c.tap_frame(f">{tag2}\n|win|SomeOpponent")
        _rate(c, tag2)
        await asyncio.sleep(0.05)
        assert sum(1 for js, _ in page.sent if "leaveRoom" in js) == 1

    asyncio.run(main())


# ── private challenge ─────────────────────────────────────────────────────────
def test_challenge_flow():
    async def main():
        page = FakePage()
        c = _ctrl(asyncio.get_running_loop(), page)
        await c.send_challenge("Some User", "beta_team")
        cmds = _sent_cmds(page)
        assert any("/utm" in s for s in cmds)
        assert any("/challenge" in s and f"Some User, {FMT}" in s for s in cmds)
        assert c.challenge_out == "Some User"
        await c.cancel_challenge()
        assert c.challenge_out is None
        assert any("/cancelchallenge" in s and "someuser" in s for s in _sent_cmds(page))
        with pytest.raises(ValueError):
            await c.send_challenge("   ", None)

    asyncio.run(main())


# ── options / team / format ───────────────────────────────────────────────────
def test_auto_accept_toggle_flips_consumer_flag():
    async def main():
        c = _ctrl(asyncio.get_running_loop())
        assert _pvhb.AUTO_ACCEPT is True            # module default preserves prior behaviour
        try:
            c.set_auto_accept(False)
            assert _pvhb.AUTO_ACCEPT is False
            assert c.status()["auto_accept"] is False
        finally:
            c.set_auto_accept(True)

    asyncio.run(main())


def test_set_team_validates_against_pool():
    async def main():
        c = _ctrl(asyncio.get_running_loop())
        c.set_team("beta_team")
        assert c.team_pin == "beta_team"
        c.set_team("")                              # '' = Teambuilder-open / default behaviour
        assert c.team_pin is None
        with pytest.raises(ValueError):
            c.set_team("not_a_team")

    asyncio.run(main())


def test_save_format_writes_env_atomically(tmp_path):
    env = tmp_path / ".env"
    env.write_text("PS_USERNAME=bot\nPS_PASSWORD=secret\n", encoding="utf-8")

    async def main():
        c = _ctrl(asyncio.get_running_loop(), env_path=env)
        note = c.save_format("gen9championsvgc2026regma")
        assert "restart" in note
        text = env.read_text(encoding="utf-8")
        assert "VDANCE_BATTLE_FORMAT=gen9championsvgc2026regma" in text
        assert "PS_PASSWORD=secret" in text          # credentials untouched
        assert c.env_fmt_saved == "gen9championsvgc2026regma"
        # re-selecting the ACTIVE format clears the pending note (no restart needed)
        assert "already" in c.save_format(FMT)
        assert c.env_fmt_saved is None

    asyncio.run(main())


# ── HTTP smoke (real server, GET only — POSTs are exercised via the controller) ──
def test_http_panel_serves_status_and_page():
    prev_hook = _pvhb.RATING_HOOK

    async def main():
        page = FakePage()
        ctrl = start_control_ui(page=page, host=FakeHost(), tally={"ai": 1, "you": 2, "draw": 0},
                                ai_pool=POOL, fmt=FMT, username="VictoriousDancing",
                                loop=asyncio.get_running_loop(), env_path=Path("unused.env"),
                                port=18877, open_browser=False)
        try:
            loop = asyncio.get_running_loop()
            status_raw = await loop.run_in_executor(
                None, lambda: urllib.request.urlopen(ctrl.url + "api/status", timeout=5).read())
            html_raw = await loop.run_in_executor(
                None, lambda: urllib.request.urlopen(ctrl.url, timeout=5).read())
        finally:
            ctrl.stop()
        st = json.loads(status_raw)
        assert st["username"] == "VictoriousDancing" and st["format"] == FMT
        assert st["teams"] == POOL and st["tally"] == {"ai": 1, "you": 2, "draw": 0}
        assert b"Victory Dance" in html_raw and b"Battle!" in html_raw
        # 2026-09-03 (USER): the bandit arms PANEL (no bandit here -> rule None, the table still ships)
        assert st["bandit"] is None and st["bandit_on"] is False and st["bandit_rule"] is None
        assert b'id="arms"' in html_raw and b"function renderArms" in html_raw
        assert "🧠 learning".encode("utf-8") in html_raw

    try:
        asyncio.run(main())
    finally:
        _pvhb.RATING_HOOK = prev_hook
