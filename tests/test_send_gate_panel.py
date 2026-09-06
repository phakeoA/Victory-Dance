"""2026-09-05 — the throttle notices' last gap + the B2 ratings fix.

(1) The PANEL's own sends (``/utm`` + ``/search`` at every battle end, ``/cancelsearch``, ``/challenge``) and the
client's own ``|/noreply /leave`` on a tab close never went through the consumer's SendGate, so the gate's token
math was off by up to three at exactly the moment five lanes ship decisions — all 5 notices of the 06:59 session
followed such a burst, and the dropped message was a live room's decision. Now: room-less commands route through
the gate when one is installed (byte-identical ``page.evaluate`` without one — tests, local play), and a tab close
DEBITS a token.

(2) B2's opponent-rating weights were inert ('known 0/289'): the recorder sealed both ratings as None because
``battle.rating`` is never set on the browser transport. ``player_ratings_from_battle`` reads the PRE-battle ratings
from poke-env's ``|player|`` parse instead.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from v_dance.play import bot_control_ui as bcu
from v_dance.play import play_vs_human_browser as _pvhb
from v_dance.play.bot_control_ui import BotController
from v_dance.play.play_online_browser import player_ratings_from_battle

FMT = "gen9championsvgc2026regmb"


class FakePage:
    def __init__(self):
        self.sent = []

    async def evaluate(self, js, arg=None):
        self.sent.append((js, arg))
        return None


class _FakeTeam:
    def yield_team(self):
        return "PACKED"


class FakePlayer:
    def __init__(self):
        self._team = _FakeTeam()
        self._team_name = None

    def update_team(self, team):
        pass


class FakeHost:
    def __init__(self):
        self.player = FakePlayer()


class FakeGate:
    def __init__(self):
        self.sent, self.debited = [], []

    async def send(self, room, msg, on_sent=None):
        self.sent.append((room, msg))
        return True

    def debit(self, msg, room=None):
        self.debited.append((msg, room))


@pytest.fixture(autouse=True)
def _no_fs(monkeypatch):
    monkeypatch.setattr(bcu, "discover_teams", lambda reg=None: [])
    monkeypatch.setenv("VD_SITE_POLL", "0")
    monkeypatch.setattr(_pvhb, "SEND_GATE", [None])          # no consumer gate unless a test installs one
    yield


def _ctrl(loop, page):
    c = BotController(page=page, host=FakeHost(), tally={"ai": 0, "you": 0, "draw": 0},
                      ai_pool=["alpha"], fmt=FMT, username="VictoriousDancing", loop=loop,
                      env_path=Path("unused.env"))
    c._load_scoped_team = lambda scoped: "LOADED"
    return c


def _socket_sends(page):
    return [js for js, _ in page.sent if "app.socket.send" in js]


def test_without_a_gate_the_panel_sends_exactly_as_before():
    async def main():
        page = FakePage()
        c = _ctrl(asyncio.get_running_loop(), page)
        await c._do_search()
        js = _socket_sends(page)
        assert any("'|/utm '" in s for s in js) and any("'|/search '" in s for s in js)
        assert [a for _, a in page.sent if a in ("PACKED", FMT)] == ["PACKED", FMT]      # utm first, then search
        c.searching = True
        await c.stop_ladder()
        assert any("'|/cancelsearch'" in s for s in _socket_sends(page))
    asyncio.run(main())


def test_with_a_gate_the_panels_room_less_commands_route_through_it_in_order():
    async def main():
        page, gate = FakePage(), FakeGate()
        _pvhb.SEND_GATE[0] = gate
        c = _ctrl(asyncio.get_running_loop(), page)
        await c._do_search()
        assert gate.sent == [("", "/utm PACKED"), ("", f"/search {FMT}")]            # same wire text, gated
        assert _socket_sends(page) == []                                             # nothing bypassed the gate
        assert c._search_outstanding is True                                         # the run's bookkeeping unchanged
        c.searching = True
        await c.stop_ladder()
        assert gate.sent[-1] == ("", "/cancelsearch")
        await c.send_challenge("Some Rival", None)
        assert gate.sent[-1] == ("", f"/challenge Some Rival, {FMT}") and c.challenge_out == "Some Rival"
        await c.cancel_challenge()
        assert gate.sent[-1] == ("", "/cancelchallenge somerival") and c.challenge_out is None
    asyncio.run(main())


def test_closing_a_battle_tab_debits_the_gate_for_the_clients_own_leave():
    async def main():
        page, gate = FakePage(), FakeGate()
        _pvhb.SEND_GATE[0] = gate
        c = _ctrl(asyncio.get_running_loop(), page)
        base = f"battle-{FMT}-7001"
        c._full_tags[base] = base + "-privatesuffix0000000000000000"
        c._close_room(base)
        await asyncio.sleep(0.05)
        assert any("app.leaveRoom" in js for js, _ in page.sent)                     # the tab still closes
        assert gate.debited == [("/noreply /leave", base + "-privatesuffix0000000000000000")]
        _pvhb.SEND_GATE[0] = None                                                    # no gate → no debit, no error
        c._close_room(base)
        await asyncio.sleep(0.05)
        assert len(gate.debited) == 1
    asyncio.run(main())


def test_player_ratings_come_from_the_player_lines_with_the_battle_attrs_as_fallback():
    b = SimpleNamespace(_players=[{"username": "VictoriousDancing", "player": "p1", "avatar": "102", "rating": "1347"},
                                  {"username": "Rival", "player": "p2", "avatar": "169", "rating": "1262"}],
                        rating=None, opponent_rating=None)
    assert player_ratings_from_battle(b, "victorious dancing") == (1347, 1262)       # id compare, case/space-free
    assert player_ratings_from_battle(b, "VICTORIOUSDANCING") == (1347, 1262)
    # an unrated game: no rating in the player lines, nothing to fall back on
    u = SimpleNamespace(_players=[{"username": "VictoriousDancing", "player": "p1", "avatar": "102"},
                                  {"username": "Rival", "player": "p2", "avatar": "169", "rating": ""}],
                        rating=None, opponent_rating=None)
    assert player_ratings_from_battle(u, "VictoriousDancing") == (None, None)
    # the poke-env attributes remain the fallback (a transport that does set them)
    f = SimpleNamespace(_players=[], rating=1500, opponent_rating=1490)
    assert player_ratings_from_battle(f, "VictoriousDancing") == (1500, 1490)
    # junk never raises
    j = SimpleNamespace(_players=[None, {"username": "x", "rating": "abc"}, "str"], rating=None, opponent_rating=None)
    assert player_ratings_from_battle(j, "") == (None, None)
