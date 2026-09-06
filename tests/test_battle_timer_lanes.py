"""Battle timer under LANES (2026-09-02) — the opponent-stall ``/timer on`` judged PER ROOM, on every
consumer pass, plus the USER's IMMEDIATE mode (timer at the first frame of every game).

USER report 09-02: "sometimes in the team pick window, in some battles while doing parallel, it
forgets to turn on the timer and people just leave, so it permanently takes up one of the lanes."
Root cause locked here: the consumer kept ONE ``active_tag`` (the last room to send a frame) and ONE
``last_ship`` stamp (our last decision in ANY room), and only checked them on idle ticks. Under lanes
a neighbouring room's frames both re-pointed ``active_tag`` and refreshed ``last_ship``, so a room
whose opponent left at team preview was never looked at — its lane stayed occupied until the panel's
5-minute ghost sweep, and the server still counted the room against the 5-game cap.

Fixes: (1) ``last_ship`` is per room, swept every loop pass; (2) a toggle sends ``/timer on`` at a
room's FIRST frame instead (panel / Mission Control checkbox, launch default VD_TIMER_IMMEDIATE).
Everything is driven with fakes — no Playwright, no server."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("poke_env")

import v_dance.play.play_vs_human_browser as _pvhb
from v_dance.play import bot_control_ui as bcu
from v_dance.play import play_online_browser as pob
from v_dance.play.bot_control_ui import BotController

FMT = "gen9championsvgc2026regmb"
A = f"battle-{FMT}-9001"
B = f"battle-{FMT}-9002"


# ── fakes ──────────────────────────────────────────────────────────────────────
class FakePage:
    def __init__(self):
        self.evals = []

    async def evaluate(self, js, arg=None):
        self.evals.append((js, arg))
        return None

    def is_closed(self):
        return False


class FakeHost:
    """Ships a ``/team`` for every ``|request|`` frame (what the real host does at team preview)."""

    def __init__(self):
        self._ended = set()
        self.fed = []
        self.player = type("P", (), {"_battles": {}})()

    async def feed_async(self, payload):
        self.fed.append(payload)
        tag = payload.split("\n", 1)[0].lstrip(">")
        if "|request|" in payload:
            return [(tag, "/team 1234")]
        return []

    def end_battle(self, t):
        self._ended.add(t)

    def abandon_battle(self, t):
        self._ended.add(t)


def _timer_sends(page):
    return [a["r"] for js, a in page.evals if isinstance(a, dict) and a.get("m") == "/timer on"]


def _ships(page):
    return [(a["r"], a["m"]) for js, a in page.evals
            if isinstance(a, dict) and str(a.get("m", "")).startswith(("/team", "/choose"))]


async def _drive(frames, *, total_s: float, page=None, host=None):
    """Run the consumer over a scripted (delay, frame) list, stop at total_s; returns (page, host)."""
    page, host = page or FakePage(), host or FakeHost()
    q: asyncio.Queue = asyncio.Queue()
    stop = asyncio.Event()

    async def feeder():
        for delay, frame in frames:
            await asyncio.sleep(delay)
            q.put_nowait(frame)

    async def stopper():
        await asyncio.sleep(total_s)
        stop.set()

    asyncio.create_task(feeder())
    asyncio.create_task(stopper())
    await _pvhb._ai_consumer(page, host, q, ["a", "b"], {"ai": 0, "you": 0, "draw": 0}, stop)
    return page, host


@pytest.fixture(autouse=True)
def _fast_grace(monkeypatch):
    monkeypatch.setattr(_pvhb, "_OPP_TIMER_S", 0.4)     # the 30 s grace, shrunk for the test clock
    monkeypatch.setattr(_pvhb, "TIMER_IMMEDIATE", False)
    # 2026-09-04: these scenarios stream a decision every 50 ms (20/s — the real server refuses that too), so the
    # outgoing pacing gate would rightly hold the lower-priority /timer on past the test window. The gate has its
    # own timing tests (test_send_gate*.py); here it must not be the clock.
    monkeypatch.setattr(_pvhb.SendGate, "REFILL_S", 0.01)
    _pvhb.REJOINING.clear()
    yield
    _pvhb.REJOINING.clear()


# ── the bug: a stalled room hidden behind a busy neighbour ───────────────────
def test_grace_timer_fires_for_a_stalled_room_while_a_neighbour_streams_frames():
    # Room A: team preview request → we ship /team → the opponent walks away (no more A frames).
    # Room B: keeps requesting + shipping every 0.1 s (a live game next door) — its frames used to
    # re-point the single active_tag and refresh the single last_ship, hiding A forever.
    frames = [(0.0, f">{A}\n|init|battle"), (0.02, f">{A}\n|request|{{\"teamPreview\":true}}")]
    frames += [(0.1, f">{B}\n|request|{{\"turn\":{i}}}") for i in range(14)]
    page, _ = asyncio.run(_drive(frames, total_s=1.7))
    assert _timer_sends(page) == [A]                      # A once; B (always freshly shipped) never
    assert any(r == A for r, _ in _ships(page)) and sum(1 for r, _ in _ships(page) if r == B) >= 10


def test_grace_timer_fires_on_an_idle_consumer_too_and_only_once_per_room():
    # Idle consumer: the sweep runs on the ~1 s queue-timeout tick (the old idle cadence), so the
    # grace (0.4 s here) is judged at t≈1.0; the game then moves on at 1.3 s → we ship again.
    frames = [(0.0, f">{A}\n|request|{{\"teamPreview\":true}}"),
              (1.3, f">{A}\n|request|{{\"turn\":1}}")]
    page, _ = asyncio.run(_drive(frames, total_s=2.2))
    assert _timer_sends(page) == [A]                      # once; never re-sent after the second ship
    assert len(_ships(page)) == 2


def test_grace_state_is_reclaimed_when_the_room_ends():
    frames = [(0.0, f">{A}\n|request|{{\"teamPreview\":true}}"),
              (0.1, f">{A}\n|win|Opponent"),                  # ended well inside the grace
              (0.2, f">{A}\n|raw|late chat")]                # a stray frame for the ended room
    page, host = asyncio.run(_drive(frames, total_s=1.0))
    assert A in host._ended and _timer_sends(page) == []  # no timer for a finished room


def test_grace_timer_is_not_sent_for_a_room_that_went_away_noinit():
    frames = [(0.0, f">{A}\n|request|{{\"teamPreview\":true}}"),
              (0.1, f">{A}\n|noinit|nonexistent|The room does not exist.")]
    page, host = asyncio.run(_drive(frames, total_s=1.0))
    assert A in host._ended and _timer_sends(page) == []


def test_timer_sweep_survives_a_dead_page(capsys):
    class DeadPage(FakePage):
        async def evaluate(self, js, arg=None):
            if isinstance(arg, dict) and arg.get("m") == "/timer on":
                raise RuntimeError("page gone")
            return await super().evaluate(js, arg)

    frames = [(0.0, f">{A}\n|request|{{\"teamPreview\":true}}")]
    page, _ = asyncio.run(_drive(frames, total_s=2.6, page=DeadPage()))   # two idle ticks pass
    out = capsys.readouterr().out
    assert "timer-on failed (non-fatal)" in out and out.count("timer-on failed") == 1   # once, no spin


# ── the toggle: /timer on at the first frame ────────────────────────────────
def test_immediate_mode_sends_the_timer_at_each_rooms_first_frame_once(monkeypatch):
    monkeypatch.setattr(_pvhb, "TIMER_IMMEDIATE", True)
    frames = [(0.0, f">{A}\n|init|battle"),
              (0.05, f">{A}\n|request|{{\"teamPreview\":true}}"),
              (0.05, f">{B}\n|init|battle"),
              (0.05, f">{A}\n|init|battle\n|title|replayed on rejoin")]   # a reconnect replays |init|
    page, _ = asyncio.run(_drive(frames, total_s=0.6))
    assert _timer_sends(page) == [A, B]                    # once per room, rejoin never re-sends
    order = [(a.get("r"), a.get("m")) for _, a in page.evals if isinstance(a, dict)]
    assert order.index((A, "/timer on")) < order.index((A, "/team 1234"))   # before our first ship


def test_immediate_mode_skips_ended_rooms_and_noinit_frames(monkeypatch):
    monkeypatch.setattr(_pvhb, "TIMER_IMMEDIATE", True)
    frames = [(0.0, f">{A}\n|request|{{\"teamPreview\":true}}"),
              (0.05, f">{A}\n|win|Opponent"),
              (0.05, f">{A}\n|raw|late chat"),              # ended → gated
              (0.05, f">{B}\n|noinit|nonexistent|gone")]    # never existed → gated
    page, host = asyncio.run(_drive(frames, total_s=0.5))
    assert _timer_sends(page) == [A] and B in host._ended


def test_grace_mode_stays_the_default_and_the_sweep_is_the_backstop_under_immediate(monkeypatch):
    assert _pvhb.TIMER_IMMEDIATE is False                  # module default = the 07-10 behaviour
    monkeypatch.setattr(_pvhb, "TIMER_IMMEDIATE", True)
    frames = [(0.0, f">{A}\n|request|{{\"teamPreview\":true}}")]
    page, _ = asyncio.run(_drive(frames, total_s=1.0))
    assert _timer_sends(page) == [A]                       # immediate sent it; the sweep did NOT double


# ── panel / Mission Control / launch plumbing ───────────────────────────────
class _PanelPage:
    def __init__(self):
        self.sent = []

    async def evaluate(self, js, arg=None):
        self.sent.append((js, arg))


def _ctrl(loop, monkeypatch):
    monkeypatch.setattr(bcu, "discover_teams", lambda reg=None: [])
    monkeypatch.setenv("VD_SITE_POLL", "0")
    host = type("H", (), {"player": type("P", (), {"_team_name": None})()})()
    return BotController(page=_PanelPage(), host=host, tally={"ai": 0, "you": 0, "draw": 0},
                         ai_pool=["alpha"], fmt=FMT, username="VictoriousDancing", loop=loop,
                         env_path=Path("unused.env"), matchups=False)


def test_panel_toggle_flips_the_consumer_flag_and_reports_in_status(monkeypatch):
    async def main():
        c = _ctrl(asyncio.get_running_loop(), monkeypatch)
        s = c.status()
        assert s["timer_immediate"] is False and s["timer_grace_s"] == pytest.approx(0.4)
        c.set_timer_immediate(True)
        assert _pvhb.TIMER_IMMEDIATE is True and c.status()["timer_immediate"] is True
        assert any("IMMEDIATE" in e for e in c.events)
        c.set_timer_immediate(False)
        assert _pvhb.TIMER_IMMEDIATE is False
        assert any("GRACE" in e for e in c.events)

    asyncio.run(main())


def test_panel_html_carries_the_toggle_wired_to_the_options_api():
    html = bcu._PANEL_HTML
    assert 'id="timerNow"' in html and "timer_immediate: $('timerNow').checked" in html
    assert "s.timer_immediate" in html


def test_mission_control_exposes_the_timer_env_key_and_both_cards():
    from v_dance.datatools import mission_control as mc
    assert "VD_TIMER_IMMEDIATE" in mc._ENV_WRITE_KEYS and "VD_TIMER_IMMEDIATE" in mc._ENV_READ_KEYS
    html = mc._HTML_PATH.read_text(encoding="utf-8")
    assert 'envRow("VD_TIMER_IMMEDIATE", "bool")' in html          # Deploy row
    assert 'id="ob-launch-timer"' in html and 'key: "VD_TIMER_IMMEDIATE"' in html   # launch card
    assert 'id="ob-timer"' in html and "timer_immediate: $(\"ob-timer\").checked" in html  # live card
    assert "OB.timer_immediate" in html


def test_online_launch_reads_the_timer_mode_from_env_and_echoes_it(monkeypatch):
    monkeypatch.delenv("VD_TIMER_IMMEDIATE", raising=False)
    monkeypatch.setattr(pob, "_ENV", {})
    assert pob._timer_immediate_env() is False
    monkeypatch.setenv("VD_TIMER_IMMEDIATE", "1")
    assert pob._timer_immediate_env() is True
    monkeypatch.setenv("VD_TIMER_IMMEDIATE", "0")
    assert pob._timer_immediate_env() is False
    monkeypatch.delenv("VD_TIMER_IMMEDIATE", raising=False)
    monkeypatch.setattr(pob, "_ENV", {"VD_TIMER_IMMEDIATE": "1"})   # .env fallback
    assert pob._timer_immediate_env() is True
    assert "IMMEDIATE" in pob._timer_banner(True, 30.0)
    assert "GRACE" in pob._timer_banner(False, 30.0) and "30s" in pob._timer_banner(False, 30.0)
