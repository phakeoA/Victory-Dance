"""2026-09-04 — the outgoing pacing gate (USER: "in the team picker a message said it couldn't be sent because it
was typed too quickly, and it didn't pick any Pokémon").

Showdown drops chat commands beyond THROTTLE_BUFFER_LIMIT (6) per THROTTLE_DELAY (600 ms) per connection — /choose
and /team included — and five lanes at team preview ship up to 15 at once. The gate: 5 inline sends, then one per
0.65 s by priority (decisions > OTS answer > timer), FIFO inside a class; inline errors propagate, pump errors are
logged; server notices are counted. Virtual clock — no sleeping, no Playwright."""
from __future__ import annotations

import asyncio

import pytest

from v_dance.play.send_gate import SendGate


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    async def sleep(self, s):
        self.t += s


def _gate(sender, **kw):
    c = Clock()
    return SendGate(sender, clock=c, sleep=c.sleep, log=lambda *_: None, **kw), c


def test_burst_then_paced_by_priority_with_fifo_inside_a_class():
    async def main():
        out = []
        async def sender(room, msg):
            out.append((room, msg, round(c.t, 2)))
        g, c = _gate(sender)
        rooms = [f"battle-x-{i}" for i in range(5)]
        for r in rooms:                                       # 5 immediate timers use the burst
            assert await g.send(r, "/timer on") is True
        for r in rooms:                                       # a room's OTS answer then its team choice
            assert await g.send(r, "/acceptopenteamsheets") is False
            assert await g.send(r, "/team 1234") is False
        assert g.queued == 10 and g.deferred == 10 and g.sent == 5
        await g.drain()
        assert g.sent == 15 and g.queued == 0
        deferred = out[5:]
        assert [m for _, m, _ in deferred] == ["/team 1234"] * 5 + ["/acceptopenteamsheets"] * 5   # decisions first
        assert [r for r, _, _ in deferred[:5]] == rooms and [r for r, _, _ in deferred[5:]] == rooms   # FIFO per class
        times = [t for _, _, t in deferred]
        assert times[0] == pytest.approx(0.65) and all(b - a == pytest.approx(0.65) for a, b in zip(times, times[1:]))
        assert g.max_wait_s == pytest.approx(6.5) and g.stats()["notices"] == 0
    asyncio.run(main())


def test_priority_table_and_inline_path_when_idle():
    assert SendGate.priority("/choose move 1") == 0 and SendGate.priority("/team 1234") == 0
    assert SendGate.priority("/forfeit") == 0 and SendGate.priority("/acceptopenteamsheets") == 1
    assert SendGate.priority("/rejectopenteamsheets") == 1 and SendGate.priority("/timer on") == 2
    assert SendGate.priority("/utm abc") == 1                 # anything else: the middle class

    async def main():
        calls = []
        async def sender(room, msg):
            calls.append((room, msg))
        g, c = _gate(sender)
        assert await g.send(None, "/utm packed|team") is True and calls == [("", "/utm packed|team")]
        assert g._task is None                                # nothing queued → no pump task
        c.t += 10.0                                           # tokens refill (capped at the burst)
        g._refill()
        assert g._tokens == pytest.approx(5.0)
    asyncio.run(main())


def test_inline_errors_propagate_and_pump_errors_are_logged_not_fatal():
    async def main():
        logs, ok = [], []
        async def sender(room, msg):
            if "bad" in msg:
                raise RuntimeError("page closed")
            ok.append(msg)
        c = Clock()
        g = SendGate(sender, clock=c, sleep=c.sleep, log=logs.append)
        with pytest.raises(RuntimeError):                     # inline: the caller's try/except sees it (timer-on)
            await g.send("r", "/timer bad")
        assert g.errors == 1
        for _ in range(4):
            await g.send("r", "/choose ok")                   # spend the remaining tokens
        await g.send("r", "/choose bad")                      # deferred
        await g.send("r", "/choose ok2")                      # deferred, after the failing one
        await g.drain()
        assert ok[-1] == "/choose ok2" and g.errors == 2 and any("send failed" in s for s in logs)
    asyncio.run(main())


def test_on_sent_fires_at_delivery_and_notices_are_counted():
    async def main():
        stamps = []
        async def sender(room, msg):
            pass
        g, c = _gate(sender)
        for i in range(5):
            await g.send("r", f"/choose {i}", on_sent=lambda: stamps.append(c.t))
        await g.send("r", "/choose late", on_sent=lambda: stamps.append(c.t))   # deferred
        assert stamps == [0.0] * 5
        await g.drain()
        assert stamps[-1] == pytest.approx(0.65)              # the think-clock stamp rides the ACTUAL send
        assert g.note_frame(">battle-x-1\n|raw|<strong class=\"message-throttle-notice\">Your message was not sent "
                            "because you've been typing too quickly.</strong>", "battle-x-1") is True
        assert g.note_frame(">battle-x-1\n|turn|3") is False and g.notices == 1
        st = g.stats()
        assert st["sent"] == 6 and st["deferred"] == 1 and st["queued"] == 0 and st["burst"] == 5 and st["refill_s"] == 0.65
    asyncio.run(main())


def test_debit_accounts_for_ungated_sends_and_a_notice_logs_the_recent_burst():
    """2026-09-05: the 5 notices of the 06:59 session all followed a tab-close ``/leave`` + the panel's ``/utm`` +
    ``/search`` — sends the gate never saw, so its tokens were 3 too generous at the moment five lanes ship decisions.
    ``debit`` takes a token for such a send (into the negative), the next gated send waits for it, and a notice now
    prints the last sends with relative times so the burst is attributable."""
    async def main():
        out, logs = [], []
        async def sender(room, msg):
            out.append((room, msg, round(c.t, 2)))
        c = Clock()
        g = SendGate(sender, clock=c, sleep=c.sleep, log=logs.append)
        for i in range(4):
            await g.send(f"battle-x-{i}", "/choose move 1")        # 4 of the 5 burst tokens
        g.debit("/noreply /leave", "battle-x-9")                    # the client's own leave: the 5th slot
        assert g.external == 1 and g._tokens == pytest.approx(0.0)
        g.debit("/utm PACKED")                                      # a second un-gated send → negative
        assert g._tokens == pytest.approx(-1.0)
        assert await g.send("battle-x-4", "/choose move 2") is False    # no token: deferred, not dropped
        await g.drain()
        assert out[-1][2] == pytest.approx(1.3)                     # waited for the debt + one token (2 × 0.65)
        assert g.sent == 5 and g.deferred == 1 and g.stats()["external"] == 2
        assert g.note_frame(">battle-x-4\n|raw|<strong class=\"message-throttle-notice\">x</strong>", "battle-x-4")
        line = logs[-1]
        assert "NOTICE #1 in battle-x-4" in line and "external 2" in line and "last sends:" in line
        assert "ext:/noreply /leave @ttle-x-9" in line and "ext:/utm PACKED" in line and "gate:/choose move 2" in line
        assert "-1.3s gate:/choose move 1 @ttle-x-0" in line       # relative seconds, oldest first
        # the debt never exceeds one burst (a runaway caller cannot stall the pump for ever)
        for _ in range(20):
            g.debit("/x")
        assert g._tokens == pytest.approx(-5.0)
    asyncio.run(main())
