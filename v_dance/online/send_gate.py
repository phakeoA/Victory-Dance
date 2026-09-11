"""Outgoing message pacing for the browser transport — 2026-09-04 (USER: "in the team picker a message said it
couldn't be sent because it was typed too quickly, and it didn't pick any Pokémon").

Showdown throttles EVERY chat command a connection sends — ``/choose``, ``/team``, ``/timer on``,
``/acceptopenteamsheets`` included — through ONE per-user queue (``server/users.ts``, pinned @4880d36):
``THROTTLE_DELAY`` 600 ms between parses once the queue is busy, a buffer of ``THROTTLE_BUFFER_LIMIT`` 6, and
anything past the buffer is DROPPED with
``|raw|<strong class="message-throttle-notice">Your message was not sent because you've been typing too quickly.</strong>``.
Five lanes filling at the same moment ship up to three messages per room at team preview (the immediate
``/timer on``, the OTS answer, the team choice) = 15 in a burst — the later rooms' team choices were the
messages the server threw away. (``/cmd userdetails``, the link probe, is exempt server-side.)

``SendGate`` keeps the connection UNDER the server's queue: a token bucket of ``BURST`` 5 immediate sends (the
server accepts 6 in 600 ms) refilled one per ``REFILL_S`` 0.65 s (slower than the 600 ms drain), and when
messages must wait they go out by priority — game decisions first (``/choose`` ``/team`` ``/forfeit``), the OTS
answer next, ``/timer on`` last (a 2 s delay on a convenience command costs nothing) — FIFO inside a class, so a
room's accept-before-choose order holds. While tokens remain ``send`` delivers inline (the pre-gate behaviour,
errors propagate to the caller); beyond that it queues and a pump task on the page's loop drains it, so the
consumer never blocks. A server throttle notice seen in a frame is counted (``notices``) and logged — with the
gate in place it should stay at 0; if it ticks, lower ``BURST`` / raise ``REFILL_S``.
"""
from __future__ import annotations

import asyncio
import heapq
import time
from collections import deque
from typing import Awaitable, Callable, Optional


class SendGate:
    BURST = 5                 # immediate sends before pacing (the server buffers 6 per 600 ms)
    REFILL_S = 0.65           # seconds per token once the burst is spent (the server drains one per 600 ms)
    NOTICE = "message-throttle-notice"
    PRIORITY = (("/choose", 0), ("/team", 0), ("/forfeit", 0), ("/leave", 0),
                ("/acceptopenteamsheets", 1), ("/rejectopenteamsheets", 1),
                ("/timer", 2))

    def __init__(self, sender: Callable[[str, str], Awaitable[None]], *, burst: Optional[float] = None,
                 refill_s: Optional[float] = None, clock=time.monotonic, sleep=asyncio.sleep, log=print):
        self._sender = sender
        self.burst = float(burst if burst is not None else self.BURST)
        self.refill_s = float(refill_s if refill_s is not None else self.REFILL_S)
        self._clock, self._sleep, self._log = clock, sleep, log
        self._tokens = self.burst
        self._last = self._clock()
        self._heap: list = []
        self._seq = 0
        self._task: Optional[asyncio.Task] = None
        self.sent = 0            # delivered to the socket
        self.deferred = 0        # had to wait for a token
        self.notices = 0         # server throttle notices seen in frames (should stay 0)
        self.errors = 0          # sender exceptions (pump path: logged; inline path: raised)
        self.max_wait_s = 0.0    # longest queue wait so far
        # 2026-09-05: messages that left OUTSIDE the gate but count against the same server throttle (``debit``),
        # and the last sends of either kind — logged with a throttle notice so the dropped burst is attributable
        self.external = 0
        self._recent: deque = deque(maxlen=12)     # (t, room, msg, "gate" | "ext")

    @classmethod
    def priority(cls, msg: str) -> int:
        for prefix, prio in cls.PRIORITY:
            if msg.startswith(prefix):
                return prio
        return 1

    _EPS = 1e-9               # a refill that lands at 0.999999… IS a whole token (float rounding)
    _MIN_SLEEP_S = 0.005      # never spin: a deficit below a float's resolution would re-sleep forever

    def _refill(self) -> None:
        now = self._clock()
        self._tokens = min(self.burst, self._tokens + max(0.0, now - self._last) / self.refill_s)
        self._last = now

    def _take_token(self) -> bool:
        if self._tokens >= 1.0 - self._EPS:
            self._tokens = max(0.0, self._tokens - 1.0)
            return True
        return False

    @property
    def queued(self) -> int:
        return len(self._heap)

    async def send(self, room: Optional[str], msg: str, on_sent: Optional[Callable[[], None]] = None) -> bool:
        """Deliver ``room|msg`` now if a token is free and nothing is waiting (returns True), else queue it for
        the pump (returns False; ``on_sent`` fires when it actually leaves). ``room`` None/"" = a global command."""
        self._refill()
        if not self._heap and self._take_token():
            await self._deliver(room, msg, on_sent, raise_errors=True)
            return True
        self.deferred += 1
        heapq.heappush(self._heap, (self.priority(msg), self._seq, self._clock(), room or "", msg, on_sent))
        self._seq += 1
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(self._pump())
        return False

    async def _deliver(self, room, msg, on_sent, *, raise_errors: bool) -> None:
        try:
            await self._sender(room or "", msg)
        except Exception as exc:                          # noqa: BLE001 — the pump must survive a dead page
            self.errors += 1
            if raise_errors:
                raise
            self._log(f"[send-gate] send failed (non-fatal): {exc!r}  ({room} {msg[:40]})")
            return
        self.sent += 1
        self._recent.append((self._clock(), room or "", msg, "gate"))
        if on_sent is not None:
            try:
                on_sent()
            except Exception as exc:                      # noqa: BLE001
                self._log(f"[send-gate] on_sent hook failed (non-fatal): {exc!r}")

    async def _pump(self) -> None:
        while self._heap:
            self._refill()
            if not self._take_token():
                await self._sleep(max(self._MIN_SLEEP_S, (1.0 - self._tokens) * self.refill_s))
                continue
            _prio, _seq, t_in, room, msg, on_sent = heapq.heappop(self._heap)
            self.max_wait_s = max(self.max_wait_s, self._clock() - t_in)
            await self._deliver(room, msg, on_sent, raise_errors=False)

    async def drain(self) -> None:
        """Wait for the pump to empty the queue (tests / shutdown)."""
        if self._task is not None and not self._task.done():
            await self._task

    def debit(self, msg: str, room: Optional[str] = None) -> None:
        """2026-09-05: account for a message that left OUTSIDE the gate — the client's own ``|/noreply /leave`` when
        the panel closes a battle tab, or anything a caller ships directly. It spent one of the server's 6-per-600-ms
        slots, so take a token, into the negative when none is free (the pump then waits that much longer). Every
        throttle notice of 2026-09-05 followed an un-gated leave + ``/utm`` + ``/search`` burst at a battle's end
        that the gate could not see — and the dropped message was a live room's decision."""
        self._refill()
        self._tokens = max(-self.burst, self._tokens - 1.0)
        self.external += 1
        self._recent.append((self._clock(), room or "", msg, "ext"))

    def note_frame(self, payload: str, room: Optional[str] = None) -> bool:
        """Count + log a server throttle notice inside an incoming frame — with the last sends (relative seconds,
        gate / ext, the room's tail) so the burst that caused it is visible in the log."""
        if self.NOTICE not in (payload or ""):
            return False
        self.notices += 1
        now = self._clock()
        recent = ", ".join(f"{t - now:+.1f}s {via}:{m[:24]}{(' @' + r[-8:]) if r else ''}"
                           for t, r, m, via in self._recent)
        self._log(f"[send-gate] SERVER THROTTLE NOTICE #{self.notices}{' in ' + room if room else ''}: Showdown dropped one "
                  f"of our messages ('typing too quickly'). The gate should prevent this — lower BURST / raise REFILL_S "
                  f"if it repeats (queued {self.queued}, sent {self.sent}, external {self.external}); "
                  f"last sends: {recent or '-'}")
        return True

    def stats(self) -> dict:
        return {"sent": self.sent, "deferred": self.deferred, "queued": self.queued, "notices": self.notices,
                "errors": self.errors, "max_wait_s": round(self.max_wait_s, 2),
                "burst": self.burst, "refill_s": self.refill_s, "external": self.external}
