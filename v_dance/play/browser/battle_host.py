"""battle_host.py — drive the production decision pipeline from RAW protocol frames (no websocket).

THE reusable linchpin for the "AI plays through the browser" transport (LOCAL now, ONLINE later). The
problem: gen141 reads a PARSED battle state, which today comes from poke-env BEING the websocket player.
When the AI is instead a BROWSER tab, poke-env can't also hold that session — so we capture the tab's
websocket frames and feed them to a poke-env player that is NOT connected to anything.

How it reuses EVERYTHING (no brain re-implementation, zero change to the production play path):
  * Construct the real ``VGCPlayer`` with ``start_listening=False`` → it builds the encoder + model +
    gap-#6 splice exactly as in live play, but its ``PSClient`` never opens a socket (PSClient only
    starts ``listen()`` when ``start_listening=True``).
  * FEED a captured frame to ``ps_client._handle_message(raw)`` — poke-env's own split + dispatch
    (``>battle`` → ``_handle_battle_message`` → on ``|request|`` → ``choose_move``/``teampreview``).
  * The decision is normally shipped via ``ps_client.send_message(msg, room)``; we PATCH that to CAPTURE
    the ``/choose …`` (and ``/timer on`` / ``/acceptopenteamsheets``) instead of writing to the (absent)
    socket. The transport then relays the captured messages into the tab.

So the splice / retry-on-reject / gimmick / teampreview / value-head logic is reused byte-for-byte. The
ONLY new code per transport is: (a) supplying frames and (b) shipping the captured messages.

⚠ EVENT-LOOP: poke-env's async primitives (queues/conditions/locks) are bound to the loop the player was
built on — ALWAYS the default ``POKE_LOOP`` (a Selector loop on its own background thread). Reach the host
from ANY other loop (e.g. Playwright's Proactor loop) via ``run_coroutine_threadsafe(host.feed_async(raw),
POKE_LOOP)`` + ``asyncio.wrap_future`` — that is exactly what ``play_vs_human_browser`` does. ``feed`` (sync)
wraps that bridge for non-async callers. Do NOT build the host on the Playwright loop (``loop=``): its prims
would land on a loop the consumer never targets → "Future attached to a different loop". Do NOT call the
sync ``feed`` from inside POKE_LOOP itself (it would deadlock) — use ``feed_async`` there.
"""
from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

from poke_env.concurrency import POKE_LOOP
from poke_env.ps_client import AccountConfiguration

from v_dance.formats import DEFAULT_FORMAT
from v_dance.play.model_io import DEFAULT_BC_CHECKPOINT, DEFAULT_TP_CHECKPOINT
from v_dance.play.player import VGCPlayer

log = logging.getLogger(__name__)

# (room, message) — e.g. ("battle-gen9...-123", "/choose move 1 1, move 2 2"). room is None for
# roomless commands. The transport ships these into the AI tab (DOM/routeWebSocket).
Outgoing = Tuple[Optional[str], str]


class BattleHost:
    """A connection-less poke-env ``VGCPlayer`` driven by raw protocol frames.

    Usage (transport side)::

        host = BattleHost(team=ai_team_paste)              # loads gen141 + SBDA once
        # ... on each captured websocket text frame `raw` from the AI tab:
        for room, msg in await host.feed_async(raw):       # decisions produced by THIS frame
            await send_into_tab(room, msg)                 # e.g. ship "/choose ..." back

    ``feed``/``feed_async`` return ONLY the messages produced by that frame, so the transport can ship
    them immediately. ``drain`` returns + clears anything buffered (e.g. if you feed several frames
    before shipping).
    """

    def __init__(
        self,
        *,
        team: Optional[str] = None,
        model_path: Optional[Path] = DEFAULT_BC_CHECKPOINT,
        team_chooser_path: Optional[Path] = DEFAULT_TP_CHECKPOINT,
        username: str = "VictoryDanceAI",
        battle_format: str = DEFAULT_FORMAT,
        device: str = "cpu",
        # MUST match production (run_local_battle.make_player → poke-env default False = REJECT open team
        # sheets → hidden teampreview, which is how gen141 + SBDA were trained/served). A mismatch desyncs
        # the teampreview timing (the host would defer to a |showteam| that a reject means never arrives).
        accept_open_team_sheet: bool = False,
        max_concurrent_battles: int = 100000,
        adapt_rules: bool = False,
        loop=POKE_LOOP,
    ):
        self._loop = loop
        self._outgoing: List[Outgoing] = []
        # Battles we've already torn down (end_battle). A late stray frame for one of these (e.g. the
        # opponent's |l|/leave or post-|win| chat) must NOT be routed to poke-env: _handle_battle_message
        # sends any non-init battle message through _get_battle(), which BLOCKS FOREVER on a tag no longer
        # in _battles → the single-threaded feed (and the whole consumer loop) wedges. Skip such frames.
        self._ended: set = set()
        self._ended_order: "deque[str]" = deque()      # eviction order so _ended stays bounded (audit leak)
        # Serialize feed_async on the player's loop so the per-frame ``_outgoing[before:]`` accounting is
        # sound even if a caller ever overlaps frames (the shipped consumer feeds one-at-a-time, but the
        # online roadmap may drain in parallel). Lazily created on the player's loop (see _get_lock).
        self._lock = None
        # The REAL production player — encoder + gen141 + SBDA + gap-#6 splice — but NOT connected.
        # poke-env's _battle_count_queue (sized by max_concurrent_battles) gets a put() on every battle
        # init and a get() on its |win|/|tie|. A battle that ends ABNORMALLY (tab closed mid-game, room
        # deinit with no win reaching us) leaks one slot; a huge maxsize means feed() can never block on a
        # full queue over a long session. We serve one battle at a time, so this only guards that edge.
        self.player = VGCPlayer(
            model_path=model_path,
            team_chooser_path=team_chooser_path,
            device=device,
            account_configuration=AccountConfiguration(username, None),
            battle_format=battle_format,
            team=team,
            start_listening=False,                 # ← no socket, no login, no listen()
            accept_open_team_sheet=accept_open_team_sheet,
            max_concurrent_battles=max_concurrent_battles,
            adapt_rules=adapt_rules,
            log_level=logging.WARNING,
            loop=loop,
        )
        # Capture decisions instead of websocket-sending them. poke-env calls
        # ``await ps_client.send_message(message, room="", message_2=None)`` for every /choose,
        # /timer on, /acceptopenteamsheets, /forfeit, /leave — we record (room, message) for the
        # transport (room="" → a roomless/global command → stored as None).
        async def _capture(message, room="", message_2=None):  # matches PSClient.send_message
            self._outgoing.append((room or None, message))

        self.player.ps_client.send_message = _capture  # type: ignore[assignment]

    # ── Drive ────────────────────────────────────────────────────────────────────
    def _get_lock(self):
        import asyncio
        if self._lock is None:                         # created on first use, on the player's loop
            self._lock = asyncio.Lock()
        return self._lock

    async def feed_async(self, raw_frame: str) -> List[Outgoing]:
        """Push ONE raw websocket text frame through poke-env's own routing, on the player's loop.
        Returns the (room, message) decisions produced by this frame (usually 0 or 1). Serialized per
        host so concurrent feeds can't cross-attribute decisions via the ``_outgoing`` slice."""
        # A stray frame for an ALREADY-ENDED battle (late |l|/leave/chat after |win|) would route through
        # poke-env's _get_battle() and block forever on _battle_start_condition (the tag is gone from
        # _battles) — wedging this feed and the entire consumer loop. Drop it before it reaches poke-env.
        if raw_frame.startswith(">battle"):
            tag = raw_frame.split("\n", 1)[0].lstrip(">")
            if tag in self._ended:
                return []
        async with self._get_lock():
            before = len(self._outgoing)               # captured INSIDE the lock so it can't race
            await self.player.ps_client._handle_message(raw_frame)
            return self._outgoing[before:]

    def feed(self, raw_frame: str, timeout: Optional[float] = 30.0) -> List[Outgoing]:
        """Sync bridge for callers on a DIFFERENT thread than the player's loop (e.g. a test, or a
        transport whose loop ≠ the player's). Blocks until the frame is handled. NEVER call this from
        within the player's own loop (deadlock) — use ``feed_async`` there."""
        import asyncio
        fut = asyncio.run_coroutine_threadsafe(self.feed_async(raw_frame), self._loop)
        return fut.result(timeout)                   # propagate exceptions; wait for completion + lock

    def drain(self) -> List[Outgoing]:
        """Return + clear all buffered outgoing messages (across every fed frame so far)."""
        out = self._outgoing[:]
        self._outgoing.clear()
        return out

    def end_battle(self, tag: str) -> None:
        """Reclaim a finished battle's state so a long session doesn't accumulate every battle's protocol
        log + captured commands. The transport calls this when it sees the battle's ``|win|``/``|tie|``."""
        tag = (tag or "").lstrip(">")
        # gate late stray frames for this tag (see feed_async), BOUNDED to the most-recent tags — only a
        # freshly-ended battle can emit a late frame, so retaining every tag forever was an unbounded leak
        # in this explicitly long-lived (human-vs-AI / online) host.
        if tag not in self._ended:
            self._ended.add(tag)
            self._ended_order.append(tag)
            while len(self._ended_order) > 512:
                self._ended.discard(self._ended_order.popleft())
        self.player._battles.pop(tag, None)
        proto = getattr(self.player, "_proto_log", None)
        if isinstance(proto, dict):
            proto.pop(tag, None)
        locks = getattr(self.player.ps_client, "_battle_locks", None)
        if isinstance(locks, dict):
            locks.pop(tag, None)
        self._outgoing.clear()                         # one battle at a time → safe to drop the buffer

    # ── Introspection (for tests / the transport's UI) ────────────────────────────
    @property
    def battles(self) -> dict:
        """poke-env battle objects this host has reconstructed, keyed by battle tag."""
        return self.player._battles

    @property
    def source_counts(self):
        """Per-source decision tally (model / retry / forced_switch / …) — proves the MODEL drove."""
        return self.player._source_counts
