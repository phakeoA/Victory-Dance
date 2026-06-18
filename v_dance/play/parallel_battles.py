"""Shared parallel-battle runner (task #13) — the ONE place poke-env battles are
played with bounded concurrency, watchdog timeouts, and clean teardown.

Three live call-sites had grown private copies of the same async glue:

  * ``v_dance/eval/gauntlet.run_gauntlet``              (model vs the scripted / prev_best ladder)
  * ``v_dance/selfplay/generation.collect_with_league`` (league self-play collection)
  * ``v_dance/selfplay/game_runner.run_self_play_games``(Phase-0 plumbing smoke)

each re-implementing the same three pieces: a ``Semaphore(workers)`` + ``gather`` to
bound the in-flight battle count (the dominant throughput lever, docs sec 20), a
``wait_for(battle_against, timeout)`` watchdog so a hung battle can't freeze the run
for hours, and a best-effort ``stop_listening()`` + ``close()`` teardown. Extracting
them here defines + tests the behaviour ONCE and gives the multiprocessing layer
(task #14) a single seam to partition across processes.

Deliberately import-light — only ``asyncio`` + ``logging``, NO poke-env / torch — so the
primitives unit-test offline against fake players, and ``play`` stays the lowest layer
(both ``eval`` and ``selfplay`` import it, so there is no import cycle)."""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

DEFAULT_BATTLE_TIMEOUT = 90.0


async def play_pairing(model_player, opponent, n: int, *,
                       battle_timeout: Optional[float] = DEFAULT_BATTLE_TIMEOUT,
                       label: str = "") -> Tuple[int, int]:
    """Play ``n`` battles ``model_player`` vs ``opponent`` and return the
    ``(model_wins, model_finished)`` DELTA over this chunk (read from the player's
    own counters, so the figure is meaningful even when the caller tallies
    trajectories instead).

    ``battle_timeout`` (seconds PER BATTLE) bounds the whole chunk at
    ``battle_timeout * n``: if a battle hangs (a forced-switch / illusion desync that
    never resolves) the chunk is ABANDONED and the caller continues instead of
    freezing for hours — the battles that finished before the hang still count.
    ``battle_timeout`` <= 0 / ``None`` disables the watchdog (await to completion)."""
    won_before = model_player.n_won_battles
    fin_before = model_player.n_finished_battles
    coro = model_player.battle_against(opponent, n_battles=n)
    try:
        if battle_timeout and battle_timeout > 0:
            await asyncio.wait_for(coro, timeout=battle_timeout * n)
        else:
            await coro
    except asyncio.TimeoutError:
        fin = model_player.n_finished_battles - fin_before
        lab = f" {label}" if label else ""
        log.warning("battle pairing WATCHDOG fired after %.0fs (%d requested, %d "
                    "finished%s) — abandoning chunk, continuing.",
                    (battle_timeout or 0) * n, n, fin, lab)
    return (model_player.n_won_battles - won_before,
            model_player.n_finished_battles - fin_before)


async def close_players(*players) -> None:
    """Best-effort teardown for a set of finished players: ``stop_listening()`` each
    (drop the websocket so the next chunk's accounts can reconnect) then ``close()``
    each. Every step is guarded and ``None`` players are skipped — teardown of one
    player must never mask another's or raise out of a caller's ``finally``."""
    for p in players:
        if p is None:
            continue
        try:
            await p.ps_client.stop_listening()
        except Exception:
            log.debug("stop_listening failed (non-fatal)", exc_info=True)
    for p in players:
        if p is None:
            continue
        try:
            p.close()
        except Exception:
            log.debug("close failed (non-fatal)", exc_info=True)


async def run_jobs(jobs: Sequence[Callable[[], Awaitable]], *, workers: int,
                   stop_check: Optional[Callable[[], bool]] = None,
                   return_exceptions: bool = False) -> List:
    """Run independent async battle ``jobs`` (0-arg coroutine factories) with at most
    ``workers`` in flight at once — the dominant collection / eval throughput lever
    (docs sec 20). This is the shared concurrency seam the multiprocessing layer
    (task #14) partitions across processes.

    ``stop_check`` (a sync predicate) is polled AFTER a job acquires its concurrency
    slot but BEFORE it launches battles, so a soft stop (Ctrl-C / time budget) drains
    the QUEUED backlog cheaply instead of launching battles against a torn-down server
    (which would flood the log with ConnectionRefused). Each job owns its own
    try/finally (watchdog + teardown); ``return_exceptions`` forwards to ``gather``
    (default ``False`` = a job that raises propagates — the gauntlet's original
    semantics; ``True`` = collect per-job exceptions and keep going)."""
    sem = asyncio.Semaphore(max(1, int(workers)))

    async def _guard(job):
        async with sem:
            if stop_check is not None and stop_check():
                return None
            return await job()

    return await asyncio.gather(*[_guard(j) for j in jobs],
                                return_exceptions=return_exceptions)
