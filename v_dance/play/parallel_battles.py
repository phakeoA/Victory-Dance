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


# ── battle account naming (22d) ────────────────────────────────────────────────
# The shared Showdown server keeps per-username challenge/battle state, but every collection /
# eval path resets its chunk ``uid`` to 0 each generation — so without a per-gen salt the account
# names (``BC1`` / ``OPprev1`` / ``LG0x1`` …) REPEAT across gens, and a stale server-side challenge
# left by one gen (an abandoned chunk, a slow handshake) collides with the next gen's reuse →
# ``There's already a challenge between you and OPprev…``. Folding the generation into the name
# makes every (gen, chunk) pair unique. These helpers are the ONE source of truth for the names
# (used by mp_collect / mp_eval / gauntlet / collect_with_league) so the scheme can't drift.
def gen_salt(generation) -> str:
    """A toID-safe per-generation salt for battle account names (22d).

    Returns ``str(int(generation))`` (digits survive Showdown's ``toID`` id-normalisation, which
    strips everything outside ``[a-z0-9]``); ``""`` when ``generation is None`` so the standalone
    gauntlet CLI / offline tests keep the legacy un-salted names."""
    return "" if generation is None else str(int(generation))


def _salted_uid(uid, salt: str) -> str:
    # ``<salt>x<uid>`` when salted, bare ``<uid>`` otherwise. 'x' is a toID-safe delimiter
    # (Showdown strips '_', so a ``gen_uid`` split would be ambiguous after normalisation); with
    # 'x' the (gen, uid) runs stay distinct and the pair is recoverable from the id.
    return f"{salt}x{uid}" if salt else f"{uid}"


def eval_account_names(kind: str, uid, *, salt: str = "") -> Tuple[str, str]:
    """The ``(model, opponent)`` Showdown account names for one EVAL chunk (22d).

    Keyed to the generation ``salt`` so names never repeat across gens. Stays under Showdown's
    18-char username limit (``OP`` + kind≤4 + gen≤5 + ``x`` + uid≤4 ≈ 16). ``salt=""`` reproduces
    the legacy ``BC{uid}`` / ``OP{kind4}{uid}`` names exactly (backward compatible)."""
    s = _salted_uid(uid, salt)
    return f"BC{s}", f"OP{kind[:4]}{s}"


def collect_account_names(kind: str, uid, *, salt: str = "", opp_ref=None) -> Tuple[str, str]:
    """The ``(our, opponent)`` Showdown account names for one COLLECTION chunk (22d).

    ``our`` is the model recorder (seat 0); the opponent name depends on the league kind
    (``latest`` ⇒ the seat-1 recorder, ``snapshot`` ⇒ a frozen-ckpt player, else a scripted
    anchor tagged by its kind prefix). Keyed to the generation ``salt`` (see ``gen_salt``);
    ``salt=""`` reproduces the legacy ``LG0x{uid}`` / ``LG1x{uid}`` / ``LGsx{uid}`` /
    ``LGc{ref3}{uid}`` names exactly."""
    s = _salted_uid(uid, salt)
    our = f"LG0x{s}"
    if kind == "latest":
        opp = f"LG1x{s}"
    elif kind == "snapshot":
        opp = f"LGsx{s}"
    else:                                       # scripted anchor
        opp = f"LGc{(opp_ref or '')[:3]}{s}"
    return our, opp


def _to_id(name) -> str:
    """Showdown user-id normalisation (lowercased, alphanumerics only) — poke-env's ``to_id_str``
    when importable, else a pure fallback so this module stays poke-env-free for offline tests."""
    try:
        from poke_env.data import to_id_str
        return to_id_str(name)
    except Exception:
        return "".join(ch for ch in str(name).lower() if ch.isalnum())


async def abandon_server_state(model_player, opponent) -> None:
    """Best-effort cleanup of SERVER-side state for an ABANDONED chunk (22e).

    When ``play_pairing``'s stall watchdog gives up on a chunk, the battles still in flight and
    any challenge that was sent-but-never-accepted are left HUNG on the shared Showdown server.
    Dropping our websocket (``close_players``) clears OUR view, but the server keeps those battle
    rooms + the stale challenge for a while — state that accumulates over a long run (the #22c
    server bloat) and, before 22d's per-gen names, produced ``already a challenge between you and
    …`` on the next reuse. So FORFEIT every unfinished battle each player holds (ends the room
    server-side) and CANCEL the model's outgoing challenge, reclaiming that state immediately.

    Every step is guarded — this runs on the failure path and must never raise. If the socket is
    already dead (the stall WAS a dropped connection), the sends fail harmlessly and teardown
    proceeds. Duck-typed (``battles`` / ``ps_client.send_message`` / ``username``) so the module
    stays import-light and the cleanup unit-tests offline against fake players."""
    # 1) forfeit each unfinished battle in its room (the server ends the room → reclaims it).
    abandoned = 0   # distinct unfinished battles we abandon-by-forfeit (counted from OUR view only)
    for p in (model_player, opponent):
        if p is None:
            continue
        try:
            battles = dict(getattr(p, "battles", {}) or {})
        except Exception:
            battles = {}
        for tag, battle in battles.items():
            try:
                if battle is not None and not getattr(battle, "finished", True):
                    if p is model_player:
                        abandoned += 1          # count once per battle (model's view), not per side
                    await p.ps_client.send_message("/forfeit", tag)
            except Exception:
                log.debug("forfeit of %s failed (non-fatal)", tag, exc_info=True)
    # fs-monitor: record the watchdog/abandon forfeits on the model player's source_counts so they
    # flow through the existing per-gen aggregation (a runner-issued /forfeit never touches the
    # player's choose_move, so without this the stalled-chunk forfeits would be invisible).
    if abandoned:
        sc = getattr(model_player, "_source_counts", None)
        if sc is not None:
            try:
                sc["abandon_forfeit"] += abandoned
            except Exception:
                log.debug("abandon_forfeit tally failed (non-fatal)", exc_info=True)
    # 2) cancel the model's outgoing challenge to the opponent (sent but never accepted).
    if model_player is not None and opponent is not None:
        try:
            await model_player.ps_client.send_message(
                f"/cancelchallenge {_to_id(getattr(opponent, 'username', ''))}")
        except Exception:
            log.debug("cancelchallenge failed (non-fatal)", exc_info=True)


async def play_pairing(model_player, opponent, n: int, *,
                       battle_timeout: Optional[float] = DEFAULT_BATTLE_TIMEOUT,
                       label: str = "", poll_interval: float = 5.0) -> Tuple[int, int]:
    """Play ``n`` battles ``model_player`` vs ``opponent`` and return the
    ``(model_wins, model_finished)`` DELTA over this chunk (read from the player's
    own counters, so the figure is meaningful even when the caller tallies
    trajectories instead).

    ``battle_timeout`` is a **STALL** watchdog (seconds): the chunk is abandoned only when
    NO battle has FINISHED for ``battle_timeout`` seconds — a true hang (a forced-switch /
    illusion desync that never resolves, or a websocket the server stopped servicing under
    load, #22). This unblocks the worker ~``battle_timeout`` after a battle wedges instead of
    after ``battle_timeout * n`` (a big eval chunk used to freeze a worker for minutes — the
    overnight deadlock), while NEVER falsely abandoning a chunk that's slow but still
    completing battles. Battles that finished before the stall still count.
    ``battle_timeout`` <= 0 / ``None`` disables the watchdog (await to completion)."""
    won_before = model_player.n_won_battles
    fin_before = model_player.n_finished_battles
    task = asyncio.ensure_future(model_player.battle_against(opponent, n_battles=n))

    if battle_timeout and battle_timeout > 0:
        loop = asyncio.get_running_loop()
        step = min(float(poll_interval), float(battle_timeout)) or float(battle_timeout)
        last_fin = fin_before
        last_progress = loop.time()
        while True:
            done, _ = await asyncio.wait({task}, timeout=step)
            if done:                                        # battle_against returned or raised
                break
            fin_now = model_player.n_finished_battles
            if fin_now > last_fin:                          # a battle FINISHED -> reset stall clock
                last_fin = fin_now
                last_progress = loop.time()
            elif (loop.time() - last_progress) >= battle_timeout:   # no progress for too long = hang
                task.cancel()
                fin = model_player.n_finished_battles - fin_before
                lab = f" {label}" if label else ""
                log.warning("battle pairing STALL watchdog fired (no battle finished in %.0fs; "
                            "%d/%d done%s) — abandoning chunk, continuing.",
                            battle_timeout, fin, n, lab)
                try:
                    await task                              # let the cancellation settle
                except asyncio.CancelledError:
                    pass
                except Exception:                           # battle_against raised on the way out
                    log.debug("abandoned battle_against raised on cancel (non-fatal)", exc_info=True)
                # 22e: reclaim the hung battle rooms + stale challenge SERVER-side before the
                # caller tears the sockets down — otherwise they linger and bloat the shared
                # server (and, pre-22d, collided with the next gen's reused names).
                await abandon_server_state(model_player, opponent)
                break
        # RETRIEVE a completed task's exception so it isn't logged as "Task exception was never
        # retrieved" (the scary-looking but harmless dropped-connection traceback in the log).
        if task.done() and not task.cancelled():
            exc = task.exception()
            if exc is not None:
                lab = f" {label}" if label else ""
                log.warning("battle pairing errored%s: %r — continuing.", lab, exc)
    else:
        await task

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
