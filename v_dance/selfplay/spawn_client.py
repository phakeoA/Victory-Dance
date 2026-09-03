"""Spawn client — W2 throughput step 2 (2026-09-03, docs/ps_ppo_review_2026-09-02.md §4).

Drives a PAIR of self-play accounts against the server-side spawner (``rlspawn.ts``, step 1) instead
of the challenge protocol: both accounts log in, set their teams with ``/utm``, one of them asks the
server to keep ``rooms`` battles alive between them (``/rlautospawn``), and poke-env plays every room
the server opens (it creates a battle on each ``|init|``; the ``max_concurrent_battles`` queue must be
sized ABOVE the target — a full queue blocks the single listen loop, and the slot only frees on a
``|win|`` that the same loop must process → set 2·rooms+2). The existing ``SelfPlayVGCPlayer``
recorder + ``pair_trajectories`` work unchanged: one account per seat, trajectories keyed by tag.

Housekeeping mirrors ps-ppo's worker: every ``reconcile_s`` the client asks ``/rlactive`` (the
reply is intercepted before poke-env's dispatcher, which would only log it) and ``/rlrescue``s any
GHOST room (live on the server, unknown to poke-env — it holds no slot, so expiring it is safe);
a room whose turn clock stops for ``stall_s`` is FORFEITED from our seat (the server then sends a
real ``|win|`` to both accounts, both slots free, and ``_forfeited_tags`` makes the recorder seal it
FALLBACK → dropped, exactly the T3.1 rule). ``/rlrescue`` is never used on a room poke-env tracks:
an expired room sends no ``|win|`` and would leak the slot. When ``n`` pairs are in, ``/rlautooff``,
a short grace for the in-flight rooms, then the leftovers are forfeited (``abandon_server_state``).

Throughput (step 4's number) comes out of ``play_pairing_spawned``'s stats: games and model
decisions per minute for the pairing.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Set

log = logging.getLogger(__name__)


@dataclass
class SpawnConfig:
    rooms: int = 4                 # battles the server keeps alive between the pair
    poll_s: float = 1.0            # progress poll cadence
    reconcile_s: float = 30.0      # /rlactive cadence (ghost rooms → /rlrescue)
    stall_s: float = 90.0          # a room whose turn has not advanced for this long is forfeited
    battle_timeout: Optional[float] = 300.0   # NO pair finished for this long → abandon the chunk
    grace_s: float = 60.0          # after /rlautooff, wait this long for in-flight rooms to end
    login_s: float = 45.0


@dataclass
class SpawnStats:
    pairs: int = 0
    finished: Dict[str, int] = field(default_factory=dict)   # per account username
    decisions: int = 0
    stalls_forfeited: int = 0
    ghosts_rescued: int = 0
    abandoned: int = 0
    reconciles: int = 0
    elapsed_s: float = 0.0
    timed_out: bool = False
    status: Optional[dict] = None                              # the last /rlstatus JSON, if any

    @property
    def games_per_min(self) -> float:
        return 60.0 * self.pairs / self.elapsed_s if self.elapsed_s > 0 else 0.0

    @property
    def decisions_per_min(self) -> float:
        return 60.0 * self.decisions / self.elapsed_s if self.elapsed_s > 0 else 0.0

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["games_per_min"] = round(self.games_per_min, 2)
        d["decisions_per_min"] = round(self.decisions_per_min, 1)
        return d


def max_concurrent_for(rooms: int) -> int:
    """The poke-env battle-count queue must never fill while the spawner refills: the server can
    open a replacement inside the same tick the previous room ended, before the ``|win|`` freed its
    slot. 2·rooms + 2 leaves room for every ended-not-yet-freed battle."""
    return 2 * max(1, int(rooms)) + 2


# ── the poke-env bridge ────────────────────────────────────────────────────────
def _bridge():
    try:
        from poke_env.concurrency import handle_threaded_coroutines
        return handle_threaded_coroutines
    except Exception:
        return None


async def _run_on(player, coro):
    """Await ``coro`` on the player's own loop (POKE_LOOP) from any loop (#02-fix pattern)."""
    loop = getattr(getattr(player, "ps_client", None), "loop", None)
    b = _bridge()
    if b is not None and loop is not None:
        return await b(coro, loop)
    return await coro


async def send(player, message: str, room: str = ""):
    return await _run_on(player, player.ps_client.send_message(message, room))


def install_queryresponse_hook(player) -> None:
    """Intercept ``|queryresponse|rlactive|`` / ``|queryresponse|rlstatus|`` BEFORE poke-env's
    dispatcher (which only logs them as unhandled): parsed into ``player._rl_active`` (set of live
    room ids the server holds for this account) and ``player._rl_status`` (dict). Idempotent."""
    client = player.ps_client
    if getattr(client, "_rl_hooked", False):
        return
    base = client._handle_message
    player._rl_active = set()
    player._rl_active_at = 0.0
    player._rl_status = None

    async def _handle(message: str):
        if message.startswith("|queryresponse|rlactive|"):
            payload = message[len("|queryresponse|rlactive|"):].strip()
            player._rl_active = {t.strip() for t in payload.split(",") if t.strip()}
            player._rl_active_at = time.monotonic()
            return None
        if message.startswith("|queryresponse|rlstatus|"):
            import json
            try:
                player._rl_status = json.loads(message[len("|queryresponse|rlstatus|"):])
            except Exception:
                player._rl_status = {"raw": message}
            return None
        return await base(message)

    client._handle_message = _handle
    client._rl_hooked = True


# ── helpers over a player's view ───────────────────────────────────────────────
def _norm(tag) -> str:
    return (str(tag) or "").strip().lstrip(">").strip()


def unfinished_tags(player) -> Set[str]:
    try:
        return {_norm(t) for t, b in dict(getattr(player, "battles", {}) or {}).items()
                if b is not None and not getattr(b, "finished", True)}
    except Exception:
        return set()


def battle_turns(player) -> Dict[str, int]:
    out = {}
    try:
        for t, b in dict(getattr(player, "battles", {}) or {}).items():
            if b is not None and not getattr(b, "finished", True):
                out[_norm(t)] = int(getattr(b, "turn", 0) or 0)
    except Exception:
        pass
    return out


def finished_pairs(p1, p2) -> int:
    f1 = getattr(p1, "_finished", None)
    f2 = getattr(p2, "_finished", None)
    if not isinstance(f1, dict) or not isinstance(f2, dict):
        return 0
    return sum(1 for t in f1 if t in f2)


def decisions_recorded(*players) -> int:
    n = 0
    for p in players:
        f = getattr(p, "_finished", None)
        if isinstance(f, dict):
            for tr in f.values():
                n += len(getattr(tr, "transitions", ()) or ())
    return n


def _tag_forfeited(player, tag: str) -> None:
    ff = getattr(player, "_forfeited_tags", None)
    if ff is None:
        ff = set()
        try:
            player._forfeited_tags = ff
        except Exception:
            return
    ff.add(_norm(tag))


async def forfeit_room(p1, p2, tag: str) -> None:
    """Forfeit ``tag`` from p1's seat: the server ends the room with a real ``|win|`` for BOTH
    accounts (slots free), and both recorders seal it FALLBACK (dropped)."""
    _tag_forfeited(p1, tag)
    _tag_forfeited(p2, tag)
    await send(p1, "/forfeit", tag)


async def reconcile(p1, p2, stats: SpawnStats, *, sleep=asyncio.sleep, reply_s: float = 0.5) -> int:
    """``/rlactive`` for p1; rescue the GHOST rooms (server-live, unknown to poke-env). Returns
    the ghost count. Rooms poke-env tracks are never rescued (see the module docstring)."""
    await send(p1, "/rlactive")
    await sleep(reply_s)                           # the reply lands on the listen loop
    active = set(getattr(p1, "_rl_active", set()) or set())
    stats.reconciles += 1
    if not active:
        return 0
    known = set()
    for p in (p1, p2):
        try:
            known |= {_norm(t) for t in dict(getattr(p, "battles", {}) or {})}
        except Exception:
            pass
    ghosts = [t for t in active if _norm(t) not in known]
    for t in ghosts:
        await send(p1, f"/rlrescue {t}")
    stats.ghosts_rescued += len(ghosts)
    return len(ghosts)


# ── the pairing driver ─────────────────────────────────────────────────────────
async def play_pairing_spawned(p1, p2, n: int, *, fmt: str, cfg: SpawnConfig = SpawnConfig(),
                               label: str = "spawn", now: Callable[[], float] = time.monotonic,
                               sleep=asyncio.sleep, wait_login: bool = True) -> SpawnStats:
    """Collect ``n`` finished pairs between ``p1`` and ``p2`` through the server-side spawner.
    Returns the throughput + housekeeping stats; the trajectories are on the players
    (``finished_trajectories``) exactly as after ``play_pairing``."""
    stats = SpawnStats()
    t0 = now()
    for p in (p1, p2):
        install_queryresponse_hook(p)
    if wait_login:
        for p in (p1, p2):
            await _run_on(p, asyncio.wait_for(p.ps_client.logged_in.wait(), cfg.login_s))
    for p in (p1, p2):
        await send(p, "/utm " + p._team.yield_team())
    spawn_args = f"{p1.username}, {p2.username}, {fmt}, {int(cfg.rooms)}"
    await send(p1, f"/rlautospawn {spawn_args}")
    log.info("[%s] /rlautospawn %s", label, spawn_args)

    last_turn: Dict[str, int] = {}
    last_change: Dict[str, float] = {}
    last_pairs = 0
    last_progress = now()
    next_reconcile = now() + cfg.reconcile_s
    target = int(cfg.rooms)
    try:
        while True:
            await sleep(cfg.poll_s)
            pairs = finished_pairs(p1, p2)
            if pairs > last_pairs:
                last_pairs, last_progress = pairs, now()
            if pairs >= n:
                break
            # taper: never keep more rooms alive than games still wanted (the plugin updates a
            # running scheduler's target in place) — bounds the over-collection to the rooms in flight
            remaining = max(1, n - pairs)
            if remaining < target:
                target = remaining
                await send(p1, f"/rlautospawn {p1.username}, {p2.username}, {fmt}, {target}")
            # stall watchdog per room: the turn clock stopped → forfeit from our seat
            turns = battle_turns(p1)
            t = now()
            for tag, turn in turns.items():
                if last_turn.get(tag) != turn:
                    last_turn[tag], last_change[tag] = turn, t
                elif t - last_change.get(tag, t) > cfg.stall_s:
                    log.warning("[%s] room %s stalled at turn %d for %.0fs — forfeiting", label, tag, turn,
                                t - last_change[tag])
                    await forfeit_room(p1, p2, tag)
                    stats.stalls_forfeited += 1
                    last_change[tag] = t                # one forfeit per stall window
            for tag in list(last_turn):
                if tag not in turns:
                    last_turn.pop(tag, None)
                    last_change.pop(tag, None)
            if t >= next_reconcile:
                try:
                    await reconcile(p1, p2, stats, sleep=sleep, reply_s=min(0.5, cfg.poll_s or 0.5))
                except Exception:
                    log.debug("reconcile failed (non-fatal)", exc_info=True)
                next_reconcile = t + cfg.reconcile_s
            if cfg.battle_timeout and cfg.battle_timeout > 0 and t - last_progress > cfg.battle_timeout:
                log.warning("[%s] no pair finished for %.0fs — abandoning the chunk", label, t - last_progress)
                stats.timed_out = True
                break
    finally:
        try:
            await send(p1, f"/rlautooff {spawn_args}")
        except Exception:
            log.debug("rlautooff failed (non-fatal)", exc_info=True)
        # grace for the in-flight rooms, then forfeit the leftovers (FALLBACK → dropped)
        deadline = now() + cfg.grace_s
        while now() < deadline and (unfinished_tags(p1) or unfinished_tags(p2)):
            await sleep(cfg.poll_s)
        leftovers = unfinished_tags(p1) | unfinished_tags(p2)
        if leftovers:
            try:
                from v_dance.play.parallel_battles import abandon_server_state
                abandoned = await abandon_server_state(p1, p2)
                stats.abandoned = len(abandoned) or len(leftovers)
            except Exception:
                log.debug("abandon failed (non-fatal)", exc_info=True)
                stats.abandoned = len(leftovers)
        try:
            await send(p1, "/rlstatus")
            await sleep(min(cfg.poll_s, 0.5))
            stats.status = getattr(p1, "_rl_status", None)
        except Exception:
            pass
        stats.pairs = finished_pairs(p1, p2)
        stats.finished = {str(getattr(p, "username", i)): int(getattr(p, "n_finished_battles", 0) or 0)
                          for i, p in enumerate((p1, p2))}
        stats.decisions = decisions_recorded(p1, p2)
        stats.elapsed_s = max(1e-9, now() - t0)
        log.info("[%s] spawn pairing: %d pairs, %.1f games/min, %.0f decisions/min, stalls %d, ghosts %d, "
                 "abandoned %d, elapsed %.0fs", label, stats.pairs, stats.games_per_min,
                 stats.decisions_per_min, stats.stalls_forfeited, stats.ghosts_rescued, stats.abandoned,
                 stats.elapsed_s)
    return stats
