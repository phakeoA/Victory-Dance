"""Offline unit tests for the shared parallel-battle runner (task #13).

No server / poke-env: ``play_pairing`` / ``close_players`` / ``run_jobs`` only touch
``asyncio`` + duck-typed fake players, so the concurrency seam the multiprocessing
layer (#14) builds on is pinned down here. Mirrors the gauntlet's existing
``_play_chunk`` watchdog tests so the extraction is provably behaviour-preserving."""
from __future__ import annotations

import asyncio
import types as _t

from v_dance.play import parallel_battles as PB


# ── fake players ──────────────────────────────────────────────────────────────
class FakePlayer:
    """Minimal poke-env-player stand-in: win/finished counters, an async
    ``battle_against`` that advances them, an async ``ps_client.stop_listening``,
    and a ``close``."""

    def __init__(self, wr: float = 0.0):
        self.n_won_battles = 0
        self.n_finished_battles = 0
        self._wr = wr
        self.closed = False
        self.stopped = False

        async def _stop():
            self.stopped = True
        self.ps_client = _t.SimpleNamespace(stop_listening=_stop)

    async def battle_against(self, opp, n_battles):
        self.n_finished_battles += n_battles
        self.n_won_battles += int(round(self._wr * n_battles))

    def close(self):
        self.closed = True


# ── play_pairing ──────────────────────────────────────────────────────────────
def test_play_pairing_counts_win_delta():
    p = FakePlayer(wr=1.0)
    wins, fin = asyncio.run(PB.play_pairing(p, object(), 5, battle_timeout=None))
    assert (wins, fin) == (5, 5)


def test_play_pairing_returns_delta_not_absolute():
    """The (wins, finished) figure is the delta OVER THIS CHUNK — pre-existing
    counts on the player must not leak in (each pairing reads before/after)."""
    p = FakePlayer(wr=1.0)
    p.n_won_battles = 100          # stale counts from an earlier life
    p.n_finished_battles = 100
    wins, fin = asyncio.run(PB.play_pairing(p, object(), 3, battle_timeout=None))
    assert (wins, fin) == (3, 3)


def test_play_pairing_watchdog_abandons_hang():
    """A battle that never resolves must not hang the run: the watchdog fires and
    play_pairing returns promptly with nothing counted (the 3-hour-hang guard)."""
    class HangPlayer(FakePlayer):
        async def battle_against(self, opp, n_battles):
            await asyncio.sleep(30)        # never finishes within the timeout

    wins, fin = asyncio.run(PB.play_pairing(HangPlayer(), object(), 1, battle_timeout=0.05))
    assert (wins, fin) == (0, 0)


def test_play_pairing_partial_finish_before_hang_still_counts():
    """Battles that finished BEFORE the hang are kept — the watchdog only abandons
    the tail."""
    class PartialPlayer(FakePlayer):
        async def battle_against(self, opp, n_battles):
            self.n_finished_battles += 2   # two resolve immediately
            self.n_won_battles += 2
            await asyncio.sleep(30)        # then hang on the rest

    wins, fin = asyncio.run(PB.play_pairing(PartialPlayer(), object(), 5, battle_timeout=0.05))
    assert (wins, fin) == (2, 2)


# ── close_players ─────────────────────────────────────────────────────────────
def test_close_players_stops_and_closes_each():
    a, b = FakePlayer(), FakePlayer()
    asyncio.run(PB.close_players(a, b))
    assert a.stopped and a.closed and b.stopped and b.closed


def test_close_players_skips_none():
    a = FakePlayer()
    asyncio.run(PB.close_players(a, None))    # must not raise on a None slot
    assert a.stopped and a.closed


def test_close_players_tolerates_raising_teardown():
    """One player's failing teardown must not mask another's — a raising
    stop_listening / close is swallowed and the rest still close."""
    bad = FakePlayer()

    async def _boom():
        raise RuntimeError("ws already gone")
    bad.ps_client = _t.SimpleNamespace(stop_listening=_boom)
    good = FakePlayer()
    asyncio.run(PB.close_players(bad, good))   # no raise
    assert good.stopped and good.closed and bad.closed   # bad still got close()d


# ── run_jobs ──────────────────────────────────────────────────────────────────
def test_run_jobs_runs_every_job_and_collects_results():
    ran = []

    def make(i):
        async def job():
            ran.append(i)
            return i * 10
        return job

    results = asyncio.run(PB.run_jobs([make(i) for i in range(5)], workers=2))
    assert sorted(ran) == [0, 1, 2, 3, 4]
    assert sorted(results) == [0, 10, 20, 30, 40]


def test_run_jobs_bounds_concurrency_to_workers():
    """At most ``workers`` jobs run their body at once — the CPU-cap guarantee."""
    state = {"cur": 0, "peak": 0}

    def make():
        async def job():
            state["cur"] += 1
            state["peak"] = max(state["peak"], state["cur"])
            await asyncio.sleep(0.02)      # hold the slot so others would pile up
            state["cur"] -= 1
        return job

    asyncio.run(PB.run_jobs([make() for _ in range(8)], workers=3))
    assert state["peak"] <= 3              # never more than 3 in flight
    assert state["peak"] >= 2              # but DID overlap (proves real concurrency)


def test_run_jobs_stop_check_drops_queued_backlog():
    """A soft stop must abort the QUEUED jobs (so we don't launch battles at a dead
    server). With workers=1 the order is deterministic: the first job runs, flips the
    stop flag, and every queued job after it is skipped."""
    ran = []
    flag = {"stop": False}

    def make(i):
        async def job():
            ran.append(i)
            flag["stop"] = True            # after the first body runs, stop
        return job

    asyncio.run(PB.run_jobs([make(i) for i in range(6)], workers=1,
                            stop_check=lambda: flag["stop"]))
    assert ran == [0]                      # only the first executed; rest dropped


def test_run_jobs_stop_check_already_set_runs_nothing():
    ran = []

    def make(i):
        async def job():
            ran.append(i)
        return job

    asyncio.run(PB.run_jobs([make(i) for i in range(4)], workers=2,
                            stop_check=lambda: True))
    assert ran == []                       # stop set up front → no bodies run


def test_run_jobs_return_exceptions_keeps_going():
    """``return_exceptions=True`` captures a failing job's error and still runs the
    rest (robust overnight mode); the default re-raises (faithful gauntlet mode)."""
    def good():
        async def job():
            return "ok"
        return job

    def boom():
        async def job():
            raise ValueError("chunk blew up")
        return job

    results = asyncio.run(PB.run_jobs([good(), boom(), good()], workers=2,
                                      return_exceptions=True))
    assert results[0] == "ok" and results[2] == "ok"
    assert isinstance(results[1], ValueError)

    # default (return_exceptions=False) → the exception propagates
    import pytest
    with pytest.raises(ValueError):
        asyncio.run(PB.run_jobs([good(), boom()], workers=2))
