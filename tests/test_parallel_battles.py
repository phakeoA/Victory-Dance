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


def test_play_pairing_stall_watchdog_allows_slow_but_progressing():
    """#22: the STALL watchdog must NOT abandon a chunk that's slow overall but steadily
    finishing battles — only a true no-progress hang is abandoned. Here total time (5*0.02s)
    exceeds battle_timeout, yet every battle finishes within the stall window, so all count."""
    class SteadyPlayer(FakePlayer):
        async def battle_against(self, opp, n_battles):
            for _ in range(n_battles):
                await asyncio.sleep(0.02)
                self.n_finished_battles += 1
                self.n_won_battles += 1

    wins, fin = asyncio.run(PB.play_pairing(SteadyPlayer(), object(), 5,
                                            battle_timeout=0.05, poll_interval=0.01))
    assert (wins, fin) == (5, 5)


def test_play_pairing_stall_abandons_fast_regardless_of_chunk_size():
    """#22 (the overnight deadlock fix): a hang is abandoned ~battle_timeout after it stalls,
    NOT battle_timeout*n — so a big eval chunk can't freeze a worker for minutes."""
    import time

    class HangPlayer(FakePlayer):
        async def battle_against(self, opp, n_battles):
            await asyncio.sleep(60)            # never finishes

    t0 = time.monotonic()
    wins, fin = asyncio.run(PB.play_pairing(HangPlayer(), object(), 100,
                                            battle_timeout=0.05, poll_interval=0.01))
    elapsed = time.monotonic() - t0
    assert (wins, fin) == (0, 0)
    assert elapsed < 1.0                        # ~0.05s, NOT 0.05*100=5s (the old per-chunk budget)


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


# ── account naming (22d) ──────────────────────────────────────────────────────
def test_gen_salt_formats():
    assert PB.gen_salt(None) == ""        # un-salted (standalone CLI / tests)
    assert PB.gen_salt(0) == "0"
    assert PB.gen_salt(54) == "54"


def test_eval_account_names_unsalted_is_legacy():
    """salt='' must reproduce the pre-22d names EXACTLY (no behaviour change off the live loop)."""
    assert PB.eval_account_names("prev_best", 3) == ("BC3", "OPprev3")
    assert PB.eval_account_names("max_damage", 7) == ("BC7", "OPmax_7")


def test_collect_account_names_unsalted_is_legacy():
    assert PB.collect_account_names("latest", 5) == ("LG0x5", "LG1x5")
    assert PB.collect_account_names("snapshot", 5)[1] == "LGsx5"
    assert PB.collect_account_names("scripted", 5, opp_ref="max_damage")[1] == "LGcmax5"


def test_account_names_unique_and_short_across_generations():
    """22d core: across the whole run no two distinct chunks share an account id (after Showdown's
    toID normalisation, which strips '_'), so a stale server-side challenge from one gen can't
    collide with the next gen's reuse — and every name stays under the 18-char limit. Faithfully
    mirrors the builders: a MONOTONIC uid counter per phase (eval / collection) per gen, exactly
    as ``build_eval_specs`` / ``build_chunk_specs`` assign them."""
    seen = set()

    def _check(name):
        assert len(name) <= 18, f"{name!r} exceeds Showdown's 18-char limit"
        nid = PB._to_id(name)
        assert nid not in seen, f"account-id collision: {name!r} -> {nid!r}"
        seen.add(nid)

    for gen in range(120):                 # well past a real overnight run's gen count
        salt = PB.gen_salt(gen)
        uid = 0                            # eval phase: one counter across all (kind, matchup) chunks
        for kind in ("random", "max_damage", "heuristic", "prev_best"):
            for _ in range(15):
                uid += 1
                for name in PB.eval_account_names(kind, uid, salt=salt):
                    _check(name)
        uid = 0                            # collection phase: its OWN counter (separate LG… prefix)
        for kind, ref in (("latest", None), ("snapshot", None), ("scripted", "heuristic")):
            for _ in range(20):
                uid += 1
                for name in PB.collect_account_names(kind, uid, salt=salt, opp_ref=ref):
                    _check(name)


def test_hof_salt_distinct_from_main_eval_salt():
    """22d: the HoF gauntlet (salt='Nh M') must NOT collide with the same-gen prev_best mirror
    (salt=str(N)) — both use kind='prev_best' and the same uids, so only the salt keeps them apart."""
    main = set(PB.eval_account_names("prev_best", 1, salt=PB.gen_salt(23)))
    hof = set(PB.eval_account_names("prev_best", 1, salt="23h11"))
    assert main.isdisjoint(hof)
    # the HoF salt ("NhM") is the longest form — even at big gens it must stay under 18 chars.
    for n in PB.eval_account_names("prev_best", 99, salt="9999h9999"):
        assert len(n) <= 18, f"{n!r} exceeds Showdown's 18-char limit"


# ── abandon_server_state (22e) ────────────────────────────────────────────────
class _FakeBattle:
    def __init__(self, finished):
        self.finished = finished


class _CleanupPlayer:
    """Fake poke-env player exposing just what ``abandon_server_state`` duck-types: a ``battles``
    dict, an async ``ps_client.send_message(message, room)`` that records its sends, and a
    ``username``. ``boom=True`` simulates a dead socket (every send raises)."""
    def __init__(self, username, battles=None, boom=False):
        self.username = username
        self._battles = battles or {}
        self.sent = []

        async def _send(message, room="", message_2=None):
            if boom:
                raise RuntimeError("socket already dead")
            self.sent.append((message, room))
        self.ps_client = _t.SimpleNamespace(send_message=_send)

    @property
    def battles(self):
        return self._battles


def test_abandon_server_state_forfeits_unfinished_and_cancels_challenge():
    model = _CleanupPlayer("BC5x3", battles={
        "battle-x-1": _FakeBattle(finished=False),    # still live -> forfeit it
        "battle-x-2": _FakeBattle(finished=True),     # already done -> leave alone
    })
    opp = _CleanupPlayer("OPprev5x3", battles={"battle-x-1": _FakeBattle(finished=False)})
    asyncio.run(PB.abandon_server_state(model, opp))
    assert ("/forfeit", "battle-x-1") in model.sent          # unfinished room forfeited
    assert all(room != "battle-x-2" for _m, room in model.sent)   # finished room untouched
    assert ("/forfeit", "battle-x-1") in opp.sent
    # the model cancels its outgoing challenge to the opponent (toID-normalised id, no '_'/case)
    assert ("/cancelchallenge opprev5x3", "") in model.sent


def test_abandon_server_state_guards_dead_socket_and_none():
    boom = _CleanupPlayer("BC1", battles={"b": _FakeBattle(False)}, boom=True)
    asyncio.run(PB.abandon_server_state(boom, _CleanupPlayer("OP1", boom=True)))   # must not raise
    asyncio.run(PB.abandon_server_state(None, None))                               # must not raise


def test_abandon_server_state_counts_abandon_forfeit_once_per_battle():
    # fs-monitor #2: a watchdog/abandon /forfeit never goes through the player's choose_move, so
    # abandon_server_state records it on the model's _source_counts (which the gen aggregation reads).
    from collections import Counter
    model = _CleanupPlayer("BC5x3", battles={
        "battle-x-1": _FakeBattle(finished=False),    # abandoned -> counted
        "battle-x-2": _FakeBattle(finished=False),    # abandoned -> counted
        "battle-x-3": _FakeBattle(finished=True),     # already done -> NOT counted
    })
    model._source_counts = Counter()
    opp = _CleanupPlayer("OP", battles={"battle-x-1": _FakeBattle(finished=False)})
    opp._source_counts = Counter()
    asyncio.run(PB.abandon_server_state(model, opp))
    assert model._source_counts["abandon_forfeit"] == 2           # 2 unfinished, counted from OUR view
    assert opp._source_counts.get("abandon_forfeit", 0) == 0      # the shared battle is not double-counted


def test_abandon_server_state_no_counter_attr_is_safe():
    # the fake without _source_counts must not crash (guarded getattr)
    model = _CleanupPlayer("BC", battles={"b": _FakeBattle(finished=False)})
    asyncio.run(PB.abandon_server_state(model, None))             # must not raise


def test_play_pairing_stall_cleans_up_server_state():
    """22e end-to-end: the stall watchdog must trigger the forfeit + cancelchallenge cleanup before
    the caller tears the sockets down (else the hung rooms/challenge bloat the shared server)."""
    sent = []

    class HangWithBattle(FakePlayer):
        def __init__(self):
            super().__init__()
            self.username = "BC9x1"
            self._battles = {"battle-z": _FakeBattle(finished=False)}

            async def _send(message, room="", message_2=None):
                sent.append((message, room))
            self.ps_client = _t.SimpleNamespace(stop_listening=self.ps_client.stop_listening,
                                                send_message=_send)

        @property
        def battles(self):
            return self._battles

        async def battle_against(self, opp, n_battles):
            await asyncio.sleep(60)                 # never finishes -> stall watchdog fires

    opp = _CleanupPlayer("OPprev9x1")
    wins, fin = asyncio.run(PB.play_pairing(HangWithBattle(), opp, 1,
                                            battle_timeout=0.05, poll_interval=0.01))
    # T4.4: the abandoned (forfeited) hung battle is folded into the FINISHED delta as a non-win, so the
    # stalled chunk doesn't silently shrink the eval denominator. 1 battle abandoned → (wins=0, fin=1).
    assert (wins, fin) == (0, 1)
    assert ("/forfeit", "battle-z") in sent                  # the hung room was forfeited
    assert ("/cancelchallenge opprev9x1", "") in sent        # the stale challenge was cancelled


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
