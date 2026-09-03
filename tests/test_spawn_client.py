"""W2 throughput step 2 (2026-09-03) — the spawn client (`v_dance/selfplay/spawn_client.py`).

Offline, with fakes: the queryresponse hook (rlactive / rlstatus intercepted before poke-env's
dispatcher, everything else passed through), the pairing driver's command order (`/utm` for BOTH
accounts before `/rlautospawn`, `/rlautooff` + `/rlstatus` at the end), the stop-at-n rule, the
per-room stall forfeit (our seat, both `_forfeited_tags`, once per stall window), the ghost-only
rescue in reconciliation, the no-progress timeout with leftover forfeits, and the throughput stats.
LIVE (opt-in `VD_SPAWN_LIVE_TEST=1`): the real `run_self_play_games(..., spawn_rooms=4)` with the
served era-4 2b checkpoint on a local server — the first throughput number of the stack.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from v_dance.selfplay import spawn_client as SC

FMT = "gen9championsvgc2026regmb"


# ── fakes ──────────────────────────────────────────────────────────────────────
class FakeClient:
    loop = None                                     # → the direct-await bridge path

    def __init__(self):
        self.sent = []
        self.handled = []
        self.logged_in = asyncio.Event()
        self.logged_in.set()

    async def send_message(self, message, room=""):
        self.sent.append((room, message))

    async def _handle_message(self, message):
        self.handled.append(message)


class FakeBattle:
    def __init__(self, turn=0, finished=False):
        self.turn, self.finished = turn, finished


class FakePlayer:
    def __init__(self, name, packed="PACKED"):
        self.username = name
        self.ps_client = FakeClient()
        self._team = SimpleNamespace(yield_team=lambda: packed)
        self.battles = {}
        self._finished = {}
        self.n_finished_battles = 0

    def finish(self, tag, n_steps=3):
        self.battles[tag] = FakeBattle(turn=9, finished=True)
        self._finished[tag] = SimpleNamespace(transitions=[0] * n_steps)
        self.n_finished_battles += 1


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _sent(p, needle):
    return [(r, m) for r, m in p.ps_client.sent if needle in m]


# ── the hook ───────────────────────────────────────────────────────────────────
def test_hook_intercepts_rlactive_and_rlstatus_and_passes_the_rest_through():
    p = FakePlayer("A")
    SC.install_queryresponse_hook(p)
    SC.install_queryresponse_hook(p)                # idempotent

    async def main():
        await p.ps_client._handle_message("|queryresponse|rlactive|battle-x-1, battle-x-2,")
        await p.ps_client._handle_message('|queryresponse|rlstatus|{"schedulers":[],"rooms":3}')
        await p.ps_client._handle_message("|updateuser| A|1|...")
        await p.ps_client._handle_message("|queryresponse|rlstatus|not json")

    asyncio.run(main())
    assert p._rl_active == {"battle-x-1", "battle-x-2"} and p._rl_active_at > 0
    assert p._rl_status == {"raw": "|queryresponse|rlstatus|not json"}
    assert p.ps_client.handled == ["|updateuser| A|1|..."]   # only the non-plugin message reached poke-env


def test_max_concurrent_sits_above_the_target():
    assert SC.max_concurrent_for(4) == 10 and SC.max_concurrent_for(0) == 4 and SC.max_concurrent_for(1) == 4


# ── the pairing driver ────────────────────────────────────────────────────────
def _world(finish_per_tick=1, steps=4):
    p1, p2 = FakePlayer("SP1x1", "P1PACK"), FakePlayer("SP2x1", "P2PACK")
    clock = Clock()
    state = {"tick": 0}

    async def sleep(_s):
        clock.t += 1.0
        state["tick"] += 1
        for _ in range(finish_per_tick):            # the server finishes games as time passes
            tag = f"battle-{FMT}-{state['tick']}{_}"
            p1.finish(tag, steps)
            p2.finish(tag, steps)

    return p1, p2, clock, sleep


def test_pairing_sets_teams_first_then_spawns_and_stops_at_n():
    p1, p2, clock, sleep = _world()
    cfg = SC.SpawnConfig(rooms=2, poll_s=0.0, reconcile_s=1e9, grace_s=0.0, battle_timeout=None)
    st = asyncio.run(SC.play_pairing_spawned(p1, p2, 3, fmt=FMT, cfg=cfg, now=clock, sleep=sleep,
                                             wait_login=False))
    m1 = [m for _, m in p1.ps_client.sent]
    m2 = [m for _, m in p2.ps_client.sent]
    assert m2 == ["/utm P2PACK"]                                     # the partner only sets its team
    assert m1[0] == "/utm P1PACK"
    assert m1[1] == f"/rlautospawn SP1x1, SP2x1, {FMT}, 2"           # after BOTH teams are set
    assert f"/rlautospawn SP1x1, SP2x1, {FMT}, 1" in m1              # taper: 1 game left → 1 room
    assert m1[-2] == f"/rlautooff SP1x1, SP2x1, {FMT}, 2" and m1[-1] == "/rlstatus"
    # ≥ n: rooms in flight when the scheduler stops still finish and are kept (real behaviour)
    assert st.pairs >= 3 and st.finished == {"SP1x1": st.pairs, "SP2x1": st.pairs}
    assert st.decisions == st.pairs * 2 * 4 and st.elapsed_s > 0 and st.games_per_min > 0
    assert st.stalls_forfeited == 0 and st.abandoned == 0 and not st.timed_out
    assert st.as_dict()["decisions_per_min"] > 0


def test_stalled_room_is_forfeited_from_our_seat_once_per_window():
    p1, p2 = FakePlayer("A"), FakePlayer("B")
    clock = Clock()
    p1.battles["battle-x-7"] = FakeBattle(turn=5)                     # a live room whose turn never moves
    p2.battles["battle-x-7"] = FakeBattle(turn=5)
    ticks = {"n": 0}

    async def sleep(_s):
        clock.t += 10.0
        ticks["n"] += 1
        if ticks["n"] == 6:                                           # someone finally finishes a pair
            p1.finish("battle-x-8"); p2.finish("battle-x-8")

    cfg = SC.SpawnConfig(rooms=1, poll_s=0.0, reconcile_s=1e9, grace_s=0.0, stall_s=25.0, battle_timeout=None)
    st = asyncio.run(SC.play_pairing_spawned(p1, p2, 1, fmt=FMT, cfg=cfg, now=clock, sleep=sleep,
                                             wait_login=False))
    forfeits = _sent(p1, "/forfeit")
    assert forfeits and forfeits[0] == ("battle-x-7", "/forfeit")     # in the room, from our seat
    assert "battle-x-7" in p1._forfeited_tags and "battle-x-7" in p2._forfeited_tags
    assert st.stalls_forfeited >= 1 and st.pairs == 1
    # the leftover unfinished room is abandoned at the end (grace 0) — still one forfeit per window
    assert st.abandoned >= 1


def test_reconcile_rescues_ghost_rooms_only():
    p1, p2 = FakePlayer("A"), FakePlayer("B")
    SC.install_queryresponse_hook(p1)
    p1.battles["battle-x-1"] = FakeBattle(turn=2)                     # known to poke-env
    p1._rl_active = {"battle-x-1", "battle-x-2"}                      # the server also holds a ghost
    stats = SC.SpawnStats()

    async def sleep(_s):
        pass

    n = asyncio.run(SC.reconcile(p1, p2, stats, sleep=sleep))
    assert n == 1 and stats.ghosts_rescued == 1 and stats.reconciles == 1
    assert _sent(p1, "/rlactive") and _sent(p1, "/rlrescue battle-x-2") and not _sent(p1, "/rlrescue battle-x-1")


def test_no_progress_timeout_abandons_and_forfeits_leftovers():
    p1, p2 = FakePlayer("A"), FakePlayer("B")
    clock = Clock()
    p1.battles["battle-x-3"] = FakeBattle(turn=1)
    p2.battles["battle-x-3"] = FakeBattle(turn=1)

    async def sleep(_s):
        clock.t += 100.0                                              # nothing ever finishes

    cfg = SC.SpawnConfig(rooms=2, poll_s=0.0, reconcile_s=1e9, grace_s=0.0, stall_s=1e9, battle_timeout=250.0)
    st = asyncio.run(SC.play_pairing_spawned(p1, p2, 5, fmt=FMT, cfg=cfg, now=clock, sleep=sleep,
                                             wait_login=False))
    assert st.timed_out and st.pairs == 0 and st.abandoned >= 1
    assert _sent(p1, "/rlautooff") and _sent(p1, "/forfeit")           # leftovers reclaimed server-side
    assert "battle-x-3" in getattr(p1, "_forfeited_tags", set())


def test_helpers_on_a_players_view():
    p = FakePlayer("A")
    p.battles = {">battle-x-1": FakeBattle(turn=3), "battle-x-2": FakeBattle(turn=1, finished=True)}
    assert SC.unfinished_tags(p) == {"battle-x-1"} and SC.battle_turns(p) == {"battle-x-1": 3}
    q = FakePlayer("B")
    p.finish("t1", 2); q.finish("t1", 5); p.finish("t2", 1)
    assert SC.finished_pairs(p, q) == 1 and SC.decisions_recorded(p, q) == 8


# ── live: the first throughput number of the stack ────────────────────────────
@pytest.mark.skipif(os.environ.get("VD_SPAWN_LIVE_TEST") != "1",
                    reason="opt-in: VD_SPAWN_LIVE_TEST=1 starts a local Showdown server and plays real games")
def test_live_spawned_self_play_collects_pairs_and_reports_throughput(tmp_path: Path):
    pytest.importorskip("torch")
    pytest.importorskip("poke_env")
    from v_dance.selfplay.actor_critic import ActorCritic
    from v_dance.selfplay.game_runner import run_self_play_games
    repo = Path(__file__).resolve().parents[1]
    ckpt = repo / "ai_train_scripts" / "BC_model" / "checkpoints_attn_era4_2b" / "battle_base.pt"
    if not ckpt.is_file():
        pytest.skip("served checkpoint not present")
    ac = ActorCritic.from_bc_checkpoint(ckpt)
    rooms = int(os.environ.get("VD_SPAWN_LIVE_ROOMS", "4"))
    n = int(os.environ.get("VD_SPAWN_LIVE_GAMES", "6"))
    import time
    t0 = time.monotonic()
    pairs, sources = asyncio.run(run_self_play_games(
        ac, team_pool=["Barnacle_meg", "Cybertron_Star"], n_games=n, store_path=tmp_path / "store.jsonl",
        tau=1.0, seed=1, manage_server=True, battle_timeout=300.0, n_workers=1, spawn_rooms=rooms))
    el = time.monotonic() - t0
    dec = sum(len(t.transitions) for pair in pairs for t in pair)
    print(f"[live] spawn x{rooms}: {len(pairs)} games in {el:.0f}s = {60 * len(pairs) / el:.1f} games/min, "
          f"{60 * dec / el:.0f} decisions/min; sources {dict((k, v) for k, v in sources.items() if k.startswith('spawn_'))}")
    assert len(pairs) >= n and sources.get("spawn_pairs", 0) >= n
    assert all(len(a.transitions) > 0 and len(b.transitions) > 0 for a, b in pairs)
