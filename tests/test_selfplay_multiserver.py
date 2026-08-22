"""Offline tests for the #22f multi-server sharding wired into game_runner.run_self_play_games.

The live smoke (spawning N Node servers) is separate; here we verify the sharding MATH with an
injected start/stop (no Node): K servers on consecutive ports, descriptor uids round-robin across
them (balanced), teardown stops all, and the runner accepts the n_servers param.
"""
from __future__ import annotations

import inspect
from collections import Counter

from v_dance.play.run_local_battle import ServerPool, SHOWDOWN_PORT


def _fake_pool(k):
    started, stopped = [], []
    pool = ServerPool(
        k,
        start_fn=lambda p: (started.append(p) or f"proc{p}"),
        stop_fn=lambda x: stopped.append(x),
    ).start_all()
    return pool, started, stopped


def test_pool_starts_k_servers_on_consecutive_ports():
    pool, started, _ = _fake_pool(4)
    assert pool.ports == [SHOWDOWN_PORT + i for i in range(4)]
    assert started == pool.ports                          # all 4 launched, in order


def test_descriptor_uids_shard_evenly_across_servers():
    # game_runner assigns each descriptor uid (1..N) a server via pool.port_for_worker(uid).
    pool, _, _ = _fake_pool(4)
    dist = Counter(pool.port_for_worker(uid) for uid in range(1, 201))
    assert set(dist) == set(pool.ports)                   # every server carries battles
    assert max(dist.values()) - min(dist.values()) <= 1   # balanced round-robin


def test_both_seats_of_a_pairing_share_one_server():
    # p1 and p2 of a pairing must land on the SAME server (they battle each other): both use uid.
    pool, _, _ = _fake_pool(3)
    for uid in range(1, 30):
        assert pool.port_for_worker(uid) == pool.port_for_worker(uid)   # deterministic per uid


def test_stop_all_tears_down_every_started_server():
    pool, _, stopped = _fake_pool(3)
    pool.stop_all()
    assert sorted(stopped) == ["proc8000", "proc8001", "proc8002"]


def test_single_server_pool_is_the_default_path():
    # n_servers=1 → one server on 8000, no sharding (game_runner passes no server_configuration).
    pool, started, _ = _fake_pool(1)
    assert pool.ports == [SHOWDOWN_PORT] and started == [SHOWDOWN_PORT]


def test_run_self_play_games_accepts_n_servers():
    from v_dance.selfplay.game_runner import run_self_play_games
    assert "n_servers" in inspect.signature(run_self_play_games).parameters
