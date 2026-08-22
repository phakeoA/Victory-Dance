"""Tests for run_local_battle server lifecycle — specifically stop_showdown's
process-TREE kill (regression for the orphaned-Showdown-server-on-port-8000 bug:
``node pokemon-showdown start`` forks a child server that the old single-process
terminate() left running, which the next start_showdown then silently reused)."""
from __future__ import annotations

import sys
import types

import pytest

pytest.importorskip("poke_env")
import v_dance.play.run_local_battle as R


class _FakeP:
    """Minimal psutil.Process stand-in: records terminate()/kill() and exposes a
    children() tree."""

    def __init__(self, pid, children=()):
        self.pid = pid
        self._children = list(children)
        self.terminated = False
        self.killed = False

    def children(self, recursive=False):
        return list(self._children)

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


def _fake_proc(pid=999, exited=False):
    """A subprocess.Popen stand-in for stop_showdown."""
    state = {"terminated": False, "killed": False}
    return types.SimpleNamespace(
        pid=pid,
        poll=lambda: (0 if exited else None),
        terminate=lambda: state.__setitem__("terminated", True),
        kill=lambda: state.__setitem__("killed", True),
        wait=lambda timeout=None: None,
        _state=state,
    )


def test_stop_showdown_none_is_noop():
    R.stop_showdown(None)            # must not raise


def test_stop_showdown_already_exited_skips_kill():
    proc = _fake_proc(exited=True)
    R.stop_showdown(proc)
    assert proc._state["terminated"] is False   # already dead → no terminate attempt


def test_stop_showdown_kills_whole_process_tree(monkeypatch):
    """The regression: stop_showdown must terminate the launcher AND its forked child
    server (the one holding the port), not just the launcher."""
    import psutil
    child = _FakeP(1000)
    parent = _FakeP(999, children=[child])
    monkeypatch.setattr(psutil, "Process", lambda pid: parent)
    monkeypatch.setattr(psutil, "wait_procs", lambda procs, timeout=None: (list(procs), []))

    proc = _fake_proc(pid=999)
    R.stop_showdown(proc)
    assert parent.terminated is True            # launcher killed
    assert child.terminated is True             # AND the forked child server (the orphan fix)


def test_stop_showdown_kills_stragglers_that_survive_terminate(monkeypatch):
    """A child that ignores SIGTERM (still 'alive' after wait_procs) gets SIGKILLed."""
    import psutil
    stubborn = _FakeP(1000)
    parent = _FakeP(999, children=[stubborn])
    monkeypatch.setattr(psutil, "Process", lambda pid: parent)
    # wait_procs reports the stubborn child as still alive → it must be kill()ed
    monkeypatch.setattr(psutil, "wait_procs", lambda procs, timeout=None: ([], [stubborn]))

    R.stop_showdown(_fake_proc(pid=999))
    assert stubborn.killed is True


def test_stop_showdown_falls_back_when_psutil_unavailable(monkeypatch):
    """If psutil can't enumerate the tree, fall back to taskkill /T on Windows (or a plain
    terminate elsewhere) — never silently leave the tree alive."""
    import psutil
    def _boom(pid):
        raise RuntimeError("psutil unavailable")
    monkeypatch.setattr(psutil, "Process", _boom)

    taskkill = {"called": False, "tree_flag": False}
    def fake_run(cmd, **kw):
        if cmd and "taskkill" in str(cmd[0]):
            taskkill["called"] = True
            taskkill["tree_flag"] = "/T" in cmd
        return None
    monkeypatch.setattr(R.subprocess, "run", fake_run)

    proc = _fake_proc(pid=4321)
    R.stop_showdown(proc)
    if sys.platform == "win32":
        assert taskkill["called"] and taskkill["tree_flag"]   # taskkill /T walked the tree
    else:
        assert proc._state["terminated"] is True              # plain terminate fallback


# ── ServerPool (22f.1) — start/recycle/stop orchestration, offline (no Node spawn) ──
def test_server_pool_assigns_consecutive_ports_and_starts_each():
    started = []
    pool = R.ServerPool(3, start_fn=lambda p: started.append(p) or f"proc{p}",
                        stop_fn=lambda x: None).start_all()
    assert pool.ports == [8000, 8001, 8002]
    assert started == [8000, 8001, 8002]                       # one server launched per port


def test_server_pool_port_for_worker_is_round_robin():
    pool = R.ServerPool(3, start_fn=lambda p: ("proc", p), stop_fn=lambda x: None).start_all()
    assert [pool.port_for_worker(i) for i in range(7)] == [8000, 8001, 8002, 8000, 8001, 8002, 8000]


def test_server_pool_custom_base_port():
    pool = R.ServerPool(2, base_port=9100, start_fn=lambda p: ("proc", p),
                        stop_fn=lambda x: None).start_all()
    assert pool.ports == [9100, 9101] and pool.port_for_worker(1) == 9101


def test_server_pool_recycle_restarts_only_that_server():
    started, stopped = [], []
    pool = R.ServerPool(2, start_fn=lambda p: started.append(p) or ("proc", p),
                        stop_fn=lambda x: stopped.append(x)).start_all()
    pool.recycle(8001)
    assert stopped == [("proc", 8001)]                         # the old 8001 server was stopped
    assert started == [8000, 8001, 8001]                       # 8001 restarted; 8000 untouched


def test_server_pool_stop_all_tree_kills_each():
    stopped = []
    pool = R.ServerPool(3, start_fn=lambda p: ("proc", p), stop_fn=lambda x: stopped.append(x)).start_all()
    pool.stop_all()
    assert sorted(stopped) == [("proc", 8000), ("proc", 8001), ("proc", 8002)]
    assert pool._procs == {}                                   # all handles released


def test_server_pool_unmanaged_never_spawns():
    """``manage=False`` (server started elsewhere / single-server mode) → start/recycle/stop are
    all no-ops, so a pre-existing server is never touched."""
    pool = R.ServerPool(2, manage=False,
                        start_fn=lambda p: pytest.fail("must not start"),
                        stop_fn=lambda x: pytest.fail("must not stop")).start_all()
    pool.recycle(8000)
    pool.stop_all()                                            # no raise; nothing spawned/stopped


def test_server_pool_context_manager_starts_and_stops():
    started, stopped = [], []
    with R.ServerPool(2, start_fn=lambda p: started.append(p) or ("proc", p),
                      stop_fn=lambda x: stopped.append(x)) as pool:
        assert started == [8000, 8001]
    assert sorted(stopped) == [("proc", 8000), ("proc", 8001)]   # stopped on __exit__


def test_server_pool_defaults_to_single_server():
    pool = R.ServerPool()                                      # n=1, base 8000 — the legacy shape
    assert pool.ports == [8000] and pool.port_for_worker(5) == 8000


# ── per-server player binding (22f.2) ─────────────────────────────────────────
def test_localhost_server_config_swaps_port():
    sc = R.localhost_server_config(8005)
    assert sc.websocket_url == "ws://localhost:8005/showdown/websocket"
    from poke_env.ps_client.server_configuration import LocalhostServerConfiguration as L
    assert sc.authentication_url == L.authentication_url     # auth endpoint preserved


def test_make_player_threads_port_into_server_config(monkeypatch):
    """A ``port`` makes the player bind that pool server (22f); no port keeps poke-env's default."""
    captured = {}

    class _Stub:
        def __init__(self, **kw):
            captured.clear()
            captured.update(kw)

    monkeypatch.setattr(R, "RandomVGCPlayer", _Stub)            # model_path=None -> random player
    R.make_player("u", "team", model_path=None, port=8003)
    sc = captured.get("server_configuration")
    assert sc is not None and sc.websocket_url == "ws://localhost:8003/showdown/websocket"

    R.make_player("u", "team", model_path=None)                # no port
    assert "server_configuration" not in captured              # poke-env's localhost:8000 default


def test_make_opponent_forwards_port(monkeypatch):
    import v_dance.eval.gauntlet as GA
    seen = {}
    monkeypatch.setattr(R, "make_player",
                        lambda *a, **k: seen.setdefault("port", k.get("port")))
    GA._make_opponent("random", "OPrand", "team", port=8002)
    assert seen["port"] == 8002                                # threaded down to make_player


def test_start_showdown_on_passes_port_as_positional_arg(monkeypatch):
    """The Node launch must put the PORT in argv (Showdown scans argv for the first numeric token)
    and not double-launch when the port is already serving."""
    calls = {}
    monkeypatch.setattr(R, "_port_open", lambda host, port: False)   # nothing listening yet
    monkeypatch.setattr(R, "_find_node", lambda: "node")
    class _Proc:
        returncode = 0
        def poll(self): return None
    def fake_popen(cmd, **kw):
        calls["cmd"] = cmd
        # flip the port-open check so the readiness loop returns immediately
        monkeypatch.setattr(R, "_port_open", lambda host, port: True)
        return _Proc()
    monkeypatch.setattr(R.subprocess, "Popen", fake_popen)
    R.start_showdown_on(8003)
    assert "8003" in calls["cmd"] and "start" in calls["cmd"]   # PORT passed positionally
    # the numeric port token precedes --no-security (Showdown takes the first numeric arg)
    assert calls["cmd"].index("8003") > calls["cmd"].index("start")
