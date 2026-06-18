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
