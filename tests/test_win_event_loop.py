"""Fix B: Ctrl-C must not wedge the terminal.

poke-env runs its websocket I/O on a background event loop created at import time. On Windows the
default ProactorEventLoop (IOCP) has a known CPython race in ``_poll`` under Ctrl-C: it calls
``future.set_exception()`` on an already-resolved future -> ``InvalidStateError`` propagates out of
``run_forever``, killing the loop thread; afterwards every poke-env coroutine (and poke-env's own
``atexit`` cleanup) blocks forever and the process never exits. ``v_dance/__init__`` forces the
SelectorEventLoop policy BEFORE poke_env is imported to remove the race.

These tests are Windows-only (the bug + the fix only exist there)."""
from __future__ import annotations

import asyncio
import sys

import pytest

import v_dance  # noqa: F401  importing the package sets the loop policy as a side effect


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only event-loop fix")
def test_selector_policy_is_active_on_windows():
    assert isinstance(asyncio.get_event_loop_policy(), asyncio.WindowsSelectorEventLoopPolicy)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only event-loop fix")
def test_new_event_loop_is_selector_not_proactor():
    # The functional guarantee: a freshly-created loop (poke-env's POKE_LOOP is one) is a
    # SelectorEventLoop, NOT the ProactorEventLoop whose _poll races under Ctrl-C.
    loop = asyncio.new_event_loop()
    try:
        assert isinstance(loop, asyncio.SelectorEventLoop)
        assert not isinstance(loop, asyncio.ProactorEventLoop)
    finally:
        loop.close()
