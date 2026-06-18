"""v_dance — Victory-Dance VGC Pokemon Showdown bot.

The single importable package for the project. Subpackages:

- ``encoders``  state/live state encoding (state_encoder, live_state_encoder)
- ``parser``    VOD/replay parsing + belief state (vod_parser, belief_state)
- ``models``    network definitions (BC model, team-preview model, network)
- ``training``  training + dataset + eval scripts (train_bc, bc_dataset, ...)
- ``play``      live battle players (vgc_base, live_vgc_base, player, model_io)
- ``eval``      gauntlet + scripted opponents
- ``selfplay``  PPO self-play stack (the former local_battle/self_play)
- ``datatools`` scrapers, bulk export, corpus QA, team sheets, the flask server

NOTE: subpackages are populated incrementally during restructure Stage 2. The
package is intentionally named ``v_dance`` (not ``victory_dance``) because
"Victory Dance" is an actual in-game Pokemon move and would be confusing.
"""

import sys as _sys

# ── Windows event-loop fix: Ctrl-C must not wedge the terminal ─────────────────
# poke-env runs all its websocket I/O on a background event loop (``POKE_LOOP``)
# created at IMPORT TIME (poke_env/concurrency.py). On Windows the default is the
# ProactorEventLoop (IOCP), whose ``_poll`` has a known CPython race under Ctrl-C:
# it calls ``future.set_exception()`` on an ALREADY-RESOLVED future ->
# ``InvalidStateError`` propagates out of ``run_forever`` and KILLS the POKE_LOOP
# thread. After that, every poke-env coroutine bridged via
# ``run_coroutine_threadsafe(...).result()`` blocks forever — including poke-env's
# own ``atexit`` cleanup — so the process never exits and the terminal stays stuck
# (the exact symptom: an ``asyncio ... InvalidStateError`` traceback, then a hang).
# The SelectorEventLoop (plain ``select()`` over TCP sockets; websockets needs
# nothing more — the repo uses NO asyncio subprocesses/pipes) has no such race.
# This MUST run before any submodule imports ``poke_env`` (which builds POKE_LOOP
# at import time). This package __init__ runs first for every ``python -m
# v_dance.*`` entry AND for the spawn collection workers (they re-import v_dance),
# so setting the policy here covers them all.
if _sys.platform == "win32":
    import asyncio as _asyncio

    try:
        _asyncio.set_event_loop_policy(_asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:  # pragma: no cover - extremely defensive (always present on win32)
        pass

__version__ = "0.1.0"
