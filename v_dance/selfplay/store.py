"""Compatibility shim (refactor Phase 2, 2026-09-10) — this module MOVED to ``v_dance.rl.store``.

Importing the old path yields the SAME module object (``sys.modules`` alias), so ``import v_dance.selfplay.store``,
``from v_dance.selfplay import store`` and ``monkeypatch.setattr`` on either path all keep working.
Removed in refactor Phase 6 (the reference sweep); new code imports ``v_dance.rl.store``.
"""
import sys as _sys

import v_dance.rl.store as _m

_sys.modules[__name__] = _m
