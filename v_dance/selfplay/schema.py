"""Compatibility shim (refactor Phase 2, 2026-09-10) — this module MOVED to ``v_dance.rl.schema``.

Importing the old path yields the SAME module object (``sys.modules`` alias), so ``import v_dance.selfplay.schema``,
``from v_dance.selfplay import schema`` and ``monkeypatch.setattr`` on either path all keep working.
Removed in refactor Phase 6 (the reference sweep); new code imports ``v_dance.rl.schema``.
"""
import sys as _sys

import v_dance.rl.schema as _m

_sys.modules[__name__] = _m
