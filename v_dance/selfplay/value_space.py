"""Compatibility shim (refactor Phase 2, 2026-09-10) — this module MOVED to ``v_dance.rl.value_space``.

Importing the old path yields the SAME module object (``sys.modules`` alias), so ``import v_dance.selfplay.value_space``,
``from v_dance.selfplay import value_space`` and ``monkeypatch.setattr`` on either path all keep working.
Removed in refactor Phase 6 (the reference sweep); new code imports ``v_dance.rl.value_space``.
"""
import sys as _sys

import v_dance.rl.value_space as _m

_sys.modules[__name__] = _m
