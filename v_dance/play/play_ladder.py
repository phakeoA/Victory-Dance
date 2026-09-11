"""Compatibility shim (refactor Phase 3, 2026-09-10) — this module MOVED to ``v_dance.online.play_ladder``.

Imported: the old path yields the SAME module object (``sys.modules`` alias). Run as ``python -m v_dance.play.play_ladder``:
re-runs ``v_dance.online.play_ladder`` as ``__main__`` (exactly what ``-m`` on the new path does). Removed in Phase 6.
"""
import sys as _sys

if __name__ == "__main__":
    import runpy as _runpy

    _runpy.run_module("v_dance.online.play_ladder", run_name="__main__", alter_sys=True)
else:
    import v_dance.online.play_ladder as _m

    _sys.modules[__name__] = _m
