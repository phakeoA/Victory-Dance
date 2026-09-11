"""Compatibility shim (refactor Phase 4, 2026-09-11) — this module MOVED to ``v_dance.ui.team_builder_server``.

Imported: the old path yields the SAME module object (``sys.modules`` alias). Run as ``python -m v_dance.datatools.server`` (or by file
path): re-runs ``v_dance.ui.team_builder_server`` as ``__main__``. Removed in Phase 6.
"""
import sys as _sys

if __name__ == "__main__":
    import runpy as _runpy

    _runpy.run_module("v_dance.ui.team_builder_server", run_name="__main__", alter_sys=True)
else:
    import v_dance.ui.team_builder_server as _m

    _sys.modules[__name__] = _m
