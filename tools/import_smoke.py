"""Import smoke — every module under ``v_dance`` must import cleanly (refactor safety net, 2026-09-10).

Run from the repo root:  ``PYTHONUTF8=1 .venv/Scripts/python.exe -X utf8 tools/import_smoke.py``

Walks the package with ``pkgutil.walk_packages`` and imports each module in THIS process (one torch / poke-env /
playwright import cost, then cheap), printing the wall time per module and a final PASS / FAIL line. Exit 1 on any
failure. It does not run anything: modules that start servers or battles do so only under ``__main__``.
Pass ``--only <prefix>`` to restrict to a subpackage (e.g. ``--only v_dance.play``).
"""
from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys
import time
import traceback
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
assert (_REPO / "pyproject.toml").is_file(), _REPO
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def discover(prefix: str = "v_dance") -> list[str]:
    root = importlib.import_module(prefix.split(".")[0])
    names = [root.__name__]
    for info in pkgutil.walk_packages(root.__path__, root.__name__ + "."):
        names.append(info.name)
    return sorted(n for n in names if n.startswith(prefix))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="v_dance", help="module prefix to restrict to")
    ap.add_argument("--quiet", action="store_true", help="print failures + the summary only")
    args = ap.parse_args(argv)

    names = discover(args.only)
    failures: list[tuple[str, str]] = []
    t_all = time.perf_counter()
    for name in names:
        t0 = time.perf_counter()
        try:
            importlib.import_module(name)
            status = "ok"
        except BaseException as exc:  # noqa: BLE001 — SystemExit at import is a failure too
            status = f"FAIL {type(exc).__name__}: {exc}"
            failures.append((name, traceback.format_exc()))
        dt = time.perf_counter() - t0
        if not args.quiet or status != "ok":
            print(f"{dt:7.2f}s  {name:55s} {status}")
    total = time.perf_counter() - t_all
    for name, tb in failures:
        print(f"\n--- {name} ---\n{tb}")
    print(f"\n{'PASS' if not failures else 'FAIL'}: {len(names) - len(failures)}/{len(names)} modules imported "
          f"in {total:.1f}s ({len(failures)} failure(s))")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
