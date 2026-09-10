"""Import-graph scan of ``v_dance`` (refactor instrument, 2026-09-10). Read-only.

Run from the repo root:  ``PYTHONUTF8=1 .venv/Scripts/python.exe -X utf8 tools/import_graph.py [--fanout PKG] [--json OUT]``

Prints: package sizes · the package→package import matrix · package-level cycles (pairs importing each other) ·
the 25 biggest modules · the most-imported modules · (optionally) per-module fan-out for one package. Parses imports
with ``ast`` (static; conditional / function-level imports count the same as top-level ones). Used to verify that a
refactor phase removed the cycle it claimed to remove — run it before and after and diff the matrix.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
assert (_REPO / "pyproject.toml").is_file(), _REPO
PKG = _REPO / "v_dance"          # the ONLY tree walked — never data/, artifacts/, checkpoints, the venv or the clone
SKIP = {"__pycache__", "showdown_plugins", "static", "archive"}   # non-Python subtrees inside the package


def module_name(p: Path) -> str:
    parts = list(p.relative_to(_REPO).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def pkg_of(mod: str) -> str:
    parts = mod.split(".")
    return parts[1] if len(parts) >= 3 else "(root)"


def imports_of(p: Path, mod: str) -> list[str]:
    tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [a.name for a in node.names if a.name.startswith("v_dance")]
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import → resolve against this module's package
                base = mod.split(".")
                if p.name != "__init__.py":
                    base = base[:-1]
                base = base[: len(base) - (node.level - 1)] if node.level > 1 else base
                out.append(".".join(base + ([node.module] if node.module else [])))
            elif node.module and node.module.startswith("v_dance"):
                out.append(node.module)
    return out


def scan():
    lines: dict[str, int] = {}
    file_edges: dict[str, set[str]] = defaultdict(set)
    for p in sorted(PKG.rglob("*.py")):
        if any(s in p.parts for s in SKIP):
            continue
        mod = module_name(p)
        lines[mod] = sum(1 for _ in p.open(encoding="utf-8", errors="replace"))
        for dst in imports_of(p, mod):
            file_edges[mod].add(dst)
    return lines, file_edges


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fanout", help="print per-module in-repo imports for this subpackage (e.g. play)")
    ap.add_argument("--json", help="write the raw edges + sizes to this JSON file")
    args = ap.parse_args(argv)

    lines, file_edges = scan()
    edges: dict[str, Counter] = defaultdict(Counter)
    for src, dsts in file_edges.items():
        for dst in dsts:
            s, d = pkg_of(src), pkg_of(dst)
            if s != d:
                edges[s][d] += 1
    pkgs = sorted({pkg_of(m) for m in lines})

    print("=== package sizes ===")
    for pk in pkgs:
        mods = [m for m in lines if pkg_of(m) == pk]
        print(f"{pk:12s} files={len(mods):3d} lines={sum(lines[m] for m in mods):6d}")

    print("\n=== package -> package import counts (row imports column) ===")
    print("src\\dst".ljust(12) + "".join(d[:9].rjust(10) for d in pkgs))
    for s in pkgs:
        print(s.ljust(12) + "".join(str(edges[s].get(d, 0) or "-").rjust(10) for d in pkgs))

    print("\n=== package-level cycles ===")
    cycles = [(a, b, edges[a][b], edges[b][a]) for a in pkgs for b in pkgs if a < b and edges[a].get(b) and edges[b].get(a)]
    for a, b, ab, ba in cycles:
        print(f"{a} <-> {b}: {ab} / {ba}")
    if not cycles:
        print("(none)")

    print("\n=== 25 biggest modules ===")
    for m, n in sorted(lines.items(), key=lambda kv: -kv[1])[:25]:
        print(f"{n:6d}  {m}")

    inbound: dict[str, set[str]] = defaultdict(set)
    for src, dsts in file_edges.items():
        for d in dsts:
            inbound[d].add(src)
    print("\n=== 25 most-imported modules ===")
    for d, srcs in sorted(inbound.items(), key=lambda kv: -len(kv[1]))[:25]:
        print(f"{len(srcs):3d}  {d}")

    if args.fanout:
        print(f"\n=== fan-out of v_dance.{args.fanout}.* ===")
        for m in sorted(x for x in lines if x.startswith(f"v_dance.{args.fanout}.")):
            short = sorted(d.replace("v_dance.", "") for d in file_edges[m])
            print(f"{m.split('.', 2)[-1]:28s} <- {', '.join(short)}")

    if args.json:
        Path(args.json).write_text(json.dumps({"lines": lines, "edges": {k: sorted(v) for k, v in file_edges.items()}},
                                              indent=1), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
