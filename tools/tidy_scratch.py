"""Phase-1 tidy of ``scratch/`` (2026-09-10): move concluded probes / audit artifacts into ``scratch/archive/<topic>/``.

Idempotent and additive: nothing is deleted except ``__pycache__``; every move is appended to ``scratch/archive/INDEX.md``
(old path → new path) so a docstring or memory note that still says ``scratch/<file>`` can be resolved. Files kept at the
``scratch/`` root are the ones a current instrument or note points to. ``scratch/`` stays gitignored, so this has no
effect on git or CI. Run from the repo root; ``--dry-run`` lists the plan.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import date
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
assert (_REPO / "pyproject.toml").is_file(), _REPO
SCRATCH = _REPO / "scratch"
ARCHIVE = SCRATCH / "archive"

KEEP = {  # live instruments referenced by memory / package comments
    "ladder_analysis_2026_09_07", "spawn_throughput_ab.py", "_hof_smoke.py", "ci_docker_check.ps1", "archive",
}


def topic_for(p: Path) -> str | None:
    n = p.name
    if n in KEEP:
        return None
    if n.startswith("tp_"):
        return "tp"
    if n.startswith("v11_"):
        return "v11"
    if n.startswith("browser_") or n in {"online_multitransport_plan.md", "play_vs_human_browser_plan.md",
                                         "repro_rechallenge_hang.py"}:
        return "browser"
    if n.startswith("wf_") and n.endswith(".js"):
        return "wf"
    if n.startswith("levelC_") or n.startswith("levelB_"):
        return "levelC"
    if n.startswith("type_eff_probe"):
        return "type_eff"
    if n.startswith("_suite_out") or n == "_ruler_2b.txt":
        return "suite_outputs"
    if n.endswith("_synthesis.json") or n.endswith("_synthesis.md"):
        return "audits"
    return "misc"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    moves: list[tuple[Path, Path]] = []
    for p in sorted(SCRATCH.iterdir()):
        if p.name == "__pycache__":
            continue
        topic = topic_for(p)
        if topic is None:
            continue
        moves.append((p, ARCHIVE / topic / p.name))

    for src, dst in moves:
        print(f"{src.relative_to(_REPO)}  ->  {dst.relative_to(_REPO)}")
    print(f"\n{len(moves)} entries to archive; kept at scratch/: {sorted(KEEP - {'archive'})}")
    if args.dry_run:
        return 0

    ARCHIVE.mkdir(exist_ok=True)
    index = ARCHIVE / "INDEX.md"
    if not index.exists():
        index.write_text("# scratch/archive — index of moved files (old path → new path)\n\n"
                         "Concluded probes, one-off harnesses, audit synthesis files and suite outputs. Nothing here is "
                         "imported or launched by the package, the tests, Mission Control or a runbook (verified with "
                         "`tools/find_refs.py` before each move). Add a dated section per tidy.\n", encoding="utf-8")
    lines = [f"\n## {date.today().isoformat()}\n"]
    for src, dst in moves:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            print(f"SKIP (exists): {dst}")
            continue
        shutil.move(str(src), str(dst))
        lines.append(f"- `{src.relative_to(_REPO).as_posix()}` → `{dst.relative_to(_REPO).as_posix()}`")
    pyc = SCRATCH / "__pycache__"
    if pyc.exists():
        shutil.rmtree(pyc)
    with index.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"done; index at {index.relative_to(_REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
