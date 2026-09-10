"""find_refs — where is a path / module string referenced in the repo's CODE and CONFIG? (refactor instrument, 2026-09-10)

Run from the repo root:
    PYTHONUTF8=1 .venv/Scripts/python.exe -X utf8 tools/find_refs.py "scratch/" ["v_dance.play.play_online_browser" ...]
    ... --docs        also search docs/*.md and the repo-root *.md notes (off by default: history, not code)
    ... --names-only  print only the file names that match

Searches ONLY the roots that can carry a live reference (package, tests, tools, the web assets under data/scripts, the
top-level runbooks, setup.sh, pyproject, CI, README, .claude/launch.json). It NEVER descends into data (replays, corpora,
dex JSON), artifacts (logs, replays, bandit state, archives), checkpoints, the venv, the Showdown clone, memory archives
or __pycache__ — those are gigabytes of text that cannot hold a code reference and only waste time and tokens.
Use it before AND after every refactor phase: the "after" run for the old string must return only historical notes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
assert (_REPO / "pyproject.toml").is_file(), _REPO

# Directories that are never searched, wherever they appear.
SKIP_DIRS = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", "node_modules", ".idea", ".vscode",
    "pokemon-showdown",           # the pinned server clone (TypeScript + data, not ours)
    "ai_train_scripts",           # checkpoints (binary)
    "artifacts",                  # logs / replays / bandit state / archives (top-level *.sh are added explicitly)
    "data",                       # dex JSON, corpora, vods, teams (data/scripts is added explicitly)
    "logs", "memory_archive", "teams", "config", "dist", "build",
    "archive",                    # scratch/archive once Phase 1 tidies it
}
CODE_SUFFIXES = {".py", ".pyi", ".js", ".mjs", ".ts", ".html", ".css", ".sh", ".ps1", ".bat", ".yml", ".yaml",
                 ".toml", ".json", ".cfg", ".ini", ".txt"}
ROOTS = ["v_dance", "tests", "tools", "scratch", "data/scripts", ".github", ".claude"]
ROOT_FILES = ["setup.sh", "pyproject.toml", "README.md", "requirements.txt", "config.example.json", ".gitignore"]
DOC_ROOTS = ["docs"]


def iter_files(with_docs: bool):
    seen: set[Path] = set()

    def walk(root: Path):
        if not root.exists():
            return
        if root.is_file():
            yield root
            return
        for p in root.rglob("*"):
            if any(part in SKIP_DIRS for part in p.relative_to(_REPO).parts[:-1]):
                continue
            if p.is_file() and p.suffix.lower() in CODE_SUFFIXES and p.stat().st_size < 4_000_000:
                yield p

    for r in ROOTS + (DOC_ROOTS if with_docs else []):
        for p in walk(_REPO / r):
            if p not in seen:
                seen.add(p)
                yield p
    for f in ROOT_FILES:
        p = _REPO / f
        if p.is_file() and p not in seen:
            seen.add(p)
            yield p
    for p in sorted((_REPO / "artifacts").glob("*.sh")):   # the runbooks only, never the run outputs
        if p not in seen:
            seen.add(p)
            yield p
    if with_docs:
        for p in sorted(_REPO.glob("*.md")):
            if p not in seen:
                seen.add(p)
                yield p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("patterns", nargs="+", help="plain substrings (a backslash variant is tried for path-like ones)")
    ap.add_argument("--docs", action="store_true")
    ap.add_argument("--names-only", action="store_true")
    args = ap.parse_args(argv)

    variants: dict[str, list[str]] = {}
    for pat in args.patterns:
        alts = [pat]
        if "/" in pat:
            alts.append(pat.replace("/", "\\"))
            alts.append(pat.replace("/", "\\\\"))
        variants[pat] = alts

    n_files = 0
    hits: dict[str, list[tuple[str, int, str]]] = {p: [] for p in args.patterns}
    for path in iter_files(args.docs):
        n_files += 1
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = str(path.relative_to(_REPO)).replace("\\", "/")
        for pat, alts in variants.items():
            if not any(a in text for a in alts):
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if any(a in line for a in alts):
                    hits[pat].append((rel, i, line.strip()))
    for pat in args.patterns:
        rows = hits[pat]
        files = sorted({r[0] for r in rows})
        print(f"=== {pat!r}: {len(rows)} line(s) in {len(files)} file(s) ===")
        if args.names_only:
            for f in files:
                print(f"  {f}")
        else:
            for rel, i, line in rows:
                print(f"  {rel}:{i}: {line[:160]}")
    print(f"\n({n_files} files searched; data/, artifacts/ outputs, checkpoints, venv and the Showdown clone skipped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
