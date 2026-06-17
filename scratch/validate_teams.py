"""Validate every team under teams/Champions/ against the live format using SHOWDOWN'S OWN
validator (authoritative — same check the server applies). Prints which files are illegal and
the per-mon reasons. Run after editing team pastes to confirm they're now legal:

    .venv/Scripts/python.exe scratch/validate_teams.py

Exit 0 = all legal, 1 = some illegal. (Scans M-A now + M-B as you add it.)
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from poke_env.teambuilder import Teambuilder

REPO = Path(__file__).resolve().parents[1]
NODE = REPO / ".venv" / "node" / ("node.exe" if sys.platform == "win32" else "node")
SD = REPO / "pokemon-showdown"
FMT = "gen9championsvgc2026regma"
TEAMS = REPO / "teams" / "Champions"
_SKIP_SUFFIX = {".json", ".md", ".txt", ".py"}


def validate(paste: str):
    packed = Teambuilder.join_team(Teambuilder.parse_showdown_team(paste))
    r = subprocess.run([str(NODE), "pokemon-showdown", "validate-team", FMT],
                       input=packed, cwd=str(SD), capture_output=True, text=True)
    msg = ((r.stdout or "") + (r.stderr or "")).strip()
    reasons = [ln.strip(" |") for ln in msg.replace("||", "\n").splitlines()
               if ln.strip(" |") and "rejected" not in ln.lower()]
    return r.returncode == 0, reasons


def main() -> int:
    files = sorted(p for p in TEAMS.rglob("*") if p.is_file()
                   and p.suffix.lower() not in _SKIP_SUFFIX and not p.name.startswith("."))
    bad = {}
    for f in files:
        try:
            ok, reasons = validate(f.read_text(encoding="utf-8"))
        except Exception as e:                       # noqa: BLE001
            ok, reasons = False, [f"PACK/RUN ERROR: {e}"]
        if not ok:
            bad[f] = reasons
    print(f"TOTAL {len(files)} | LEGAL {len(files) - len(bad)} | ILLEGAL {len(bad)}\n")
    for f, reasons in bad.items():
        print(f"### {f.relative_to(TEAMS).as_posix()}")
        for r in reasons:
            print(f"   - {r}")
        print()
    if not bad:
        print("All teams are format-legal.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
