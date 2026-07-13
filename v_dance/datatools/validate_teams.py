"""Showdown-authoritative team validation (M4 B3 — promoted from scratch/validate_teams.py).

Runs the pinned server's OWN ``validate-team`` (node subprocess, no server process) so the
verdict is exactly what the ladder would apply. ``fmt=None`` uses the stack's active format.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

_REPO = Path(__file__).resolve().parents[2]
NODE = _REPO / ".venv" / "node" / ("node.exe" if sys.platform == "win32" else "node")
SD = _REPO / "pokemon-showdown"


def validate(paste: str, fmt: Optional[str] = None):
    """(ok, reasons) for a Showdown-export paste, judged by the pinned validator."""
    from poke_env.teambuilder import Teambuilder
    if fmt is None:
        from v_dance.formats import default_format
        fmt = default_format()
    packed = Teambuilder.join_team(Teambuilder.parse_showdown_team(paste))
    r = subprocess.run([str(NODE), "pokemon-showdown", "validate-team", fmt],
                       input=packed, cwd=str(SD), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")  # accented species w/o PYTHONUTF8
    msg = ((r.stdout or "") + (r.stderr or "")).strip()
    reasons = [ln.strip(" |") for ln in msg.replace("||", "\n").splitlines()
               if ln.strip(" |") and "rejected" not in ln.lower()]
    return r.returncode == 0, reasons
