"""Backfill the opponent dossiers for the 2026-09-02 mega fix (USER: "fix at the source").

Before the fix a mega'd mon was stored with ``ability`` = the MEGA forme's ability (Pixilate on a
Gardevoir) and ``item`` = None (poke-env never sets the stone). This rewrites every such record to
the shape the capture now writes: ``item`` = the stone id, ``mega`` = the forme, ``mega_ability`` =
that ability, ``ability`` = None (the base ability is unknown), ``mega_seen`` = 1 (a lower bound —
per-game evidence was never stored). Per-game ``megas`` lists cannot be reconstructed and stay
absent; the matchup book infers from the record for those games.

Only mega-ONLY abilities are rewritten (Pixilate, Drought-on-Charizard, …); a shared base/mega
ability (Latios Levitate, Scizor Technician, Abomasnow Snow Warning) proves nothing and is left.

    .venv/Scripts/python.exe -m v_dance.datatools.dossier_mega_backfill            # dry-run: what would change
    .venv/Scripts/python.exe -m v_dance.datatools.dossier_mega_backfill --apply    # backup the folder, then write
    (moved from scratch/dossier_mega_backfill.py in the 2026-09-10 refactor, Phase 1)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]          # v_dance/datatools/ -> repo root
assert (_REPO / "pyproject.toml").is_file(), _REPO

from v_dance.play.matchup_book import mega_of, toid  # noqa: E402
from v_dance.play.opponent_dossier import DOSSIER_DIR  # noqa: E402


def plan(dossier_dir: Path) -> tuple:
    """(changes, files_scanned, unreadable) — changes = [(path, doc, [(species, mega)])]."""
    changes, scanned, bad = [], 0, 0
    for p in sorted(Path(dossier_dir).glob("*.json")):
        scanned += 1
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            bad += 1
            continue
        mons = doc.get("mons") if isinstance(doc.get("mons"), dict) else {}
        todo = []
        for sp, rec in mons.items():
            if not isinstance(rec, dict) or rec.get("mega"):
                continue                                    # already the new shape
            mg = mega_of(sp, rec.get("ability"))            # ability-only evidence
            if mg:
                todo.append((sp, mg))
        if todo:
            changes.append((p, doc, todo))
    return changes, scanned, bad


def rewrite(doc: dict, todo: list) -> None:
    for sp, mg in todo:
        rec = doc["mons"][sp]
        rec["mega"] = mg.get("forme")
        rec["mega_ability"] = toid(mg.get("ability") or "") or None
        rec["mega_seen"] = int(rec.get("mega_seen", 0)) or 1
        stone = toid(mg.get("stone") or "")
        if stone:
            rec["item"] = stone
        rec["ability"] = None                               # the base ability is unknown


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", default=str(DOSSIER_DIR))
    ap.add_argument("--apply", action="store_true", help="write the files (after a folder backup)")
    args = ap.parse_args()
    d = Path(args.dir)
    changes, scanned, bad = plan(d)
    by_species = Counter(sp for _p, _doc, todo in changes for sp, _mg in todo)
    n_recs = sum(by_species.values())
    print(f"dossiers: {scanned} scanned, {bad} unreadable, {len(changes)} file(s) / {n_recs} record(s) to rewrite")
    for sp, n in by_species.most_common(15):
        stone = next((mg.get("stone") for _p, _doc, todo in changes for s, mg in todo if s == sp), "?")
        print(f"  {sp:<16} {n:>3}  → item {stone}, ability → None, mega_ability = the mega's")
    if not changes:
        return
    if not args.apply:
        print("dry-run only — re-run with --apply to write (the folder is backed up first).")
        return
    backup = d.parent / f"{d.name}_backup_{time.strftime('%Y%m%d-%H%M%S')}"
    shutil.copytree(d, backup)
    print(f"backup → {backup}")
    for p, doc, todo in changes:
        rewrite(doc, todo)
        p.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(f"rewrote {len(changes)} file(s), {n_recs} record(s).")


if __name__ == "__main__":
    main()
