"""Reg M-C (or any new regulation) release checklist — era-5 W4 (docs/era5_specialist_design.md §3.4).

Run it the day a new regulation appears, and again after each step of the release checklist:

    .venv/Scripts/python.exe -m v_dance.datatools.release_check --reg mc \
        --expect lucariomegaz=aurabreak absolmegaz=sharpness garchompmegaz=levitate heatranmega=? \
                 rillaboom=grassysurge baxcalibur=thermalexchange

It prints one PASS / WARN / FAIL row per item: format registry entry, roster file, Pikalytics /
belief file, team pool folder, dex entries for the named formes (with the BUNDLED abilities next
to what the USER was told — the pinned Showdown data carried PLACEHOLDER abilities on 2026-09-01),
required mega stones in the pinned items.ts, and whether each expected ability is known to the
encoder's mechanic tables (text-grounded: the id appears in battle_mechanics.py / encoder_layout.py
/ tp_features.py). Nothing here mutates anything.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]          # v_dance/datatools/ -> repo root (moved from scratch/mc_release_check.py 2026-09-10)
assert (_REPO / "pyproject.toml").is_file(), _REPO


def _id(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reg", default="mc", help="reg token letters, e.g. mc → gen9championsvgc2026regmc")
    ap.add_argument("--expect", nargs="*", default=[],
                    help="dex_id=ability pairs the USER confirmed (use ? for unknown)")
    ap.add_argument("--year", default="2026")
    args = ap.parse_args(argv)

    reg = args.reg.lower().lstrip("reg")
    fmt = f"gen9championsvgc{args.year}reg{reg}"
    label = f"M-{reg[-1].upper()}" if len(reg) == 2 else reg.upper()
    rows = []

    def row(status, item, detail=""):
        rows.append((status, item, detail))

    # 1. format registry
    reg_path = _REPO / "data" / "regulations" / "vod_parser_format_names.json"
    try:
        formats = json.loads(reg_path.read_text(encoding="utf-8")).get("formats", [])
        ids = {f.get("id") for f in formats}
        row("PASS" if fmt in ids else "FAIL", "format registry", f"{fmt} {'present' if fmt in ids else 'MISSING'} in {reg_path.name}")
    except Exception as exc:
        row("FAIL", "format registry", repr(exc))

    # 2. roster / pikalytics / team pool
    for name, p, hint in (
        ("roster file", _REPO / "data" / f"champions_roster_reg{reg}.json", "enum_champions_roster.js --reg"),
        ("Pikalytics / belief", _REPO / "data" / f"pikalytics_reg{reg}.json", "scrape_pikalytics.py --format"),
    ):
        if p.is_file():
            try:
                n = len(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                n = "?"
            row("PASS", name, f"{p.name} ({n} entries)")
        else:
            row("FAIL", name, f"{p.name} missing → run {hint}")
    pool = _REPO / "teams" / "Champions" / label
    teams = [t for t in pool.iterdir() if t.is_file() and t.suffix.lower() not in (".json", ".md")] if pool.is_dir() else []
    row("PASS" if teams else "FAIL", "team pool", f"{pool.relative_to(_REPO)}: {len(teams)} team(s)")

    # 3. dex entries + expected abilities, mega stones, encoder coverage
    dex = json.loads((_REPO / "data" / "pokedex.json").read_text(encoding="utf-8"))
    dex_l = {k.lower(): v for k, v in dex.items()}
    items_src = ""
    try:
        items_src = (_REPO / "pokemon-showdown" / "data" / "items.ts").read_text(encoding="utf-8", errors="replace")
    except Exception:
        pass
    enc_src = ""
    for rel in ("v_dance/encoders/battle_mechanics.py", "v_dance/encoders/encoder_layout.py",
                "v_dance/training/tp_features.py", "v_dance/encoders/mechanic_tags.py"):
        try:
            enc_src += (_REPO / rel).read_text(encoding="utf-8", errors="replace").lower()
        except Exception:
            pass
    known_untagged = ""
    try:
        known_untagged = (_REPO / "v_dance" / "training" / "mechanic_coverage.py").read_text(encoding="utf-8").lower()
    except Exception:
        pass

    for pair in args.expect:
        did, _, ability = pair.partition("=")
        did, ability = _id(did), (ability or "").strip()
        e = dex_l.get(did)
        if not e:
            row("FAIL", f"dex {did}", "not in data/pokedex.json → update_pokedex.py after the Showdown re-pin")
            continue
        bundled = sorted(set(str(v) for v in (e.get("abilities") or {}).values()))
        req = e.get("requiredItem")
        detail = f"bundled abilities {bundled}"
        if ability and ability != "?":
            if any(_id(b) == _id(ability) for b in bundled):
                row("PASS", f"dex {did}", detail + f"; matches expected {ability}")
            else:
                row("WARN", f"dex {did}", detail + f"; USER expects {ability} → PLACEHOLDER data, re-pin + diff")
        else:
            row("WARN", f"dex {did}", detail + "; expected ability unknown yet")
        if req:
            ok = _id(req) in _id(items_src)
            row("PASS" if ok else "FAIL", f"stone {req}", "in pinned items.ts" if ok else "MISSING from pinned items.ts")
        if ability and ability != "?":
            aid = _id(ability)
            if aid in enc_src:
                row("PASS", f"encoder {ability}", "known to the mechanic tables")
            elif ability.lower() in known_untagged or aid in _id(known_untagged):
                row("PASS", f"encoder {ability}", "deliberately untagged (_KNOWN_UNTAGGED)")
            else:
                row("WARN", f"encoder {ability}", "NOT in the mechanic tables → decide: category (layout bump) or _KNOWN_UNTAGGED")

    width = max(len(r[1]) for r in rows) if rows else 10
    print(f"Reg {label} readiness — format {fmt}")
    for status, item, detail in rows:
        print(f"  {status:4s}  {item:<{width}}  {detail}")
    fails = sum(1 for r in rows if r[0] == "FAIL")
    warns = sum(1 for r in rows if r[0] == "WARN")
    print(f"  → {fails} FAIL, {warns} WARN, {len(rows) - fails - warns} PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
