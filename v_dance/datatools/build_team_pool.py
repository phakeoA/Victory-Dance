"""Build a pool of legal Reg M-C teams from tournament LINEUPS + the live usage sets (self-play league, 2026-09-29).

A lineup (six species, Mega formes labelled) is real tournament data, so the six already work together; each member's
SET comes from ``data/belief_<reg>.json`` (MunchStats first, Pikalytics fallback): top ability, top four moves, top
spread with the modal nature, and the most common item — with Item Clause enforced (the next item when one is taken),
the Mega's stone from the pinned Showdown dex, and no Fake Out on a team whose own Psychic Surge would block it.
Every team is checked by Showdown's own validator; an illegal one is reported and skipped.

    python -m v_dance.datatools.build_team_pool --lineups data/regulations/baltimore_2027_lineups.txt --dry-run
    python -m v_dance.datatools.build_team_pool --lineups data/regulations/baltimore_2027_lineups.txt
Lineup file: one team per line, ``<place>: A, B, C, D, E, F`` (e.g. ``1: Salamence Mega, Tyranitar Mega, ...``).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_STATS = ("HP", "Atk", "Def", "SpA", "SpD", "Spe")
_NODE = _REPO / ".venv" / "node" / "node.exe"
_RESOLVE_JS = """
const {Dex} = require('./dist/sim'); const out = {};
for (const id of JSON.parse(process.argv[1])) {
  const s = Dex.species.get(id);
  if (!s.exists) { out[id] = null; continue; }
  const mega = !!s.isMega;
  const base = mega ? (s.changesFrom || s.baseSpecies) : (s.isCosmeticForme ? s.baseSpecies : s.name);
  out[id] = {name: base, mega, item: mega ? (s.requiredItem || null) : null};
}
console.log(JSON.stringify(out));
"""


def _id(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def read_lineups(path: Path) -> list[tuple[str, list[str]]]:
    teams, seen = [], set()
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*(\d+)\s*:\s*(.+)", line)
        if not m:
            continue
        mons = [x.strip() for x in m.group(2).split(",") if x.strip()]
        key = tuple(sorted(_id(x) for x in mons))
        if len(mons) == 6 and key not in seen:          # identical lineups add nothing to the pool
            seen.add(key)
            teams.append((m.group(1), mons))
    return teams


def resolve_species(labels: set[str]) -> dict:
    ids = sorted({_id(x) for x in labels})
    res = subprocess.run([str(_NODE), "-e", _RESOLVE_JS, json.dumps(ids)], cwd=_REPO / "pokemon-showdown",
                         capture_output=True, text=True, check=True)
    return json.loads(res.stdout)


def build_set(name: str, entry: dict, item: str, blocked_moves: set[str]) -> str:
    ability = entry["abilities"][0]["name"] if entry.get("abilities") else None
    moves = [m["name"] for m in entry.get("moves", []) if m["name"] not in blocked_moves][:4]
    spread = (entry.get("spreads") or [None])[0]
    lines = [f"{name} @ {item}" if item else name]
    if ability:
        lines.append(f"Ability: {ability}")
    if spread:
        evs = " / ".join(f"{v} {s}" for v, s in zip(spread["evs"], _STATS) if v)
        lines.append(f"EVs: {evs}")
        if spread.get("nature"):
            lines.append(f"{spread['nature']} Nature")
    lines += [f"- {m}" for m in moves]
    return "\n".join(lines)


def build_team(mons: list[str], species: dict, belief: dict) -> tuple[str, list[str]]:
    """Return (paste, problems)."""
    problems, used_items, sets = [], set(), []
    infos = [species.get(_id(m)) for m in mons]
    psychic_surge = any(((belief.get(i["name"]) or {}).get("abilities") or [{}])[0].get("name") == "Psychic Surge"
                        for i in infos if i)
    for label, info in zip(mons, infos):
        if not info:
            problems.append(f"unknown species {label!r}")
            continue
        entry = belief.get(info["name"]) or {}
        if not entry.get("moves"):
            problems.append(f"no usage data for {info['name']}")
            continue
        if info["mega"]:
            item = info["item"]
        else:
            item = next((it["name"] for it in entry.get("items", [])
                         if it["name"] not in used_items and not it["name"].endswith(("ite", "ite X", "ite Y", "ite Z"))),
                        None)
        if item:
            used_items.add(item)
        blocked = {"Fake Out"} if psychic_surge else set()
        sets.append(build_set(info["name"], entry, item, blocked))
    return "\n\n".join(sets) + "\n", problems


def team_name(place: str, mons: list[str]) -> str:
    megas = [m.split(" Mega")[0].replace(" ", "") for m in mons if " Mega" in m]
    return f"Balt27_{int(place):02d}_{'_'.join(megas) or _id(mons[0])}"


def main() -> None:
    from v_dance.datatools.validate_teams import validate
    from v_dance.formats import default_format, reg_token

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lineups", required=True)
    ap.add_argument("--format", default=None, help="format id (default: the active format)")
    ap.add_argument("--out", default="teams/Champions/M-C", help="team folder (the league's M-C pool)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    fmt = a.format or default_format()
    belief = json.loads((_REPO / "data" / f"belief_{reg_token(fmt)}.json").read_text(encoding="utf-8"))["pokemon"]
    lineups = read_lineups(_REPO / a.lineups)
    species = resolve_species({m for _, mons in lineups for m in mons})
    out_dir = _REPO / a.out
    ok = bad = 0
    for place, mons in lineups:
        paste, problems = build_team(mons, species, belief)
        legal, errs = (False, problems) if problems else validate(paste, fmt)
        name = team_name(place, mons)
        if not legal:
            bad += 1
            print(f"  SKIP {name}: {'; '.join(errs)[:160]}")
            continue
        ok += 1
        if not a.dry_run:
            (out_dir / name).write_text(paste, encoding="utf-8")
    print(f"[pool] {len(lineups)} distinct lineups -> {ok} legal teams{' (dry run)' if a.dry_run else f' in {a.out}'}, "
          f"{bad} skipped")


if __name__ == "__main__":
    main()
