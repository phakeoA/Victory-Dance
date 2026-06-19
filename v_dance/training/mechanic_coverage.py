"""15b-feat.robust.1 — dex-grounded, regulation-agnostic mechanic COVERAGE GUARD.

The curated mechanic-tag tables in tp_features.py are the SERVING source (regulation-
independent game facts). Their one risk is OMISSION: a NEW reg's ability/move that bears a
mechanic would be SILENTLY under-tagged (a zeroed channel, no error). This guard closes that
hole by grounding RECALL in the bundled, version-pinned Showdown dex (PINS.md): it enumerates
every ability/move the GAME defines as mechanic-bearing (setWeather/setTerrain in abilities;
weather:/terrain:/sideCondition:/volatileStatus:/pseudoWeather: fields in moves), and flags any
that is PRESENT in the loaded pikalytics data but ABSENT from the curated tables. So a data swap
that introduces an untagged setter/abuser/role fails a test instead of degrading silently — and
fixing it is a one-line table addition the guard names for you.

Two layers (severity):
  * HIGH (must-have): weather/terrain SETTERS + ABUSERS — these are the synergy carriers.
  * WARN: role heuristics (screens/tailwind/redirect/trick_room) — softer, dex pattern-based.
Anything reviewed-and-deliberately-untagged goes in _KNOWN_UNTAGGED.

Used by tests (a data swap goes red on an unmapped must-have) and importable as a runtime audit.
"""
from __future__ import annotations

import re
from pathlib import Path

from v_dance.training import tp_features as TF

_REPO = Path(__file__).resolve().parents[2]
DEX_DIR = _REPO / "pokemon-showdown" / "data"

_WEATHER_ID = {"sandstorm": "sand", "raindance": "rain", "sunnyday": "sun",
               "snowscape": "snow", "snow": "snow", "hail": "snow"}
_TERRAIN_ID = {"electricterrain": "electric", "grassyterrain": "grassy",
               "psychicterrain": "psychic", "mistyterrain": "misty"}

MUST_HAVE = {"weather_setter", "terrain_setter", "weather_abuser", "terrain_abuser", "spread"}

# Dex mechanic-bearers we have REVIEWED and deliberately do NOT tag (so they don't trip the guard).
_KNOWN_UNTAGGED = {
    # strong/illegal-in-format primal weathers (not in M-A; mapped only as future tripwires elsewhere)
    "Desolate Land", "Primordial Sea", "Delta Stream",
    # terrain-AMBIGUOUS moves (type/effect changes with the active terrain — no single axis)
    "Terrain Pulse", "Steel Roller",
    "Mimicry",
    # manual weather move that is also a pivot; weather handled via the setter table already — n/a
}


def _parse_blocks(path: Path):
    """Yield (display_name, block_text) for each top-level dex entry (``\\tid: { ... }``)."""
    text = path.read_text(encoding="utf-8")
    starts = [m.start() for m in re.finditer(r"(?m)^\t(\w+): \{", text)]
    starts.append(len(text))
    for i in range(len(starts) - 1):
        block = text[starts[i]:starts[i + 1]]
        nm = re.search(r'\n\t\tname: "([^"]+)"', block) or re.search(r'name: "([^"]+)"', block)
        if nm:
            yield nm.group(1), block


_STAT_HOOK = re.compile(r"onModifySpe|onModifyAtk|onModifySpA|onModifyDef|onBasePower")


def dex_mechanic_bearers(dex_dir: Path = DEX_DIR) -> dict:
    """{display_name: set(mechanic_class)} for every dex ability/move that bears a mechanic.
    Empty if the dex is unavailable (the guard then no-ops rather than crashing)."""
    out: dict = {}

    def add(name, cls):
        out.setdefault(name, set()).add(cls)

    abilities, moves = dex_dir / "abilities.ts", dex_dir / "moves.ts"
    if abilities.exists():
        for name, block in _parse_blocks(abilities):
            if any(w in _WEATHER_ID for w in re.findall(r"setWeather\('(\w+)'\)", block)):
                add(name, "weather_setter")
            if any(t in _TERRAIN_ID for t in re.findall(r"setTerrain\('(\w+)'\)", block)):
                add(name, "terrain_setter")
            if re.search(r"isWeather\(|effectiveWeather\(", block) and _STAT_HOOK.search(block):
                add(name, "weather_abuser")
            if re.search(r"isTerrain\(", block) and _STAT_HOOK.search(block):
                add(name, "terrain_abuser")
            if re.search(r"onChangeBoost", block) and re.search(r"\*=?\s*-1", block):
                add(name, "stat_reverser")     # Contrary-style (reverses stat drops into boosts); warn
    if moves.exists():
        for name, block in _parse_blocks(moves):
            if any(w in _WEATHER_ID for w in re.findall(r"weather: '(\w+)'", block)):
                add(name, "weather_setter")
            if any(t in _TERRAIN_ID for t in re.findall(r"terrain: '(\w+)'", block)):
                add(name, "terrain_setter")
            if re.search(r"sideCondition: '(reflect|lightscreen|auroraveil)'", block):
                add(name, "screens")
            if re.search(r"sideCondition: 'tailwind'", block):
                add(name, "tailwind")
            if re.search(r"volatileStatus: '(followme|ragepowder)'", block):
                add(name, "redirect")
            if re.search(r"pseudoWeather: 'trickroom'", block):
                add(name, "trick_room")
            if any(t == "allAdjacent" for t in re.findall(r'target: "(\w+)"', block)):
                add(name, "spread")            # ally-hitting spread move (15b-feat.spread)
    return out


def curated_names() -> set:
    """Every ability/move NAME the tp_features tag tables map (the serving source)."""
    names = (set(TF.SETTER_ABILITY) | set(TF.ABUSER_ABILITY) | set(TF.SETTER_MOVE)
             | set(TF.TERRAIN_SETTER_ABILITY) | set(TF.TERRAIN_ABUSER_ABILITY)
             | set(TF.TERRAIN_SETTER_MOVE) | set(TF.TERRAIN_ABUSER_MOVE)
             | set(TF.SPREAD_MOVE) | set(TF.IMMUNITY_ABILITY)
             | set(TF.STAT_REVERSER_ABILITY) | set(TF.ALLY_DEBUFF_MOVE) | set(TF.ORDER_ABILITY))
    for s in TF.ROLE_MOVE.values():
        names |= set(s)
    for s in TF.ROLE_ABILITY.values():
        names |= set(s)
    return names


def _diff(bearers: dict, present: set, curated: set) -> list:
    """Pure: mechanic-bearers PRESENT in the data but unmapped (and not deliberately untagged)."""
    findings = []
    for name, classes in bearers.items():
        if name not in present or name in curated or name in _KNOWN_UNTAGGED:
            continue
        findings.append({"name": name, "classes": sorted(classes),
                         "severity": "high" if classes & MUST_HAVE else "warn"})
    return sorted(findings, key=lambda f: (f["severity"] != "high", f["name"]))


def names_present_in_data(belief) -> set:
    """Every ability + move NAME any species runs in the loaded pikalytics data."""
    present = set()
    for sp in belief.all_pokemon():
        for a in belief.ability_distribution(sp, top_k=8):
            present.add(a["name"])
        for m in belief.move_distribution(sp, top_k=20):
            present.add(m["name"])
    return present


def audit_mechanic_coverage(belief, dex_dir: Path = DEX_DIR) -> list:
    """Findings for mechanic-bearing abilities/moves PRESENT in the loaded data but UNMAPPED.
    [] means full coverage. severity 'high' = a must-have setter/abuser is missing (FAIL the
    build); 'warn' = a softer role heuristic is missing (review)."""
    return _diff(dex_mechanic_bearers(dex_dir), names_present_in_data(belief), curated_names())
