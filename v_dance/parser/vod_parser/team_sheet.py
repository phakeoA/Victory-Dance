"""
team_sheet.py
=============
Parse a Pokémon Showdown team paste (the "export to text" format) into the
Victory-Dance inject shapes used for Type A (own-VOD) exact-stats enrichment.

A paste looks like::

    Sneasler @ White Herb
    Ability: Unburden
    Level: 50
    EVs: 2 HP / 32 Atk / 32 Spe
    Jolly Nature
    - Fake Out
    - Dire Claw
    - Protect
    - Close Combat

    Charizard-Mega-Y @ Charizardite Y
    ...

Public surface
--------------
    parse_showdown_team(text)   -> list[dict]   # one entry per Pokémon
    team_to_known_side(mons)    -> dict          # {base_species: inject-dict}
    detect_our_side(rosters, base_species) -> "p1"|"p2"|None

The same parsing logic is mirrored client-side in tb_parser.js
(``parseShowdownTeam``) so the team-builder UI's "Import team" button and the
headless bulk exporter produce identical inject data.
"""

from __future__ import annotations

import re
from typing import Optional

from v_dance.parser.vod_parser.pokedex import get_pokedex, is_mega_species_name, norm_species

# Showdown EV/IV stat labels → internal stat keys.
_STAT_LABELS = {
    "hp": "hp", "atk": "atk", "def": "def",
    "spa": "spa", "spd": "spd", "spe": "spe",
}
_STAT_ORDER = ("hp", "atk", "def", "spa", "spd", "spe")

_NATURES = {
    "hardy", "lonely", "brave", "adamant", "naughty",
    "bold", "docile", "relaxed", "impish", "lax",
    "timid", "hasty", "serious", "jolly", "naive",
    "modest", "mild", "quiet", "bashful", "rash",
    "calm", "gentle", "sassy", "careful", "quirky",
}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_showdown_team(text: str) -> list[dict]:
    """Parse a full team paste into a list of per-Pokémon dicts.

    Each dict::

        {
          "species":   "Charizard-Mega-Y",   # as written in the paste
          "nickname":  None,
          "gender":    None | "M" | "F" | "N",
          "item":      "Charizardite Y" | None,
          "ability":   "Solar Power" | None,  # the BASE forme's ability
          "level":     50,
          "nature":    "Modest" | None,
          "evs":       {"hp": 15, ...},        # only the invested stats
          "ivs":       {"atk": 0, ...},        # only the specified stats
          "moves":     ["Heat Wave", ...],     # up to 4
          "tera_type": "Fire" | None,
        }
    """
    mons: list[dict] = []
    # Normalise CRLF/CR → LF: Path.read_text already does this, but a caller
    # may hand us a raw upload (e.g. a web POST body) with Windows \r\n, which
    # the blank-line block separator below would otherwise fail to split on.
    text = re.sub(r"\r\n?", "\n", text or "")
    # Pokémon are separated by one or more blank lines.
    for block in re.split(r"\n[ \t]*\n", text.strip()):
        lines = [ln.rstrip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        mon = _parse_mon(lines)
        if mon and mon.get("species"):
            mons.append(mon)
    return mons


def _parse_mon(lines: list[str]) -> Optional[dict]:
    species, nickname, item, gender = _parse_first_line(lines[0])
    mon: dict = {
        "species": species, "nickname": nickname, "gender": gender,
        "item": item, "ability": None, "level": 50, "nature": None,
        "evs": {}, "ivs": {}, "moves": [], "tera_type": None,
    }
    for raw in lines[1:]:
        line = raw.strip()
        low = line.lower()
        if low.startswith("ability:"):
            mon["ability"] = line.split(":", 1)[1].strip() or None
        elif low.startswith("level:"):
            try:
                mon["level"] = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif low.startswith("tera type:"):
            mon["tera_type"] = line.split(":", 1)[1].strip() or None
        elif low.startswith("evs:"):
            mon["evs"] = _parse_stat_line(line)
        elif low.startswith("ivs:"):
            mon["ivs"] = _parse_stat_line(line)
        elif line.startswith("-"):
            mv = line[1:].strip()
            # Strip any "[Hidden Power Fire]" style bracket and the trailing
            # PP/Tera glyphs Showdown never emits in a clean export.
            if mv and len(mon["moves"]) < 4:
                mon["moves"].append(mv)
        elif low.endswith("nature") and line.split()[0].lower() in _NATURES:
            mon["nature"] = line.rsplit(" ", 1)[0].strip()
        # Shiny:/Happiness:/Gigantamax: etc. are intentionally ignored.
    return mon


def _parse_first_line(line: str) -> tuple[str, Optional[str], Optional[str], Optional[str]]:
    """Split the first line into (species, nickname, item, gender).

    Handles ``Species @ Item``, ``Nickname (Species) @ Item``,
    ``Species (M) @ Item`` and ``Nickname (Species) (F) @ Item``.
    """
    item = None
    namepart = line.strip()
    if " @ " in namepart:
        namepart, item = namepart.rsplit(" @ ", 1)
        item = item.strip() or None
        namepart = namepart.strip()

    gender = None
    m = re.search(r"\(([MFN])\)\s*$", namepart)
    if m:
        gender = m.group(1)
        namepart = namepart[: m.start()].strip()

    species = namepart
    nickname = None
    m2 = re.match(r"^(.*?)\s*\(([^)]+)\)\s*$", namepart)
    if m2:
        nickname = m2.group(1).strip() or None
        species = m2.group(2).strip()

    return species, nickname, item, gender


def _parse_stat_line(line: str) -> dict:
    """Parse ``EVs: 2 HP / 32 Atk / 32 Spe`` → {"hp": 2, "atk": 32, "spe": 32}."""
    out: dict = {}
    body = line.split(":", 1)[1] if ":" in line else line
    for part in body.split("/"):
        m = re.match(r"\s*(\d+)\s+([A-Za-z]+)", part)
        if not m:
            continue
        key = _STAT_LABELS.get(m.group(2).lower())
        if key:
            out[key] = int(m.group(1))
    return out


# ---------------------------------------------------------------------------
# Conversion to inject shapes
# ---------------------------------------------------------------------------

def base_species(species: str) -> str:
    """Resolve a paste species to the name used in a replay roster.

    Mega formes appear under their BASE name at teampreview (a mon only mega
    evolves mid-battle), so ``Charizard-Mega-Y`` → ``Charizard``.  Non-mega
    formes keep their own name (``Floette-Eternal``, ``Rotom-Wash`` stay as-is).
    """
    dex = get_pokedex()
    if dex and dex.is_mega_forme(species):
        e = dex.entry(species) or {}
        return e.get("baseSpecies") or re.sub(r"-Mega(-[XY])?$", "", species)
    if is_mega_species_name(species):
        return re.sub(r"-Mega(-[XY])?$", "", species)
    return species


def team_to_known_side(mons: list[dict]) -> dict:
    """Convert parsed mons into a ``known_teams`` side dict.

    Keyed by BASE species (matching replay roster / inject-card keys), each
    value is the inject shape ``_known_entry_side_to_sheet`` consumes:
    ``{nature, ev_spread, item, ability, moves}``.  ev_spread carries all six
    stats (0 for the uninvested ones) so the exact-stat computation is total.
    """
    out: dict = {}
    for mon in mons:
        key = base_species(mon.get("species") or "")
        if not key:
            continue
        evs = mon.get("evs") or {}
        out[key] = {
            "nature": mon.get("nature"),
            "ev_spread": {s: int(evs.get(s, 0)) for s in _STAT_ORDER},
            "item": mon.get("item"),
            "ability": mon.get("ability"),
            "moves": list(mon.get("moves") or [])[:4],
        }
    return out


def detect_our_side(rosters: dict, base_species_set) -> Optional[str]:
    """Return the side ("p1"/"p2") whose roster best matches the team.

    For a Type A own-VOD the player's six revealed mons equal the team paste,
    so the matching side is unambiguous.  Returns None if neither side overlaps.
    """
    want = {norm_species(s) for s in base_species_set}
    best_pid, best_overlap = None, 0
    for pid in ("p1", "p2"):
        have = {norm_species(s) for s in (rosters.get(pid) or [])}
        overlap = len(want & have)
        if overlap > best_overlap:
            best_pid, best_overlap = pid, overlap
    return best_pid
