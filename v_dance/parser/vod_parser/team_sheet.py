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
    parse_packed_team(packed)   -> list[dict]   # Showdown PACKED format (|showteam| payload)
    parse_showteam_sides(log)   -> dict         # {"p1": [mon,...], "p2": [...]} from |showteam| lines
    team_to_known_side(mons)    -> dict          # {base_species: inject-dict}
    detect_our_side(rosters, base_species) -> "p1"|"p2"|None

The same parsing logic is mirrored client-side in tb_parser.js
(``parseShowdownTeam``) so the team-builder UI's "Import team" button and the
headless bulk exporter produce identical inject data.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
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
# Packed-team parsing (the |showteam| open-team-sheet payload)
# ---------------------------------------------------------------------------
# Showdown packs OTS team sheets as `]`-joined mons of `|`-joined fields:
#   nick|species|item|ability|moves|nature|evs|gender|ivs|shiny|level|misc
# Species keep their display name ("Floette-Eternal", "Mr. Rime"); item/
# ability/move names are packName'd — non-alphanumerics stripped, case kept
# ("FocusSash", "WillOWisp", "Uturn").  Champions OTS strips EVs/IVs but
# DOES reveal nature (verified on the VGC-Bench regmb logs, 2026-07-01).

# data/ lives three levels up from this file (v_dance/parser/vod_parser/).
_DATA_DIR = Path(__file__).resolve().parents[3] / "data"

_CAMEL_SPLIT = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

_display_maps: Optional[dict] = None


def _get_display_maps() -> dict:
    """norm_species(key) → display name, for moves / items / abilities.

    Built once from the repo data files; every map degrades to {} when its
    source is missing (the CamelCase-split fallback still applies, and all
    downstream comparisons normalise via norm_species anyway).
    """
    global _display_maps
    if _display_maps is None:
        moves: dict = {}
        items: dict = {}
        abilities: dict = {}
        try:
            data = json.loads((_DATA_DIR / "moves.json").read_text(encoding="utf-8"))
            moves = {k: (v.get("name") or k) for k, v in data.items()}
        except (OSError, json.JSONDecodeError, AttributeError):
            pass
        try:
            dexdata = json.loads((_DATA_DIR / "pokedex.json").read_text(encoding="utf-8"))
            for e in dexdata.values():
                for ab in (e.get("abilities") or {}).values():
                    if ab:
                        abilities.setdefault(norm_species(ab), ab)
        except (OSError, json.JSONDecodeError, AttributeError):
            pass
        # Serebii item lists wrap names in markdown links: "[Focus Sash](url)".
        for fname in ("serebii_champions_items_regma.json",
                      "serebii_champions_items_regmb.json"):
            try:
                d = json.loads((_DATA_DIR / fname).read_text(encoding="utf-8"))
                for it in d.get("items") or []:
                    m = re.match(r"\[([^\]]+)\]", str(it.get("name") or ""))
                    if m:
                        items.setdefault(norm_species(m.group(1)), m.group(1))
            except (OSError, json.JSONDecodeError, AttributeError):
                pass
        _display_maps = {"moves": moves, "items": items, "abilities": abilities}
    return _display_maps


def _packed_display(kind: str, raw: str) -> Optional[str]:
    """Map a packName'd token ("FocusSash") to its display name ("Focus Sash").

    Lookup via the data files first (restores hyphens/apostrophes: "Uturn" →
    "U-turn"); falls back to a CamelCase split, which is norm_species-identical
    to the true display name for anything the data files miss.
    """
    raw = (raw or "").strip()
    if not raw:
        return None
    return _get_display_maps().get(kind, {}).get(norm_species(raw)) or _CAMEL_SPLIT.sub(" ", raw)


def parse_packed_team(packed: str) -> list[dict]:
    """Parse Showdown's packed team format (the ``|showteam|pN|`` payload)
    into the same per-mon dicts :func:`parse_showdown_team` produces.

    OTS payloads carry no EV/IV data (``evs``/``ivs`` stay ``{}``) — those
    mons must remain belief-estimated, never sheet-exact.  Nature IS present
    and is preserved for spread narrowing.
    """
    mons: list[dict] = []
    for chunk in (packed or "").split("]"):
        if not chunk.strip():
            continue
        f = chunk.split("|")
        f += [""] * (12 - len(f))
        name, species_f, item, ability, moves, nature, evs, gender, ivs, _shiny, level, misc = f[:12]
        # Field 1 is the nickname-or-species; field 2 is the packName'd species
        # ONLY when it differs from field 1 (i.e. the mon is nicknamed).
        species = (species_f or "").strip() or (name or "").strip()
        if not species:
            continue
        nickname = None
        if (species_f or "").strip() and norm_species(name) != norm_species(species_f):
            nickname = (name or "").strip() or None
            # A packName'd species field lost its punctuation — best-effort
            # display restore (norm_species-identical either way).
            species = _CAMEL_SPLIT.sub(" ", species)
        mon: dict = {
            "species": species,
            "nickname": nickname,
            "gender": (gender or "").strip() or None,
            "item": _packed_display("items", item),
            "ability": _packed_display("abilities", ability),
            "level": int(level) if (level or "").strip().isdigit() else 50,
            "nature": (nature or "").strip() or None,
            "evs": {},
            "ivs": {},
            "moves": [_packed_display("moves", mv)
                      for mv in (moves or "").split(",") if mv.strip()][:4],
            "tera_type": None,
        }
        # EV/IV fields (comma-sep hp,atk,def,spa,spd,spe; blank slot = absent).
        # Stripped in Champions OTS — parsed defensively for other sources.
        for field, key in ((evs, "evs"), (ivs, "ivs")):
            parts = (field or "").split(",")
            if any(p.strip() for p in parts):
                for stat, val in zip(_STAT_ORDER, parts):
                    v = val.strip()
                    if v.lstrip("-").isdigit():
                        mon[key][stat] = int(v)
        # misc = happiness,pokeball,hiddenpowertype,gigantamax,dynamaxlevel,teratype
        tail = (misc or "").split(",")
        if len(tail) >= 6 and tail[5].strip():
            mon["tera_type"] = tail[5].strip()
        mons.append(mon)
    return mons


def parse_showteam_sides(log: str) -> dict[str, list[dict]]:
    """Extract both sides' open team sheets from a raw Showdown protocol log.

    Returns ``{"p1": [mon, ...], "p2": [...]}`` — empty dict when the log has
    no ``|showteam|`` lines (closed team sheets).  The packed payload contains
    literal ``|`` separators, so the line is captured whole, never split.
    """
    out: dict[str, list[dict]] = {}
    for m in re.finditer(r"^\|showteam\|(p[12])\|(.*)$", log or "", re.MULTILINE):
        mons = parse_packed_team(m.group(2))
        if mons:
            out[m.group(1)] = mons
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


def packed_team_to_known_side(mons: list[dict]) -> dict:
    """OTS variant of :func:`team_to_known_side` for packed (|showteam|) mons.

    A packed OTS payload has its EVs STRIPPED — unknown, not zero — so
    ``ev_spread`` stays ``None`` (``team_to_known_side``'s all-zero default
    would let the exact-stats path bake a 0-EV spread as ground truth).
    Nature/item/ability/moves are revealed and carried through.
    """
    out: dict = {}
    for mon in mons:
        key = base_species(mon.get("species") or "")
        if not key:
            continue
        evs = mon.get("evs") or {}
        out[key] = {
            "nature": mon.get("nature"),
            "ev_spread": ({s: int(evs.get(s, 0)) for s in _STAT_ORDER} if evs else None),
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
