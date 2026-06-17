"""
pokedex.py
==========
Thin lookup layer over data/pokedex.json.

This is the single source of truth for species → ability resolution, used by:
  - replay_parser.py  — to resolve the (single, fixed) ability of a mega forme
                        the instant |detailschange| fires.
  - transitions.py    — to inject the correct ability into training transitions
                        (pre-mega ability before mega, mega ability after).
  - server.py         — exposes pokedex_loaded on /health.

Key domain rule encoded here
----------------------------
A Pokémon that mega evolves has TWO ability contexts:
  * pre-mega:  one of its base forme's possible abilities (player's choice,
               unknown until revealed or injected by the user), and
  * post-mega: the mega forme's ability, which is ALWAYS exactly one and is
               fully determined by the mega species — never a player choice.

So the moment a mega evolution is observed, the active ability is known with
certainty from the pokedex, and any previously revealed ability must be
demoted to "pre-mega ability" rather than kept as the current one.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

# v_dance/parser/vod_parser/pokedex.py → parents[3] == repo root, holds data/
_DEFAULT_POKEDEX_PATH = Path(__file__).resolve().parents[3] / "data" / "pokedex.json"

_NON_ALNUM = re.compile(r"[^a-z0-9]")


def norm_species(species: Optional[str]) -> str:
    """Normalise a display species name to a pokedex key.

    "Floette-Mega" → "floettemega", "Rotom-Wash" → "rotomwash",
    "Mr. Mime" → "mrmime".
    """
    if not species:
        return ""
    return _NON_ALNUM.sub("", species.lower())


def is_mega_species_name(species: Optional[str]) -> bool:
    """Heuristic check on the display name alone (no dex needed).

    True for "Venusaur-Mega", "Charizard-Mega-X"; False for "Meganium",
    "Palafin-Hero", "Yanmega" (suffix match only, not substring).
    """
    if not species:
        return False
    return species.endswith("-Mega") or "-Mega-" in species


class Pokedex:
    """Loaded pokedex with ability/forme lookup helpers.

    All lookups are safe on unknown species — they return None / [] rather
    than raising, so the parser degrades gracefully when the dex doesn't
    cover a custom format species.
    """

    def __init__(self, path: Optional[Path] = None, data: Optional[dict] = None):
        if data is not None:
            self._dex = data
        else:
            p = Path(path) if path else _DEFAULT_POKEDEX_PATH
            self._dex = json.loads(Path(p).read_text(encoding="utf-8"))

    # ── basic lookups ────────────────────────────────────────────────────
    def entry(self, species: Optional[str]) -> Optional[dict]:
        return self._dex.get(norm_species(species))

    def has(self, species: Optional[str]) -> bool:
        return norm_species(species) in self._dex

    def abilities_for(self, species: Optional[str]) -> list[str]:
        """All possible abilities for a forme, in dex slot order (0, 1, H, S)."""
        e = self.entry(species)
        if not e:
            return []
        ab = e.get("abilities") or {}
        order = ["0", "1", "H", "S"]
        out = [ab[k] for k in order if k in ab and ab[k]]
        # Catch any non-standard slot keys
        out += [v for k, v in ab.items() if k not in order and v and v not in out]
        return out

    # ── mega-forme logic ─────────────────────────────────────────────────
    def is_mega_forme(self, species: Optional[str]) -> bool:
        """True if `species` IS a mega forme (dex-confirmed when possible)."""
        e = self.entry(species)
        if e is not None:
            forme = (e.get("forme") or "")
            return forme.startswith("Mega")
        # Unknown to the dex — fall back to the name heuristic
        return is_mega_species_name(species)

    def mega_ability_for(self, mega_species: Optional[str]) -> Optional[str]:
        """The single fixed ability of a mega forme, or None if unknown.

        Mega formes always have exactly one ability; if the dex entry for a
        supposed mega lists several, we refuse to guess and return None.
        """
        if not self.is_mega_forme(mega_species):
            return None
        abilities = self.abilities_for(mega_species)
        return abilities[0] if len(abilities) == 1 else None

    def mega_formes_for(self, base_species: Optional[str]) -> list[dict]:
        """All mega formes reachable from a base species.

        Returns [{"forme": "Venusaur-Mega", "ability": "Thick Fat"}, ...].
        Handles X/Y twin megas (two entries). Empty list if none.
        """
        e = self.entry(base_species)
        if not e:
            return []
        # Resolve any non-base forme (Eternal, Wash, Mega, …) to its base
        # entry, since otherFormes lives on the base species only.
        if e.get("baseSpecies"):
            e = self.entry(e["baseSpecies"]) or e

        out = []
        for forme_name in e.get("otherFormes", []) or []:
            fe = self.entry(forme_name)
            if fe and (fe.get("forme") or "").startswith("Mega"):
                out.append({
                    "forme": forme_name,
                    "ability": self.mega_ability_for(forme_name),
                })
        return out


# ── Lazy module-level singleton ──────────────────────────────────────────
_singleton: Optional[Pokedex] = None
_singleton_failed = False


def get_pokedex() -> Optional[Pokedex]:
    """Return the shared Pokedex, or None if data/pokedex.json is missing.

    Never raises — callers must handle None (graceful degradation: mega
    abilities then stay unknown until an |-ability| line reveals them).
    """
    global _singleton, _singleton_failed
    if _singleton is None and not _singleton_failed:
        try:
            _singleton = Pokedex()
        except (OSError, json.JSONDecodeError):
            _singleton_failed = True
    return _singleton
