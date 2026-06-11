"""
battle_models.py
================
Dataclasses that represent the in-battle state of a single Pokémon Showdown
match.  Imported by replay_parser.py and transitions.py; no circular deps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PokemonSlot:
    """Tracks a single Pokémon's in-battle state snapshot."""
    species: str
    nickname: str
    player: str          # "p1" or "p2"
    slot: str            # "a" or "b"  (active field position)
    hp_current: Optional[float] = None   # percentage 0-100
    hp_max: Optional[float] = 100.0
    status: Optional[str] = None
    boosts: dict = field(default_factory=dict)
    is_mega: bool = False
    is_fainted: bool = False
    # --- new in v2 ---
    revealed_moves: list = field(default_factory=list)   # moves seen this match
    known_item: Optional[str] = None                     # item if revealed
    known_tera_type: Optional[str] = None                # tera type if revealed
    is_terastallized: bool = False
    known_ability: Optional[str] = None                  # ability if revealed via |-ability|
    # Bug 7: teampreview/roster species name, frozen at first switch-in.
    # `species` mutates on mega evolution (|detailschange|), so bench/roster
    # reconciliation must compare against this instead.
    base_species: Optional[str] = None

    def key(self) -> str:
        return f"{self.player}{self.slot}"

    def to_dict(self) -> dict:
        return {
            "species": self.species,
            "nickname": self.nickname,
            "player": self.player,
            "slot": self.slot,
            "hp_pct": self.hp_current,
            "status": self.status,
            "boosts": dict(self.boosts),
            "is_mega": self.is_mega,
            "is_fainted": self.is_fainted,
            "revealed_moves": list(self.revealed_moves),
            "known_item": "mega stone" if self.is_mega else self.known_item,
            "known_tera_type": self.known_tera_type,
            "is_terastallized": self.is_terastallized,
            # EVs/IVs unknown for Type B — left as distribution placeholder
            "ev_spread": None,
            "iv_spread": None,
            "nature": None,
        }


@dataclass
class SideConditions:
    tailwind: int = 0        # turns remaining (0 = inactive)
    screens: dict = field(default_factory=dict)   # reflect / light screen / aurora veil

    def to_dict(self) -> dict:
        return {
            "tailwind_turns_remaining": self.tailwind,
            "screens": dict(self.screens),
        }


@dataclass
class FieldConditions:
    weather: Optional[str] = None
    terrain: Optional[str] = None
    trick_room: int = 0      # turns remaining

    def to_dict(self) -> dict:
        return {
            "weather": self.weather,
            "terrain": self.terrain,
            "trick_room_turns_remaining": self.trick_room,
        }
