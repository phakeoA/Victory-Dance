"""
belief_state.py
===============
Converts raw Pikalytics JSON into a BeliefState lookup used by the
VOD parser and live battle state encoder.

When an opponent Pokémon's true moveset/EVs are unknown, the belief
state fills in the *most likely* values from meta usage data — exactly
what a top human VGC player does during Team Preview.

USAGE
─────
    from belief_state import BeliefState

    bs = BeliefState("pikalytics_regma.json")

    # Most likely ability for an unseen mon
    ability = bs.top_ability("Kingambit")     # → "Defiant"

    # Weighted moveset (up to 4 most-used moves)
    moves = bs.top_moves("Kingambit", n=4)
    # → [("Sucker Punch", 99.3), ("Kowtow Cleave", 88.3), ...]

    # Best item
    item = bs.top_item("Kingambit")           # → "Chople Berry"

    # Best EV spread (nature + 6 EVs in HP/Atk/Def/SpA/SpD/Spe order)
    nature, evs = bs.top_spread("Kingambit")  # → ("Adamant", [32,32,0,0,2,0])

    # Full belief dict (for serialization / display)
    belief = bs.get_belief("Kingambit")

INTEGRATION WITH state_encoder.py
──────────────────────────────────
The StateEncoder._write_pokemon() method already reads from a
poke-env Pokemon object.  The VOD parser (vod_parser.py) creates
lightweight FakePokemon objects — dicts that quack like Pokemon —
populated from this BeliefState wherever true data is unknown.

See vod_parser.py for the integration pattern.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional


# ── Stat formula helpers ───────────────────────────────────────────────────────
# EV buckets from Pikalytics use 32-step notation (0–32 in steps of 1,
# corresponding to 0–252 EVs in steps of 8 approx).
# Exact formula: actual_ev = ev_bucket * 8   (max 252 → bucket 31.5 → use 32)
# For damage-formula back-solving we need actual EVs, not buckets.

NATURE_BOOSTS: dict[str, tuple[str, str]] = {
    # nature: (boosted_stat, reduced_stat) — None stat = neutral
    "Adamant":  ("atk", "spa"),
    "Modest":   ("spa", "atk"),
    "Jolly":    ("spe", "spa"),
    "Timid":    ("spe", "atk"),
    "Brave":    ("atk", "spe"),
    "Quiet":    ("spa", "spe"),
    "Impish":   ("def", "spa"),
    "Bold":     ("def", "atk"),
    "Careful":  ("spd", "spa"),
    "Calm":     ("spd", "atk"),
    "Sassy":    ("spd", "spe"),
    "Relaxed":  ("def", "spe"),
    "Gentle":   ("spd", "def"),
    "Hasty":    ("spe", "def"),
    "Naive":    ("spe", "spd"),
    "Lax":      ("def", "spd"),
    "Rash":     ("spa", "spd"),
    "Naughty":  ("atk", "spd"),
    "Lonely":   ("atk", "def"),
    "Mild":     ("spa", "def"),
    # Neutral natures (no boost/drop)
    "Hardy": ("", ""), "Docile": ("", ""), "Serious": ("", ""),
    "Bashful": ("", ""), "Quirky": ("", ""),
}

STAT_ORDER = ["hp", "atk", "def", "spa", "spd", "spe"]


def ev_bucket_to_evs(buckets: list[int]) -> list[int]:
    """Convert Pikalytics 0–32 bucket notation to actual 0–252 EVs."""
    return [min(b * 8, 252) for b in buckets]


def calc_stat(
    base: int,
    ev: int,
    iv: int,
    level: int,
    nature: str,
    stat_name: str,
) -> int:
    """
    Standard Gen 6+ stat formula (used in Gen 9).
    Returns the in-battle stat value.
    """
    if stat_name == "hp":
        return int((2 * base + iv + ev // 4) * level / 100) + level + 10
    else:
        raw = int((2 * base + iv + ev // 4) * level / 100) + 5
        boost, drop = NATURE_BOOSTS.get(nature, ("", ""))
        if stat_name == boost:
            raw = int(raw * 1.1)
        elif stat_name == drop:
            raw = int(raw * 0.9)
        return raw


def calc_full_stats(
    base_stats: dict[str, int],
    evs: list[int],           # HP Atk Def SpA SpD Spe (actual EVs 0-252)
    nature: str,
    ivs: Optional[list[int]] = None,  # defaults to 31 all
    level: int = 50,
) -> dict[str, int]:
    """Return the full 6-stat dict for a mon at L50 with given EVs/nature."""
    if ivs is None:
        ivs = [31] * 6
    return {
        stat: calc_stat(base_stats[stat], evs[i], ivs[i], level, nature, stat)
        for i, stat in enumerate(STAT_ORDER)
    }


# ── BeliefState ────────────────────────────────────────────────────────────────
class BeliefState:
    """
    Wraps pikalytics_regma.json and provides fast lookups for the most
    likely moveset, item, ability, and EV spread for any meta Pokémon.
    """

    def __init__(self, json_path: str | Path = "pikalytics_regma.json"):
        self._path = Path(json_path)
        with self._path.open(encoding="utf-8") as f:
            raw = json.load(f)
        self._data: dict[str, dict] = raw.get("pokemon", {})
        # Build a case-insensitive lookup index
        self._name_map: dict[str, str] = {
            k.lower(): k for k in self._data
        }

    # ── Name resolution ───────────────────────────────────────────────────────
    def _resolve(self, species: str) -> Optional[str]:
        """
        Resolve a poke-env species string to the Pikalytics key.
        Handles common mismatches: 'aerodactyl' → 'Aerodactyl',
        'charizardmegay' → 'Charizard-Mega-Y', etc.
        """
        # Normalise: lowercase, collapse spaces/hyphens
        key = species.lower().strip()
        if key in self._name_map:
            return self._name_map[key]
        # Try stripping hyphens
        stripped = key.replace("-", "").replace(" ", "")
        for k_lower, k_orig in self._name_map.items():
            if k_lower.replace("-", "").replace(" ", "") == stripped:
                return k_orig
        return None

    def _entry(self, species: str) -> Optional[dict]:
        key = self._resolve(species)
        return self._data.get(key) if key else None

    # ── Public API ────────────────────────────────────────────────────────────
    def known(self, species: str) -> bool:
        """True if we have Pikalytics data for this species."""
        return self._resolve(species) is not None

    def top_moves(self, species: str, n: int = 4) -> list[tuple[str, float]]:
        """Return top-N (move_name, pct) tuples, most popular first."""
        entry = self._entry(species)
        if not entry:
            return []
        return [(m["name"], m["pct"]) for m in sorted(
            entry.get("moves", []), key=lambda x: x["pct"], reverse=True
        )[:n]]

    def top_item(self, species: str) -> Optional[str]:
        entry = self._entry(species)
        if not entry or not entry.get("items"):
            return None
        return max(entry["items"], key=lambda x: x["pct"])["name"]

    def top_ability(self, species: str) -> Optional[str]:
        entry = self._entry(species)
        if not entry or not entry.get("abilities"):
            return None
        return max(entry["abilities"], key=lambda x: x["pct"])["name"]

    def top_spread(self, species: str) -> tuple[str, list[int]]:
        """
        Return (nature, actual_evs_list) for the most common spread.
        EVs are actual values (0–252), not Pikalytics buckets.
        Falls back to Adamant + all-zeros if no data.
        """
        entry = self._entry(species)
        if not entry or not entry.get("spreads"):
            return ("Adamant", [0] * 6)
        best = max(entry["spreads"], key=lambda x: x["pct"])
        actual_evs = ev_bucket_to_evs(best["evs"])
        return (best["nature"], actual_evs)

    def expected_stats(
        self,
        species: str,
        base_stats: dict[str, int],
        level: int = 50,
    ) -> dict[str, int]:
        """
        Return the most likely in-battle stats for an opponent mon
        given its base stats and meta EV/nature data.

        base_stats: dict with keys hp/atk/def/spa/spd/spe
        """
        nature, evs = self.top_spread(species)
        return calc_full_stats(base_stats, evs, nature, level=level)

    def get_belief(self, species: str) -> dict:
        """Full belief dict for one Pokémon — useful for debugging."""
        entry = self._entry(species)
        if not entry:
            return {"error": f"No data for {species}"}
        nature, evs = self.top_spread(species)
        return {
            "species":   species,
            "usage_pct": entry.get("usage_pct"),
            "top_moves": self.top_moves(species),
            "top_item":  self.top_item(species),
            "top_ability": self.top_ability(species),
            "top_nature": nature,
            "top_evs_actual": evs,  # HP Atk Def SpA SpD Spe
            "teammates": entry.get("teammates", [])[:5],
        }

    def all_pokemon(self) -> list[str]:
        """Return all Pokémon names in the dataset."""
        return list(self._data.keys())

    def usage_ranking(self) -> list[tuple[str, float]]:
        """Return (name, usage_pct) sorted by usage descending."""
        result = []
        for name, entry in self._data.items():
            pct = entry.get("usage_pct")
            if pct is not None:
                result.append((name, pct))
        return sorted(result, key=lambda x: x[1], reverse=True)
