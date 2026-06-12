"""
belief_state.py
===============
Two responsibilities, one module:

1. ``BeliefState`` — wraps data/pikalytics_regma.json and answers "what is the
   most likely moveset / item / ability / EV spread for this species?".
   Used by server.py at startup and by the fill pipeline below.

2. ``fill_blanks()`` — takes a parsed VOD JSON (output of vod_parser) plus a
   ``VodType`` and fills in the blanks the parser could not know.  It is
   **never called automatically**: invoke it from the CLI with
   ``--fill-beliefs`` (or from a future UI button).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VOD TYPES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A — my own VODs
      our side : exact team sheet injected from an uploaded paste file
                 (e.g. teams/M-A/team1) or a known_teams.json entry
      opp side : EV/item/ability/move *distributions* from Pikalytics
      predicted_action_by_bot : stays null

B — ranked player VODs
      both sides unknown → distributions from Pikalytics, then the
      back_calculate_evs() hook narrows defensive spreads from the
      damage log  (TODO: damage-calc API — currently a stub)
      predicted_action_by_bot : stays null (behavioural-cloning data)

C — bot vs. ranked player (live)
      our side : exact;  opp side : distribution
      predicted_action_by_bot / opp_actions_predicted are written *live*
      by the poke-env hook BEFORE the turn resolves (TODO marker below);
      fill_blanks validates they are present and computes a per-turn
      ``prediction_error`` from opp_actions_predicted vs opp_actions_actual.
      chosen_action (≡ our_actions in this schema) is logged after the
      turn resolves by the same hook.

D — self-play
      both sides 100 % exact (bot controls both) → exact sheets injected
      for both players, then every mon and every turn is validated for
      completeness; missing fields are flagged (or raise with --strict).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT GETS WRITTEN INTO EACH MON DICT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
exact fill (our side A/C, both sides D):
    ev_spread        : [hp atk def spa spd spe]  bucket scale 0–32
    ev_spread_actual : [..]                      classic 0–252 scale
    iv_spread        : [..]                      defaults 31
    nature, known_item, known_ability, known_moves   (replay reveals win
                                                      conflicts; conflicts
                                                      are reported)
    exact            : {evs, evs_actual, ivs, nature, stats, source}
    stats_estimate   : {"mode": "exact", "stats": {...L50 stats...}}

distribution fill (opp side A/C, both sides B):
    belief           : {spreads:[{nature,evs,evs_actual,p}], expected_stats,
                        items:[{name,p}], abilities:[{name,p}],
                        moves_known:[..], moves_predicted:[{name,p}],
                        usage_pct, source:"pikalytics"}
                       — item/ability distributions collapse to p=1.0 when
                         the replay already revealed them
    stats_estimate   : {"mode": "distribution", "stats": {...expected...}}

EV SCALE NOTE (Pokémon Champions / Reg M-A): team sheets and Pikalytics both
use 0–32 stat points ("buckets").  Standard 0–252 pastes are auto-detected
(any value > 32) and converted.  Both representations are always emitted so
the damage-calc back-solver can use real EVs.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CLI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # dry run — reports what WOULD be filled, writes nothing
    python belief_state.py parsed.json --vod-type A --team teams/M-A/team1

    # actually fill (the manual trigger)
    python belief_state.py parsed.json --vod-type A --team teams/M-A/team1 \
        --fill-beliefs --out parsed.beliefs.json

    # type B (no team sheet needed)
    python belief_state.py parsed.json --vod-type B --fill-beliefs

    # type D self-play with both sheets, strict validation
    python belief_state.py parsed.json --vod-type D --team my.txt \
        --opp-team my.txt --fill-beliefs --strict

    # team sheet from a known_teams.json registry instead of a paste file
    python belief_state.py parsed.json --vod-type A \
        --known-teams known_teams.json --team-key "stevenhevgc" --fill-beliefs
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

# ── Bootstrap: make sibling package `vod_parser` importable when this file is
# run directly (python data/scripts/belief_state.py …).  No-op on normal import.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

# Reuse existing project logic — do not duplicate (pokedex lookups + the
# Bug-8-aware known-stats merge used by transitions.py).
from vod_parser.pokedex import get_pokedex, norm_species
from vod_parser.transitions import _inject_known_stats

# Default Pikalytics path: belief_state.py lives at data/scripts/ → data/
_DEFAULT_PIKALYTICS_PATH = Path(__file__).resolve().parents[1] / "pikalytics_regma.json"


# ── Stat formula helpers ───────────────────────────────────────────────────────
# EV buckets from Pikalytics use 32-step notation (0–32 in steps of 1,
# corresponding to 0–252 EVs in steps of 8 approx).
# Exact formula: actual_ev = ev_bucket * 8   (max 252 → bucket 31.5 → use 32)
# For damage-formula back-solving we need actual EVs, not buckets.

NATURE_BOOSTS: dict[str, tuple[str, str]] = {
    # nature: (boosted_stat, reduced_stat) — "" stat = neutral
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

# pokedex.json stores base stats under long keys
_DEX_STAT_KEYS = {
    "hp": "hp", "atk": "attack", "def": "defense",
    "spa": "special-attack", "spd": "special-defense", "spe": "speed",
}

DEFAULT_LEVEL = 50  # VGC


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
    level: int = DEFAULT_LEVEL,
) -> dict[str, int]:
    """Return the full 6-stat dict for a mon at L50 with given EVs/nature."""
    if ivs is None:
        ivs = [31] * 6
    return {
        stat: calc_stat(base_stats[stat], evs[i], ivs[i], level, nature, stat)
        for i, stat in enumerate(STAT_ORDER)
    }


def dex_base_stats(species: Optional[str]) -> Optional[dict[str, int]]:
    """Base stats for a species from data/pokedex.json, in hp/atk/…/spe keys."""
    dex = get_pokedex()
    entry = dex.entry(species) if dex else None
    if not entry:
        return None
    raw = entry.get("baseStats") or {}
    out = {k: raw.get(v) for k, v in _DEX_STAT_KEYS.items()}
    if any(v is None for v in out.values()):
        return None
    return out


def normalize_ev_map(ev_map: dict) -> tuple[list[int], list[int]]:
    """
    Take a {stat: value} EV mapping (missing stats → 0) and return
    (buckets 0–32, actual 0–252) lists in STAT_ORDER.

    Scale is auto-detected: any value > 32 means a classic 0–252 paste;
    otherwise values are Champions stat points (this format's native scale).
    """
    vals = [int(ev_map.get(s) or 0) for s in STAT_ORDER]
    if any(v > 32 for v in vals):
        actual = [min(v, 252) for v in vals]
        buckets = [round(v / 8) for v in actual]
    else:
        buckets = vals
        actual = ev_bucket_to_evs(vals)
    return buckets, actual


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

    @property
    def source_path(self) -> Path:
        return self._path

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
        level: int = DEFAULT_LEVEL,
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

    # ── Distribution API (used by fill_blanks) ────────────────────────────────
    def spread_distribution(self, species: str, top_k: int = 5) -> list[dict]:
        """
        Top-K EV spreads as a renormalised probability distribution::

            [{"nature": "Jolly", "evs": [0,32,2,0,0,32],
              "evs_actual": [0,252,16,0,0,252], "p": 0.41}, ...]

        ``evs`` are bucket scale (0–32), ``evs_actual`` classic 0–252.
        Probabilities are renormalised over the returned subset so they sum
        to 1.0 (the raw Pikalytics pcts are kept implicitly via the ratios).
        """
        entry = self._entry(species)
        if not entry or not entry.get("spreads"):
            return []
        spreads = sorted(entry["spreads"], key=lambda x: x["pct"], reverse=True)[:top_k]
        total = sum(s["pct"] for s in spreads) or 1.0
        return [
            {
                "nature": s["nature"],
                "evs": list(s["evs"]),
                "evs_actual": ev_bucket_to_evs(s["evs"]),
                "p": round(s["pct"] / total, 4),
            }
            for s in spreads
        ]

    def item_distribution(self, species: str, top_k: int = 5) -> list[dict]:
        """[{"name": item, "p": pct/100}, ...] — raw marginals, NOT renormalised."""
        entry = self._entry(species)
        if not entry:
            return []
        items = sorted(entry.get("items", []), key=lambda x: x["pct"], reverse=True)
        return [{"name": i["name"], "p": round(i["pct"] / 100.0, 4)} for i in items[:top_k]]

    def ability_distribution(self, species: str, top_k: int = 4) -> list[dict]:
        """[{"name": ability, "p": pct/100}, ...] — raw marginals."""
        entry = self._entry(species)
        if not entry:
            return []
        abs_ = sorted(entry.get("abilities", []), key=lambda x: x["pct"], reverse=True)
        return [{"name": a["name"], "p": round(a["pct"] / 100.0, 4)} for a in abs_[:top_k]]

    def move_distribution(self, species: str, top_k: int = 8) -> list[dict]:
        """[{"name": move, "p": pct/100}, ...] — usage marginals (sum can exceed 1)."""
        entry = self._entry(species)
        if not entry:
            return []
        mvs = sorted(entry.get("moves", []), key=lambda x: x["pct"], reverse=True)
        return [{"name": m["name"], "p": round(m["pct"] / 100.0, 4)} for m in mvs[:top_k]]

    def expected_stats_weighted(
        self,
        species: str,
        base_stats: dict[str, int],
        level: int = DEFAULT_LEVEL,
        top_k: int = 5,
    ) -> Optional[dict[str, float]]:
        """
        Probability-weighted in-battle stats over the top-K spreads
        (a softer estimate than expected_stats(), which uses only the #1).
        """
        spreads = self.spread_distribution(species, top_k=top_k)
        if not spreads:
            return None
        acc = {s: 0.0 for s in STAT_ORDER}
        for sp in spreads:
            stats = calc_full_stats(base_stats, sp["evs_actual"], sp["nature"], level=level)
            for s in STAT_ORDER:
                acc[s] += sp["p"] * stats[s]
        return {s: round(v, 1) for s, v in acc.items()}

    def belief_block(
        self,
        species: str,
        *,
        top_k: int = 5,
        revealed_moves: Optional[list[str]] = None,
        revealed_item: Optional[str] = None,
        revealed_ability: Optional[str] = None,
        stats_species: Optional[str] = None,
        level: int = DEFAULT_LEVEL,
    ) -> Optional[dict]:
        """
        Build the full per-mon ``belief`` dict written by fill_blanks().

        Revealed information conditions the distributions: a revealed item or
        ability collapses that distribution to p=1.0; revealed moves are listed
        as known and excluded from the predicted-move marginals.

        stats_species: forme to pull base stats from when it differs from the
        Pikalytics key (e.g. expected stats for "Charizard-Mega-Y" while the
        usage entry is "Charizard").
        """
        key = self._resolve(species)
        if key is None:
            return None
        entry = self._data[key]

        revealed_moves = revealed_moves or []
        revealed_norm = {norm_species(m) for m in revealed_moves}

        if revealed_item and revealed_item != "mega stone":
            items = [{"name": revealed_item, "p": 1.0, "revealed": True}]
        else:
            items = self.item_distribution(key, top_k=top_k)

        if revealed_ability:
            abilities = [{"name": revealed_ability, "p": 1.0, "revealed": True}]
        else:
            abilities = self.ability_distribution(key)

        free_slots = max(0, 4 - len(revealed_moves))
        moves_predicted = [
            m for m in self.move_distribution(key, top_k=top_k + 4)
            if norm_species(m["name"]) not in revealed_norm
        ][:free_slots]

        base = dex_base_stats(stats_species or species) or dex_base_stats(key)
        expected = (
            self.expected_stats_weighted(key, base, level=level, top_k=top_k)
            if base else None
        )

        return {
            "source": "pikalytics",
            "species_key": key,
            "usage_pct": entry.get("usage_pct"),
            "spreads": self.spread_distribution(key, top_k=top_k),
            "expected_stats": expected,
            "items": items,
            "abilities": abilities,
            "moves_known": list(revealed_moves),
            "moves_predicted": moves_predicted,
        }


# ══════════════════════════════════════════════════════════════════════════════
# VOD types
# ══════════════════════════════════════════════════════════════════════════════
class VodType(Enum):
    A = "A"   # my own VODs           — our side exact, opp distribution
    B = "B"   # ranked player VODs    — both sides distribution (+ back-calc)
    C = "C"   # bot vs ranked (live)  — exact/distribution + prediction error
    D = "D"   # self-play             — both sides exact, validated

    @classmethod
    def coerce(cls, value: "VodType | str") -> "VodType":
        if isinstance(value, cls):
            return value
        v = str(value).strip().upper()
        aliases = {
            "OWN_VOD": "A", "MY_VOD": "A",
            "RANKED_PLAYER_VOD": "B", "RANKED": "B",
            "LIVE_BOT_BATTLE": "C", "LIVE": "C", "BOT_VOD": "C",
            "SELF_PLAY": "D", "SELFPLAY": "D",
        }
        v = aliases.get(v, v)
        try:
            return cls(v)
        except ValueError:
            raise ValueError(f"Unknown vod_type: {value!r} (expected A|B|C|D)") from None


# source_type / stats_quality stamped onto the enriched output per type
_SOURCE_TYPE = {
    VodType.A: "own_vod",
    VodType.B: "ranked_player_vod",
    VodType.C: "live_bot_battle",
    VodType.D: "self_play",
}
_STATS_QUALITY = {
    VodType.A: {"our_side": "exact", "opp_side": "distribution"},
    VodType.B: {"our_side": "distribution", "opp_side": "distribution"},
    VodType.C: {"our_side": "exact", "opp_side": "distribution"},
    VodType.D: {"our_side": "exact", "opp_side": "exact"},
}


# ══════════════════════════════════════════════════════════════════════════════
# Team sheet loading (Showdown paste format — see teams/M-A/team1)
# ══════════════════════════════════════════════════════════════════════════════
# Self-contained on purpose: teams/team_converter.py depends on fp.helpers,
# which is not part of this repo, and it normalises away the display names
# that the parser JSON uses ("Sucker Punch", "White Herb").  We keep display
# names so injected values are directly comparable with revealed_* fields.

def parse_team_sheet(text: str) -> list[dict]:
    """
    Parse a Showdown export paste into a list of mon dicts::

        {"species": "Kingambit", "nickname": None, "item": "Chople Berry",
         "ability": "Defiant", "moves": ["Sucker Punch", ...],
         "nature": "Serious", "evs": {"hp": 31, "atk": 31, "def": 2},
         "ivs": {}, "tera_type": None, "level": 50}

    EV/IV values are kept exactly as written (scale auto-detected later by
    normalize_ev_map).  Item/ability/move/nature keep display capitalisation.
    """
    team: list[dict] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        mon: dict = {
            "species": None, "nickname": None, "item": None, "ability": None,
            "moves": [], "nature": None, "evs": {}, "ivs": {},
            "tera_type": None, "level": DEFAULT_LEVEL,
        }
        # Header: "Nickname (Species) (F) @ Item" — every part optional
        header = lines[0]
        if "@" in header:
            header, item = header.split("@", 1)
            mon["item"] = item.strip() or None
        header = re.sub(r"\((M|F)\)", "", header).strip()
        m = re.match(r"^(.*?)\s*\(([^()]+)\)\s*$", header)
        if m:
            mon["nickname"], mon["species"] = m.group(1).strip() or None, m.group(2).strip()
        else:
            mon["species"] = header.strip()

        for ln in lines[1:]:
            if ln.startswith("Ability:"):
                mon["ability"] = ln.split(":", 1)[1].strip() or None
            elif ln.startswith("Tera Type:"):
                mon["tera_type"] = ln.split(":", 1)[1].strip() or None
            elif ln.startswith("Level:"):
                try:
                    mon["level"] = int(ln.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif ln.startswith("EVs:") or ln.startswith("IVs:"):
                target = "evs" if ln.startswith("EVs:") else "ivs"
                for chunk in ln.split(":", 1)[1].split("/"):
                    bits = chunk.strip().split()
                    if len(bits) >= 2 and bits[0].lstrip("-").isdigit():
                        stat = bits[1].lower().replace("spatk", "spa").replace("spdef", "spd")
                        stat = {"hp": "hp", "atk": "atk", "def": "def",
                                "spa": "spa", "spd": "spd", "spe": "spe"}.get(stat, stat)
                        if stat in STAT_ORDER:
                            mon[target][stat] = int(bits[0])
            elif ln.endswith("Nature"):
                mon["nature"] = ln[: -len("Nature")].strip() or None
            elif ln.startswith("-"):
                mv = ln[1:].strip()
                if mv:
                    mon["moves"].append(mv)
        if mon["species"]:
            team.append(mon)
    return team


def load_team_sheet(path: str | Path) -> list[dict]:
    """Read + parse a team paste file (e.g. teams/M-A/team1)."""
    return parse_team_sheet(Path(path).read_text(encoding="utf-8"))


def load_known_team(
    known_teams_path: str | Path,
    team_key: Optional[str] = None,
) -> list[dict]:
    """
    Look a team up in known_teams.json.  Supported value shapes per key::

        { "<key>": "<raw showdown paste>" }
        { "<key>": [ {mon dict as produced by parse_team_sheet}, ... ] }

    With no team_key and exactly one entry, that entry is returned.
    """
    data = json.loads(Path(known_teams_path).read_text(encoding="utf-8"))
    if team_key is None:
        if len(data) == 1:
            team_key = next(iter(data))
        else:
            raise ValueError(
                f"known_teams.json has {len(data)} entries — pass --team-key "
                f"(available: {', '.join(sorted(data))})"
            )
    if team_key not in data:
        raise KeyError(f"Team key {team_key!r} not found in {known_teams_path}")
    entry = data[team_key]
    if isinstance(entry, str):
        return parse_team_sheet(entry)
    return entry


# ══════════════════════════════════════════════════════════════════════════════
# Damage-calc back-calculation hook (Type B) — STUB
# ══════════════════════════════════════════════════════════════════════════════
def back_calculate_evs(
    damage_events: list[dict],
    *,
    defender_species: str,
    prior_spreads: Optional[list[dict]] = None,
    level: int = DEFAULT_LEVEL,
) -> dict:
    """
    Narrow a defender's EV-spread distribution using observed damage rolls.

    TODO(damage-calc): implement via the damage calculator API.  Plan:
      1. For each damage event, resolve the attacker's stats (exact if our
         side / injected, else its own belief distribution) and the move used
         (event["source_move"]).
      2. For every candidate spread in ``prior_spreads``, compute the legal
         damage roll range (0.85–1.00 × base damage, modifiers from
         weather/screens/items in the turn snapshot) against the defender's
         resulting Def/SpD/HP stats.
      3. Zero out spreads whose roll range cannot produce the observed
         |hp_pct_delta|; renormalise the survivors' ``p``.
      4. Multiple events multiply (Bayesian update) — intersect constraints.
    Integration candidates: @smogon/calc via a tiny node sidecar, or the
    poke-env damage helpers once the live bot env is wired in.

    Currently a stub: returns the prior unchanged with ``narrowed=False``.
    """
    return {
        "defender_species": defender_species,
        "spreads": prior_spreads or [],
        "narrowed": False,
        "events_considered": len(damage_events),
        "todo": "damage-calc API not yet implemented",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Type C — prediction error
# ══════════════════════════════════════════════════════════════════════════════
def _action_pair_score(pred: Optional[dict], actual: Optional[dict]) -> float:
    """
    Similarity of one predicted action vs the action actually taken, in [0,1].
    Weights: action kind 0.4, move name / switch species 0.4 (0.6 for switch),
    move target 0.2.
    """
    if not pred or not actual:
        return 0.0
    if pred.get("action") != actual.get("action"):
        return 0.0
    score = 0.4
    if pred.get("action") == "move":
        if norm_species(pred.get("move")) == norm_species(actual.get("move")):
            score += 0.4
        if (pred.get("target_slot") or None) == (actual.get("target_slot") or None):
            score += 0.2
    else:  # switch
        if norm_species(pred.get("species")) == norm_species(actual.get("species")):
            score += 0.6
    return score


def compute_prediction_error(
    predicted: Optional[list[dict]],
    actual: Optional[list[dict]],
) -> Optional[float]:
    """
    Per-turn prediction error in [0,1]: 0 = every slot predicted perfectly,
    1 = nothing matched.  Slots are matched by their "slot" key; a slot that
    appears on only one side scores 0.  Returns None when there is no
    prediction to grade (predicted is None) — this is the neural-net
    Prediction Error target for opponent modelling.
    """
    if predicted is None:
        return None
    actual = actual or []
    pred_by_slot = {a.get("slot"): a for a in predicted}
    act_by_slot = {a.get("slot"): a for a in actual}
    slots = set(pred_by_slot) | set(act_by_slot)
    if not slots:
        return None
    total = sum(
        _action_pair_score(pred_by_slot.get(s), act_by_slot.get(s)) for s in slots
    )
    return round(1.0 - total / len(slots), 4)


def _other(pid: str) -> str:
    return "p2" if pid == "p1" else "p1"


# ══════════════════════════════════════════════════════════════════════════════
# UI auto-fill suggestions (team-builder inject panel)
# ══════════════════════════════════════════════════════════════════════════════
def ui_fill_suggestions(
    belief: BeliefState,
    vod_type: "VodType | str",
    players: dict,
    revealed_info: Optional[dict] = None,
    top_k: int = 5,
) -> dict:
    """
    Build per-species auto-fill suggestions for the team-builder UI
    ("Use Belief Integration" button → server /fill-beliefs → this).

    Which sides get suggestions depends on the VOD type:
      A own VOD        → opponent side only (our side comes from a team sheet)
      B ranked VOD     → BOTH sides
      C bot vs ranked  → opponent side only
      D self-play      → nothing (both sides are exact by definition)

    Fields the replay already revealed are NOT suggested (item/ability return
    None, revealed move slots are left empty) — the UI shows those with its
    own VOD badges and they outrank any belief.

    Returns::

        {
          "vod_type": "B",
          "filled_sides": ["p1", "p2"],
          "suggestions": {
            "p1:Kingambit": {
              "nature": "Adamant",
              "ev_spread": {"hp": 31, "atk": 31, ...},   # bucket scale 0–32
              "item": "Chople Berry" | None,
              "ability": "Defiant" | None,
              "moves": ["", "Sucker Punch", ...],  # ""-padded after revealed
              "meta": {"spread_p": .30, "item_p": .35, "ability_p": .87,
                       "moves_p": [None, .93, ...], "usage_pct": 52.0},
            }, ...
          },
          "skipped": ["p2:Meganium (no Pikalytics data)"],
          "note": str | None,
        }
    """
    vt = VodType.coerce(vod_type)
    revealed_info = revealed_info or {}
    our = (players.get("our_side") if isinstance(players, dict) else None) or "p1"
    sides = {
        VodType.A: [_other(our)],
        VodType.B: ["p1", "p2"],
        VodType.C: [_other(our)],
        VodType.D: [],
    }[vt]

    suggestions: dict[str, dict] = {}
    skipped: list[str] = []

    for pid in sides:
        roster = (players.get(pid) or {}).get("roster") or []
        for species in roster:
            key = f"{pid}:{species}"
            rev = revealed_info.get(key) or {}
            # The base ability is the only one a player chooses; a mega'd
            # mon's known_ability is the (fixed) mega ability — never suggest
            # against that context (Bug 8).
            rev_ability = rev.get("pre_mega_ability") or (
                rev.get("known_ability") if not rev.get("is_mega") else None
            )
            block = belief.belief_block(
                species,
                top_k=top_k,
                revealed_moves=rev.get("revealed_moves") or [],
                revealed_item=rev.get("known_item"),
                revealed_ability=rev_ability,
            )
            if block is None:
                skipped.append(f"{key} (no Pikalytics data)")
                continue

            top_sp = (block.get("spreads") or [None])[0]
            top_item = (block.get("items") or [None])[0]
            top_ability = (block.get("abilities") or [None])[0]
            n_revealed = len(block.get("moves_known") or [])
            predicted = block.get("moves_predicted") or []

            moves = [""] * n_revealed + [m["name"] for m in predicted]
            moves_p: list = [None] * n_revealed + [m["p"] for m in predicted]
            moves, moves_p = (moves + [""] * 4)[:4], (moves_p + [None] * 4)[:4]

            suggestions[key] = {
                "nature": top_sp["nature"] if top_sp else None,
                "ev_spread": dict(zip(STAT_ORDER, top_sp["evs"])) if top_sp else None,
                # revealed → None: the UI's VOD prefill already covers it
                "item": None if (not top_item or top_item.get("revealed")) else top_item["name"],
                "ability": None if (not top_ability or top_ability.get("revealed")) else top_ability["name"],
                "moves": moves,
                "meta": {
                    "spread_p": top_sp["p"] if top_sp else None,
                    "item_p": top_item["p"] if (top_item and not top_item.get("revealed")) else None,
                    "ability_p": top_ability["p"] if (top_ability and not top_ability.get("revealed")) else None,
                    "moves_p": moves_p,
                    "usage_pct": block.get("usage_pct"),
                },
            }

    return {
        "vod_type": vt.value,
        "filled_sides": sides,
        "suggestions": suggestions,
        "skipped": skipped,
        "note": (
            "Self-play (Type D) is fully exact — nothing to auto-fill."
            if vt is VodType.D else None
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# fill_blanks — the manual, button-triggered entry point
# ══════════════════════════════════════════════════════════════════════════════


def _sheet_index(team_sheet: Optional[list[dict]]) -> dict[str, dict]:
    """norm_species(species) → sheet mon dict."""
    if not team_sheet:
        return {}
    return {norm_species(m["species"]): m for m in team_sheet if m.get("species")}


def _iter_snapshots(turn: dict):
    """
    Yield (snapshot, perspective_pid) for every snapshot in a turn, handling
    both the raw parser shape ({"p1": snap, "p2": snap}) and the flattened
    preview/transition shape (snap directly, perspective unknown → None,
    resolved by the caller).
    """
    for key in ("state_before_actions", "state_after_actions"):
        container = turn.get(key)
        if not isinstance(container, dict):
            continue
        if "our_active" in container:          # flattened single-perspective
            yield container, None
        else:                                   # nested both-perspective
            for pid in ("p1", "p2"):
                snap = container.get(pid)
                if isinstance(snap, dict):
                    yield snap, pid


def _iter_snapshot_mons(snap: dict, perspective: str):
    """Yield (mon_dict, physical_pid) for every mon dict in one snapshot."""
    opp = _other(perspective)
    for d in (snap.get("our_active") or {}).values():
        yield d, perspective
    for d in (snap.get("opp_active") or {}).values():
        yield d, opp
    for d in (snap.get("our_bench") or []):
        yield d, perspective
    for d in (snap.get("opp_bench") or []):
        yield d, opp


def _enrich_mon_exact(
    mon: dict,
    sheet_entry: dict,
    warnings: list[str],
    level: int = DEFAULT_LEVEL,
) -> None:
    """Merge an exact team-sheet entry into one mon dict (replay reveals win)."""
    buckets, actual = normalize_ev_map(sheet_entry.get("evs") or {})
    ivs = [int(sheet_entry.get("ivs", {}).get(s, 31) or 31) for s in STAT_ORDER]
    nature = sheet_entry.get("nature") or "Serious"

    # Conflict checks: the replay log is ground truth — report, don't overwrite.
    rev_item = mon.get("known_item")
    sheet_item = sheet_entry.get("item")
    if rev_item and sheet_item and rev_item != "mega stone" \
            and norm_species(rev_item) != norm_species(sheet_item):
        warnings.append(
            f"{mon.get('species')}: replay revealed item '{rev_item}' but team "
            f"sheet says '{sheet_item}' — keeping revealed"
        )
    rev_ability = mon.get("known_ability")
    sheet_ability = sheet_entry.get("ability")
    if rev_ability and sheet_ability and not mon.get("is_mega") \
            and norm_species(rev_ability) != norm_species(sheet_ability):
        warnings.append(
            f"{mon.get('species')}: replay revealed ability '{rev_ability}' but "
            f"team sheet says '{sheet_ability}' — keeping revealed"
        )

    # Reuse the Bug-8-aware merge from transitions.py (mega ability handling).
    _inject_known_stats(mon, {
        "ev_spread": buckets,
        "nature": nature,
        "item": sheet_item,
        "moves": list(sheet_entry.get("moves") or []),
        "ability": sheet_ability,
    })
    mon["iv_spread"] = ivs
    mon["ev_spread_actual"] = actual
    if sheet_entry.get("tera_type") and not mon.get("known_tera_type"):
        mon["known_tera_type"] = sheet_entry["tera_type"]

    # In-battle stats from the current forme (mega forme base stats if mega'd).
    base = dex_base_stats(mon.get("species")) or dex_base_stats(sheet_entry["species"])
    stats = calc_full_stats(base, actual, nature, ivs, level=level) if base else None
    if stats is None:
        warnings.append(f"{mon.get('species')}: no pokedex base stats — exact stats not computed")
    mon["exact"] = {
        "evs": buckets, "evs_actual": actual, "ivs": ivs,
        "nature": nature, "stats": stats, "source": "team_sheet",
    }
    mon["stats_estimate"] = {"mode": "exact", "stats": stats}


def _enrich_mon_belief(
    mon: dict,
    belief: BeliefState,
    top_k: int,
    warnings: list[str],
    warned_species: set[str],
    level: int = DEFAULT_LEVEL,
) -> None:
    """Attach a Pikalytics belief block to one (opponent / unknown) mon dict."""
    species = mon.get("base_species") or mon.get("species")
    lookup = species if belief.known(species) else mon.get("species")
    if not lookup or not belief.known(lookup):
        if species and species not in warned_species:
            warned_species.add(species)
            warnings.append(f"{species}: no Pikalytics data — left unfilled")
        return
    block = belief.belief_block(
        lookup,
        top_k=top_k,
        revealed_moves=mon.get("revealed_moves") or [],
        revealed_item=mon.get("known_item"),
        revealed_ability=mon.get("known_ability"),
        # expected stats from the CURRENT forme (mega base stats if mega'd)
        stats_species=mon.get("species"),
        level=level,
    )
    if block is None:
        return
    mon["belief"] = block
    mon["stats_estimate"] = {"mode": "distribution", "stats": block["expected_stats"]}


# Fields a Type-D mon must have to count as "fully populated"
_TYPE_D_REQUIRED = ("ev_spread", "nature", "iv_spread")


def _validate_mon_complete(mon: dict) -> list[str]:
    """Return the list of missing fields for a supposedly-exact mon dict."""
    missing = [f for f in _TYPE_D_REQUIRED if mon.get(f) in (None, "", [])]
    if not mon.get("exact"):
        missing.append("exact")
    if not (mon.get("known_moves") or mon.get("revealed_moves")):
        missing.append("moves")
    if not (mon.get("known_ability") or mon.get("pre_mega_ability")):
        missing.append("ability")
    # known_item may legitimately be None (itemless mon) once a sheet was
    # injected — only flag it when no exact source vouched for the mon at all.
    return missing


def fill_blanks(
    parsed: dict,
    vod_type: "VodType | str",
    *,
    belief: Optional[BeliefState] = None,
    team_sheet: Optional[list[dict]] = None,
    opp_team_sheet: Optional[list[dict]] = None,
    our_side: Optional[str] = None,
    top_k: int = 5,
    strict: bool = False,
    level: int = DEFAULT_LEVEL,
) -> dict:
    """
    Fill the blanks in a parsed VOD JSON according to its VOD type and return
    an enriched **copy** (the input is never mutated).  Only invoked manually
    (CLI ``--fill-beliefs`` / future UI button) — never automatically.

    Parameters
    ----------
    parsed : dict
        Output of ShowdownReplayParser.parse() / vod_parser CLI (raw nested
        snapshots) or parse_replay_for_preview() (flattened) — both work.
    vod_type : VodType | "A"|"B"|"C"|"D"
    belief : BeliefState, optional
        Loaded Pikalytics data.  Auto-loaded from data/pikalytics_regma.json
        when omitted and the type needs distributions.
    team_sheet / opp_team_sheet : list[dict], optional
        parse_team_sheet() output for our / opponent side (opp only for D).
    our_side : "p1"|"p2", optional — overrides parsed["players"]["our_side"].
    top_k : how many EV spreads to keep per belief distribution.
    strict : Type D — raise ValueError on validation failures instead of
        only flagging them.

    Returns
    -------
    dict — deep copy of ``parsed`` with mon dicts enriched (see module
    docstring), per-turn ``prediction_error`` (Type C), and a top-level
    ``belief_fill`` metadata block.  Ready for training ingestion /
    state_encoder.encode_snapshot().
    """
    vt = VodType.coerce(vod_type)
    enriched = copy.deepcopy(parsed)
    warnings: list[str] = []
    warned_species: set[str] = set()

    our_pid = our_side or (enriched.get("players") or {}).get("our_side") or "p1"
    opp_pid = _other(our_pid)

    # ── Resolve fill mode per physical player ────────────────────────────────
    fill_modes: dict[str, str] = {
        VodType.A: {our_pid: "exact", opp_pid: "distribution"},
        VodType.B: {our_pid: "distribution", opp_pid: "distribution"},
        VodType.C: {our_pid: "exact", opp_pid: "distribution"},
        VodType.D: {our_pid: "exact", opp_pid: "exact"},
    }[vt]

    sheets: dict[str, dict[str, dict]] = {
        our_pid: _sheet_index(team_sheet),
        opp_pid: _sheet_index(opp_team_sheet if vt is VodType.D else None),
    }
    for pid, mode in fill_modes.items():
        if mode == "exact" and not sheets[pid]:
            warnings.append(
                f"{pid} marked exact but no team sheet supplied — "
                f"existing/injected values left as-is"
            )

    needs_belief = "distribution" in fill_modes.values()
    if needs_belief and belief is None:
        if _DEFAULT_PIKALYTICS_PATH.exists():
            belief = BeliefState(_DEFAULT_PIKALYTICS_PATH)
        else:
            warnings.append(
                f"No BeliefState supplied and {_DEFAULT_PIKALYTICS_PATH} missing "
                f"— distribution fill skipped"
            )

    # ── Walk every snapshot of every turn ────────────────────────────────────
    turns = enriched.get("turns") or []
    for turn in turns:
        for snap, perspective in _iter_snapshots(turn):
            persp = perspective or our_pid
            for mon, pid in _iter_snapshot_mons(snap, persp):
                mode = fill_modes.get(pid)
                if mode == "exact":
                    entry = (
                        sheets[pid].get(norm_species(mon.get("base_species")))
                        or sheets[pid].get(norm_species(mon.get("species")))
                    )
                    if entry:
                        _enrich_mon_exact(mon, entry, warnings, level=level)
                    elif sheets[pid] and mon.get("species") not in warned_species:
                        warned_species.add(mon.get("species"))
                        warnings.append(
                            f"{mon.get('species')}: not found on {pid} team sheet"
                        )
                elif mode == "distribution" and belief is not None:
                    _enrich_mon_belief(
                        mon, belief, top_k, warnings, warned_species, level=level
                    )

    # ── Type-specific passes ─────────────────────────────────────────────────
    back_calc: dict[str, dict] = {}
    prediction_errors: list[float] = []
    validation: dict = {}

    if vt is VodType.B:
        # Group damage events by (player, defender species) and run the
        # back-calc hook once per defender.  TODO(damage-calc): the stub
        # currently returns priors unchanged — see back_calculate_evs().
        by_defender: dict[tuple[str, str], list[dict]] = {}
        for turn in turns:
            for ev in turn.get("damage_events") or []:
                if ev.get("event") != "damage" or not ev.get("source_move"):
                    continue
                slot = ev.get("slot") or ""
                pid = slot[:2] if slot[:2] in ("p1", "p2") else None
                if pid and ev.get("species"):
                    by_defender.setdefault((pid, ev["species"]), []).append(ev)
        for (pid, species), events in by_defender.items():
            prior = belief.spread_distribution(species, top_k=top_k) if belief else []
            back_calc[f"{pid}:{species}"] = back_calculate_evs(
                events, defender_species=species, prior_spreads=prior, level=level
            )

    if vt is VodType.C:
        # TODO(live-poke-env): the live battle hook must write, per turn,
        #   turn["predicted_action_by_bot"]  — our own planned action(s),
        #   turn["opp_actions_predicted"]    — predicted opponent action(s),
        # BEFORE the turn resolves, and rely on the parser's our_actions /
        # opp_actions_actual as the post-resolution chosen actions.
        missing_pred = []
        for turn in turns:
            if turn.get("predicted_action_by_bot") is None:
                missing_pred.append(turn.get("turn"))
            err = compute_prediction_error(
                turn.get("opp_actions_predicted"),
                turn.get("opp_actions_actual"),
            )
            turn["prediction_error"] = err
            if err is not None:
                prediction_errors.append(err)
        if missing_pred:
            warnings.append(
                f"Type C: predicted_action_by_bot missing on turns {missing_pred} "
                f"— was the live poke-env logger running?"
            )

    if vt is VodType.D:
        missing_by_turn: dict[int, dict[str, list[str]]] = {}
        for turn in turns:
            turn_missing: dict[str, list[str]] = {}
            for snap, perspective in _iter_snapshots(turn):
                persp = perspective or our_pid
                for mon, pid in _iter_snapshot_mons(snap, persp):
                    if mon.get("seen") is False:
                        # unrevealed roster stub with no sheet entry
                        if not mon.get("exact"):
                            cur = turn_missing.setdefault(
                                f"{pid}:{mon.get('species')}", []
                            )
                            if "unrevealed_and_no_sheet" not in cur:
                                cur.append("unrevealed_and_no_sheet")
                        continue
                    missing = _validate_mon_complete(mon)
                    if missing:
                        key = f"{pid}:{mon.get('species')}"
                        cur = turn_missing.setdefault(key, [])
                        for f in missing:
                            if f not in cur:
                                cur.append(f)
            if turn.get("predicted_action_by_bot") is None:
                turn_missing.setdefault("_turn", []).append("predicted_action_by_bot")
            if turn_missing:
                missing_by_turn[turn.get("turn")] = turn_missing
        validation = {
            "complete": not missing_by_turn,
            "missing_by_turn": missing_by_turn,
        }
        if missing_by_turn:
            msg = (
                f"Type D validation: {len(missing_by_turn)}/{len(turns)} turns "
                f"have missing fields (see belief_fill.validation)"
            )
            if strict:
                raise ValueError(msg)
            warnings.append(msg)

    # ── Stamp metadata ───────────────────────────────────────────────────────
    enriched["source_type"] = _SOURCE_TYPE[vt]
    enriched["stats_quality"] = dict(_STATS_QUALITY[vt])
    enriched["belief_fill"] = {
        "vod_type": vt.value,
        "filled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "our_side": our_pid,
        "fill_modes": fill_modes,
        "pikalytics_source": str(belief.source_path) if belief else None,
        "top_k_spreads": top_k,
        "level": level,
        "warnings": warnings,
        **({"back_calc": back_calc} if back_calc else {}),
        **({"prediction_error_mean":
            round(sum(prediction_errors) / len(prediction_errors), 4)
            if prediction_errors else None} if vt is VodType.C else {}),
        **({"validation": validation} if vt is VodType.D else {}),
    }
    return enriched


# ══════════════════════════════════════════════════════════════════════════════
# CLI — fill is opt-in via --fill-beliefs; without it you get a dry run
# ══════════════════════════════════════════════════════════════════════════════
def _dry_run_report(parsed: dict, vt: VodType, args) -> None:
    players = parsed.get("players") or {}
    our = args.side or players.get("our_side") or "p1"
    turns = parsed.get("turns") or []
    print(f"[dry-run] vod_type={vt.value}  our_side={our}  turns={len(turns)}")
    for pid in ("p1", "p2"):
        info = players.get(pid) or {}
        roster = info.get("roster") or []
        print(f"[dry-run]   {pid} ({info.get('username')}): roster={len(roster)} mons")
    if vt in (VodType.A, VodType.C, VodType.D):
        print(f"[dry-run]   team sheet: {'provided' if (args.team or args.known_teams) else 'MISSING (required for exact fill)'}")
    if vt is VodType.D:
        print(f"[dry-run]   opp team sheet: {'provided' if args.opp_team else 'MISSING'}")
    pika = Path(args.pikalytics)
    print(f"[dry-run]   pikalytics: {pika} ({'found' if pika.exists() else 'NOT FOUND'})")
    print("[dry-run] nothing written — add --fill-beliefs to fill.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fill the blanks in a parsed VOD JSON (manual trigger)."
    )
    ap.add_argument("parsed_json", help="Path to vod_parser output JSON")
    ap.add_argument("--vod-type", required=True,
                    help="A=own VOD, B=ranked VOD, C=live bot battle, D=self-play")
    ap.add_argument("--fill-beliefs", action="store_true",
                    help="Actually fill (without this flag: dry-run report only)")
    ap.add_argument("--team", default=None,
                    help="Showdown paste file with OUR exact team (types A/C/D)")
    ap.add_argument("--opp-team", default=None,
                    help="Showdown paste file with the OPPONENT team (type D)")
    ap.add_argument("--known-teams", default=None,
                    help="known_teams.json registry to look our team up in")
    ap.add_argument("--team-key", default=None,
                    help="Key inside known_teams.json (default: sole entry)")
    ap.add_argument("--side", choices=["p1", "p2"], default=None,
                    help="Override which side is ours (default: from JSON)")
    ap.add_argument("--pikalytics", default=str(_DEFAULT_PIKALYTICS_PATH),
                    help="Path to pikalytics_regma.json")
    ap.add_argument("--top-k", type=int, default=5,
                    help="EV spreads kept per belief distribution (default 5)")
    ap.add_argument("--strict", action="store_true",
                    help="Type D: raise on validation failure")
    ap.add_argument("--out", default=None,
                    help="Output path (default: <input>.beliefs.json)")
    args = ap.parse_args()

    vt = VodType.coerce(args.vod_type)
    src = Path(args.parsed_json)
    if not src.exists():
        print(f"ERROR: File not found: {src}", file=sys.stderr)
        sys.exit(1)
    parsed = json.loads(src.read_text(encoding="utf-8"))

    if not args.fill_beliefs:
        _dry_run_report(parsed, vt, args)
        return

    team_sheet = None
    if args.team:
        team_sheet = load_team_sheet(args.team)
    elif args.known_teams:
        team_sheet = load_known_team(args.known_teams, args.team_key)
    opp_sheet = load_team_sheet(args.opp_team) if args.opp_team else None

    belief = None
    pika = Path(args.pikalytics)
    if pika.exists():
        belief = BeliefState(pika)

    enriched = fill_blanks(
        parsed, vt,
        belief=belief,
        team_sheet=team_sheet,
        opp_team_sheet=opp_sheet,
        our_side=args.side,
        top_k=args.top_k,
        strict=args.strict,
    )

    out_path = Path(args.out) if args.out else src.with_suffix(".beliefs.json")
    out_path.write_text(
        json.dumps(enriched, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    fills = enriched["belief_fill"]
    print(f"Wrote {out_path}  (vod_type={fills['vod_type']}, "
          f"{len(enriched.get('turns') or [])} turns)")
    for w in fills["warnings"]:
        print(f"  [warn] {w}")


if __name__ == "__main__":
    main()
