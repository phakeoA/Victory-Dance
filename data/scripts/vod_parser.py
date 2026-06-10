"""
vod_parser.py
=============
Parses a Pokémon Showdown replay .html file into the Victory-Dance training JSON schema.

Supports Type B VODs (Ranked Player replays):
  - Both sides are public information extracted from the replay log.
  - Exact EVs/IVs are unknown; damage events are recorded so a back-calculator
    can later narrow down spread distributions.
  - `predicted_action_by_bot` is always null (bot wasn't present).
  - `source_type` is always "ranked_player_vod".

Usage:
    python vod_parser.py <replay.html> [--out output.json] [--player p1|p2]

The `--player` flag marks which side is "our" side vs. opponent.
If omitted, defaults to p1.

Changes vs. v1:
  1. state_before_actions / state_after_actions — no future leakage.
  2. revealed_moves tracking on PokemonSlot.
  3. Item reveal tracking (known_item) via |-item| and |-enditem|.
  4. Terastallization tracking via |-terastallize|.
  5. Explicit is_protect flag on move events.
  6. Damage source attribution — damage events carry source_slot/source_move.
  7. Screen timer decrementing on upkeep.
  8. Action execution order (execution_index) on move events.
  9. Evolving known_team per player tracked per turn.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
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
            "known_item": self.known_item,
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


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class ShowdownReplayParser:
    """
    Walks the Showdown protocol line-by-line and emits a list of turn snapshots.

    Each snapshot captures:
      - state_before_actions: board state at the START of the turn (before any actions mutate it)
      - actions: all events that occurred during the turn
      - state_after_actions: board state at the END of the turn
    """

    def __init__(self, raw_log: str, our_player: str = "p1"):
        self.lines = [l.strip() for l in raw_log.splitlines() if l.strip()]
        self.our_player = our_player

        # ---- top-level metadata ----
        self.replay_id: Optional[str] = None
        self.format: Optional[str] = None
        self.players: dict[str, str] = {}          # "p1" -> username
        self.ratings: dict[str, int] = {}
        self.rating_deltas: dict[str, int] = {}
        self.winner: Optional[str] = None

        # ---- roster (6-mon pools revealed at teampreview) ----
        self.rosters: dict[str, list[str]] = {"p1": [], "p2": []}

        # ---- evolving known team (species revealed so far this match) ----
        self.known_team: dict[str, list[str]] = {"p1": [], "p2": []}

        # ---- active field ----
        self.field_conditions = FieldConditions()
        self.side_conditions: dict[str, SideConditions] = {
            "p1": SideConditions(),
            "p2": SideConditions(),
        }

        # slot_key -> PokemonSlot  (e.g. "p1a", "p1b", "p2a", "p2b")
        self.active_slots: dict[str, PokemonSlot] = {}

        # species -> PokemonSlot  (all mons seen, for back-reference and move history)
        self.seen_mons: dict[str, PokemonSlot] = {}

        # ---- turn tracking ----
        self.current_turn: int = 0
        self.turns: list[dict] = []
        self._current_turn_actions: list[dict] = []
        self._current_turn_damage_events: list[dict] = []

        # Snapshot captured at the START of each turn (before any mutation)
        self._state_before: Optional[dict] = None

        # Tracks the most recent move action so damage events can be attributed
        self._last_move_action: Optional[dict] = None

        # Monotone counter for action execution order within a turn
        self._execution_index: int = 0

        # ---- team sizes chosen at preview ----
        self.team_sizes: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def parse(self) -> dict:
        """Parse all lines and return the full structured output."""
        for line in self.lines:
            self._handle_line(line)

        # Flush any last in-progress turn
        if self._current_turn_actions or self._current_turn_damage_events:
            self._flush_turn()

        return self._build_output()

    # ------------------------------------------------------------------
    # Line dispatcher
    # ------------------------------------------------------------------

    def _handle_line(self, line: str) -> None:
        if not line.startswith("|"):
            return

        parts = line.split("|")
        # parts[0] is always "" because line starts with "|"
        if len(parts) < 2:
            return
        cmd = parts[1]

        # ---- metadata ----
        if cmd == "player":
            # |player|p1|steven he vgc|101|1733
            pid = parts[2]
            if len(parts) > 3 and parts[3]:
                self.players[pid] = parts[3]
            if len(parts) > 5 and parts[5]:
                try:
                    self.ratings[pid] = int(parts[5])
                except ValueError:
                    pass

        elif cmd == "tier":
            self.format = parts[2] if len(parts) > 2 else None

        elif cmd == "poke":
            # |poke|p1|Floette-Eternal, L50, F|
            pid = parts[2]
            species_info = parts[3].split(",")[0].strip() if len(parts) > 3 else "Unknown"
            self.rosters[pid].append(species_info)

        elif cmd == "teamsize":
            self.team_sizes[parts[2]] = int(parts[3])

        elif cmd == "win":
            self.winner = parts[2] if len(parts) > 2 else None

        elif cmd == "raw":
            # Capture rating deltas: "p1's rating: 1733 → 1746"
            raw_text = parts[2] if len(parts) > 2 else ""
            m = re.search(r"(\d+)\s*(?:&rarr;|→)\s*<strong>(\d+)<", raw_text)
            if m:
                for pid, uname in self.players.items():
                    if uname and uname.lower() in raw_text.lower():
                        self.rating_deltas[pid] = int(m.group(2)) - int(m.group(1))

        # ---- start of new turn ----
        elif cmd == "turn":
            if self.current_turn > 0:
                self._flush_turn()
            self.current_turn = int(parts[2])
            self._current_turn_actions = []
            self._current_turn_damage_events = []
            self._last_move_action = None
            self._execution_index = 0
            # Capture state BEFORE any actions mutate it
            self._state_before = self._snapshot_state()

        # ---- field events ----
        elif cmd == "switch" or cmd == "drag":
            self._handle_switch(parts)

        elif cmd == "detailschange":
            self._handle_mega(parts)

        elif cmd == "move":
            self._handle_move(parts)

        elif cmd == "-damage" or cmd == "-heal":
            self._handle_damage(parts, cmd)

        elif cmd == "faint":
            slot_key = self._slot_key_from_ident(parts[2])
            if slot_key in self.active_slots:
                self.active_slots[slot_key].is_fainted = True
                self.active_slots[slot_key].hp_current = 0.0
            self._current_turn_actions.append({
                "event": "faint",
                "slot": slot_key,
                "species": self._species_from_ident(parts[2]),
            })

        elif cmd == "-boost" or cmd == "-unboost":
            self._handle_boost(parts, cmd)

        elif cmd == "-status":
            slot_key = self._slot_key_from_ident(parts[2])
            status = parts[3] if len(parts) > 3 else None
            if slot_key in self.active_slots:
                self.active_slots[slot_key].status = status
            self._current_turn_actions.append({
                "event": "status",
                "slot": slot_key,
                "status": status,
            })

        elif cmd == "-curestatus":
            slot_key = self._slot_key_from_ident(parts[2])
            if slot_key in self.active_slots:
                self.active_slots[slot_key].status = None

        elif cmd == "-sidestart":
            self._handle_sidestart(parts)

        elif cmd == "-sideend":
            self._handle_sideend(parts)

        elif cmd == "-weather":
            weather_val = parts[2] if len(parts) > 2 else None
            self.field_conditions.weather = None if weather_val in (None, "none") else weather_val

        elif cmd == "-fieldstart":
            raw = parts[2] if len(parts) > 2 else ""
            if "Trick Room" in raw:
                self.field_conditions.trick_room = 5
            elif "terrain" in raw.lower():
                self.field_conditions.terrain = raw

        elif cmd == "-fieldend":
            raw = parts[2] if len(parts) > 2 else ""
            if "Trick Room" in raw:
                self.field_conditions.trick_room = 0
            elif "terrain" in raw.lower():
                self.field_conditions.terrain = None

        # ---- item reveals ----
        elif cmd == "-item":
            # |-item|p2a: Sneasler|White Herb
            self._handle_item_reveal(parts, revealed=True)

        elif cmd == "-enditem":
            # |-enditem|p2a: Sneasler|White Herb
            self._handle_item_reveal(parts, revealed=True, consumed=True)

        # ---- terastallization ----
        elif cmd == "-terastallize":
            # |-terastallize|p1a: Flutter Mane|Fairy
            self._handle_tera(parts)

        elif cmd == "-ability":
            # |-ability|p1a: Aerodactyl|Unnerve  (ability activation / reveal)
            self._handle_ability_reveal(parts)

        elif cmd == "upkeep":
            # Decrement all time-based counters
            for sc in self.side_conditions.values():
                if sc.tailwind > 0:
                    sc.tailwind -= 1
                # Decrement screen timers
                expired = [k for k, v in sc.screens.items() if v <= 1]
                for k in expired:
                    del sc.screens[k]
                for k in list(sc.screens):
                    if sc.screens[k] > 1:
                        sc.screens[k] -= 1
            if self.field_conditions.trick_room > 0:
                self.field_conditions.trick_room -= 1

    # ------------------------------------------------------------------
    # Sub-handlers
    # ------------------------------------------------------------------

    def _handle_switch(self, parts: list[str]) -> None:
        # |switch|p1b: Incineroar|Incineroar, L50, F|100/100
        ident = parts[2]
        details = parts[3]
        hp_str = parts[4] if len(parts) > 4 else "100/100"

        slot_key = self._slot_key_from_ident(ident)
        nickname = ident.split(": ", 1)[1] if ": " in ident else ident
        species = details.split(",")[0].strip()
        player = slot_key[:2]

        hp_current, hp_max = self._parse_hp(hp_str)

        # Reuse existing PokemonSlot if already seen (preserves revealed_moves, known_item, etc.)
        seen_key = f"{player}:{species}"
        if seen_key in self.seen_mons:
            mon = self.seen_mons[seen_key]
            mon.slot = slot_key[2]
            mon.hp_current = hp_current
            mon.is_fainted = False
        else:
            mon = PokemonSlot(
                species=species,
                nickname=nickname,
                player=player,
                slot=slot_key[2],
                hp_current=hp_current,
                hp_max=hp_max,
            )
            self.seen_mons[seen_key] = mon

        self.active_slots[slot_key] = mon

        # Track evolving known team
        if species not in self.known_team[player]:
            self.known_team[player].append(species)

        self._current_turn_actions.append({
            "event": "switch",
            "slot": slot_key,
            "species": species,
            "player": player,
        })

    def _handle_mega(self, parts: list[str]) -> None:
        # |detailschange|p1a: Floette|Floette-Mega, L50, F
        ident = parts[2]
        new_species = parts[3].split(",")[0].strip() if len(parts) > 3 else None
        slot_key = self._slot_key_from_ident(ident)
        if slot_key in self.active_slots and new_species:
            self.active_slots[slot_key].species = new_species
            self.active_slots[slot_key].is_mega = True
        self._current_turn_actions.append({
            "event": "mega_evolution",
            "slot": slot_key,
            "new_species": new_species,
        })

    def _handle_move(self, parts: list[str]) -> None:
        # |move|p1b: Sneasler|Fake Out|p2a: Sneasler
        user_ident = parts[2] if len(parts) > 2 else ""
        move_name = parts[3] if len(parts) > 3 else ""
        target_ident = parts[4] if len(parts) > 4 else None

        user_slot = self._slot_key_from_ident(user_ident)
        target_slot = self._slot_key_from_ident(target_ident) if target_ident else None

        # Record the move on the Pokémon's revealed_moves list
        if user_slot in self.active_slots:
            mon = self.active_slots[user_slot]
            if move_name and move_name not in mon.revealed_moves:
                mon.revealed_moves.append(move_name)

        is_protect = move_name.lower() in {
            "protect", "detect", "wide guard", "quick guard",
            "baneful bunker", "spiky shield", "silk trap", "burning bulwark",
            "max guard",
        }

        action = {
            "event": "move",
            "execution_index": self._execution_index,
            "user_slot": user_slot,
            "user_species": self._species_from_ident(user_ident),
            "move": move_name,
            "target_slot": target_slot,
            "target_species": self._species_from_ident(target_ident) if target_ident else None,
            "is_protect": is_protect,
        }
        self._execution_index += 1
        self._last_move_action = action
        self._current_turn_actions.append(action)

    def _handle_damage(self, parts: list[str], cmd: str) -> None:
        # |-damage|p2a: Sneasler|74/100
        ident = parts[2]
        hp_str = parts[3] if len(parts) > 3 else "0/100"
        slot_key = self._slot_key_from_ident(ident)

        hp_current, hp_max = self._parse_hp(hp_str)
        prev_hp = None
        if slot_key in self.active_slots:
            prev_hp = self.active_slots[slot_key].hp_current
            self.active_slots[slot_key].hp_current = hp_current

        delta_pct = None
        if prev_hp is not None and hp_current is not None:
            delta_pct = round(hp_current - prev_hp, 2)

        # Attribute source from the most recent move action
        source_slot = None
        source_species = None
        source_move = None
        if self._last_move_action and cmd == "-damage":
            source_slot = self._last_move_action.get("user_slot")
            source_species = self._last_move_action.get("user_species")
            source_move = self._last_move_action.get("move")

        event = {
            "event": cmd.lstrip("-"),   # "damage" or "heal"
            "slot": slot_key,
            "species": self._species_from_ident(ident),
            "hp_pct_after": hp_current,
            "hp_pct_delta": delta_pct,
            "source_slot": source_slot,
            "source_species": source_species,
            "source_move": source_move,
        }
        self._current_turn_damage_events.append(event)
        self._current_turn_actions.append(event)

    def _handle_boost(self, parts: list[str], cmd: str) -> None:
        # |-boost|p1a: Floette|spa|1
        ident = parts[2]
        stat = parts[3] if len(parts) > 3 else ""
        amount = int(parts[4]) if len(parts) > 4 else 1
        if cmd == "-unboost":
            amount = -amount

        slot_key = self._slot_key_from_ident(ident)
        if slot_key in self.active_slots:
            current = self.active_slots[slot_key].boosts.get(stat, 0)
            self.active_slots[slot_key].boosts[stat] = current + amount

        self._current_turn_actions.append({
            "event": "stat_change",
            "slot": slot_key,
            "stat": stat,
            "stages": amount,
        })

    def _handle_sidestart(self, parts: list[str]) -> None:
        # |-sidestart|p2: speedyturtle87|move: Tailwind
        raw_side = parts[2]
        effect = parts[3] if len(parts) > 3 else ""
        pid = raw_side.split(":")[0].strip()

        if "Tailwind" in effect:
            self.side_conditions[pid].tailwind = 4
        elif "Reflect" in effect:
            self.side_conditions[pid].screens["reflect"] = 5
        elif "Light Screen" in effect:
            self.side_conditions[pid].screens["light_screen"] = 5
        elif "Aurora Veil" in effect:
            self.side_conditions[pid].screens["aurora_veil"] = 5

    def _handle_sideend(self, parts: list[str]) -> None:
        raw_side = parts[2]
        effect = parts[3] if len(parts) > 3 else ""
        pid = raw_side.split(":")[0].strip()

        if "Tailwind" in effect:
            self.side_conditions[pid].tailwind = 0
        elif "Reflect" in effect:
            self.side_conditions[pid].screens.pop("reflect", None)
        elif "Light Screen" in effect:
            self.side_conditions[pid].screens.pop("light_screen", None)
        elif "Aurora Veil" in effect:
            self.side_conditions[pid].screens.pop("aurora_veil", None)

    def _handle_item_reveal(self, parts: list[str], revealed: bool, consumed: bool = False) -> None:
        # |-item|p2a: Sneasler|White Herb
        # |-enditem|p2a: Sneasler|White Herb
        ident = parts[2] if len(parts) > 2 else ""
        item = parts[3] if len(parts) > 3 else None
        slot_key = self._slot_key_from_ident(ident)

        if slot_key in self.active_slots and item:
            self.active_slots[slot_key].known_item = item

        self._current_turn_actions.append({
            "event": "item_consumed" if consumed else "item_revealed",
            "slot": slot_key,
            "species": self._species_from_ident(ident),
            "item": item,
        })

    def _handle_tera(self, parts: list[str]) -> None:
        # |-terastallize|p1a: Flutter Mane|Fairy
        ident = parts[2] if len(parts) > 2 else ""
        tera_type = parts[3] if len(parts) > 3 else None
        slot_key = self._slot_key_from_ident(ident)

        if slot_key in self.active_slots:
            self.active_slots[slot_key].known_tera_type = tera_type
            self.active_slots[slot_key].is_terastallized = True

        self._current_turn_actions.append({
            "event": "terastallize",
            "slot": slot_key,
            "species": self._species_from_ident(ident),
            "tera_type": tera_type,
        })

    def _handle_ability_reveal(self, parts: list[str]) -> None:
        # |-ability|p1a: Aerodactyl|Unnerve
        # |-ability|p2b: Incineroar|Intimidate|[from] ability: Trace
        ident = parts[2] if len(parts) > 2 else ""
        ability = parts[3] if len(parts) > 3 else None
        slot_key = self._slot_key_from_ident(ident)

        if slot_key in self.active_slots and ability:
            self.active_slots[slot_key].known_ability = ability
            # Propagate to seen_mons so it survives switches
            player = slot_key[:2]
            species = self.active_slots[slot_key].species
            seen_key = f"{player}:{species}"
            if seen_key in self.seen_mons:
                self.seen_mons[seen_key].known_ability = ability

        self._current_turn_actions.append({
            "event": "ability_revealed",
            "slot": slot_key,
            "species": self._species_from_ident(ident),
            "ability": ability,
        })

    # ------------------------------------------------------------------
    # State snapshot (deep copy so mutations don't bleed through)
    # ------------------------------------------------------------------

    def _snapshot_state(self) -> dict:
        return {
            "field": copy.deepcopy(self.field_conditions.to_dict()),
            "side_conditions": {
                pid: copy.deepcopy(sc.to_dict())
                for pid, sc in self.side_conditions.items()
            },
            "active_pokemon": {
                slot: copy.deepcopy(mon.to_dict())
                for slot, mon in self.active_slots.items()
                if not mon.is_fainted
            },
            "known_team": {
                pid: list(team) for pid, team in self.known_team.items()
            },
        }

    # ------------------------------------------------------------------
    # Turn flush
    # ------------------------------------------------------------------

    def _flush_turn(self) -> None:
        """Snapshot the board state before + after and record all actions."""
        turn_snapshot = {
            "turn": self.current_turn,
            # State captured at the START of the turn (no future leakage)
            "state_before_actions": self._state_before or self._snapshot_state(),
            "actions": list(self._current_turn_actions),
            "damage_events": list(self._current_turn_damage_events),
            # State captured AFTER all actions have resolved
            "state_after_actions": self._snapshot_state(),
            # ---- Type B: prediction fields always null ----
            "predicted_action_by_bot": None,
        }
        self.turns.append(turn_snapshot)

    # ------------------------------------------------------------------
    # Output assembly
    # ------------------------------------------------------------------

    def _build_output(self) -> dict:
        # Aggregate everything the parser learned across the whole match into a
        # flat dict keyed by "pid:species".  This is what the inject panel uses
        # to pre-populate revealed moves, items, tera type, and ability.
        revealed_info: dict[str, dict] = {}
        for seen_key, mon in self.seen_mons.items():
            # seen_key is already "pid:species"
            revealed_info[seen_key] = {
                "revealed_moves": list(mon.revealed_moves),
                "known_item": mon.known_item,
                "known_tera_type": mon.known_tera_type,
                "is_terastallized": mon.is_terastallized,
                "known_ability": mon.known_ability,
            }

        return {
            "source_type": "ranked_player_vod",
            "replay_id": None,      # caller can inject from filename
            "format": self.format,
            "players": {
                "our_side": self.our_player,
                "p1": {
                    "username": self.players.get("p1"),
                    "rating_before": self.ratings.get("p1"),
                    "rating_delta": self.rating_deltas.get("p1"),
                    "roster": self.rosters["p1"],
                    "team_size_chosen": self.team_sizes.get("p1"),
                },
                "p2": {
                    "username": self.players.get("p2"),
                    "rating_before": self.ratings.get("p2"),
                    "rating_delta": self.rating_deltas.get("p2"),
                    "roster": self.rosters["p2"],
                    "team_size_chosen": self.team_sizes.get("p2"),
                },
            },
            "winner": self.winner,
            "stats_quality": {
                "our_side": "distribution",
                "opp_side": "distribution",
            },
            # Everything the parser learned from the replay log — revealed moves,
            # consumed/revealed items, tera type.  Used by the inject panel to
            # pre-populate fields and show "confirmed" badges.
            "revealed_info": revealed_info,
            "turns": self.turns,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _slot_key_from_ident(ident: Optional[str]) -> str:
        if not ident:
            return ""
        m = re.match(r"(p[12][ab])", ident)
        return m.group(1) if m else ident.split(":")[0].strip()

    @staticmethod
    def _species_from_ident(ident: Optional[str]) -> Optional[str]:
        if not ident:
            return None
        if ": " in ident:
            return ident.split(": ", 1)[1].strip()
        return None

    @staticmethod
    def _parse_hp(hp_str: str) -> tuple[Optional[float], Optional[float]]:
        hp_str = hp_str.split(" ")[0]  # strip " fnt"
        if "/" in hp_str:
            parts = hp_str.split("/")
            try:
                return float(parts[0]), float(parts[1])
            except ValueError:
                pass
        try:
            return float(hp_str), 100.0
        except ValueError:
            return None, None


# ---------------------------------------------------------------------------
# HTML extraction
# ---------------------------------------------------------------------------

def extract_log_from_html(html: str) -> str:
    m = re.search(
        r'<script[^>]+class="battle-log-data"[^>]*>(.*?)</script>',
        html,
        re.DOTALL,
    )
    if not m:
        raise ValueError(
            "Could not find battle-log-data script block in HTML. "
            "Is this a valid Showdown replay file?"
        )
    raw = m.group(1)
    raw = raw.replace(r"\/", "/")
    return raw


def extract_replay_id_from_html(html: str) -> Optional[str]:
    m = re.search(r'name="replayid"\s+value="([^"]+)"', html)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Server API helpers
# ---------------------------------------------------------------------------

def parse_replay_for_preview(
    html_content: str,
    belief=None,
    known_entry: Optional[dict] = None,
) -> dict:
    """
    Parse a Showdown replay HTML string and return the structured battle dict
    that the team-builder UI expects.

    Parameters
    ----------
    html_content : str
        Raw contents of the .html replay file.
    belief : BeliefState, optional
        Pikalytics belief state (unused for now; accepted so the signature
        stays stable when belief inference is wired in).
    known_entry : dict, optional
        A partial known_teams entry the user has already filled in.  Shape::

            {
              "p1": { "Kingambit": { "nature": "Adamant", ... } },
              "p2": { ... },
              "_meta": { "yourSide": "p1", "winner": "p1", ... }
            }

        When supplied, the ``_meta`` fields are merged into the preview's
        top-level ``players`` dict and the per-species dicts are surfaced as
        ``known_team_overrides`` so the UI can pre-populate injection cards.

    Returns
    -------
    dict
        Full battle object matching the shape produced by the JS
        ``parseShowdownLog()`` function.
    """
    log = extract_log_from_html(html_content)
    replay_id = extract_replay_id_from_html(html_content)

    # Determine our_player from known_entry if present
    our_player = "p1"
    if known_entry:
        meta = known_entry.get("_meta", {})
        our_player = meta.get("yourSide", "p1") or "p1"

    parser = ShowdownReplayParser(log, our_player=our_player)
    result = parser.parse()
    result["replay_id"] = replay_id
    result["known_team_overrides"] = {}

    # Merge known_entry into the result
    if known_entry:
        meta = known_entry.get("_meta", {})

        if meta.get("winner"):
            result["winner"] = meta["winner"]

        # Surface per-species data as known_team_overrides
        for pid in ("p1", "p2"):
            for species, data in (known_entry.get(pid) or {}).items():
                key = f"{pid}:{species}"
                if any(v is not None for v in data.values()):
                    result["known_team_overrides"][key] = data

    return result


def replay_to_transitions(
    replay_path,
    belief=None,
    encoder=None,
    players: Optional[list] = None,
    known_teams: Optional[dict] = None,
) -> list:
    """
    Parse a replay and convert every turn into a JSONL-ready transition dict.

    Each transition records the state visible to one player at decision time,
    the actions taken that turn, and the resulting state.  Fields that require
    a trained encoder (numeric feature vectors) are left as null for now.

    Parameters
    ----------
    replay_path : path-like
        Path to the .html replay file.
    belief : BeliefState, optional
        Reserved for Pikalytics-based hidden-info inference.
    encoder : StateEncoder, optional
        Reserved for converting state dicts to float tensors.
    players : list of str, optional
        Which player perspectives to emit (e.g. ["p1"]).
        Defaults to ["p1", "p2"].
    known_teams : dict, optional
        Map of battle_id -> known_teams_entry.  Used to inject exact stats
        when available.

    Returns
    -------
    list of dict
        One dict per (turn, player) combination, ready for json.dumps.
    """
    html_content = Path(replay_path).read_text(encoding="utf-8")
    replay_id = extract_replay_id_from_html(html_content) or Path(replay_path).stem

    # Look up any injected team data for this battle
    known_entry: dict = {}
    if known_teams:
        known_entry = known_teams.get(replay_id, {})

    our_player = known_entry.get("_meta", {}).get("yourSide", "p1") or "p1"

    log = extract_log_from_html(html_content)
    parser = ShowdownReplayParser(log, our_player=our_player)
    battle = parser.parse()
    battle["replay_id"] = replay_id

    if players is None:
        players = ["p1", "p2"]

    transitions = []

    for turn in battle["turns"]:
        for perspective in players:
            # Inject known stats into active_pokemon for our side
            after = copy.deepcopy(turn["state_after_actions"])
            for slot_key, mon_dict in after.get("active_pokemon", {}).items():
                pid = slot_key[:2]
                species = mon_dict.get("species", "")
                inj = (known_entry.get(pid) or {}).get(species, {})
                if inj:
                    mon_dict["ev_spread"] = inj.get("ev_spread")
                    mon_dict["nature"] = inj.get("nature")
                    mon_dict["known_item"] = mon_dict.get("known_item") or inj.get("item")
                    if inj.get("moves"):
                        mon_dict["known_moves"] = inj["moves"]

            t = {
                "source_type": battle.get("source_type", "ranked_player_vod"),
                "replay_id": replay_id,
                "format": battle.get("format"),
                "perspective": perspective,
                "turn": turn["turn"],
                "winner": battle.get("winner"),
                "state_before_actions": turn["state_before_actions"],
                "state_after_actions": after,
                "actions": turn["actions"],
                "damage_events": turn["damage_events"],
                # Placeholders for future encoder output
                "state_vector": None,
                "action_mask": None,
                "predicted_action_by_bot": turn.get("predicted_action_by_bot"),
                "players": {
                    "our_side": our_player,
                    "p1": battle["players"]["p1"],
                    "p2": battle["players"]["p2"],
                },
            }
            transitions.append(t)

    return transitions


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Parse a Showdown replay HTML into Victory-Dance JSON.")
    ap.add_argument("replay_html", help="Path to the .html replay file")
    ap.add_argument("--out", default=None, help="Output JSON path (default: <replay>.json)")
    ap.add_argument(
        "--player",
        choices=["p1", "p2"],
        default="p1",
        help="Which side is 'our' side (default: p1)",
    )
    args = ap.parse_args()

    src = Path(args.replay_html)
    if not src.exists():
        print(f"ERROR: File not found: {src}", file=sys.stderr)
        sys.exit(1)

    html = src.read_text(encoding="utf-8")

    log = extract_log_from_html(html)
    replay_id = extract_replay_id_from_html(html)

    parser = ShowdownReplayParser(log, our_player=args.player)
    result = parser.parse()
    result["replay_id"] = replay_id
    result["source_file"] = src.name

    out_path = Path(args.out) if args.out else src.with_suffix(".json")
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(result['turns'])} turns → {out_path}")


if __name__ == "__main__":
    main()
