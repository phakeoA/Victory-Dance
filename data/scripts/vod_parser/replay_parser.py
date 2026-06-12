"""
replay_parser.py
================
Parses a Pokémon Showdown replay log line-by-line and emits structured turn
snapshots.  Depends only on battle_models; no circular imports.

Public surface
--------------
    ShowdownReplayParser   — main parser class
    extract_log_from_html  — pull the battle-log-data block out of an .html file
    extract_replay_id_from_html — pull the replay id out of an .html file
"""

from __future__ import annotations

import copy
import re
from typing import Optional

from vod_parser.battle_models import (
    FieldConditions,
    PokemonSlot,
    SideConditions,
)
from vod_parser.pokedex import get_pokedex, is_mega_species_name


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

        # Bug 9: many abilities never get a standalone |-ability| line — the
        # reveal rides as a "[from] ability:" tag on some other line, e.g.
        #   |-weather|RainDance|[from] ability: Drizzle|[of] p2a: Pelipper
        # Scan every line for that pattern.  |-ability| is excluded: it has
        # its own handler, and a [from] tag there (e.g. Trace) names a
        # DIFFERENT ability than the one being activated.
        if cmd != "-ability":
            self._learn_ability_from_tags(parts)

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
            self._state_before = {
                "p1": self._snapshot_state("p1"),
                "p2": self._snapshot_state("p2"),
            }

        # ---- field events ----
        elif cmd == "switch" or cmd == "drag":
            self._handle_switch(parts)

        elif cmd == "detailschange":
            self._handle_mega(parts)

        elif cmd == "-mega":
            # |-mega|p1a: Floette|Floette|Floettite  — explicit stone reveal
            self._handle_mega_stone(parts)

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
            cured = parts[3] if len(parts) > 3 else None
            if slot_key in self.active_slots:
                self.active_slots[slot_key].status = None
            # Bug 3 fix: record the event so status changes between
            # state_before and state_after are explained in the action log.
            self._current_turn_actions.append({
                "event": "curestatus",
                "slot": slot_key,
                "species": self._species_from_ident(parts[2]),
                "status": cured,
            })

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
                # Bug 4 fix: store a normalised token ("electric", "grassy",
                # "misty", "psychic"), not the raw effect string.
                self.field_conditions.terrain = self._normalize_terrain(raw)

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
        mon = self.seen_mons.get(seen_key)
        if mon is None:
            # Bug 8 continuity: a mega'd mon switching back in shows its MEGA
            # forme on the |switch| line ("Floette-Mega"), but seen_mons is
            # keyed by base species.  Match on current (mutated) species so
            # we don't fork a duplicate mon and lose its ability state.
            mon = next(
                (m for k, m in self.seen_mons.items()
                 if k.startswith(player + ":") and m.species == species),
                None,
            )
        if mon is not None:
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
                base_species=species,   # Bug 7: frozen pre-mega/forme name
            )
            self.seen_mons[seen_key] = mon

        # Bug 1 fix: stat boosts do not persist through a switch.  Clear the
        # outgoing mon's boosts before it leaves the field so they don't
        # silently accumulate across the whole match (and so they're already
        # clean if it later switches back in).
        if slot_key in self.active_slots and self.active_slots[slot_key] is not mon:
            self.active_slots[slot_key].boosts = {}

        self.active_slots[slot_key] = mon
        # Incoming mon always starts with neutral boosts (covers |drag| too,
        # and guards against any stale state on the reused PokemonSlot).
        mon.boosts = {}
        # A switch-in starts a fresh stay on the field — a Choice lock from a
        # previous stint no longer applies, so the working move set restarts.
        mon.stint_moves = []

        # Track evolving known team (by BASE species — a mega'd mon switching
        # back in must not appear as a second team member)
        known_species = mon.base_species or species
        if known_species not in self.known_team[player]:
            self.known_team[player].append(known_species)

        self._current_turn_actions.append({
            "event": "switch",
            "slot": slot_key,
            "species": species,
            "player": player,
        })

    def _handle_mega(self, parts: list[str]) -> None:
        # |detailschange|p1a: Floette|Floette-Mega, L50, F
        # Mutates species + sets is_mega.  The |-mega| line that fires immediately
        # after carries the explicit stone name and calls _handle_mega_stone().
        # The suffix check is only a fallback for replays that lack |-mega|.
        #
        # Bug 8 (mega ability split): |detailschange| also fires for NON-mega
        # forme changes (Palafin → Palafin-Hero, Terapagos, etc.), so we must
        # not blindly set is_mega.  For genuine megas, the new forme has
        # exactly ONE possible ability, fully determined by the pokedex — so
        # the moment we see the mega we:
        #   1. demote any previously revealed ability to pre_mega_ability
        #      (it can no longer be the active ability), and
        #   2. set known_ability/mega_ability from the pokedex (or None if
        #      the dex doesn't cover this species — a later |-ability| line
        #      will fill it in).
        ident = parts[2]
        new_species = parts[3].split(",")[0].strip() if len(parts) > 3 else None
        slot_key = self._slot_key_from_ident(ident)

        dex = get_pokedex()
        is_mega = bool(new_species) and (
            dex.is_mega_forme(new_species) if dex else is_mega_species_name(new_species)
        )

        if slot_key in self.active_slots and new_species:
            mon = self.active_slots[slot_key]
            mon.species = new_species
            if is_mega:
                mon.is_mega = True
                self._apply_mega_ability(mon, new_species)
            # Non-mega forme change: species updated, ability state untouched.

        if is_mega:
            self._current_turn_actions.append({
                "event": "mega_evolution",
                "slot": slot_key,
                "new_species": new_species,
                "mega_stone": self.active_slots[slot_key].known_item if slot_key in self.active_slots else None,
                "pre_mega_ability": self.active_slots[slot_key].pre_mega_ability if slot_key in self.active_slots else None,
                "mega_ability": self.active_slots[slot_key].mega_ability if slot_key in self.active_slots else None,
            })
        else:
            self._current_turn_actions.append({
                "event": "forme_change",
                "slot": slot_key,
                "new_species": new_species,
            })

    def _apply_mega_ability(self, mon: PokemonSlot, mega_species: str) -> None:
        """Swap a mon's ability state over to its (single, fixed) mega ability.

        Idempotent — safe to call from both |detailschange| and |-mega|.
        """
        # Demote whatever was known before the mega: it is now definitively
        # NOT the active ability.  Never overwrite an already-recorded
        # pre-mega ability (e.g. on a second, redundant call).
        if mon.pre_mega_ability is None and mon.known_ability and mon.known_ability != mon.mega_ability:
            mon.pre_mega_ability = mon.known_ability

        dex = get_pokedex()
        mega_ability = dex.mega_ability_for(mega_species) if dex else None
        if mega_ability:
            mon.mega_ability = mega_ability
            mon.known_ability = mega_ability
        elif mon.mega_ability:
            # Already learned via |-ability| — keep it current.
            mon.known_ability = mon.mega_ability
        else:
            # Dex can't resolve it and nothing revealed yet: the pre-mega
            # ability is stale, so the current ability is simply unknown.
            mon.known_ability = None

    def _handle_mega_stone(self, parts: list[str]) -> None:
        # |-mega|p1a: Floette|Floette|Floettite
        # parts: ['', '-mega', 'p1a: Floette', 'Floette', 'Floettite']
        ident    = parts[2] if len(parts) > 2 else ""
        stone    = parts[4] if len(parts) > 4 else None
        slot_key = self._slot_key_from_ident(ident)

        if stone and slot_key in self.active_slots:
            mon = self.active_slots[slot_key]
            mega_species  = mon.species   # already the Mega form
            pre_mega      = parts[3] if len(parts) > 3 else mega_species
            self._set_known_item(slot_key, mega_species, pre_mega, stone)
            # Bug 8 safety net: if |detailschange| was missing/odd, make sure
            # the mega flag and ability swap still happened (idempotent).
            if not mon.is_mega:
                mon.is_mega = True
                self._apply_mega_ability(mon, mega_species)
            # Patch the mega_stone field on the most recent mega_evolution action
            for action in reversed(self._current_turn_actions):
                if action.get("event") == "mega_evolution" and action.get("slot") == slot_key:
                    action["mega_stone"] = stone
                    break

    def _set_known_item(
        self,
        slot_key: str,
        mega_species: str,
        pre_mega_species: str,
        stone: str,
    ) -> None:
        """Write the mega stone into the active slot and both seen_mons keys."""
        player = slot_key[:2]
        self.active_slots[slot_key].known_item = stone
        for sk in (f"{player}:{mega_species}", f"{player}:{pre_mega_species}"):
            if sk in self.seen_mons:
                self.seen_mons[sk].known_item = stone

    def _handle_move(self, parts: list[str]) -> None:
        # |move|p1b: Sneasler|Fake Out|p2a: Sneasler
        user_ident = parts[2] if len(parts) > 2 else ""
        move_name = parts[3] if len(parts) > 3 else ""
        target_ident = parts[4] if len(parts) > 4 else None

        user_slot = self._slot_key_from_ident(user_ident)
        target_slot = self._slot_key_from_ident(target_ident) if target_ident else None
        # Bug 5 fix: self-targeting / no-target moves must always serialise as
        # null, never as "" — the encoder treats them as the same situation.
        if target_slot == "":
            target_slot = None

        # Record the move on the Pokémon's revealed_moves list
        if user_slot in self.active_slots:
            mon = self.active_slots[user_slot]
            if move_name and move_name not in mon.revealed_moves:
                mon.revealed_moves.append(move_name)
            # Choice-item constraint (see PokemonSlot.can_have_choice_item):
            # a [from]-tagged move was CALLED by another effect (Sleep Talk,
            # Dancer, Instruct, locked-move continuations) — the player never
            # selected it, so it proves nothing about a Choice lock.  Struggle
            # is excluded too: a choice-locked mon Struggles once its locked
            # move runs out of PP.
            called = any(p.startswith("[from]") for p in parts[4:] if p)
            if move_name and not called and move_name.lower() != "struggle":
                if move_name not in mon.stint_moves:
                    mon.stint_moves.append(move_name)
                if len(mon.stint_moves) >= 2:
                    mon.can_have_choice_item = False

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

        # Attribute source from the most recent move action.
        # Bug 2 fix: indirect damage (burn, poison, Rocky Helmet, recoil,
        # Leech Seed, weather, ...) carries a [from] tag on the protocol line.
        # Those must NOT be attributed to whatever move happened to fire last.
        has_from_tag = any("[from]" in (p or "") for p in parts[3:])
        source_slot = None
        source_species = None
        source_move = None
        if self._last_move_action and cmd == "-damage" and not has_from_tag:
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

        if slot_key in self.active_slots:
            mon = self.active_slots[slot_key]
            if item:
                mon.known_item = item
            # The item just changed hands or was consumed/revealed (Trick,
            # Knock Off, berry, Frisk, …) — moves used from here on prove
            # nothing about the item the mon BROUGHT, so the choice-constraint
            # stint restarts.  Conservative: never creates a false positive.
            mon.stint_moves = []

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
        self._record_ability(
            self._slot_key_from_ident(ident),
            ability,
            self._species_from_ident(ident),
        )

    def _learn_ability_from_tags(self, parts: list[str]) -> None:
        """Learn an ability revealed via a [from] ability: tag (Bug 9).

        Weather/terrain setters and many passive effects announce the
        ability inline on another protocol line instead of |-ability|::

            |-weather|RainDance|[from] ability: Drizzle|[of] p2a: Pelipper
            |-damage|p1a: X|90/100|[from] ability: Rough Skin|[of] p2b: Y
            |-heal|p2a: X|100/100|[from] ability: Water Absorb

        The ability holder is the [of] mon when present, otherwise the
        subject of the line.  (Showdown points [of] at a different mon for
        a couple of rare item-theft abilities — Pickpocket/Magician — which
        we accept as a known limitation.)
        """
        ability = None
        of_ident = None
        for p in parts[2:]:
            if p.startswith("[from] ability:"):
                ability = p.split(":", 1)[1].strip()
            elif p.startswith("[of]"):
                of_ident = p[len("[of]"):].strip()
        if not ability:
            return
        holder = of_ident or (parts[2] if len(parts) > 2 else "")
        slot_key = self._slot_key_from_ident(holder)
        if not re.fullmatch(r"p[12][ab]", slot_key):
            return
        self._record_ability(slot_key, ability, self._species_from_ident(holder))

    def _record_ability(
        self,
        slot_key: str,
        ability: Optional[str],
        species: Optional[str],
    ) -> None:
        """Write a revealed ability into the holder's slot + seen_mons state."""
        if slot_key in self.active_slots and ability:
            mon = self.active_slots[slot_key]
            # Bug 8: route the reveal into the correct ability context.
            # A mega'd mon's ability line IS its mega ability; a base-forme
            # mon's line is its (chosen) base ability.  Either way the
            # currently-active ability is updated.
            if mon.is_mega:
                mon.mega_ability = ability
            else:
                mon.pre_mega_ability = ability
            mon.known_ability = ability
            # active_slots and seen_mons share the same PokemonSlot object
            # (see _handle_switch), so no extra propagation is needed — but
            # keep an explicit write keyed by BASE species (Bug 7: `species`
            # mutates on mega, seen_mons keys don't) for refactor safety.
            player = slot_key[:2]
            seen_key = f"{player}:{mon.base_species or mon.species}"
            if seen_key in self.seen_mons:
                self.seen_mons[seen_key].known_ability = ability

        self._current_turn_actions.append({
            "event": "ability_revealed",
            "slot": slot_key,
            "species": species,
            "ability": ability,
            "is_mega_ability": (
                self.active_slots[slot_key].is_mega
                if slot_key in self.active_slots else False
            ),
        })

    # ------------------------------------------------------------------
    # State snapshot (deep copy so mutations don't bleed through)
    # ------------------------------------------------------------------

    def _snapshot_state(self, perspective: Optional[str] = None) -> dict:
        our = perspective or self.our_player
        opp = "p2" if our == "p1" else "p1"

        # ── Active slots split by side ────────────────────────────────────
        our_active: dict[str, dict] = {}
        opp_active: dict[str, dict] = {}
        for slot_key, mon in self.active_slots.items():
            if mon.is_fainted:
                continue
            d = copy.deepcopy(mon.to_dict())
            if slot_key.startswith(our):
                # Store under perspective-relative key: p1a -> our_a, p1b -> our_b
                rel_key = "our_" + slot_key[2:]
                our_active[rel_key] = d
            else:
                rel_key = "opp_" + slot_key[2:]
                opp_active[rel_key] = d

        # ── Bench policy (Bug 6) ──────────────────────────────────────────
        # DESIGN DECISION: fainted mons are KEPT in the bench, on both sides,
        # with is_fainted=True / hp_pct=0.  The model needs them to know which
        # switch-ins are legal and how many team members remain.  The rule is
        # applied identically to our_bench and opp_bench:
        #   bench = every revealed mon that is not currently active-and-alive.
        # (A mon that fainted on the field appears in the bench until/after
        # its replacement switches in — it is filtered out of *_active either
        # way, so it is never represented twice.)

        # Bug 7: |detailschange| mutates `species` (Meganium -> Meganium-Mega)
        # but rosters and seen_mons keys use the teampreview name.  All bench
        # vs. active reconciliation below therefore compares BASE species.
        def _base(mon: PokemonSlot) -> str:
            return mon.base_species or mon.species

        # ── Our bench ─────────────────────────────────────────────────────
        our_active_species = {_base(m) for m in self.active_slots.values()
                              if m.player == our and not m.is_fainted}
        our_bench: list[dict] = []
        for seen_key, mon in self.seen_mons.items():
            if not seen_key.startswith(our + ":"):
                continue
            if _base(mon) in our_active_species:
                continue
            d = copy.deepcopy(mon.to_dict())
            d["seen"] = True   # own bench is always fully known
            our_bench.append(d)

        # ── Opponent bench ────────────────────────────────────────────────
        # Same rule as our_bench (see Bug 6 note above): revealed mons —
        # fainted ones included — plus unseen roster slots as stubs.
        opp_active_species = {_base(m) for m in self.active_slots.values()
                              if m.player == opp and not m.is_fainted}
        opp_seen: dict[str, dict] = {}
        for seen_key, mon in self.seen_mons.items():
            if not seen_key.startswith(opp + ":"):
                continue
            if _base(mon) in opp_active_species:
                continue
            d = copy.deepcopy(mon.to_dict())
            d["seen"] = True
            opp_seen[_base(mon)] = d

        # Fill remaining roster slots as unseen stubs
        opp_bench: list[dict] = []
        for species in self.rosters.get(opp, []):
            if species in opp_active_species:
                continue
            if species in opp_seen:
                opp_bench.append(opp_seen[species])
            else:
                opp_bench.append({
                    "species": species,
                    "hp_pct": None,
                    "status": None,
                    "boosts": {},
                    "is_mega": False,
                    "is_fainted": False,
                    "revealed_moves": [],
                    "known_item": None,
                    "known_tera_type": None,
                    "is_terastallized": False,
                    "seen": False,
                })

        return {
            "field": copy.deepcopy(self.field_conditions.to_dict()),
            # Keyed by "our_side" / "opp_side" so the consumer never needs to
            # know which physical player the perspective corresponds to.
            "side_conditions": {
                "our_side": copy.deepcopy(self.side_conditions[our].to_dict()),
                "opp_side": copy.deepcopy(self.side_conditions[opp].to_dict()),
            },
            "our_active": our_active,
            "opp_active": opp_active,
            "our_bench": our_bench,
            "opp_bench": opp_bench,
            "known_team": {
                pid: list(team) for pid, team in self.known_team.items()
            },
        }

    # ------------------------------------------------------------------
    # Turn flush
    # ------------------------------------------------------------------

    def _flush_turn(self) -> None:
        """Snapshot the board state before + after and record all actions."""
        our = self.our_player
        opp = "p2" if our == "p1" else "p1"

        # Extract decision-level actions (move / switch) per side.
        # These are what the model needs to learn from — distinct from all the
        # passive events (damage, boosts, faints) that also live in actions[].
        def _extract_actions(player: str) -> list[dict]:
            out = []
            for ev in self._current_turn_actions:
                if ev.get("event") not in ("move", "switch"):
                    continue
                slot = ev.get("user_slot") or ev.get("slot") or ""
                if not slot.startswith(player):
                    continue
                if ev["event"] == "move":
                    out.append({
                        "slot": slot,
                        "action": "move",
                        "move": ev.get("move"),
                        "target_slot": ev.get("target_slot"),
                        "is_protect": ev.get("is_protect", False),
                        "execution_index": ev.get("execution_index"),
                    })
                else:  # switch
                    out.append({
                        "slot": slot,
                        "action": "switch",
                        "species": ev.get("species"),
                    })
            return out

        turn_snapshot = {
            "turn": self.current_turn,
            # Both-perspective snapshots captured at the START of the turn.
            # transitions.py indexes by perspective; the parser's own our_player
            # is stored so callers can resolve which side is "ours" without
            # having to thread our_player through every layer.
            "state_before_actions": self._state_before or {
                "p1": self._snapshot_state("p1"),
                "p2": self._snapshot_state("p2"),
            },
            # Flat event log (moves, damage, boosts, faints, etc.) — for replays
            "actions": list(self._current_turn_actions),
            "damage_events": list(self._current_turn_damage_events),
            # Both-perspective snapshots captured AFTER all actions have resolved.
            "state_after_actions": {
                "p1": self._snapshot_state("p1"),
                "p2": self._snapshot_state("p2"),
            },
            # ── Training-schema decision fields ──────────────────────────
            "our_actions": _extract_actions(our),
            "opp_actions_actual": _extract_actions(opp),
            # ---- Type B: prediction fields always null ----
            "predicted_action_by_bot": None,
            "opp_actions_predicted": None,
        }
        self.turns.append(turn_snapshot)

    # ------------------------------------------------------------------
    # Output assembly
    # ------------------------------------------------------------------

    def _build_output(self) -> dict:
        # Aggregate everything the parser learned across the whole match into a
        # flat dict keyed by "pid:species".  This is what the inject panel uses
        # to pre-populate revealed moves, items, tera type, and ability.
        dex = get_pokedex()
        revealed_info: dict[str, dict] = {}
        for seen_key, mon in self.seen_mons.items():
            # seen_key is already "pid:species" (BASE species — Bug 7)
            base = mon.base_species or mon.species
            revealed_info[seen_key] = {
                "revealed_moves": list(mon.revealed_moves),
                "known_item": "mega stone" if mon.is_mega else mon.known_item,
                "known_tera_type": mon.known_tera_type,
                "is_terastallized": mon.is_terastallized,
                # Bug 8: split ability contexts.  known_ability == the ability
                # active at the END of the match; pre_mega_ability is what the
                # base forme ran (the only user-editable one for mega mons);
                # mega_ability is the mega forme's fixed ability.
                "known_ability": mon.known_ability,
                "pre_mega_ability": mon.pre_mega_ability,
                "mega_ability": mon.mega_ability,
                "is_mega": mon.is_mega,
                "mega_species": mon.species if mon.is_mega else None,
                # Choice-item constraint: False = used 2+ different moves in
                # one stay on the field → cannot hold Choice Scarf/Band/Specs.
                "can_have_choice_item": mon.can_have_choice_item,
                # Pokedex-derived dropdown data for the inject panel.
                "possible_abilities": dex.abilities_for(base) if dex else [],
                "mega_formes": dex.mega_formes_for(base) if dex else [],
            }

        return {
            "source_type": "ranked_player_vod",
            "replay_id": None,      # caller can inject from filename
            "format": self.format,
            "players": {
                "our_side": self.our_player,
                # roster = the full 6-mon teampreview pool; brought = the
                # (≤4) mons that actually entered the battle, in first
                # switch-in order — the first two entries are the leads.
                # Teampreview-choice models train on roster → brought.
                "p1": {
                    "username": self.players.get("p1"),
                    "rating_before": self.ratings.get("p1"),
                    "rating_delta": self.rating_deltas.get("p1"),
                    "roster": self.rosters["p1"],
                    "brought": list(self.known_team["p1"]),
                    "team_size_chosen": self.team_sizes.get("p1"),
                },
                "p2": {
                    "username": self.players.get("p2"),
                    "rating_before": self.ratings.get("p2"),
                    "rating_delta": self.rating_deltas.get("p2"),
                    "roster": self.rosters["p2"],
                    "brought": list(self.known_team["p2"]),
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
    # Slot normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_slots(
        events: list[dict],
        perspective: str,
        slot_fields: tuple[str, ...] = ("slot", "target_slot", "source_slot", "user_slot"),
    ) -> list[dict]:
        """
        Rewrite absolute slot strings (``p1a``, ``p2b``, …) to perspective-
        relative ones (``our_a``, ``our_b``, ``opp_a``, ``opp_b``) in a list
        of event dicts.

        This is applied at transition-write time in transitions.py so the
        state encoder always sees the same slot schema regardless of which
        physical player a transition was generated for.

        The parser itself always stores absolute notation internally; this
        function is the boundary where we switch to relative notation.
        """
        opp = "p2" if perspective == "p1" else "p1"

        def _remap(val: Optional[str]) -> Optional[str]:
            if not val:
                return val
            if val.startswith(perspective):
                return "our_" + val[2:]   # e.g. p1a -> our_a
            if val.startswith(opp):
                return "opp_" + val[2:]   # e.g. p2b -> opp_b
            return val                     # non-slot strings pass through

        out = []
        for ev in events:
            ev2 = dict(ev)
            for field in slot_fields:
                if field in ev2:
                    ev2[field] = _remap(ev2[field])
            out.append(ev2)
        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    # Bug 4: canonical terrain tokens the encoder can rely on.
    _TERRAIN_MAP = {
        "electric terrain": "electric",
        "grassy terrain":   "grassy",
        "misty terrain":    "misty",
        "psychic terrain":  "psychic",
    }

    @classmethod
    def _normalize_terrain(cls, raw: Optional[str]) -> Optional[str]:
        """
        Map a raw -fieldstart effect string to a canonical terrain token.

        Handles all the formats Showdown emits, e.g.::

            "move: Electric Terrain"
            "Electric Terrain"
            "move: Grassy Terrain|[from] ability: Grassy Surge"

        Returns "electric" / "grassy" / "misty" / "psychic", or None if the
        string doesn't contain a recognised terrain.
        """
        if not raw:
            return None
        text = raw.lower()
        if text.startswith("move:"):
            text = text.split(":", 1)[1].strip()
        for key, token in cls._TERRAIN_MAP.items():
            if key in text:
                return token
        return None

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
