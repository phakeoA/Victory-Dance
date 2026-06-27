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

from v_dance.parser.vod_parser.battle_models import (
    FieldConditions,
    PokemonSlot,
    SideConditions,
)
from v_dance.parser.vod_parser.pokedex import get_pokedex, is_mega_species_name, norm_species


# Species with the Illusion ability (normalised).  Used to resolve a disguise
# when Species Clause makes a duplicate-active species provably an Illusion.
_ILLUSION_SPECIES = {"zoroark", "zoroarkhisui"}

# v11 P3: the 7 boost stats in canonical order (== state_encoder._BOOST_KEYS; defined LOCALLY to avoid a
# parser->encoder import). Order is irrelevant for the whole-dict clear/set/copy/swap/invert ops below.
_ALL_BOOST_STATS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")


def _perceived_roster_name(display: Optional[str]) -> str:
    """Map a PERCEIVED active species onto the base name the teampreview roster
    lists it under, so disguise/active exclusion matches the roster.

    The roster (|poke|) lists BASE formes, so a perceived mega forme — e.g. a
    Zoroark copying a mega-evolved teammate appears as ``"Charizard-Mega-Y"`` —
    must map back to its base ``"Charizard"`` to consume the right roster slot.
    Non-mega formes (regional, etc.) are already listed under their own name.
    Mirrors the non-illusion branch's use of a mon's frozen ``base_species``.
    """
    if not display:
        return ""
    if is_mega_species_name(display):
        return display.split("-Mega")[0]
    return display


# Items that change hands SILENTLY (no |-item|/|-enditem|) and so are held by at most one mon at a
# time — used to clear a stale known_item off a former holder when the item resurfaces elsewhere.
_TRANSFER_ONLY_ITEMS = frozenset({"Sticky Barb"})

# v11 A.3: moves that increment poke-env's consecutive-Protect counter (poke_env battle/move.py
# _PROTECT_COUNTER_MOVES = _PROTECT_MOVES | {wideguard, quickguard, endure}). The offline counter is
# driven from the |move| line to byte-match poke-env Pokemon.moved (pokemon.py:462-465). NOTE this is
# DISTINCT from the per-move ``is_protect`` set below (which omits endure/kingsshield/obstruct and
# includes matblock); Mat Block does NOT increment the counter.
_PROTECT_COUNTER_MOVES = frozenset({
    "protect", "detect", "endure", "spikyshield", "kingsshield",
    "banefulbunker", "burningbulwark", "obstruct", "maxguard",
    "silktrap", "wideguard", "quickguard",
})
# |move|-line suffixes poke-env reads as a FAILED move (abstract_battle.py:589-595); a failed
# protect-counter move resets the counter to 0 (never increments).
_MOVE_FAILED_SUFFIXES = frozenset({"[miss]", "[still]", "[notarget]"})


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
        # Post-faint replacement decisions made during this turn.  Each entry is
        # {player, slot, species, state:{p1,p2}} captured AFTER the faint and
        # BEFORE the replacement switches in — its own decision state for the
        # "who to send in after a faint" policy target.
        self._current_turn_replacements: list[dict] = []

        # Snapshot captured at the START of each turn (before any mutation)
        self._state_before: Optional[dict] = None

        # Tracks the most recent move action so damage events can be attributed
        self._last_move_action: Optional[dict] = None

        # Monotone counter for action execution order within a turn
        self._execution_index: int = 0

        # ---- team sizes chosen at preview ----
        self.team_sizes: dict[str, int] = {}

        # ---- Illusion (Zoroark) pre-scan state ----
        # Index of the line currently being handled, so _handle_switch can look
        # up whether *this* switch is really a disguised Zoroark (resolved by a
        # later |replace|).  _illusion_truth maps switch-line index -> the TRUE
        # DETAILS string ("Zoroark-Hisui, L50, M") revealed for it.
        self._line_idx: int = -1
        self._illusion_truth: dict[int, str] = {}
        # Match-level flags surfaced in the output for downstream awareness.
        self._illusion_seen: bool = False
        self._transform_seen: bool = False

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def parse(self) -> dict:
        """Parse all lines and return the full structured output."""
        # First pass: resolve Illusion disguises so the main pass can build each
        # disguised slot as the real Zoroark from its first frame on the field.
        self._illusion_truth = self._prescan_illusions()
        for idx, line in enumerate(self.lines):
            self._line_idx = idx
            self._handle_line(line)

        # Flush any last in-progress turn
        if (self._current_turn_actions or self._current_turn_damage_events
                or self._current_turn_replacements):
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

        # Items announce themselves by their effect (Life Orb recoil, Leftovers/Black Sludge
        # heal, Rocky Helmet contact, Flame/Toxic Orb status) via a "[from] item:" tag — learn the
        # HELD item from it. |-item|/|-enditem| carry the item in a field, not this tag, so they
        # are handled separately and never double-counted here.
        if cmd not in ("-item", "-enditem"):
            self._learn_item_from_tags(parts)

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
            self._current_turn_replacements = []
            self._last_move_action = None
            self._execution_index = 0
            # Field-duration: advance turns-ACTIVE at the START of each turn, anchored on the SET turn,
            # so a pre-|turn|1 lead weather/terrain (Drought / Electric Surge on switch-in) reads
            # age = current_turn − set_turn, matching poke-env's ``turn − start-turn`` on the live path.
            # Done here (NOT at upkeep) so the snapshot below reflects the same elapsed both paths see.
            if self.field_conditions.weather:
                self.field_conditions.weather_turns += 1
            if self.field_conditions.terrain:
                self.field_conditions.terrain_turns += 1
            for sc in self.side_conditions.values():
                for k in list(sc.screens):       # screens count UP; cap at the 8-turn Light Clay max
                    sc.screens[k] += 1
                    if sc.screens[k] > 8:
                        del sc.screens[k]
            # v11 C.3: active-turns +1 for every mon on the field at turn-start (mirrors poke-env
            # end_turn, called per active mon on |turn|). A fainted-but-not-yet-replaced slot does NOT
            # advance (it is leaving); its replacement is reset to 0 by _handle_switch and reaches 1 on
            # the NEXT |turn|. Done BEFORE the snapshot so first_turn=(active_turns==1) reflects the
            # decision state both paths see.
            for _m in self.active_slots.values():
                if not _m.is_fainted:
                    _m.active_turns += 1
            # Capture state BEFORE any actions mutate it
            self._state_before = {
                "p1": self._snapshot_state("p1"),
                "p2": self._snapshot_state("p2"),
            }

        # ---- field events ----
        elif cmd == "switch" or cmd == "drag":
            self._handle_switch(parts)

        elif cmd == "replace":
            # |replace|p2a: Zoroark|Zoroark-Hisui, L50, M
            # Illusion broke — the slot's true identity is revealed.
            self._handle_replace(parts)

        elif cmd == "swap":
            # |swap|p2a: Farigiraf|1|[from] move: Ally Switch
            # A position-shifting move (Ally Switch) moved an active mon to a
            # different field slot — swap the slot pointers so later faints /
            # replacements target the right slot (Gap #6 sub-cause 1).
            self._handle_swap(parts)

        elif cmd == "-transform":
            # |-transform|p1a: Ditto|p2b: Whimsicott|[from] ability: Imposter
            self._handle_transform(parts)

        elif cmd == "detailschange":
            self._handle_mega(parts)

        elif cmd == "-formechange":
            # |-formechange|p1a: Aegislash|Aegislash-Blade  — NON-permanent forme (Stance Change/Forecast/
            # Hunger Switch). isPermanent formes (Mimikyu-Busted, Palafin-Hero, Eiscue-Noice) arrive as
            # |detailschange| above. Same [ident, species] shape → reuse _handle_mega's non-mega branch
            # (mutates mon.species, no is_mega). In-meta: Aegislash Stance (the Atk/Def-SpA swap). Reversion
            # (Blade→Shield) is a free -formechange|…|Aegislash line; a SILENT switch-out revert keeps the
            # forme stats on BOTH encoders (poke-env's _update_from_details early-returns on identical switch
            # details → keeps Blade), so parity holds with NO switch-in species reset.
            self._handle_mega(parts)

        elif cmd == "-mega":
            # |-mega|p1a: Floette|Floette|Floettite  — explicit stone reveal
            self._handle_mega_stone(parts)

        elif cmd == "move":
            self._handle_move(parts)

        elif cmd == "-damage" or cmd == "-heal":
            self._handle_damage(parts, cmd)

        elif cmd == "-sethp":          # v11 P4: Pain Split etc. — set HP directly (poke-env set_hp)
            self._handle_sethp(parts)

        elif cmd == "faint":
            slot_key = self._slot_key_from_ident(parts[2])
            fainted_species = self._species_from_ident(parts[2])
            self._faint_mon(slot_key, fainted_species)
            self._current_turn_actions.append({
                "event": "faint",
                "slot": slot_key,
                "species": fainted_species,
            })

        elif cmd == "cant":
            # |cant|p1a: X|flinch (and other reasons) — poke-env cant_move resets the consecutive-
            # Protect counter (pokemon.py:189-191). v11 A.3: mirror that and nothing else.
            slot_key = self._slot_key_from_ident(parts[2]) if len(parts) > 2 else None
            if slot_key in self.active_slots:
                self.active_slots[slot_key].protect_counter = 0

        elif cmd == "-singleturn" or cmd == "-singlemove":
            self._handle_singleturn(parts, cmd)

        elif cmd == "-boost" or cmd == "-unboost":
            self._handle_boost(parts, cmd)

        # v11 P3: the remaining boost-mutation messages (parser-only; offline boosts dict mirrors poke-env)
        elif cmd == "-clearboost":
            self._handle_clearboost(parts)
        elif cmd == "-clearallboost":
            self._handle_clearallboost()
        elif cmd == "-clearnegativeboost":
            self._handle_clear_signed(parts, "neg")
        elif cmd == "-clearpositiveboost":
            self._handle_clear_signed(parts, "pos")
        elif cmd == "-setboost":
            self._handle_setboost(parts)
        elif cmd == "-copyboost":
            self._handle_copyboost(parts)
        elif cmd == "-swapboost":
            self._handle_swapboost(parts)
        elif cmd == "-invertboost":
            self._handle_invertboost(parts)

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
            else:
                # #12: a team-cure (Heal Bell / Aromatherapy) or switch-out cure (Natural Cure) emits a
                # BARE 'pN: Nickname' ident (no a/b) for a BENCHED member; _slot_key_from_ident then falls
                # back to 'pN' (not in active_slots) and the bench mon kept its stale status forever.
                # Resolve by nickname (poke-env keys its team dict by 'pN: Nickname'), matching the live
                # encoder so offline parse == live serve.
                player = slot_key[:2]
                nick = parts[2].split(": ", 1)[1] if ": " in parts[2] else None
                if nick:
                    for m in self.seen_mons.values():
                        if m.player == player and m.nickname == nick and not m.is_fainted:
                            m.status = None
            # Bug 3 fix: record the event so status changes between
            # state_before and state_after are explained in the action log.
            self._current_turn_actions.append({
                "event": "curestatus",
                "slot": slot_key,
                "species": self._species_from_ident(parts[2]),
                "status": cured,
            })

        elif cmd == "-cureteam":
            # #12: legacy whole-party cure (older Heal Bell form). Clear status on every NON-fainted
            # member of that side (bench seen_mons + active), mirroring poke-env's team.values() cure loop.
            player = self._slot_key_from_ident(parts[2])[:2]
            for m in self.seen_mons.values():
                if m.player == player and not m.is_fainted:
                    m.status = None
            for k, m in self.active_slots.items():
                if k.startswith(player) and not getattr(m, "is_fainted", False):
                    m.status = None

        elif cmd == "-start":
            self._handle_volatile(parts, start=True)

        elif cmd == "-end":
            self._handle_volatile(parts, start=False)

        elif cmd == "-sidestart":
            self._handle_sidestart(parts)

        elif cmd == "-sideend":
            self._handle_sideend(parts)

        elif cmd == "-swapsideconditions":
            # |-swapsideconditions (Court Change) — swap ALL side conditions (hazards/screens/tailwind)
            # between the two sides, mirroring poke-env (abstract_battle swaps the two dicts).
            self.side_conditions["p1"], self.side_conditions["p2"] = (
                self.side_conditions["p2"], self.side_conditions["p1"])

        elif cmd == "-weather":
            weather_val = parts[2] if len(parts) > 2 else None
            new_w = None if weather_val in (None, "none") else weather_val
            if new_w != self.field_conditions.weather:
                # New weather (or cleared) → restart the elapsed counter. Repeated
                # ``|-weather|X|[upkeep]`` lines carry the same token and are ignored here;
                # the ``upkeep`` command increments the counter while weather is active.
                self.field_conditions.weather = new_w
                self.field_conditions.weather_turns = 0

        elif cmd == "-fieldstart":
            raw = parts[2] if len(parts) > 2 else ""
            if "Trick Room" in raw:
                self.field_conditions.trick_room = 5
            elif "Gravity" in raw:
                # v11 C.2e: base 5; the Persistent ability extends to 7
                # (|-fieldstart|move: Gravity|[persistent]).
                self.field_conditions.gravity = (
                    7 if any("persistent" in str(p).lower() for p in parts) else 5)
            elif "Magic Room" in raw:                    # v11 P5: base 5; Persistent -> 7 (same as Gravity)
                self.field_conditions.magic_room = (
                    7 if any("persistent" in str(p).lower() for p in parts) else 5)
            elif "Wonder Room" in raw:
                self.field_conditions.wonder_room = (
                    7 if any("persistent" in str(p).lower() for p in parts) else 5)
            elif "terrain" in raw.lower():
                # Bug 4 fix: store a normalised token ("electric", "grassy",
                # "misty", "psychic"), not the raw effect string.
                self.field_conditions.terrain = self._normalize_terrain(raw)
                self.field_conditions.terrain_turns = 0

        elif cmd == "-fieldend":
            raw = parts[2] if len(parts) > 2 else ""
            if "Trick Room" in raw:
                self.field_conditions.trick_room = 0
            elif "Gravity" in raw:
                self.field_conditions.gravity = 0
            elif "Magic Room" in raw:                    # v11 P5
                self.field_conditions.magic_room = 0
            elif "Wonder Room" in raw:
                self.field_conditions.wonder_room = 0
            elif "terrain" in raw.lower():
                self.field_conditions.terrain = None
                self.field_conditions.terrain_turns = 0

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

        elif cmd == "-activate":
            # Symbiosis (silent ally item pass) is the one -activate that moves an item;
            # the line names giver, item and receiver, so record the hand-off.
            self._handle_activate(parts)

        elif cmd == "upkeep":
            # tailwind / trick room = REMAINING (base 4 / 5, no item extension); their age is derived
            # at encode time as base − remaining. Weather/terrain/screens count UP at turn-start (above).
            for sc in self.side_conditions.values():
                if sc.tailwind > 0:
                    sc.tailwind -= 1
            if self.field_conditions.trick_room > 0:
                self.field_conditions.trick_room -= 1
            if self.field_conditions.gravity > 0:        # v11 C.2e
                self.field_conditions.gravity -= 1
            if self.field_conditions.magic_room > 0:     # v11 P5
                self.field_conditions.magic_room -= 1
            if self.field_conditions.wonder_room > 0:
                self.field_conditions.wonder_room -= 1

    # ------------------------------------------------------------------
    # Sub-handlers
    # ------------------------------------------------------------------

    def _faint_mon(self, slot_key: str, fainted_species: Optional[str]) -> None:
        """Mark the mon that fainted, by the SPECIES the |faint| line names.

        Trust the named species over the slot pointer: an Illusion that mimicked
        one of its OWN teammates' species can desync slot tracking (active_slots
        [slot] ends up pointing at the wrong twin), which would otherwise faint
        the wrong Pokémon (and leave the real one wrongly alive).  When the
        slot's occupant doesn't match the named species, faint the mon the log
        actually names — searching the player's active mons first, then their
        bench (seen_mons).  Falls back to the slot pointer when nothing matches
        (e.g. an unbroken disguise the log never resolves)."""
        mon = self.active_slots.get(slot_key)
        want = norm_species(fainted_species) if fainted_species else None
        # #13: when the slot's occupant is the ACTIVE disguise the |faint| line names (a Species-Clause-
        # relabeled Zoroark whose base_species is now 'Zoroark' but which faints still showing the
        # teammate's disguise name), TRUST the slot pointer — else the named-species search below would
        # faint the genuine teammate of that species and leave the dead Zoroark alive.
        slot_is_named_disguise = (
            mon is not None
            and getattr(mon, "illusion_active", False)
            and want is not None
            and norm_species(getattr(mon, "disguise_species", None)) == want
        )
        if want and not slot_is_named_disguise and (mon is None
                     or norm_species(mon.base_species or mon.species) != want):
            player = slot_key[:2]
            match = next(
                (m for k, m in self.active_slots.items()
                 if k.startswith(player)
                 and norm_species(m.base_species or m.species) == want),
                None,
            )
            if match is None:
                match = next(
                    (m for k, m in self.seen_mons.items()
                     if k.startswith(player + ":")
                     and norm_species(m.base_species or m.species) == want),
                    None,
                )
            if match is not None:
                mon = match
        if mon is not None:
            mon.is_fainted = True
            mon.hp_current = 0.0

    def _handle_swap(self, parts: list[str]) -> None:
        """Handle ``|swap|POKEMON|POSITION`` — a position-shifting move.

        Ally Switch (and a few field-control moves) relocate an already-active
        mon to a different field slot: the named mon moves to active-field
        ``POSITION`` (0 = slot a, 1 = slot b, …) and whatever was there moves to
        its old slot.  poke-env applies this to its DoubleBattle, so the offline
        parser must too — otherwise the two slot pointers stay desynced and a
        later ``|faint|``/forced ``|switch|`` on a relabeled slot overwrites the
        WRONG (still-living) mon, dropping it out of the active set entirely
        (Gap #6 sub-cause 1: Ally Switch before a faint+replacement).
        """
        ident = parts[2] if len(parts) > 2 else ""
        src_slot = self._slot_key_from_ident(ident)
        if not src_slot or len(src_slot) < 3:
            return
        player = src_slot[:2]
        try:
            pos = int(parts[3]) if len(parts) > 3 else -1
        except (ValueError, TypeError):
            return
        dst_letter = chr(ord("a") + pos)
        dst_slot = player + dst_letter
        if pos < 0 or dst_slot == src_slot:
            return
        # Pop both, then reassign swapped (an empty slot simply stays empty).
        src_mon = self.active_slots.pop(src_slot, None)
        dst_mon = self.active_slots.pop(dst_slot, None)
        if dst_mon is not None:
            self.active_slots[src_slot] = dst_mon
            dst_mon.slot = src_slot[2]
        if src_mon is not None:
            self.active_slots[dst_slot] = src_mon
            src_mon.slot = dst_letter
        self._current_turn_actions.append({
            "event": "swap",
            "slot": src_slot,
            "to_slot": dst_slot,
        })

    def _handle_switch(self, parts: list[str]) -> None:
        # |switch|p1b: Incineroar|Incineroar, L50, F|100/100
        ident = parts[2]
        details = parts[3]
        hp_str = parts[4] if len(parts) > 4 else "100/100"

        slot_key = self._slot_key_from_ident(ident)
        # Illusion (Zoroark): if the pre-scan flagged this switch as a disguised
        # Zoroark — resolved by a later |replace| at this slot — swap in the
        # TRUE details now, so the slot gets its own (correct) seen_mons card,
        # brought entry, and move history from its first frame on the field
        # rather than poisoning the disguise species it was masquerading as.
        disguise_truth = self._illusion_truth.get(self._line_idx)
        disguise_species = None
        if disguise_truth:
            disguise_species = details.split(",")[0].strip()   # the DISGUISE name
            details = disguise_truth
            self._illusion_seen = True
        nickname = ident.split(": ", 1)[1] if ": " in ident else ident
        species = details.split(",")[0].strip()
        player = slot_key[:2]

        hp_current, hp_max = self._parse_hp(hp_str)

        # Forced replacement detection: this |switch| fills a slot whose mon
        # fainted earlier THIS turn (its PokemonSlot is still in active_slots,
        # flagged is_fainted) — as opposed to a voluntary turn-start switch of a
        # LIVING mon.  Capture the post-faint board (fainted mon excluded from
        # active, living bench available) as its own decision state BEFORE the
        # switch mutates anything, so the in-battle policy can learn the
        # "who to send in after a faint" choice.  ``species`` is already the
        # true (de-disguised) identity at this point.
        # #00: capture this BEFORE line ~660 reassigns active_slots[slot_key] to the incoming mon.
        is_forced_replacement = (slot_key in self.active_slots
                                 and self.active_slots[slot_key].is_fainted)
        if is_forced_replacement:
            self._current_turn_replacements.append({
                "player": player,
                "slot": slot_key,
                "species": species,
                "state": {
                    "p1": self._snapshot_state("p1"),
                    "p2": self._snapshot_state("p2"),
                },
            })

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
            outgoing = self.active_slots[slot_key]
            outgoing.boosts = {}
            # v9: volatiles + ability suppression end when the mon leaves the field.
            outgoing.volatiles = set()
            outgoing.ability_suppressed = False
            outgoing.protect_counter = 0          # v11 A.3: counter resets on switch-out (poke-env)
            outgoing.taunt = False                # v11 exact-mask: move-restrictions end on leaving
            outgoing.disabled_move = None
            outgoing.encore_move = None
            # Transform reverts the instant the mon leaves the field, so the
            # outgoing mon is its real self again on the bench.
            outgoing.is_transformed = False
            outgoing.transformed_into = None
            outgoing.runtime_types = None      # v11 P2: typechange ends on leaving the field (poke-env switch_out)

        self.active_slots[slot_key] = mon
        # Incoming mon always starts with neutral boosts (covers |drag| too,
        # and guards against any stale state on the reused PokemonSlot).
        mon.boosts = {}
        # v9 (B1-mechanics): volatiles (Substitute/Ingrain/Taunt/Encore/partial-trap/…) and ability
        # suppression (Gastro Acid) end when a mon leaves the field — the incoming mon starts clean
        # (also guards a reused PokemonSlot).
        mon.volatiles = set()
        mon.ability_suppressed = False
        mon.protect_counter = 0          # v11 A.3: counter resets on switch-in (poke-env switch_out)
        mon.active_turns = 0             # v11 C.3: switch-in starts a fresh stay (==1 next |turn|)
        mon.times_attacked = 0           # v11 C.4: per-stint Rage Fist hit counter resets on switch-in
        mon.taunt = False                # v11 exact-mask: move-restrictions end on switch-in
        mon.disabled_move = None
        mon.encore_move = None
        # A switch-in starts a fresh stay on the field — a Choice lock from a
        # previous stint no longer applies, so the working move set restarts.
        mon.stint_moves = []
        # Transform reverts on switch-out: a mon coming back in is its real
        # self again.  (An Imposter Ditto re-fires |-transform| immediately, so
        # this is correctly re-set right after; a Transform-move user that left
        # the field returns un-transformed.)
        mon.is_transformed = False
        mon.transformed_into = None
        mon.runtime_types = None          # v11 P2: incoming mon starts with its real types (no typechange)
        # Illusion: reset on every switch-in, then flag if THIS switch-in was a
        # disguised Zoroark (the pre-scan paired it with a later |replace|).
        # Opponents see `disguise_species` until the disguise breaks.
        mon.illusion_active = bool(disguise_truth)
        mon.disguise_species = disguise_species

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
            # #00: a post-faint forced refill is NOT a turn-start decision (the genuine "who to send in"
            # choice is the separate decision_type="replacement" transition). Flag it so _extract_actions
            # drops it from our_actions / opp_actions_actual — else it becomes a FALSE turn-start switch
            # label (especially when the KO'd mon never acted, so it's the slot's ONLY entry).
            "forced_replacement": is_forced_replacement,
        })

    def _prescan_illusions(self) -> dict[int, str]:
        """Resolve Illusion disguises in a cheap first pass over the log.

        Zoroark / Zoroark-Hisui (ability Illusion) enter the field disguised as
        the last unfainted party member, so the |switch| line names the DISGUISE
        species.  The mask only drops when a direct damaging hit breaks it, at
        which point Showdown emits::

            |replace|p2a: Zoroark|Zoroark-Hisui, L50, M
            |-end|p2a: Zoroark|Illusion

        We walk the log once, remembering the most recent switch-in line per
        slot, and when a |replace| reveals the truth we tag that originating
        switch with the real DETAILS string.  The main pass then constructs the
        slot as the real species from its first frame, so the disguise species
        is never credited with Zoroark's moves and Zoroark gets its own card.

        Mapping to the *most recent* switch at the slot is what keeps a genuine
        copy of the disguise species apart from the impostor: in a battle where
        the real Charizard also switches in later, only the pre-reveal switch is
        relabeled.

        Returns ``{switch_line_index: true_details_string}``.  An Illusion that
        is never broken (Zoroark faints disguised, or the log ends first) is
        genuinely unobservable and is left as the disguise — the same blind
        spot a human spectator has.

        SPECIES-CLAUSE RESOLUTION (the hard cross-stint case): a Zoroark can
        disguise as a teammate that the team ALSO genuinely brings, and switch
        out before any |replace| breaks that stint.  Because Species Clause
        forbids two of the same species on a team, seeing the SAME species
        active in both slots at once is *provably* an Illusion — the disguise is
        the one already on the field (Illusion mimics a benched teammate, who
        enters later), so we relabel its switch to the side's roster Illusion
        mon.  This keeps the disguise's moves credited to Zoroark and frees the
        real teammate to keep its own card (otherwise it collides and is lost).
        """
        truth: dict[int, str] = {}
        last_switch: dict[str, int] = {}
        active_base: dict[str, str] = {}   # slot -> base species currently SHOWN
        rosters: dict[str, list[str]] = {}
        for idx, line in enumerate(self.lines):
            if not line.startswith("|"):
                continue
            parts = line.split("|")
            if len(parts) < 2:
                continue
            cmd = parts[1]
            if cmd == "poke":
                pid = parts[2] if len(parts) > 2 else ""
                sp = parts[3].split(",")[0].strip() if len(parts) > 3 else ""
                if pid and sp:
                    rosters.setdefault(pid, []).append(sp)
            elif cmd in ("switch", "drag"):
                slot = self._slot_key_from_ident(parts[2] if len(parts) > 2 else "")
                player = slot[:2]
                species = parts[3].split(",")[0].strip() if len(parts) > 3 else ""
                base = norm_species(species)
                if base:
                    # Duplicate-active species ⇒ provably an Illusion (Species
                    # Clause).  Relabel the disguise already on the field.
                    for oslot, obase in active_base.items():
                        if (oslot != slot and oslot[:2] == player and obase == base
                                and last_switch.get(oslot) is not None):
                            illu = next((r for r in rosters.get(player, [])
                                         if norm_species(r) in _ILLUSION_SPECIES), None)
                            o_switch = last_switch[oslot]
                            if illu and o_switch not in truth:
                                truth[o_switch] = f"{illu}, L50"
                            break
                    active_base[slot] = base
                last_switch[slot] = idx
            elif cmd == "replace":
                slot = self._slot_key_from_ident(parts[2] if len(parts) > 2 else "")
                details = parts[3] if len(parts) > 3 else ""
                origin = last_switch.get(slot)
                if origin is not None and details:
                    truth[origin] = details          # a reveal is ground truth
                    active_base[slot] = norm_species(details.split(",")[0].strip())
        return truth

    def _handle_replace(self, parts: list[str]) -> None:
        """Handle |replace| — an Illusion has just been unmasked.

        In the normal case the pre-scan already relabeled the originating switch
        (see _prescan_illusions), so the slot is *already* tracked as the true
        species and this only emits the reveal event.  The defensive relabel
        below is a fallback for malformed logs where the pre-scan couldn't pair
        the |replace| with a switch.
        """
        ident = parts[2] if len(parts) > 2 else ""
        details = parts[3] if len(parts) > 3 else ""
        slot_key = self._slot_key_from_ident(ident)
        true_species = details.split(",")[0].strip() if details else None
        self._illusion_seen = True

        mon = self.active_slots.get(slot_key)
        if mon is not None:
            # The disguise just broke — from now on opponents see the truth.
            mon.illusion_active = False
        if mon is not None and true_species:
            current_base = mon.base_species or mon.species
            if current_base != true_species:
                # Fallback: the pre-scan didn't catch this disguise.  Relabel
                # the live slot forward so post-reveal state is at least correct.
                player = slot_key[:2]
                self.seen_mons.pop(f"{player}:{current_base}", None)
                mon.species = true_species
                mon.base_species = true_species
                mon.nickname = ident.split(": ", 1)[1] if ": " in ident else mon.nickname
                self.seen_mons[f"{player}:{true_species}"] = mon
                if true_species not in self.known_team[player]:
                    self.known_team[player].append(true_species)

        self._current_turn_actions.append({
            "event": "illusion_revealed",
            "slot": slot_key,
            "species": true_species,
        })

    def _handle_transform(self, parts: list[str]) -> None:
        """Handle |-transform| — Ditto/Imposter or the Transform move.

        Marks the slot as transformed and records the copied species.  From
        here until it leaves the field, the mon's move usage is filtered out of
        revealed_moves and the Choice-item constraint (see _handle_move): those
        are the target's moves, not this mon's.
        """
        ident = parts[2] if len(parts) > 2 else ""
        target_ident = parts[3] if len(parts) > 3 else ""
        slot_key = self._slot_key_from_ident(ident)
        into_species = self._species_from_ident(target_ident)
        if into_species is None and target_ident:
            into_species = target_ident.split(",")[0].strip() or None
        self._transform_seen = True

        if slot_key in self.active_slots:
            mon = self.active_slots[slot_key]
            mon.is_transformed = True
            mon.transformed_into = into_species
            mon.ever_transformed = True
            # v11 P2: transform overwrites poke-env _temporary_types with the copied dex types; offline
            # resolves those via transformed_into, so clear any prior typechange (transform branch wins).
            mon.runtime_types = None
            tgt = self._slot_key_from_ident(target_ident)   # v11 C.4: transform copies timesAttacked
            if tgt in self.active_slots:
                mon.times_attacked = self.active_slots[tgt].times_attacked

        self._current_turn_actions.append({
            "event": "transform",
            "slot": slot_key,
            "species": self._species_from_ident(ident),
            "into_species": into_species,
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
            # A [from]-tagged move was CALLED by another effect (Sleep Talk, Dancer, Instruct, Copycat,
            # locked-move continuations) — the player never selected it.  Used by both the Choice-lock
            # logic below and the v11 A.3 protect-counter (poke-env does not count a called move).
            called = any(p.startswith("[from]") for p in parts[4:] if p)

            # v11 A.3: consecutive-Protect counter — byte-parity with poke-env Pokemon.moved
            # (pokemon.py:462-465): increment a self-selected, non-failed protect-counter move; else
            # reset to 0.  Driven from the |move| line (NOT |-singleturn|/|-fail|).  Runs regardless of
            # is_transformed (poke-env updates the counter on the copied move too).  ``failed`` is read
            # only from the |move|-line suffix, exactly as poke-env does.  ([from]-called protect-family
            # moves — e.g. a Sleep-Talk'd Protect — are a documented near-zero residual: excluded here.)
            move_failed = any(p in _MOVE_FAILED_SUFFIXES for p in parts[4:] if p)
            if norm_species(move_name) in _PROTECT_COUNTER_MOVES and not move_failed and not called:
                mon.protect_counter += 1
            else:
                mon.protect_counter = 0

            # Transform (Ditto/Imposter, Mew, …): while transformed the mon is
            # using the COPIED foe's moves with its borrowed stats.  Those moves
            # belong to the target, not this mon, so they must not enter its
            # revealed_moves nor count toward the Choice-item constraint.  The
            # Transform move itself is recorded normally — it fires BEFORE the
            # |-transform| line that flips this flag.
            if mon.is_transformed:
                pass
            else:
                if move_name and move_name not in mon.revealed_moves:
                    mon.revealed_moves.append(move_name)
                # Choice-item constraint (see PokemonSlot.can_have_choice_item):
                # the move must be self-selected (not `called` above) and not
                # Struggle (a choice-locked mon Struggles once its locked move
                # runs out of PP).
                if move_name and not called and move_name.lower() != "struggle":
                    if move_name not in mon.stint_moves:
                        mon.stint_moves.append(move_name)
                    if len(mon.stint_moves) >= 2:
                        mon.can_have_choice_item = False
                    # PP tracking (gap #7): count this self-selected use.  The
                    # transformed / [from]-called / Struggle exclusions above
                    # mirror poke-env's PP `use` flag, so the offline use-count
                    # matches poke-env's current_pp on a replay (Pressure aside).
                    mid = norm_species(move_name)
                    mon.move_pp_used[mid] = mon.move_pp_used.get(mid, 0) + 1

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
            # Keep the denominator current so to_dict()'s hp_pct stays a true
            # percentage even for real-HP (X/Y, Y!=100) replays (gap #5).
            if hp_max:
                self.active_slots[slot_key].hp_max = hp_max

        # hp_pct_after / delta are reported as true 0-100 PERCENTAGES, normalising
        # the raw numerator by the denominator so real-HP logs (e.g. 175/200) are
        # not over-reported (gap #5).
        denom = hp_max or 100.0
        hp_pct_after = None if hp_current is None else round(hp_current / denom * 100.0, 2)
        delta_pct = None
        if prev_hp is not None and hp_current is not None:
            delta_pct = round((hp_current - prev_hp) / denom * 100.0, 2)

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

        # v11 C.4: count DIRECT damaging-move hits toward Rage Fist BP. The SAME [from] filter that gates
        # source_move above excludes chip/recoil/confusion/Stealth-Rock/Rocky-Helmet/Pain-Split/Future-
        # Sight (all carry [from]). A multi-hit move emits one bare |-damage| per hit → +N (matches
        # Showdown timesAttacked += hit). The self-hit guard drops bare self-cost lines (Substitute
        # creation, Belly Drum, Curse) where the user damages itself with no [from] tag. PARITY: the live
        # path sources this same count from the shared offline parse (C.4 Architecture B), so there is no
        # second filter to drift. (We exclude Future Sight via [from] for parity, accepting the minor
        # fidelity gap vs Showdown which would count its delayed hit.)
        if (cmd == "-damage" and not has_from_tag and slot_key in self.active_slots
                and not (self._last_move_action
                         and self._last_move_action.get("user_slot") == slot_key)):
            self.active_slots[slot_key].times_attacked += 1

        event = {
            "event": cmd.lstrip("-"),   # "damage" or "heal"
            "slot": slot_key,
            "species": self._species_from_ident(ident),
            "hp_pct_after": hp_pct_after,
            "hp_pct_delta": delta_pct,
            "source_slot": source_slot,
            "source_species": source_species,
            "source_move": source_move,
        }
        self._current_turn_damage_events.append(event)
        self._current_turn_actions.append(event)

    def _handle_sethp(self, parts: list[str]) -> None:
        """|-sethp|MON|X/Y (Pain Split) — set HP directly (poke-env Pokemon.set_hp / set_hp_status). Mirrors
        the HP-setting half of _handle_damage: parse the numerator/denominator and keep the denominator
        current so to_dict()'s hp_pct stays a true percentage even for real-HP (X/Y, Y!=100) replays
        (gap #5). Status is NOT set here — the offline parser sources status from the dedicated -status /
        -curestatus / faint events (consistent with _handle_damage, whose _parse_hp also strips the suffix);
        Pain Split's -sethp carries no status. NOT counted toward times_attacked (it is not a direct hit)."""
        if len(parts) < 4:
            return
        slot_key = self._slot_key_from_ident(parts[2])
        hp_current, hp_max = self._parse_hp(parts[3])
        if slot_key in self.active_slots and hp_current is not None:
            self.active_slots[slot_key].hp_current = hp_current
            if hp_max:
                self.active_slots[slot_key].hp_max = hp_max

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
            # v11 P3: clamp to [-6, +6] to match poke-env Pokemon.boost() (the logged stages stay RAW)
            self.active_slots[slot_key].boosts[stat] = max(-6, min(6, current + amount))

        self._current_turn_actions.append({
            "event": "stat_change",
            "slot": slot_key,
            "stat": stat,
            "stages": amount,
        })

    # ── v11 P3: boost-manipulation messages (parser-only — the live encoder reads poke-env's native
    # mon.boosts, so each handler just mirrors the matching poke-env Pokemon/Battle method in value) ──
    def _handle_clearboost(self, parts: list[str]) -> None:
        """|-clearboost|MON — zero all 7 stages (poke-env Pokemon.clear_boosts). Trailing [from]/[of] ignored."""
        slot = self._slot_key_from_ident(parts[2]) if len(parts) > 2 else ""
        if slot in self.active_slots:
            self.active_slots[slot].boosts = {s: 0 for s in _ALL_BOOST_STATS}

    def _handle_clearallboost(self) -> None:
        """|-clearallboost (Haze) — zero EVERY active mon both sides (poke-env DoubleBattle.clear_all_boosts).
        active_slots holds exactly the active mons, so benched mons are untouched."""
        for slot in self.active_slots:
            self.active_slots[slot].boosts = {s: 0 for s in _ALL_BOOST_STATS}

    def _handle_clear_signed(self, parts: list[str], sign: str) -> None:
        """|-clearnegativeboost|MON (White Herb) / |-clearpositiveboost|MON — zero only the negative /
        positive stages (poke-env Pokemon.clear_negative_boosts / clear_positive_boosts)."""
        slot = self._slot_key_from_ident(parts[2]) if len(parts) > 2 else ""
        if slot not in self.active_slots:
            return
        b = self.active_slots[slot].boosts
        for stat, val in list(b.items()):
            if (sign == "neg" and val < 0) or (sign == "pos" and val > 0):
                b[stat] = 0

    def _handle_setboost(self, parts: list[str]) -> None:
        """|-setboost|MON|STAT|AMOUNT (Anger Point sets atk +6, Belly Drum +6) — ABSOLUTE set (poke-env
        Pokemon.set_boost), NOT additive. Clamped to [-6,6] defensively (protocol guarantees that range)."""
        slot = self._slot_key_from_ident(parts[2]) if len(parts) > 2 else ""
        stat = parts[3] if len(parts) > 3 else ""
        amount = int(parts[4]) if (len(parts) > 4 and parts[4].lstrip("-").isdigit()) else 0
        if slot in self.active_slots and stat:
            self.active_slots[slot].boosts[stat] = max(-6, min(6, amount))

    def _handle_copyboost(self, parts: list[str]) -> None:
        """|-copyboost|SOURCE|TARGET (Psych Up) — TARGET (parts[3]) RECEIVES a copy of SOURCE (parts[2]),
        matching poke-env get_pokemon(target).copy_boosts(source). Direction is NOT symmetric."""
        if len(parts) < 4:
            return
        src = self._slot_key_from_ident(parts[2])
        tgt = self._slot_key_from_ident(parts[3])
        if src in self.active_slots and tgt in self.active_slots:
            s = self.active_slots[src].boosts
            self.active_slots[tgt].boosts = {st: s.get(st, 0) for st in _ALL_BOOST_STATS}

    def _handle_swapboost(self, parts: list[str]) -> None:
        """|-swapboost|SOURCE|TARGET|STATS (Heart/Guard/Power Swap) — SYMMETRIC exchange. If '[from]' is in
        the STATS arg (parts[4]; full-swap moves like Heart Swap send no explicit list) swap all 7, else the
        ', '-split named subset — matching poke-env (abstract_battle.py:1045-1061, split on ', ')."""
        if len(parts) < 4:
            return
        src = self._slot_key_from_ident(parts[2])
        tgt = self._slot_key_from_ident(parts[3])
        if src not in self.active_slots or tgt not in self.active_slots:
            return
        stats_arg = parts[4] if len(parts) > 4 else ""
        stats = (list(_ALL_BOOST_STATS) if ("[from]" in stats_arg or not stats_arg)
                 else [s for s in stats_arg.split(", ") if s])
        sb = self.active_slots[src].boosts
        tb = self.active_slots[tgt].boosts
        for st in stats:
            sb[st], tb[st] = tb.get(st, 0), sb.get(st, 0)

    def _handle_invertboost(self, parts: list[str]) -> None:
        """|-invertboost|MON (Topsy-Turvy) — negate every stage (poke-env Pokemon.invert_boosts)."""
        slot = self._slot_key_from_ident(parts[2]) if len(parts) > 2 else ""
        if slot in self.active_slots:
            b = self.active_slots[slot].boosts
            self.active_slots[slot].boosts = {st: -v for st, v in b.items()}

    def _handle_volatile(self, parts: list[str], start: bool) -> None:
        """|-start| / |-end| volatile tracking (v9 B1-mechanics): Substitute, Ingrain, Taunt/Encore/
        Disable/Torment, the partial-trap moves, Mean Look/Block, Gastro Acid, recharge/locked states.
        Feeds the per-mon volatile block; cleared on switch-in."""
        if len(parts) < 4:
            return
        mon = self.active_slots.get(self._slot_key_from_ident(parts[2]))
        if mon is None:
            return
        raw = parts[3]
        # strip a "move: " / "ability: " / "item: " prefix, then normalise to a Showdown id
        name = raw.split(":", 1)[-1] if ":" in raw else raw
        vid = "".join(c for c in name.lower() if c.isalnum())
        if not vid:
            return
        # v11 P2: TYPECHANGE (Protean/Libero/Color Change/Soak/Conversion/Burn Up/Double Shock) REPLACES
        # the mon's types. Stored in a dedicated field (NOT the volatile set), read by _effective_types at
        # 2nd precedence (tera > runtime_types > dex). Mirrors poke-env's typechange handler.
        if vid == "typechange":
            mon.runtime_types = self._resolve_typechange(parts) if start else None
            return
        if start:
            mon.volatiles.add(vid)
            # v11 exact-mask: capture the SPECIFIC restricted move so the offline action mask can drop
            # the exact illegal slot. Disable names the move in parts[4]; Encore names none → it locks to
            # the LAST self-selected move (stint_moves[-1], appended in _handle_move BEFORE this -start).
            if vid == "disable":
                mon.disabled_move = parts[4] if len(parts) > 4 else None
            elif vid == "encore":
                mon.encore_move = mon.stint_moves[-1] if mon.stint_moves else None
            elif vid == "taunt":
                mon.taunt = True
        else:
            mon.volatiles.discard(vid)
            if vid == "disable":
                mon.disabled_move = None
            elif vid == "encore":
                mon.encore_move = None
            elif vid == "taunt":
                mon.taunt = False
        if vid == "gastroacid":
            mon.ability_suppressed = start

    def _resolve_typechange(self, parts: list[str]) -> Optional[list]:
        """Resolve the RESULT types of a |-start|MON|typechange| line, mirroring poke-env's handler
        (abstract_battle.py:799-806): if parts[5] is an '[of] SOURCE' tag (Reflect Type) copy that
        source's CURRENT effective types; else the slash-joined type string in parts[4] (e.g. 'Water'
        or 'Normal/Water'). Returns None on a truncated/unresolvable line (fail-safe: no drift)."""
        if len(parts) > 5 and parts[5].startswith("[of] "):
            src = self.active_slots.get(self._slot_key_from_ident(parts[5][5:]))
            return (self._slot_effective_types(src) or None) if src is not None else None
        if len(parts) > 4 and parts[4]:
            return [t for t in parts[4].split("/") if t] or None
        return None

    def _slot_effective_types(self, mon) -> list:
        """A slot's CURRENT effective types (for the Reflect-Type [of] copy), matching the encoder's
        _effective_types precedence: tera > runtime typechange > transform dex > species dex. Uses the
        SHARED pokedex (the same source as the encoder's _dex_types) so the copied types stay parity-clean."""
        if mon is None:
            return []
        if mon.is_terastallized and mon.known_tera_type:
            return [mon.known_tera_type]
        if mon.runtime_types:
            return list(mon.runtime_types)
        dex = get_pokedex()
        if not dex:
            return []
        species = mon.transformed_into if (mon.is_transformed and mon.transformed_into) else mon.species
        e = dex.entry(species) or (dex.entry(mon.base_species) if mon.base_species else None)
        return list(e.get("types") or []) if e else []

    def _handle_singleturn(self, parts: list[str], cmd: str) -> None:
        """v11 A.3: |-singleturn| / |-singlemove| audit hook. These mark THIS-turn-only effects
        (Protect/Wide Guard/Helping Hand/Follow Me/redirection/flinch/…) that Showdown clears by the
        next turn-start snapshot, so they are NOT decision-state and must NOT be encoded. We record a
        transient audit entry only — it must NOT mutate mon.volatiles (which persist to switch and would
        drift vs poke-env's end-of-turn clearing) and must NOT drive the protect_counter (that is sourced
        from the |move| line in _handle_move). The persistent Protect signal is protect_counter."""
        if len(parts) < 4:
            return
        slot_key = self._slot_key_from_ident(parts[2])
        name = parts[3].split(":", 1)[-1].strip() if parts[3] else ""
        self._current_turn_actions.append({
            "event": cmd.lstrip("-"),     # "singleturn" / "singlemove"
            "slot": slot_key,
            "effect": name,
        })

    def _handle_sidestart(self, parts: list[str]) -> None:
        # |-sidestart|p2: speedyturtle87|move: Tailwind
        raw_side = parts[2]
        effect = parts[3] if len(parts) > 3 else ""
        pid = raw_side.split(":")[0].strip()

        sc = self.side_conditions[pid]
        if "Tailwind" in effect:
            sc.tailwind = 4
        elif "Reflect" in effect:
            sc.screens["reflect"] = 0          # turns ACTIVE (elapsed)
        elif "Light Screen" in effect:
            sc.screens["light_screen"] = 0
        elif "Aurora Veil" in effect:
            sc.screens["aurora_veil"] = 0
        # v11 C.1: entry hazards. NOTE "Spikes" is a substring of "Toxic Spikes" → match Toxic FIRST.
        # Spikes/Toxic Spikes STACK (one |-sidestart| per layer, parity w/ poke-env's +1); cap 3 / 2.
        elif "Stealth Rock" in effect:
            sc.stealth_rock = True
        elif "Toxic Spikes" in effect:
            sc.toxic_spikes = min(2, sc.toxic_spikes + 1)
        elif "Spikes" in effect:
            sc.spikes = min(3, sc.spikes + 1)
        elif "Sticky Web" in effect:
            sc.sticky_web = True
        # v11 C.2b gap-fix: persistent team-protection side conditions.
        elif "Safeguard" in effect:
            sc.safeguard = True
        elif "Mist" in effect:
            sc.mist = True
        elif "Lucky Chant" in effect:
            sc.lucky_chant = True

    def _handle_sideend(self, parts: list[str]) -> None:
        raw_side = parts[2]
        effect = parts[3] if len(parts) > 3 else ""
        pid = raw_side.split(":")[0].strip()

        sc = self.side_conditions[pid]
        if "Tailwind" in effect:
            sc.tailwind = 0
        elif "Reflect" in effect:
            sc.screens.pop("reflect", None)
        elif "Light Screen" in effect:
            sc.screens.pop("light_screen", None)
        elif "Aurora Veil" in effect:
            sc.screens.pop("aurora_veil", None)
        # v11 C.1: hazard removal (Defog / Rapid Spin / Court Change / Tidy Up) clears ALL layers at once.
        elif "Stealth Rock" in effect:
            sc.stealth_rock = False
        elif "Toxic Spikes" in effect:          # match Toxic before Spikes (substring)
            sc.toxic_spikes = 0
        elif "Spikes" in effect:
            sc.spikes = 0
        elif "Sticky Web" in effect:
            sc.sticky_web = False
        elif "Safeguard" in effect:
            sc.safeguard = False
        elif "Mist" in effect:
            sc.mist = False
        elif "Lucky Chant" in effect:
            sc.lucky_chant = False

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
            # |-enditem| means the item was consumed/removed (berry eaten, Sash
            # popped, Herb used, Knock Off): the mon no longer HOLDS it, so the
            # encoder must stop treating it as held (gap #5 parity with poke-env,
            # which nulls mon.item on consumption).  A plain |-item| reveal (Frisk
            # /Trick) leaves the item in hand, so it does NOT set this.
            if consumed:
                mon.item_consumed = True
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
            self.active_slots[slot_key].runtime_types = None   # v11 P2: tera clears poke-env _temporary_types

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

    def _handle_activate(self, parts: list[str]) -> None:
        """Record a Symbiosis item hand-off (the only -activate that moves an item).

            |-activate|p2a: Florges|ability: Symbiosis|Leftovers|[of] p2b: Flapple

        The transfer is otherwise SILENT (no |-item|/|-enditem|), so without this the GIVER keeps a
        stale known_item and the RECEIVER's new item stays invisible. Mark the giver itemless (it
        passed its only item) and set the receiver's known held item.
        """
        if len(parts) < 5 or parts[3] != "ability: Symbiosis":
            return
        item = parts[4].strip()
        of_ident = next((p[len("[of]"):].strip() for p in parts[5:] if p.startswith("[of]")), None)
        giver = self._slot_key_from_ident(parts[2])
        receiver = self._slot_key_from_ident(of_ident) if of_ident else ""
        if giver in self.active_slots:
            self.active_slots[giver].item_consumed = True       # gave its item away → itemless
        if item and receiver in self.active_slots:
            self.active_slots[receiver].known_item = item
            self.active_slots[receiver].item_consumed = False   # now holds the passed item

    def _learn_item_from_tags(self, parts: list[str]) -> None:
        """Learn a HELD item revealed by its OWN effect via a ``[from] item:`` tag.

        Many items announce themselves when they trigger, with no |-item| line::

            |-damage|p1a: X|90/100|[from] item: Life Orb               (recoil — X holds it)
            |-heal|p2a: Y|100/100|[from] item: Leftovers              (heal — Y holds it)
            |-damage|p1a: Atk|80/100|[from] item: Rocky Helmet|[of] p2b: Def   (Def holds it)
            |-status|p1a: Z|brn|[from] item: Flame Orb

        The holder is the ``[of]`` mon ONLY on ``-damage``/``-status`` lines (reactive
        contact-punish items — Rocky Helmet / Jaboca / Rowap — hurt a DIFFERENT mon and
        emit the HOLDER as ``[of]``). On a ``-heal`` line the item heals its OWN holder
        (the line subject) and Showdown emits the damaged FOE as ``[of]`` (e.g. Shell Bell:
        ``|-heal|<HOLDER>|hp|[from] item: Shell Bell|[of] <VICTIM>``), so ``[of]`` there is
        the victim, NOT the holder — trusting it would label the opponent with the item.
        Sets ``known_item`` (the held identity) but does NOT mark it consumed —
        |-enditem| owns consumption, and a Sitrus/Sash also emits its own |-enditem|.
        """
        item = None
        of_ident = None
        for p in parts[2:]:
            if p.startswith("[from] item:"):
                item = p.split(":", 1)[1].strip()
            elif p.startswith("[of]"):
                of_ident = p[len("[of]"):].strip()
        if not item:
            return
        cmd = parts[1] if len(parts) > 1 else ""
        use_of = cmd in ("-damage", "-status")     # [of] is the holder only for damage/status reactors
        holder = of_ident if (use_of and of_ident) else (parts[2] if len(parts) > 2 else "")
        slot_key = self._slot_key_from_ident(holder)
        if not re.fullmatch(r"p[12][ab]", slot_key):
            return
        if slot_key in self.active_slots:
            self.active_slots[slot_key].known_item = item
            # Sticky Barb transfers SILENTLY on contact (no |-item|/|-enditem|): when it surfaces on a
            # NEW holder via its residual self-damage, the PRIOR holder lost it. Clear the stale label so
            # the former holder isn't encoded as DEFINITELY holding it. Scoped to transfer-prone items
            # only — a common item like Leftovers can legitimately be on two mons at once.
            if item in _TRANSFER_ONLY_ITEMS:
                for k, m in self.active_slots.items():
                    if k != slot_key and m.known_item == item:
                        m.known_item = None

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

    @staticmethod
    def _observed_opp(d: dict) -> dict:
        """Apply an active Illusion to an OPPONENT-side mon dict (Zoroark
        Solution B / fooled view): the observer sees the disguise species — its
        typing/stats via the dex — carrying the moves it was seen using (the
        tells), until a hit breaks the disguise (|replace| → illusion_active
        False).  The owner's own side keeps the true identity.  Mutates the
        already-deepcopied dict."""
        if d.get("illusion_active") and d.get("disguise_species"):
            d["species"] = d["disguise_species"]
            d["base_species"] = d["disguise_species"]
            d["appears_disguised"] = True
        return d

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
                opp_active[rel_key] = self._observed_opp(d)

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
        #
        # Two distinct exclusion sets keep this consistent with the LIVE bot:
        #   • opp_active_species (TRUE identities) keeps the genuinely-active
        #     mons out of opp_seen.
        #   • opp_bench_skip (PERCEIVED identities) decides which roster slots
        #     the observer treats as already on the field.  A disguised opponent
        #     Zoroark is perceived as its DISGUISE species, so THAT roster slot
        #     is consumed by the illusion while the TRUE Zoroark surfaces as an
        #     unseen roster stub — exactly what a live bot sees (it cannot
        #     un-see the disguise until |replace|).  With no active illusion the
        #     two sets are identical, so non-illusion battles are unaffected.
        #     This runs for the opp side only; the owner's own side (our_bench,
        #     this perspective) always tracks the truth.
        #
        # The perceived skip key is normalised to the roster's base name
        # (_perceived_roster_name): the roster lists base formes, so a Zoroark
        # disguised as a MEGA-evolved teammate (perceived "Charizard-Mega-Y")
        # must consume the base "Charizard" slot.  Without this the disguise's
        # roster slot is missed AND a genuinely-on-bench same-species teammate is
        # also listed → the species shows up BOTH active and benched (a phantom
        # double-count; gap-#6 re-disguise residual).  Mirrors the non-illusion
        # branch, which already keys on the mon's base species.
        opp_active_species = {_base(m) for m in self.active_slots.values()
                              if m.player == opp and not m.is_fainted}
        opp_bench_skip: set[str] = set()
        for m in self.active_slots.values():
            if m.player != opp or m.is_fainted:
                continue
            if m.illusion_active and m.disguise_species:
                opp_bench_skip.add(_perceived_roster_name(m.disguise_species))
            else:
                opp_bench_skip.add(_base(m))

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
            if species in opp_bench_skip:
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
                    "is_transformed": False,
                    "transformed_into": None,
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
                # #00: skip a forced post-faint replacement — it is recorded separately as a
                # decision_type="replacement" transition; emitting it here as a turn-start switch teaches
                # a FALSE label (the real turn-start command was a move that never executed, or nothing).
                if ev["event"] == "switch" and ev.get("forced_replacement"):
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

            # ── Gimmick (mega-evolution) label join ──────────────────────────
            # A mega is a CHECKBOX on the chosen move, not a competing action, so
            # we stamp it onto the SAME-turn, SAME-slot move action.  Join by SLOT,
            # not execution_index: mega_evolution events carry no execution_index,
            # and a slot megas at most once per game, so the slot key is an
            # unambiguous join key.  forme_change events (Palafin/Terapagos auto-
            # formes) are involuntary, not a player decision, and are deliberately
            # NOT joined.  The flag is added ONLY when a mega actually occurred, so
            # non-mega move actions serialise byte-identically to before.
            megaed_slots = {
                ev.get("slot")
                for ev in self._current_turn_actions
                if ev.get("event") == "mega_evolution"
                and (ev.get("slot") or "").startswith(player)
            }
            if megaed_slots:
                for act in out:
                    if act["action"] == "move" and act["slot"] in megaed_slots:
                        act["mega"] = True
            # v11 Phase D: tera label-join, EXACT mirror of the mega join. _handle_tera appends
            # {"event": "terastallize", "slot": ...} identically to mega_evolution. ⚠ INERT today (tera is
            # mod-disabled → teraed_slots is always empty), so non-tera move actions serialise byte-identically;
            # the path is ready if the Champions mod ever flips canTerastallize.
            teraed_slots = {
                ev.get("slot")
                for ev in self._current_turn_actions
                if ev.get("event") == "terastallize"
                and (ev.get("slot") or "").startswith(player)
            }
            if teraed_slots:
                for act in out:
                    if act["action"] == "move" and act["slot"] in teraed_slots:
                        act["tera"] = True
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
            # Post-faint replacement decisions made this turn (each carries its
            # own both-perspective decision state).  transitions.py emits one
            # extra transition per replacement, from the replacing player's view.
            "replacements": list(self._current_turn_replacements),
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
                # Transform/Imposter: True if this mon transformed at any point
                # this match (latched — survives the switch-out revert).  Its
                # revealed_moves are deliberately empty (copied moves filtered
                # out) so belief fill falls back to its own usage distribution.
                "is_transformed": mon.ever_transformed,
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
            # Match-level edge-case flags (Zoroark Illusion / Ditto-Imposter
            # Transform).  The parser handles both correctly; these surface that
            # the replay exercised the handling, for auditing/filtering.
            "contains_illusion": self._illusion_seen,
            "contains_transform": self._transform_seen,
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
        # HP is "num/den", optionally followed by a status ("50/100 brn") or, in
        # some replays, a stray suffix with no separator ("50/100y").  Extract the
        # LEADING digits of each side so malformed tokens still parse the way
        # poke-env's client does (gap #5: an unparsed token previously left an
        # active mon with hp_pct=None / encoded 0 while the live path saw 0.5).
        hp_str = hp_str.split(" ")[0]  # strip " fnt" / " brn" etc.
        m = re.match(r"\s*(\d+)\s*/\s*(\d+)", hp_str)
        if m:
            return float(m.group(1)), float(m.group(2))
        m = re.match(r"\s*(\d+)", hp_str)   # bare number (rare) → assume /100
        if m:
            return float(m.group(1)), 100.0
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
