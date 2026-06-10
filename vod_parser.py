"""
vod_parser.py
=============
Converts Pokémon Showdown HTML replay logs into replay buffer JSONL
entries compatible with Victory-Dance's existing format.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT  (one JSON object per line — same as self-play buffer)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{
  "battle_id":  "gen9championsvgc2026regma-XXXX",
  "source":     "vod",
  "player":     "p1",        ← which side is being encoded
  "turn":       3,
  "state":      [0.0, ...],  ← 882 floats from StateEncoder
  "action_s0":  4,           ← action for active slot A (0-15, -1 = unknown)
  "action_s1":  12,          ← action for active slot B
  "outcome":    1            ← +1 win / -1 loss / 0 draw
}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMPERFECT-INFORMATION POLICY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For the perspective player ("p1" by default):
  Own mons:  full info (moves revealed as used, HP from log)
  Opp mons:  belief state until revealed, then actual revealed info

This mirrors exactly what the network sees during live play, making
VOD training directly compatible with self-play training.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KNOWN TEAM OVERRIDES  (--known-teams)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For replays where you know the exact sets (e.g. your own VODs),
pass a JSON file structured as:

  {
    "battle-gen9championsvgc2026regma-1234": {
      "p1": {
        "Kingambit": {
          "nature": "Adamant",
          "evs": [252, 252, 0, 0, 4, 0],   ← actual EVs 0-252
          "ivs": [31, 31, 31, 31, 31, 31],  ← optional, defaults to 31s
          "item": "Chople Berry",
          "ability": "Defiant",
          "moves": ["Sucker Punch", "Kowtow Cleave", "Protect", "Low Kick"]
        },
        ...
      }
    }
  }

When a known-team entry exists for a battle, it completely overrides
the belief-state for that player's mons — giving perfect information
for your own replays.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  # Single replay, both perspectives
  python vod_parser.py replay.html

  # Directory of replays
  python vod_parser.py replays/ --out replay_buffer/vods.jsonl

  # With exact team knowledge for your own replays
  python vod_parser.py replays/ --known-teams known_teams.json

  # Only your side (p1), with known teams
  python vod_parser.py replays/ --player p1 --known-teams known_teams.json
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# ── Project imports ────────────────────────────────────────────────────────────
try:
    from state_encoder import StateEncoder, STATE_DIM, ACTIONS_PER_SLOT
    from belief_state import BeliefState
except ImportError:
    raise ImportError(
        "vod_parser.py must be run from the Victory-Dance project root "
        "alongside state_encoder.py and belief_state.py."
    )

from poke_env.battle import PokemonType, Status, MoveCategory


# ══════════════════════════════════════════════════════════════════════════════
# FakeMove — minimal duck-type for StateEncoder._write_move()
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class FakeMove:
    name: str
    base_power: int = 80
    type: PokemonType = PokemonType.NORMAL
    category: MoveCategory = MoveCategory.PHYSICAL
    priority: int = 0
    accuracy: float = 1.0
    max_pp: int = 10
    current_pp: int = 10
    is_protect_move: bool = False

    def __post_init__(self):
        self.is_protect_move = "protect" in self.name.lower()


_MOVE_CACHE: dict[str, FakeMove] = {}


def _lookup_move(move_name: str) -> FakeMove:
    if move_name in _MOVE_CACHE:
        return _MOVE_CACHE[move_name]
    try:
        from poke_env.data import GenData
        gen_data = GenData.from_gen(9)
        move_id = move_name.lower().replace(" ", "").replace("-", "").replace("'", "")
        if move_id in gen_data.moves:
            m = gen_data.moves[move_id]
            fake = FakeMove(
                name=move_name,
                base_power=getattr(m, "base_power", 0) or 0,
                type=getattr(m, "type", PokemonType.NORMAL),
                category=getattr(m, "category", MoveCategory.PHYSICAL),
                priority=getattr(m, "priority", 0),
                accuracy=getattr(m, "accuracy", 1.0),
                max_pp=getattr(m, "pp", 10),
                current_pp=getattr(m, "pp", 10),
            )
        else:
            fake = FakeMove(name=move_name)
    except Exception:
        fake = FakeMove(name=move_name)
    _MOVE_CACHE[move_name] = fake
    return fake


# ══════════════════════════════════════════════════════════════════════════════
# FakePokemon — duck-type for StateEncoder._write_pokemon()
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class FakePokemon:
    species: str
    level: int = 50
    current_hp_fraction: float = 1.0
    fainted: bool = False
    type_1: PokemonType = PokemonType.NORMAL
    type_2: Optional[PokemonType] = None
    base_stats: dict = field(default_factory=lambda: {
        "hp": 80, "atk": 80, "def": 80, "spa": 80, "spd": 80, "spe": 80
    })
    moves: dict = field(default_factory=dict)  # name → FakeMove
    status: Optional[Status] = None
    boosts: dict = field(default_factory=lambda: {
        "atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0,
        "accuracy": 0, "evasion": 0,
    })
    is_terastallized: bool = False
    revealed: bool = False

    @property
    def types(self) -> frozenset:
        t = {self.type_1}
        if self.type_2:
            t.add(self.type_2)
        return frozenset(t)

    def reveal_move(self, move_name: str) -> None:
        if move_name not in self.moves:
            m = _lookup_move(move_name)
            m.current_pp = max(0, m.current_pp - 1)
            self.moves[move_name] = m

    def apply_belief(self, belief: BeliefState) -> None:
        """Fill unknown move slots from meta usage data."""
        known = set(self.moves.keys())
        for move_name, _pct in belief.top_moves(self.species, n=4):
            if len(self.moves) >= 4:
                break
            if move_name not in known:
                self.moves[move_name] = _lookup_move(move_name)

    def apply_known_set(self, known: dict) -> None:
        """
        Override with exact set data (for replays you produced).
        known dict keys: nature, evs, ivs, item, ability, moves
        """
        if "moves" in known:
            self.moves = {
                m: _lookup_move(m) for m in known["moves"]
            }
        # EVs/nature/item/ability are stored on the mon for reference
        # but don't map directly to FakePokemon fields used by the encoder
        # (the encoder reads base_stats + hp_fraction, not actual EVs).
        # They're preserved here for future use (damage calc, etc.).
        self._known_nature  = known.get("nature", "Hardy")
        self._known_evs     = known.get("evs", [0]*6)
        self._known_ivs     = known.get("ivs", [31]*6)
        self._known_item    = known.get("item")
        self._known_ability = known.get("ability")


def _load_base_data(species: str) -> dict:
    try:
        from poke_env.data import GenData
        gen_data = GenData.from_gen(9)
        sid = species.lower().replace("-", "").replace(" ", "")
        if sid in gen_data.pokedex:
            p = gen_data.pokedex[sid]
            return {
                "base_stats": p.get("baseStats", {}),
                "type_1": p.get("types", ["Normal"])[0],
                "type_2": p.get("types", [None, None])[1]
                          if len(p.get("types", [])) > 1 else None,
            }
    except Exception:
        pass
    return {}


def _parse_type(t_str: Optional[str]) -> Optional[PokemonType]:
    if not t_str:
        return None
    try:
        return PokemonType[t_str.upper()]
    except KeyError:
        return PokemonType.NORMAL


def _make_fake_pokemon(species: str, belief: BeliefState) -> FakePokemon:
    base = _load_base_data(species)
    mon = FakePokemon(
        species=species,
        type_1=_parse_type(base.get("type_1", "Normal")),
        type_2=_parse_type(base.get("type_2")),
        base_stats=base.get("base_stats", {
            "hp": 80, "atk": 80, "def": 80, "spa": 80, "spd": 80, "spe": 80,
        }),
    )
    mon.apply_belief(belief)
    return mon


# ══════════════════════════════════════════════════════════════════════════════
# ParsedBattle — game state tracked as we parse turns
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class ParsedBattle:
    battle_id: str
    p1_name: str = ""
    p2_name: str = ""
    roster: dict = field(default_factory=lambda: {"p1": {}, "p2": {}})
    active: dict = field(default_factory=lambda: {
        "p1a": None, "p1b": None,
        "p2a": None, "p2b": None,
    })
    turn: int = 0
    winner: Optional[str] = None
    fields: dict = field(default_factory=dict)
    p1_side: dict = field(default_factory=dict)
    p2_side: dict = field(default_factory=dict)

    def get_active_mon(self, slot: str) -> Optional[FakePokemon]:
        species = self.active.get(slot)
        if not species:
            return None
        side = slot[:2]
        return self.roster[side].get(species)

    def own_active(self, player: str) -> list[Optional[FakePokemon]]:
        return [self.get_active_mon(f"{player}a"), self.get_active_mon(f"{player}b")]

    def bench(self, player: str) -> list[FakePokemon]:
        active_species = {
            self.active.get(f"{player}a"),
            self.active.get(f"{player}b"),
        }
        return [
            mon for sp, mon in self.roster[player].items()
            if sp not in active_species and not mon.fainted
        ]


# ══════════════════════════════════════════════════════════════════════════════
# FakeDoubleBattle — duck-type for StateEncoder.encode()
# ══════════════════════════════════════════════════════════════════════════════
class FakeDoubleBattle:
    def __init__(self, battle: ParsedBattle, player: str, belief: BeliefState):
        self._battle = battle
        self._player = player
        self._opp    = "p2" if player == "p1" else "p1"
        self._belief = belief

    @property
    def active_pokemon(self):
        return self._battle.own_active(self._player)

    @property
    def opponent_active_pokemon(self):
        mons = self._battle.own_active(self._opp)
        for mon in mons:
            if mon is not None:
                mon.apply_belief(self._belief)
        return mons

    @property
    def team(self):
        return self._battle.roster[self._player]

    @property
    def weather(self):
        return self._battle.fields.get("weather", {})

    @property
    def fields(self):
        return self._battle.fields.get("fields", {})

    @property
    def side_conditions(self):
        return getattr(self._battle, f"{self._player}_side", {})

    @property
    def opponent_side_conditions(self):
        return getattr(self._battle, f"{self._opp}_side", {})

    @property
    def turn(self):
        return self._battle.turn


# ══════════════════════════════════════════════════════════════════════════════
# Action inference
# ══════════════════════════════════════════════════════════════════════════════
def _infer_action(
    turn_lines: list[str],
    player: str,
    slot: str,           # "a" or "b"
    battle: ParsedBattle,
) -> int:
    """
    Infer action integer (0-15) from this turn's log lines for one slot.
    Returns -1 if action cannot be determined.

    Encoding (matches state_encoder.py):
      0-11  → move_idx (0-3) × target_idx (0=opp_a, 1=opp_b, 2=ally)
      12-15 → switch to bench slot 0-3
    """
    player_slot = f"{player}{slot}"
    opp = "p2" if player == "p1" else "p1"

    species = battle.active.get(player_slot)
    if not species:
        return -1
    mon = battle.roster[player].get(species)
    if not mon or mon.fainted:
        return -1

    # Check switch
    switch_re = re.compile(rf"\|switch\|{player_slot}: ([^|]+)\|")
    for line in turn_lines:
        m = switch_re.match(line)
        if m:
            new_species = _normalize_species(m.group(1))
            bench = battle.bench(player)
            for i, bmon in enumerate(bench[:4]):
                if bmon.species == new_species:
                    return 12 + i
            return 12  # fallback: bench slot 0

    # Check move
    move_re = re.compile(rf"\|move\|{player_slot}: [^|]+\|([^|]+)\|([^|]*)")
    for line in turn_lines:
        m = move_re.match(line)
        if m:
            move_name  = m.group(1).strip()
            target_str = m.group(2).strip()

            move_names = list(mon.moves.keys())
            move_idx   = move_names.index(move_name) if move_name in move_names else 0

            # Target
            target_idx = 0
            if f"{opp}a" in target_str:
                target_idx = 0
            elif f"{opp}b" in target_str:
                target_idx = 1
            elif player in target_str:
                target_idx = 2

            return move_idx * 3 + target_idx

    return -1


# ══════════════════════════════════════════════════════════════════════════════
# HTML replay parser
# ══════════════════════════════════════════════════════════════════════════════
def _normalize_species(raw: str) -> str:
    return raw.split(",")[0].strip()


def _extract_log_lines(html: str) -> Optional[list[str]]:
    """Pull the raw battle log out of the Showdown HTML replay."""
    m = re.search(
        r'<script[^>]+class="battle-log-data"[^>]*>(.*?)</script>',
        html, re.DOTALL,
    )
    if not m:
        return None
    return m.group(1).split("\n")


def parse_html_replay(
    html_path: Path,
    belief: BeliefState,
    known_teams: Optional[dict] = None,  # battle_id → {p1: {...}, p2: {...}}
) -> Optional[tuple["ParsedBattle", list[str]]]:
    html  = html_path.read_text(encoding="utf-8")
    lines = _extract_log_lines(html)
    if lines is None:
        return None

    # Battle ID
    bid_m = re.search(r'name="replayid"\s+value="([^"]+)"', html)
    battle_id = bid_m.group(1) if bid_m else html_path.stem

    battle = ParsedBattle(battle_id=battle_id)

    # ── Pass 1: roster + winner ───────────────────────────────────────────────
    for line in lines:
        if line.startswith("|player|p1|"):
            battle.p1_name = line.split("|")[3]
        elif line.startswith("|player|p2|"):
            battle.p2_name = line.split("|")[3]
        elif line.startswith("|poke|"):
            parts   = line.split("|")
            side    = parts[2]
            species = _normalize_species(parts[3])
            mon     = _make_fake_pokemon(species, belief)
            battle.roster[side][species] = mon
        elif line.startswith("|win|"):
            winner_name = line[5:].strip()
            if winner_name == battle.p1_name:
                battle.winner = "p1"
            elif winner_name == battle.p2_name:
                battle.winner = "p2"

    # ── Apply known-team overrides ────────────────────────────────────────────
    if known_teams and battle_id in known_teams:
        for side, mon_overrides in known_teams[battle_id].items():
            for species, set_data in mon_overrides.items():
                # Match by species (case-insensitive)
                for roster_species, mon in battle.roster[side].items():
                    if roster_species.lower() == species.lower():
                        mon.apply_known_set(set_data)
                        break

    return battle, lines


# ══════════════════════════════════════════════════════════════════════════════
# Main transition extractor
# ══════════════════════════════════════════════════════════════════════════════
def replay_to_transitions(
    html_path: Path,
    belief: BeliefState,
    encoder: StateEncoder,
    players: list[str] = ("p1", "p2"),
    known_teams: Optional[dict] = None,
) -> list[dict]:
    parsed = parse_html_replay(html_path, belief, known_teams)
    if parsed is None:
        print(f"  [skip] {html_path.name}: not a valid replay.")
        return []

    battle, lines = parsed
    outcome_map = (
        {"p1": 1,  "p2": -1} if battle.winner == "p1" else
        {"p1": -1, "p2": 1}  if battle.winner == "p2" else
        {"p1": 0,  "p2": 0}
    )

    transitions: list[dict] = []
    current_turn_lines: list[str] = []
    in_turn = False

    def _flush_turn(turn_num: int) -> None:
        for player in players:
            fake = FakeDoubleBattle(battle, player, belief)
            try:
                state_vec = encoder.encode(fake)
            except Exception:
                return

            a0 = _infer_action(current_turn_lines, player, "a", battle)
            a1 = _infer_action(current_turn_lines, player, "b", battle)

            if a0 < 0 and a1 < 0:
                return

            transitions.append({
                "battle_id": battle.battle_id,
                "source":    "vod",
                "player":    player,
                "turn":      turn_num,
                "state":     state_vec.tolist(),
                "action_s0": max(a0, 0),
                "action_s1": max(a1, 0),
                "outcome":   outcome_map.get(player, 0),
            })

    for line in lines:
        # ── Turn boundary ─────────────────────────────────────────────────────
        if line.startswith("|turn|"):
            if in_turn and battle.turn > 0:
                _flush_turn(battle.turn)
            battle.turn = int(line.split("|")[2])
            current_turn_lines = []
            in_turn = True
            continue

        if not in_turn:
            continue
        current_turn_lines.append(line)

        # ── State mutations ───────────────────────────────────────────────────

        # Switch in
        m = re.match(r"\|switch\|(\w+): ([^|]+)\|[^|]+\|(\d+)\/(\d+)", line)
        if m:
            slot_key, species_raw, cur_hp, max_hp = m.groups()
            species  = _normalize_species(species_raw)
            side     = slot_key[:2]
            if species not in battle.roster[side]:
                mon = _make_fake_pokemon(species, belief)
                battle.roster[side][species] = mon
            battle.active[slot_key] = species
            mon = battle.roster[side][species]
            mon.revealed = True
            mon.current_hp_fraction = int(cur_hp) / int(max_hp)
            continue

        # Damage
        m = re.match(r"\|-damage\|(\w+): [^|]+\|(\d+)\/(\d+)", line)
        if m:
            slot_key, cur_hp, max_hp = m.groups()
            mon = battle.get_active_mon(slot_key)
            if mon:
                mon.current_hp_fraction = int(cur_hp) / int(max_hp)
            continue

        # Faint
        m = re.match(r"\|faint\|(\w+):", line)
        if m:
            slot_key = m.group(1)
            mon = battle.get_active_mon(slot_key)
            if mon:
                mon.fainted = True
                mon.current_hp_fraction = 0.0
            battle.active[slot_key] = None
            continue

        # Move (reveals the move)
        m = re.match(r"\|move\|(\w+): [^|]+\|([^|]+)", line)
        if m:
            slot_key, move_name = m.group(1), m.group(2).strip()
            mon = battle.get_active_mon(slot_key)
            if mon:
                mon.reveal_move(move_name)
                mon.revealed = True
            continue

        # Heal
        m = re.match(r"\|-heal\|(\w+): [^|]+\|(\d+)\/(\d+)", line)
        if m:
            slot_key, cur_hp, max_hp = m.groups()
            mon = battle.get_active_mon(slot_key)
            if mon:
                mon.current_hp_fraction = int(cur_hp) / int(max_hp)
            continue

        # Status
        m = re.match(r"\|-status\|(\w+): [^|]+\|(\w+)", line)
        if m:
            slot_key, status_str = m.group(1), m.group(2)
            mon = battle.get_active_mon(slot_key)
            if mon:
                try:
                    mon.status = Status[status_str.upper()]
                except KeyError:
                    pass
            continue

        # Terastallize
        m = re.match(r"\|-terastallize\|(\w+):", line)
        if m:
            mon = battle.get_active_mon(m.group(1))
            if mon:
                mon.is_terastallized = True
            continue

        # Boost / Unboost
        m = re.match(r"\|-(boost|unboost)\|(\w+): [^|]+\|(\w+)\|(\d+)", line)
        if m:
            direction, slot_key, stat, amount = m.groups()
            mon = battle.get_active_mon(slot_key)
            if mon and stat in mon.boosts:
                delta = int(amount) * (1 if direction == "boost" else -1)
                mon.boosts[stat] = max(-6, min(6, mon.boosts[stat] + delta))
            continue

        # Detailschange (e.g. Floette → Floette-Mega)
        m = re.match(r"\|detailschange\|(\w+): [^|]+\|(.+)", line)
        if m:
            slot_key, new_raw = m.group(1), m.group(2)
            new_species = _normalize_species(new_raw)
            side        = slot_key[:2]
            old_species = battle.active.get(slot_key)
            if old_species and old_species in battle.roster[side]:
                old_mon = battle.roster[side].pop(old_species)
                old_mon.species = new_species
                battle.roster[side][new_species] = old_mon
                battle.active[slot_key] = new_species
            continue

    # Flush last turn
    if in_turn and battle.turn > 0:
        _flush_turn(battle.turn)

    return transitions


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse Showdown replay VODs into replay buffer JSONL.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        help="Path to a single .html replay OR a directory containing .html files.",
    )
    parser.add_argument(
        "--out", default="replay_buffer/vods.jsonl",
        help="Output JSONL file (appended to, not overwritten). Default: replay_buffer/vods.jsonl",
    )
    parser.add_argument(
        "--player", choices=["p1", "p2", "both"], default="both",
        help="Which player perspective(s) to encode. Default: both",
    )
    parser.add_argument(
        "--belief", default="pikalytics_regma.json",
        help="Path to pikalytics_regma.json belief-state database.",
    )
    parser.add_argument(
        "--known-teams", default=None, dest="known_teams",
        help=(
            "Path to a JSON file with exact set data for replays you produced. "
            "See module docstring for the expected format."
        ),
    )
    args = parser.parse_args()

    # ── Load supporting data ──────────────────────────────────────────────────
    belief  = BeliefState(args.belief)
    encoder = StateEncoder()

    known_teams: Optional[dict] = None
    if args.known_teams:
        with open(args.known_teams, encoding="utf-8") as f:
            known_teams = json.load(f)
        print(f"[known-teams] Loaded overrides for {len(known_teams)} battle(s).")

    players = ["p1", "p2"] if args.player == "both" else [args.player]

    # ── Collect replay files ──────────────────────────────────────────────────
    input_path = Path(args.input)
    replay_files = (
        sorted(input_path.glob("*.html"))
        if input_path.is_dir()
        else [input_path]
    )
    print(f"[parse] {len(replay_files)} replay file(s) found.")

    # ── Parse and write ───────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with out_path.open("a", encoding="utf-8") as out_f:
        for replay_file in replay_files:
            print(f"  {replay_file.name}")
            transitions = replay_to_transitions(
                replay_file, belief, encoder, players, known_teams,
            )
            for t in transitions:
                out_f.write(json.dumps(t) + "\n")
            total += len(transitions)
            print(f"    → {len(transitions)} transitions written")

    print(f"\n[done] {total} total transitions → {out_path}")


if __name__ == "__main__":
    main()
