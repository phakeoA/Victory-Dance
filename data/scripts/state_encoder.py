"""
State encoder for VGC Reg M-A (Pokémon Champions doubles format).

Encodes a battle state into a fixed-size float32 numpy array for the
AlphaZero neural network.  Two input paths produce the SAME tensor layout:

  1. LIVE   — encode(battle)            poke-env DoubleBattle (bot play, Type C/D)
  2. OFFLINE — encode_snapshot(snap)    a state_before/after_actions snapshot
               encode_transition(t)     from vod_parser / belief_state JSON
                                        (Types A/B + training ingestion)

poke-env is OPTIONAL: the offline path works without it (training machines),
the live path raises a clear error if it is missing.  All enum orderings are
FROZEN in this file (not derived from poke-env at import time) so STATE_DIM
can never silently change with a poke-env upgrade and invalidate a trained net.

BELIEF INTEGRATION (belief_state.py)
────────────────────────────────────
fill_blanks() writes per-mon ``stats_estimate`` ({"mode": "exact"|"distribution",
"stats": {...}}) and ``belief`` blocks into the parsed JSON.  The encoder
consumes them:
  * est_stats (6 floats)  — exact L50 stats (own side / self-play) or the
    Pikalytics probability-weighted expected stats (opponent side)
  * stats_known (1 float) — 1.0 exact, 0.5 distribution estimate, 0.0 unknown
  * unknown move slots are padded with belief moves_predicted, with the
    is_known feature carrying the usage probability instead of 1.0
Live path: own mons use poke-env's real stats; opponent mons use the
BeliefState passed to the constructor (StateEncoder(belief=...)).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TENSOR LAYOUT  (total STATE_DIM floats)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [A] 4 active slots × POKEMON_FEATURES
        slot 0 → own active mon 0 (our_a)
        slot 1 → own active mon 1 (our_b)
        slot 2 → opp active mon 0 (opp_a)
        slot 3 → opp active mon 1 (opp_b)

  [B] 4 bench slots × POKEMON_FEATURES
        own bench mons (up to 4, fainted excluded; zeros if fewer)
        TODO(layout-v2): consider adding opp bench + fainted flags so the
        net can count remaining team members / legal switch-ins.

  [C] GLOBAL_FEATURES
        weather multi-hot (9)
        terrain/field multi-hot (15)
        own side conditions multi-hot (24)
        opp side conditions multi-hot (24)
        turn normalised (1)
        trick room flag (1)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
POKEMON_FEATURES per slot (108 floats):
    hp_frac       (1)
    type1 one-hot (20)
    type2 one-hot (20)   ← zeros if single-type
    base_stats    (6)    ← hp atk def spa spd spe / 255
    est_stats     (6)    ← in-battle L50 stats / 300 (exact or belief-expected)
    stats_known   (1)    ← 1.0 exact · 0.5 distribution · 0.0 unknown
    is_mega       (1)
    is_tera       (1)
    status one-hot(7)    ← BRN FNT FRZ PAR PSN SLP TOX
    boosts        (7)    ← atk def spa spd spe acc eva / 6
    4 × MOVE_FEATURES    ← 4 × 9 = 36
    is_active     (1)
    is_revealed   (1)    ← always 1 for own mons
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MOVE_FEATURES per move (9 floats):
    base_power / 150     (1)
    type_idx / 19        (1)   ← ordinal, not one-hot (keep dim small)
    category             (1)   ← 0=phys 0.5=spec 1=status
    priority / 7         (1)
    accuracy             (1)   ← 0–1
    pp_fraction          (1)   ← 1.0 offline (TODO: track PP in parser)
    is_protect           (1)
    is_stab              (1)
    is_known             (1)   ← 1.0 revealed/exact · p(usage) belief-padded
                                 · 0 empty slot
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ACTION SPACE (per active slot):
    0-11  → move index (0-3) × target (0=opp_a, 1=opp_b, 2=ally)
    12-15 → switch to bench slot 0-3
    Total: 16 actions per slot (illegal actions masked during MCTS)

    Conventions (action_to_index / build_action_mask):
    · move index m = the move's slot in move_slots_for_mon(mon) — the SAME
      ordering encode_snapshot writes the move features in, so policy
      logits and move features always line up.
    · moves without a player-choosable target (self, spread, field, …)
      canonicalise to target bucket 0; only that index is legal for them.
    · ally-targeting kinds (adjacentAlly / adjacentAllyOrSelf) live in
      bucket 2.
    · switch index 12+i = i-th LIVING mon of our_bench (encoder bench
      order).  Forced replacements picked mid-turn (pivot chains) may
      reference a mon that was active at turn start — those encode as
      None rather than a wrong index.
    · masks are decision-time legality from state_before_actions; they do
      not model trapping, Encore, Choice locks, or Taunt.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ── Bootstrap: sibling imports work when run/imported from anywhere ────────────
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from vod_parser.pokedex import get_pokedex, norm_species
from belief_state import BeliefState, dex_base_stats, STAT_ORDER

# ── poke-env is optional (offline training machines don't need it) ────────────
try:  # poke-env ≥ 0.8 layout
    from poke_env.battle import (
        DoubleBattle, Pokemon, Move, PokemonType, Status, Weather, Field,
        SideCondition, MoveCategory,
    )
    _HAS_POKE_ENV = True
except ImportError:
    try:  # older poke-env layout
        from poke_env.environment import (
            DoubleBattle, Pokemon, Move, PokemonType, Status, Weather, Field,
            SideCondition, MoveCategory,
        )
        _HAS_POKE_ENV = True
    except ImportError:
        DoubleBattle = Pokemon = Move = None  # type: ignore
        PokemonType = Status = Weather = Field = SideCondition = MoveCategory = None  # type: ignore
        _HAS_POKE_ENV = False


# ══════════════════════════════════════════════════════════════════════════════
# FROZEN feature orderings — single source of truth for tensor indices.
# Mirrors poke-env's enum definition order at the time of freezing; when
# poke-env is present we map its enum members onto these by NAME (members the
# canonical lists don't know are ignored, so a poke-env upgrade cannot move
# or grow any index).
# ══════════════════════════════════════════════════════════════════════════════
TYPE_NAMES = [
    "BUG", "DARK", "DRAGON", "ELECTRIC", "FAIRY", "FIGHTING", "FIRE",
    "FLYING", "GHOST", "GRASS", "GROUND", "ICE", "NORMAL", "POISON",
    "PSYCHIC", "ROCK", "STEEL", "STELLAR", "THREE_QUESTION_MARKS", "WATER",
]
STATUS_NAMES = ["BRN", "FNT", "FRZ", "PAR", "PSN", "SLP", "TOX"]
WEATHER_NAMES = [
    "UNKNOWN", "DESOLATELAND", "DELTASTREAM", "HAIL", "PRIMORDIALSEA",
    "RAINDANCE", "SANDSTORM", "SNOW", "SUNNYDAY",
]
FIELD_NAMES = [
    "UNKNOWN", "ELECTRIC_TERRAIN", "FAIRY_LOCK", "GRASSY_TERRAIN", "GRAVITY",
    "HEAL_BLOCK", "MAGIC_ROOM", "MISTY_TERRAIN", "MUD_SPORT", "MUD_SPOT",
    "NEUTRALIZING_GAS", "PSYCHIC_TERRAIN", "TRICK_ROOM", "WATER_SPORT",
    "WONDER_ROOM",
]
SIDE_COND_NAMES = [
    "UNKNOWN", "AURORA_VEIL", "CRAFTY_SHIELD", "FIRE_PLEDGE",
    "G_MAX_CANNONADE", "G_MAX_STEELSURGE", "G_MAX_VINE_LASH",
    "G_MAX_VOLCALITH", "G_MAX_WILDFIRE", "GRASS_PLEDGE", "LIGHT_SCREEN",
    "LUCKY_CHANT", "MAT_BLOCK", "MIST", "QUICK_GUARD", "REFLECT",
    "SAFEGUARD", "SPIKES", "STEALTH_ROCK", "STICKY_WEB", "TAILWIND",
    "TOXIC_SPIKES", "WATER_PLEDGE", "WIDE_GUARD",
]

_TYPE_IDX    = {n: i for i, n in enumerate(TYPE_NAMES)}
_STATUS_IDX  = {n: i for i, n in enumerate(STATUS_NAMES)}
_WEATHER_IDX = {n: i for i, n in enumerate(WEATHER_NAMES)}
_FIELD_IDX   = {n: i for i, n in enumerate(FIELD_NAMES)}
_SC_IDX      = {n: i for i, n in enumerate(SIDE_COND_NAMES)}

# ── Dimension constants ────────────────────────────────────────────────────────
NUM_TYPES      = len(TYPE_NAMES)        # 20
NUM_STATUS     = len(STATUS_NAMES)      # 7
NUM_WEATHER    = len(WEATHER_NAMES)     # 9
NUM_FIELDS     = len(FIELD_NAMES)       # 15
NUM_SIDE_CONDS = len(SIDE_COND_NAMES)   # 24
NUM_BOOSTS     = 7                      # atk def spa spd spe acc eva
NUM_MOVES      = 4
MOVE_FEATURES  = 9

POKEMON_FEATURES = (
    1               # hp_frac
    + NUM_TYPES     # type1 one-hot   (20)
    + NUM_TYPES     # type2 one-hot   (20)
    + 6             # base stats      (6)
    + 6             # est in-battle stats (6)   ← belief integration
    + 1             # stats_known flag          ← belief integration
    + 1             # is_mega
    + 1             # is_tera
    + NUM_STATUS    # status one-hot  (7)
    + NUM_BOOSTS    # stat boosts     (7)
    + NUM_MOVES * MOVE_FEATURES  # 4 × 9 = 36
    + 1             # is_active
    + 1             # is_revealed
)
# POKEMON_FEATURES = 1+20+20+6+6+1+1+1+7+7+36+1+1 = 108

ACTIVE_SLOTS = 4   # 2 own + 2 opp
BENCH_SLOTS  = 4   # own bench (up to 4 in a 6-mon VGC team)

GLOBAL_FEATURES = (
    NUM_WEATHER       # 9
    + NUM_FIELDS      # 15
    + NUM_SIDE_CONDS  # 24  own
    + NUM_SIDE_CONDS  # 24  opp
    + 1               # turn
    + 1               # trick room explicit flag
)  # = 74

STATE_DIM = (ACTIVE_SLOTS + BENCH_SLOTS) * POKEMON_FEATURES + GLOBAL_FEATURES
# = 8 × 108 + 74 = 938

# ── Action space ───────────────────────────────────────────────────────────────
ACTIONS_PER_SLOT = 16    # 12 move-target + 4 switch
ACTION_DIM       = ACTIONS_PER_SLOT

# Decode helpers
MOVE_TARGET_PAIRS = [(m, t) for m in range(4) for t in range(3)]  # indices 0–11
SWITCH_OFFSET     = 12   # actions 12–15 → bench slots 0–3


# ── Boost key order (must match encoder) ──────────────────────────────────────
_BOOST_KEYS = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]

# Protect-family moves (mirrors replay_parser._handle_move's inline set)
_PROTECT_MOVES = {
    "protect", "detect", "wideguard", "quickguard", "banefulbunker",
    "spikyshield", "silktrap", "burningbulwark", "maxguard",
}

_EST_STAT_NORM = 300.0   # L50 stats top out around ~290 (max-HP Blissey)

_NON_ALNUM = re.compile(r"[^A-Z0-9_]")


def _canon(name: Optional[str]) -> str:
    """'RainDance' → 'RAINDANCE', 'electric' → 'ELECTRIC'."""
    if not name:
        return ""
    return _NON_ALNUM.sub("", str(name).upper().replace(" ", "_"))


# Showdown protocol weather tokens → canonical WEATHER_NAMES entries
_WEATHER_ALIASES = {"SNOWSCAPE": "SNOW", "NONE": ""}
# vod_parser terrain tokens → canonical FIELD_NAMES entries
_TERRAIN_TO_FIELD = {
    "electric": "ELECTRIC_TERRAIN", "grassy": "GRASSY_TERRAIN",
    "misty": "MISTY_TERRAIN", "psychic": "PSYCHIC_TERRAIN",
}
# vod_parser screen keys → canonical SIDE_COND_NAMES entries
_SCREEN_TO_SC = {
    "reflect": "REFLECT", "light_screen": "LIGHT_SCREEN",
    "aurora_veil": "AURORA_VEIL",
}

# data/moves.json — lazy-loaded for the offline path
_MOVES_JSON_PATH = Path(__file__).resolve().parents[1] / "moves.json"
_moves_data: Optional[dict] = None


def _get_moves_data() -> dict:
    global _moves_data
    if _moves_data is None:
        try:
            _moves_data = json.loads(_MOVES_JSON_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            _moves_data = {}
    return _moves_data


def _dex_types(species: Optional[str]) -> list[str]:
    """Dex types for a species as canonical names, e.g. ['DARK', 'STEEL']."""
    dex = get_pokedex()
    entry = dex.entry(species) if dex else None
    if not entry:
        return []
    return [_canon(t) for t in (entry.get("types") or [])]


# poke-env member names that differ from the frozen canonical names
_LIVE_NAME_ALIASES = {"SNOWSCAPE": "SNOW", "MATBLOCK": "MAT_BLOCK"}


def _enum_to_idx(enum_cls, idx_map: dict[str, int]) -> dict:
    """Map live poke-env enum members onto frozen indices by member name."""
    out = {}
    for member in enum_cls:
        name = _LIVE_NAME_ALIASES.get(member.name, member.name)
        i = idx_map.get(name)
        if i is not None:
            out[member] = i
    return out


# ══════════════════════════════════════════════════════════════════════════════
class StateEncoder:
    """
    Encodes a battle state into a 1D float32 numpy array of shape (STATE_DIM,).

    Example (live, requires poke-env)::

        encoder = StateEncoder(belief=BeliefState("pikalytics_regma.json"))
        vec = encoder.encode(battle)                  # shape (937,)

    Example (offline, parsed/enriched VOD JSON)::

        encoder = StateEncoder()
        snap = enriched["turns"][0]["state_before_actions"]["p1"]
        vec = encoder.encode_snapshot(snap, turn=1)   # shape (937,)
    """

    def __init__(self, belief: Optional[BeliefState] = None, level: int = 50):
        self.state_dim  = STATE_DIM
        self.action_dim = ACTION_DIM
        self.belief     = belief   # opponent stat estimates on the live path
        self.level      = level

        # Live-path enum→index lookups (poke-env members mapped by name onto
        # the frozen orderings; only built when poke-env is importable)
        if _HAS_POKE_ENV:
            self._type_idx    = _enum_to_idx(PokemonType, _TYPE_IDX)
            self._status_idx  = _enum_to_idx(Status, _STATUS_IDX)
            self._weather_idx = _enum_to_idx(Weather, _WEATHER_IDX)
            self._field_idx   = _enum_to_idx(Field, _FIELD_IDX)
            self._sc_idx      = _enum_to_idx(SideCondition, _SC_IDX)

    # ══════════════════════════════════════════════════════════════════════════
    # LIVE PATH — poke-env DoubleBattle
    # ══════════════════════════════════════════════════════════════════════════
    def encode(self, battle: "DoubleBattle") -> np.ndarray:
        """
        Return a float32 vector of shape (STATE_DIM,) for the given battle.
        Safe to call at any point during a battle, including teampreview.
        """
        if not _HAS_POKE_ENV:
            raise RuntimeError(
                "poke-env is not installed — the live encode(battle) path is "
                "unavailable. Use encode_snapshot()/encode_transition() for "
                "parsed VOD JSON instead."
            )
        vec = np.zeros(STATE_DIM, dtype=np.float32)
        cursor = 0

        # ── [A] Active slots ─────────────────────────────────────────────────
        # active_pokemon / opponent_active_pokemon are List[Optional[Pokemon]]
        # of length 2; None means that slot is empty (fainted / not yet sent).
        try:
            own_active = battle.active_pokemon          # List[Optional[Pokemon]]
        except ValueError:
            own_active = [None, None]
        try:
            opp_active = battle.opponent_active_pokemon
        except ValueError:
            opp_active = [None, None]

        own_active_set = {p for p in own_active if p is not None}

        for mon in list(own_active):
            self._write_pokemon(vec, cursor, mon, is_active=True, is_own=True)
            cursor += POKEMON_FEATURES
        for mon in list(opp_active):
            self._write_pokemon(vec, cursor, mon, is_active=True, is_own=False)
            cursor += POKEMON_FEATURES

        # ── [B] Bench slots ──────────────────────────────────────────────────
        bench = [
            p for p in battle.team.values()
            if p not in own_active_set and not p.fainted
        ][:BENCH_SLOTS]

        for i in range(BENCH_SLOTS):
            mon = bench[i] if i < len(bench) else None
            self._write_pokemon(vec, cursor, mon, is_active=False, is_own=True)
            cursor += POKEMON_FEATURES

        # ── [C] Global features ──────────────────────────────────────────────

        # Weather: Dict[Weather, int] — multi-hot (normally one at a time)
        for w in battle.weather:
            if w in self._weather_idx:
                vec[cursor + self._weather_idx[w]] = 1.0
        cursor += NUM_WEATHER

        # Fields: Dict[Field, int]
        trick_room_active = False
        for f in battle.fields:
            if f.name == "TRICK_ROOM":
                trick_room_active = True
            if f in self._field_idx:
                vec[cursor + self._field_idx[f]] = 1.0
        cursor += NUM_FIELDS

        # Own side conditions: Dict[SideCondition, int]
        # value = layer count (stackable) or turn activated (non-stackable)
        # Normalise to [0,1] using max 3 layers
        for sc, val in battle.side_conditions.items():
            if sc in self._sc_idx:
                vec[cursor + self._sc_idx[sc]] = min(val, 3) / 3.0
        cursor += NUM_SIDE_CONDS

        # Opponent side conditions
        for sc, val in battle.opponent_side_conditions.items():
            if sc in self._sc_idx:
                vec[cursor + self._sc_idx[sc]] = min(val, 3) / 3.0
        cursor += NUM_SIDE_CONDS

        # Turn (cap at 60 for normalisation)
        vec[cursor] = min(battle.turn, 60) / 60.0
        cursor += 1

        # Trick Room explicit flag (already in fields but surfaced separately
        # because it flips speed priority — critical strategic signal)
        vec[cursor] = 1.0 if trick_room_active else 0.0
        cursor += 1

        assert cursor == STATE_DIM, (
            f"StateEncoder cursor mismatch: wrote {cursor}, expected {STATE_DIM}"
        )
        return vec

    # ── Pokémon encoder (live) ────────────────────────────────────────────────
    def _write_pokemon(
        self,
        vec: np.ndarray,
        start: int,
        mon: Optional["Pokemon"],
        is_active: bool,
        is_own: bool,
    ) -> None:
        """Write POKEMON_FEATURES floats into vec starting at `start`."""
        # Empty / unrevealed slot → all zeros (already zeroed by np.zeros)
        if mon is None:
            return

        i = start

        # hp_frac — current_hp_fraction returns 0 if current_hp is falsy
        vec[i] = mon.current_hp_fraction
        i += 1

        # type1 one-hot
        if mon.type_1 in self._type_idx:
            vec[i + self._type_idx[mon.type_1]] = 1.0
        i += NUM_TYPES

        # type2 one-hot (None if single-type)
        if mon.type_2 is not None and mon.type_2 in self._type_idx:
            vec[i + self._type_idx[mon.type_2]] = 1.0
        i += NUM_TYPES

        # Base stats: hp atk def spa spd spe (normalised by 255)
        base_stats = mon.base_stats
        for key in STAT_ORDER:
            vec[i] = base_stats.get(key, 0) / 255.0
            i += 1

        # Est in-battle stats + confidence flag (belief integration):
        #   own mon  → exact stats from the request (stats_known = 1.0)
        #   opp mon  → Pikalytics-weighted expectation   (stats_known = 0.5)
        est, known = self._live_est_stats(mon, is_own)
        for key in STAT_ORDER:
            vec[i] = (est.get(key) or 0) / _EST_STAT_NORM if est else 0.0
            i += 1
        vec[i] = known
        i += 1

        # Mega: species string contains 'mega' after mega_evolve() updates it
        vec[i] = 1.0 if "mega" in mon.species.lower() else 0.0
        i += 1

        # Terastallized
        vec[i] = 1.0 if mon.is_terastallized else 0.0
        i += 1

        # Status one-hot (None = healthy)
        if mon.status is not None and mon.status in self._status_idx:
            vec[i + self._status_idx[mon.status]] = 1.0
        i += NUM_STATUS

        # Boosts (Dict[str, int], values in [-6, +6])
        boosts = mon.boosts
        for j, key in enumerate(_BOOST_KEYS):
            vec[i + j] = boosts.get(key, 0) / 6.0
        i += NUM_BOOSTS

        # Moves (up to 4; remaining slots stay zero = unknown)
        move_list = list(mon.moves.values())[:NUM_MOVES]
        for m_idx in range(NUM_MOVES):
            if m_idx < len(move_list):
                self._write_move(vec, i, move_list[m_idx], mon)
            i += MOVE_FEATURES

        # is_active slot flag
        vec[i] = 1.0 if is_active else 0.0
        i += 1

        # is_revealed (own mons always 1; opp mons 1 once sent out / identified)
        vec[i] = 1.0 if mon.revealed else 0.0
        i += 1

    def _live_est_stats(
        self, mon: "Pokemon", is_own: bool
    ) -> tuple[Optional[dict], float]:
        """(stats dict | None, stats_known flag) for the live path."""
        if is_own:
            stats = getattr(mon, "stats", None) or {}
            if all(stats.get(k) for k in ("atk", "def", "spa", "spd", "spe")):
                est = dict(stats)
                if not est.get("hp"):
                    est["hp"] = mon.max_hp
                return est, 1.0
        if self.belief is not None:
            est = self.belief.expected_stats_weighted(
                mon.species, mon.base_stats, level=self.level
            )
            if est:
                return est, 0.5
        return None, 0.0

    # ── Move encoder (live) ───────────────────────────────────────────────────
    def _write_move(
        self,
        vec: np.ndarray,
        start: int,
        move: "Move",
        user: "Pokemon",
    ) -> None:
        """Write MOVE_FEATURES floats into vec starting at `start`."""
        i = start

        # Base power (cap at 250 since a few moves are absurdly high)
        vec[i] = min(move.base_power, 250) / 150.0
        i += 1

        # Move type as ordinal index (compact; type embeddings learned by net)
        if move.type in self._type_idx:
            vec[i] = self._type_idx[move.type] / (NUM_TYPES - 1)
        i += 1

        # Category: 0.0=physical, 0.5=special, 1.0=status
        cat = move.category
        if cat == MoveCategory.PHYSICAL:
            vec[i] = 0.0
        elif cat == MoveCategory.SPECIAL:
            vec[i] = 0.5
        else:
            vec[i] = 1.0
        i += 1

        # Priority (range roughly -7 to +5 in practice)
        vec[i] = move.priority / 7.0
        i += 1

        # Accuracy (already 0–1 float; always-hit moves return 1.0)
        vec[i] = move.accuracy
        i += 1

        # PP fraction remaining
        max_pp = move.max_pp or 1
        vec[i] = move.current_pp / max_pp
        i += 1

        # Protect-family move flag (high strategic value in doubles)
        vec[i] = 1.0 if move.is_protect_move else 0.0
        i += 1

        # STAB: move type in user's current types (accounts for tera)
        vec[i] = 1.0 if move.type in user.types else 0.0
        i += 1

        # Known flag (always 1 here — zero-slot means unknown)
        vec[i] = 1.0
        i += 1

    # ══════════════════════════════════════════════════════════════════════════
    # OFFLINE PATH — parsed / belief-enriched VOD JSON
    # ══════════════════════════════════════════════════════════════════════════
    def encode_snapshot(self, snap: dict, turn: int = 0) -> np.ndarray:
        """
        Encode one perspective snapshot (the value of state_before_actions /
        state_after_actions for one player) into the same (STATE_DIM,) layout
        as the live path.

        Accepts plain vod_parser output; belief_state.fill_blanks() enrichment
        (stats_estimate / belief / known_moves) is used when present.
        """
        vec = np.zeros(STATE_DIM, dtype=np.float32)
        cursor = 0

        # ── [A] Active slots: our_a, our_b, opp_a, opp_b ────────────────────
        our_active = snap.get("our_active") or {}
        opp_active = snap.get("opp_active") or {}
        for key_map, prefix in ((our_active, "our"), (opp_active, "opp")):
            for slot in ("a", "b"):
                mon = key_map.get(f"{prefix}_{slot}")
                self._write_pokemon_json(vec, cursor, mon, is_active=True)
                cursor += POKEMON_FEATURES

        # ── [B] Own bench (fainted excluded — matches the live path; the
        # snapshot keeps fainted mons by design, see replay_parser Bug 6) ────
        bench = [
            m for m in (snap.get("our_bench") or []) if not m.get("is_fainted")
        ][:BENCH_SLOTS]
        for i in range(BENCH_SLOTS):
            mon = bench[i] if i < len(bench) else None
            self._write_pokemon_json(vec, cursor, mon, is_active=False)
            cursor += POKEMON_FEATURES

        # ── [C] Global features ──────────────────────────────────────────────
        field = snap.get("field") or {}

        # Weather (parser stores the raw protocol token, e.g. "RainDance")
        w = _canon(field.get("weather"))
        w = _WEATHER_ALIASES.get(w, w)
        if w in _WEATHER_IDX:
            vec[cursor + _WEATHER_IDX[w]] = 1.0
        cursor += NUM_WEATHER

        # Terrain + trick room into the field multi-hot
        terrain = _TERRAIN_TO_FIELD.get(field.get("terrain") or "")
        if terrain in _FIELD_IDX:
            vec[cursor + _FIELD_IDX[terrain]] = 1.0
        trick_room = (field.get("trick_room_turns_remaining") or 0) > 0
        if trick_room:
            vec[cursor + _FIELD_IDX["TRICK_ROOM"]] = 1.0
        cursor += NUM_FIELDS

        # Side conditions.  Live path stores layer-count/turn-activated and
        # normalises min(val,3)/3; offline we have TURNS REMAINING, which is
        # strictly more informative — same normalisation, value semantics
        # differ slightly (documented divergence).
        side_conds = snap.get("side_conditions") or {}
        for side_key in ("our_side", "opp_side"):
            sc = side_conds.get(side_key) or {}
            tw = sc.get("tailwind_turns_remaining") or 0
            if tw > 0:
                vec[cursor + _SC_IDX["TAILWIND"]] = min(tw, 3) / 3.0
            for screen, turns_left in (sc.get("screens") or {}).items():
                name = _SCREEN_TO_SC.get(screen)
                if name and turns_left:
                    vec[cursor + _SC_IDX[name]] = min(turns_left, 3) / 3.0
            cursor += NUM_SIDE_CONDS

        # Turn (cap at 60 for normalisation)
        vec[cursor] = min(turn, 60) / 60.0
        cursor += 1

        # Trick Room explicit flag
        vec[cursor] = 1.0 if trick_room else 0.0
        cursor += 1

        assert cursor == STATE_DIM, (
            f"StateEncoder cursor mismatch: wrote {cursor}, expected {STATE_DIM}"
        )
        return vec

    def encode_transition(self, transition: dict) -> np.ndarray:
        """Encode a transitions.py dict (uses state_before_actions + turn)."""
        return self.encode_snapshot(
            transition.get("state_before_actions") or {},
            turn=transition.get("turn") or 0,
        )

    def encode_transitions_inplace(self, transitions: list[dict]) -> list[dict]:
        """Fill the reserved ``state_vector`` field on every transition."""
        for t in transitions:
            t["state_vector"] = self.encode_transition(t).tolist()
        return transitions

    # ── Pokémon encoder (offline JSON) ────────────────────────────────────────
    def _write_pokemon_json(
        self,
        vec: np.ndarray,
        start: int,
        mon: Optional[dict],
        is_active: bool,
    ) -> None:
        """JSON twin of _write_pokemon — same POKEMON_FEATURES layout."""
        if not mon:
            return

        i = start

        # hp_frac (parser stores 0–100 pct; None = unrevealed → 0…
        # …except unrevealed-but-alive bench stubs, which are at full HP)
        hp_pct = mon.get("hp_pct")
        if hp_pct is None:
            vec[i] = 0.0 if mon.get("seen", True) else 1.0
        else:
            vec[i] = max(0.0, min(hp_pct, 100.0)) / 100.0
        i += 1

        # Types: tera overrides; otherwise dex types of the CURRENT forme
        if mon.get("is_terastallized") and mon.get("known_tera_type"):
            types = [_canon(mon["known_tera_type"])]
        else:
            types = _dex_types(mon.get("species")) or _dex_types(mon.get("base_species"))
        if types and types[0] in _TYPE_IDX:
            vec[i + _TYPE_IDX[types[0]]] = 1.0
        i += NUM_TYPES
        if len(types) > 1 and types[1] in _TYPE_IDX:
            vec[i + _TYPE_IDX[types[1]]] = 1.0
        i += NUM_TYPES

        # Base stats from the dex (current forme, falling back to base forme)
        base = dex_base_stats(mon.get("species")) or dex_base_stats(mon.get("base_species"))
        for key in STAT_ORDER:
            vec[i] = (base.get(key, 0) / 255.0) if base else 0.0
            i += 1

        # Est in-battle stats + confidence (from belief_state.fill_blanks)
        est_block = mon.get("stats_estimate") or {}
        est = est_block.get("stats")
        known = {"exact": 1.0, "distribution": 0.5}.get(est_block.get("mode"), 0.0)
        for key in STAT_ORDER:
            vec[i] = (est.get(key) or 0) / _EST_STAT_NORM if est else 0.0
            i += 1
        vec[i] = known if est else 0.0
        i += 1

        # Mega / tera flags
        vec[i] = 1.0 if mon.get("is_mega") else 0.0
        i += 1
        vec[i] = 1.0 if mon.get("is_terastallized") else 0.0
        i += 1

        # Status one-hot ('brn'/'par'/… tokens; fainted → FNT)
        status = _canon(mon.get("status"))
        if mon.get("is_fainted"):
            status = "FNT"
        if status in _STATUS_IDX:
            vec[i + _STATUS_IDX[status]] = 1.0
        i += NUM_STATUS

        # Boosts
        boosts = mon.get("boosts") or {}
        for j, key in enumerate(_BOOST_KEYS):
            vec[i + j] = (boosts.get(key) or 0) / 6.0
        i += NUM_BOOSTS

        # Moves: canonical slot order shared with the action codec — see
        # move_slots_for_mon().
        slots = move_slots_for_mon(mon)
        for m_idx in range(NUM_MOVES):
            if m_idx < len(slots):
                name, confidence = slots[m_idx]
                self._write_move_json(vec, i, name, confidence, types)
            i += MOVE_FEATURES

        # is_active / is_revealed
        vec[i] = 1.0 if is_active else 0.0
        i += 1
        vec[i] = 1.0 if mon.get("seen", True) else 0.0
        i += 1

    # ── Move encoder (offline JSON) ───────────────────────────────────────────
    def _write_move_json(
        self,
        vec: np.ndarray,
        start: int,
        move_name: str,
        confidence: float,
        user_types: list[str],
    ) -> None:
        """JSON twin of _write_move using data/moves.json."""
        i = start
        data = _get_moves_data().get(norm_species(move_name))
        if not data:
            # Unknown move id: only the is_known confidence is encoded
            vec[start + MOVE_FEATURES - 1] = confidence
            return

        vec[i] = min(data.get("basePower") or 0, 250) / 150.0
        i += 1

        mtype = _canon(data.get("type"))
        if mtype in _TYPE_IDX:
            vec[i] = _TYPE_IDX[mtype] / (NUM_TYPES - 1)
        i += 1

        cat = (data.get("category") or "").lower()
        vec[i] = {"physical": 0.0, "special": 0.5}.get(cat, 1.0)
        i += 1

        vec[i] = (data.get("priority") or 0) / 7.0
        i += 1

        acc = data.get("accuracy")
        vec[i] = 1.0 if acc is True else (acc or 0) / 100.0
        i += 1

        # PP fraction — the parser does not track PP yet.
        # TODO(parser): count move uses per mon to derive real PP fractions.
        vec[i] = 1.0
        i += 1

        vec[i] = 1.0 if norm_species(move_name) in _PROTECT_MOVES else 0.0
        i += 1

        vec[i] = 1.0 if mtype in user_types else 0.0
        i += 1

        # is_known: 1.0 revealed/exact, p(usage) for belief-padded moves
        vec[i] = confidence
        i += 1


# ══════════════════════════════════════════════════════════════════════════════
# Action codec — string actions ⇄ fixed indices + legality masks
# (see ACTION SPACE in the module docstring for the frozen conventions)
# ══════════════════════════════════════════════════════════════════════════════

def move_slots_for_mon(mon: Optional[dict]) -> list[tuple[str, float]]:
    """
    Canonical ≤4-slot move list for one snapshot mon dict, as
    ``[(move_name, confidence), …]``.

    This is THE single source of truth shared by the feature encoder
    (_write_pokemon_json) and the action codec (action_to_index /
    build_action_mask): policy action ``m*3+t`` always refers to the move
    written into feature slot ``m`` of the same snapshot.

    Order: known_moves (exact/injected) wins over revealed_moves, then the
    remaining slots are padded with belief moves_predicted carrying p(usage)
    as confidence.
    """
    if not mon:
        return []
    slots: list[tuple[str, float]] = []
    for mv in (mon.get("known_moves") or mon.get("revealed_moves") or []):
        if mv:
            slots.append((mv, 1.0))
    belief = mon.get("belief") or {}
    for predicted in belief.get("moves_predicted") or []:
        if len(slots) >= NUM_MOVES:
            break
        slots.append((predicted["name"], float(predicted.get("p") or 0.0)))
    return slots[:NUM_MOVES]


# Showdown dex target kinds with a player-choosable single target
_CHOOSABLE_SINGLE = {"normal", "any", "adjacentFoe"}
# Kinds that always point at our own side (bucket 2)
_ALLY_KINDS = {"adjacentAlly", "adjacentAllyOrSelf"}


def _move_target_kind(move_name: Optional[str]) -> Optional[str]:
    """data/moves.json 'target' field for a move ('normal', 'self', …)."""
    if not move_name:
        return None
    data = _get_moves_data().get(norm_species(move_name))
    return (data or {}).get("target")


def _target_bucket(
    move_name: Optional[str],
    target_slot: Optional[str],
    opp_present: Optional[dict] = None,
) -> int:
    """Resolve a logged move action to its target bucket (0/1/2).

    ``opp_present`` ({0: bool, 1: bool} for opp_a / opp_b presence at decision
    time) lets a single-target move whose target Showdown did not log — it omits
    the target when auto-targeting the ONLY legal foe — resolve to the foe that
    is actually on the field, instead of defaulting to a possibly-empty opp_a
    slot.  Without this the index disagrees with build_action_mask (which marks
    only the present foe legal), producing an action that is illegal under its
    own mask.  Passing None preserves the legacy "trust the logged slot" path.
    """
    kind = _move_target_kind(move_name)
    if kind in _ALLY_KINDS:
        return 2
    if kind in _CHOOSABLE_SINGLE or kind is None:
        # kind None = move missing from moves.json → trust the logged slot
        if target_slot == "opp_a" and (opp_present is None or opp_present.get(0)):
            return 0
        if target_slot == "opp_b" and (opp_present is None or opp_present.get(1)):
            return 1
        if target_slot in ("our_a", "our_b"):
            return 2
        # No usable foe target logged (auto-targeted), or the logged foe slot is
        # empty at decision time (redirect) → point at whichever single foe is
        # actually present, matching Showdown auto-targeting and the mask.
        if opp_present is not None:
            if opp_present.get(0) and not opp_present.get(1):
                return 0
            if opp_present.get(1) and not opp_present.get(0):
                return 1
        return 0          # ambiguous (both/neither present) → canonical bucket
    return 0              # non-choosable (self / spread / field / …)


def _living_bench(snap: dict) -> list[dict]:
    """Our bench in encoder order: living mons only, capped at BENCH_SLOTS."""
    return [m for m in (snap.get("our_bench") or [])
            if not m.get("is_fainted")][:BENCH_SLOTS]


def _species_matches(mon: dict, species: Optional[str]) -> bool:
    if not species:
        return False
    want = norm_species(species)
    return want in (
        norm_species(mon.get("species") or ""),
        norm_species(mon.get("base_species") or ""),
    )


def action_to_index(
    action: dict,
    mon: Optional[dict],
    snap: Optional[dict] = None,
) -> Optional[int]:
    """
    Encode one perspective-relative action dict (transitions.py our_actions
    entry) into the fixed 0–15 index space.

    ``mon``  — the acting mon's dict from snap["our_active"][action["slot"]]
               (needed for move actions; switches only need ``snap``).
    ``snap`` — the decision-time snapshot (state_before_actions), used for
               the bench ordering of switch actions.

    Returns None when the action cannot be expressed: move not among the
    mon's 4 encoded slots, or switch target not on the living bench (e.g.
    mid-turn forced replacement by a mon that was active at turn start).
    """
    kind = action.get("action")
    if kind == "switch":
        for i, bench_mon in enumerate(_living_bench(snap or {})):
            if _species_matches(bench_mon, action.get("species")):
                return SWITCH_OFFSET + i
        return None
    if kind == "move":
        want = norm_species(action.get("move") or "")
        if not want:
            return None
        opp_active = (snap or {}).get("opp_active") or {}
        opp_present = {0: "opp_a" in opp_active, 1: "opp_b" in opp_active}
        for m_idx, (name, _conf) in enumerate(move_slots_for_mon(mon)):
            if norm_species(name) == want:
                return m_idx * 3 + _target_bucket(
                    name, action.get("target_slot"), opp_present
                )
        return None
    return None


def index_to_action(idx: int) -> dict:
    """Decode a 0–15 action index into its structural meaning."""
    if 0 <= idx < SWITCH_OFFSET:
        m_idx, bucket = divmod(idx, 3)
        return {
            "kind": "move",
            "move_slot": m_idx,
            "target_bucket": bucket,
            "target": ("opp_a", "opp_b", "ally")[bucket],
        }
    if SWITCH_OFFSET <= idx < ACTIONS_PER_SLOT:
        return {"kind": "switch", "bench_slot": idx - SWITCH_OFFSET}
    raise ValueError(f"action index out of range: {idx}")


def build_action_mask(snap: dict) -> dict[str, list[int]]:
    """
    Decision-time legality mask for one state_before_actions snapshot:
    ``{"our_a": [16×0/1], "our_b": [16×0/1]}``.

    Legal = the agent could have selected it at the START of the turn:
      · every named move slot, restricted to its reachable target buckets
        (occupied opposing slots / present ally; non-choosable moves expose
        only their canonical bucket 0; no legal foe → bucket-0 fallback,
        mirroring Showdown's auto-targeting),
      · one switch index per living bench mon.
    An empty/fainted active slot has an all-zero row.  Trapping, Encore,
    Choice locks and Taunt are NOT modelled (approximate legality).
    """
    our_active = snap.get("our_active") or {}
    opp_active = snap.get("opp_active") or {}
    n_bench    = len(_living_bench(snap))
    opp_present = {0: "opp_a" in opp_active, 1: "opp_b" in opp_active}

    mask: dict[str, list[int]] = {}
    for slot in ("our_a", "our_b"):
        row = [0] * ACTIONS_PER_SLOT
        mon = our_active.get(slot)
        if mon and not mon.get("is_fainted"):
            ally_slot = "our_b" if slot == "our_a" else "our_a"
            ally_present = ally_slot in our_active
            for m_idx, (name, _conf) in enumerate(move_slots_for_mon(mon)):
                kind = _move_target_kind(name)
                if kind in _ALLY_KINDS:
                    if kind == "adjacentAllyOrSelf" or ally_present:
                        row[m_idx * 3 + 2] = 1
                elif kind in _CHOOSABLE_SINGLE:
                    buckets = [b for b in (0, 1) if opp_present[b]]
                    if kind != "adjacentFoe" and ally_present:
                        buckets.append(2)
                    for b in (buckets or [0]):
                        row[m_idx * 3 + b] = 1
                else:
                    # self / spread / field / unknown → canonical bucket only
                    row[m_idx * 3] = 1
            for i in range(n_bench):
                row[SWITCH_OFFSET + i] = 1
        mask[slot] = row
    return mask


def annotate_transition_actions(transition: dict) -> dict:
    """
    Stamp the action-space view onto one transitions.py dict, in place:

      · each our_actions entry gains ``action_index`` (0–15 or None),
      · ``action_mask`` becomes the build_action_mask() of
        state_before_actions (the decision-time state).

    Pure-python and deterministic given the stored snapshot (the belief
    padding that orders move slots is baked into the same JSON), so exports
    can carry it without the state_vector being encoded.

    Invariant guaranteed here: every non-null ``action_index`` is legal under
    the same transition's ``action_mask``.  If an action cannot be resolved to a
    mask-legal index (an unresolved auto-target / redirect edge), it is stamped
    None — never a wrong index — so a downstream masked policy can never be told
    the answer was an illegal action.
    """
    snap = transition.get("state_before_actions") or {}
    our_active = snap.get("our_active") or {}
    mask = build_action_mask(snap)
    for act in transition.get("our_actions") or []:
        slot = act.get("slot")
        idx = action_to_index(act, our_active.get(slot), snap)
        if idx is not None:
            row = mask.get(slot) or []
            if idx >= len(row) or row[idx] != 1:
                idx = None  # not expressible in the turn-start frame → unencoded
        act["action_index"] = idx
    transition["action_mask"] = mask
    return transition


# ── Convenience accessors ──────────────────────────────────────────────────────
def get_state_dim() -> int:
    return STATE_DIM

def get_action_dim() -> int:
    return ACTION_DIM
