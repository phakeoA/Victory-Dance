"""
State encoder for VGC Reg M-A (Pokémon Champions doubles format).

Converts a DoubleBattle into a fixed-size float32 numpy array for the
AlphaZero neural network.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TENSOR LAYOUT  (total STATE_DIM floats)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [A] 4 active slots × POKEMON_FEATURES
        slot 0 → own active mon 0
        slot 1 → own active mon 1
        slot 2 → opp active mon 0
        slot 3 → opp active mon 1

  [B] 4 bench slots × POKEMON_FEATURES
        own bench mons (up to 4; zeros if fewer)

  [C] GLOBAL_FEATURES
        weather multi-hot (9)
        terrain/field multi-hot (15)
        own side conditions multi-hot (24)
        opp side conditions multi-hot (24)
        turn normalised (1)
        trick room flag (1)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
POKEMON_FEATURES per slot (101 floats):
    hp_frac       (1)
    type1 one-hot (20)
    type2 one-hot (20)   ← zeros if single-type
    base_stats    (6)    ← hp atk def spa spd spe / 255
    is_mega       (1)    ← species name contains 'mega'
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
    pp_fraction          (1)
    is_protect           (1)
    is_stab              (1)
    is_known             (1)   ← 0 for unknown/empty slot
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ACTION SPACE (per active slot):
    0-11  → move index (0-3) × target (0=opp0, 1=opp1, 2=ally)
    12-15 → switch to bench slot 0-3
    Total: 16 actions per slot (illegal actions masked during MCTS)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np

from poke_env.battle import (
    DoubleBattle,
    Pokemon,
    Move,
    PokemonType,
    Status,
    Weather,
    Field,
    SideCondition,
    MoveCategory,
)


# ── Ordered enum lists (index stability matters for encoding) ──────────────────
_TYPES      = list(PokemonType)        # 20
_STATUSES   = list(Status)             # 7
_WEATHERS   = list(Weather)            # 9
_FIELDS     = list(Field)              # 15
_SIDE_CONDS = list(SideCondition)      # 24

# ── Dimension constants ────────────────────────────────────────────────────────
NUM_TYPES         = len(_TYPES)        # 20
NUM_STATUS        = len(_STATUSES)     # 7
NUM_WEATHER       = len(_WEATHERS)     # 9
NUM_FIELDS        = len(_FIELDS)       # 15
NUM_SIDE_CONDS    = len(_SIDE_CONDS)   # 24
NUM_BOOSTS        = 7                  # atk def spa spd spe acc eva
NUM_MOVES         = 4
MOVE_FEATURES     = 9

POKEMON_FEATURES = (
    1               # hp_frac
    + NUM_TYPES     # type1 one-hot   (20)
    + NUM_TYPES     # type2 one-hot   (20)
    + 6             # base stats      (6)
    + 1             # is_mega
    + 1             # is_tera
    + NUM_STATUS    # status one-hot  (7)
    + NUM_BOOSTS    # stat boosts     (7)
    + NUM_MOVES * MOVE_FEATURES  # 4 × 9 = 36
    + 1             # is_active
    + 1             # is_revealed
)
# POKEMON_FEATURES = 1+20+20+6+1+1+7+7+36+1+1 = 101

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

# ── Action space ───────────────────────────────────────────────────────────────
ACTIONS_PER_SLOT = 16    # 12 move-target + 4 switch
ACTION_DIM       = ACTIONS_PER_SLOT

# Decode helpers
MOVE_TARGET_PAIRS = [(m, t) for m in range(4) for t in range(3)]  # indices 0–11
SWITCH_OFFSET     = 12   # actions 12–15 → bench slots 0–3


# ── Boost key order (must match encoder) ──────────────────────────────────────
_BOOST_KEYS = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]


# ══════════════════════════════════════════════════════════════════════════════
class StateEncoder:
    """
    Encodes a DoubleBattle into a 1D float32 numpy array of shape (STATE_DIM,).

    Example::

        encoder = StateEncoder()
        vec = encoder.encode(battle)   # shape (882,)
    """

    def __init__(self):
        self.state_dim  = STATE_DIM
        self.action_dim = ACTION_DIM

        # Pre-build lookup dicts for O(1) index access
        self._type_idx    = {t: i for i, t in enumerate(_TYPES)}
        self._status_idx  = {s: i for i, s in enumerate(_STATUSES)}
        self._weather_idx = {w: i for i, w in enumerate(_WEATHERS)}
        self._field_idx   = {f: i for i, f in enumerate(_FIELDS)}
        self._sc_idx      = {sc: i for i, sc in enumerate(_SIDE_CONDS)}

    # ──────────────────────────────────────────────────────────────────────────
    def encode(self, battle: DoubleBattle) -> np.ndarray:
        """
        Return a float32 vector of shape (STATE_DIM,) for the given battle.
        Safe to call at any point during a battle, including teampreview.
        """
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

        for mon in list(own_active) + list(opp_active):
            self._write_pokemon(vec, cursor, mon, is_active=True)
            cursor += POKEMON_FEATURES

        # ── [B] Bench slots ──────────────────────────────────────────────────
        bench = [
            p for p in battle.team.values()
            if p not in own_active_set and not p.fainted
        ][:BENCH_SLOTS]

        for i in range(BENCH_SLOTS):
            mon = bench[i] if i < len(bench) else None
            self._write_pokemon(vec, cursor, mon, is_active=False)
            cursor += POKEMON_FEATURES

        # ── [C] Global features ──────────────────────────────────────────────

        # Weather: Dict[Weather, int] — multi-hot (normally one at a time)
        for w in battle.weather:
            if w in self._weather_idx:
                vec[cursor + self._weather_idx[w]] = 1.0
        cursor += NUM_WEATHER

        # Fields: Dict[Field, int]
        for f in battle.fields:
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
        vec[cursor] = 1.0 if Field.TRICK_ROOM in battle.fields else 0.0
        cursor += 1

        assert cursor == STATE_DIM, (
            f"StateEncoder cursor mismatch: wrote {cursor}, expected {STATE_DIM}"
        )
        return vec

    # ── Pokémon encoder ───────────────────────────────────────────────────────
    def _write_pokemon(
        self,
        vec: np.ndarray,
        start: int,
        mon: Optional[Pokemon],
        is_active: bool,
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
        stats = mon.base_stats
        for key in ("hp", "atk", "def", "spa", "spd", "spe"):
            vec[i] = stats.get(key, 0) / 255.0
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
        for key in _BOOST_KEYS:
            vec[i] = boosts.get(key, 0) / 6.0
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

    # ── Move encoder ──────────────────────────────────────────────────────────
    def _write_move(
        self,
        vec: np.ndarray,
        start: int,
        move: Move,
        user: Pokemon,
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


# ── Convenience accessors ──────────────────────────────────────────────────────
def get_state_dim() -> int:
    return STATE_DIM

def get_action_dim() -> int:
    return ACTION_DIM
