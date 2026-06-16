"""
State encoder for VGC Reg M-A (Pokémon Champions doubles format) — OFFLINE
(VOD-parsing) path + the FROZEN tensor layout shared by both paths.

This module is the canonical owner of the 1398-dim layout: every feature
ordering, dimension constant, and the action codec live here, and the LIVE
encoder (``live_state_encoder.LiveStateEncoder``) imports them so the two paths
can never drift.  It encodes parsed/belief-enriched VOD JSON:

  OFFLINE — encode_snapshot(snap)    a state_before/after_actions snapshot
            encode_transition(t)     from vod_parser / belief_state JSON
                                     (Types A/B + training ingestion)

  LIVE    — encode(battle)           lives in live_state_encoder.py (poke-env
                                     DoubleBattle; bot play, Type C/D).  It is
                                     kept in a separate module so this one never
                                     imports poke-env.

This module has NO poke-env dependency: it works on training machines without
it.  All enum orderings are FROZEN here (not derived from poke-env at import
time) so STATE_DIM can never silently change with a poke-env upgrade and
invalidate a trained net.  ``StateEncoder`` remains as a backward-compatible
alias of ``VodStateEncoder`` (the offline encoder) at the bottom of the file.

BELIEF INTEGRATION (belief_state.py)
────────────────────────────────────
fill_blanks() writes per-mon ``stats_estimate`` ({"mode": "exact"|"distribution",
"stats": {...}}) and ``belief`` blocks into the parsed JSON.  The offline
encoder consumes them:
  * est_stats (6 floats)  — exact L50 stats (own side / self-play) or the
    Pikalytics probability-weighted expected stats (opponent side)
  * stats_known (1 float) — 1.0 exact, 0.5 distribution estimate, 0.0 unknown
  * unknown move slots are padded with belief moves_predicted, with the
    is_known feature carrying the usage probability instead of 1.0
(The live path's equivalent belief handling — own mons from poke-env's real
stats, opponent mons from the BeliefState — lives in live_state_encoder.py.)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TENSOR LAYOUT  (total STATE_DIM floats)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [A] 4 active slots × POKEMON_FEATURES
        slot 0 → own active mon 0 (our_a)
        slot 1 → own active mon 1 (our_b)
        slot 2 → opp active mon 0 (opp_a)
        slot 3 → opp active mon 1 (opp_b)

  [B] 4 OWN bench slots × POKEMON_FEATURES
        own bench mons (living switch-ins, fainted excluded; zeros if fewer).
        Kept living-only so switch slots line up with the action codec.

  [B2] 4 OPP bench slots × POKEMON_FEATURES   (layout-v2)
        opponent's non-active roster mons, ordered seen-alive → seen-fainted →
        unseen teampreview stubs (so the 4 slots keep the known switch-ins).
        is_revealed / is_fainted distinguish them; lets the net read the
        opponent's remaining team and likely switch-ins.

  [C] GLOBAL_FEATURES
        weather multi-hot (9)
        terrain/field multi-hot (15)
        own side conditions multi-hot (24)
        opp side conditions multi-hot (24)
        turn normalised (1)
        trick room flag (1)
        team counts (4)      ← own/opp living-bench, own/opp fainted (each /4)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
POKEMON_FEATURES per slot (110 floats):
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
    is_fainted    (1)    ← layout-v2; KO flag (opp bench / counting)
    is_transformed(1)    ← layout-v2; Ditto copied a forme (types/stats above
                           are the COPY's); reverts on switch-out
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
    + 1             # is_fainted      (layout-v2: see/count KO'd mons)
    + 1             # is_transformed  (layout-v2: Ditto copies a forme; reverts)
)
# POKEMON_FEATURES = 1+20+20+6+6+1+1+1+7+7+36+1+1+1+1 = 110

ACTIVE_SLOTS    = 4   # 2 own + 2 opp
BENCH_SLOTS     = 4   # own bench (living switch-ins; aligned with the action codec)
OPP_BENCH_SLOTS = 4   # opp non-active roster mons (seen incl. fainted + stubs)  (layout-v2)

GLOBAL_FEATURES = (
    NUM_WEATHER       # 9
    + NUM_FIELDS      # 15
    + NUM_SIDE_CONDS  # 24  own
    + NUM_SIDE_CONDS  # 24  opp
    + 1               # turn
    + 1               # trick room explicit flag
    + 4               # team counts (layout-v2): own/opp living-bench, own/opp fainted
)  # = 78

STATE_DIM = (ACTIVE_SLOTS + BENCH_SLOTS + OPP_BENCH_SLOTS) * POKEMON_FEATURES + GLOBAL_FEATURES
# = 12 × 110 + 78 = 1398

# ── Action space ───────────────────────────────────────────────────────────────
ACTIONS_PER_SLOT = 16    # 12 move-target + 4 switch
ACTION_DIM       = ACTIONS_PER_SLOT

# Decode helpers
MOVE_TARGET_PAIRS = [(m, t) for m in range(4) for t in range(3)]  # indices 0–11
SWITCH_OFFSET     = 12   # actions 12–15 → bench slots 0–3

# ── Gimmick (mega-evolution) decision ────────────────────────────────────────
# A SEPARATE per-slot 2-way head, ORTHOGONAL to the 16-way move/switch head: a
# mega is a checkbox alongside the chosen move, not a competing action.  Keeping
# it out of ACTION_DIM (frozen 16) leaves the move/switch policy untouched and
# stops the rare positive mega signal from competing in the move softmax.
# TODO(tera): when Terastallization enters Reg M-A AND replays carry it, bump to
# 3 {none, mega, tera}, add the tera label-join + a serve branch, and re-export.
# Tera is NOT in the format yet — build NOTHING tera-specific now.
GIMMICK_NONE = 0
GIMMICK_MEGA = 1
GIMMICK_DIM  = 2


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


# ══════════════════════════════════════════════════════════════════════════════
class VodStateEncoder:
    """
    Offline VOD-parsing encoder: parsed / belief-enriched replay JSON →
    a (STATE_DIM,) float32 vector.  Shares the FROZEN tensor layout with the
    live encoder (live_state_encoder.LiveStateEncoder); that module imports
    the constants defined here, so both paths are guaranteed identical.

        enc = VodStateEncoder()
        snap = enriched["turns"][0]["state_before_actions"]["p1"]
        vec = enc.encode_snapshot(snap, turn=1)   # shape (STATE_DIM,)
    """

    def __init__(self, belief: Optional[BeliefState] = None, level: int = 50):
        self.state_dim  = STATE_DIM
        self.action_dim = ACTION_DIM
        self.belief     = belief   # accepted for signature compatibility
        self.level      = level

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

        # ── [B2] Opponent bench (layout-v2): opp non-active roster mons.
        # Ordered seen-alive → seen-fainted → unseen stubs, so the 4 slots
        # prioritise known switch-ins; is_revealed/is_fainted distinguish them.
        opp_bench_all    = snap.get("opp_bench") or []
        opp_seen_alive   = [m for m in opp_bench_all if m.get("seen", True) and not m.get("is_fainted")]
        opp_seen_fainted = [m for m in opp_bench_all if m.get("seen", True) and m.get("is_fainted")]
        opp_unseen       = [m for m in opp_bench_all if not m.get("seen", True)]
        opp_bench = (opp_seen_alive + opp_seen_fainted + opp_unseen)[:OPP_BENCH_SLOTS]
        for i in range(OPP_BENCH_SLOTS):
            mon = opp_bench[i] if i < len(opp_bench) else None
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

        # Side conditions — BINARY presence (1.0 = active), gap #5.  The two paths
        # cannot agree on a magnitude: offline has true turns-remaining (from the
        # protocol) while poke-env exposes only the TURN a condition STARTED, and
        # a live bot cannot know an opponent's screen duration anyway (Light Clay).
        # A presence bit is the one representation both paths reproduce identically.
        side_conds = snap.get("side_conditions") or {}
        for side_key in ("our_side", "opp_side"):
            sc = side_conds.get(side_key) or {}
            if (sc.get("tailwind_turns_remaining") or 0) > 0:
                vec[cursor + _SC_IDX["TAILWIND"]] = 1.0
            for screen, turns_left in (sc.get("screens") or {}).items():
                name = _SCREEN_TO_SC.get(screen)
                if name and turns_left:
                    vec[cursor + _SC_IDX[name]] = 1.0
            cursor += NUM_SIDE_CONDS

        # Turn (cap at 60 for normalisation)
        vec[cursor] = min(turn, 60) / 60.0
        cursor += 1

        # Trick Room explicit flag
        vec[cursor] = 1.0 if trick_room else 0.0
        cursor += 1

        # ── Team counts (layout-v2): living-bench + fainted per side ──────────
        # Own bench keeps brought-but-unentered stubs (all our switch-ins);
        # opp counts use only SEEN mons (information asymmetry).
        our_bench_all = snap.get("our_bench") or []
        own_live = sum(1 for m in our_bench_all if not m.get("is_fainted"))
        own_fnt  = sum(1 for m in our_bench_all if m.get("is_fainted"))
        vec[cursor] = min(own_live, 4) / 4.0;               cursor += 1
        vec[cursor] = min(len(opp_seen_alive), 4) / 4.0;    cursor += 1
        vec[cursor] = min(own_fnt, 4) / 4.0;                cursor += 1
        vec[cursor] = min(len(opp_seen_fainted), 4) / 4.0;  cursor += 1

        assert cursor == STATE_DIM, (
            f"VodStateEncoder cursor mismatch: wrote {cursor}, expected {STATE_DIM}"
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

        # Transform (Ditto / Imposter, Solution A): a transformed mon copies the
        # target's typing + base stats, so read the dex for the COPIED forme.
        # (HP stays the mon's own — handled above; reverts on switch-out, which
        # the parser already clears.)
        if mon.get("is_transformed") and mon.get("transformed_into"):
            dex_species = mon["transformed_into"]
        else:
            dex_species = mon.get("species")
        base_fallback = mon.get("base_species")

        # Types: tera overrides; otherwise dex types of the current/copied forme
        if mon.get("is_terastallized") and mon.get("known_tera_type"):
            types = [_canon(mon["known_tera_type"])]
        else:
            types = _dex_types(dex_species) or _dex_types(base_fallback)
        if types and types[0] in _TYPE_IDX:
            vec[i + _TYPE_IDX[types[0]]] = 1.0
        i += NUM_TYPES
        if len(types) > 1 and types[1] in _TYPE_IDX:
            vec[i + _TYPE_IDX[types[1]]] = 1.0
        i += NUM_TYPES

        # Base stats from the dex (copied forme when transformed, else current,
        # falling back to base forme)
        base = dex_base_stats(dex_species) or dex_base_stats(base_fallback)
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

        # is_active / is_revealed / is_fainted / is_transformed (layout-v2)
        vec[i] = 1.0 if is_active else 0.0
        i += 1
        vec[i] = 1.0 if mon.get("seen", True) else 0.0
        i += 1
        vec[i] = 1.0 if mon.get("is_fainted") else 0.0
        i += 1
        vec[i] = 1.0 if mon.get("is_transformed") else 0.0
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


def action_to_gimmick(action: dict) -> int:
    """Map one our_actions entry to its gimmick bucket (0=none, 1=mega).

    The parser stamps ``mega=True`` onto the chosen move for the slot that mega-
    evolved that turn (replay_parser._extract_actions).  Switches/replacements
    never gimmick, so they map to GIMMICK_NONE.
    """
    return GIMMICK_MEGA if action.get("mega") else GIMMICK_NONE


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


# ── Move-slot permutation (training augmentation, task #22) ──────────────────
# The BC net is ~96% move-ORDER sensitive: it leans on a move's slot POSITION (a
# "slot 0 = main move" prior baked in by reveal-order training) instead of the
# move's FEATURES — and at serve the own moves arrive in poke-env REQUEST order,
# so the prior is misapplied.  Train-only fix: randomly permute a mon's 4 move
# sub-blocks AND the matching action label so the net must read move features,
# not position.  These helpers are the single source for WHERE the move blocks
# live (so the dataset augmentation can never drift from the encoder layout).
#
# Within a POKEMON_FEATURES block the 4 move sub-blocks start here (after hp,
# both types, base+est stats, stats-known, mega, tera, status, boosts):
_MOVE_BLOCK_REL = 1 + 2 * NUM_TYPES + 6 + 6 + 1 + 1 + 1 + NUM_STATUS + NUM_BOOSTS  # 70
# Own ACTIVE mons are state blocks 0 (our_a) and 1 (our_b); opp/bench moves have
# no action head, so they are never permuted.


def own_active_move_base(slot: int) -> int:
    """Absolute index of own active ``slot``'s (0=our_a, 1=our_b) first move
    feature in an encoded state vector."""
    return slot * POKEMON_FEATURES + _MOVE_BLOCK_REL


def permute_move_slots(vec: np.ndarray, slot: int, perm: Sequence[int]) -> None:
    """In place: reorder the 4 move sub-blocks of own active ``slot`` so the new
    move-slot ``i`` holds the OLD move-slot ``perm[i]``.  ``perm`` is a
    permutation of ``range(NUM_MOVES)``; all other features are left untouched."""
    base = own_active_move_base(slot)
    blocks = [vec[base + m * MOVE_FEATURES: base + (m + 1) * MOVE_FEATURES].copy()
              for m in range(NUM_MOVES)]
    for i, src in enumerate(perm):
        vec[base + i * MOVE_FEATURES: base + (i + 1) * MOVE_FEATURES] = blocks[src]


def permute_action_index(idx: Optional[int], perm: Sequence[int]) -> Optional[int]:
    """Remap an action index under move-slot permutation ``perm`` (``perm[i]`` =
    the old move-slot now at position ``i``).  Switch indices (>=SWITCH_OFFSET),
    None and out-of-range pass through unchanged; a move index ``m*3+bucket`` maps
    to ``j*3+bucket`` where ``perm[j] == m``."""
    if idx is None or idx < 0 or idx >= SWITCH_OFFSET:
        return idx
    m, bucket = divmod(idx, 3)
    return list(perm).index(m) * 3 + bucket


def permute_action_mask_row(row: Sequence, perm: Sequence[int]) -> list:
    """Remap a 16-wide action mask row under ``perm``: new move block ``j`` takes
    old move block ``perm[j]``; switch entries (>=SWITCH_OFFSET) are unchanged."""
    out = list(row)
    for j in range(NUM_MOVES):
        src = perm[j]
        for b in range(3):
            out[j * 3 + b] = row[src * 3 + b]
    return out


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


def _species_is_mega_capable(species: Optional[str]) -> bool:
    """True iff this (base) species has a mega forme in the dex.

    We gate the gimmick mask on mega-CAPABILITY (species) rather than on holding
    a mega stone (item), because at the moment a mon mega-evolves its stone is
    revealed DURING that turn — the decision-time snapshot (state_before_actions)
    does NOT yet carry the item for ~99% of real megas.  An item-gated mask would
    mark the actual mega label illegal and annotate_transition_actions would drop
    it, leaving the head nothing to learn.  Capability is the stable, offline-
    computable signal that keeps every real mega label legal; the LIVE serve path
    additionally gates the emitted order on ``battle.can_mega_evolve`` (true item-
    aware legality), so an illegal mega order is never sent.
    """
    dex = get_pokedex()
    return bool(dex.mega_formes_for(species)) if dex else False


def _own_team_has_megaed(snap: dict) -> bool:
    """Has any own mon already mega-evolved this game?  A team megas at most once
    per game, so the moment any own active/bench mon shows ``is_mega`` the gimmick
    is spent and mega is no longer a legal choice for either slot."""
    mons = list((snap.get("our_active") or {}).values()) + list(snap.get("our_bench") or [])
    return any(m.get("is_mega") for m in mons)


def build_gimmick_mask(snap: dict) -> dict[str, list[int]]:
    """
    Decision-time gimmick legality for one state_before_actions snapshot:
    ``{"our_a": [2×0/1], "our_b": [2×0/1]}`` over (none, mega).

    bucket 0 (none) is legal for any acting (present, non-fainted) slot — not
    gimmicking is always allowed.  bucket 1 (mega) is legal iff the acting mon is
    mega-capable AND no own mon has used mega this game.  An empty/fainted active
    slot gets an all-zero row, mirroring build_action_mask.
    """
    our_active  = snap.get("our_active") or {}
    team_megaed = _own_team_has_megaed(snap)

    mask: dict[str, list[int]] = {}
    for slot in ("our_a", "our_b"):
        row = [0] * GIMMICK_DIM
        mon = our_active.get(slot)
        if mon and not mon.get("is_fainted"):
            row[GIMMICK_NONE] = 1
            base = mon.get("base_species") or mon.get("species")
            if not team_megaed and _species_is_mega_capable(base):
                row[GIMMICK_MEGA] = 1
        mask[slot] = row
    return mask


def annotate_transition_actions(transition: dict) -> dict:
    """
    Stamp the action-space view onto one transitions.py dict, in place:

      · each our_actions entry gains ``action_index`` (0–15 or None) and
        ``gimmick_index`` (0=none / 1=mega / None),
      · ``action_mask`` becomes the build_action_mask() of
        state_before_actions (the decision-time state),
      · ``gimmick_mask`` becomes the build_gimmick_mask() of the same state.

    Pure-python and deterministic given the stored snapshot (the belief
    padding that orders move slots is baked into the same JSON), so exports
    can carry it without the state_vector being encoded.

    Invariant guaranteed here: every non-null ``action_index`` is legal under
    the same transition's ``action_mask``, and every non-null ``gimmick_index``
    is legal under its ``gimmick_mask``.  An action that cannot be resolved to a
    mask-legal index is stamped None (never a wrong index); its gimmick is then
    None too (a gimmick is a checkbox on an action we could not encode).  A
    decoded action whose gimmick bucket is not legal (a mega by a mon the mask
    says cannot mega) likewise drops the gimmick to None — never a wrong label.
    """
    snap = transition.get("state_before_actions") or {}
    our_active = snap.get("our_active") or {}
    mask = build_action_mask(snap)
    gimmick_mask = build_gimmick_mask(snap)
    for act in transition.get("our_actions") or []:
        slot = act.get("slot")
        idx = action_to_index(act, our_active.get(slot), snap)
        if idx is not None:
            row = mask.get(slot) or []
            if idx >= len(row) or row[idx] != 1:
                idx = None  # not expressible in the turn-start frame → unencoded
        act["action_index"] = idx

        # Gimmick (mega) is a checkbox on the chosen action — only meaningful
        # when that action is itself encodable and the gimmick bucket is legal.
        if idx is None:
            g = None
        else:
            g = action_to_gimmick(act)
            grow = gimmick_mask.get(slot) or []
            if g >= len(grow) or grow[g] != 1:
                g = None  # gimmick illegal under its mask (never on real data)
        act["gimmick_index"] = g

    transition["action_mask"] = mask
    transition["gimmick_mask"] = gimmick_mask
    return transition


# ── Opponent-perspective action codec (auxiliary head, task #9) ──────────────
# The aux opponent head predicts the OPPONENT's action from the same (our-
# perspective) state vector — a representation-shaping signal so the trunk
# captures opponent threats (e.g. for board-dependent mega timing).  It is
# TRAIN-ONLY (never served), so it lives entirely offline.
#
# Rather than fork the battle-tested codec, we FLIP the snapshot (swap our<->opp
# actives/benches + relabel slot keys) and reuse build_action_mask /
# action_to_index unchanged: from the opponent's seat, OUR mons are the foes and
# the opp bench holds the switch-ins.
OPP_HEADS: tuple[str, str] = ("opp_a", "opp_b")


def _relabel_slots(d: Optional[dict], frm: str, to: str) -> dict:
    return {(k.replace(frm, to, 1) if isinstance(k, str) else k): v
            for k, v in (d or {}).items()}


def _flip_perspective(snap: dict) -> dict:
    """A snapshot seen from the OPPONENT's seat: our<->opp actives/benches swapped
    and the active-slot keys relabeled (opp_a->our_a, our_a->opp_a, …).  Only the
    fields the action codec reads are populated."""
    return {
        "our_active": _relabel_slots(snap.get("opp_active"), "opp_", "our_"),
        "opp_active": _relabel_slots(snap.get("our_active"), "our_", "opp_"),
        "our_bench": snap.get("opp_bench") or [],
        "opp_bench": snap.get("our_bench") or [],
    }


def _flip_action_slots(action: dict) -> dict:
    """Swap our_<->opp_ in an action's ``slot`` / ``target_slot`` so an opp action
    reads correctly under the flipped snapshot."""
    out = dict(action)
    for field in ("slot", "target_slot"):
        v = out.get(field)
        if isinstance(v, str):
            if v.startswith("opp_"):
                out[field] = "our_" + v[4:]
            elif v.startswith("our_"):
                out[field] = "opp_" + v[4:]
    return out


def build_opp_action_mask(snap: dict) -> dict[str, list[int]]:
    """Decision-time legality mask for the OPPONENT's active slots:
    ``{"opp_a": [16], "opp_b": [16]}`` — build_action_mask over the flipped snap."""
    m = build_action_mask(_flip_perspective(snap))
    return {"opp_a": m.get("our_a") or [0] * ACTIONS_PER_SLOT,
            "opp_b": m.get("our_b") or [0] * ACTIONS_PER_SLOT}


def opp_action_to_index(action: dict, snap: dict) -> Optional[int]:
    """Encode one opp_actions_actual entry into the 0–15 codec from the
    opponent's perspective (our mons are its foes, opp bench its switch-ins)."""
    flipped = _flip_perspective(snap)
    fa = _flip_action_slots(action)
    mon = (flipped.get("our_active") or {}).get(fa.get("slot"))
    return action_to_index(fa, mon, flipped)


def annotate_opp_actions(transition: dict) -> dict:
    """Stamp ``opp_action_index`` on each opp_actions_actual entry + the
    ``opp_action_mask`` onto a transition, in place — the auxiliary opponent-head
    target/mask.  Same invariant as annotate_transition_actions: a non-null
    opp_action_index is always legal under opp_action_mask (else stamped None)."""
    snap = transition.get("state_before_actions") or {}
    opp_mask = build_opp_action_mask(snap)
    for act in transition.get("opp_actions_actual") or []:
        slot = act.get("slot")
        idx = opp_action_to_index(act, snap)
        if idx is not None:
            row = opp_mask.get(slot) or []
            if idx >= len(row) or row[idx] != 1:
                idx = None
        act["opp_action_index"] = idx
    transition["opp_action_mask"] = opp_mask
    return transition


# ── Convenience accessors ──────────────────────────────────────────────────────
def get_state_dim() -> int:
    return STATE_DIM

def get_action_dim() -> int:
    return ACTION_DIM

def get_gimmick_dim() -> int:
    return GIMMICK_DIM


# Backward-compatibility alias: the offline encoder kept the name
# StateEncoder before the live/vod split (live encode() now lives in
# live_state_encoder.LiveStateEncoder).
StateEncoder = VodStateEncoder
