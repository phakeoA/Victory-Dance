"""
State encoder for VGC Reg M-A (Pokémon Champions doubles format) — OFFLINE
(VOD-parsing) path + the FROZEN tensor layout shared by both paths.

This module is the canonical owner of the 1806-dim layout: every feature
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
        turn-order block (12) ← B1.4: 4 effective-speed + 4 moves-first margins + 4 pair-confidences
        team counts (4)      ← own/opp living-bench, own/opp fainted (each /4) — MUST stay last
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
POKEMON_FEATURES per slot (144 floats):
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
    item effects  (16)   ← gap #5; multi-hot over ITEM_EFFECT_NAMES (mechanics,
                           NOT identity — scalable across formats)
    item_known    (1)    ← 1.0 revealed/exact · 0.5 belief-top · 0.0 unknown
    ability effects(16)  ← gap #5; multi-hot over ABILITY_EFFECT_NAMES
    ability_known (1)    ← 1.0 revealed/exact · 0.5 belief-top · 0.0 unknown
    is_active     (1)
    is_revealed   (1)    ← always 1 for own mons
    is_fainted    (1)    ← layout-v2; KO flag (opp bench / counting)
    is_transformed(1)    ← layout-v2; Ditto copied a forme (types/stats above
                           are the COPY's); reverts on switch-out
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MOVE_FEATURES per move (22 floats):
    base_power / 150     (1)
    type_idx / 19        (1)   ← ordinal, not one-hot (keep dim small)
    category             (1)   ← 0=phys 0.5=spec 1=status
    priority / 7         (1)
    accuracy             (1)   ← 0–1
    pp_fraction          (1)   ← remaining PP / max (gap #7); max = base·8//5
    is_protect           (1)
    is_stab              (1)
    is_spread            (1)   ← gap #6; hits both foes (allAdjacentFoes/allAdjacent)
    type_eff vs enemy0/1 (4)   ← B1.1: signed log2(mult)/2 + immune (0× distinct from 0.25×), ×2 enemies
    damage vs enemy0/1   (4)   ← B1.2: [min,max] roll as frac of current HP, ×2 enemies (core: no situ. mods)
    intrinsics           (4)   ← B1.3: contact · recoil · drain · multihit-count/5 (per-move, public)
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
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ── Bootstrap: sibling imports work when run/imported from anywhere ────────────
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
from v_dance.parser.vod_parser.pokedex import get_pokedex, norm_species
from v_dance.parser.belief_state import BeliefState, dex_base_stats, STAT_ORDER

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

# ══════════════════════════════════════════════════════════════════════════════
# ITEM + ABILITY EFFECT CATEGORIES (gap #5) — FORMAT-SCALABLE, MECHANICS-BASED
# ══════════════════════════════════════════════════════════════════════════════
# We do NOT encode item/ability IDENTITY (a one-hot over the current meta's item
# list would not generalise to a future format's items, and an opaque ordinal id
# is a poor MLP feature).  Instead each mon's held item and ability are encoded as
# a multi-hot over their strategic EFFECT CATEGORIES — the game *mechanics*, which
# are stable across formats.  A brand-new Choice item in a later regulation is
# still ``choice``; a new weather-setter ability is still ``weather_setter``; so
# the LAYOUT never changes when the item/ability roster does, only the id→category
# membership tables below grow.  This is the SAME philosophy the encoder already
# uses for Pokémon themselves (types + base stats, never a species one-hot).
#
# The categories are FROZEN here (single source of tensor indices); both encode
# paths collapse their different raw forms — the offline display name ("Focus
# Sash") and poke-env's id ("focussash") — to the same id via norm_species and
# look up the SAME category function, so parity is at the EFFECT level, not the
# raw-string level.  Add a new id to the relevant ``_ITEM_*`` / ``_ABIL_*`` set
# (below, near the resolvers) to teach a new format's item/ability — no STATE_DIM
# change, no retrain of the layout.
ITEM_EFFECT_NAMES = [
    "has_item",         # holds any item at all (0 = itemless / Knock Off'd)
    "choice",           # Choice Band / Specs / Scarf — move-lock + stat boost
    "choice_speed",     # Choice Scarf specifically (1.5× speed)
    "focus_sash",       # survive a KO at 1 HP from full
    "life_orb",         # +30% damage, recoil
    "assault_vest",     # +50% SpD, status moves locked out
    "passive_recovery", # Leftovers / Black Sludge / Shell Bell — per-turn heal
    "heal_berry",       # Sitrus / pinch HP-restore berries (one-shot heal)
    "resist_berry",     # type-resist berries (Occa/Passho/…) — one-shot dmg cut
    "status_cure",      # Lum / Chesto / … — consumable status cure
    "booster_energy",   # Protosynthesis / Quark Drive trigger
    "contact_punish",   # Rocky Helmet — chip the attacker on contact
    "effect_shield",    # Covert Cloak / Safety Goggles — block secondary/spread
    "drop_guard",       # White Herb / Mental Herb / Clear Amulet — drop/taunt guard
    "luck_item",        # Bright Powder / Quick Claw / Scope Lens — rng items
    "type_boost",       # type-boosting held items (Charcoal/Mystic Water/plates…)
]
ABILITY_EFFECT_NAMES = [
    "intimidate",       # lower foes' Atk on switch-in
    "weather_setter",   # Drought / Drizzle / Sand Stream / Snow Warning / …
    "terrain_setter",   # Electric / Grassy / Misty / Psychic Surge
    "booster_ability",  # Protosynthesis / Quark Drive (paradox stat boost)
    "weather_speed",    # Swift Swim / Chlorophyll / Sand Rush / Slush Rush
    "speed_control",    # Speed Boost / Unburden (self speed escalation)
    "type_immunity",    # Levitate / Volt Absorb / Flash Fire / redirection / …
    "damage_boost",     # Adaptability / Tough Claws / Sheer Force / -ate / …
    "regenerator",      # heal on switch-out
    "prankster",        # +priority to status moves
    "magic_bounce",     # reflect status moves back
    "defensive_reduce", # Multiscale / Thick Fat / Filter / Ice Scales / …
    "status_immunity",  # Limber / Insomnia / Own Tempo / Purifying Salt / …
    "reactive_boost",   # Justified / Anger Point / Weak Armor / Berserk — buff on being HIT
    "mold_breaker",     # ignore the target's ability
    "guts_boost",       # Guts / Quick Feet / Marvel Scale — status-as-buff
    "statdrop_boost",   # Defiant / Competitive — offensive boost when a stat is LOWERED (Intimidate-punish)  (state-rep #B, layout v4)
]

_ITEM_EFFECT_IDX = {n: i for i, n in enumerate(ITEM_EFFECT_NAMES)}
_ABIL_EFFECT_IDX = {n: i for i, n in enumerate(ABILITY_EFFECT_NAMES)}

NUM_ITEM_EFFECTS    = len(ITEM_EFFECT_NAMES)      # 16  (v8 internal helpers — att_ctx life_orb/choice)
NUM_ABILITY_EFFECTS = len(ABILITY_EFFECT_NAMES)   # 17  (v8 internal helpers)
ITEM_FEATURES    = NUM_ITEM_EFFECTS + 1     # effect multi-hot + 1 known/confidence (v8 internal)
ABILITY_FEATURES = NUM_ABILITY_EFFECTS + 1  # effect multi-hot + 1 known/confidence (v8 internal)

# ── v9 (B1-mechanics): data-grounded mechanic substrate ──────────────────────────
# Tag NAME counts are FIXED (dex-independent) so the flat LAYOUT is stable; the per-mon identity INDEX is
# 1 float the model embeds; vocab sizes feed the model's nn.Embeddings (NOT the flat dim). No circular
# import — mechanic_tags/vocab/damage_mechanics never import the encoder.
from v_dance.encoders.mechanic_tags import (  # noqa: E402
    NUM_MOVE_TAGS, NUM_ABILITY_TAGS, NUM_ITEM_TAGS,
    move_tag_indices, ability_tag_indices, item_tag_indices, ABILITY_SUPPRESSED_IDX)
from v_dance.encoders.mechanic_vocab import (  # noqa: E402
    ability_index, move_index, item_index, ABILITY_VOCAB_SIZE, MOVE_VOCAB_SIZE, ITEM_VOCAB_SIZE)
from v_dance.encoders import damage_mechanics as _DMG  # noqa: E402

WEIGHT_FEATURES   = 1
VOLATILE_FEATURES = 6     # rooted, trapped, has_substitute, move_restricted, can_switch, locked_action
ITEM_BLOCK_V9     = NUM_ITEM_TAGS + 1 + 1      # tags + identity index + known
ABILITY_BLOCK_V9  = NUM_ABILITY_TAGS + 1 + 1   # tags + identity index + known

# ── Dimension constants ────────────────────────────────────────────────────────
NUM_TYPES      = len(TYPE_NAMES)        # 20
NUM_STATUS     = len(STATUS_NAMES)      # 7
NUM_WEATHER    = len(WEATHER_NAMES)     # 9
NUM_FIELDS     = len(FIELD_NAMES)       # 15
NUM_SIDE_CONDS = len(SIDE_COND_NAMES)   # 24
NUM_BOOSTS     = 7                      # atk def spa spd spe acc eva
NUM_MOVES      = 4
MOVE_FEATURES  = 21 + NUM_MOVE_TAGS + 1 + 1   # v9: 21 v8-core feats (base_power/type/category/priority/
                      # accuracy/pp/protect/stab/spread + type-eff×2 + damage-band×2 + intrinsics×4) +
                      # 29 effect-tags + 1 identity index + is_known(last) = 21+29+1+1 = 52

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
    + WEIGHT_FEATURES            # v9: normalized species weight (1)
    + NUM_MOVES * MOVE_FEATURES  # v9: 4 × 52
    + ITEM_BLOCK_V9              # v9: item tags + identity index + known
    + ABILITY_BLOCK_V9          # v9: ability tags + identity index + known
    + VOLATILE_FEATURES         # v9: per-mon volatile block (6)
    + 1             # is_active
    + 1             # is_revealed
    + 1             # is_fainted      (layout-v2: see/count KO'd mons)
    + 1             # is_transformed  (layout-v2: Ditto copies a forme; reverts)
)
# POKEMON_FEATURES = 1+20+20+6+6+1+1+1+7+7+40 + 17+18 + 1+1+1+1 = 149  (layout v4)
#   item block = ITEM_FEATURES 17 (16 effects + known); ability block =
#   ABILITY_FEATURES 18 (17 effects + known) after the Defiant/Competitive split.
# (item/ability blocks sit AFTER the move block and BEFORE the 4 trailing slot
#  flags, so the move-block offset _MOVE_BLOCK_REL and the move-perm augmentation
#  are untouched, and the flags stay last — addressed by negative offsets.)

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
    + 12              # B1.4 turn-order: 4 effective-speed + 4 moves-first margins + 4 pair-confidences
    + 11              # FIELD-DURATION (v10): turns-active age for weather/terrain/TR + per-side
                      #   (tailwind + reflect/light_screen/aurora_veil); age>base ⇒ rock/Light-Clay 8-turn
    + 4               # team counts (layout-v2): own/opp living-bench, own/opp fainted  (MUST stay last)
)  # = 101

STATE_DIM = (ACTIVE_SLOTS + BENCH_SLOTS + OPP_BENCH_SLOTS) * POKEMON_FEATURES + GLOBAL_FEATURES
# = 12 × 149 + 78 = 1866   (POKEMON_FEATURES 149 = 148 + 1×(NUM_ABILITY_EFFECTS 16→17))

# Monotonic layout version: bump whenever the tensor LAYOUT changes (a new
# feature block, a reorder, a dim change) so a checkpoint trained on an older
# layout can be REJECTED at load instead of silently matmul-mismatching.  History:
#   1 → STATE_DIM 1398 (layout-v2 + Ditto/Zoroark v2, pre item/ability)
#   2 → STATE_DIM 1806 (gap #5: per-mon item/ability effect-category blocks)
#   3 → STATE_DIM 1854 (gap #6 is_spread move flag; gap #7 real pp_fraction — no
#       dim change, a value not a slot)
#   4 → STATE_DIM 1866 (state-rep #B: Defiant/Competitive split into statdrop_boost)
#   8 → STATE_DIM 2454 (Epic B: type-eff/damage-band/intrinsics/turn-order; PF 197, MOVE_FEATURES 22)
#   9 → STATE_DIM 4398 (B1-mechanics: ability/move/item identity embeddings + expanded mechanic tags +
#       per-mon weight + volatile block + ability-type/stat/BP-aware damage band; PF 359, MOVE_FEATURES 52)
#  10 → STATE_DIM 4409 (field-duration: +11 turns-active age channels for weather/terrain/TR + per-side
#       tailwind/reflect/light_screen/aurora_veil; GLOBAL_FEATURES 90→101 — captures rock/Light-Clay 8-turn)
# train_bc stamps this into the checkpoint config; model_io.load_bc_policy asserts
# it (and the dim) match the running code.
STATE_LAYOUT_VERSION = 10

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

# data/moves.json — lazy-loaded for the offline path (v_dance/encoders/ → repo data/)
_MOVES_JSON_PATH = Path(__file__).resolve().parents[2] / "data" / "moves.json"
_moves_data: Optional[dict] = None


def _get_moves_data() -> dict:
    global _moves_data
    if _moves_data is None:
        try:
            _moves_data = json.loads(_MOVES_JSON_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            _moves_data = {}
    return _moves_data


# Spread moves (hit BOTH foes) — the core doubles tradeoff (gap #6): wider reach
# for 0.75× damage.  ``allAdjacentFoes`` = both foes (Heat Wave, Rock Slide);
# ``allAdjacent`` = both foes AND the ally (Earthquake, Surf).  Both count as
# spread (the model can read the ally-hit risk from is_protect / the board).
_SPREAD_TARGET_IDS = frozenset({"alladjacentfoes", "alladjacent"})


def is_spread_target(target) -> bool:
    """True for a move whose target hits both foes.  Accepts the offline camelCase
    string from data/moves.json ("allAdjacentFoes") OR a poke-env ``Target`` enum
    member — both normalised to a common id, so the two encode paths agree."""
    if target is None:
        return False
    name = getattr(target, "name", target)     # poke-env Target enum → its NAME
    return str(name).replace("_", "").lower() in _SPREAD_TARGET_IDS


def pp_max(base_pp: Optional[int]) -> int:
    """Max PP assuming 3 PP Ups — ``base_pp × 8 // 5`` — matching poke-env's
    ``Move.max_pp`` (gap #7), so the offline and live ``pp_fraction`` use the same
    denominator (e.g. Heat Wave base 10 → max 16)."""
    return (int(base_pp) * 8 // 5) if base_pp else 0


def _dex_types(species: Optional[str]) -> list[str]:
    """Dex types for a species as canonical names, e.g. ['DARK', 'STEEL']."""
    dex = get_pokedex()
    entry = dex.entry(species) if dex else None
    if not entry:
        return []
    return [_canon(t) for t in (entry.get("types") or [])]


# ── B1/#25 type-effectiveness (resolved cross — the encoder otherwise never learns the chart) ──
_TYPE_CHART_CACHE = None


def _type_chart():
    """poke-env's canonical Gen-9 chart {DEFENDING_TYPE: {ATTACKING_TYPE: multiplier}}, cached.
    Reg-independent game rule; identical on the offline + live encoder paths (parity)."""
    global _TYPE_CHART_CACHE
    if _TYPE_CHART_CACHE is None:
        from poke_env.data import GenData
        _TYPE_CHART_CACHE = GenData.from_gen(9).type_chart
    return _TYPE_CHART_CACHE


def _effective_types(mon: Optional[dict]) -> list[str]:
    """A mon's CURRENT effective types (canonical upper): TERA overrides (tera is prepared-for but
    INACTIVE in Reg M-A) else the dex types of the current/copied forme. The SINGLE source shared by
    the per-mon type one-hots AND the per-move type-eff, so the defender typing stays in lockstep."""
    if not mon:
        return []
    if mon.get("is_terastallized") and mon.get("known_tera_type"):
        return [_canon(mon["known_tera_type"])]
    dex_species = (mon["transformed_into"] if (mon.get("is_transformed") and mon.get("transformed_into"))
                   else mon.get("species"))
    return _dex_types(dex_species) or _dex_types(mon.get("base_species")) or []


def _type_mult(move_type: str, defender_types: list[str]) -> float:
    """Raw type-eff multiplier (product over the defender's types) of an attacking ``move_type``;
    1.0 when either side is unknown. Shared by the signed type-eff channel AND the damage band."""
    if not defender_types or not move_type:
        return 1.0
    chart = _type_chart()
    at = move_type.upper()
    mult = 1.0
    for dt in defender_types:
        mult *= float(chart.get(dt.upper(), {}).get(at, 1.0))
    return mult


def _type_eff_signed_immune(move_type: str, defender_types: list[str]) -> tuple[float, float]:
    """(signed_mult, immune) of ``move_type`` vs a defender. signed = clip(log2(mult)/2, -1, 1) for
    mult>0 → 4×=+1 · 2×=+0.5 · 1×=0 · 0.5×=-0.5 · 0.25×=-1; a 0× TYPING immunity (Ground→Flying, tera)
    → signed -1 AND immune 1, so 0× stays DISTINCT from 0.25×. Ability immunities (Levitate etc.)
    fold into the resolved damage band (B1.2), not here — this is the pure typing cross."""
    if not defender_types or not move_type:
        return 0.0, 0.0
    mult = _type_mult(move_type, defender_types)
    if mult <= 0.0:
        return -1.0, 1.0
    return float(np.clip(math.log2(mult) / 2.0, -1.0, 1.0)), 0.0


def _damage_band(base_power, A, D, hp_stat, hp_frac, type_mult, is_stab, is_spread,
                 situational_mult=1.0) -> tuple[float, float]:
    """(min, max) damage as a fraction of the defender's CURRENT HP, clamped [0,1] — the L50 Gen-9
    formula folding base_power·(A/D)·STAB·type_mult·spread·situational(B1.2b: weather/terrain/screens/
    item/burn)·the 0.85–1.0 roll. (0,0) for a non-damaging move or missing belief. Belief-dependent →
    rides stats_known. Defender-ability resists (Thick Fat/Multiscale/Filter) NOT modelled."""
    if (not base_power or base_power <= 0 or not A or not D
            or not hp_stat or not hp_frac or hp_frac <= 0):
        return 0.0, 0.0
    base = ((2 * 50 / 5 + 2) * base_power * A / D) / 50.0 + 2.0
    mult = type_mult * (1.5 if is_stab else 1.0) * (0.75 if is_spread else 1.0) * situational_mult
    cur_hp = hp_stat * hp_frac
    if cur_hp <= 0:
        return 0.0, 0.0
    return (float(np.clip(base * mult * 0.85 / cur_hp, 0.0, 1.0)),
            float(np.clip(base * mult / cur_hp, 0.0, 1.0)))


def _side_screens(side_dict: Optional[dict]) -> tuple:
    """(phys_reduced, spec_reduced) — does this side have a screen reducing physical / special damage
    (Reflect / Light Screen / Aurora Veil)? (B1.2b damage modifier.)"""
    screens = (side_dict or {}).get("screens") or {}
    # screens values are now turns-ACTIVE (elapsed; 0 on the set turn) — any key present is active.
    names = {_SCREEN_TO_SC.get(k) for k in screens}
    return (("REFLECT" in names or "AURORA_VEIL" in names),
            ("LIGHT_SCREEN" in names or "AURORA_VEIL" in names))


# v10 field-duration: the 11 turns-ACTIVE (age) channels, in fixed order, normalised to [0,1] by
# each condition's MAX duration. age > base ⇒ an item-EXTENDED instance (weather rock / Light Clay /
# Terrain Extender = 8 turns vs the 5/4 base). SHARED by the offline + live encoders so the age math
# is byte-identical; only the INPUTS differ (offline reads the parser's elapsed counters; live derives
# them from ``turn − poke-env start-turn``).
_SCREEN_AGE_KEYS = ("reflect", "light_screen", "aurora_veil")
# per-condition MAX duration (with the item extension) for normalising the age to [0,1].
_AGE_MAX = {"weather": 8.0, "terrain": 8.0, "trick_room": 5.0, "tailwind": 4.0, "screen": 8.0}


def field_duration_scalars(weather_age, terrain_age, tr_age, sides) -> list:
    """11 floats: [weather_age, terrain_age, trick_room_age, then per side (own, opp):
    tailwind_age, reflect_age, light_screen_age, aurora_veil_age].  ALL inputs are turns-ELAPSED
    (age, 0 on the set turn; None/absent → 0); each is normalised by its condition's max duration.
    ``sides`` = [own, opp], each a dict {'tailwind_age': int|None, 'screens': {name: elapsed}}.  The
    offline encoder converts its parser counters to elapsed; the live encoder uses turn − start-turn.
    Identical math on both paths."""
    out = [min((weather_age or 0) / _AGE_MAX["weather"], 1.0),
           min((terrain_age or 0) / _AGE_MAX["terrain"], 1.0),
           min((tr_age or 0) / _AGE_MAX["trick_room"], 1.0)]
    for side in sides:
        out.append(min(((side or {}).get("tailwind_age") or 0) / _AGE_MAX["tailwind"], 1.0))
        scr = (side or {}).get("screens") or {}
        for s in _SCREEN_AGE_KEYS:
            out.append(min((scr.get(s) or 0) / _AGE_MAX["screen"], 1.0) if s in scr else 0.0)
    return out


def _defender_profile(mon: Optional[dict], side_screens: tuple = (False, False)) -> Optional[dict]:
    """Uniform DEFENDER profile {types, def, spd, hp, hp_frac, grounded, screen_phys, screen_spec} for
    the per-move type-eff + damage cross. None for an empty slot. Belief est stats; hp_frac mirrors the
    per-mon hp encoding. grounded = static v1 (not Flying / Levitate / Air-Balloon; volatiles skipped)."""
    if not mon:
        return None
    est = (mon.get("stats_estimate") or {}).get("stats") or {}
    hp_pct = mon.get("hp_pct")
    if hp_pct is None:
        hp_frac = 0.0 if mon.get("seen", True) else 1.0
    else:
        hp_frac = max(0.0, min(hp_pct, 100.0)) / 100.0
    types = _effective_types(mon)
    grounded = ("FLYING" not in types and resolve_ability_json(mon)[0] != "levitate"
                and resolve_item_json(mon)[0] != "airballoon")
    return {"types": types, "def": est.get("def"), "spd": est.get("spd"), "hp": est.get("hp"),
            "hp_frac": hp_frac, "grounded": grounded,
            "screen_phys": side_screens[0], "screen_spec": side_screens[1]}


# B1.2b weather-speed abilities → the weather they double speed in.
_WEATHER_SPEED_ABILITY = {"swiftswim": ("RAINDANCE",), "chlorophyll": ("SUNNYDAY",),
                          "sandrush": ("SANDSTORM",), "slushrush": ("SNOW", "HAIL")}


def _situational_damage_mult(move_type, is_physical, weather, terrain, defender,
                             att_burned, att_life_orb, att_choice) -> float:
    """B1.2b damage situational multiplier: weather (Sun/Rain on Fire/Water) · terrain ×1.3 (grounded
    defender, matching type) · screens ×0.667 (doubles) · Life Orb ×1.3 · Choice Band/Specs ×1.5 ·
    burn ×0.5 (physical). Defender-ability resists (Thick Fat/Multiscale/Filter) NOT modelled here."""
    mult = 1.0
    mt = (move_type or "").upper()
    if weather == "SUNNYDAY":
        mult *= 1.5 if mt == "FIRE" else (0.5 if mt == "WATER" else 1.0)
    elif weather == "RAINDANCE":
        mult *= 1.5 if mt == "WATER" else (0.5 if mt == "FIRE" else 1.0)
    if defender and defender.get("grounded"):
        if (terrain == "ELECTRIC_TERRAIN" and mt == "ELECTRIC") \
                or (terrain == "GRASSY_TERRAIN" and mt == "GRASS") \
                or (terrain == "PSYCHIC_TERRAIN" and mt == "PSYCHIC"):
            mult *= 1.3
    if defender and ((is_physical and defender.get("screen_phys"))
                     or (not is_physical and defender.get("screen_spec"))):
        mult *= 0.667                                  # doubles screen reduction (≈2732/4096)
    if att_life_orb:
        mult *= 1.3
    if att_choice:
        mult *= 1.5                                    # Choice Band/Specs (not Scarf) boost the A stat
    if att_burned and is_physical:
        mult *= 0.5
    return mult


def _effective_speed(mon: Optional[dict], side_tailwind: bool,
                     weather: Optional[str] = None) -> tuple[float, float]:
    """(speed, known) — resolved in-battle speed folding the spe boost stage · paralysis ×0.5 · Choice
    Scarf ×1.5 · Tailwind ×2 · a weather-speed ability ×2 (Swift Swim/Chlorophyll/Sand Rush/Slush Rush
    in matching weather, B1.2b). None / no belief → (0, 0). Belief-dependent → formula-parity."""
    if not mon:
        return 0.0, 0.0
    est_block = mon.get("stats_estimate") or {}
    spe = (est_block.get("stats") or {}).get("spe")
    if not spe:
        return 0.0, 0.0
    known = {"exact": 1.0, "distribution": 0.5}.get(est_block.get("mode"), 0.0)
    stage = (mon.get("boosts") or {}).get("spe", 0) or 0
    spe *= ((2 + stage) / 2.0) if stage >= 0 else (2.0 / (2 - stage))
    if _canon(mon.get("status")) == "PAR":
        spe *= 0.5
    item_id, _ = resolve_item_json(mon)
    if _ITEM_EFFECT_IDX.get("choice_speed") in item_effect_indices(item_id):
        spe *= 1.5
    if side_tailwind:
        spe *= 2.0
    if weather and weather in _WEATHER_SPEED_ABILITY.get(resolve_ability_json(mon)[0], ()):
        spe *= 2.0
    return float(spe), known


def _moves_first(spd_a: float, spd_b: float, trick_room: bool) -> float:
    """Soft signed margin in [-1,1]: tanh(log(spd_a/spd_b)) with the Trick-Room sign-flip (positive →
    a moves first). 0 for a tie / missing speed — NEVER a hard ±1, so a surprise-Scarf / min-speed-TR
    read stays representable (the panel's over-anchoring guard)."""
    if not spd_a or not spd_b or spd_a <= 0 or spd_b <= 0:
        return 0.0
    m = math.tanh(math.log(spd_a / spd_b))
    return -m if trick_room else m


def dex_unique_ability(species: Optional[str]) -> Optional[str]:
    """The species' ability when it has exactly ONE possible ability, else None.

    A single-ability species' ability is public knowledge (fixed by the species,
    visible from team preview), so BOTH encode paths can know it without any
    in-battle reveal — and poke-env already sets ``mon.ability`` for such mons.
    Deriving it here from OUR pokedex keeps the offline path in lockstep with the
    live one (gap #5 parity): e.g. Zoroark-Hisui's only ability is Illusion."""
    if not species:
        return None
    dex = get_pokedex()
    entry = dex.entry(species) if dex else None
    abilities = (entry or {}).get("abilities") or {}
    distinct = list(dict.fromkeys(abilities.values()))
    return distinct[0] if len(distinct) == 1 else None


# ══════════════════════════════════════════════════════════════════════════════
# Item / ability id → EFFECT-CATEGORY resolvers (gap #5)
# ──────────────────────────────────────────────────────────────────────────────
# Membership tables keyed on the NORMALISED id (norm_species form: lowercase,
# alphanumerics only — "Focus Sash"→"focussash", matching poke-env's id).  To
# teach a new format's item/ability, add its id to the relevant set; the tensor
# LAYOUT (ITEM_EFFECT_NAMES / ABILITY_EFFECT_NAMES) never changes.  Sets are
# deliberately non-exhaustive of every item that ever existed — they cover the
# Reg M-A meta and the evergreen staples; unknown ids simply set only ``has_item``
# (still strictly more than the zero the net saw before).
_ITEM_NONE_IDS   = frozenset({"", "nothing", "none", "noitem", "unknownitem", "unknown_item"})
_RECOVERY_ITEMS  = frozenset({"leftovers", "blacksludge", "shellbell"})
_HEAL_BERRIES    = frozenset({
    "sitrusberry", "oranberry", "aguavberry", "figyberry", "wikiberry",
    "magoberry", "iapapaberry", "enigmaberry",
})
_RESIST_BERRIES  = frozenset({
    "occaberry", "passhoberry", "wacanberry", "rindoberry", "yacheberry",
    "chopleberry", "kebiaberry", "shucaberry", "cobaberry", "payapaberry",
    "tangaberry", "chartiberry", "kasibberry", "habanberry", "colburberry",
    "babiriberry", "chilanberry", "roseliberry",
})
_STATUS_CURE_BERRIES = frozenset({
    "lumberry", "chestoberry", "cheriberry", "pechaberry", "rawstberry",
    "aspearberry", "persimberry", "mintberry",
})
_CONTACT_PUNISH_ITEMS = frozenset({"rockyhelmet"})
_EFFECT_SHIELD_ITEMS  = frozenset({"covertcloak", "safetygoggles"})
_DROP_GUARD_ITEMS     = frozenset({"whiteherb", "mentalherb", "clearamulet"})
_LUCK_ITEMS           = frozenset({
    "brightpowder", "quickclaw", "scopelens", "widelens", "zoomlens",
    "laggingtail", "focusband", "kingsrock", "razorclaw", "razorfang",
})
_TYPE_BOOST_ITEMS     = frozenset({
    "charcoal", "mysticwater", "blackbelt", "magnet", "miracleseed",
    "nevermeltice", "dragonfang", "blackglasses", "sharpbeak", "poisonbarb",
    "softsand", "hardstone", "silkscarf", "spelltag", "twistedspoon",
    "metalcoat", "silverpowder", "oddincense", "rockincense", "seaincense",
    "roseincense", "waveincense", "fairyfeather", "pixieplate",
})


def item_effect_indices(item_id: Optional[str]) -> list[int]:
    """ITEM_EFFECT_NAMES indices set for a normalised item id.

    Empty / itemless / unknown → ``[]`` (no flags, including no ``has_item``).
    Any concrete item sets ``has_item`` plus every effect category it belongs to.
    A type-boosting *plate* (id ending in 'plate') counts as ``type_boost`` even
    if not in the explicit set, so new plates need no table edit.
    """
    iid = (item_id or "")
    if iid in _ITEM_NONE_IDS:
        return []
    out = [_ITEM_EFFECT_IDX["has_item"]]
    if iid.startswith("choice"):
        out.append(_ITEM_EFFECT_IDX["choice"])
        if iid == "choicescarf":
            out.append(_ITEM_EFFECT_IDX["choice_speed"])
    if iid == "focussash":
        out.append(_ITEM_EFFECT_IDX["focus_sash"])
    if iid == "lifeorb":
        out.append(_ITEM_EFFECT_IDX["life_orb"])
    if iid == "assaultvest":
        out.append(_ITEM_EFFECT_IDX["assault_vest"])
    if iid in _RECOVERY_ITEMS:
        out.append(_ITEM_EFFECT_IDX["passive_recovery"])
    if iid in _HEAL_BERRIES:
        out.append(_ITEM_EFFECT_IDX["heal_berry"])
    if iid in _RESIST_BERRIES:
        out.append(_ITEM_EFFECT_IDX["resist_berry"])
    if iid in _STATUS_CURE_BERRIES:
        out.append(_ITEM_EFFECT_IDX["status_cure"])
    if iid == "boosterenergy":
        out.append(_ITEM_EFFECT_IDX["booster_energy"])
    if iid in _CONTACT_PUNISH_ITEMS:
        out.append(_ITEM_EFFECT_IDX["contact_punish"])
    if iid in _EFFECT_SHIELD_ITEMS:
        out.append(_ITEM_EFFECT_IDX["effect_shield"])
    if iid in _DROP_GUARD_ITEMS:
        out.append(_ITEM_EFFECT_IDX["drop_guard"])
    if iid in _LUCK_ITEMS:
        out.append(_ITEM_EFFECT_IDX["luck_item"])
    if iid in _TYPE_BOOST_ITEMS or iid.endswith("plate") or iid.endswith("gem"):
        out.append(_ITEM_EFFECT_IDX["type_boost"])
    return out


_INTIMIDATE_AB   = frozenset({"intimidate"})
_WEATHER_SET_AB  = frozenset({
    "drought", "drizzle", "sandstream", "snowwarning", "desolateland",
    "primordialsea", "deltastream", "orichalcumpulse",
})
_TERRAIN_SET_AB  = frozenset({
    "electricsurge", "grassysurge", "mistysurge", "psychicsurge", "hadronengine",
})
_BOOSTER_AB      = frozenset({"protosynthesis", "quarkdrive"})
_WEATHER_SPEED_AB = frozenset({"swiftswim", "chlorophyll", "sandrush", "slushrush"})
_SPEED_CTRL_AB   = frozenset({"speedboost", "unburden"})
_TYPE_IMMUNE_AB  = frozenset({
    "levitate", "voltabsorb", "waterabsorb", "flashfire", "sapsipper",
    "lightningrod", "stormdrain", "motordrive", "dryskin", "eartheater",
    "wellbakedbody", "windrider", "purifyingsalt",
})
_DMG_BOOST_AB    = frozenset({
    "adaptability", "toughclaws", "sheerforce", "technician", "ironfist",
    "strongjaw", "reckless", "pixilate", "refrigerate", "aerilate",
    "galvanize", "steelworker", "punkrock", "sharpness", "rockypayload",
    "transistor", "dragonsmaw", "steelyspirit", "hugepower", "purepower",
    "hustle", "guts", "tintedlens",
})
_REGENERATOR_AB  = frozenset({"regenerator"})
_PRANKSTER_AB    = frozenset({"prankster"})
_MAGIC_BOUNCE_AB = frozenset({"magicbounce"})
_DEF_REDUCE_AB   = frozenset({
    "multiscale", "shadowshield", "thickfat", "filter", "solidrock",
    "prismarmor", "fluffy", "icescales", "furcoat", "heatproof", "wonderguard",
})
_STATUS_IMMUNE_AB = frozenset({
    "limber", "insomnia", "vitalspirit", "waterveil", "magmaarmor", "immunity",
    "owntempo", "innerfocus", "sweetveil", "leafguard", "pastelveil",
    "thermalexchange", "goodasgold", "shieldsdown",
})
_REACTIVE_AB     = frozenset({
    "justified", "angerpoint", "berserk", "weakarmor",
    "rattled", "stamina", "watercompaction", "electromorphosis", "windpower",
    "angershell", "sandspit", "seedsower", "toxicdebris",
})
# Defiant / Competitive boost an OFFENSIVE stat (Atk / SpA) when ANY of the holder's
# stats is LOWERED — the Intimidate-punish abilities.  SPLIT out of _REACTIVE_AB
# (which is buff-on-being-HIT) into their own category so the model can tell
# "punishes my stat-drops / Intimidate" apart from "buffs when I land a hit".
# (state-rep #B, layout v4.)
_STATDROP_AB     = frozenset({"defiant", "competitive"})
_MOLD_BREAKER_AB = frozenset({"moldbreaker", "teravolt", "turboblaze"})
_GUTS_AB         = frozenset({"guts", "quickfeet", "marvelscale", "flareboost", "toxicboost"})


def ability_effect_indices(ability_id: Optional[str]) -> list[int]:
    """ABILITY_EFFECT_NAMES indices set for a normalised ability id (``[]`` for
    empty/unknown).  An ability can belong to several categories (e.g. Guts is
    both a damage_boost and a guts_boost)."""
    aid = (ability_id or "")
    if not aid:
        return []
    out: list[int] = []
    if aid in _INTIMIDATE_AB:    out.append(_ABIL_EFFECT_IDX["intimidate"])
    if aid in _WEATHER_SET_AB:   out.append(_ABIL_EFFECT_IDX["weather_setter"])
    if aid in _TERRAIN_SET_AB:   out.append(_ABIL_EFFECT_IDX["terrain_setter"])
    if aid in _BOOSTER_AB:       out.append(_ABIL_EFFECT_IDX["booster_ability"])
    if aid in _WEATHER_SPEED_AB: out.append(_ABIL_EFFECT_IDX["weather_speed"])
    if aid in _SPEED_CTRL_AB:    out.append(_ABIL_EFFECT_IDX["speed_control"])
    if aid in _TYPE_IMMUNE_AB:   out.append(_ABIL_EFFECT_IDX["type_immunity"])
    if aid in _DMG_BOOST_AB:     out.append(_ABIL_EFFECT_IDX["damage_boost"])
    if aid in _REGENERATOR_AB:   out.append(_ABIL_EFFECT_IDX["regenerator"])
    if aid in _PRANKSTER_AB:     out.append(_ABIL_EFFECT_IDX["prankster"])
    if aid in _MAGIC_BOUNCE_AB:  out.append(_ABIL_EFFECT_IDX["magic_bounce"])
    if aid in _DEF_REDUCE_AB:    out.append(_ABIL_EFFECT_IDX["defensive_reduce"])
    if aid in _STATUS_IMMUNE_AB: out.append(_ABIL_EFFECT_IDX["status_immunity"])
    if aid in _REACTIVE_AB:      out.append(_ABIL_EFFECT_IDX["reactive_boost"])
    if aid in _STATDROP_AB:      out.append(_ABIL_EFFECT_IDX["statdrop_boost"])
    if aid in _MOLD_BREAKER_AB:  out.append(_ABIL_EFFECT_IDX["mold_breaker"])
    if aid in _GUTS_AB:          out.append(_ABIL_EFFECT_IDX["guts_boost"])
    return out


def resolve_item_json(mon: dict) -> tuple[str, float]:
    """(normalised item id, confidence) for an OFFLINE mon dict.

    A revealed/injected ``known_item`` (incl. the ``"mega stone"`` placeholder the
    parser stamps on a mega'd mon) wins at confidence 1.0; an exact (team-sheet)
    mon with no item is a confirmed-itemless 1.0; else the belief top item
    (belief["items"][0]) at 0.5; else unknown ("", 0.0).

    A CONSUMED item (item_consumed; berry eaten / Sash popped / Knocked Off) is
    no longer held → unknown ("", 0.0), matching poke-env's nulled mon.item — the
    historical identity stays in known_item for belief/UI but is not encoded as a
    held item."""
    if mon.get("item_consumed"):
        return "", 0.0
    ki = mon.get("known_item")
    if ki:
        return norm_species(ki), 1.0
    if mon.get("exact"):
        return "", 1.0                      # team sheet vouches: itemless
    items = (mon.get("belief") or {}).get("items") or []
    if items and items[0].get("name"):
        return norm_species(items[0]["name"]), 0.5
    return "", 0.0


def resolve_ability_json(mon: dict) -> tuple[str, float]:
    """(normalised ability id, confidence) for an OFFLINE mon dict — the
    USER-CHOOSABLE ability (Bug 8 mega split): ``pre_mega_ability`` when present,
    else ``known_ability`` for a non-mega mon (a mega'd mon's ``known_ability`` is
    the fixed mega ability, NOT the chosen one, so it is ignored); else the belief
    top ability (which fill_blanks derives from the BASE forme for a mega) at 0.5;
    else unknown."""
    abil = mon.get("pre_mega_ability") or (
        None if mon.get("is_mega") else mon.get("known_ability")
    )
    if abil:
        return norm_species(abil), 1.0
    # Single-ability species → publicly known (poke-env sets it live too).
    if not mon.get("is_mega"):
        uniq = dex_unique_ability(mon.get("base_species") or mon.get("species"))
        if uniq:
            return norm_species(uniq), 1.0
    abilities = (mon.get("belief") or {}).get("abilities") or []
    if abilities and abilities[0].get("name"):
        return norm_species(abilities[0]["name"]), 0.5
    return "", 0.0


def resolve_active_ability_json(mon: dict) -> tuple[str, float]:
    """The CURRENTLY-ACTIVE ability for the v9 ability block (tags + identity). For a MEGA'd mon this is
    the mega forme's FIXED ability (Shadow Tag for Mega Gengar, Parental Bond for Mega Kangaskhan) at
    confidence 1.0 — NOT the user-chosen base ability — so mega abilities (trapping / parental_bond / …)
    are correctly tagged. Non-mega → resolve_ability_json (the chosen ability). Mirrors the live path,
    which reads poke-env's mon.ability (the mega forme ability) for a mega'd mon, so the index + conf agree
    on both paths (resolving the v8 mega-clamp 1.0-vs-0.5 asymmetry)."""
    if mon.get("is_mega"):
        active = mon.get("mega_ability") or mon.get("known_ability")
        if active:
            return norm_species(active), 1.0
    return resolve_ability_json(mon)


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
        # B1/B1.2b: resolve enemy-active DEFENDER PROFILES (types/stats/HP/grounded/screens) + the global
        # field mods (weather/terrain) ONCE — own mons attack the opp actives & vice-versa.
        _field = snap.get("field") or {}
        _wraw = _field.get("weather")
        _weather = _canon(_wraw) if _wraw else None
        _terrain = _TERRAIN_TO_FIELD.get(_field.get("terrain") or "")
        field_mods = (_weather, _terrain)
        _sc = snap.get("side_conditions") or {}
        _our_scr = _side_screens(_sc.get("our_side"))
        _opp_scr = _side_screens(_sc.get("opp_side"))
        own_enemy = [_defender_profile(opp_active.get("opp_a"), _opp_scr),
                     _defender_profile(opp_active.get("opp_b"), _opp_scr)]
        opp_enemy = [_defender_profile(our_active.get("our_a"), _our_scr),
                     _defender_profile(our_active.get("our_b"), _our_scr)]
        for key_map, prefix in ((our_active, "our"), (opp_active, "opp")):
            enemy = own_enemy if prefix == "our" else opp_enemy
            for slot in ("a", "b"):
                mon = key_map.get(f"{prefix}_{slot}")
                self._write_pokemon_json(vec, cursor, mon, is_active=True, enemy_defenders=enemy, field_mods=field_mods)
                cursor += POKEMON_FEATURES

        # ── [B] Own bench (fainted excluded — matches the live path; the
        # snapshot keeps fainted mons by design, see replay_parser Bug 6) ────
        bench = [
            m for m in (snap.get("our_bench") or []) if not m.get("is_fainted")
        ][:BENCH_SLOTS]
        for i in range(BENCH_SLOTS):
            mon = bench[i] if i < len(bench) else None
            self._write_pokemon_json(vec, cursor, mon, is_active=False, enemy_defenders=own_enemy,
                                     field_mods=field_mods)
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
            self._write_pokemon_json(vec, cursor, mon, is_active=False, enemy_defenders=opp_enemy,
                                     field_mods=field_mods)
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
            for screen in (sc.get("screens") or {}):      # values = turns-active; any key = present
                name = _SCREEN_TO_SC.get(screen)
                if name:
                    vec[cursor + _SC_IDX[name]] = 1.0
            cursor += NUM_SIDE_CONDS

        # Turn (cap at 60 for normalisation)
        vec[cursor] = min(turn, 60) / 60.0
        cursor += 1

        # Trick Room explicit flag
        vec[cursor] = 1.0 if trick_room else 0.0
        cursor += 1

        # ── B1.4 turn-order block: 4 effective speeds + 4 moves-first margins + 4 pair-confidences.
        # Folds spe boost/para/Scarf/Tailwind into a resolved speed and the TR sign-flip into the
        # who-moves-first comparison. Belief-dependent (est speed) → formula-parity like damage.
        _our_tw = ((side_conds.get("our_side") or {}).get("tailwind_turns_remaining") or 0) > 0
        _opp_tw = ((side_conds.get("opp_side") or {}).get("tailwind_turns_remaining") or 0) > 0
        _sp = {
            "our_a": _effective_speed(our_active.get("our_a"), _our_tw, _weather),
            "our_b": _effective_speed(our_active.get("our_b"), _our_tw, _weather),
            "opp_a": _effective_speed(opp_active.get("opp_a"), _opp_tw, _weather),
            "opp_b": _effective_speed(opp_active.get("opp_b"), _opp_tw, _weather),
        }
        for _k in ("our_a", "our_b", "opp_a", "opp_b"):      # 4 effective-speed channels (/600 clamped)
            vec[cursor] = min(_sp[_k][0] / 600.0, 1.0)
            cursor += 1
        for _ok in ("our_a", "our_b"):                        # 4 our×opp pairs × [margin, confidence]
            for _pk in ("opp_a", "opp_b"):
                vec[cursor] = _moves_first(_sp[_ok][0], _sp[_pk][0], trick_room)
                vec[cursor + 1] = _sp[_ok][1] * _sp[_pk][1]
                cursor += 2

        # ── Field-duration block (11, v10): turns-active AGE per condition. age > base ⇒ an
        # item-extended (weather rock / Light Clay / Terrain Extender) 8-turn instance. Convert the
        # parser's REMAINING counters (TR/tailwind) to elapsed so the math matches the live path.
        _tr_rem = field.get("trick_room_turns_remaining") or 0
        _fd_sides = []
        for _sk in ("our_side", "opp_side"):
            _s = side_conds.get(_sk) or {}
            _tw_rem = _s.get("tailwind_turns_remaining") or 0
            _fd_sides.append({"tailwind_age": (4 - _tw_rem) if _tw_rem > 0 else 0,
                              "screens": _s.get("screens")})
        for _v in field_duration_scalars(field.get("weather_turns_active"),
                                         field.get("terrain_turns_active"),
                                         (5 - _tr_rem) if _tr_rem > 0 else 0, _fd_sides):
            vec[cursor] = _v
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
        enemy_defenders: Optional[list] = None,
        field_mods: tuple = (None, None),
    ) -> None:
        """JSON twin of _write_pokemon — same POKEMON_FEATURES layout. ``enemy_defenders`` = the 2 enemy
        actives' defender profiles; ``field_mods`` = (weather, terrain) for the B1.2b damage modifiers."""
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

        # Types: tera overrides; otherwise dex types of the current/copied forme.
        # _effective_types is the SINGLE source also used by the per-move type-eff (lockstep).
        types = _effective_types(mon)
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

        # v9: normalized species weight (weight-based moves Heavy Slam/Low Kick + Heavy/Light Metal)
        _wt = _DMG.species_weight(dex_species) or _DMG.species_weight(base_fallback) or 0.0
        vec[i] = min(_wt / 500.0, 1.0)
        i += 1

        # Resolve the (current) ACTIVE ability once — needed for the move-type fix (Normalize/Pixilate)
        # AND the ability block below. Mega'd mons use the mega forme ability (so Mega Gengar = Shadow Tag).
        abil_id, abil_known = resolve_active_ability_json(mon)
        vol = mon.get("volatiles") or {}

        # Moves: canonical slot order shared with the action codec — see move_slots_for_mon().
        pp_used_map = mon.get("move_pp_used") or {}
        # B1.2/B1.2b attacker context: A stat + burn + Life-Orb/Choice item (fold into the damage band).
        _est = (mon.get("stats_estimate") or {}).get("stats") or {}
        _aidx = item_effect_indices(resolve_item_json(mon)[0])
        att_ctx = {"atk": _est.get("atk"), "spa": _est.get("spa"),
                   "burned": _canon(mon.get("status")) == "BRN",
                   "life_orb": _ITEM_EFFECT_IDX.get("life_orb") in _aidx,
                   "choice": (_ITEM_EFFECT_IDX.get("choice") in _aidx
                              and _ITEM_EFFECT_IDX.get("choice_speed") not in _aidx)}
        slots = move_slots_for_mon(mon)
        for m_idx in range(NUM_MOVES):
            if m_idx < len(slots):
                name, confidence = slots[m_idx]
                pp_used = pp_used_map.get(norm_species(name), 0)
                self._write_move_json(vec, i, name, confidence, types, pp_used,
                                      enemy_defenders, att_ctx, field_mods, ability_id=abil_id)
            i += MOVE_FEATURES

        # ── v9 ITEM block: identity index + effect-tag multi-hot + known ──
        item_id, item_known = resolve_item_json(mon)
        vec[i] = float(item_index(item_id))
        i += 1
        for idx in item_tag_indices(item_id):
            vec[i + idx] = 1.0
        i += NUM_ITEM_TAGS
        vec[i] = item_known
        i += 1

        # ── v9 ABILITY block: identity index + effect-tag multi-hot + known. Gastro Acid / Neutralizing
        # Gas suppress the FUNCTIONAL tags but keep the identity (+ set the ability_suppressed tag). ──
        vec[i] = float(ability_index(abil_id))
        i += 1
        if vol.get("ability_suppressed"):
            vec[i + ABILITY_SUPPRESSED_IDX] = 1.0
        else:
            for idx in ability_tag_indices(abil_id):
                vec[i + idx] = 1.0
        i += NUM_ABILITY_TAGS
        vec[i] = abil_known
        i += 1

        # ── v9 VOLATILE block: rooted, trapped(move), has_substitute, move_restricted, can_switch,
        # locked_action.  can_switch = NOT(move-trapped OR rooted); ability-trapping (Shadow Tag) is a
        # documented follow-up (the opponent's `trapping` tag already lets the net learn the interaction). ──
        vec[i] = 1.0 if vol.get("rooted") else 0.0
        vec[i + 1] = 1.0 if vol.get("trapped") else 0.0
        vec[i + 2] = 1.0 if vol.get("has_substitute") else 0.0
        vec[i + 3] = 1.0 if vol.get("move_restricted") else 0.0
        vec[i + 4] = 0.0 if (vol.get("trapped") or vol.get("rooted")) else 1.0
        vec[i + 5] = 1.0 if vol.get("locked_action") else 0.0
        i += VOLATILE_FEATURES

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
        pp_used: int = 0,
        enemy_defenders: Optional[list] = None,
        att_ctx: Optional[dict] = None,
        field_mods: tuple = (None, None),
        ability_id: Optional[str] = None,
    ) -> None:
        """JSON twin of _write_move using data/moves.json. ``enemy_defenders`` = the 2 enemy actives'
        defender profiles; ``att_ctx`` = attacker stats/burn/item; ``field_mods`` = (weather, terrain);
        ``ability_id`` = the user's current ability (v9: effective-type change Normalize/Pixilate/-ate)."""
        i = start
        data = _get_moves_data().get(norm_species(move_name))
        if not data:
            # Unknown move id: only the is_known confidence is encoded
            vec[start + MOVE_FEATURES - 1] = confidence
            return

        vec[i] = min(data.get("basePower") or 0, 250) / 150.0
        i += 1

        mtype = _canon(data.get("type"))
        mtype = _DMG.effective_move_type(move_name, mtype, ability_id)   # v9: Normalize/Pixilate/-ate/Liquid Voice
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

        # PP fraction (gap #7): remaining PP / max, where max = base·8//5 (3 PP
        # Ups) to match poke-env's Move.max_pp.  pp_used is the parser's count of
        # this mon's self-selected uses (move_pp_used); a belief-padded / unused
        # move has pp_used 0 → 1.0.  Moves with no PP data fall back to 1.0.
        mx = pp_max(data.get("pp"))
        vec[i] = (max(0, mx - pp_used) / mx) if mx else 1.0
        i += 1

        vec[i] = 1.0 if norm_species(move_name) in _PROTECT_MOVES else 0.0
        i += 1

        vec[i] = 1.0 if mtype in user_types else 0.0
        i += 1

        # is_spread (gap #6): hits both foes (the core doubles tradeoff).
        _spread = is_spread_target(data.get("target"))
        vec[i] = 1.0 if _spread else 0.0
        i += 1

        # B1.1 type-eff cross: signed multiplier + immune flag of THIS move vs each of the 2 enemy
        # actives (the resolved chart lookup the net otherwise never learns). 4 channels = 2×[signed,immune].
        for e in range(2):
            d = enemy_defenders[e] if (enemy_defenders and e < len(enemy_defenders)) else None
            signed, immune = _type_eff_signed_immune(mtype, (d.get("types") if d else []))
            vec[i] = signed
            vec[i + 1] = immune
            i += 2

        # B1.2 damage band + B1.2b situational mods: [min, max] roll as a fraction of each enemy
        # active's CURRENT HP. 4 channels = 2×[min,max].
        _phys = (data.get("category") or "").lower() == "physical"
        _ac = att_ctx or {}
        _A = _ac.get("atk") if _phys else _ac.get("spa")
        _stab = mtype in user_types
        _weather, _terrain = field_mods if field_mods else (None, None)
        for e in range(2):
            d = enemy_defenders[e] if (enemy_defenders and e < len(enemy_defenders)) else None
            if d:
                _sit = _situational_damage_mult(mtype, _phys, _weather, _terrain, d,
                                                _ac.get("burned"), _ac.get("life_orb"), _ac.get("choice"))
                dmin, dmax = _damage_band(
                    data.get("basePower") or 0, _A, (d.get("def") if _phys else d.get("spd")),
                    d.get("hp"), d.get("hp_frac"), _type_mult(mtype, d.get("types") or []),
                    _stab, _spread, _sit)
            else:
                dmin, dmax = 0.0, 0.0
            vec[i] = dmin
            vec[i + 1] = dmax
            i += 2

        # B1.3 move intrinsics (per-move, public, from moves.json) — contact crosses with the
        # Rocky-Helmet/Static/Flame-Body tags the encoder already has; recoil/drain shift the HP
        # math the value head reasons over; multihit re-procs Life-Orb/Tough-Claws + ignores Sash.
        flags = data.get("flags") or {}
        vec[i] = 1.0 if flags.get("contact") else 0.0
        vec[i + 1] = 1.0 if data.get("recoil") else 0.0
        vec[i + 2] = 1.0 if data.get("drain") else 0.0
        _mh = data.get("multihit")
        _mh_max = (_mh[-1] if isinstance(_mh, (list, tuple)) and _mh
                   else (_mh if isinstance(_mh, int) else 0))
        vec[i + 3] = min(_mh_max or 0, 5) / 5.0
        i += 4

        # v9: move effect-tags (multi-hot priors) + identity index, BEFORE the trailing is_known
        for idx in move_tag_indices(move_name):
            vec[i + idx] = 1.0
        i += NUM_MOVE_TAGS
        vec[i] = float(move_index(move_name))
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
_WEIGHT_REL = 1 + 2 * NUM_TYPES + 6 + 6 + 1 + 1 + 1 + NUM_STATUS + NUM_BOOSTS  # 70 (v9 weight feature)
_MOVE_BLOCK_REL = _WEIGHT_REL + WEIGHT_FEATURES  # 71 (v9: moves start after weight)

# v9 within-mon-row offsets of the IDENTITY-INDEX columns (the model extracts + embeds these; never a
# Linear input). Computed from the block sizes so encoder + model can't drift.
_ITEM_BLOCK_REL    = _MOVE_BLOCK_REL + NUM_MOVES * MOVE_FEATURES   # 279
ITEM_ID_REL        = _ITEM_BLOCK_REL                              # item identity index
_ABILITY_BLOCK_REL = _ITEM_BLOCK_REL + ITEM_BLOCK_V9              # 304
ABILITY_ID_REL     = _ABILITY_BLOCK_REL                           # ability identity index
ABILITY_KNOWN_REL  = _ABILITY_BLOCK_REL + 1 + NUM_ABILITY_TAGS    # ability known/confidence (conf-scales the emb)
# per-move identity index = 2nd-to-last float in each MOVE_FEATURES block (before is_known)
MOVE_ID_RELS       = tuple(_MOVE_BLOCK_REL + m * MOVE_FEATURES + (MOVE_FEATURES - 2) for m in range(NUM_MOVES))
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

def get_state_layout_version() -> int:
    return STATE_LAYOUT_VERSION

def get_action_dim() -> int:
    return ACTION_DIM

def get_gimmick_dim() -> int:
    return GIMMICK_DIM


# Backward-compatibility alias: the offline encoder kept the name
# StateEncoder before the live/vod split (live encode() now lives in
# live_state_encoder.LiveStateEncoder).
StateEncoder = VodStateEncoder
