"""Frozen tensor LAYOUT: dims, vocab, index maps, action-space + gimmick constants.

AUTO-SPLIT from state_encoder.py (v17 refactor) — PURE verbatim moves; the
frozen tensor layout + parity are unchanged.  state_encoder.py re-exports every
name moved here, so all existing imports keep working.
"""
from __future__ import annotations

from v_dance.encoders.mechanic_tags import NUM_ABILITY_TAGS, NUM_ITEM_TAGS, NUM_MOVE_TAGS


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

WEIGHT_FEATURES   = 1
VOLATILE_FEATURES = 12    # rooted, trapped, has_substitute, move_restricted, can_switch, locked_action,
                          # protect_counter (A.3), residual_damage/confused/perish_norm (C.2), drowsy (C.2b),
                          # first_turn (C.3: Fake Out / First Impression / Mat Block legal this turn)
_PROTECT_COUNTER_CAP = 3.0   # v11 A.3: success decays ~1/3ⁿ; clamp to [0,1] via min(c,CAP)/CAP. SHARED
                             # byte-identically by the offline + live encoders (a mismatch breaks parity).
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
MOVE_FEATURES  = 28 + NUM_MOVE_TAGS + 1 + 1   # v9/v11: 25 core feats (base_power/type/category/priority/
                      # accuracy/pp/protect/stab/spread + type-eff×2 + damage-band×2 + MOVES-FIRST×2 [v11 B.1b
                      # priority-aware who-moves-first vs each enemy] + intrinsics×4 + HIT-CHANCE×2 [v11 B2b
                      # per-enemy realized accuracy: defender evasion stage/Sand-Veil/No-Guard/acc−eva boost])
                      # + 1 REDUNDANT-CONDITION bit (v18: this move re-sets an already-active screen/weather/
                      # terrain on the relevant side) + 2 REDUNDANT-STATUS bits (v19/Option 1c: per-enemy —
                      # this PURE status move is wasted on enemy[e] (already-statused / type-or-ability immune))
                      # + 29 effect-tags + 1 identity index + is_known(last) = 28+29+1+1 = 59

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
    + NUM_MOVES * MOVE_FEATURES  # v9: 4 × MOVE_FEATURES (v18 = 57)
    + ITEM_BLOCK_V9              # v9: item tags + identity index + known
    + ABILITY_BLOCK_V9          # v9: ability tags + identity index + known
    + VOLATILE_FEATURES         # v9: per-mon volatile block (6)
    + NUM_TYPES     # v11 Phase D (D9): tera_type one-hot (20) — REVEALED tera type; 0 while tera mod-disabled
    + 1             # is_active
    + 1             # is_revealed
    + 1             # is_fainted      (layout-v2: see/count KO'd mons)
    + 1             # is_transformed  (layout-v2: Ditto copies a forme; reverts)
)
# POKEMON_FEATURES sums the blocks above (v18 = 405; full dim history in the
#   STATE_LAYOUT_VERSION block below).
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
# Current STATE_DIM = 4961 @ layout v18 (= 12 × POKEMON_FEATURES + GLOBAL_FEATURES);
# see the STATE_LAYOUT_VERSION history below for every prior layout.

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
#  11 → STATE_DIM 4421 (v11 A.3: +1 per-mon volatile channel = consecutive-Protect counter; VOLATILE_FEATURES
#       6→7, POKEMON_FEATURES 359→360, +1×12 mon slots)
#  12 → STATE_DIM 4517 (v11 B.1b: +2 per-move PRIORITY-aware who-moves-first channels vs each enemy active;
#       MOVE_FEATURES 52→54, POKEMON_FEATURES 360→368, +2×4 moves×12 slots = +96)
#  13 → STATE_DIM 4553 (v11 C.2: +3 per-mon volatile channels = residual_damage / confused / perish_norm;
#       VOLATILE_FEATURES 7→10, POKEMON_FEATURES 368→371, +3×12 mon slots = +36)
#  14 → STATE_DIM 4565 (v11 C.2b gap-fix: +1 per-mon volatile channel = drowsy (Yawn); VOLATILE_FEATURES
#       10→11, POKEMON_FEATURES 371→372, +1×12. ALSO lit up SC channels Safeguard/Mist/Lucky Chant — no dim)
#  15 → STATE_DIM 4577 (v11 C.3: +1 per-mon volatile channel = first_turn — Fake Out / First Impression /
#       Mat Block first-turn-out legality; VOLATILE_FEATURES 11→12, POKEMON_FEATURES 372→373, +1×12 slots)
#  16 → STATE_DIM 4673 (v11 B2b: +2 per-move per-enemy HIT-CHANCE channels = realized accuracy vs each enemy
#       active (defender evasion stage / Sand Veil / Snow Cloak / Tangled Feet / Bright Powder / No Guard / the
#       Showdown acc−eva combined boost); MOVE_FEATURES 54→56, POKEMON_FEATURES 373→381, +2×4 moves×12 = +96)
#  17 → STATE_DIM 4913 (v11 Phase D / D9: +1 per-mon tera_type one-hot block = NUM_TYPES 20; POKEMON_FEATURES
#       381→401, +20×12 slots = +240. REVEALED tera type only (parity-clean) → PERMANENTLY ZERO while the
#       Champions mod hard-disables tera (canTerastallize→null); forward-compat. Gimmick codec also 2→3 here
#       but GIMMICK_DIM is an action-head width, NOT part of STATE_DIM.)
#  18 → STATE_DIM 4961 (Option 1, support-move discipline: +1 per-move REDUNDANT-CONDITION bit — 1.0 if
#       this move re-sets an already-active screen/weather/terrain on the relevant side → MOVE_FEATURES
#       56→57, POKEMON_FEATURES 401→405, +1×4 moves×12 slots = +48. A PURE feature (no action masked) so a
#       deliberate re-cast vs a predicted Brick Break / before expiry stays learnable. ⚠ Inert until a
#       RETRAIN: a v17 checkpoint (gen141) is REJECTED at load against v18. (Option 1b — an explicit
#       opp-has-screen-breaker GLOBAL — was DEFERRED: it is opp-derived but not in the gap-#6 splice, and
#       the opp's breaker move is already encoded on the opp mon's move block.))
#  19 → STATE_DIM 5057 (Option 1c, status-move discipline: +2 per-move REDUNDANT-STATUS bits — per enemy
#       active e, 1.0 if this PURE status move (Will-O-Wisp / Thunder Wave / Toxic / Spore / Glare …, i.e.
#       category 'Status' with a top-level major-status field) is WASTED on enemy[e] — it already carries a
#       major status (a mon holds only ONE) or is type/ability immune (Fire↔brn, Poison/Steel↔psn/tox,
#       Grass↔powder, Electric↔par, Ground↔Thunder-Wave, Comatose/Purifying-Salt/Limber/… ). DAMAGING moves
#       with a status RIDER (Zap Cannon, Scald, Nuzzle, Discharge) are category dmg → NEVER flagged. A PURE
#       feature (no action masked) so statusing a predicted switch-in / setup stays learnable. MOVE_FEATURES
#       57→59, POKEMON_FEATURES 405→413, +2×4 moves×12 slots = +96. ⚠ Inert until a RETRAIN: a v18 checkpoint
#       is REJECTED at load against v19. ALSO bundled in v19 (VALUE-only, no slot change): FIELD-DEPENDENT
#       move type — Weather Ball follows the weather (rain→Water/sun→Fire/snow→Ice/sand→Rock) and Terrain
#       Pulse the terrain, so the type-eff / damage-band channels rate them correctly (a rain Weather Ball
#       now reads Water/super-effective, not neutral Normal). See battle_mechanics.field_dependent_move_type.
# train_bc stamps this into the checkpoint config; model_io.load_bc_policy asserts
# it (and the dim) match the running code.
STATE_LAYOUT_VERSION = 19

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
# v11 Phase D: 3-way gimmick {none, mega, tera}. ⚠ Tera is HARD-DISABLED in the Champions mod
# (champions/scripts.ts actions.canTerastallize → null, inherited by regma; regmb uses champions
# directly) → tera can NEVER occur in regma/regmb, so the GIMMICK_TERA head trains never-tera and the
# live can_tera gate is always False. This is FORWARD-COMPAT plumbing (user-approved): the capability is
# fully wired so the bot is tera-ready if the mod ever flips canTerastallize. The tera-recognition
# channels (is_terastallized / known_tera_type / the D9 tera_type one-hot) are likewise dormant today.
GIMMICK_NONE = 0
GIMMICK_MEGA = 1
GIMMICK_TERA = 2
GIMMICK_DIM  = 3


# ── Boost key order (must match encoder) ──────────────────────────────────────
_BOOST_KEYS = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]

# Protect-family moves (mirrors replay_parser._handle_move's inline set)
_PROTECT_MOVES = {
    "protect", "detect", "wideguard", "quickguard", "banefulbunker",
    "spikyshield", "silktrap", "burningbulwark", "maxguard",
}

_EST_STAT_NORM = 300.0   # L50 stats top out around ~290 (max-HP Blissey)


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


# v11 C.2e: Gravity onModifyAccuracy chainModify([6840, 4096]) ≈ 1.6699 (NOT 5/3); numeric-accuracy only.
_GRAVITY_ACC_MULT = 6840.0 / 4096.0


# ── Convenience accessors ──────────────────────────────────────────────────────
def get_state_dim() -> int:
    return STATE_DIM

def get_state_layout_version() -> int:
    return STATE_LAYOUT_VERSION

def get_action_dim() -> int:
    return ACTION_DIM

def get_gimmick_dim() -> int:
    return GIMMICK_DIM
