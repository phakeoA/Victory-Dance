"""
battle_models.py
================
Dataclasses that represent the in-battle state of a single Pokémon Showdown
match.  Imported by replay_parser.py and transitions.py; no circular deps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

# v9 (B1-mechanics): volatile id -> the encoder's per-mon volatile flag it feeds. Normalised ids.
_RESTRICT_VOL = frozenset({"taunt", "encore", "disable", "torment", "healblock", "imprison"})
_TRAP_VOL = frozenset({"partiallytrapped", "bind", "wrap", "firespin", "whirlpool", "sandtomb",
                       "magmastorm", "infestation", "clamp", "snaptrap", "thundercage", "jawlock",
                       "meanlook", "block", "spiderweb", "octolock", "noretreat", "trapped"})
_LOCK_VOL = frozenset({"mustrecharge", "lockedmove", "bide", "uproar", "rollout", "iceball",
                       "twoturnmove", "phantomforce", "shadowforce"})
# v11 C.2: ongoing residual-DAMAGE volatiles (lose HP each turn; switching cleanses) — grouped into ONE
# switch-pressure flag. Curse fires this id ONLY for the Ghost 1/4-HP variant (the non-Ghost stat-drop
# Curse deletes its volatileStatus). All four normalise identically offline (parser) and live (poke-env).
_RESIDUAL_VOL = frozenset({"leechseed", "saltcure", "curse", "nightmare"})


def _paradox_boosted_stat(ids) -> Optional[str]:
    """The stat suffix (atk/def/spa/spd/spe) of an ACTIVE Protosynthesis / Quark-Drive boost volatile
    (``|-start|MON|protosynthesisatk`` / ``quarkdrivespa`` …), or ``None``. A paradox mon boosts its
    HIGHEST stat, so this reveals which stat is highest — a strong EV-spread constraint the within-game
    belief narrows on. (``paradox_speed`` above is the SPE-only bool the speed calc consumes; this is the
    full stat, byte-parity offline/live since poke-env normalises the same ``protosynthesis<stat>`` ids.)"""
    for st in ("atk", "def", "spa", "spd", "spe"):
        if f"protosynthesis{st}" in ids or f"quarkdrive{st}" in ids:
            return st
    return None


def volatile_flags(vol_ids) -> dict:
    """Map a set of normalised volatile ids -> the encoder's per-mon volatile booleans/scalars. SHARED by
    the offline (PokemonSlot.to_dict) and live (live_state_encoder) paths so the volatile block is byte-
    parity by construction. ``ability_suppressed`` is handled by the caller (Gastro Acid only; Neutralizing
    Gas not yet tracked). ``perish_norm`` is a FLOAT in [0,1] (not a bool)."""
    ids = set(vol_ids)
    _perish = [n for n in (3, 2, 1, 0) if f"perish{n}" in ids]   # the countdown ACCUMULATES; lowest = now
    return {
        "rooted": "ingrain" in ids,
        "has_substitute": "substitute" in ids,
        "move_restricted": bool(ids & _RESTRICT_VOL),
        "trapped": bool(ids & _TRAP_VOL),
        "locked_action": bool(ids & _LOCK_VOL),
        # v11 B.1: Protosynthesis/Quark Drive (incl. Booster Energy) boosting SPEED — the stat suffix is
        # in the volatile id (|-start|MON|protosynthesisspe). NOT a volatile-block channel; consumed only
        # by _effective_speed (×1.5). Byte-parity: poke-env Effect.PROTOSYNTHESISSPE normalises identically.
        "paradox_speed": bool(ids & {"protosynthesisspe", "quarkdrivespe"}),
        # the FULL boosted stat (atk/def/spa/spd/spe or None) — reveals the paradox mon's HIGHEST stat to the
        # within-game belief (NOT an encoder channel; the encoder reads only the named volatile flags above).
        "paradox_boosted_stat": _paradox_boosted_stat(ids),
        # ── v11 C.2: dropped-volatile channels (pure id-set functions ⇒ offline/live byte-parity) ──
        "residual_damage": bool(ids & _RESIDUAL_VOL),    # Leech Seed / Salt Cure / Ghost-Curse / Nightmare
        "confused": "confusion" in ids,                  # 33% self-hit risk
        # perish_norm: perish0→1.0 (faint imminent) · perish1→0.75 · perish2→0.5 · perish3→0.25 · none→0.0
        "perish_norm": (4 - min(_perish)) / 4.0 if _perish else 0.0,
        # v11 C.2b gap-fix: Yawn — the mon falls asleep NEXT turn unless it switches (a switch-or-sleep
        # signal). 'yawn' lingers in the id set until the |-end| Showdown emits when sleep lands.
        "drowsy": "yawn" in ids,
        # v11 C.2c grounding gap-fix — NOT volatile-block channels; consumed by _is_grounded / the band's
        # Ground-immunity + Tar-Shot type-eff (like paradox_speed). Byte-parity: ids normalise identically.
        "levitating": bool(ids & {"magnetrise", "telekinesis"}),   # temporarily UNgrounded (Ground-immune)
        "force_grounded": bool(ids & {"smackdown", "ingrain"}),    # forced grounded (overrides Flying/Levitate)
        "tar_shot": "tarshot" in ids,                              # +1 step Fire effectiveness (×2)
        # v11 C.2d type-immunity NEGATION (consumed by the move-writer's chart negation, NOT channels —
        # like the grounding flags above). Foresight/Odor Sleuth both apply the 'foresight' volatile →
        # Normal/Fighting vs Ghost 0×→1×; Miracle Eye → Psychic vs Dark 0×→1×. Byte-parity: ids normalise
        # identically (poke-env Effect.FORESIGHT / Effect.MIRACLE_EYE map to the same ids).
        "foresight": "foresight" in ids,
        "miracleeye": "miracleeye" in ids,
        # v11: Embargo volatile — suppresses the holder's item effects for 5 turns (consumed by the
        # _item_active gate, NOT a volatile-block channel). Pure id-set membership ⇒ offline/live byte-parity.
        "embargo": "embargo" in ids,
    }


@dataclass
class PokemonSlot:
    """Tracks a single Pokémon's in-battle state snapshot."""
    species: str
    nickname: str
    player: str          # "p1" or "p2"
    slot: str            # "a" or "b"  (active field position)
    hp_current: Optional[float] = None   # raw numerator from the log (NOT a pct)
    hp_max: Optional[float] = 100.0      # raw denominator (100 for %-scale logs)
    status: Optional[str] = None
    boosts: dict = field(default_factory=dict)
    is_mega: bool = False
    is_fainted: bool = False
    # --- new in v2 ---
    revealed_moves: list = field(default_factory=list)   # moves seen this match
    known_item: Optional[str] = None                     # item if revealed
    # Item consumed/removed (|-enditem|: berry eaten, Sash popped, Herb used,
    # Knock Off, …).  known_item keeps the item's IDENTITY for belief/UI, but the
    # mon no longer HOLDS it — the encoder treats it as itemless (gap #5), matching
    # poke-env, which sets mon.item = None on consumption.  Not cleared on switch
    # (a consumed item is gone for the game).
    item_consumed: bool = False
    known_tera_type: Optional[str] = None                # tera type if revealed
    is_terastallized: bool = False
    known_ability: Optional[str] = None                  # CURRENTLY-active ability, if known
    # Bug 8 (mega ability split): a mega-evolving Pokémon has TWO ability
    # contexts.  `known_ability` always reflects the ability that is active
    # RIGHT NOW.  Before mega it equals pre_mega_ability (if revealed); the
    # instant the mon mega evolves it becomes the mega forme's single fixed
    # ability (resolved from the pokedex), and the old value is preserved in
    # pre_mega_ability instead of being silently lost/left stale.
    pre_mega_ability: Optional[str] = None               # base-forme ability (revealed/injected)
    mega_ability: Optional[str] = None                   # mega forme's fixed ability (deterministic)
    # Bug 7: teampreview/roster species name, frozen at first switch-in.
    # `species` mutates on mega evolution (|detailschange|), so bench/roster
    # reconciliation must compare against this instead.
    base_species: Optional[str] = None
    # Choice-item constraint: Choice items (Scarf/Band/Specs) lock the holder
    # into the first move it selects until it leaves the field.  A mon
    # observed using 2+ DIFFERENT self-selected moves during one continuous
    # stay on the field therefore cannot have brought a Choice item — the
    # belief fill uses this to drop Choice items from its item distribution.
    # Flips False permanently once proven; the per-stint working set lives in
    # stint_moves (internal — never serialised, reset on switch-in and on any
    # item change because moves after a Trick/Knock Off prove nothing about
    # the ORIGINAL item).
    can_have_choice_item: bool = True
    stint_moves: list = field(default_factory=list)
    # PP tracking (gap #7): {norm_species(move) -> uses} for the pp_fraction
    # feature.  PERSISTS across switch-outs (PP is not restored), unlike
    # stint_moves.  Counts only self-selected uses (transformed/called moves are
    # skipped in _handle_move, mirroring poke-env's `use` flag); Pressure (2 PP)
    # is not modelled — a rare, documented residual.
    move_pp_used: dict = field(default_factory=dict)
    # Transform / Imposter (Ditto, Mew, …): once a mon Transforms it borrows
    # the target's species/moves/stats for the rest of its stay on the field.
    # Moves used while transformed are the COPIED foe's moves and reveal
    # nothing about this mon's real set — `is_transformed` gates them out of
    # revealed_moves and the Choice-item constraint.  Reverts on switch-out.
    is_transformed: bool = False             # CURRENT state — reverts on switch-out
    transformed_into: Optional[str] = None   # species currently copied, if any
    ever_transformed: bool = False           # latched: transformed at any point
    #   (match-level signal for revealed_info — never reverts)
    # v11 P2: runtime TYPE CHANGE that REPLACES the type list (Protean/Libero/Color Change/Soak/
    # Conversion/Conversion2/Burn Up/Double Shock — any |-start|MON|typechange|). A type-name list
    # (the encoder re-canons it); None = no active typechange. Mirrors poke-env Pokemon._temporary_types:
    # SET by the typechange protocol line, CLEARED on switch-out/in, terastallize and transform.
    # _effective_types reads it at 2nd precedence: tera > runtime_types > transform > dex.
    runtime_types: Optional[list] = None
    # Illusion (Zoroark): the parser tracks the TRUE identity (so moves credit
    # correctly), but an OPPONENT is fooled into seeing the disguise until a hit
    # breaks it (|replace|).  These let _snapshot_state present the disguise on
    # the opponent's side of a snapshot while the illusion is up — the realistic
    # "fooled view" the in-battle policy must learn from.
    illusion_active: bool = False            # disguise currently up (pre-|replace|)
    disguise_species: Optional[str] = None   # species the disguise appears as
    # v9 (B1-mechanics): per-mon volatile state for the encoder's volatile block. CLEARED on switch-in
    # (volatiles end when a mon leaves the field — handled in replay_parser._handle_switch). MOVE-based
    # trapping only; ABILITY trapping (Shadow Tag) is derived at encode time from the opponent's ability.
    volatiles: set = field(default_factory=set)   # active normalised volatile ids
    ability_suppressed: bool = False              # Gastro Acid (per-mon ability suppression) in effect
    # v11 A.3: consecutive-Protect counter (poke-env Pokemon.protect_counter). Driven from the |move|
    # line; resets on switch / a non-protect move / |cant|. The one |-singleturn| signal that survives
    # to the next decision snapshot and is exposed identically by poke-env (so it is parity-safe).
    protect_counter: int = 0
    # v11 C.3: active-turns counter mirroring poke-env Pokemon._active_turns. RESET to 0 on switch-IN
    # (replay_parser._handle_switch incoming branch); +1 in the |turn| handler for every ACTIVE,
    # non-fainted slot, BEFORE the decision snapshot. first_turn = (active_turns == 1) gates Fake Out /
    # First Impression / Mat Block. poke-env: _active_turns inits 0, is zeroed in switch_OUT (NOT
    # switch_in), +=1 in end_turn() (one per active mon per |turn|), first_turn = (_active_turns == 1).
    # NOT reset on switch-out here (poke-env freezes a benched mon's value) → a benched slot's first_turn
    # stays frozen identically on both paths.
    active_turns: int = 0
    # v11 C.4: per-stint Rage Fist hit counter (Showdown Pokemon.timesAttacked). +1 per DIRECT damaging-
    # move |-damage| hit (chip/recoil/confusion/Future-Sight carry [from] → excluded; self-cost lines
    # guarded); reset 0 on switch-IN (offline twin of clearVolatile). Feeds Rage Fist BP = 50+50×count.
    times_attacked: int = 0
    # v11 exact-mask: specific move-restriction capture, split out of the bundled `move_restricted`
    # volatile bool so the OFFLINE action mask can drop the EXACT illegal slot (the bundle can't tell
    # taunt from encore from disable). MASK-ONLY metadata (NOT encoder feature channels); reset on switch.
    taunt: bool = False                    # |-start|MON|move: Taunt  (blocks status moves)
    disabled_move: Optional[str] = None    # the move named by |-start|MON|Disable|<Move>
    encore_move: Optional[str] = None      # move Encore locks into (derived from the last self-selected move)
    # audit: the ACTUAL last self-selected move this stint (NOT de-duped, unlike stint_moves) — Encore locks
    # into the move used ON the Encore turn, which differs from stint_moves[-1] when that move was a repeat.
    last_self_move: Optional[str] = None

    def key(self) -> str:
        return f"{self.player}{self.slot}"

    def to_dict(self) -> dict:
        return {
            "species": self.species,
            "base_species": self.base_species or self.species,
            "nickname": self.nickname,
            "player": self.player,
            "slot": self.slot,
            # hp_pct is a true 0-100 PERCENTAGE.  The log expresses HP either as
            # a percentage (X/100) or, when the replay was recorded from the
            # owner's client, as REAL HP (e.g. 175/200) — gap #5.  hp_current is
            # the raw numerator and hp_max the raw denominator, so we must divide
            # to normalise; storing the bare numerator over-reports a real-HP
            # mon (175 → clamped to full 1.0 by the encoder) and breaks live
            # parity (poke-env's current_hp_fraction is always a true fraction).
            "hp_pct": (
                None if self.hp_current is None
                else self.hp_current / (self.hp_max or 100.0) * 100.0
            ),
            "status": self.status,
            "boosts": dict(self.boosts),
            "is_mega": self.is_mega,
            "is_fainted": self.is_fainted,
            "revealed_moves": list(self.revealed_moves),
            "known_item": "mega stone" if self.is_mega else self.known_item,
            "item_consumed": self.item_consumed and not self.is_mega,
            "known_tera_type": self.known_tera_type,
            "is_terastallized": self.is_terastallized,
            "known_ability": self.known_ability,
            "pre_mega_ability": self.pre_mega_ability,
            "mega_ability": self.mega_ability,
            "can_have_choice_item": self.can_have_choice_item,
            "move_pp_used": dict(self.move_pp_used),
            "times_attacked": self.times_attacked,        # v11 C.4: Rage Fist BP (top-level → att_ctx)
            "stint_moves": list(self.stint_moves),        # v11 exact-mask: known-Choice lock target
            "is_transformed": self.is_transformed,
            "transformed_into": self.transformed_into,
            "runtime_types": list(self.runtime_types) if self.runtime_types else None,  # v11 P2 typechange
            "illusion_active": self.illusion_active,
            "disguise_species": self.disguise_species,
            # v9 volatile block (B1-mechanics) — booleans the encoder reads directly. `trapped` is the
            # MOVE-based component only; ability-trapping (Shadow Tag) is NOT yet OR'd into can_switch
            # (a documented Phase-B follow-up — the encoder currently uses move-trap | rooted only).
            # protect_counter (v11 A.3) is a non-flag scalar carried here alongside ability_suppressed.
            "volatiles": {**volatile_flags(self.volatiles),
                          "ability_suppressed": self.ability_suppressed,
                          "protect_counter": self.protect_counter,
                          "first_turn": self.active_turns == 1,   # v11 C.3
                          # v11 exact-mask (MASK-ONLY — consumed by build_action_mask, NOT a feature channel):
                          "taunt": self.taunt,
                          "disabled_move": self.disabled_move,
                          "encore_move": self.encore_move},
            # EVs/IVs unknown for Type B — left as distribution placeholder
            "ev_spread": None,
            "iv_spread": None,
            "nature": None,
        }


@dataclass
class SideConditions:
    tailwind: int = 0        # turns REMAINING (0 = inactive); base 4, no item extension
    # screens: {reflect/light_screen/aurora_veil: turns ACTIVE (elapsed, 0 on set)} — counted UP and
    # removed ONLY by the explicit |-sideend| event (not auto-expired at 5), so a Light Clay 8-turn
    # screen is tracked correctly. The elapsed value IS the field-duration signal: elapsed > 5 ⇒ the
    # setter held Light Clay (8-turn), inferrable identically offline (parser counts) and live
    # (poke-env start-turn). See state_encoder field-duration block.
    screens: dict = field(default_factory=dict)
    # v11 C.1: entry hazards on THIS side (hit a mon SWITCHING IN). Stealth Rock / Sticky Web are 1-layer
    # (bool); Spikes (0-3) / Toxic Spikes (0-2) STACK — store the layer count (parity with poke-env's
    # STACKABLE_CONDITIONS). Persist until removed (Defog / Rapid Spin / Court Change); no turn decay.
    stealth_rock: bool = False
    spikes: int = 0
    toxic_spikes: int = 0
    sticky_web: bool = False
    # v11 C.2b gap-fix: persistent team-protection side conditions (5 turns) that had a SIDE_COND channel
    # but were never populated. Safeguard = status immunity, Mist = stat-drop/Intimidate immunity,
    # Lucky Chant = crit immunity. Presence bits (no field-duration channel for these).
    safeguard: bool = False
    mist: bool = False
    lucky_chant: bool = False

    def to_dict(self) -> dict:
        return {
            "tailwind_turns_remaining": self.tailwind,
            "screens": dict(self.screens),
            "stealth_rock": self.stealth_rock,
            "spikes": self.spikes,
            "toxic_spikes": self.toxic_spikes,
            "sticky_web": self.sticky_web,
            "safeguard": self.safeguard,
            "mist": self.mist,
            "lucky_chant": self.lucky_chant,
        }


@dataclass
class FieldConditions:
    weather: Optional[str] = None
    terrain: Optional[str] = None
    trick_room: int = 0      # turns remaining; base 5, no item extension
    weather_turns: int = 0   # turns ACTIVE (elapsed); >5 ⇒ a weather-rock 8-turn instance
    terrain_turns: int = 0   # turns ACTIVE (elapsed); >5 ⇒ a Terrain-Extender 8-turn instance
    gravity: int = 0         # v11 C.2e: turns remaining; base 5 (7 if announced [persistent]); no item ext
    magic_room: int = 0      # v11 P5: turns remaining; base 5 (7 if [persistent]); suppresses ALL item effects
    wonder_room: int = 0     # v11 P5: turns remaining; base 5 (7 if [persistent]); swaps Def<->SpD in the band

    def to_dict(self) -> dict:
        return {
            "weather": self.weather,
            "terrain": self.terrain,
            "trick_room_turns_remaining": self.trick_room,
            "weather_turns_active": self.weather_turns,
            "terrain_turns_active": self.terrain_turns,
            "gravity_turns_remaining": self.gravity,
            "magic_room_turns_remaining": self.magic_room,
            "wonder_room_turns_remaining": self.wonder_room,
        }
