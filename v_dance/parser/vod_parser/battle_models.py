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


def volatile_flags(vol_ids) -> dict:
    """Map a set of normalised volatile ids -> the encoder's per-mon volatile booleans. SHARED by the
    offline (PokemonSlot.to_dict) and live (live_state_encoder) paths so the volatile block is byte-parity
    by construction. ``ability_suppressed`` is handled by the caller (Gastro Acid / Neutralizing Gas)."""
    ids = set(vol_ids)
    return {
        "rooted": "ingrain" in ids,
        "has_substitute": "substitute" in ids,
        "move_restricted": bool(ids & _RESTRICT_VOL),
        "trapped": bool(ids & _TRAP_VOL),
        "locked_action": bool(ids & _LOCK_VOL),
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
            "is_transformed": self.is_transformed,
            "transformed_into": self.transformed_into,
            "illusion_active": self.illusion_active,
            "disguise_species": self.disguise_species,
            # v9 volatile block (B1-mechanics) — booleans the encoder reads directly. `trapped` is the
            # MOVE-based component only; the encoder ORs in ability-trapping (Shadow Tag) at encode time.
            "volatiles": {**volatile_flags(self.volatiles),
                          "ability_suppressed": self.ability_suppressed},
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

    def to_dict(self) -> dict:
        return {
            "tailwind_turns_remaining": self.tailwind,
            "screens": dict(self.screens),
        }


@dataclass
class FieldConditions:
    weather: Optional[str] = None
    terrain: Optional[str] = None
    trick_room: int = 0      # turns remaining; base 5, no item extension
    weather_turns: int = 0   # turns ACTIVE (elapsed); >5 ⇒ a weather-rock 8-turn instance
    terrain_turns: int = 0   # turns ACTIVE (elapsed); >5 ⇒ a Terrain-Extender 8-turn instance

    def to_dict(self) -> dict:
        return {
            "weather": self.weather,
            "terrain": self.terrain,
            "trick_room_turns_remaining": self.trick_room,
            "weather_turns_active": self.weather_turns,
            "terrain_turns_active": self.terrain_turns,
        }
