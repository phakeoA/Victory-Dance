"""
State encoder for VGC Reg M-A (Pokémon Champions doubles format) — OFFLINE
(VOD-parsing) path + the FROZEN tensor layout shared by both paths.

As of the v17 split this module hosts the ``VodStateEncoder`` class (the OFFLINE
encode path) and RE-EXPORTS the frozen layout + helpers factored into sibling
modules, so its public import surface is byte-identical to before:
  * ``encoder_layout``    — every feature ordering, dimension constant, vocab
                            size, index map, and the action-space / gimmick consts
  * ``battle_mechanics``  — pure damage / type-chart / speed / ability+item
                            multiplier / accuracy / crit helpers (shared with live)
  * ``action_codec``      — legality masks, gimmick mask, move-slot permutation,
                            opponent-perspective relabelling
The LIVE encoder (``live_state_encoder.LiveStateEncoder``) imports the SAME names
so the two paths can never drift.  This module encodes parsed/belief-enriched VOD JSON:

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

  [C] GLOBAL_FEATURES  (= 101)
        weather multi-hot (9)
        terrain/field multi-hot (15)
        own side conditions (24)  ← multi-hot; layered hazards (Spikes/Toxic Spikes) carry a fractional
        opp side conditions (24)    layer count (C.1), so this is NOT strictly binary
        turn normalised (1)
        trick room flag (1)
        turn-order block (12) ← B1.4: 4 effective-speed + 4 moves-first margins + 4 pair-confidences
        field-duration (11)  ← v10: turns-active age for weather/terrain/Trick-Room + per-side tailwind +
                               reflect/light_screen/aurora_veil (age>base ⇒ Light-Clay/Terrain-Extender 8-turn)
        team counts (4)      ← own/opp living-bench, own/opp fainted (each /4) — MUST stay last
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
POKEMON_FEATURES per slot (= 405; the *_FEATURES constants in encoder_layout.py are canonical, but this
list is the authoritative per-field breakdown in WRITE order — keep it in sync on a layout bump):
    hp_frac          (1)
    type1 one-hot    (20)
    type2 one-hot    (20)   ← zeros if single-type
    base_stats       (6)    ← hp atk def spa spd spe / 255
    est_stats        (6)    ← in-battle L50 stats / 300 (exact or belief-expected)
    stats_known      (1)    ← 1.0 exact · 0.5 distribution · 0.0 unknown
    is_mega          (1)
    is_tera          (1)
    status one-hot   (7)    ← BRN FNT FRZ PAR PSN SLP TOX
    boosts           (7)    ← atk def spa spd spe acc eva / 6
    weight           (1)    ← v9: normalized species weight (Heavy Slam / Low Kick / Heavy-Light Metal)
    4 × MOVE_FEATURES (228) ← 4 × 57 (per-move breakdown below)
    item block       (25)   ← v9: 23 item-mechanic tags (multi-hot) + identity index (1) + item_known (1);
                              tags are MECHANICS (not identity) so the layout is format-scalable
    ability block    (45)   ← v9: 43 ability-mechanic tags (multi-hot) + identity index (1) + ability_known (1)
    volatile block   (12)   ← v9/v11 C.2: rooted · substitute · move_restricted · trapped · locked_action ·
                              paradox_speed · residual_damage · confused · perish_norm · drowsy + 2 more
                              (grounding/negation/embargo flags are helper-only, NOT all channels)
    tera_type one-hot(20)   ← v11 Phase D (D9): REVEALED tera type — PERMANENTLY ZERO (tera mod-disabled),
                              forward-compat plumbing
    is_active        (1)
    is_revealed      (1)    ← always 1 for own mons
    is_fainted       (1)    ← layout-v2; KO flag (opp bench / counting)
    is_transformed   (1)    ← layout-v2; Ditto copied a forme (types/stats above are the COPY's)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MOVE_FEATURES per move (= 57 = 25 core + 1 redundant-condition + 29 move-tags + identity + is_known; in WRITE order):
    base_power / 150     (1)   ← v11 N1 Champions BP override, capped 250
    type_idx / 19        (1)   ← ordinal, not one-hot; v9 -ate / N1 Champions type
    category             (1)   ← 0=phys 0.5=spec 1=status
    priority / 7         (1)
    accuracy             (1)   ← 0–1; v11 B2a attacker fold (acc-stage · Compound Eyes · Hustle · Wide Lens ·
                                 Victory Star · Gravity); an always-hit move → 1.0
    pp_fraction          (1)   ← remaining PP / max (gap #7); max = base·8//5
    is_protect           (1)
    is_stab              (1)
    is_spread            (1)   ← gap #6; hits both foes (allAdjacentFoes/allAdjacent)
    type_eff vs enemy0/1 (4)   ← B1.1: signed log2(mult)/2 + immune flag, ×2 enemies (C.2d negation)
    damage vs enemy0/1   (4)   ← B1.2 + A1-A4/B2-B3/N1/N4/G7: [min,max] roll as frac of current HP, ×2 enemies
    who-moves-first 0/1  (2)   ← B1.4b: priority-aware signed speed margin of THIS move vs each enemy
    intrinsics           (4)   ← B1.3: contact · recoil · drain · multihit-count/5 (per-move, public)
    hit-chance vs enemy0/1(2)  ← B2b: realized hit chance vs each enemy (evasion / Sand Veil / No Guard / …)
    redundant-condition  (1)   ← v18 Option 1: this move re-sets a screen/weather/terrain already active on the caster's side
    move tags            (29)  ← v9: mechanic-tag multi-hot prior
    identity index       (1)   ← v9: move embedding index (feeds nn.Embedding, NOT the flat dim)
    is_known             (1)   ← 1.0 revealed/exact · p(usage) belief-padded · 0 empty slot
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
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from v_dance.parser.vod_parser.pokedex import get_pokedex, norm_species
from v_dance.parser.belief_state import BeliefState, dex_base_stats, STAT_ORDER

# ── v9 (B1-mechanics): data-grounded mechanic substrate ──────────────────────────
# Tag NAME counts are FIXED (dex-independent) so the flat LAYOUT is stable; the per-mon identity INDEX is
# 1 float the model embeds; vocab sizes feed the model's nn.Embeddings (NOT the flat dim). No circular
# import — mechanic_tags/vocab/damage_mechanics never import the encoder.
from v_dance.encoders.mechanic_tags import (  # noqa: E402
    NUM_MOVE_TAGS, NUM_ABILITY_TAGS, NUM_ITEM_TAGS,
    move_tag_indices, ability_tag_indices, item_tag_indices, ABILITY_SUPPRESSED_IDX)
from v_dance.encoders.mechanic_vocab import (  # noqa: E402
    ability_index, move_index, item_index,
    ABILITY_VOCAB_SIZE, MOVE_VOCAB_SIZE, ITEM_VOCAB_SIZE)  # noqa: F401 (re-exported for bc_model_attn + tests)
from v_dance.encoders import damage_mechanics as _DMG  # noqa: E402
from v_dance.encoders.champions_move_overrides import CHAMP_BP, CHAMP_TYPE, CHAMP_ACC  # v11 N1  # noqa: E402

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)

# ── v17 split: re-export every name moved to the sibling modules so this
#    module's public import surface is byte-identical to before the split. ──
from v_dance.encoders.encoder_layout import (  # noqa: F401
    ABILITY_BLOCK_V9, ABILITY_EFFECT_NAMES, ABILITY_FEATURES, ACTIONS_PER_SLOT, ACTION_DIM,
    ACTIVE_SLOTS, BENCH_SLOTS, FIELD_NAMES, GIMMICK_DIM, GIMMICK_MEGA, GIMMICK_NONE,
    GIMMICK_TERA, GLOBAL_FEATURES, ITEM_BLOCK_V9, ITEM_EFFECT_NAMES, ITEM_FEATURES,
    MOVE_FEATURES, MOVE_TARGET_PAIRS, NUM_ABILITY_EFFECTS, NUM_BOOSTS, NUM_FIELDS,
    NUM_ITEM_EFFECTS, NUM_MOVES, NUM_SIDE_CONDS, NUM_STATUS, NUM_TYPES, NUM_WEATHER,
    OPP_BENCH_SLOTS, POKEMON_FEATURES, SIDE_COND_NAMES, STATE_DIM, STATE_LAYOUT_VERSION,
    STATUS_NAMES, SWITCH_OFFSET, TYPE_NAMES, VOLATILE_FEATURES, WEATHER_NAMES, WEIGHT_FEATURES,
    _ABIL_EFFECT_IDX, _BOOST_KEYS, _EST_STAT_NORM, _FIELD_IDX, _GRAVITY_ACC_MULT,
    _ITEM_EFFECT_IDX, _PROTECT_COUNTER_CAP, _PROTECT_MOVES, _SCREEN_TO_SC, _SC_IDX,
    _STATUS_IDX, _TERRAIN_TO_FIELD, _TYPE_IDX, _WEATHER_ALIASES, _WEATHER_IDX, get_action_dim,
    get_gimmick_dim, get_state_dim, get_state_layout_version,
)
from v_dance.encoders.battle_mechanics import (  # noqa: F401
    _ABILITY_FLAG_IMMUNITY, _ABILITY_TYPE_IMMUNITY, _AGE_MAX, _ATE_BOOST_AB, _ATE_NOMODIFY,
    _BAND_ITEM_MULT, _BOOSTER_AB, _CONTACT_PUNISH_ITEMS, _CRIT_MULT, _DEF_HIT_MOVES, _GRASSY_WEAKENED,
    _DEF_MOLDBREAKER_IMMUNE, _DEF_REDUCE_AB, _DEF_SIGNED_RESIST_AB, _DMG_BOOST_AB,
    _DROP_GUARD_ITEMS, _EFFECT_SHIELD_ITEMS, _GEN9_MOVES_CACHE, _GUTS_AB, _HEAL_BERRIES,
    _IGNORE_EVA_AB, _INTIMIDATE_AB, _ITEM_NONE_IDS, _LUCK_ITEMS, _MAGIC_BOUNCE_AB,
    _MOLD_BREAKER_AB, _MOVES_JSON_PATH, _M_1_2, _M_1_3, _NON_ALNUM, _OFF_EXTRA_AB,
    _PRANKSTER_AB, _REACTIVE_AB, _RECOVERY_ITEMS, _REGENERATOR_AB, _RESIST_BERRIES,
    _RESIST_BERRY_TYPE, _SCREEN_AGE_KEYS, _SPEED_CTRL_AB, _SPREAD_TARGET_IDS, _STATDROP_AB,
    _STATUS_CURE_BERRIES, _STATUS_IMMUNE_AB, _SUPREME_OVERLORD_MULT, _TERRAIN_SET_AB,
    _TYPE_BOOST_ITEMS, _TYPE_BOOST_TYPE, _TYPE_CHART_CACHE, _TYPE_IMMUNE_AB, _WEATHER_SET_AB,
    _WEATHER_SPEED_AB, _WEATHER_SPEED_ABILITY, _ability_damage_mult, _ability_immunizes,
    _ability_trapped, _accuracy_modifiers, _canon, _damage_band, _defender_profile, _dex_types,
    _disguise_intact, _effective_speed, _effective_types, _expected_crit_mult, _gen9_moves,
    _get_moves_data, _ground_immune, _immunity_neg_ctx, _immunity_negated, _is_grounded,
    _item_active, _move_always_hit, _move_damage_props, _move_ground_pierces, _move_hit_range,
    _move_ignores_all_immunity, _move_ignores_evasion, _move_immune, _move_is_ohko,
    _moves_data, _moves_first, _per_enemy_hit_chance, _side_screens, _situational_damage_mult,
    _type_chart, _type_eff_signed_immune, _type_mult, ability_effect_indices, champ_acc_raw,
    champ_bp, champ_type, dex_unique_ability, field_dependent_move_type, field_duration_scalars, is_spread_target,
    item_effect_indices, move_redundant_condition, move_redundant_status, pp_max,
    resolve_ability_json, resolve_active_ability_json, resolve_item_json,
)
from v_dance.encoders.action_codec import (  # noqa: F401
    ABILITY_ID_REL, ABILITY_KNOWN_REL, ITEM_ID_REL, MOVE_ID_RELS, OPP_HEADS,
    _ABILITY_BLOCK_REL, _ALLY_KINDS, _CHOICE_ITEMS, _CHOOSABLE_SINGLE, _FIRST_TURN_ONLY,
    _ITEM_BLOCK_REL, _MOVE_BLOCK_REL, _WEIGHT_REL, _certainly_illegal_move_slots,
    _flip_action_slots, _flip_perspective, _living_bench, _move_target_kind,
    _own_team_has_megaed, _own_team_has_teraed, _relabel_slots, _species_is_mega_capable,
    _species_matches, _target_bucket, action_to_gimmick, action_to_index, annotate_opp_actions,
    annotate_transition_actions, build_action_mask, build_gimmick_mask, build_opp_action_mask,
    index_to_action, move_slots_for_mon, opp_action_to_index, own_active_move_base,
    permute_action_index, permute_action_mask_row, permute_move_slots,
)


def _active_side_conditions_offline(side: dict | None) -> frozenset:
    """v18 (Option 1): canonical SIDE_COND names ACTIVE on a side, from the offline snapshot's side dict —
    MIRRORS the global SC block exactly (tailwind + screens + safeguard/mist/lucky_chant), so the per-move
    redundant-condition bit is parity-clean with the live path's analogue."""
    s = side or {}
    out = set()
    if (s.get("tailwind_turns_remaining") or 0) > 0:
        out.add("TAILWIND")
    for _scr in (s.get("screens") or {}):
        _nm = _SCREEN_TO_SC.get(_scr)
        if _nm:
            out.add(_nm)
    if s.get("safeguard"):
        out.add("SAFEGUARD")
    if s.get("mist"):
        out.add("MIST")
    if s.get("lucky_chant"):
        out.add("LUCKY_CHANT")
    return frozenset(out)


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
        # v18 (Option 1): canonical active SIDE_COND names per side → the per-move redundant-condition bit.
        _our_side_active = _active_side_conditions_offline(_sc.get("our_side"))
        _opp_side_active = _active_side_conditions_offline(_sc.get("opp_side"))
        # v11 B.1b: per-side tailwind + Trick Room resolved ONCE for the move-block who-moves-first
        # channel — the defender eff_speed and the attacker att_ctx.eff_speed each need the mon's OWN
        # side tailwind (and the global weather/TR).
        _our_tw = ((_sc.get("our_side") or {}).get("tailwind_turns_remaining") or 0) > 0
        _opp_tw = ((_sc.get("opp_side") or {}).get("tailwind_turns_remaining") or 0) > 0
        _trick_room = (_field.get("trick_room_turns_remaining") or 0) > 0
        _gravity = (_field.get("gravity_turns_remaining") or 0) > 0      # v11 C.2e
        _magic_room = (_field.get("magic_room_turns_remaining") or 0) > 0     # v11 P5
        _wonder_room = (_field.get("wonder_room_turns_remaining") or 0) > 0   # v11 P5
        # v11 B.2 gap-fix (Last Respects): per-side fainted count = the move's BP driver (50+50×faints).
        # SAME expressions as the parity-proven team-count block below (own bench fainted / opp seen-fainted).
        _own_fnt = sum(1 for m in (snap.get("our_bench") or []) if m and m.get("is_fainted"))
        _opp_fnt = sum(1 for m in (snap.get("opp_bench") or [])
                       if m and m.get("seen", True) and m.get("is_fainted"))
        own_enemy = [_defender_profile(opp_active.get("opp_a"), _opp_scr, _opp_tw, _weather, _gravity, _magic_room),
                     _defender_profile(opp_active.get("opp_b"), _opp_scr, _opp_tw, _weather, _gravity, _magic_room)]
        opp_enemy = [_defender_profile(our_active.get("our_a"), _our_scr, _our_tw, _weather, _gravity, _magic_room),
                     _defender_profile(our_active.get("our_b"), _our_scr, _our_tw, _weather, _gravity, _magic_room)]
        for key_map, prefix in ((our_active, "our"), (opp_active, "opp")):
            enemy = own_enemy if prefix == "our" else opp_enemy
            _tw = _our_tw if prefix == "our" else _opp_tw
            _fnt = _own_fnt if prefix == "our" else _opp_fnt
            _side_act = _our_side_active if prefix == "our" else _opp_side_active
            for slot in ("a", "b"):
                mon = key_map.get(f"{prefix}_{slot}")
                # v11 Victory Star: the ACTIVE ally's ability (other slot) — boosts this mon's accuracy.
                _ally = key_map.get(f"{prefix}_{'b' if slot == 'a' else 'a'}")
                _ally_ab = resolve_active_ability_json(_ally)[0] if _ally else None
                self._write_pokemon_json(vec, cursor, mon, is_active=True, enemy_defenders=enemy,
                                         field_mods=field_mods, side_tailwind=_tw, trick_room=_trick_room,
                                         gravity=_gravity, fainted_allies=_fnt,
                                         magic_room=_magic_room, wonder_room=_wonder_room,
                                         ally_ability=_ally_ab, side_active=_side_act)
                cursor += POKEMON_FEATURES

        # ── [B] Own bench (fainted excluded — matches the live path; the
        # snapshot keeps fainted mons by design, see replay_parser Bug 6) ────
        bench = [
            m for m in (snap.get("our_bench") or []) if not m.get("is_fainted")
        ][:BENCH_SLOTS]
        for i in range(BENCH_SLOTS):
            mon = bench[i] if i < len(bench) else None
            self._write_pokemon_json(vec, cursor, mon, is_active=False, enemy_defenders=own_enemy,
                                     field_mods=field_mods, side_tailwind=_our_tw, trick_room=_trick_room,
                                     gravity=_gravity, fainted_allies=_own_fnt,
                                     magic_room=_magic_room, wonder_room=_wonder_room,
                                     side_active=_our_side_active)
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
                                     field_mods=field_mods, side_tailwind=_opp_tw, trick_room=_trick_room,
                                     gravity=_gravity, fainted_allies=_opp_fnt,
                                     magic_room=_magic_room, wonder_room=_wonder_room,
                                     side_active=_opp_side_active)
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
        if (field.get("gravity_turns_remaining") or 0) > 0:     # v11 C.2e (channel already in FIELD_NAMES)
            vec[cursor + _FIELD_IDX["GRAVITY"]] = 1.0
        if (field.get("magic_room_turns_remaining") or 0) > 0:  # v11 P5 (channel already in FIELD_NAMES)
            vec[cursor + _FIELD_IDX["MAGIC_ROOM"]] = 1.0
        if (field.get("wonder_room_turns_remaining") or 0) > 0:  # v11 P5
            vec[cursor + _FIELD_IDX["WONDER_ROOM"]] = 1.0
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
            # v11 C.1: entry hazards — Spikes/Toxic Spikes as a normalised LAYER count, SR/Web as presence.
            if sc.get("stealth_rock"):
                vec[cursor + _SC_IDX["STEALTH_ROCK"]] = 1.0
            if sc.get("sticky_web"):
                vec[cursor + _SC_IDX["STICKY_WEB"]] = 1.0
            if sc.get("spikes"):
                vec[cursor + _SC_IDX["SPIKES"]] = min(sc["spikes"], 3) / 3.0
            if sc.get("toxic_spikes"):
                vec[cursor + _SC_IDX["TOXIC_SPIKES"]] = min(sc["toxic_spikes"], 2) / 2.0
            # v11 C.2b gap-fix: persistent team protections (presence bits).
            if sc.get("safeguard"):
                vec[cursor + _SC_IDX["SAFEGUARD"]] = 1.0
            if sc.get("mist"):
                vec[cursor + _SC_IDX["MIST"]] = 1.0
            if sc.get("lucky_chant"):
                vec[cursor + _SC_IDX["LUCKY_CHANT"]] = 1.0
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
        _sp = {  # v11 E4-fix: pass _magic_room so the global turn-order speed suppresses Choice Scarf under
                 # Magic Room (matching the per-move who-moves-first eff_speed); Klutz/Embargo are gated inside.
            "our_a": _effective_speed(our_active.get("our_a"), _our_tw, _weather, _magic_room),
            "our_b": _effective_speed(our_active.get("our_b"), _our_tw, _weather, _magic_room),
            "opp_a": _effective_speed(opp_active.get("opp_a"), _opp_tw, _weather, _magic_room),
            "opp_b": _effective_speed(opp_active.get("opp_b"), _opp_tw, _weather, _magic_room),
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
        side_tailwind: bool = False,
        trick_room: bool = False,
        gravity: bool = False,
        fainted_allies: int = 0,
        magic_room: bool = False,
        wonder_room: bool = False,
        ally_ability: Optional[str] = None,
        side_active: frozenset = frozenset(),
    ) -> None:
        """JSON twin of _write_pokemon — same POKEMON_FEATURES layout. ``enemy_defenders`` = the 2 enemy
        actives' defender profiles; ``field_mods`` = (weather, terrain) for the B1.2b damage modifiers;
        ``side_tailwind``/``trick_room`` (v11 B.1b) feed this mon's resolved speed + the who-moves-first
        channel against each enemy active. ``gravity`` (v11 C.2e) force-grounds this mon (Arena Trap
        can_switch) and boosts numeric move accuracy ×6840/4096. ``fainted_allies`` (v11 B.2 gap-fix) =
        this mon's side faint count → Last Respects base power (50+50×faints)."""
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
        _att_item = _item_active(resolve_item_json(mon)[0], magic_room,   # v11 P5: Magic Room
                                 klutz=abil_id == "klutz",                # v11 Klutz: suppress held item
                                 embargo=bool(vol.get("embargo")))        # v11 Embargo volatile
        _aidx = item_effect_indices(_att_item)
        att_ctx = {"atk": _est.get("atk"), "spa": _est.get("spa"),
                   "burned": _canon(mon.get("status")) == "BRN",
                   "statused": bool(_canon(mon.get("status"))),    # v11 A.2: Guts triggers on ANY status
                   "life_orb": _ITEM_EFFECT_IDX.get("life_orb") in _aidx,
                   "choice": (_ITEM_EFFECT_IDX.get("choice") in _aidx
                              and _ITEM_EFFECT_IDX.get("choice_speed") not in _aidx),
                   # v11 B.1b: this mon's resolved speed + Trick Room for the who-moves-first channel.
                   "eff_speed": _effective_speed(mon, side_tailwind,
                                                 field_mods[0] if field_mods else None, magic_room)[0],
                   "trick_room": trick_room,
                   "wonder_room": wonder_room,                      # v11 P5: swaps Def<->SpD in the band
                   # v11 B.2: damage-band R1 context — user's Def (Body Press), raw weight (Heavy Slam),
                   # hp_frac (Eruption family; SAME stub convention as the per-mon hp channel above) and
                   # the sum of positive boost stages (Stored Power / Power Trip).
                   "def": _est.get("def"), "weight": _wt,
                   "hp_frac": (0.0 if mon.get("seen", True) else 1.0) if hp_pct is None
                              else max(0.0, min(hp_pct, 100.0)) / 100.0,
                   "pos_boosts": sum(v for v in (mon.get("boosts") or {}).values() if v > 0),
                   "fainted_allies": fainted_allies,      # v11 B.2 gap-fix: Last Respects BP
                   "times_attacked": mon.get("times_attacked", 0),   # v11 C.4: Rage Fist BP
                   "loaded_dice": _att_item == "loadeddice",  # v11 A1: 2-5 moves hit 4-5 (P5: gated)
                   "acc_stage": (mon.get("boosts") or {}).get("accuracy", 0),  # v11 B2: attacker accuracy stage
                   "wide_lens": _att_item == "widelens",       # v11 B2: Wide Lens ×1.1 accuracy (P5: gated)
                   "expert_belt": _att_item == "expertbelt",   # v11 B3: ×1.2 super-effective (P5: gated)
                   "type_boost": _TYPE_BOOST_TYPE.get(_att_item),  # v11 B3: ×1.2 matching-type move (P5: gated)
                   "scope_lens": _att_item in ("scopelens", "razorclaw"),  # v11 N4: +1 crit stage (P5: gated)
                   # v11: Victory Star ×1.1 accuracy — the holder OR its active ally has the ability.
                   "victory_star": abil_id == "victorystar" or ally_ability == "victorystar",
                   "side_active": side_active}      # v18 (Option 1): this mon's own-side active conditions
        slots = move_slots_for_mon(mon)
        for m_idx in range(NUM_MOVES):
            if m_idx < len(slots):
                name, confidence = slots[m_idx]
                pp_used = pp_used_map.get(norm_species(name), 0)
                self._write_move_json(vec, i, name, confidence, types, pp_used,
                                      enemy_defenders, att_ctx, field_mods, ability_id=abil_id,
                                      gravity=gravity)
            i += MOVE_FEATURES

        # ── v9 ITEM block: identity index + effect-tag multi-hot + known ──
        # v11 P5: under Magic Room an ACTIVE mon's item is HELD but non-functional → keep the identity index
        # + known confidence (the item is still known), but ZERO the EFFECT tags (no effect applies).
        item_id, item_known = resolve_item_json(mon)
        vec[i] = float(item_index(item_id))
        i += 1
        for idx in item_tag_indices(_item_active(item_id, magic_room and is_active,
                                                 klutz=abil_id == "klutz", embargo=bool(vol.get("embargo")))):
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
        # locked_action.  can_switch = NOT(move-trapped OR rooted OR ability-trapped). v11 B.3: ability
        # trapping (Shadow Tag / Arena Trap / Magnet Pull on an opposing active) now zeroes can_switch. ──
        _item_id = resolve_item_json(mon)[0]
        # Use the ACTIVE ability (abil_id) for BOTH the Levitate (grounded) and Shadow-Tag checks so the
        # offline computation matches the live twin (_live_ability is the active-ability resolver).
        # v11 C.2c: grounded now folds Magnet Rise/Telekinesis + Smack Down/Ingrain/Iron Ball.
        _trap = _ability_trapped(
            "GHOST" in types, "STEEL" in types,
            _is_grounded(types, abil_id, _item_id, bool(vol.get("levitating")),
                         bool(vol.get("force_grounded")) or gravity),     # v11 C.2e: Gravity grounds
            abil_id == "shadowtag", _item_id == "shedshell",
            [d.get("ability") for d in (enemy_defenders or []) if d])
        vec[i] = 1.0 if vol.get("rooted") else 0.0
        vec[i + 1] = 1.0 if vol.get("trapped") else 0.0
        vec[i + 2] = 1.0 if vol.get("has_substitute") else 0.0
        vec[i + 3] = 1.0 if vol.get("move_restricted") else 0.0
        vec[i + 4] = 0.0 if (vol.get("trapped") or vol.get("rooted") or _trap) else 1.0
        vec[i + 5] = 1.0 if vol.get("locked_action") else 0.0
        vec[i + 6] = min(vol.get("protect_counter", 0), _PROTECT_COUNTER_CAP) / _PROTECT_COUNTER_CAP  # v11 A.3
        vec[i + 7] = 1.0 if vol.get("residual_damage") else 0.0    # v11 C.2: Leech Seed/Salt Cure/Curse/Nightmare
        vec[i + 8] = 1.0 if vol.get("confused") else 0.0           # v11 C.2
        vec[i + 9] = vol.get("perish_norm", 0.0)                   # v11 C.2 (already in [0,1])
        vec[i + 10] = 1.0 if vol.get("drowsy") else 0.0            # v11 C.2b: Yawn (sleeps next turn)
        vec[i + 11] = 1.0 if vol.get("first_turn") else 0.0        # v11 C.3: Fake Out/First Impression/Mat Block legal
        i += VOLATILE_FEATURES

        # v11 Phase D (D9): tera_type one-hot (NUM_TYPES). PARITY-clean = the REVEALED tera type only (same
        # condition as _effective_types' tera branch): set iff is_terastallized AND known_tera_type. ⚠ Tera is
        # mod-disabled → PERMANENTLY ZERO today (forward-compat; populated only if the mod ever enables tera).
        if mon.get("is_terastallized") and mon.get("known_tera_type"):
            _tt = _TYPE_IDX.get(_canon(mon["known_tera_type"]))
            if _tt is not None:
                vec[i + _tt] = 1.0
        i += NUM_TYPES

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
        gravity: bool = False,
    ) -> None:
        """JSON twin of _write_move using data/moves.json. ``enemy_defenders`` = the 2 enemy actives'
        defender profiles; ``att_ctx`` = attacker stats/burn/item; ``field_mods`` = (weather, terrain);
        ``ability_id`` = the user's current ability (v9: effective-type change Normalize/Pixilate/-ate).
        ``gravity`` (v11 C.2e) boosts numeric-accuracy moves ×6840/4096 (the Gravity-Hypnosis payoff)."""
        i = start
        _mid = norm_species(move_name)             # v11 N1: move-id key (Champions overrides + the type-eff cross)
        data = _get_moves_data().get(_mid)
        if not data:
            # Unknown move id: only the is_known confidence is encoded
            vec[start + MOVE_FEATURES - 1] = confidence
            return

        vec[i] = min(champ_bp(_mid, data.get("basePower") or 0), 250) / 150.0   # v11 N1: Champions BP override
        i += 1

        mtype = _canon(champ_type(_mid, data.get("type")))              # v11 N1: Champions type (snaptrap Grass→Steel)
        mtype = _canon(field_dependent_move_type(_mid, mtype,           # v19b: Weather Ball / Terrain Pulse follow
                       field_mods[0] if field_mods else None,          #        the weather / terrain (else base type)
                       field_mods[1] if field_mods else None))
        mtype = _DMG.effective_move_type(move_name, mtype, ability_id)   # v9: Normalize/Pixilate/-ate/Liquid Voice
        if mtype in _TYPE_IDX:
            vec[i] = _TYPE_IDX[mtype] / (NUM_TYPES - 1)
        i += 1

        cat = (data.get("category") or "").lower()
        vec[i] = {"physical": 0.0, "special": 0.5}.get(cat, 1.0)
        i += 1

        vec[i] = (data.get("priority") or 0) / 7.0
        i += 1

        acc = champ_acc_raw(_mid, data.get("accuracy"))   # v11 N1: Champions accuracy override (raw percent / True)
        _acc = 1.0 if acc is True else (acc or 0) / 100.0
        if gravity and acc is not True:        # v11 C.2e: Gravity ×6840/4096 on numeric accuracy, cap 1.0
            _acc = min(_acc * _GRAVITY_ACC_MULT, 1.0)
        _base_acc = _acc                       # v11 B2b: post-Gravity numeric base for the per-enemy hit-chance
        if not _move_always_hit(_mid):         # v11 B2: attacker-side accuracy mods (skip always-hit moves)
            _ac2 = att_ctx or {}
            _acc = _accuracy_modifiers(_acc, ability_id=ability_id,
                                       is_physical=(data.get("category") or "").lower() == "physical",
                                       is_ohko=_move_is_ohko(_mid),
                                       acc_stage=_ac2.get("acc_stage", 0), wide_lens=_ac2.get("wide_lens", False),
                                       victory_star=_ac2.get("victory_star", False))   # v11 Victory Star
        vec[i] = _acc
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
        # (_mid computed at the top of the writer for the Champions overrides; reused here.)
        _neg = _immunity_neg_ctx(enemy_defenders, ability_id, _mid)   # v11 C.2d (None per slot = legacy)
        for e in range(2):
            d = enemy_defenders[e] if (enemy_defenders and e < len(enemy_defenders)) else None
            if d and _move_immune(mtype, d, ability_id, _mid):
                signed, immune = -1.0, 1.0          # ability 0× (A.1/A.1b) OR Ground-immune (C.2c)
            else:
                signed, immune = _type_eff_signed_immune(mtype, (d.get("types") if d else []), _neg[e])
                if d and mtype == "FIRE" and immune == 0.0 and d.get("tar_shot"):
                    signed = float(np.clip(signed + 0.5, -1.0, 1.0))   # v11 C.2c: Tar Shot +1 step (×2)
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
        # v11 B.2 stat-override: Psyshock-family (special) hit Def → physical-screen side too.
        _hits_def = _phys or _mid in _DEF_HIT_MOVES
        _raw_bp = champ_bp(_mid, data.get("basePower") or 0)   # v11 N1: Champions BP override (band + variable-BP)
        # v11 A1: multi-hit band scaling — a 2-5 move's min = min-hits × low-roll, max = max-hits × high.
        # Skill Link forces a variable move to max hits; Loaded Dice makes a 2-5 move hit 4-5 (min 4).
        _hmin, _hmax = _move_hit_range(_mid)
        if ability_id == "skilllink" and _hmin != _hmax:
            _hmin = _hmax
        elif _ac.get("loaded_dice") and (_hmin, _hmax) == (2, 5):
            _hmin = 4
        # v11 N4: expected-crit multiplier (Super Luck / Sniper / Scope Lens + high-crit moves) — the band is
        # otherwise crit-blind. Defender-independent → computed once; folded into each enemy's _sit below.
        _crit = _expected_crit_mult(_mid, ability_id, _ac.get("scope_lens"))
        for e in range(2):
            d = enemy_defenders[e] if (enemy_defenders and e < len(enemy_defenders)) else None
            if d and not _move_immune(mtype, d, ability_id, _mid):
                _tmult = _type_mult(mtype, d.get("types") or [], _neg[e])   # v11 C.2d negation
                if mtype == "FIRE" and d.get("tar_shot"):
                    _tmult *= 2.0                                  # v11 C.2c: Tar Shot Fire ×2
                _abm = _ability_damage_mult(                       # v11 A.2 ability damage multipliers
                    _mid, ability_id, d.get("ability"), eff_move_type=mtype, is_stab=_stab,
                    is_physical=_phys, type_mult=_tmult, hp_frac=d.get("hp_frac") or 0.0,
                    att_burned=bool(_ac.get("burned")), att_statused=bool(_ac.get("statused")),
                    is_spread=_spread,                             # v11 A2: Parental Bond eligibility
                    fainted_allies=_ac.get("fainted_allies", 0),  # v11 A3: Supreme Overlord faint-count
                    defender_intact_disguise=d.get("intact_disguise", False),  # v11 B1: Disguise block
                    weather=_weather)                             # v11 G5/G6: Sand Force / Solar Power
                _sit = _situational_damage_mult(mtype, _phys, _weather, _terrain, d,
                                                _ac.get("burned"), _ac.get("life_orb"), _ac.get("choice"),
                                                hits_def=_hits_def,
                                                grassy_eq=_mid in _GRASSY_WEAKENED) * _abm  # v11 G7
                # v11 B3 attacker item band mults (×4915/4096): type-boost on a matching-type move + Expert
                # Belt on a super-effective hit. v11 B3b: defender resist berry ×0.5 on a SE hit of its type
                # (Chilan = all Normal — Normal is never SE). _tmult = the resolved per-enemy type multiplier.
                if _ac.get("type_boost") == mtype:
                    _sit *= _BAND_ITEM_MULT
                if _ac.get("expert_belt") and _tmult > 1.0:
                    _sit *= _BAND_ITEM_MULT
                _rb = d.get("resist_berry")
                if _rb == mtype and (mtype == "NORMAL" or _tmult > 1.0):
                    _sit *= 0.5
                _sit *= _crit                                   # v11 N4: expected-crit EV (band crit-blind otherwise)
                # v11 B.2: variable base power (Low Kick/Heavy Slam/Gyro Ball/Eruption/Stored Power/…) —
                # a no-op (returns _raw_bp) for the ~99% of moves with a fixed BP. Per-enemy context (d).
                _bp = _DMG.variable_base_power(
                    _mid, _raw_bp,
                    attacker_weight=_ac.get("weight"), target_weight=d.get("weight"),
                    attacker_hp_frac=_ac.get("hp_frac"),
                    attacker_speed=_ac.get("eff_speed"), target_speed=d.get("eff_speed"),
                    attacker_pos_boosts=_ac.get("pos_boosts", 0), target_pos_boosts=d.get("pos_boosts", 0),
                    fainted_allies=_ac.get("fainted_allies", 0),      # v11 B.2 gap-fix: Last Respects
                    times_hit=_ac.get("times_attacked", 0))           # v11 C.4: Rage Fist
                # v11 B.2: offensive-stat override — Body Press = user Def, Foul Play = TARGET Atk.
                # (Body Press under Wonder Room is UNAFFECTED — its def->spd override swap and calculateStat's
                #  spd->def read cancel, so it still uses Def: do NOT swap the offensive read.)
                _Aov = (_ac.get("def") if _mid == "bodypress"
                        else d.get("atk") if _mid == "foulplay" else _A)
                _wr = bool(_ac.get("wonder_room"))   # v11 P5: Wonder Room swaps the DEFENSIVE stat (Def<->SpD)
                dmin, dmax = _damage_band(
                    _bp, _Aov, (d.get("def") if (_hits_def ^ _wr) else d.get("spd")),
                    d.get("hp"), d.get("hp_frac"), _tmult,
                    _stab, _spread, _sit, hits_min=_hmin, hits_max=_hmax)   # v11 A1: multi-hit
            else:
                dmin, dmax = 0.0, 0.0                # empty slot OR defender-ability immunity → 0 damage
            vec[i] = dmin
            vec[i + 1] = dmax
            i += 2

        # v11 B.1b: priority-aware who-moves-first vs each enemy active (2 channels). prio_delta = THIS
        # move's priority bracket − the enemy's assumed bracket (0; the enemy move is unknown at decision
        # time). A nonzero bracket hard-overrides speed (the Fake Out / Sucker Punch read), TR-invariant;
        # an equal bracket falls back to the soft speed/TR margin. Empty enemy slot → 0.
        _att_spd = _ac.get("eff_speed") or 0.0
        _prio = data.get("priority") or 0
        _tr = bool(_ac.get("trick_room"))
        for e in range(2):
            d = enemy_defenders[e] if (enemy_defenders and e < len(enemy_defenders)) else None
            if d:
                vec[i] = _moves_first(_att_spd, d.get("eff_speed") or 0.0, _tr, prio_delta=_prio)
            i += 1

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

        # v11 B2b: per-enemy realized HIT CHANCE vs each enemy active (2 channels). Folds the defender's
        # evasion stage / Sand Veil / Snow Cloak / Tangled Feet / Bright Powder + No Guard + the Showdown
        # acc−eva combined boost into the actual hit chance of THIS move vs that enemy. Empty slot → 0.0;
        # an always-hit move → 1.0 (evasion never applies). _base_acc = the post-Gravity numeric accuracy.
        _ah_b2b = _move_always_hit(_mid)
        _oh_b2b = _move_is_ohko(_mid)
        for e in range(2):
            d = enemy_defenders[e] if (enemy_defenders and e < len(enemy_defenders)) else None
            if d is None:
                vec[i] = 0.0
            elif _ah_b2b:
                vec[i] = 1.0
            else:
                vec[i] = _per_enemy_hit_chance(
                    _base_acc, attacker_ability=ability_id, acc_stage=_ac.get("acc_stage", 0),
                    wide_lens=_ac.get("wide_lens", False), is_physical=_phys, is_ohko=_oh_b2b,
                    defender=d, move_id=_mid, weather=_weather,
                    victory_star=_ac.get("victory_star", False))   # v11 Victory Star
            i += 1

        # v18 (Option 1): REDUNDANT-CONDITION bit — 1.0 if this move re-sets a screen/weather/terrain
        # already active on the caster's side (Light Screen under Light Screen, Rain Dance under rain, …).
        # A pure feature: no action is masked, so a deliberate re-cast stays learnable. ``side_active`` is
        # the mon's own-side conditions (att_ctx); _weather/_terrain are the global field (field_mods).
        vec[i] = move_redundant_condition(_mid, (att_ctx or {}).get("side_active") or frozenset(),
                                          _weather, _terrain)
        i += 1

        # v19 (Option 1c): per-ENEMY REDUNDANT-STATUS bits (2). 1.0 if this PURE status move is WASTED on
        # enemy_defenders[e] (target already carries a major status, or is type/ability immune). Reuses the
        # SAME enemy iteration as the type-eff / hit-chance channels; damaging moves & empty slots → 0.0.
        for e in range(2):
            d = enemy_defenders[e] if (enemy_defenders and e < len(enemy_defenders)) else None
            vec[i] = move_redundant_status(_mid, d)
            i += 1

        # v9: move effect-tags (multi-hot priors) + identity index, BEFORE the trailing is_known
        for idx in move_tag_indices(move_name):
            vec[i + idx] = 1.0
        i += NUM_MOVE_TAGS
        vec[i] = float(move_index(move_name))
        i += 1

        # is_known: 1.0 revealed/exact, p(usage) for belief-padded moves
        vec[i] = confidence
        i += 1



# Backward-compatibility alias: the offline encoder kept the name
# StateEncoder before the live/vod split (live encode() now lives in
# live_state_encoder.LiveStateEncoder).
StateEncoder = VodStateEncoder
