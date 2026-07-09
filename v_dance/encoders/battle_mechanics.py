"""Pure battle-mechanics helpers (type chart, damage band, speed, ability/item mults).

AUTO-SPLIT from state_encoder.py (v17 refactor) — PURE verbatim moves; the
frozen tensor layout + parity are unchanged.  state_encoder.py re-exports every
name moved here, so all existing imports keep working.
"""
from __future__ import annotations

import json
import math
import numpy as np
import re
from v_dance.encoders import damage_mechanics as _DMG
from pathlib import Path
from typing import Optional
from v_dance.encoders.champions_move_overrides import CHAMP_ACC, CHAMP_BP, CHAMP_TYPE
from v_dance.parser.vod_parser.pokedex import get_pokedex, norm_species
from v_dance.encoders.encoder_layout import (_ABIL_EFFECT_IDX, _ITEM_EFFECT_IDX, _SCREEN_TO_SC)


_NON_ALNUM = re.compile(r"[^A-Z0-9_]")


def _canon(name: Optional[str]) -> str:
    """'RainDance' → 'RAINDANCE', 'electric' → 'ELECTRIC'."""
    if not name:
        return ""
    return _NON_ALNUM.sub("", str(name).upper().replace(" ", "_"))

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


# ── v11 A.2: canonical Gen-9 MOVE data (the SAME dataset poke-env's live Move wraps) — sourced here so
# the per-move ability-damage props are byte-identical on the offline + live paths (data/moves.json is a
# stale/partial dump: it lacks `secondaries` and `hasCrashDamage`, diverging from live for ~14 moves). ──
_GEN9_MOVES_CACHE = None


def _gen9_moves() -> dict:
    """poke-env's Gen-9 move dict {move_id: entry}, cached. move_id keys match norm_species(name)
    (offline) and poke-env Move.id (live), so both encoders look up the SAME entry (parity)."""
    global _GEN9_MOVES_CACHE
    if _GEN9_MOVES_CACHE is None:
        from poke_env.data import GenData
        _GEN9_MOVES_CACHE = GenData.from_gen(9).moves
    return _GEN9_MOVES_CACHE


# v11 N1 — Champions-mod move overrides. The custom Champions VGC format overrides ~30 moves' BP / type /
# accuracy vs standard Gen-9; BOTH encoders otherwise read standard data (offline GenData(9), live poke-env
# Move). These id-keyed helpers overlay the modded value. move_id == norm_species(name) (offline) == Move.id
# (live) → the overlay is byte-identical on both paths (parity by construction). Imported by the live twin.
# Champions-only by construction (the battle encoder is hard-scoped to Champions; see champions_move_overrides).
# Scraper-baked BP approximations (data/scripts/scrapers/update_moves.py hand-edits moves.json): the OFFLINE
# corpus represents Acrobatics as its no-item 110 (halved when the holder has an item) and Return as its
# max-friendship 102. poke-env's LIVE Move.base_power returns the UN-baked 55 / 0, so without this overlay an
# OWN mon carrying these serves a BP channel + damage band that diverge from the trained corpus (the opp side
# is already neutralized by _splice_opponent). Consulted INSIDE champ_bp so BOTH encoders apply it identically
# (parity by construction); the values MATCH what moves.json already bakes, so the offline corpus is unchanged
# (no re-export / retrain) — this only lifts the LIVE own-mon path up to the value the net was trained on.
_BAKED_BP = {"acrobatics": 110, "return": 102}


def champ_bp(move_id: Optional[str], std):
    """Champions basePower override for ``move_id`` (else the scraper-baked ``_BAKED_BP``, else ``std``).
    Feeds the band + the Technician/-ate gates."""
    mid = move_id or ""
    if mid in CHAMP_BP:
        return CHAMP_BP[mid]
    return _BAKED_BP.get(mid, std)


def champ_type(move_id: Optional[str], std):
    """Champions move-type override (TitleCase, e.g. 'Steel') for ``move_id`` (else ``std``)."""
    return CHAMP_TYPE.get(move_id or "", std)


def champ_acc_raw(move_id: Optional[str], std):
    """Champions accuracy override in RAW moves.ts form (percent int / True) for ``move_id`` (else ``std``)."""
    return CHAMP_ACC.get(move_id or "", std)


def _move_damage_props(move_id: Optional[str]) -> dict:
    """The move properties the A.2 ability multipliers gate on, read from GenData(9).moves so the
    booleans are byte-identical offline↔live. has_recoil_or_crash folds recoil/hasCrashDamage/
    struggleRecoil (matching live Move.recoil which returns 0.25 for struggleRecoil); has_secondaries
    folds singular `secondary` + plural `secondaries` (matching live Move.secondary's merged list);
    raw_is_normal is the PRE-`effective_move_type` type (for the -ate boost gate)."""
    e = _gen9_moves().get(move_id or "") or {}
    fl = e.get("flags") or {}
    return {
        "is_contact": bool(fl.get("contact")),
        "is_punch": bool(fl.get("punch")),
        "is_bite": bool(fl.get("bite")),
        "is_slicing": bool(fl.get("slicing")),
        "is_sound": bool(fl.get("sound")),
        "is_bullet": bool(fl.get("bullet")),     # v11 A.1b: Bulletproof immunity
        "is_wind": bool(fl.get("wind")),         # v11 A.1b: Wind Rider immunity
        "has_recoil_or_crash": bool(e.get("recoil") or e.get("hasCrashDamage") or e.get("struggleRecoil")),
        "has_secondaries": bool(e.get("secondary") or e.get("secondaries")),
        "raw_is_normal": (champ_type(move_id, e.get("type")) == "Normal"),   # v11 N1: -ate gate
        "base_power": champ_bp(move_id, e.get("basePower") or 0),            # v11 N1: Technician <=60 gate
    }


def _move_hit_range(move_id: Optional[str]) -> tuple[int, int]:
    """(min_hits, max_hits) for a move, from the shared GenData (byte-identical offline↔live). Single-hit
    → (1,1); a fixed N-hit move (Dragon Darts=2, Triple Axel=3, Population Bomb=10) → (N,N); a variable
    move (Bullet Seed/Rock Blast/Icicle Spear/Scale Shot/Water Shuriken) → its [min,max] (usually (2,5)).
    v11 A1: drives the multi-hit DAMAGE-BAND scaling (the band applies BP once; a 2-5 move did ~1/N)."""
    mh = (_gen9_moves().get(move_id or "") or {}).get("multihit")
    if isinstance(mh, (list, tuple)) and len(mh) == 2:
        return int(mh[0]), int(mh[1])
    if isinstance(mh, int) and mh:
        return int(mh), int(mh)
    return 1, 1


def _effective_types(mon: Optional[dict]) -> list[str]:
    """A mon's CURRENT effective types (canonical upper): TERA overrides (ACTIVE in the Champions
    format), else a runtime TYPECHANGE (v11 P2: Protean/Libero/Color Change/Soak/Conversion/Burn Up/
    Double Shock), else the dex types of the current/copied forme. The SINGLE source shared by the
    per-mon type one-hots AND the per-move type-eff, so the defender typing stays in lockstep."""
    if not mon:
        return []
    if mon.get("is_terastallized") and mon.get("known_tera_type"):
        return [_canon(mon["known_tera_type"])]
    # v11 P2: a runtime typechange REPLACES the dex types until switch/tera/transform. 2nd precedence
    # (below tera, above transform/dex), mirroring poke-env Pokemon.type_1/type_2 (tera > _temporary_types
    # > dex). The parser already clears runtime_types on tera/transform, so the ordering can't conflict.
    rt = mon.get("runtime_types")
    if rt:
        return [_canon(t) for t in rt]
    dex_species = (mon["transformed_into"] if (mon.get("is_transformed") and mon.get("transformed_into"))
                   else mon.get("species"))
    return _dex_types(dex_species) or _dex_types(mon.get("base_species")) or []


# ── v11 C.2d — TYPE-IMMUNITY NEGATION ───────────────────────────────────────────────────────────
# Game rules that turn a per-(move_type → defender_type) 0× TYPING immunity into 1× at decision time.
# ``neg`` is a frozenset of active tokens for THIS (move, defender) or None (the ~99% byte-identical
# fast path). Tokens: 'ringtarget' (blanket — Ring Target negates ALL the holder's chart immunities),
# 'force_ground' (Smack Down/Ingrain/Iron Ball/Thousand Arrows → GROUND-vs-FLYING only), 'ghost'
# (Scrappy/Mind's Eye attacker OR Foresight/Odor Sleuth defender → NORMAL/FIGHTING vs GHOST), 'dark'
# (Miracle Eye defender → PSYCHIC vs DARK). Mold-Breaker-INDEPENDENT (these are typing, not abilities).
def _immunity_negated(move_type: str, defender_type: str, neg) -> bool:
    if not neg:
        return False
    mt, dt = move_type.upper(), defender_type.upper()
    if "ringtarget" in neg:
        return True
    if "force_ground" in neg and mt == "GROUND" and dt == "FLYING":
        return True
    if "ghost" in neg and dt == "GHOST" and mt in ("NORMAL", "FIGHTING"):
        return True
    if "dark" in neg and mt == "PSYCHIC" and dt == "DARK":
        return True
    return False


def _move_ground_pierces(move_id: Optional[str]) -> bool:
    """True iff the MOVE intrinsically ignores Ground immunity (Thousand Arrows, ignoreImmunity
    {'Ground': True}). Read from the shared _gen9_moves() dict so it is byte-identical offline↔live —
    NOT the live Move.ignore_immunity attribute (a PokemonType set, which would diverge)."""
    ii = (_gen9_moves().get(move_id or "") or {}).get("ignoreImmunity")
    return isinstance(ii, dict) and bool(ii.get("Ground"))


def _move_ignores_all_immunity(move_id: Optional[str]) -> bool:
    """True iff the MOVE ignores ALL type immunities (``ignoreImmunity: true``) — the standard 120-BP
    Psychic move FUTURE SIGHT (which therefore hits Dark). Same blanket effect as Ring Target on the
    chart; read from the shared _gen9_moves() dict for offline↔live parity."""
    return _gen9_moves().get(move_id or "", {}).get("ignoreImmunity") is True


def _immunity_neg_ctx(enemy_defenders, attacker_ability, move_id) -> list:
    """Per-enemy-slot C.2d negation tokens for THIS move — a 2-element list of (frozenset | None);
    None = no negation active (the byte-identical fast path). SHARED by the offline + live move-writers
    (imported by the live twin) so the token set is identical. Attacker side: Scrappy / Mind's Eye (and
    Thousand Arrows' Ground-pierce) are move/attacker-wide; defender side: Ring Target / force-ground /
    Foresight / Miracle Eye are read per slot. Foresight is Ghost-only and Miracle Eye Dark-only — but
    _immunity_negated already gates on the defender TYPE, so passing the token unconditionally is safe."""
    ghost_att = attacker_ability in ("scrappy", "mindseye")
    ta_pierce = _move_ground_pierces(move_id)
    all_ignore = _move_ignores_all_immunity(move_id)      # Future Sight: blanket any-0× → 1×
    out = []
    for e in range(2):
        d = enemy_defenders[e] if (enemy_defenders and e < len(enemy_defenders)) else None
        if not d:
            out.append(None)
            continue
        f = []
        if d.get("ring_target") or all_ignore:
            f.append("ringtarget")
        if d.get("force_grounded") or ta_pierce:
            f.append("force_ground")
        if ghost_att or d.get("foresight"):
            f.append("ghost")
        if d.get("miracleeye"):
            f.append("dark")
        out.append(frozenset(f) if f else None)
    return out


def _type_mult(move_type: str, defender_types: list[str], neg=None) -> float:
    """Raw type-eff multiplier (product over the defender's types) of an attacking ``move_type``;
    1.0 when either side is unknown. Shared by the signed type-eff channel AND the damage band. ``neg``
    (v11 C.2d) negates the relevant per-type 0× → 1× (default None = byte-identical legacy)."""
    if not defender_types or not move_type:
        return 1.0
    chart = _type_chart()
    at = move_type.upper()
    mult = 1.0
    for dt in defender_types:
        m = float(chart.get(dt.upper(), {}).get(at, 1.0))
        if m == 0.0 and neg is not None and _immunity_negated(move_type, dt, neg):
            m = 1.0
        mult *= m
    return mult


def _type_eff_signed_immune(move_type: str, defender_types: list[str], neg=None) -> tuple[float, float]:
    """(signed_mult, immune) of ``move_type`` vs a defender. signed = clip(log2(mult)/2, -1, 1) for
    mult>0 → 4×=+1 · 2×=+0.5 · 1×=0 · 0.5×=-0.5 · 0.25×=-1; a 0× TYPING immunity (Ground→Flying, tera)
    → signed -1 AND immune 1, so 0× stays DISTINCT from 0.25×. Ability immunities (Levitate etc.)
    fold into the resolved damage band (B1.2), not here — this is the pure typing cross. ``neg`` (v11
    C.2d) forwards to _type_mult so the channel and band agree on a negated immunity."""
    if not defender_types or not move_type:
        return 0.0, 0.0
    mult = _type_mult(move_type, defender_types, neg)
    if mult <= 0.0:
        return -1.0, 1.0
    return float(np.clip(math.log2(mult) / 2.0, -1.0, 1.0)), 0.0


def _damage_band(base_power, A, D, hp_stat, hp_frac, type_mult, is_stab, is_spread,
                 situational_mult=1.0, hits_min=1, hits_max=1) -> tuple[float, float]:
    """(min, max) damage as a fraction of the defender's CURRENT HP, clamped [0,1] — the L50 Gen-9
    formula folding base_power·(A/D)·STAB·type_mult·spread·situational(B1.2b: weather/terrain/screens/
    item/burn)·the 0.85–1.0 roll. (0,0) for a non-damaging move or missing belief. Belief-dependent →
    rides stats_known. Defender-ability resists (Thick Fat/Multiscale/Filter) fold in via the A.2
    ``_ability_damage_mult`` (passed as situational_mult), NOT here. v11 A1: ``hits_min``/``hits_max``
    scale the band for a MULTI-HIT move (a 2–5 move's min = 2 hits at the low roll, max = 5 at the high)."""
    if (not base_power or base_power <= 0 or not A or not D
            or not hp_stat or not hp_frac or hp_frac <= 0):
        return 0.0, 0.0
    base = ((2 * 50 / 5 + 2) * base_power * A / D) / 50.0 + 2.0
    mult = type_mult * (1.5 if is_stab else 1.0) * (0.75 if is_spread else 1.0) * situational_mult
    cur_hp = hp_stat * hp_frac
    if cur_hp <= 0:
        return 0.0, 0.0
    return (float(np.clip(base * mult * 0.85 * hits_min / cur_hp, 0.0, 1.0)),
            float(np.clip(base * mult * hits_max / cur_hp, 0.0, 1.0)))


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


# v11 P5/Klutz/Embargo: item-effect suppression (Showdown Pokemon.ignoringItem) → resolve the item to None
# so every downstream read (damage-band mults, accuracy, speed, defensive mults, the per-mon effect channels,
# the AV/Choice action-mask) sees no item. Suppressed by: field Magic Room (``suppressed``) · the Klutz
# ability · the Embargo volatile. Klutz suppresses every item EXCEPT the ignoreKlutz set (Ability Shield /
# Macho Brace / Power items) — NONE of which is an ENCODED effect, so the encoder treats Klutz as full item
# suppression. Shared by both encoders (live imports it); callers pass the per-mon klutz/embargo flags.
def _item_active(item_id, suppressed: bool, klutz: bool = False, embargo: bool = False):
    return None if (suppressed or klutz or embargo) else item_id


# v11 N4: crit system (the damage band is otherwise crit-BLIND by design). Gen-9 crit chance by stage
# (sim/battle-actions.ts critMult): p_crit = 1/_CRIT_MULT[stage]; a crit deals ×1.5 (Sniper stacks → ×2.25).
# stage = move.critRatio (base 1) + Super Luck (+1) + Scope Lens/Razor Claw (+1), capped at 4. critRatio is
# read from the SHARED _gen9_moves (GenData, what poke-env wraps) so it is byte-identical offline↔live.
_CRIT_MULT = (0, 24, 8, 2, 1)


def _expected_crit_mult(move_id, attacker_ability, scope_lens) -> float:
    """Deterministic expected-damage multiplier from the crit system, folded into the situational band mult.
    Scope Lens is an ITEM so the caller passes the Magic-Room-gated value → suppressed under Magic Room."""
    stage = (_gen9_moves().get(move_id or "", {}).get("critRatio") or 1)
    if attacker_ability == "superluck":
        stage += 1
    if scope_lens:
        stage += 1
    stage = max(1, min(stage, 4))
    return 1.0 + (1.0 / _CRIT_MULT[stage]) * (1.25 if attacker_ability == "sniper" else 0.5)


def _defender_profile(mon: Optional[dict], side_screens: tuple = (False, False),
                      side_tailwind: bool = False, weather: Optional[str] = None,
                      gravity: bool = False, magic_room: bool = False) -> Optional[dict]:
    """Uniform DEFENDER profile {types, def, spd, hp, hp_frac, grounded, screen_phys, screen_spec,
    eff_speed} for the per-move type-eff + damage + who-moves-first cross. None for an empty slot. Belief
    est stats; hp_frac mirrors the per-mon hp encoding. grounded = static v1 (not Flying / Levitate /
    Air-Balloon; volatiles skipped). eff_speed (v11 B.1b) = the resolved in-battle speed for the
    move-block who-moves-first channel. ``gravity`` (v11 C.2e) force-grounds EVERY active (isGrounded
    checks Gravity first) → feeds grounding + the C.2d type-immunity negation + ground_immune."""
    if not mon:
        return None
    est = (mon.get("stats_estimate") or {}).get("stats") or {}
    hp_pct = mon.get("hp_pct")
    if hp_pct is None:
        hp_frac = 0.0 if mon.get("seen", True) else 1.0
    else:
        hp_frac = max(0.0, min(hp_pct, 100.0)) / 100.0
    types = _effective_types(mon)
    _vol = mon.get("volatiles") or {}
    # v11 A.1: the ACTIVE ability (mega forme ability included) — parity twin of the live _live_ability.
    _active_ab = resolve_active_ability_json(mon)[0]
    _item = _item_active(resolve_item_json(mon)[0], magic_room,            # v11 P5: Magic Room
                         klutz=_active_ab == "klutz",                      # v11 Klutz: suppress held item
                         embargo=bool(_vol.get("embargo")))                # v11 Embargo volatile
    _levit, _fg = bool(_vol.get("levitating")), bool(_vol.get("force_grounded"))
    _fg_g = _fg or gravity            # v11 C.2e: Gravity grounds everything (isGrounded checks it first)
    # v11 C.2c: grounded now folds Magnet Rise/Telekinesis (ungrounded) + Smack Down/Ingrain/Iron Ball
    # (force-grounded). Uses the ACTIVE ability for Levitate (parity with live _live_ability).
    grounded = _is_grounded(types, _active_ab, _item, _levit, _fg_g)
    _dex_sp = (mon["transformed_into"] if (mon.get("is_transformed") and mon.get("transformed_into"))
               else mon.get("species"))
    return {"types": types, "def": est.get("def"), "spd": est.get("spd"), "hp": est.get("hp"),
            "hp_frac": hp_frac, "grounded": grounded, "ability": _active_ab,
            "status": mon.get("status"),    # v19 Option 1c: token ('brn'/…); helper lowercases (live = enum.name)
            "screen_phys": side_screens[0], "screen_spec": side_screens[1],
            "eff_speed": _effective_speed(mon, side_tailwind, weather, magic_room)[0],   # v11 B.1b / P5
            # v11 B.2: target Atk (Foul Play), raw weight (Low Kick / Heavy Slam), positive-boost sum
            # (Punishment) — for the variable-BP + stat-override damage band.
            "atk": est.get("atk"),
            "weight": _DMG.species_weight(_dex_sp) or _DMG.species_weight(mon.get("base_species")) or 0.0,
            "pos_boosts": sum(v for v in (mon.get("boosts") or {}).values() if v > 0),
            # v11 C.2c: Ground-move immunity (Air Balloon / Magnet Rise / Telekinesis) + Tar Shot Fire ×2.
            "ground_immune": _ground_immune(_item, _levit, _fg_g),
            "tar_shot": bool(_vol.get("tar_shot")),
            # v11 C.2d: type-immunity NEGATION inputs (force-ground incl. Iron Ball / Gravity; Ring Target
            # item; Foresight/Miracle Eye defender volatiles). Consumed by the move-writer's chart negation.
            "force_grounded": _fg_g or _item == "ironball",
            "ring_target": _item == "ringtarget",
            "foresight": bool(_vol.get("foresight")),
            "miracleeye": bool(_vol.get("miracleeye")),
            # v11 B1: intact Mimikyu Disguise → the move-writer zeros the band (Mold Breaker bypasses).
            "intact_disguise": _disguise_intact(_active_ab, hp_frac),
            # v11 B2b: defender-side accuracy/evasion inputs for the per-enemy hit-chance channel.
            "eva_stage": (mon.get("boosts") or {}).get("evasion", 0),
            "confused": bool(_vol.get("confused")),
            "no_guard": _active_ab == "noguard",
            "evasion_item": _item in ("brightpowder", "laxincense"),
            # v11 B3: defensive items — Assault Vest (SpD ×1.5) / Eviolite (Def+SpD ×1.5, NFE) → incoming ×2/3
            # (folded in _situational_damage_mult); resist berry (×0.5 on a SE hit of its type / all Normal for
            # Chilan) folded in the move-writer band (needs the per-enemy type_mult).
            "assault_vest": _item == "assaultvest",
            "evio": _item == "eviolite" and _DMG.is_nfe(_dex_sp),
            "resist_berry": _RESIST_BERRY_TYPE.get(_item)}


# B1.2b weather-speed abilities → the weather they double speed in. ⚠ These key on the RAW poke-env /
# parser weather token that _effective_speed / _live_effective_speed actually receive (un-aliased: offline
# _canon(parser_token), live Weather.name), NOT the encoder's canonical one-hot token. For gen-9 SNOW that
# raw token is "SNOWSCAPE" (poke-env Weather.SNOWSCAPE; Showdown emits |-weather|Snowscape) — the old
# "SNOW" was the canonical-one-hot alias target and never matched here, so Slush Rush never boosted.
_WEATHER_SPEED_ABILITY = {"swiftswim": ("RAINDANCE", "PRIMORDIALSEA"),       # also extreme rain (audit 2026-06-30)
                          "chlorophyll": ("SUNNYDAY", "DESOLATELAND"),       # also extreme sun
                          "sandrush": ("SANDSTORM",), "slushrush": ("SNOWSCAPE", "HAIL")}

# v11 B.2: SPECIAL moves that hit the target's physical DEF instead of SpD (overrideDefensiveStat:'def').
_DEF_HIT_MOVES = frozenset({"psyshock", "psystrike", "secretsword"})

# v11 G7: Grassy Terrain halves these moves vs a GROUNDED target (moves.ts grassyterrain onBasePower
# chainModify(0.5) for ['earthquake','bulldoze','magnitude'] when defender.isGrounded()).
_GRASSY_WEAKENED = frozenset({"earthquake", "bulldoze", "magnitude"})


def _situational_damage_mult(move_type, is_physical, weather, terrain, defender,
                             att_burned, att_life_orb, att_choice, hits_def=None,
                             grassy_eq: bool = False) -> float:
    """B1.2b damage situational multiplier: weather (Sun/Rain on Fire/Water) · terrain ×1.3 (grounded
    defender, matching type) + Misty ×0.5 Dragon (grounded; v11 G4) + Grassy ×0.5 Earthquake/Bulldoze/
    Magnitude (grounded; v11 G7) · N3 DEFENSIVE weather (Sand ×2/3 on the SpD-side vs a ROCK defender,
    Snowscape ×2/3 on the Def-side vs an ICE defender) · screens ×0.667 (doubles) · Life Orb ×1.3 ·
    Choice Band/Specs ×1.5 · burn ×0.5 (physical). Defender-ability resists NOT modelled here.
    v11 B.2: ``hits_def`` selects the SCREEN side — None keeps the legacy category-based pick, True forces
    the physical (Reflect) side for a special move that hits Def (Psyshock/Psystrike/Secret Sword).
    v11 G7: ``grassy_eq`` (caller: move_id in _GRASSY_WEAKENED) → ×0.5 under Grassy Terrain vs a grounded
    defender (the semi-invulnerable exemption is a rare two-turn state the band does not model)."""
    mult = 1.0
    mt = (move_type or "").upper()
    # audit: extreme weather (Desolate Land harsh sun / Primordial Sea heavy rain) acts like Sun/Rain for
    # the Fire/Water modifiers AND NULLS the opposite type (Water=0 under harsh sun, Fire=0 under heavy rain).
    # Both encoders share this helper (parity-clean). (Dormant in Reg-M — restricted-legend setters only.)
    if weather in ("SUNNYDAY", "DESOLATELAND"):
        _opp = 0.0 if weather == "DESOLATELAND" else 0.5
        mult *= 1.5 if mt == "FIRE" else (_opp if mt == "WATER" else 1.0)
    elif weather in ("RAINDANCE", "PRIMORDIALSEA"):
        _opp = 0.0 if weather == "PRIMORDIALSEA" else 0.5
        mult *= 1.5 if mt == "WATER" else (_opp if mt == "FIRE" else 1.0)
    if defender and defender.get("grounded"):
        if (terrain == "ELECTRIC_TERRAIN" and mt == "ELECTRIC") \
                or (terrain == "GRASSY_TERRAIN" and mt == "GRASS") \
                or (terrain == "PSYCHIC_TERRAIN" and mt == "PSYCHIC"):
            mult *= 1.3
        elif terrain == "MISTY_TERRAIN" and mt == "DRAGON":
            mult *= 0.5                                # v11 gap-scan G4: Misty Terrain halves Dragon (grounded)
        if grassy_eq and terrain == "GRASSY_TERRAIN":
            mult *= 0.5                                # v11 G7: Grassy Terrain halves EQ/Bulldoze/Magnitude (grounded)
    _phys_side = is_physical if hits_def is None else hits_def
    # v11 N3: DEFENSIVE weather stat boosts (conditions.ts). Sandstorm onModifySpD ×1.5 for a ROCK defender →
    # incoming SpD-side damage ×2/3; Snowscape onModifyDef ×1.5 for an ICE defender → incoming Def-side damage
    # ×2/3. ⚠ The band weather token is the PRE-alias "SNOWSCAPE"/"SANDSTORM" on BOTH paths (never "SNOW" — the
    # SNOWSCAPE→SNOW alias is applied only at the global one-hot, not here). Snow ONLY (hail has no Def boost).
    # _phys_side = "hits Def" → Sand keys on the SpD side (not _phys_side), Snow on the Def side (_phys_side).
    if defender:
        _dtypes = defender.get("types") or []
        if weather == "SANDSTORM" and not _phys_side and "ROCK" in _dtypes:
            mult *= 2.0 / 3.0
        elif weather == "SNOWSCAPE" and _phys_side and "ICE" in _dtypes:
            mult *= 2.0 / 3.0
        # v11 B3: defensive ITEMS (items.ts). Assault Vest SpD ×1.5 → incoming SpD-side dmg ×2/3 (any holder);
        # Eviolite Def AND SpD ×1.5 (NFE holder) → both sides ×2/3. Stacks with N3 weather + screens (independent
        # chainModify). No new arg → both writers call the shared helper unchanged = parity automatic.
        if defender.get("evio") or (defender.get("assault_vest") and not _phys_side):
            mult *= 2.0 / 3.0
    if defender and ((_phys_side and defender.get("screen_phys"))
                     or (not _phys_side and defender.get("screen_spec"))):
        mult *= 0.667                                  # doubles screen reduction (≈2732/4096)
    if att_life_orb:
        mult *= 1.3
    if att_choice:
        mult *= 1.5                                    # Choice Band/Specs (not Scarf) boost the A stat
    if att_burned and is_physical:
        mult *= 0.5
    return mult


def _accuracy_modifiers(base_acc: float, *, ability_id: Optional[str], is_physical: bool,
                        is_ohko: bool, acc_stage: int, wide_lens: bool,
                        victory_star: bool = False) -> float:
    """v11 B2: ATTACKER-side accuracy modifiers folded into the (numeric) per-move accuracy channel — the
    attacker's accuracy boost STAGE (3+n)/3 or 3/(3-n), Compound Eyes ×1.3, Hustle ×0.8 (PHYSICAL only — its
    Atk ×1.5 is the separate _DMG_BOOST_AB entry, so no double-count), Wide Lens ×1.1, No Guard → always-hit.
    Exact chainModify rationals (abilities.ts / items.ts). OHKO moves left as-is (gap G3). The caller passes
    the post-Gravity numeric acc and skips this for always-hit moves (acc is True). DEFENDER-side evasion
    (Sand Veil/Snow Cloak/evasion stage) is per-enemy → NOT here (deferred, would need a layout channel).
    SHARED by both encoders (parity by construction)."""
    if is_ohko:
        return base_acc
    if ability_id == "noguard":
        return 1.0
    acc = base_acc
    n = max(-6, min(6, acc_stage or 0))
    if n > 0:
        acc *= (3 + n) / 3.0
    elif n < 0:
        acc *= 3.0 / (3 - n)
    if ability_id == "compoundeyes":
        acc *= 5325 / 4096                            # ≈1.30005
    elif ability_id == "hustle" and is_physical:
        acc *= 3277 / 4096                            # ≈0.80005
    if wide_lens:
        acc *= 4505 / 4096                            # ≈1.09985
    if victory_star:
        acc *= 4506 / 4096                            # v11: Victory Star ×1.1 (holder OR ally; abilities.ts chainModify([4506,4096]))
    return min(acc, 1.0)


def _move_always_hit(move_id: Optional[str]) -> bool:
    """v11 B2: True iff the move ALWAYS hits (accuracy is True / never misses). Sourced from the shared
    GenData(9) + the Champions override so BOTH encoders agree by construction (poke-env coerces always-hit
    to a 1.0 float, indistinguishable from a 100%-numeric move, so the live path can't use move.accuracy for
    this). The B2 accuracy fold is SKIPPED for these (a reduction like Hustle must not lower an always-hit move)."""
    return champ_acc_raw(move_id, (_gen9_moves().get(move_id or "", {}) or {}).get("accuracy")) is True


def _move_is_ohko(move_id: Optional[str]) -> bool:
    """v11 B2 (gap G3): True iff the move is a one-hit-KO (Fissure/Sheer Cold/Guillotine/Horn Drill). Sourced
    from the shared GenData(9) (moves.json may lack the ``ohko`` field → using it would drift). The B2 accuracy
    fold leaves OHKO moves' 30%-accuracy channel untouched (their accuracy is a separate mechanic)."""
    return bool((_gen9_moves().get(move_id or "", {}) or {}).get("ohko"))


# v11 B2b: attacker abilities that make the move IGNORE the defender's evasion STAGE (Keen Eye/Illuminate/
# Mind's Eye set move.ignoreEvasion). Sand Veil/Snow Cloak/Tangled Feet are SEPARATE accuracy mults, NOT negated.
_IGNORE_EVA_AB = frozenset({"keeneye", "illuminate", "mindseye"})


def _move_ignores_evasion(move_id: Optional[str]) -> bool:
    """v11 B2b: True iff the move ignores the defender's evasion stage (ignoreEvasion: Chip Away/Sacred Sword/
    Darkest Lariat/Nihil Light). Shared GenData (parity by construction; mirror of _move_always_hit/_is_ohko)."""
    return (_gen9_moves().get(move_id or "", {}) or {}).get("ignoreEvasion") is True


def _per_enemy_hit_chance(base_acc: float, *, attacker_ability: Optional[str], acc_stage: int,
                          wide_lens: bool, is_physical: bool, is_ohko: bool, defender: dict,
                          move_id: Optional[str], weather: Optional[str],
                          victory_star: bool = False) -> float:
    """v11 B2b: realized HIT CHANCE of THIS move vs ONE enemy active, Showdown-faithful (sim/battle-actions.ts
    :706-722). ``base_acc`` = the POST-Gravity numeric accuracy in [0,1] WITHOUT the B2a attacker fold (this
    recomputes the whole chain from scratch). Order: No Guard (attacker OR defender) → 1.0 (non-multiplicative
    override); OHKO → base_acc (bypasses all accuracy mods + evasion); attacker ModifyAccuracy mults (Compound
    Eyes/Hustle-phys/Wide Lens) × defender mults (Sand Veil/Snow Cloak in the matching weather / Tangled Feet if
    confused / Bright Powder·Lax Incense item); then ONE combined boost = clamp(clamp(acc_stage,-6,6) − eva_stage,
    -6,6) — the eva-stage is DROPPED if the attacker ignores evasion (Keen Eye/Illuminate/Mind's Eye) or the move
    does (ignoreEvasion); min(,1.0). Always-hit moves are handled by the CALLER (→1.0; evasion never applies).
    SHARED by both encoders → byte-parity by construction."""
    if attacker_ability == "noguard" or defender.get("no_guard"):
        return 1.0
    if is_ohko:
        return base_acc
    acc = base_acc
    if attacker_ability == "compoundeyes":
        acc *= 5325 / 4096
    elif attacker_ability == "hustle" and is_physical:
        acc *= 3277 / 4096
    if wide_lens:
        acc *= 4505 / 4096
    if victory_star:
        acc *= 4506 / 4096                            # v11: Victory Star ×1.1 (holder OR ally)
    _dab = defender.get("ability")
    if _dab == "sandveil" and weather == "SANDSTORM":
        acc *= 3277 / 4096
    elif _dab == "snowcloak" and weather in ("HAIL", "SNOWSCAPE"):
        acc *= 3277 / 4096
    if _dab == "tangledfeet" and defender.get("confused"):
        acc *= 0.5
    if defender.get("evasion_item"):                     # v11 B2b gap #2: Bright Powder / Lax Incense ≈×0.9
        acc *= 3686 / 4096
    boost = max(-6, min(6, acc_stage or 0))
    if not (attacker_ability in _IGNORE_EVA_AB or _move_ignores_evasion(move_id)):
        boost = max(-6, min(6, boost - (defender.get("eva_stage", 0) or 0)))
    if boost > 0:
        acc *= (3 + boost) / 3.0
    elif boost < 0:
        acc *= 3.0 / (3 - boost)
    return min(acc, 1.0)


def _effective_speed(mon: Optional[dict], side_tailwind: bool,
                     weather: Optional[str] = None, magic_room: bool = False) -> tuple[float, float]:
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
    # v11 B.1 (#4): Quick Feet ×1.5 on ANY status AND it NEGATES the paralysis ×0.5 (Showdown skips the
    # para Spe cut for Quick Feet holders — so a paralyzed Quick Feet mon is ×1.5, not ×0.75).
    # T2.1 parity fix: resolve through resolve_ACTIVE_ability_json (the mega forme's ability), NOT the
    # base resolve_ability_json — the live path uses poke-env mon.ability (= the active/mega ability), so a
    # mega whose forme ability is a speed ability differing from its base (e.g. Mega Swampert: Torrent→Swift
    # Swim) diverged offline↔live on eff_speed under weather. Active resolver matches live + is correct.
    _status = _canon(mon.get("status"))
    if resolve_active_ability_json(mon)[0] == "quickfeet" and _status:
        spe *= 1.5
    elif _status == "PAR":
        spe *= 0.5
    item_id, _ = resolve_item_json(mon)
    item_id = _item_active(item_id, magic_room,               # v11 P5: Magic Room suppresses Choice Scarf
                           klutz=resolve_active_ability_json(mon)[0] == "klutz",   # v11 Klutz (T2.1: active)
                           embargo=bool((mon.get("volatiles") or {}).get("embargo")))  # v11 Embargo
    if _ITEM_EFFECT_IDX.get("choice_speed") in item_effect_indices(item_id):
        spe *= 1.5
    if side_tailwind:
        spe *= 2.0
    if weather and weather in _WEATHER_SPEED_ABILITY.get(resolve_active_ability_json(mon)[0], ()):  # T2.1: active
        spe *= 2.0
    # v11 B.1 (#4): Protosynthesis/Quark Drive (incl. Booster Energy) ×1.5 when boosting SPEED — gated on
    # the SPE-suffixed paradox volatile (volatile_flags.paradox_speed). A non-spe paradox boost (atk/spa/…)
    # gives ×1.3 on THAT stat and ×1.0 speed → must NOT trigger here (the ×1.3 case is the damage phase).
    if (mon.get("volatiles") or {}).get("paradox_speed"):
        spe *= 1.5
    return float(spe), known


def _moves_first(spd_a: float, spd_b: float, trick_room: bool, prio_delta: int = 0) -> float:
    """Soft signed margin in [-1,1]: tanh(log(spd_a/spd_b)) with the Trick-Room sign-flip (positive →
    a moves first). 0 for a tie / missing speed — NEVER a hard ±1, so a surprise-Scarf / min-speed-TR
    read stays representable (the panel's over-anchoring guard).

    v11 B.1b: ``prio_delta`` = a's move priority bracket − b's (default 0). A nonzero bracket OVERRIDES
    speed outright and is NEVER flipped by Trick Room (TR only reorders WITHIN a bracket), so a higher
    bracket returns a hard +1 (lower → −1). The soft speed/TR margin only applies in the equal-bracket
    (prio_delta == 0) case — the existing behaviour, so the move-agnostic GLOBAL turn-order block (which
    always passes prio_delta 0) is byte-unchanged."""
    if prio_delta:
        return 1.0 if prio_delta > 0 else -1.0
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
# v11 B3: item id → boosted TYPE (the 40 type-boost items: 23 incense/herb + 17 Arceus plates). onBasePower
# ×4915/4096 (≈1.2) for a matching-type move. Species-locked Orbs/Crystals (Adamant/Lustrous/Griseous/Soul Dew)
# + Ogerpon masks EXCLUDED (forme/species-gated, never fire for a normal holder). The full explicit MAP — the
# tag set above is endswith("plate")-incomplete and is for tagging only.
_TYPE_BOOST_TYPE = {
    "charcoal": "FIRE", "flameplate": "FIRE",
    "mysticwater": "WATER", "seaincense": "WATER", "waveincense": "WATER", "splashplate": "WATER",
    "miracleseed": "GRASS", "roseincense": "GRASS", "meadowplate": "GRASS",
    "magnet": "ELECTRIC", "zapplate": "ELECTRIC",
    "blackbelt": "FIGHTING", "fistplate": "FIGHTING",
    "blackglasses": "DARK", "dreadplate": "DARK",
    "sharpbeak": "FLYING", "skyplate": "FLYING",
    "poisonbarb": "POISON", "toxicplate": "POISON",
    "softsand": "GROUND", "earthplate": "GROUND",
    "hardstone": "ROCK", "rockincense": "ROCK", "stoneplate": "ROCK",
    "silverpowder": "BUG", "insectplate": "BUG",
    "spelltag": "GHOST", "spookyplate": "GHOST",
    "dragonfang": "DRAGON", "dracoplate": "DRAGON",
    "metalcoat": "STEEL", "ironplate": "STEEL",
    "twistedspoon": "PSYCHIC", "oddincense": "PSYCHIC", "mindplate": "PSYCHIC",
    "nevermeltice": "ICE", "icicleplate": "ICE",
    "silkscarf": "NORMAL",
    "fairyfeather": "FAIRY", "pixieplate": "FAIRY",
}
# v11 B3b (gap-scan): resist berry id → resisted TYPE. onSourceModifyDamage ×0.5 on a SUPER-EFFECTIVE hit of
# that type; ⚠ Chilan (NORMAL) is NOT SE-gated — it halves ALL Normal damage (Normal is never SE, so it's the
# always-case). Consumed one-time, but the snapshot reads the CURRENT item (gone if eaten → no fold).
_RESIST_BERRY_TYPE = {
    "occaberry": "FIRE", "passhoberry": "WATER", "wacanberry": "ELECTRIC", "rindoberry": "GRASS",
    "yacheberry": "ICE", "chopleberry": "FIGHTING", "kebiaberry": "POISON", "shucaberry": "GROUND",
    "cobaberry": "FLYING", "payapaberry": "PSYCHIC", "tangaberry": "BUG", "chartiberry": "ROCK",
    "kasibberry": "GHOST", "habanberry": "DRAGON", "colburberry": "DARK", "babiriberry": "STEEL",
    "chilanberry": "NORMAL", "roseliberry": "FAIRY",
}
_BAND_ITEM_MULT = 4915.0 / 4096.0       # v11 B3: Expert Belt + type-boost items ≈×1.19995 (items.ts chainModify)


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
    "hustle", "guts", "tintedlens", "parentalbond", "supremeoverlord", "sandforce", "solarpower",
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

# v11 Phase A.1 — DEFENDER-ability TYPE IMMUNITIES that ZERO the damage (a 0× immunity, distinct
# from a typing immunity). Data-grounded subset of _TYPE_IMMUNE_AB mapped to the absorbed move TYPE.
# (Flag-gated immunities — Bulletproof/Soundproof/Wind Rider/Overcoat — need a move FLAG not a type,
# deferred to A.1b; Purifying Salt is a Ghost ×0.5 RESIST not an immunity → handled in the damage
# multiplier, not here.) An attacker with Mold Breaker / Teravolt / Turboblaze IGNORES these.
_ABILITY_TYPE_IMMUNITY = {
    "levitate": "GROUND", "eartheater": "GROUND",
    "voltabsorb": "ELECTRIC", "lightningrod": "ELECTRIC", "motordrive": "ELECTRIC",
    "waterabsorb": "WATER", "stormdrain": "WATER", "dryskin": "WATER",
    "flashfire": "FIRE", "wellbakedbody": "FIRE",
    "sapsipper": "GRASS",
}

# v11 A.1b — DEFENDER-ability 0× immunities gated by a move FLAG (not a type): the move is nullified
# if it carries the flag. All breakable → an attacker with Mold Breaker bypasses them (handled below).
_ABILITY_FLAG_IMMUNITY = {
    "bulletproof": "is_bullet",     # ballistics (Shadow Ball, Sludge Bomb, Aura Sphere, Focus Blast, …)
    "soundproof": "is_sound",       # sound moves (Boomburst, Hyper Voice, Overdrive, …)
    "windrider": "is_wind",         # wind moves (Bleakwind/Sandsear/Springtide/Wildbolt Storm, Hurricane, Icy Wind, …)
}


def _ability_immunizes(move_type: Optional[str], defender_ability: Optional[str],
                       attacker_ability: Optional[str], move_id: Optional[str] = None) -> bool:
    """True iff the DEFENDER's ability grants a 0× immunity to this move AND the attacker does NOT ignore
    abilities (Mold Breaker family). Covers TYPE-gated immunities (Levitate→Ground, Volt Absorb→Electric,
    …) and, when ``move_id`` is given, FLAG-gated ones (Bulletproof/Soundproof/Wind Rider, v11 A.1b).
    SHARED by the offline + live move-writers so the immunity zeroing is byte-identical."""
    if not defender_ability:
        return False
    if (attacker_ability or "") in _MOLD_BREAKER_AB:
        return False
    if move_type and _ABILITY_TYPE_IMMUNITY.get(defender_ability) == move_type.upper():
        return True
    flag_key = _ABILITY_FLAG_IMMUNITY.get(defender_ability)
    return bool(flag_key and move_id is not None and _move_damage_props(move_id).get(flag_key))


# ══════════════════════════════════════════════════════════════════════════════
# v11 C.2c — GROUNDING (Magnet Rise/Telekinesis/Smack Down/Iron Ball/Air Balloon) + the Ground-move
# immunity it implies + Tar Shot. The static typing-only grounded read mis-handled these (Arena Trap #B.3,
# terrain ×1.3, and Ground immunity). Mirrors pokemon-showdown sim/pokemon.ts isGrounded() precedence,
# EXCEPT Gravity (a pseudo-weather the offline parser does not track — excluding it keeps offline↔live
# parity; documented Tier-3 follow-up).
def _is_grounded(types, ability, item, levitating: bool, force_grounded: bool) -> bool:
    """True iff the mon is grounded (affected by Ground moves / terrain / Spikes / Arena Trap). Force-
    grounded (Ingrain/Smack Down/Iron Ball) overrides the ungrounders (Flying/Levitate/Magnet Rise/
    Telekinesis/Air Balloon). SHARED by the offline + live paths (only the property extraction differs)."""
    if force_grounded or item == "ironball":
        return True
    if "FLYING" in types or ability == "levitate" or item == "airballoon" or levitating:
        return False
    return True


def _ground_immune(item, levitating: bool, force_grounded: bool) -> bool:
    """The Ground-move immunity that is NOT a typing (Flying) or an ability (Levitate, handled elsewhere)
    and is NOT bypassed by Mold Breaker: Air Balloon / Magnet Rise / Telekinesis, unless force-grounded
    (Ingrain/Smack Down/Iron Ball override the ungrounders, matching _is_grounded)."""
    return (item == "airballoon" or levitating) and not (force_grounded or item == "ironball")


def _move_immune(move_type, defender, attacker_ability, move_id) -> bool:
    """Combined per-move 0× immunity for the type-eff channel + damage band: a defender ABILITY immunity
    (Levitate/Volt Absorb/Bulletproof/…, Mold-Breaker-aware) OR a GROUND move vs an Air-Balloon/Magnet-
    Rise/Telekinesis defender (item/volatile — NOT Mold-Breaker-bypassable; Flying typing is the chart).
    v11 C.2d: a GROUND move negates the AIRBORNE-based immunities (Levitate ability + Air-Balloon/Magnet-
    Rise/Telekinesis) when the defender is force-grounded (Smack Down/Ingrain/Iron Ball) OR the move
    pierces Ground immunity (Thousand Arrows). isGrounded (sim/pokemon.ts:2154-2158) returns grounded
    BEFORE the Flying(2160)/Levitate(2161) checks, so force-ground beats both. Earth Eater (absorption,
    not airborne) is NOT negated; the Flying TYPING 0× is negated in the chart helpers, not here."""
    if not defender:
        return False
    _ab = defender.get("ability")
    if move_type == "GROUND" and (defender.get("force_grounded") or _move_ground_pierces(move_id)):
        if _ab == "levitate":
            return False                                          # airborne ability negated
        if _ability_immunizes(move_type, _ab, attacker_ability, move_id):
            return True                                           # Earth Eater etc. still absorb
        return False                                              # Air Balloon / Magnet Rise negated
    if _ability_immunizes(move_type, _ab, attacker_ability, move_id):
        return True
    return move_type == "GROUND" and bool(defender.get("ground_immune"))


# ══════════════════════════════════════════════════════════════════════════════
# v11 Phase A.2 — ability DAMAGE MULTIPLIERS (offensive boosts + defensive resists)
# ──────────────────────────────────────────────────────────────────────────────
# Exact chainModify rationals from pokemon-showdown/data/abilities.ts (decimals shown). One shared
# helper folds them into the B1.2 damage band's situational_mult so the offline + live paths are
# byte-identical (parity twin of _ability_immunizes). Adversarially verified: scratch/
# v11_a2_ability_damage_mult_synthesis.md.
_M_1_2 = 4915 / 4096   # 1.19995…  ironfist, reckless, the -ate set
_M_1_3 = 5325 / 4096   # 1.30005…  toughclaws, punkrock(off), transistor, sheerforce
# v11 A3 Supreme Overlord (Kingambit, IN META): EXACT chainModify powMod table /4096, indexed by
# min(fallen, 5). abilities.ts:4734 powMod=[4096,4506,4915,5325,5734,6144] → 1.0/1.10009/1.19995/1.30005/
# 1.40015/1.5 (indices 2,3 are exactly _M_1_2/_M_1_3). ⚠ The sim LATCHES `fallen` at switch-in (onStart →
# effectState.fallen); we use the CURRENT side-faint count (fainted_allies) because poke-env exposes NO
# latched value → the live twin physically cannot read it (parity-forced). This over-reads only in the
# "an ally faints while Kingambit is already active" sub-case (cf. Last Respects, which re-reads
# side.totalFainted per USE so its current-count is EXACT). Documented approximation; see _ability_damage_mult.
_SUPREME_OVERLORD_MULT = tuple(v / 4096 for v in (4096, 4506, 4915, 5325, 5734, 6144))

# The 4 -ate abilities that ALSO grant a ×1.2 BP boost to the Normal move they convert (in _DMG_BOOST_AB).
_ATE_BOOST_AB = frozenset({"pixilate", "refrigerate", "aerilate", "galvanize"})
# Moves whose type the -ate abilities do NOT convert (abilities.ts noModifyType) → no boost either.
_ATE_NOMODIFY = frozenset({"judgment", "multiattack", "naturalgift", "revelationdance",
                           "technoblast", "terrainpulse", "weatherball"})
# Defensive abilities Mold Breaker does NOT suppress (flags:{} = un-breakable, abilities.ts) — all the
# other _DEF_REDUCE_AB members are flags:{breakable:1} and ARE ignored by Mold Breaker/Teravolt/Turboblaze.
_DEF_MOLDBREAKER_IMMUNE = frozenset({"shadowshield", "prismarmor"})

# v11 A.1b — DEFENDER-ability PARTIAL (signed) damage modifiers, type-gated. Not in _DEF_REDUCE_AB (they
# carry other tags) so they need their own set. All breakable → Mold Breaker bypasses them.
_DEF_SIGNED_RESIST_AB = frozenset({"purifyingsalt", "dryskin", "waterbubble"})
# v11 A.1b — ATTACKER-ability offensive type boosts NOT already in _DMG_BOOST_AB (Water Bubble Water ×2).
_OFF_EXTRA_AB = frozenset({"waterbubble"})


# v11 B1: Mimikyu Disguise — intact-disguise first-hit damage block.
def _disguise_intact(ability: Optional[str], hp_frac: float) -> bool:
    """True iff a Mimikyu's Disguise is INTACT (blocks the next damaging hit → band 0). Keyed on ability +
    FULL HP (hp_frac >= 1.0), NOT the busted species: poke-env keeps mon.species at base post-bust AND
    Mimikyu/Mimikyu-Busted share identical stats/types, so the species flip is OFFLINE-ONLY and would break
    parity. The Disguise bust costs 1/8 HP, so an intact (never-move-hit) Disguise ⟺ hp_frac >= 1.0 on BOTH
    paths. (Approximation: a hazard-chipped-but-intact Mimikyu reads <1.0 → not zeroed (conservative); a
    busted-then-fully-healed Mimikyu reads >=1.0 → over-zeroed — both rare and PARITY-CLEAN. Ice Face is NOT
    modelled: its bust is HP-free so the proxy fails, and Eiscue is absent from the Champions meta.) SHARED
    by both encoders → byte-identical."""
    return ability == "disguise" and (hp_frac or 0.0) >= 1.0


def _ability_damage_mult(move_id: Optional[str], attacker_ability: Optional[str],
                         defender_ability: Optional[str], *, eff_move_type: Optional[str],
                         is_stab: bool, is_physical: bool, type_mult: float, hp_frac: float,
                         att_burned: bool, att_statused: bool, is_spread: bool = False,
                         fainted_allies: int = 0, defender_intact_disguise: bool = False,
                         weather: Optional[str] = None) -> float:
    """Combined attacker+defender ABILITY damage multiplier, multiplied INTO the B1.2 band's
    situational_mult. SHARED by the offline + live move-writers; move props come from GenData(9)
    (_move_damage_props) so both paths are byte-identical. Returns 0.0 for a Wonder-Guard non-super
    hit OR an intact-Disguise block (v11 B1) — both zero the band. Mold Breaker (att) ignores the
    breakable defensive abilities (incl. Disguise) only."""
    a = attacker_ability or ""
    da = defender_ability or ""
    # v11 B1: an INTACT Disguise (Mimikyu) blocks ANY damaging hit → band 0. Disguise is breakable, so Mold
    # Breaker/Teravolt/Turboblaze bypass it. Checked BEFORE the common-case early-return (da="disguise" is in
    # none of the ability sets below, so the early-return would otherwise skip this). A multi-hit move is
    # fully zeroed (deliberate over-block of the rare bust-turn; the dominant signal is "intact Mimikyu survives").
    if defender_intact_disguise and a not in _MOLD_BREAKER_AB:
        return 0.0
    if (a not in _DMG_BOOST_AB and a not in _OFF_EXTRA_AB
            and da not in _DEF_REDUCE_AB and da not in _DEF_SIGNED_RESIST_AB):
        return 1.0                                   # common case: no relevant ability, skip the lookup
    mult = 1.0
    mt = (eff_move_type or "").upper()
    props = _move_damage_props(move_id)

    # ── OFFENSIVE: the attacker's own ability (one ability per mon → mutually exclusive) ──
    if a in _DMG_BOOST_AB:
        if a == "adaptability":
            if is_stab:
                mult *= 2.0 / 1.5                     # band folds 1.5 STAB (see _damage_band) → net 2.0
        elif a == "technician":
            if 0 < props["base_power"] <= 60:
                mult *= 1.5
        elif a == "toughclaws":
            if props["is_contact"]:
                mult *= _M_1_3
        elif a == "tintedlens":
            if 0.0 < type_mult < 1.0:                 # resisted (not a 0× immunity, which is band-0 already)
                mult *= 2.0
        elif a in ("hugepower", "purepower"):
            if is_physical:
                mult *= 2.0
        elif a == "guts":
            if is_physical and att_statused:
                mult *= 1.5
                if att_burned:
                    mult *= 2.0                       # cancel the burn ×0.5 in _situational (Guts ignores burn)
        elif a == "hustle":
            if is_physical:
                mult *= 1.5
        elif a == "ironfist":
            if props["is_punch"]:
                mult *= _M_1_2
        elif a == "strongjaw":
            if props["is_bite"]:
                mult *= 1.5
        elif a == "reckless":
            if props["has_recoil_or_crash"]:
                mult *= _M_1_2
        elif a == "punkrock":
            if props["is_sound"]:
                mult *= _M_1_3
        elif a == "sharpness":
            if props["is_slicing"]:
                mult *= 1.5
        elif a == "rockypayload":
            if mt == "ROCK":
                mult *= 1.5
        elif a == "transistor":
            if mt == "ELECTRIC":
                mult *= _M_1_3
        elif a == "dragonsmaw":
            if mt == "DRAGON":
                mult *= 1.5
        elif a in ("steelworker", "steelyspirit"):   # steelyspirit self-applies (alliesAndSelf incl. self)
            if mt == "STEEL":
                mult *= 1.5
        elif a in _ATE_BOOST_AB:
            if props["raw_is_normal"] and (move_id or "") not in _ATE_NOMODIFY:
                mult *= _M_1_2
        elif a == "sheerforce":
            if props["has_secondaries"]:
                mult *= _M_1_3
        elif a == "parentalbond":                    # v11 A2: Mega Kangaskhan — 2nd strike ×0.25 (Gen 9)
            # Eligible = a single-target, single-hit damaging move (not spread/multihit/status/charge/Z).
            # Effective ≈ ×1.25 (1.0 + 0.25). Multihit moves are NOT eligible → no double-count with A1.
            if not is_spread and _move_hit_range(move_id) == (1, 1):
                mult *= 1.25
        elif a == "supremeoverlord":                 # v11 A3: Kingambit — ×powMod[min(fallen,5)] faint-count boost
            mult *= _SUPREME_OVERLORD_MULT[min(fainted_allies, 5)]
        elif a == "sandforce":                       # v11 gap-scan G5: ×1.3 Rock/Ground/Steel moves in Sandstorm
            if weather == "SANDSTORM" and mt in ("ROCK", "GROUND", "STEEL"):
                mult *= _M_1_3
        elif a == "solarpower":                      # v11 gap-scan G6: ×1.5 SPECIAL damage in Sun
            if weather in ("SUNNYDAY", "DESOLATELAND") and not is_physical:   # audit: harsh sun counts as Sun
                mult *= 1.5
    elif a == "waterbubble":                         # v11 A.1b: Water Bubble offensive Water ×2
        if mt == "WATER":
            mult *= 2.0

    # ── DEFENSIVE: the defender's ability (Mold Breaker att ignores breakable ones) ──
    if da in _DEF_REDUCE_AB:
        mold = a in _MOLD_BREAKER_AB
        if not (mold and da not in _DEF_MOLDBREAKER_IMMUNE):
            if da in ("multiscale", "shadowshield"):
                if hp_frac >= 1.0:
                    mult *= 0.5
            elif da == "thickfat":
                if mt in ("FIRE", "ICE"):
                    mult *= 0.5
            elif da in ("filter", "solidrock", "prismarmor"):
                if type_mult > 1.0:
                    mult *= 0.75
            elif da == "fluffy":
                if mt == "FIRE":
                    mult *= 2.0
                if props["is_contact"]:
                    mult *= 0.5
            elif da == "icescales":
                if not is_physical:
                    mult *= 0.5
            elif da == "furcoat":
                if is_physical:
                    mult *= 0.5                       # Def×2 ≈ ½ physical damage (band-level approximation)
            elif da == "heatproof":
                if mt == "FIRE":
                    mult *= 0.5
            elif da == "wonderguard":
                if type_mult <= 1.0:
                    return 0.0                        # only super-effective hits land

    # v11 A.1b — DEFENDER-ability partial (signed) type resists/weaknesses (all breakable → MB bypasses).
    if da in _DEF_SIGNED_RESIST_AB and a not in _MOLD_BREAKER_AB:
        if da == "purifyingsalt":
            if mt == "GHOST":
                mult *= 0.5
        elif da == "dryskin":
            if mt == "FIRE":
                mult *= 1.25                          # Dry Skin Water immunity is the A.1 type-immunity path
        elif da == "waterbubble":
            if mt == "FIRE":
                mult *= 0.5

    return mult


# v11 B.3: opposing-active abilities that TRAP a foe (can_switch correctness bug #5). Shadow Tag is in
# the Champions meta (Mega Gengar). Magnet Pull (Steel) / Arena Trap (grounded) are the conditional ones.
# (the trap-ability ids are inlined in _ability_trapped below — no shared constant.)


def _ability_trapped(is_ghost: bool, is_steel: bool, is_grounded: bool, has_shadowtag: bool,
                     has_shed_shell: bool, enemy_abilities) -> bool:
    """v11 B.3: True iff this mon is TRAPPED by an OPPOSING active's ability (so it cannot voluntarily
    switch). Ghost types + Shed Shell holders are immune to ALL trapping (Showdown 'trapped' status
    immunity); a Shadow Tag mon is immune to Shadow Tag (mutual). Shadow Tag traps any other foe; Arena
    Trap only grounded foes; Magnet Pull only Steel foes. SHARED by the offline + live volatile blocks so
    the can_switch zeroing is byte-identical (only the per-mon property EXTRACTION differs)."""
    if is_ghost or has_shed_shell:
        return False
    for ab in enemy_abilities:
        if ab == "shadowtag" and not has_shadowtag:
            return True
        if ab == "arenatrap" and is_grounded:
            return True
        if ab == "magnetpull" and is_steel:
            return True
    return False


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
    mon with no item is a confirmed-itemless 1.0, as is an OTS ``sheet_itemless``
    mon (the server ``|showteam|`` sheet showed no item — without the sentinel the
    belief fallback hands it a phantom top item); else the belief top item
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
    if mon.get("exact") or mon.get("sheet_itemless"):
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
# v18 (Option 1): redundant condition-setting (per-move feature).
# SHARED so the offline + live encoders compute it byte-identically (parity).
# Goal: teach the net that re-casting an ALREADY-ACTIVE non-stackable condition
# (Light Screen, Rain Dance, …) usually wastes a turn — WITHOUT forbidding it, so
# the legit double-screen (predicted Brick Break / about-to-expire) stays learnable.
# ══════════════════════════════════════════════════════════════════════════════

# Moves that set a NON-STACKABLE condition on the USER'S OWN side (re-casting while
# active = redundant). Trick Room is EXCLUDED (re-casting TOGGLES it off); entry
# hazards are EXCLUDED (opp-side + stackable — a different, noisier judgement).
_REDUNDANT_OWN_SIDE_MOVE: dict = {
    "lightscreen": "LIGHT_SCREEN", "reflect": "REFLECT", "auroraveil": "AURORA_VEIL",
    "tailwind": "TAILWIND", "safeguard": "SAFEGUARD", "mist": "MIST", "luckychant": "LUCKY_CHANT",
    # rare same-effect screen moves (LGPE-era partner moves; in the dex, defensively mapped)
    "glitzyglow": "LIGHT_SCREEN", "baddybad": "REFLECT",
}
# the canonical SIDE_COND names the redundancy bit cares about (both encoders filter their active-side
# set to these, so the frozensets match for the same battle → parity).
_REDUNDANT_OWN_SIDE_NAMES = frozenset(_REDUNDANT_OWN_SIDE_MOVE.values())
# Weather/terrain SETTING moves → a normalised condition key (matched against the
# currently-active field, normalised the same way — alias-proof across both paths).
_REDUNDANT_WEATHER_MOVE: dict = {
    "raindance": "rain", "sunnyday": "sun", "sandstorm": "sand", "snowscape": "snow",
    "hail": "snow",   # gen9 Hail sets snow (Chilly Reception EXCLUDED — it also switches, so a re-cast has value)
}
_REDUNDANT_TERRAIN_MOVE: dict = {
    "electricterrain": "electric", "grassyterrain": "grassy",
    "mistyterrain": "misty", "psychicterrain": "psychic",
}


def _weather_key(tok) -> Optional[str]:
    """Normalise any weather token (offline 'RainDance'/'SNOWSCAPE', live Weather.name) to a key."""
    if not tok:
        return None
    t = "".join(c for c in str(tok).lower() if c.isalnum())
    if t in ("raindance", "rain", "primordialsea"):
        return "rain"
    if t in ("sunnyday", "sun", "desolateland", "harshsunlight"):
        return "sun"
    if t in ("sandstorm", "sand"):
        return "sand"
    if t in ("snow", "snowscape", "hail"):
        return "snow"
    return None


def _terrain_key(tok) -> Optional[str]:
    """Normalise any terrain token ('electric'/'ELECTRIC_TERRAIN'/'electricterrain') to a key."""
    if not tok:
        return None
    t = "".join(c for c in str(tok).lower() if c.isalnum())
    for k in ("electric", "grassy", "misty", "psychic"):
        if k in t:
            return k
    return None


# v19b: FIELD-DEPENDENT move type. Weather Ball's type follows the WEATHER (rain→Water, sun→Fire, snow→Ice,
# sand→Rock, none→Normal); Terrain Pulse's follows the TERRAIN (electric→Electric, grassy→Grass, psychic→
# Psychic, misty→Fairy, none→Normal). Without this both read as base Normal → the type-eff / damage-band
# channels mis-rate them (e.g. a rain-boosted Water Weather Ball reads neutral instead of super-effective).
# ⚠ Terrain Pulse is GROUNDED-only in the sim; we approximate "always grounded" (airborne Terrain-Pulse
# users are vanishingly rare in the meta) — BOTH encoders make the SAME approximation → parity-clean. Techno
# Blast (Drive-ITEM dependent, rare Genesect) is intentionally NOT handled here. Value-only (no new slot).
_WEATHER_BALL_TYPE = {"rain": "Water", "sun": "Fire", "snow": "Ice", "sand": "Rock"}
_TERRAIN_PULSE_TYPE = {"electric": "Electric", "grassy": "Grass", "psychic": "Psychic", "misty": "Fairy"}


def field_dependent_move_type(move_id: Optional[str], base_type, weather, terrain):
    """Effective type of a FIELD-dependent move (Weather Ball / Terrain Pulse) under the current
    weather/terrain; ``base_type`` for every other move (and for these two when no field is up). SHARED by
    the offline + live writers (same _weather_key/_terrain_key normalisation) → parity by construction."""
    if move_id == "weatherball":
        return _WEATHER_BALL_TYPE.get(_weather_key(weather), base_type)
    if move_id == "terrainpulse":
        return _TERRAIN_PULSE_TYPE.get(_terrain_key(terrain), base_type)
    return base_type


def move_redundant_condition(move_id: Optional[str], side_active, weather, terrain) -> float:
    """1.0 if casting ``move_id`` would re-set a non-stackable condition ALREADY active on the relevant
    side, else 0.0. ``side_active`` = the casting mon's OWN-side active SIDE_COND names (canonical, e.g.
    {'LIGHT_SCREEN','TAILWIND'}); ``weather``/``terrain`` = the current global field tokens. A pure
    feature — the policy still decides (no action is blocked)."""
    if not move_id:
        return 0.0
    side_active = side_active or frozenset()      # defensive: tolerate a None / empty container
    name = _REDUNDANT_OWN_SIDE_MOVE.get(move_id)
    if name is not None:
        return 1.0 if name in side_active else 0.0
    wk = _REDUNDANT_WEATHER_MOVE.get(move_id)
    if wk is not None:
        return 1.0 if wk == _weather_key(weather) else 0.0
    tk = _REDUNDANT_TERRAIN_MOVE.get(move_id)
    if tk is not None:
        return 1.0 if tk == _terrain_key(terrain) else 0.0
    return 0.0


# ── v19 (Option 1c): per-TARGET status-move redundancy ─────────────────────────────────────────────────
# A PURE status move (category 'Status' whose whole point is to inflict a major status — Will-O-Wisp,
# Thunder Wave, Toxic, Spore, Glare, Sleep Powder, …) is WASTED on a target that already carries a major
# status (a mon holds only ONE) or is IMMUNE to that status by typing / ability. DAMAGING moves that carry
# a status only as a SECONDARY (Zap Cannon, Scald, Nuzzle, Discharge, Sludge Bomb) are category Physical/
# Special → they are NEVER flagged: the damage justifies them regardless of the target's status. A PURE
# feature (no action is masked) so statusing a predicted switch-in stays fully learnable.
_MAJOR_STATUS = frozenset({"brn", "par", "psn", "tox", "slp", "frz"})
# Defender TYPE → the applied status it is immune to (deterministic from typing; Corrosion edge-case skipped).
_STATUS_TYPE_IMMUNE = {
    "brn": frozenset({"FIRE"}),
    "psn": frozenset({"POISON", "STEEL"}),
    "tox": frozenset({"POISON", "STEEL"}),
    "frz": frozenset({"ICE"}),
    # 'par' is handled specially (Electric immune to ALL paralysis gen6+; Ground immune to the Thunder-Wave MOVE).
}
# Abilities granting BLANKET status immunity (any major status).
_STATUS_ABILITY_IMMUNE_ALL = frozenset({"comatose", "purifyingsalt"})
# Ability → the single status it blocks.
_STATUS_ABILITY_IMMUNE = {
    "limber": "par", "insomnia": "slp", "vitalspirit": "slp", "sweetveil": "slp",
    "immunity": "psn", "pastelveil": "psn", "waterveil": "brn", "waterbubble": "brn",
    "thermalexchange": "brn", "magmaarmor": "frz",
}


def _pure_status_move(move_id: Optional[str]):
    """(status_token, is_powder) for a PURE status move, else (None, False).

    Data-driven: category == 'Status' AND a TOP-LEVEL ``status`` field. Zap Cannon/Scald carry their status
    only in ``secondary`` (category Physical/Special) → excluded by construction (no hand list to rot)."""
    data = _gen9_moves().get(move_id or "")
    if not data or (data.get("category") or "").lower() != "status":
        return None, False
    st = (data.get("status") or "").lower()
    if st not in _MAJOR_STATUS:
        return None, False
    return st, bool((data.get("flags") or {}).get("powder"))


def _status_immune(status: str, powder: bool, move_type: Optional[str], defender: Optional[dict],
                   terrain: Optional[str] = None) -> bool:
    """True if ``defender`` cannot receive ``status`` from this move — already statused, type immunity,
    powder vs Grass/Overcoat, a status-immunity ability, or TERRAIN immunity (Misty blocks all major
    status, Electric blocks sleep — both only on a GROUNDED target)."""
    if defender is None:
        return False
    if (defender.get("status") or "").lower() in _MAJOR_STATUS:   # holds a major status already → a new one fails
        return True
    if defender.get("grounded") and terrain:                     # terrain status immunity (grounded targets only)
        if terrain == "MISTY_TERRAIN":                           # Misty Terrain: blocks ALL major status
            return True
        if terrain == "ELECTRIC_TERRAIN" and status == "slp":    # Electric Terrain: blocks sleep
            return True
    types = {str(t).upper() for t in (defender.get("types") or [])}
    ab = (defender.get("ability") or "")
    if ab in _STATUS_ABILITY_IMMUNE_ALL or _STATUS_ABILITY_IMMUNE.get(ab) == status:
        return True
    if powder and ("GRASS" in types or ab == "overcoat"):        # Spore/Sleep Powder/Stun Spore do nothing
        return True
    if status == "par":
        if "ELECTRIC" in types:                                  # gen6+: Electric-types can't be paralysed
            return True
        if (move_type or "").upper() == "ELECTRIC" and "GROUND" in types:   # Thunder Wave (Electric) vs Ground
            return True
        return False
    return status in _STATUS_TYPE_IMMUNE and bool(types & _STATUS_TYPE_IMMUNE[status])


def move_redundant_status(move_id: Optional[str], defender: Optional[dict],
                          terrain: Optional[str] = None) -> float:
    """1.0 if casting PURE status move ``move_id`` at ``defender`` is wasted (target already statused or
    immune to that status — incl. Misty/Electric Terrain on a grounded target), else 0.0. ``defender`` is a
    per-move enemy DEFENDER profile (must carry ``types``/``status``/``ability``/``grounded``). ``terrain``
    is the global field token (e.g. ``MISTY_TERRAIN``). Damaging moves and empty slots → 0.0. Pure feature."""
    st, powder = _pure_status_move(move_id)
    if st is None or defender is None:
        return 0.0
    move_type = (_gen9_moves().get(move_id, {}).get("type") or "")
    return 1.0 if _status_immune(st, powder, move_type, defender, terrain) else 0.0
