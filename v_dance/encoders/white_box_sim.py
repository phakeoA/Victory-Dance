"""
white_box_sim.py — Level C / B0: white-box turn-stepper, state-MUTATION core (B0a).
================================================================================

The forward model for Level-C search.  ``battle_mechanics.py`` already has every READ-ONLY physics
primitive (damage band, type/situational/ability mults, expected-crit, hit-chance, effective-speed,
moves-first, …) but NOTHING that mutates state.  This module adds exactly that mutation layer, operating
on the ENCODER SNAPSHOT DICT (``state_before_actions`` shape) so successors stay pure dicts that feed the
offline ``VodStateEncoder`` — NO poke-env in the tree (matching the B2 serve plan).  Design:
``docs/levelC_B0_white_box_design.md``.

**B0a (this file) = the low-level state ops + residuals**: clone, apply-damage→HP→faint, heal, status,
boost, switch-in (+ entry hazards), and end-of-turn residuals (weather/status chip, Leftovers, duration
decrement).  The 4-action SEQUENCER (B0b) and per-move damage RESOLUTION (B0c, which calls the
battle_mechanics primitives) build ON these.  HP is tracked in ``hp_pct`` (percent of max), so every op
here works in percent — no max-HP lookups needed.

**Fidelity tiers (never silent-defer):** Tier-1 exact / Tier-2 expected / **Tier-3 = note() in an
``UnmodelledLog`` and skip** — the ledger is surfaced by the B0d corpus-validation gate so coverage gaps
are visible, not hidden.
"""
from __future__ import annotations

import copy
from collections import Counter
from typing import Optional

from v_dance.encoders.battle_mechanics import (
    _effective_types, _type_mult, _gen9_moves, champ_bp, champ_type, field_dependent_move_type,
    _effective_speed, _defender_profile, _situational_damage_mult, _ability_damage_mult,
    _expected_crit_mult, _damage_band, _move_hit_range, _move_immune, _move_always_hit, _move_is_ohko,
    is_spread_target, resolve_active_ability_json, resolve_item_json, _BAND_ITEM_MULT, _TYPE_BOOST_TYPE,
    _canon, _GRASSY_WEAKENED,
)
from v_dance.encoders.damage_mechanics import variable_base_power, species_weight, effective_move_type
from v_dance.encoders.encoder_layout import _TERRAIN_TO_FIELD
from v_dance.parser.belief_state import dex_base_stats
from v_dance.parser.vod_parser.pokedex import get_pokedex, norm_species

_STAT_KEYS = ("hp", "atk", "def", "spa", "spd", "spe")
_ZERO_DMG = (0.0, 0.0, 0.0, 0.0)        # (mean, lo_nc, hi_nc, hi_crit) — a no-damage _move_damage_pct result

# (Choice Band/Specs give the ×1.5 A-stat damage boost; Choice SCARF does NOT — it boosts Speed. The damage
#  boost is gated inline on ("choiceband","choicespecs"), matching the encoder's choice AND-NOT-choice_speed.)
# Single-target Protect-class moves: the user takes NO damage this turn (Tier-1). Consecutive-use failure
# (protect_counter) is Tier-2 (assumed to succeed); Feint/Unseen-Fist break-through is Tier-3.
_PROTECT_MOVES = {"protect", "detect", "spikyshield", "kingsshield", "banefulbunker", "obstruct",
                  "silktrap", "burningbulwark", "maxguard"}
# Flat-50% self-recovery moves (Tier-1). Weather-scaled (Synthesis/Moonlight/Morning Sun/Shore Up) +
# delayed (Wish) + team heals (Life Dew) stay Tier-3 (logged).
_HEAL_MOVES = {"roost", "recover", "slackoff", "softboiled", "milkdrink", "healorder", "lunarblessing"}
# Status moves whose EFFECT is modelled elsewhere → no status_move gap note. Helping Hand boosts the ally's
# damage in _ally_damage_mult (applied when the ALLY attacks, not when Helping Hand itself resolves).
_HANDLED_STATUS = {"helpinghand"}
# Option B — survival mechanics + generic residual (the snapshot represents these only coarsely; see
# docs/levelC_B0d_option_b_survival_residual_design.md).
_SURVIVE_FLOOR = 0.5                  # ~1 HP: a Focus Sash / Sturdy / Endure survivor floors here, not faint
_RESIDUAL_CHIP_PCT = 100.0 / 8.0     # generic Leech Seed / Salt Cure / Ghost-Curse chip (bool carries no amount)
# Variable base-power moves whose REAL bp is recomputed from context via damage_mechanics.variable_base_power
# (Low Kick/Grass Knot weight · Heavy Slam/Heat Crash ratio · Gyro/Electro Ball speed · Eruption/Water Spout HP ·
#  Stored Power/Power Trip/Punishment boosts · Last Respects fainted-allies · Rage Fist times-hit).
_VBP_HANDLED = {"lowkick", "grassknot", "heavyslam", "heatcrash", "gyroball", "electroball",
                "eruption", "waterspout", "dragonenergy", "storedpower", "powertrip", "punishment",
                "lastrespects", "ragefist"}
# Variable-BP moves NOT in variable_base_power → logged residual, fall back to static BP: Bolt Beak/Fishious Rend
# (2× when moving first), Hard Press (target-HP scaled).
_VBP_RESIDUAL = {"boltbeak", "fishiousrend", "hardpress"}
_DEF_HIT_SPECIAL = {"psyshock", "psystrike", "secretsword"}   # special moves that hit Def (Tier-3 noted)

# ── status / weather residual constants ───────────────────────────────────────
_BURN_PCT = 100.0 / 16.0          # 1/16 max
_POISON_PCT = 100.0 / 8.0         # 1/8 max
_SAND_CHIP_PCT = 100.0 / 16.0
_LEFTOVERS_PCT = 100.0 / 16.0
_STEALTH_ROCK_PCT = 100.0 / 8.0   # 12.5% × Rock type-effectiveness
# Major-status chip (percent of max). Toxic is Tier-2 here (flat 1/8, no escalation — logged).
_STATUS_CHIP_PCT = {"brn": _BURN_PCT, "psn": _POISON_PCT, "tox": _POISON_PCT}
_SAND_IMMUNE_TYPES = {"ROCK", "GROUND", "STEEL"}


class UnmodelledLog(Counter):
    """Tier-3 ledger — every mechanic the stepper does NOT model is ``note()``-d here (never
    silent-deferred).  Surfaced by the B0d validation gate so the coverage gap is explicit + ranked."""

    def note(self, key, n: int = 1) -> None:
        self[str(key)] += int(n)


# ── state accessors ───────────────────────────────────────────────────────────
def clone_state(snapshot: dict) -> dict:
    """A deep copy of the snapshot — the stepper mutates the copy, the original (a search node) is kept."""
    return copy.deepcopy(snapshot)


def is_fainted(mon: Optional[dict]) -> bool:
    # hp_pct=None is the UNREVEALED-bench convention (the leaf encoder renders it at FULL HP), NOT a faint —
    # only a set is_fainted flag or a concrete hp_pct<=0 is a real KO. Active mons always carry a concrete
    # hp_pct, so this only changes the bench case: _count_fainted_side / Last Respects BP / Supreme Overlord
    # no longer over-count unrevealed allies as fallen (audit 2026-06-30).
    if not mon:
        return False
    if mon.get("is_fainted"):
        return True
    hp = mon.get("hp_pct")
    return hp is not None and float(hp) <= 0.0


def mon_max_hp(mon: Optional[dict]) -> Optional[float]:
    """The mon's est max-HP stat (for callers converting a band's fraction-of-current → percent)."""
    return ((mon or {}).get("stats_estimate") or {}).get("stats", {}).get("hp")


# ── HP mutation ───────────────────────────────────────────────────────────────
# Per-mon DISTRIBUTION shadow HPs (variance-aware faint, B0d reframe): _hp_hi = HP after MIN damage (luckiest),
# _hp_lo = HP after MAX non-crit damage, _hp_floor = HP after MAX damage WITH crit (reachability). Lazily
# initialised to hp_pct on first damage; the probe defaults to hp_pct when absent. NOT clamped (P(KO) magnitude).
_SHADOWS = ("_hp_hi", "_hp_lo", "_hp_floor")


def _shadow_shift(mon: dict, delta: float) -> None:
    """Translate any present shadows by ``delta`` (negative = damage) — for DETERMINISTIC HP changes (no roll)."""
    for k in _SHADOWS:
        if k in mon:
            mon[k] = mon[k] + delta


def apply_damage_pct(mon: Optional[dict], pct: float, *, log: Optional[UnmodelledLog] = None) -> float:
    """Subtract ``pct`` (percent of MAX hp) from ``mon``; faint at ≤ 0. Returns the percent actually dealt
    (capped at the mon's current HP). No-op on a fainted/empty mon or non-positive ``pct``. DETERMINISTIC →
    shifts the distribution shadows in lockstep (residuals / recoil / hazards have no roll)."""
    if not mon or is_fainted(mon) or pct is None or pct <= 0:
        return 0.0
    cur = float(mon.get("hp_pct") or 0.0)
    _shadow_shift(mon, -float(pct))
    new = cur - float(pct)
    if new <= 0.0:
        mon["hp_pct"] = 0.0
        mon["is_fainted"] = True
        return cur
    mon["hp_pct"] = new
    return float(pct)


def apply_damage_rolled(mon: Optional[dict], mean: float, lo_nc: float, hi_nc: float, hi_crit: float, *,
                        survive: bool = False) -> float:
    """Apply a ROLLED move hit: ``mean`` damage → ``hp_pct`` (faint logic, exactly as the EV path), and widen the
    distribution shadows (least damage ``lo_nc`` → ``_hp_hi``; most non-crit ``hi_nc`` → ``_hp_lo``; most-with-crit
    ``hi_crit`` → ``_hp_floor``). Shadows are lazily seeded to the pre-hit ``hp_pct`` and left UNCLAMPED. With
    ``survive`` (Focus Sash / Sturdy / Endure) the hit CANNOT faint: ``hp_pct`` and any ≤0 shadow floor at
    ``_SURVIVE_FLOOR``. Returns the mean percent actually dealt (for recoil)."""
    if not mon or is_fainted(mon):
        return 0.0
    cur = float(mon.get("hp_pct") or 0.0)
    for k in _SHADOWS:
        mon.setdefault(k, cur)
    mon["_hp_hi"] -= max(0.0, float(lo_nc or 0.0))
    mon["_hp_lo"] -= max(0.0, float(hi_nc or 0.0))
    mon["_hp_floor"] -= max(0.0, float(hi_crit or 0.0))
    if survive:                                       # Sash/Sturdy/Endure — no shadow may faint this hit
        for k in _SHADOWS:
            if mon[k] <= 0.0:
                mon[k] = _SURVIVE_FLOOR
    if mean is None or mean <= 0:
        return 0.0
    new = cur - float(mean)
    if new <= 0.0:
        if survive:
            mon["hp_pct"] = _SURVIVE_FLOOR
            return max(0.0, cur - _SURVIVE_FLOOR)
        mon["hp_pct"] = 0.0
        mon["is_fainted"] = True
        return cur
    mon["hp_pct"] = new
    return float(mean)


def apply_heal_pct(mon: Optional[dict], pct: float) -> float:
    """Add ``pct`` (percent of MAX hp), capped at 100. Returns the percent actually healed. No-op on a
    fainted mon (a fainted mon cannot be healed by residuals/items). DETERMINISTIC → shifts shadows up."""
    if not mon or is_fainted(mon) or pct is None or pct <= 0:
        return 0.0
    cur = float(mon.get("hp_pct") or 0.0)
    new = min(100.0, cur + float(pct))
    _shadow_shift(mon, new - cur)
    mon["hp_pct"] = new
    return new - cur


# ── status / boosts ───────────────────────────────────────────────────────────
def set_status(mon: Optional[dict], status: Optional[str]) -> bool:
    """Apply a major status if the mon is healthy + currently status-free (returns True).  Immunity
    (type / ability / Safeguard) is the CALLER's responsibility (``_status_immune``); a no-op on an
    already-statused or fainted mon, or a falsy status."""
    if not mon or not status or is_fainted(mon) or mon.get("status"):
        return False
    mon["status"] = status
    return True


def apply_boost(mon: Optional[dict], stat: str, stages: int) -> int:
    """Change a stat boost stage, clamped to [-6, +6]. Returns the net change applied."""
    if not mon or is_fainted(mon) or not stages:
        return 0
    boosts = mon.setdefault("boosts", {})
    cur = int(boosts.get(stat, 0) or 0)
    new = max(-6, min(6, cur + int(stages)))
    boosts[stat] = new
    return new - cur


# ── switch ────────────────────────────────────────────────────────────────────
def _reset_on_switch(mon: Optional[dict]) -> None:
    """Clear the boosts + volatiles a mon loses when it switches OUT (Showdown resets stat stages and
    most volatiles on switch). Status + HP persist."""
    if not mon:
        return
    mon["boosts"] = {}
    vol = mon.get("volatiles")
    if isinstance(vol, dict):
        for k, v in list(vol.items()):
            if isinstance(v, bool):
                vol[k] = False
            elif isinstance(v, (int, float)):
                vol[k] = 0
            else:
                vol[k] = None


def _apply_entry_hazards(snapshot: dict, mon: dict, side: str, log: Optional[UnmodelledLog]) -> None:
    """Entry hazards on a switched-in mon. Tier-2: Stealth Rock (type-scaled). Tier-3 (logged): Spikes /
    Toxic Spikes / Sticky Web (grounding + layer logic deferred until validation shows it matters)."""
    sc = ((snapshot.get("side_conditions") or {}).get(f"{side}_side")) or {}
    if sc.get("stealth_rock"):
        mult = _type_mult("ROCK", [t.upper() for t in (_effective_types(mon) or [])])
        apply_damage_pct(mon, _STEALTH_ROCK_PCT * mult, log=log)
    if log is not None:
        if sc.get("spikes"):
            log.note("spikes_entry")
        if sc.get("toxic_spikes"):
            log.note("toxic_spikes_entry")
        if sc.get("sticky_web"):
            log.note("sticky_web_entry")


_INTIMIDATE_IMMUNE = frozenset({          # abilities that BLOCK or INVERT Intimidate's -1 Atk → Tier-3 (not modelled)
    "clearbody", "whitesmoke", "fullmetalbody", "hypercutter", "innerfocus", "oblivious",
    "owntempo", "scrappy", "guarddog", "defiant", "competitive", "mirrorarmor",
})
_ON_ENTRY_ABILITIES = frozenset({         # impactful on-entry abilities the stepper does NOT model → LOGGED (Tier-3)
    "drought", "drizzle", "sandstream", "snowwarning", "primordialsea", "desolateland", "deltastream",
    "electricsurge", "grassysurge", "mistysurge", "psychicsurge", "download", "intrepidsword",
    "dauntlessshield", "protosynthesis", "quarkdrive", "screencleaner",
})


def _apply_entry_ability(snapshot: dict, incoming: dict, side: str, log: Optional[UnmodelledLog]) -> None:
    """On-entry ability effects for a switched-in mon. Tier-2: Intimidate lowers each LIVE opposing active's
    Atk by one stage (skipping abilities that block/invert it — Tier-3). Every other impactful on-entry ability
    (weather/terrain setters, Download, Intrepid Sword/Dauntless Shield, Protosynthesis/Quark Drive booster) is
    NOT modelled but is LOGGED — never silent-defer (module contract, audit 2026-07-01)."""
    aid = resolve_active_ability_json(incoming)[0]
    if not aid:
        return
    if aid == "intimidate":
        opp = "opp" if side == "our" else "our"
        for m in ((snapshot.get(f"{opp}_active") or {}).values()):
            if not m or is_fainted(m):
                continue
            if resolve_active_ability_json(m)[0] in _INTIMIDATE_IMMUNE:
                if log is not None:
                    log.note("intimidate_blocked")
                continue
            apply_boost(m, "atk", -1)
        if log is not None:
            log.note("intimidate_entry")
    elif aid in _ON_ENTRY_ABILITIES and log is not None:
        log.note("on_entry_ability:" + aid)


def switch_in(snapshot: dict, slot_key: str, bench_index: int, *,
              log: Optional[UnmodelledLog] = None) -> Optional[dict]:
    """Swap the active mon at ``slot_key`` (``our_a``/``our_b``/``opp_a``/``opp_b``) with the
    ``bench_index``-th mon of that side's bench. Resets the OUTGOING mon's boosts/volatiles and benches
    it; applies entry hazards to the INCOMING. Returns the incoming mon, or None on a bad index."""
    side = "our" if str(slot_key).startswith("our") else "opp"
    active = snapshot.get(f"{side}_active")
    bench = snapshot.get(f"{side}_bench")
    if not isinstance(active, dict) or not isinstance(bench, list) or not (0 <= bench_index < len(bench)):
        if log is not None:
            log.note("switch_bad_index")
        return None
    incoming = bench.pop(bench_index)
    # An UNSEEN bench mon (never revealed) carries hp_pct=None; the leaf encoder renders it at FULL HP, so the
    # stepper must too (else is_fainted(None)→fainted = a forward-model vs leaf contradiction). Seed it full.
    if incoming is not None and incoming.get("hp_pct") is None:
        incoming["hp_pct"] = 100.0
        incoming["is_fainted"] = False
    outgoing = active.get(slot_key)
    if outgoing is not None:
        _reset_on_switch(outgoing)
        bench.append(outgoing)
    active[slot_key] = incoming
    _apply_entry_hazards(snapshot, incoming, side, log)
    _apply_entry_ability(snapshot, incoming, side, log)
    return incoming


# ── end-of-turn residuals ─────────────────────────────────────────────────────
def apply_residuals(snapshot: dict, *, log: Optional[UnmodelledLog] = None) -> None:
    """End-of-turn residuals (Tier-1 common): sandstorm chip (type-aware), major-status chip
    (burn/poison; toxic flat 1/8 — Tier-2, logged), Leftovers heal; then decrement field/side
    durations. Weather/terrain duration-expiry + screens are Tier-3 (logged) for now."""
    field = snapshot.get("field") or {}
    weather = str(field.get("weather") or "").lower()
    is_sand = "sand" in weather
    if weather and not is_sand and log is not None:
        log.note(f"weather_residual:{weather}")

    for side in ("our", "opp"):
        for mon in (snapshot.get(f"{side}_active") or {}).values():
            if not mon or is_fainted(mon):
                continue
            if is_sand:
                types = {t.upper() for t in (_effective_types(mon) or [])}
                if not (types & _SAND_IMMUNE_TYPES):
                    apply_damage_pct(mon, _SAND_CHIP_PCT, log=log)
            if is_fainted(mon):
                continue
            st = str(mon.get("status") or "").lower()
            chip = _STATUS_CHIP_PCT.get(st)
            if chip:
                if st == "tox" and log is not None:
                    log.note("toxic_no_escalation")
                apply_damage_pct(mon, chip, log=log)
            if is_fainted(mon):
                continue
            if str(mon.get("known_item") or "").lower() == "leftovers" and not mon.get("item_consumed"):
                apply_heal_pct(mon, _LEFTOVERS_PCT)   # consumed/Knocked-Off Leftovers no longer heals
            if is_fainted(mon):
                continue
            vol = mon.get("volatiles") or {}
            # generic residual chip (Leech Seed / Salt Cure / Ghost-Curse / Nightmare — the snapshot bool
            # carries no magnitude, so a flat 1/8 is the modal estimate):
            if vol.get("residual_damage"):
                apply_damage_pct(mon, _RESIDUAL_CHIP_PCT, log=log)
                _note(log, "applied:residual_chip")
            if is_fainted(mon):
                continue
            # Perish Song: perish_norm == 1.0 (perish0) faints at end of THIS turn.
            if float(vol.get("perish_norm") or 0.0) >= 1.0:
                apply_damage_pct(mon, 100.0, log=log)
                _note(log, "applied:perish")

    _decrement_durations(snapshot, log=log)


def _decrement_durations(snapshot: dict, *, log: Optional[UnmodelledLog] = None) -> None:
    """Decrement the count-DOWN field/side durations (Trick Room / Gravity / Magic & Wonder Room /
    Tailwind). Weather/terrain are tracked as count-UP AGE (no clean remaining in the snapshot) and
    screens' representation is unconfirmed → both deferred to B0c/validation (logged when present)."""
    field = snapshot.get("field") or {}
    for k in ("trick_room_turns_remaining", "gravity_turns_remaining",
              "magic_room_turns_remaining", "wonder_room_turns_remaining"):
        v = field.get(k)
        if isinstance(v, (int, float)) and v > 0:
            field[k] = max(0, int(v) - 1)
    for side in ("our", "opp"):
        sc = ((snapshot.get("side_conditions") or {}).get(f"{side}_side")) or {}
        tw = sc.get("tailwind_turns_remaining")
        if isinstance(tw, (int, float)) and tw > 0:
            sc["tailwind_turns_remaining"] = max(0, int(tw) - 1)
        if sc.get("screens") and log is not None:
            log.note("screens_duration")


# ── mega evolution (gimmick, resolves first) ──────────────────────────────────
def apply_mega(mon: Optional[dict], *, log: Optional[UnmodelledLog] = None) -> bool:
    """Mega-evolve ``mon`` in place: swap to the mega forme's stats / ability / types so the SAME turn's
    damage uses them. Stats use the base-stat DELTA (``cur + (mega_base − base_base)``) — exact at L50 (the
    EV/IV term cancels) and valid for the expected (fractional) ``distribution`` stats too. Species→forme
    (so ``_effective_types`` returns the mega typing/STAB); ability→the forme's fixed ability (Technician,
    Fairy Aura, …) for the damage mults. No-op + log on a missing forme / base stats. Returns True if applied."""
    if not mon or mon.get("is_mega"):
        return False
    base = mon.get("base_species") or mon.get("species")
    formes = get_pokedex().mega_formes_for(base) or []
    if not formes:
        if log is not None:
            log.note("mega_no_forme:" + str(base))
        return False
    forme = formes[0]
    if len(formes) > 1:                                     # Charizard/Mewtwo X-vs-Y: disambiguate by the
        stone = norm_species(mon.get("known_item") or "")  # held megastone (…itex/…itey) when KNOWN
        suffix = "x" if stone.endswith("x") else "y" if stone.endswith("y") else None
        if suffix:
            forme = next((f for f in formes if str(f.get("forme", "")).lower().endswith(suffix)), forme)
        elif log is not None:
            log.note("mega_ambiguous_forme:" + str(base))   # stone unknown → first forme, surfaced
    forme_name, forme_abil = forme.get("forme"), forme.get("ability")
    base_st, mega_st = dex_base_stats(base), dex_base_stats(forme_name)
    if not base_st or not mega_st:
        if log is not None:
            log.note("mega_no_base_stats:" + str(forme_name))
        return False
    stats = ((mon.get("stats_estimate") or {}).get("stats")) or {}
    for k in _STAT_KEYS:
        if stats.get(k) is not None and base_st.get(k) is not None and mega_st.get(k) is not None:
            stats[k] = float(stats[k]) + (float(mega_st[k]) - float(base_st[k]))
    mon["species"] = forme_name
    mon["is_mega"] = True
    if forme_abil:
        mon["mega_ability"] = forme_abil
        mon["known_ability"] = forme_abil                   # mirror the parser's mega'd-mon representation
    return True


# ══════════════════════════════════════════════════════════════════════════════
# B0b — 4-action sequencer  (switches first, then moves by priority bracket + speed)
# ══════════════════════════════════════════════════════════════════════════════
def _mon_at(snapshot: dict, slot_key: str) -> Optional[dict]:
    side = "our" if str(slot_key).startswith("our") else "opp"
    return (snapshot.get(f"{side}_active") or {}).get(slot_key)


def _side_tailwind(snapshot: dict, slot_key: str) -> bool:
    side = "our" if str(slot_key).startswith("our") else "opp"
    sc = ((snapshot.get("side_conditions") or {}).get(f"{side}_side")) or {}
    return bool(sc.get("tailwind_turns_remaining"))


def _boost_mult(stage) -> float:
    s = int(stage or 0)
    return (2 + s) / 2.0 if s >= 0 else 2.0 / (2 - s)


def _move_effective_priority(move_id: str, attacker_mon: dict) -> int:
    """Effective priority bracket = Move.priority + Prankster(+1 to status) + Gale Wings(+1 to Flying at
    full HP) — the same physics as A3's analyze_speed_order."""
    data = _gen9_moves().get(move_id) or {}
    p = int(data.get("priority", 0) or 0)
    ab = resolve_active_ability_json(attacker_mon)[0]
    cat = (data.get("category") or "").lower()
    mtype = (champ_type(move_id, data.get("type")) or "").upper()
    if ab == "prankster" and cat == "status":
        p += 1
    elif ab == "galewings" and mtype == "FLYING" and float(attacker_mon.get("hp_pct") or 0.0) >= 100.0:
        p += 1
    return p


def order_moves(snapshot: dict, move_actions: dict, *, log: Optional[UnmodelledLog] = None):
    """Order MOVE actions for resolution: effective priority bracket (desc) then effective speed (desc;
    Trick Room flips the speed comparison). Returns ``[(slot_key, action)]``; fainted/empty/move-less
    movers are dropped. A speed tie keeps insertion order (Tier-2; the search may coin-flip it later)."""
    field = snapshot.get("field") or {}
    tr = bool(field.get("trick_room_turns_remaining"))
    weather = _canon(field.get("weather")) or None      # canonical token so weather-speed abilities fire (audit 2026-07-01)
    magic_room = bool(field.get("magic_room_turns_remaining"))
    rows = []
    for slot_key, act in (move_actions or {}).items():
        mon = _mon_at(snapshot, slot_key)
        if not act or mon is None or is_fainted(mon) or not act.get("move"):
            continue
        mid = norm_species(act["move"])
        prio = _move_effective_priority(mid, mon)
        spd = _effective_speed(mon, _side_tailwind(snapshot, slot_key), weather, magic_room)[0]
        rows.append((prio, (-spd if tr else spd), slot_key, act))
    rows.sort(key=lambda r: (r[0], r[1]), reverse=True)
    return [(sk, a) for (_p, _s, sk, a) in rows]


# ══════════════════════════════════════════════════════════════════════════════
# B0c — per-move damage resolution + the turn-stepper
# ══════════════════════════════════════════════════════════════════════════════
def _screens_tuple(side_sc: Optional[dict]) -> tuple:
    """(phys_screen, spec_screen) from a side's screens dict (Reflect / Light Screen / Aurora Veil)."""
    keys = {str(k).lower().replace("_", "") for k in ((side_sc or {}).get("screens") or {})}
    return ("reflect" in keys or "auroraveil" in keys,
            "lightscreen" in keys or "auroraveil" in keys)


def _foes(attacker_slot: str) -> list:
    return ["opp_a", "opp_b"] if str(attacker_slot).startswith("our") else ["our_a", "our_b"]


def _ally_slot(slot_key: str) -> str:
    """The other active slot on the SAME side (our_a↔our_b, opp_a↔opp_b)."""
    return {"our_a": "our_b", "our_b": "our_a", "opp_a": "opp_b", "opp_b": "opp_a"}.get(slot_key, slot_key)


def _count_fainted_side(snapshot: dict, side: str) -> int:
    """Fainted mons on ``side`` (active ∪ bench) — Last Respects' fainted-ally count (attacker is alive)."""
    mons = list((snapshot.get(f"{side}_active") or {}).values()) + (snapshot.get(f"{side}_bench") or [])
    return sum(1 for m in mons if m and is_fainted(m))


def _variable_bp(snapshot: dict, att: dict, defn: dict, def_slot: str, mid: str, static_bp: float,
                 weather, magic_room: bool) -> float:
    """The REAL base power of a variable-BP move from live context (weight / speed / HP / boosts / fainted
    allies / times-hit), via the SHARED ``damage_mechanics.variable_base_power`` — graceful fallback to
    ``static_bp`` when context is missing. Reuses the same primitive the encoder's damage band uses (parity)."""
    def_side = "our" if str(def_slot).startswith("our") else "opp"
    att_side = "opp" if def_side == "our" else "our"
    sc = snapshot.get("side_conditions") or {}
    att_tw = bool(((sc.get(f"{att_side}_side")) or {}).get("tailwind_turns_remaining"))
    def_tw = bool(((sc.get(f"{def_side}_side")) or {}).get("tailwind_turns_remaining"))
    pos = lambda m: sum(int(v) for v in ((m.get("boosts") or {}).values()) if isinstance(v, (int, float)) and v > 0)
    return float(variable_base_power(
        mid, static_bp,
        attacker_weight=species_weight(att.get("species") or att.get("base_species")),
        target_weight=species_weight(defn.get("species") or defn.get("base_species")),
        attacker_hp_frac=(float(att.get("hp_pct") or 0.0) / 100.0),
        attacker_speed=_effective_speed(att, att_tw, weather, magic_room)[0],
        target_speed=_effective_speed(defn, def_tw, weather, magic_room)[0],
        attacker_pos_boosts=pos(att), target_pos_boosts=pos(defn),
        fainted_allies=_count_fainted_side(snapshot, att_side),
        times_hit=int(att.get("times_attacked") or 0)))


def _note(log: Optional[UnmodelledLog], key: str) -> None:
    if log is not None:
        log.note(key)


def _ally_active(snapshot: dict, side: str, mon: dict) -> Optional[dict]:
    """The OTHER live active mon on ``side`` (the ally of ``mon``), by object identity — doubles only."""
    for m in (snapshot.get(f"{side}_active") or {}).values():
        if m is not None and m is not mon and not is_fainted(m):
            return m
    return None


def _ally_damage_mult(snapshot: dict, att: dict, defn: dict, def_slot: str, mtype: str, phys: bool, *,
                      helping_hand: bool, log: Optional[UnmodelledLog] = None) -> float:
    """Doubles ally-sourced damage modifiers (none are in _ability_damage_mult, which only sees the att/def
    OWN abilities): Helping Hand ×1.5 (per-turn) · att-ally Power Spot ×1.3 / Battery ×1.3 (special) / Steely
    Spirit ×1.5 (Steel) · def-ally Friend Guard ×0.75. Each firing notes ``applied:<id>`` (informational)."""
    mult = 1.0
    if helping_hand:
        mult *= 1.5
        _note(log, "applied:helpinghand")
    def_side = "our" if str(def_slot).startswith("our") else "opp"
    att_side = "opp" if def_side == "our" else "our"
    att_ally = _ally_active(snapshot, att_side, att)
    if att_ally:
        aa = resolve_active_ability_json(att_ally)[0]
        if aa == "powerspot":
            mult *= 1.3; _note(log, "applied:powerspot")
        elif aa == "battery" and not phys:
            mult *= 1.3; _note(log, "applied:battery")
        elif aa == "steelyspirit" and mtype == "STEEL":
            mult *= 1.5; _note(log, "applied:steelyspirit_ally")
    def_ally = _ally_active(snapshot, def_side, defn)
    if def_ally and resolve_active_ability_json(def_ally)[0] == "friendguard":
        mult *= 0.75; _note(log, "applied:friendguard")
    return mult


def _apply_self_hp(att: Optional[dict], mid: str, data: dict, total_abs: float, *,
                   log: Optional[UnmodelledLog] = None) -> None:
    """The attacker's OWN HP change from an offensive move (data-driven): recoil (frac of damage dealt),
    drain (heal frac of damage dealt, ×1.3 Big Root), Mind Blown/Steel Beam (flat 50%), Struggle (25%),
    crash on miss (HJK, expected (1−hit)×50%), Life Orb recoil (10%), self-destruct (Explosion/Final Gambit
    → faint). ``total_abs`` = absolute HP dealt this move. Runs only for damaging moves (caller gates)."""
    if not att or is_fainted(att):
        return
    att_max = mon_max_hp(att)
    item = resolve_item_json(att)[0]
    if att_max:
        recoil = data.get("recoil")
        if recoil and total_abs > 0:
            apply_damage_pct(att, (float(recoil[0]) / float(recoil[1])) * total_abs / att_max * 100.0, log=log)
            _note(log, "applied:recoil:" + mid)
        if data.get("mindBlownRecoil"):
            apply_damage_pct(att, 50.0, log=log); _note(log, "applied:mindblown:" + mid)
        if data.get("struggleRecoil"):
            apply_damage_pct(att, 25.0, log=log); _note(log, "applied:struggle")
        if data.get("hasCrashDamage"):
            acc = data.get("accuracy")
            hit = (1.0 if (acc is True or acc is None or _move_always_hit(mid))
                   else max(0.0, min(1.0, float(acc) / 100.0)))
            if (1.0 - hit) > 0:
                apply_damage_pct(att, (1.0 - hit) * 50.0, log=log); _note(log, "applied:crash:" + mid)
        drain = data.get("drain")                      # drain heals BEFORE Life Orb recoil (Showdown order),
        if drain and total_abs > 0:                     # so a drain move can't be fainted by its own Life Orb
            heal = (float(drain[0]) / float(drain[1])) * total_abs / att_max * 100.0
            apply_heal_pct(att, heal * (1.3 if item == "bigroot" else 1.0))
            _note(log, "applied:drain:" + mid)
        if item == "lifeorb" and total_abs > 0:
            apply_damage_pct(att, 10.0, log=log); _note(log, "applied:lifeorb_recoil")
    if data.get("selfdestruct"):
        if mid == "finalgambit":
            _note(log, "fixed_damage:finalgambit")   # damage = user HP (not a normal calc) — residual
        apply_damage_pct(att, 100.0, log=log)
        _note(log, "applied:selfdestruct:" + mid)


def _survives_ko(defn: dict, slot: str, endured: "frozenset[str]") -> bool:
    """Would a survive-KO mechanic spare this defender from fainting this hit? Endure (any HP, this turn) or
    Focus Sash / Sturdy (ONLY from full HP — the full-HP gate self-consumes them on the 2nd hit). Opp items/
    abilities fire only when KNOWN (belief-uncertain otherwise)."""
    if slot in endured:
        return True
    if float(defn.get("hp_pct") or 0.0) >= 100.0:
        sash = norm_species(defn.get("known_item") or "") == "focussash" and not defn.get("item_consumed")
        return sash or norm_species(defn.get("known_ability") or "") == "sturdy"
    return False


def _move_damage_pct(snapshot: dict, att: dict, mid: str, data: dict, defn: dict, def_slot: str, *,
                     is_spread: bool, helping_hand: bool = False, log: Optional[UnmodelledLog] = None) -> tuple:
    """EXPECTED damage + the roll DISTRIBUTION for one (attacker, move, defender) hit, as
    ``(mean, lo_nc, hi_nc, hi_crit)`` percent of the DEFENDER's max HP. ``mean`` (mean roll × expected-crit ×
    accuracy) is byte-identical to the prior scalar; ``lo_nc``/``hi_nc`` = the min/max NON-crit roll, ``hi_crit``
    = max roll × crit (reachability). Reuses the encoder's standalone primitives; boosts ARE applied."""
    field = snapshot.get("field") or {}
    # canonicalise to the encoder's uppercase tokens (RAINDANCE / ELECTRIC_TERRAIN): the parser snapshot stores
    # raw 'RainDance' / 'electric', but _situational_damage_mult / _defender_profile / _effective_speed all
    # compare against the CANONICAL tokens, so raw silently no-ops every weather/terrain damage mult AND every
    # weather-speed ability. Mirror state_encoder.py (audit 2026-07-01 — forward-vs-leaf divergence).
    weather = _canon(field.get("weather")) or None
    terrain = _TERRAIN_TO_FIELD.get(field.get("terrain") or "") or None
    gravity = bool(field.get("gravity_turns_remaining"))
    magic_room = bool(field.get("magic_room_turns_remaining"))
    def_side = "our" if str(def_slot).startswith("our") else "opp"
    def_sc = ((snapshot.get("side_conditions") or {}).get(f"{def_side}_side")) or {}
    d = _defender_profile(defn, _screens_tuple(def_sc), bool(def_sc.get("tailwind_turns_remaining")),
                          weather, gravity, magic_room)
    if not d or not d.get("hp") or not d.get("hp_frac"):
        if log is not None:
            log.note("no_defender_profile")
        return _ZERO_DMG
    ability_id = resolve_active_ability_json(att)[0]
    # mtype pipeline mirrors the encoder twin (state_encoder.py:748-752): Champions type → Weather Ball /
    # Terrain Pulse → -ate / Normalize / Liquid Voice retype. Without effective_move_type a Pixilate Hyper
    # Voice etc. kept its raw NORMAL type → wrong type-eff (0× into Ghost!), immunity, and STAB (audit 2026-07-01).
    base_type = _canon(champ_type(mid, data.get("type")))
    mtype = _canon(field_dependent_move_type(mid, base_type, weather, terrain)) or base_type
    mtype = effective_move_type(mid, mtype, ability_id) or mtype
    if _move_immune(mtype, d, ability_id, mid):
        return _ZERO_DMG
    raw_bp = champ_bp(mid, data.get("basePower") or 0)
    if mid in _VBP_HANDLED:                          # recompute REAL bp (Low Kick/Heavy Slam/Eruption/…)
        raw_bp = _variable_bp(snapshot, att, defn, def_slot, mid, raw_bp, weather, magic_room)
    elif mid in _VBP_RESIDUAL and log is not None:
        log.note("variable_bp_residual:" + mid)      # Bolt Beak/Fishious Rend/Hard Press — static bp used
    if not raw_bp:
        if log is not None:
            log.note("no_basepower:" + mid)
        return _ZERO_DMG
    phys = (data.get("category") or "").lower() == "physical"
    hits_def = phys or mid in _DEF_HIT_SPECIAL       # Psyshock family: special move that hits physical Def
    est = (att.get("stats_estimate") or {}).get("stats") or {}
    if mid == "bodypress":                            # Body Press uses the attacker's DEF (+ its Def boosts)
        A = est.get("def"); a_boost = (att.get("boosts") or {}).get("def", 0)
    elif mid == "foulplay":                           # Foul Play uses the TARGET's Atk (+ the target's Atk boosts)
        d_est = (defn.get("stats_estimate") or {}).get("stats") or {}
        A = d_est.get("atk"); a_boost = (defn.get("boosts") or {}).get("atk", 0)
    else:
        A = est.get("atk") if phys else est.get("spa")
        a_boost = (att.get("boosts") or {}).get("atk" if phys else "spa", 0)
    if A:
        A = A * _boost_mult(a_boost)
    use_def = hits_def ^ bool(field.get("wonder_room_turns_remaining"))   # Wonder Room swaps Def<->SpD
    D0 = d.get("def") if use_def else d.get("spd")
    if D0:
        D0 = D0 * _boost_mult((defn.get("boosts") or {}).get("def" if use_def else "spd", 0))
    if not A or not D0:
        if log is not None:
            log.note("missing_stat")
        return _ZERO_DMG
    tmult = _type_mult(mtype, d.get("types") or [])
    if mtype == "FIRE" and d.get("tar_shot"):
        tmult *= 2.0                                  # Tar Shot: Fire ×2
    if tmult <= 0:
        return _ZERO_DMG
    stab = mtype in [t.upper() for t in (_effective_types(att) or [])]
    burned = str(att.get("status") or "").lower() == "brn"
    item = resolve_item_json(att)[0]
    att_side = "opp" if def_side == "our" else "our"
    abm = _ability_damage_mult(mid, ability_id, d.get("ability"), eff_move_type=mtype, is_stab=stab,
                               is_physical=phys, type_mult=tmult, hp_frac=d.get("hp_frac") or 0.0,
                               att_burned=burned, att_statused=bool(att.get("status")), is_spread=is_spread,
                               fainted_allies=_count_fainted_side(snapshot, att_side),   # Supreme Overlord
                               defender_intact_disguise=d.get("intact_disguise", False), weather=weather)
    sit = _situational_damage_mult(mtype, phys, weather, terrain, d, burned,
                                   item == "lifeorb", item in ("choiceband", "choicespecs"),
                                   hits_def=hits_def, grassy_eq=mid in _GRASSY_WEAKENED) * abm
    if item == "expertbelt" and tmult > 1.0:
        sit *= _BAND_ITEM_MULT                         # Expert Belt: super-effective ×~1.2
    if _TYPE_BOOST_TYPE.get(item) == mtype:
        sit *= _BAND_ITEM_MULT                         # Charcoal/Mystic Water/Metal Coat/plates ×~1.2 (common)
    rb = d.get("resist_berry")
    if rb == mtype and (mtype == "NORMAL" or tmult > 1.0):
        sit *= 0.5                                     # defender resist berry halves a SE (or Normal) hit
    crit_mult = (1.0 if d.get("ability") in ("battlearmor", "shellarmor")
                 else _expected_crit_mult(mid, ability_id, item in ("scopelens", "razorclaw")))
    sit *= crit_mult                                 # EXPECTED crit folded into the mean
    sit *= _ally_damage_mult(snapshot, att, defn, def_slot, mtype, phys, helping_hand=helping_hand, log=log)
    hmin, hmax = _move_hit_range(mid)
    if ability_id == "skilllink" and hmin != hmax:
        hmin = hmax
    elif item == "loadeddice" and (hmin, hmax) == (2, 5):
        hmin = 4                                       # Loaded Dice: min 4 hits on a 2-5 move
    hp_frac = float(d.get("hp_frac"))
    scale = hp_frac * 100.0
    acc = data.get("accuracy")
    hit = (1.0 if (acc is True or acc is None or _move_always_hit(mid))
           else max(0.0, min(1.0, float(acc) / 100.0)))
    dmin, dmax = _damage_band(raw_bp, A, D0, d.get("hp"), hp_frac, tmult, stab, is_spread, sit, hmin, hmax)
    mean_pct = ((float(dmin) + float(dmax)) / 2.0) * scale * hit      # EV ×accuracy (band frac-of-current → %max)
    # non-crit roll band (the mean keeps the EXPECTED-crit fold):
    sit_nc = sit / crit_mult if crit_mult else sit
    nmin, nmax = _damage_band(raw_bp, A, D0, d.get("hp"), hp_frac, tmult, stab, is_spread, sit_nc, hmin, hmax)
    crit_f = (2.25 if ability_id == "sniper" else 1.5) if crit_mult > 1.0 else 1.0   # Sniper crit ×2.25
    # shadows are CONDITIONAL-ON-HIT (no ×accuracy): KO-reachability shouldn't be erased by a possible miss —
    # accuracy is a separate miss event, not a uniform damage scaler (audit 2026-06-30).
    return (mean_pct, float(nmin) * scale, float(nmax) * scale, float(nmax) * crit_f * scale)


def resolve_move(snapshot: dict, attacker_slot: str, action: dict, *,
                 protected: "frozenset[str]" = frozenset(),
                 helped: "frozenset[str]" = frozenset(),
                 endured: "frozenset[str]" = frozenset(),
                 log: Optional[UnmodelledLog] = None) -> None:
    """Resolve one chosen MOVE: compute + apply EXPECTED damage to its target(s), then the attacker's own
    HP change (recoil/drain/crash/self-destruct). Status / unknown moves are Tier-3 (logged, no effect)
    except the modelled ones (heal, Helping Hand). Spread moves hit both live foes (×0.75 via the band). A
    target slot in ``protected`` takes no damage; ``attacker_slot in helped`` ⇒ Helping Hand ×1.5."""
    att = _mon_at(snapshot, attacker_slot)
    if att is None or is_fainted(att) or not action or not action.get("move"):
        return
    mid = norm_species(action["move"])
    data = _gen9_moves().get(mid)
    if not data:
        if log is not None:
            log.note("unknown_move:" + mid)
        return
    if (data.get("category") or "").lower() == "status":
        if mid in _HEAL_MOVES:                       # Tier-1: flat 50% self-recovery
            apply_heal_pct(att, 50.0)
        elif mid in _HANDLED_STATUS:                 # effect modelled elsewhere (Helping Hand → ally damage)
            pass
        elif log is not None:
            log.note("status_move:" + mid)           # other status moves: B0c core models damage only
        return
    spread = bool(is_spread_target(data.get("target")))
    if spread:
        targets = [s for s in _foes(attacker_slot) if not is_fainted(_mon_at(snapshot, s))]
        # 'allAdjacent' (Earthquake/Surf/Discharge/Explosion) ALSO hits the user's own ally; 'allAdjacentFoes'
        # (Rock Slide/Heat Wave/Snarl) does not. The self-hit on the ally is a real downside the depth-1 search
        # must see, else it over-values ground/water spread lines (audit 2026-07-01). Immunity (Flying/Levitate/
        # Air Balloon) and Protect on the ally are handled by the damage loop below (tmult<=0 → 0, protected → skip).
        if norm_species(data.get("target") or "") == "alladjacent":
            ally = _ally_slot(attacker_slot)
            if ally != attacker_slot and _mon_at(snapshot, ally) is not None \
                    and not is_fainted(_mon_at(snapshot, ally)):
                targets.append(ally)
    else:
        tgt = action.get("target")
        targets = ([tgt] if (tgt and _mon_at(snapshot, tgt) is not None
                             and not is_fainted(_mon_at(snapshot, tgt)))
                   else [s for s in _foes(attacker_slot) if not is_fainted(_mon_at(snapshot, s))][:1])
    if not targets:
        if log is not None:
            log.note("no_live_target")
        return
    multi = spread and len(targets) > 1
    is_helped = attacker_slot in helped
    total_abs = 0.0                                  # absolute HP dealt this move (for recoil / drain)
    for tgt_slot in targets:
        defn = _mon_at(snapshot, tgt_slot)
        if defn is None or is_fainted(defn):
            continue
        if tgt_slot in protected:                   # Protect blocks the hit (Tier-1)
            if log is not None:
                log.note("blocked_by_protect")
            continue
        vol = defn.get("volatiles") or {}
        if vol.get("has_substitute"):               # the Substitute absorbs this hit (no sub HP in the snapshot)
            vol["has_substitute"] = False           # sub breaks; later hits this turn land normally
            # recoil/drain key off damage DEALT — incl. damage the Substitute soaks — so feed the mean to total_abs
            sub_mean = _move_damage_pct(snapshot, att, mid, data, defn, tgt_slot,
                                        is_spread=multi, helping_hand=is_helped, log=None)[0]
            total_abs += sub_mean * (mon_max_hp(defn) or 0.0) / 100.0
            _note(log, "applied:substitute_absorb")
            continue
        mean_pct, lo_nc, hi_nc, hi_crit = _move_damage_pct(snapshot, att, mid, data, defn, tgt_slot,
                                                           is_spread=multi, helping_hand=is_helped, log=log)
        if mean_pct > 0 or hi_crit > 0:
            survive = _survives_ko(defn, tgt_slot, endured)   # Focus Sash / Sturdy / Endure
            dealt = apply_damage_rolled(defn, mean_pct, lo_nc, hi_nc, hi_crit, survive=survive)
            if survive:
                _note(log, "applied:survive_ko")
            total_abs += dealt * (mon_max_hp(defn) or 0.0) / 100.0
    _apply_self_hp(att, mid, data, total_abs, log=log)        # recoil / drain / crash / self-destruct


def white_box_sim(snapshot: dict, actions: dict, *, log: Optional[UnmodelledLog] = None):
    """Step ``snapshot`` forward ONE turn under the joint ``actions`` (dict ``{slot_key: action}``).
    EXPECTED-value + deterministic → the Level-C search's forward model. Each action is
    ``{kind:'move', move:<name>, target:<slot>}`` | ``{kind:'switch', bench_index:int}`` | ``{kind:'pass'}``.
    Resolution: gimmick (Tier-3 logged) -> switches -> moves in priority/speed order (a mon fainted before
    it acts loses its move) -> end-of-turn residuals. Returns ``(next_snapshot, UnmodelledLog)``; the input
    is never mutated (deep-cloned)."""
    if log is None:
        log = UnmodelledLog()
    s = clone_state(snapshot)
    for slot_key, act in (actions or {}).items():       # gimmick resolves FIRST (mega stats affect order+damage)
        if act and act.get("mega"):
            apply_mega(_mon_at(s, slot_key), log=log)
    # Switches: resolve each target to its MON OBJECT against the pre-switch bench FIRST, then switch by the
    # target's CURRENT index — so a simultaneous double-switch's 2nd index isn't stale after the 1st pop.
    _targets = []
    for slot_key, act in (actions or {}).items():
        if act and act.get("kind") == "switch":
            side = "our" if str(slot_key).startswith("our") else "opp"
            bench = s.get(f"{side}_bench") or []
            bi = int(act.get("bench_index", act.get("bench_slot", 0)) or 0)
            _targets.append((slot_key, side, bench[bi] if 0 <= bi < len(bench) else None))
    for slot_key, side, target in _targets:
        bench = s.get(f"{side}_bench") or []
        cur = next((i for i, m in enumerate(bench) if m is target), None) if target is not None else None
        if cur is None:
            log.note("switch_bad_index")
            continue
        switch_in(s, slot_key, cur, log=log)
    move_actions = {k: v for k, v in (actions or {}).items() if v and v.get("kind") == "move"}
    # A slot that used a Protect-class move this turn blocks all incoming damage (Tier-1).
    protected = frozenset(
        k for k, v in move_actions.items()
        if norm_species(v.get("move") or "") in _PROTECT_MOVES)
    # A slot whose ALLY used Helping Hand on it gets ×1.5 damage (the target_slot, else the caster's ally slot).
    helped = set()
    for k, v in move_actions.items():
        if norm_species(v.get("move") or "") == "helpinghand":
            tgt = v.get("target")
            helped.add(tgt if tgt else _ally_slot(k))
    helped = frozenset(helped)
    # A slot that used Endure this turn survives any KO at 1 HP (all hits this turn).
    endured = frozenset(k for k, v in move_actions.items()
                        if norm_species(v.get("move") or "") == "endure")
    for slot_key, act in order_moves(s, move_actions, log=log):
        mon = _mon_at(s, slot_key)
        if mon is None or is_fainted(mon):
            log.note("mover_fainted_before_acting")
            continue
        resolve_move(s, slot_key, act, protected=protected, helped=helped, endured=endured, log=log)
    apply_residuals(s, log=log)
    return s, log
