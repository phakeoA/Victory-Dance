"""
live_belief_feed.py — per-turn feeders for the within-game opponent belief (Level C / A3c).
================================================================================

PURE helpers that drive a per-battle :class:`~v_dance.parser.match_belief.MatchBelief` from the
PREVIOUS turn's resolved events, so the live opponent belief sharpens as the game reveals damage
(A2) and — next — speed (A3). They run in ``SplicingVGCPlayerBase`` BEFORE the opp snapshot is
enriched + spliced into the encoded state (see ``live_vgc_base._feed_match_belief``).

Design notes (grounded):
  * The DAMAGE reference calculator is the OFFLINE ``_damage_band`` machinery the live encoder
    already imports + uses for its move-feature block (USER decision 2026-06-30): no poke-env
    mutation, and consistent with how opp stats are derived everywhere (from the BeliefState, not
    poke-env, whose opp ``.stats`` are ``None``). This mirrors the proven offline probe
    (``scratch/levelC_a2_damage_probe.py``) exactly — same full-HP gate, same ``_damage_band`` +
    ``damage_main_term`` reference, same both-direction logic.
  * One side of every constraint is OUR exact mon: its real stats come from the live poke-env team
    (passed in as ``our_stats``); the opponent is referenced against its modal Pikalytics spread and
    re-scaled per candidate inside ``MatchBelief``.
  * Crit is NOT captured by the parser, so the live damage feed uses the CONSERVATIVE default σ
    (``_SIGMA_LOG_DEFAULT`` ≈ 0.23, crit + mitigation unknown) — a correct, never-over-confident
    narrowing. Sharpening σ with the ``|-crit|`` flag / revealed item+ability is a clean follow-on.
  * Identity is gated via :func:`~v_dance.parser.match_belief.identity_reliable` — a Zoroark
    Illusion / Ditto-Transform mon's damage must NOT update the apparent species' belief.

This module is import-light (no poke-env) and has NO side effects at import time.
"""
from __future__ import annotations

from typing import Callable, Optional

from v_dance.parser.belief_state import dex_base_stats, calc_full_stats
from v_dance.parser.match_belief import (
    _move_category, _offensive_stat_for_move, damage_main_term, identity_reliable,
    analyze_speed_order, _SIGMA_ROLL, _SIGMA_MITIG, _SIGMA_MISC,
)
from v_dance.parser.vod_parser.pokedex import norm_species
from v_dance.encoders.battle_mechanics import (
    _damage_band, _type_mult, _effective_types, champ_bp, champ_type, is_spread_target, _gen9_moves,
)


def _live_damage_sigma(mitigation_known: bool) -> float:
    """Live log-space σ for the A2 damage likelihood. Crit is ALWAYS known live (the parser stamps
    ``|-crit|`` presence/absence and the feeder folds it into the reference), so the crit² term is
    DROPPED. The mitigation² term (an unknown resist-berry/Multiscale/screen) is dropped only when the
    DEFENDER's context is known — always for OUR mon (the offensive direction), and for the opp defender
    only once its item+ability are revealed. Sharper than the offline default (≈0.23)."""
    var = _SIGMA_ROLL ** 2 + _SIGMA_MISC ** 2
    if not mitigation_known:
        var += _SIGMA_MITIG ** 2
    return var ** 0.5


def _persp_key(slot: str, own_role: str):
    """Map a RAW protocol slot key (``'p1a'`` / ``'p2b'``) to the perspective key + ownership for
    ``own_role`` — ``('our_a', True)`` / ``('opp_b', False)`` — or ``None`` when the slot is malformed."""
    if not slot or len(slot) < 3:
        return None
    letter = slot[-1]
    if letter not in ("a", "b"):
        return None
    is_our = slot.startswith(own_role)
    return ("our_" if is_our else "opp_") + letter, is_our


def _modal_full(belief, species: Optional[str], level: int = 50):
    """``calc_full_stats`` for the species' MODAL Pikalytics spread (the reference defender/attacker),
    or ``None`` when there is no spread / base-stat data. Mirrors the offline probe's ``_modal_full``."""
    if not species:
        return None
    spreads = belief.spread_distribution(species, top_k=5)
    base = dex_base_stats(species)
    if not spreads or not base:
        return None
    return calc_full_stats(base, spreads[0]["evs_actual"], spreads[0]["nature"], level=level)


def _move_damage_ctx(move: Optional[str], attacker_mon: dict, defender_mon: dict):
    """``(category, off_stat, bp, mtype, type_mult, is_spread, is_stab)`` for a STANDARD damaging move
    (STAB from the ATTACKER's types, type_mult vs the DEFENDER's types), or ``None`` for a status /
    stat-independent / no-BP move. Mirrors the offline probe's ``_move_ctx``."""
    cat = _move_category(move)
    off_stat = _offensive_stat_for_move(move)
    std = _gen9_moves().get(norm_species(move)) or {}
    bp = champ_bp(norm_species(move), std.get("basePower") or 0)
    if off_stat is None or cat not in ("physical", "special") or not bp:
        return None
    mtype = (champ_type(norm_species(move), std.get("type") or "") or "").upper()
    tmult = _type_mult(mtype, _effective_types(defender_mon))
    is_stab = mtype in (_effective_types(attacker_mon) or [])
    return cat, off_stat, bp, mtype, tmult, is_spread_target(std.get("target")), is_stab


def feed_damage_constraints(
    mb,
    belief,
    prev_turn: Optional[dict],
    own_role: str,
    our_stats: dict,
    *,
    level: int = 50,
    on_log: Optional[Callable[[dict], None]] = None,
    damage_loglik_fn: Optional[Callable] = None,
) -> dict:
    """Feed the PREVIOUS turn's full-HP, non-KO damage events into ``mb`` (A2 live narrowing).

    For every damaging hit where exactly one side is ours and one is the opponent's:
      * **defensive** (our hit → opp defender): narrow the opp's Def/SpD/HP from how much it took;
      * **offensive** (opp hit → our defender): narrow the opp's Atk/SpA from how much it dealt.

    ``prev_turn`` is the raw parser turn dict (``state_before_actions[own_role]`` perspective snapshot
    + raw ``damage_events``). ``our_stats`` maps ``norm_species`` → our mon's exact stat dict (from the
    live poke-env team). Returns a counts dict (used/skipped) for diagnostics; ``on_log`` receives it.
    Pure + side-effect-free except for the ``observe_damage_*`` calls on ``mb``.
    """
    stats = {"used_def": 0, "used_off": 0, "skipped_hp": 0,
             "skipped_move": 0, "skipped_ctx": 0, "skipped_id": 0}
    if not mb or belief is None or not prev_turn:
        return stats
    sb = (prev_turn.get("state_before_actions") or {}).get(own_role) or {}
    our_act = sb.get("our_active") or {}
    opp_act = sb.get("opp_active") or {}

    for ev in prev_turn.get("damage_events") or []:
        if ev.get("event") != "damage" or not ev.get("source_move"):
            continue
        delta = abs(float(ev.get("hp_pct_delta") or 0.0))
        after = float(ev.get("hp_pct_after") or 0.0)
        if (after + delta) < 95.0 or after <= 0.0 or delta <= 0:   # full-HP non-KO only
            stats["skipped_hp"] += 1
            continue
        dk = _persp_key(ev.get("slot") or "", own_role)
        sk = _persp_key(ev.get("source_slot") or "", own_role)
        if not dk or not sk:
            continue
        (dkey, d_is_our), (skey, s_is_our) = dk, sk
        if d_is_our == s_is_our:        # need exactly one our-side and one opp-side mon
            continue
        move = ev.get("source_move")
        obs = delta / 100.0             # fraction of the defender's MAX hp

        if (not d_is_our) and s_is_our:                              # DEFENSIVE — narrow opp bulk
            opp_mon, our_mon = opp_act.get(dkey), our_act.get(skey)
            if not opp_mon or not our_mon:
                stats["skipped_ctx"] += 1
                continue
            if not identity_reliable(opp_mon):
                stats["skipped_id"] += 1
                continue
            sp = opp_mon.get("base_species") or opp_mon.get("species")
            # A2 v2: poke-env per-candidate calc (item + ability marginalised) when a live calc is wired,
            # else the analytic `_damage_band` path below. A None return falls through to analytic.
            if damage_loglik_fn is not None and sp:
                ll = damage_loglik_fn(
                    direction="def", opp_key=dkey, our_key=skey, opp_species=sp,
                    opp_snapshot_mon=opp_mon, move=move, crit=bool(ev.get("crit")), obs_frac=obs)
                if ll:
                    mb.observe_damage_loglik(sp, ll)
                    stats["used_def"] += 1
                    continue
            ctx = _move_damage_ctx(move, our_mon, opp_mon)
            if not ctx:
                stats["skipped_move"] += 1
                continue
            cat, _off, bp, _mt, tmult, is_spread, is_stab = ctx
            a_key = norm_species(ev.get("source_species") or our_mon.get("species") or "")
            A = (our_stats.get(a_key) or {}).get(_off)
            mf = _modal_full(belief, sp, level)
            stat = "def" if cat == "physical" else "spd"
            if not A or not mf or not mf.get(stat) or not mf.get("hp"):
                stats["skipped_ctx"] += 1
                continue
            sit = 1.5 if ev.get("crit") else 1.0          # fold a known crit into the reference
            bmin, bmax = _damage_band(bp, A, mf[stat], mf["hp"], 1.0, tmult, is_stab, is_spread,
                                      situational_mult=sit)
            if bmax <= 0:
                stats["skipped_ctx"] += 1
                continue
            # opp defender → mitigation known only once its item+ability are revealed (else σ keeps the
            # resist-berry/Multiscale/screen term); crit is known either way → its σ term is dropped.
            mitig_known = bool(opp_mon.get("known_item")) and bool(opp_mon.get("known_ability"))
            mb.observe_damage_taken(
                sp, mu_ref=(bmin + bmax) / 2.0, ref_main=damage_main_term(bp, A, mf[stat]),
                ref_def_stat=mf[stat], ref_hp=mf["hp"], category=cat, obs_frac_of_max=obs,
                sigma_log=_live_damage_sigma(mitig_known))
            stats["used_def"] += 1

        else:                                                        # OFFENSIVE — narrow opp offense
            our_mon, opp_mon = our_act.get(dkey), opp_act.get(skey)
            if not our_mon or not opp_mon:
                stats["skipped_ctx"] += 1
                continue
            if not identity_reliable(opp_mon):
                stats["skipped_id"] += 1
                continue
            sp = opp_mon.get("base_species") or opp_mon.get("species")
            # A2 v2: poke-env per-candidate calc (item + ability marginalised) when a live calc is wired,
            # else the analytic `_damage_band` path below. A None return (poke-env couldn't run) falls through.
            if damage_loglik_fn is not None and sp:
                ll = damage_loglik_fn(
                    direction="off", opp_key=skey, our_key=dkey, opp_species=sp, opp_snapshot_mon=opp_mon,
                    move=move, crit=bool(ev.get("crit")), obs_frac=obs)
                if ll:
                    mb.observe_damage_loglik(sp, ll)
                    stats["used_off"] += 1
                    continue
            ctx = _move_damage_ctx(move, opp_mon, our_mon)
            if not ctx:
                stats["skipped_move"] += 1
                continue
            cat, off_stat, bp, _mt, tmult, is_spread, is_stab = ctx
            d_key = norm_species(ev.get("species") or our_mon.get("species") or "")
            ours = our_stats.get(d_key) or {}
            dstat = "def" if cat == "physical" else "spd"
            D, hp = ours.get(dstat), ours.get("hp")
            mf = _modal_full(belief, sp, level)
            if not mf or not mf.get(off_stat) or not D or not hp:
                stats["skipped_ctx"] += 1
                continue
            sit = 1.5 if ev.get("crit") else 1.0          # fold a known crit into the reference
            bmin, bmax = _damage_band(bp, mf[off_stat], D, hp, 1.0, tmult, is_stab, is_spread,
                                      situational_mult=sit)
            if bmax <= 0:
                stats["skipped_ctx"] += 1
                continue
            # OUR defender → its item/ability/screens are known → drop the mitigation term (sharp σ).
            mb.observe_damage_dealt(
                sp, mu_ref=(bmin + bmax) / 2.0, ref_main=damage_main_term(bp, mf[off_stat], D),
                ref_atk_stat=mf[off_stat], category=cat, obs_frac_of_max=obs,
                sigma_log=_live_damage_sigma(True))
            stats["used_off"] += 1

    if on_log:
        on_log(stats)
    return stats


def _move_speed_props(move: Optional[str]) -> Optional[dict]:
    """``{priority, category, type}`` for ``analyze_speed_order`` from a move name (pinned Gen-9 data +
    Champions type override), or ``None`` when the move is unknown."""
    if not move:
        return None
    std = _gen9_moves().get(norm_species(move))
    if std is None:
        return None
    return {
        "priority": int(std.get("priority", 0) or 0),
        "category": (std.get("category") or "").lower(),
        "type": (champ_type(norm_species(move), std.get("type") or "") or ""),
    }


def feed_speed_constraints(
    mb,
    prev_turn: Optional[dict],
    own_role: str,
    speed_ctx: dict,
    *,
    on_log: Optional[Callable[[dict], None]] = None,
) -> dict:
    """Feed the PREVIOUS turn's move-ORDER into ``mb`` as speed-tier bounds (A3 live narrowing).

    For every (our active mover, opp active mover) pair, the relative ``execution_index`` tells us who
    moved first; :func:`~v_dance.parser.match_belief.analyze_speed_order` then decides whether that
    order was SPEED-determined (same effective priority bracket, not forced) and, if so, narrows the
    opp's Spe. Both directions of a sandwich (our_a, opp_Y, our_b) yield two informative bounds.

    ``speed_ctx`` is assembled by the player from the LIVE battle (poke-env), keyed by perspective:
      ``{trick_room, order_forced, our: {our_a:{eff_speed,ability,hp_frac}, ...},
         opp: {opp_a:{speed_mult_known,context_known,ability,hp_frac}, ...}}``.
    Our exact effective speed (the threshold) is known; the opp's KNOWN multipliers are divided out so
    the bound is in opp BASE-Spe units. ``order_forced`` (Quash/After-You/Quick-Claw/Instruct/Dancer,
    detected by the player) makes every read bail. Identity (Zoroark/Ditto) is gated per opp mover.
    Returns a counts dict; pure except for the ``observe_speed_bound`` calls on ``mb``.
    """
    stats = {"used": 0, "skipped_bracket": 0, "skipped_id": 0, "skipped_ctx": 0}
    if not mb or not prev_turn or not speed_ctx:
        return stats
    sb = (prev_turn.get("state_before_actions") or {}).get(own_role) or {}
    opp_act = sb.get("opp_active") or {}
    trick_room = bool(speed_ctx.get("trick_room"))
    order_forced = bool(speed_ctx.get("order_forced"))
    our_ctx = speed_ctx.get("our") or {}
    opp_ctx = speed_ctx.get("opp") or {}

    # First move per ACTIVE slot this turn (lowest execution_index); a slot moving twice is an
    # Instruct/Dancer repeat → force the whole turn's reads to bail (handled via order_forced below).
    our_moves: dict = {}
    opp_moves: dict = {}
    repeat = False
    for a in prev_turn.get("actions") or []:
        if a.get("event") != "move":
            continue
        pk = _persp_key(a.get("user_slot") or "", own_role)
        if not pk:
            continue
        key, is_our = pk
        bucket = our_moves if is_our else opp_moves
        if key in bucket:
            repeat = True
            if (a.get("execution_index") or 0) >= (bucket[key].get("execution_index") or 0):
                continue                       # keep the earliest move for the slot
        bucket[key] = a
    if repeat:
        order_forced = True                    # a repeated move (Instruct/Dancer) → order wasn't speed

    for okey, omove in opp_moves.items():
        opp_mon = opp_act.get(okey)
        if opp_mon is not None and not identity_reliable(opp_mon):
            stats["skipped_id"] += 1
            continue
        sp = (opp_mon or {}).get("base_species") or (opp_mon or {}).get("species")
        if not sp:
            stats["skipped_ctx"] += 1
            continue
        opp_props = _move_speed_props(omove.get("move"))
        octx = opp_ctx.get(okey) or {}
        # Skip when the CURRENT-board mon at this slot is known and is NOT the one that moved last turn
        # (a switch / faint+replacement since T-1): octx's speed multipliers + ability are the WRONG mon's
        # and would corrupt this species' Spe belief. Mirrors the A2 offensive guard. (audit 2026-06-30)
        if octx.get("species") and norm_species(sp) != octx["species"]:
            stats["skipped_ctx"] += 1
            continue
        if opp_props is None:
            stats["skipped_ctx"] += 1
            continue
        oidx = omove.get("execution_index") or 0
        for ukey, umove in our_moves.items():
            uctx = our_ctx.get(ukey) or {}
            our_props = _move_speed_props(umove.get("move"))
            our_eff = uctx.get("eff_speed")
            if our_props is None or not our_eff or our_eff <= 0:
                continue
            uidx = umove.get("execution_index") or 0
            if uidx == oidx:
                continue                       # same execution index (shouldn't happen) → no order
            obs = analyze_speed_order(
                opp_moved_first=oidx < uidx,
                our_eff_speed=our_eff,
                our_move=our_props, opp_move=opp_props,
                our_ability=uctx.get("ability"), opp_ability=octx.get("ability"),
                our_hp_frac=uctx.get("hp_frac"), opp_hp_frac=octx.get("hp_frac"),
                trick_room=trick_room, order_forced=order_forced,
                opp_speed_mult_known=octx.get("speed_mult_known", 1.0) or 1.0,
                opp_context_known=bool(octx.get("context_known", True)),
            )
            if obs:
                mb.observe_speed_bound(sp, **obs)
                stats["used"] += 1
            else:
                stats["skipped_bracket"] += 1

    if on_log:
        on_log(stats)
    return stats
