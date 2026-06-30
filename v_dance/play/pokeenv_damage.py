"""
pokeenv_damage.py — poke-env per-candidate damage likelihood (Level C / A2 v2, BOTH directions).
================================================================================

For a damage event, narrow the opponent's hidden EV spread by asking poke-env's full Gen-9 calc — with
each candidate spread's stats INJECTED onto the opp mon and the opp's ITEM and ABILITY marginalised over
the belief — how much damage that hypothesis would produce, and comparing to the observed roll. Returns
a per-spread log-likelihood that :meth:`MatchBelief.observe_damage_loglik` multiplies into the spread
distribution. :func:`damage_loglik` handles BOTH:
  * ``direction='off'`` — the opp ATTACKED our mon → narrow its offence (Atk/SpA). The opp's offensive
    item (Choice Band/Specs ×1.5, Life Orb ×1.3, …) and ability (Pure Power, Adaptability, Tough Claws,
    Sheer Force, Technician, …) are CONFOUNDED with its offensive stat — a single multiplier can't
    separate them, so we marginalise.
  * ``direction='def'`` — our mon HIT the opp → narrow its bulk (Def/SpD/HP). Same confound on the
    DEFENSIVE side: a resist berry / Assault Vest / Eviolite, or Multiscale / Thick Fat / Filter / Fur
    Coat / Ice Scales. The denominator is the candidate's OWN max HP, so HP disambiguates too.

Why poke-env (not the offline ``_damage_band``): it applies every one of those item/ability effects
(``calculate_attack`` / ``calculate_defense`` / final mods) plus weather / screens / type / STAB that
``situational_mult=1.0`` ignored — confirmed in ``damage_calc_gen9.py``. (One poke-env quirk: a typo
makes Huge Power a no-op, so we substitute the identical Pure Power.)

poke-env's calc asserts both mons have numeric stats; the opp's are ``None`` (filled only from our own
``|request|``), so we INJECT the candidate stats (+ item + ability) onto the live opp mon, call calc,
and RESTORE in a ``finally`` (calc only READS the mon).  Design: ``docs/levelC_A2v2_offensive_design.md``.
This module is the ONLY poke-env-coupled piece of the belief feed; it degrades to ``None`` (→ the
analytic fallback) on any failure.
"""
from __future__ import annotations

import math
from typing import Optional

from v_dance.parser.belief_state import dex_base_stats, calc_full_stats
from v_dance.parser.match_belief import _spread_key
from v_dance.parser.vod_parser.pokedex import norm_species

try:  # poke-env optional (offline machines)
    from poke_env.calc import calculate_damage
    from poke_env.battle import Move
    _HAS_POKE_ENV = True
except Exception:  # pragma: no cover
    calculate_damage = None  # type: ignore
    Move = None  # type: ignore
    _HAS_POKE_ENV = False

# log-space σ for the per-spread likelihood: the damage roll (0.85–1.0) + situational slop only. Crit is
# KNOWN (the parser stamps |-crit|, folded as is_critical), poke-env handles mitigation, and the ITEM
# uncertainty is carried by the mixture (not σ) — so this is sharp.
_SIGMA = (0.05 ** 2 + 0.05 ** 2) ** 0.5            # ≈ 0.071
_LL_FLOOR = math.log(1e-3)                          # per-event log-likelihood floor (never zero the truth)
_STATS = ("hp", "atk", "def", "spa", "spd", "spe")


_TOP_ITEMS = 4         # marginalise over at most this many belief items (+ a "no item" residual)
_TOP_ABILITIES = 3     # …and this many belief abilities


def _hyps(dist, revealed, *, residual_none: bool, top_k: int) -> list[tuple[float, Optional[str]]]:
    """``[(weight, id_or_None)]`` to marginalise an item/ability over. A REVEALED value → that value
    alone (P=1, sharp). Else the belief distribution's top-``top_k`` entries (mapped to poke-env ids),
    renormalised; for ITEMS (``residual_none``) the leftover probability mass becomes a "no item"
    hypothesis. Falls back to ``[(1.0, None)]`` when there is nothing to marginalise."""
    if revealed:
        return [(1.0, norm_species(revealed))]
    hyps: list[tuple[float, Optional[str]]] = []
    mass = 0.0
    for e in (dist or [])[:top_k]:
        p, name = float(e.get("p") or 0.0), e.get("name")
        if p > 0 and name:
            hyps.append((p, norm_species(name)))
            mass += p
    if residual_none:
        hyps.append((max(1.0 - mass, 1e-6), None))      # residual mass → "no (relevant) item"
    if not hyps:
        return [(1.0, None)]
    tot = sum(w for w, _ in hyps) or 1.0
    return [(w / tot, x) for w, x in hyps]


def damage_loglik(
    battle, *, opp_mon, our_mon, direction: str, move_name: str,
    is_critical: bool, obs_frac: float, belief, opp_snapshot_mon: Optional[dict] = None,
    level: int = 50, sigma: float = _SIGMA,
) -> Optional[dict]:
    """Per-spread ``ln L`` that the OPPONENT mon produced the observed damage fraction, marginalised over
    the belief's ITEM and ABILITY hypotheses via poke-env's full Gen-9 calc with each candidate spread's
    stats injected onto the opp mon.

    ``direction='off'`` → the opp is the ATTACKER and OUR mon the (known) defender: narrows the opp's
    offence (Atk/SpA), ``obs_frac`` is of OUR mon's max HP. ``direction='def'`` → the opp is the DEFENDER
    and OUR mon the (known) attacker: narrows the opp's bulk (Def/SpD/HP), ``obs_frac`` is of the opp's
    max HP, which is PER-CANDIDATE (the candidate spread's HP stat). Either way only the OPP side is
    injected — our side is already populated from the team sheet; the opp mon is restored exactly
    (try/finally; calc only reads). Returns ``{_spread_key: loglik}`` or ``None`` (→ analytic fallback)."""
    if (not _HAS_POKE_ENV or calculate_damage is None or belief is None or not opp_mon or not our_mon
            or direction not in ("off", "def") or not obs_frac or obs_frac <= 0):
        return None
    # Resolve the SAME lookup / stats forme MatchBelief.enrich_mon→block_for uses, so the spread keys we
    # emit MATCH the consumer's block spreads (a megaevolved opp keys off the MEGA forme). (audit 2026-06-30)
    snap = opp_snapshot_mon or {}
    sp = (snap.get("base_species") or snap.get("species")
          or getattr(opp_mon, "base_species", None) or getattr(opp_mon, "species", None))
    forme = snap.get("species")
    lookup = sp if (sp and belief.known(sp)) else forme
    if snap.get("is_mega") and forme and belief.known(forme):
        lookup = forme
    stats_species = forme or sp
    block = belief.belief_block(lookup, top_k=5) if lookup and belief.known(lookup) else None
    spreads = (block or {}).get("spreads") or []
    base = dex_base_stats(stats_species) or (dex_base_stats(sp) if sp else None)
    our_max_hp = getattr(our_mon, "max_hp", 0) or 0
    if not block or len(spreads) < 2 or not base or (direction == "off" and our_max_hp <= 0):
        return None
    try:
        att, defn = (opp_mon, our_mon) if direction == "off" else (our_mon, opp_mon)
        att_role = battle.opponent_role if direction == "off" else battle.player_role
        def_role = battle.player_role if direction == "off" else battle.opponent_role
        att_id, def_id = att.identifier(att_role), defn.identifier(def_role)
        move = Move(norm_species(move_name), 9)
    except Exception:
        return None

    item_hyps = _hyps(block.get("items"), snap.get("known_item"),
                      residual_none=True, top_k=_TOP_ITEMS)
    ability_hyps = _hyps(block.get("abilities"), snap.get("known_ability"),
                         residual_none=False, top_k=_TOP_ABILITIES)

    saved_stats = getattr(opp_mon, "_stats", None)        # restore the EXACT live objects/values
    saved_item = getattr(opp_mon, "_item", None)
    saved_ability = getattr(opp_mon, "_ability", None)
    log_obs = math.log(obs_frac)
    out: dict = {}
    try:
        for s in spreads:
            full = calc_full_stats(base, s["evs_actual"], s["nature"], level=level)
            if not full or not all(isinstance(full.get(k), (int, float)) for k in _STATS):
                continue
            # defensive denom is THIS candidate's max HP (so HP/Def/SpD all disambiguate); offensive denom
            # is our known max HP.
            denom = our_max_hp if direction == "off" else full.get("hp")
            if not denom or denom <= 0:
                continue
            opp_mon._stats = {k: full.get(k) for k in _STATS}
            like = 0.0
            for w_it, item in item_hyps:
                opp_mon._item = item
                for w_ab, ab in ability_hyps:
                    # poke-env has a typo ('hugemount' not 'hugepower') so Huge Power is never applied;
                    # Pure Power is identical (×2 physical Atk) → substitute it so the hypothesis counts.
                    opp_mon._ability = "purepower" if ab == "hugepower" else ab
                    try:
                        lo, hi = calculate_damage(att_id, def_id, move, battle,
                                                  is_critical=bool(is_critical))
                    except Exception:
                        continue
                    frac = ((float(lo) + float(hi)) / 2.0) / denom
                    if frac > 0:
                        z = (log_obs - math.log(frac)) / sigma
                        like += w_it * w_ab * math.exp(-0.5 * z * z)
            out[_spread_key(s)] = max(math.log(like), _LL_FLOOR) if like > 0 else _LL_FLOOR
    except Exception:
        return None
    finally:
        if saved_stats is not None:
            opp_mon._stats = saved_stats
        opp_mon._item = saved_item
        opp_mon._ability = saved_ability
    return out or None


def offensive_loglik(battle, opp_mon, our_mon, move_name: str, **kw):
    """Backwards-compatible offensive wrapper → :func:`damage_loglik` with ``direction='off'``."""
    return damage_loglik(battle, opp_mon=opp_mon, our_mon=our_mon, direction="off",
                         move_name=move_name, **kw)
