"""Drill PRESSURE — opponent-only logit biases that make a self-play opponent play a situation hard (2026-10-02, USER:
"have it fight for terrain control and also for weather control").

The ``field`` pressure: when the current weather / terrain is not one this player's TEAM sets, add ``+bias`` to the
logit of "switch to my weather / terrain setter" for every such setter alive on the bench — on normal turns AND on
post-faint replacements. A bias is a NUDGE on the model's own logits (the adapt_rules mechanism), never a mask
edit: an illegal index stays illegal, and the model still decides. Without this the drill's opponents — the bot's own
family — never reclaim either (the 10-02 report: our side reclaims terrain 2.9 % of the time), so there would be no
fight to learn from.

Opponent-only by construction: ``SelfPlayVGCPlayer`` (the recording learner, and the 'latest' opponent PPO also
trains on) refuses the kwargs — a biased recorder would make PPO silently off-policy.
"""
from __future__ import annotations

import math
import re
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from v_dance.encoders.encoder_layout import ACTIONS_PER_SLOT, BENCH_SLOTS, SWITCH_OFFSET
from v_dance.encoders.live_state_encoder import own_bench_mons
from v_dance.eval.field_control_report import (ABILITY_WEATHER, ITEM_TERRAIN, ITEM_WEATHER, SWITCH_IN_TERRAIN,
                                               WEATHER_KIND)


def _id(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


def setter_kinds(mon, battle=None) -> Dict[str, str]:
    """{"weather" / "terrain": kind} this mon sets on switch-in (its ability), or as its MEGA (a mega stone whose
    forme sets it — Charizardite Y, Froslassite, Raichunite X …). The mega-stone path counts only while this side
    can still mega-evolve (one mega per battle — review F3: a Charizard-Y holder whose partner already mega'd can
    never set sun, so pushing it in is no field fight). An already-mega'd mon is covered by its (mega) ability."""
    out: Dict[str, str] = {}
    ab = _id(getattr(mon, "ability", None))
    it = _id(getattr(mon, "item", None))
    mega_left = battle is None or not bool(getattr(battle, "used_mega_evolve", False))
    if ab in ABILITY_WEATHER:
        out["weather"] = ABILITY_WEATHER[ab]
    elif mega_left and it in ITEM_WEATHER:
        out["weather"] = ITEM_WEATHER[it]
    if ab in SWITCH_IN_TERRAIN:
        out["terrain"] = SWITCH_IN_TERRAIN[ab]
    elif mega_left and it in ITEM_TERRAIN:
        out["terrain"] = ITEM_TERRAIN[it]
    return out


def _fieldable(battle) -> list:
    """The side's mons that can still matter: on the field now + alive on the bench (brought, not fainted). Review F5:
    the full 6-mon ``battle.team`` included the 2 left at home and fainted setters, so a kind the side can no longer
    field still read as 'ours' and suppressed the fight (Rillaboom never pushed in when Indeedee stayed home)."""
    out = [m for m in (getattr(battle, "active_pokemon", None) or []) if m is not None
           and not getattr(m, "fainted", False)]
    out += [m for m in own_bench_mons(battle) if not getattr(m, "fainted", False)]
    return out


def current_kinds(battle) -> Dict[str, Optional[str]]:
    """The weather / terrain KIND on the field now (None = clear; poke-env's Weather.UNKNOWN counts as clear)."""
    w = None
    for k in (getattr(battle, "weather", None) or {}):
        w = WEATHER_KIND.get(_id(getattr(k, "name", k)))
        break
    t = None
    for f in (getattr(battle, "fields", None) or {}):
        if getattr(f, "is_terrain", False):
            t = _id(getattr(f, "name", f)).replace("terrain", "") or None
            break
    return {"weather": w, "terrain": t}


def field_biases(battle, bias: float, *, replacement: bool = False, fire_on_empty: bool = True
                 ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Per-slot (ACTIONS_PER_SLOT,) float32 biases: +bias on 'switch to a bench setter' whose kind is not on the
    field (and the field is not already one of this TEAM's kinds). (None, None) when nothing applies."""
    try:
        bias = float(bias)
    except (TypeError, ValueError):
        return None, None
    if not math.isfinite(bias) or bias <= 0:
        return None, None
    team_kinds: Dict[str, set] = {"weather": set(), "terrain": set()}
    for m in _fieldable(battle):
        for d, k in setter_kinds(m, battle).items():
            team_kinds[d].add(k)
    if not team_kinds["weather"] and not team_kinds["terrain"]:
        return None, None
    cur = current_kinds(battle)
    arr = np.zeros(ACTIONS_PER_SLOT, dtype=np.float32)
    fired = False
    for i, mon in enumerate(list(own_bench_mons(battle))[:BENCH_SLOTS]):
        for d, k in setter_kinds(mon, battle).items():
            c = cur[d]
            if c is not None and c in team_kinds[d]:
                continue                                # the field is already ours (any of our setters' kinds)
            if c is None and not fire_on_empty:
                continue
            arr[SWITCH_OFFSET + i] = bias
            fired = True
    if not fired:
        return None, None
    if not replacement:
        return arr, arr.copy()
    force = list(getattr(battle, "force_switch", None) or [])
    return tuple((arr.copy() if (s < len(force) and force[s]) else None) for s in (0, 1))   # type: ignore[return-value]


BIASES: Dict[str, Callable] = {"field": field_biases}    # the names drills.Drill.pressure refers to


def pressure_biases(name: Optional[str], battle, bias: float, *, replacement: bool = False):
    fn = BIASES.get(name or "")
    if fn is None:
        return None, None
    return fn(battle, bias, replacement=replacement)


def add_bias(a, b):
    """None-safe sum of two per-slot bias arrays."""
    if a is None:
        return None if b is None else np.asarray(b, dtype=np.float32)
    if b is None:
        return np.asarray(a, dtype=np.float32)
    return np.asarray(a, dtype=np.float32) + np.asarray(b, dtype=np.float32)


def note_pressure(player, biases, masks, picks) -> None:
    """Count, per slot: ``fired`` = a biased action was LEGAL there; ``taken`` = the pick was a biased action. Kept in
    ``player._pressure_stats`` — NEVER in _source_counts (the model-driven guard counts those)."""
    st = getattr(player, "_pressure_stats", None)
    if st is None:
        st = player._pressure_stats = {"fired": 0, "taken": 0}
    for b, m, a in zip(biases, masks, picks):
        if b is None or m is None:
            continue
        idx = [i for i in range(len(b)) if b[i] > 0 and bool(m[i])]
        if not idx:
            continue
        st["fired"] += 1
        if a is not None and int(a) in idx:
            st["taken"] += 1
