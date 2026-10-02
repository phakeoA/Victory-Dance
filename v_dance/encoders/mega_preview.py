"""Layout v20 — the MEGA PREVIEW block (mega audit gap 2, 2026-10-02; USER: "fix 1-5").

A mega resolves BEFORE anyone moves, and 55 % of opponent megas happen on turn 1 — but every v19 feature of a
mega-capable mon that has not mega'd yet describes its BASE forme: an opponent Golisopod about to become Bug/Steel
reads as Bug/Water (Flare Blitz 'neutral', it lands 4×; Thunderbolt '2×', it lands neutral), and our own
Golisopod's Leech Life band on its mega turn is priced with the pre-mega Atk. v20 adds, per mon (just before the
4 trailing slot flags), what the mon is about to become:

  MON part (MEGA_PREVIEW_MON_FEATURES)
    p_mega                    P(this mon mega-evolves): 1.0 for a KNOWN stone of its species (own side, an open
                              sheet, a revealed item), else its regulation's stone share (data/mega_stone_shares
                              .json); 0 once it is mega, its side has mega'd, it is transformed or fainted, or
                              its known item is not one of its stones
    mega types (NUM_TYPES)    the forme's typing, CONDITIONAL on mega-evolving (share-weighted over its stones)
    Δ base stats (6)          forme − current base stats, /255, conditional (Garchomp-Z: Spe +49)
    mega ability tags         the forme's fixed ability's mechanic tags, conditional (Charizardite Y → Drought)
  PER-MOVE part (NUM_MOVES × MEGA_PREVIEW_PER_MOVE), per enemy active e
    vs e's mega forme         signed type-eff · damage-band max · who-moves-first of THIS move into e AS ITS
                              MOST LIKELY MEGA (0 when e cannot mega)
    as OUR OWN mega forme     damage-band max · who-moves-first of THIS move used by this mon AS ITS MOST LIKELY
                              MEGA into e's current forme (0 when this mon cannot mega)

The per-move channels are computed by re-running each encoder's OWN move writer into a scratch row with the
mega-transformed defender profile / attacker context — so every damage rule (abilities, items, weather, terrain,
screens, crits, variable BP …) applies unchanged, and offline ↔ live parity rides on the existing writers.
The profile / attacker transforms below are shared by both encoders (parity by construction).

Stats of the forme: est(forme) = est(current) + Δbase per stat — exact at L50 up to the nature factor
(stat = base + IV/2 + EV/8 + 5 before nature); HP never changes on a mega.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from v_dance.encoders.encoder_layout import (
    MEGA_PREVIEW_MON_FEATURES, NUM_TYPES, _TYPE_IDX,
)

_REPO = Path(__file__).resolve().parents[2]
TABLE_PATH = _REPO / "data" / "mega_stone_shares.json"
_STATS = ("hp", "atk", "def", "spa", "spd", "spe")


def _canon(s) -> str:
    return "".join(c for c in str(s or "").lower() if c.isalnum())


# ── the frozen stone-share table ─────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _table() -> dict:
    try:
        return json.loads(TABLE_PATH.read_text(encoding="utf-8")).get("formats", {})
    except (OSError, ValueError):
        return {}


def _fingerprint() -> str:
    try:
        return hashlib.sha1(TABLE_PATH.read_bytes()).hexdigest()[:12]
    except OSError:
        return "none"


TABLE_FINGERPRINT = _fingerprint()      # part of the encoded-cache key (training/encoded_cache.py)


def _reg(fmt: Optional[str]) -> Optional[str]:
    from v_dance.formats import default_format, reg_token
    tok = reg_token(fmt or "")
    if not tok and fmt:                  # display names: '[Gen 9 Champions] VGC 2026 Reg M-C' → 'regmc'
        tok = reg_token(_canon(fmt))
    return tok or reg_token(default_format())


def stone_shares(species: Optional[str], fmt: Optional[str] = None) -> Dict[str, float]:
    """{stone id: share} for ``species`` in ``fmt``'s regulation (falls back to the dex base species)."""
    t = _table().get(_reg(fmt) or "", {})
    sid = _canon(species)
    if sid in t:
        return t[sid]
    base = _dex_base_id(sid)
    return t.get(base, {}) if base else {}


# ── dex helpers (project pokedex — the same one the offline encoder reads) ─────────────────────
@lru_cache(maxsize=4096)
def _dex_base_id(sid: str) -> Optional[str]:
    from v_dance.dex.pokedex import get_pokedex
    dex = get_pokedex()
    e = dex.entry(sid) if dex else None
    return _canon(e.get("baseSpecies")) if e and e.get("baseSpecies") else None


def mega_reachable(sid: str, forme_entry: dict) -> bool:
    """Can ``sid`` (the CURRENT, pre-mega species id) mega into this forme? The dex's ``battleOnly`` names the
    pre-mega forme when it is not the base (Floette-ETERNAL, Meowstic-F, Tatsugiri-Droopy, Magearna-Original);
    otherwise only the base species itself megas — Raichu-Alola / Slowbro-Galar never become Raichu-Mega /
    Slowbro-Mega (review 10-02)."""
    bo = forme_entry.get("battleOnly")
    if isinstance(bo, (list, tuple)):
        return any(_canon(b) == sid for b in bo)
    return _canon(bo or forme_entry.get("baseSpecies")) == sid


@lru_cache(maxsize=4096)
def _forme_for_stone(sid: str, stone: str) -> Optional[str]:
    """Dex display name of the mega forme ``stone`` makes ``sid`` into (one forme per stone and species:
    Meowsticite on 'meowsticf' → Meowstic-F-Mega, on 'meowstic' → Meowstic-M-Mega)."""
    from v_dance.dex.pokedex import get_pokedex
    dex = get_pokedex()
    if not dex or not sid or not stone:
        return None
    cands = []
    for fm in dex.mega_formes_for(sid):
        fe = dex.entry(fm["forme"]) or {}
        if _canon(fe.get("requiredItem")) == stone and mega_reachable(sid, fe):
            cands.append((fm["forme"], fe))
    return sorted(cands, key=lambda c: _canon(c[0]))[0][0] if cands else None


@lru_cache(maxsize=4096)
def species_stones(sid: str) -> Tuple[str, ...]:
    """The stones that make a mega forme reachable FROM ``sid`` (none for Raichu-Alola, Floette)."""
    from v_dance.dex.pokedex import get_pokedex
    dex = get_pokedex()
    out: List[str] = []
    for fm in (dex.mega_formes_for(sid) if dex else []):
        fe = dex.entry(fm["forme"]) or {}
        st = _canon(fe.get("requiredItem"))
        if st and st not in out and mega_reachable(sid, fe):
            out.append(st)
    return tuple(out)


@lru_cache(maxsize=4096)
def forme_view(current_species: str, forme: str) -> Optional[dict]:
    """{types, ability (fixed, canon id or None), delta {stat: Δbase}, weight} of ``forme`` vs the current one."""
    from v_dance.dex.pokedex import get_pokedex
    from v_dance.parser.belief_state import dex_base_stats
    from v_dance.encoders import damage_mechanics as _DMG
    dex = get_pokedex()
    fe = dex.entry(forme) if dex else None
    if not fe:
        return None
    b0 = dex_base_stats(current_species) or {}
    b1 = dex_base_stats(forme) or {}
    abilities = [a for a in (fe.get("abilities") or {}).values() if a]
    return {"forme": forme,
            "types": tuple(_canon_type(t) for t in (fe.get("types") or [])),
            "ability": _canon(abilities[0]) if len(abilities) == 1 else None,
            "delta": {k: float((b1.get(k) or 0) - (b0.get(k) or 0)) for k in _STATS},
            "weight": _DMG.species_weight(_canon(forme))}


def _canon_type(t) -> str:
    return "".join(c for c in str(t or "").upper() if c.isalnum() or c == "_")


# ── who can mega, into what ───────────────────────────────────────────────────────────────────────
def offline_known_item(mon: dict) -> Optional[str]:
    """The item a parsed (offline) mon is KNOWN to hold: '' = known itemless (consumed / Knocked Off, an exact or
    sheet-itemless mon), the id when revealed, None = unknown. Live twin: poke-env ``mon.item`` ('' / an id /
    'unknown_item'). A mega stone can be neither consumed nor knocked off, so '' rules a mega out."""
    if "preview_item" in mon:              # live: OUR true item stamped on the reconstruction (preview-only key)
        return mon["preview_item"]
    if mon.get("item_consumed"):
        return ""
    ki = mon.get("known_item")
    if ki:
        return ki
    if mon.get("exact") or mon.get("sheet_itemless"):
        return ""
    return None


def live_known_item(mon) -> Optional[str]:
    """poke-env twin of ``offline_known_item``."""
    it = getattr(mon, "item", None)
    if it is None or it == "unknown_item":
        return None
    return it



def mega_options(species: Optional[str], known_item: Optional[str], fmt: Optional[str] = None
                 ) -> List[Tuple[str, float]]:
    """[(forme display name, p)] this mon may mega into. A KNOWN item decides (its stone → p=1; any other known
    item, '' = itemless → none); an unknown item (None / 'unknown_item') falls back to the stone shares."""
    sid = _canon(species)
    if not sid or not species_stones(sid):
        return []
    ki = _canon(known_item) if known_item not in (None, "unknown_item") else None
    if ki is not None:
        if ki in species_stones(sid):
            f = _forme_for_stone(sid, ki)
            return [(f, 1.0)] if f else []
        return []
    out = []
    for st, p in stone_shares(sid, fmt).items():
        f = _forme_for_stone(sid, st)
        if f and p > 0:
            out.append((f, float(p)))
    tot = sum(p for _f, p in out)
    if tot > 1.0:
        out = [(f, p / tot) for f, p in out]
    return out


def best_forme(options: Sequence[Tuple[str, float]]) -> Optional[str]:
    """The most likely forme (ties → alphabetical, deterministic)."""
    return min(options, key=lambda fp: (-fp[1], fp[0]))[0] if options else None


# ── the MON part of the block ─────────────────────────────────────────────────────────────────────
def write_mon_part(vec: np.ndarray, i: int, current_species: str, options) -> None:
    """Write MEGA_PREVIEW_MON_FEATURES floats at ``vec[i:]`` (zeros when ``options`` is empty)."""
    from v_dance.encoders.mechanic_tags import ability_tag_indices, NUM_ABILITY_TAGS
    p_tot = float(sum(p for _f, p in options))
    if p_tot <= 0.0:
        return
    vec[i] = min(p_tot, 1.0)
    j = i + 1
    for forme, p in options:
        v = forme_view(_canon(current_species), forme)
        if v is None:
            continue
        w = p / p_tot                                    # conditional on mega-evolving
        for t in v["types"]:
            if t in _TYPE_IDX:
                vec[j + _TYPE_IDX[t]] += w
        for k, s in enumerate(_STATS):
            vec[j + NUM_TYPES + k] += w * float(np.clip(v["delta"][s] / 255.0, -1.0, 1.0))
        if v["ability"]:
            for idx in ability_tag_indices(v["ability"]):
                vec[j + NUM_TYPES + 6 + idx] += w
    assert NUM_TYPES + 6 + NUM_ABILITY_TAGS + 1 == MEGA_PREVIEW_MON_FEATURES


# ── profile / attacker-context transforms (shared by BOTH encoders) ───────────────────────────────
# What a mega forme sets on the field the moment it mega-evolves (before anyone moves): Charizardite Y → sun,
# Tyranitarite → sand, Abomasite → snow, Raichunite X → Electric Terrain … (tokens = both encoders' field_mods).
_ABILITY_WEATHER = {"drought": "SUNNYDAY", "drizzle": "RAINDANCE", "sandstream": "SANDSTORM",
                    "snowwarning": "SNOWSCAPE"}
_ABILITY_TERRAIN = {"electricsurge": "ELECTRIC_TERRAIN", "grassysurge": "GRASSY_TERRAIN",
                    "psychicsurge": "PSYCHIC_TERRAIN", "mistysurge": "MISTY_TERRAIN"}
_AURA_ABILITIES = frozenset({"fairyaura", "darkaura", "aurabreak"})


def forme_field(ability: Optional[str], field_mods) -> tuple:
    """(weather, terrain) once a mon with ``ability`` has mega-evolved (its setter ability fires on the mega)."""
    w, t = (field_mods or (None, None))
    return _ABILITY_WEATHER.get(ability or "", w), _ABILITY_TERRAIN.get(ability or "", t)


def speed_parts_speed(parts, ability: Optional[str], weather) -> float:
    """The in-battle speed from ``speed_parts`` = (est spe, spe stage, status token, own-side tailwind) with
    ``ability`` and ``weather`` — the offline ``_effective_speed`` / live ``_live_effective_speed`` formula WITHOUT
    the held-item and paradox terms: a mega holds its STONE (no Choice Scarf) and a mega ability is never
    Protosynthesis / Quark Drive. 0 when the speed is unknown (the who-moves-first channel then reads 0)."""
    from v_dance.encoders.battle_mechanics import _WEATHER_SPEED_ABILITY
    if not parts:
        return 0.0
    spe, stage, status, tailwind = parts
    if not spe:
        return 0.0
    s = float(spe)
    stage = int(stage or 0)
    s *= ((2 + stage) / 2.0) if stage >= 0 else (2.0 / (2 - stage))
    st = str(status or "").upper()
    if ability == "quickfeet" and st:
        s *= 1.5
    elif st == "PAR":
        s *= 0.5
    if tailwind:
        s *= 2.0
    if weather and weather in _WEATHER_SPEED_ABILITY.get(ability or "", ()):
        s *= 2.0
    return float(s)


def reweather_speed(eff, ability: Optional[str], old_weather, new_weather):
    """A resolved speed moved from one weather to another (only a weather-speed ability cares)."""
    from v_dance.encoders.battle_mechanics import _WEATHER_SPEED_ABILITY
    if eff is None or old_weather == new_weather:
        return eff
    ws = _WEATHER_SPEED_ABILITY.get(ability or "", ())
    f0 = 2.0 if (old_weather and old_weather in ws) else 1.0
    f1 = 2.0 if (new_weather and new_weather in ws) else 1.0
    return float(eff) * f1 / f0


def mega_defender_profile(d: Optional[dict], view: Optional[dict], field_weather=None) -> Optional[dict]:
    """A defender profile (offline ``_defender_profile`` / live ``_prof`` — the same dict schema) re-made for
    the mega forme: typing, ability, Def / SpD / Atk += Δbase, speed recomputed (its own setter's weather), and
    every HELD-ITEM effect dropped — a mega holds its stone, so a believed Choice Scarf / Assault Vest / resist
    berry / Bright Powder / Air Balloon / Iron Ball / Ring Target cannot apply (review 10-02)."""
    if d is None or view is None:
        return None
    from v_dance.encoders.battle_mechanics import _disguise_intact, _is_grounded
    m = dict(d)
    ab = view["ability"] or d.get("ability")
    m["types"] = list(view["types"])
    m["ability"] = ab
    m["no_guard"] = ab == "noguard"
    m["intact_disguise"] = _disguise_intact(ab, d.get("hp_frac") or 0.0)
    dl = view["delta"]
    for k in ("def", "spd", "atk"):
        if m.get(k) is not None:
            m[k] = float(m[k]) + dl[k]
    parts = d.get("speed_parts")
    if parts is not None:
        parts = (None if parts[0] is None else float(parts[0]) + dl["spe"],) + tuple(parts[1:])
        m["speed_parts"] = parts
        m["eff_speed"] = speed_parts_speed(parts, ab, forme_field(ab, (field_weather, None))[0])
        m["spe"] = parts[0]
    if view.get("weight"):
        m["weight"] = view["weight"]
    m.update(assault_vest=False, evio=False, resist_berry=None, evasion_item=False, ring_target=False)
    ga = d.get("ground_args")                                   # (item, levitating, force-grounded|gravity, ability)
    if ga is not None:
        levit, force = bool(ga[1]), bool(ga[2])
        m["grounded"] = _is_grounded(m["types"], ab, "", levit, force)
        m["ground_immune"] = levit and not force
        m["force_grounded"] = force
    return m


def mega_att_ctx(ac: Optional[dict], view: Optional[dict], field_weather=None) -> Optional[dict]:
    """The attacker context (both encoders' ``att_ctx``) for this mon AS its mega forme: stats += Δbase, speed
    recomputed, grounding re-derived, the forme's aura on the field, and every held-item effect dropped (Life
    Orb / Choice / Expert Belt / type-boost / Scope Lens / Loaded Dice / Wide Lens — it holds its stone)."""
    if ac is None or view is None:
        return None
    from v_dance.encoders.battle_mechanics import _is_grounded, field_auras
    c = dict(ac)
    ab = view["ability"] or ((ac.get("ground_args") or (None,) * 4)[3])
    dl = view["delta"]
    for k in ("atk", "spa", "def"):
        if c.get(k) is not None:
            c[k] = float(c[k]) + dl[k]
    parts = ac.get("speed_parts")
    if parts is not None:
        parts = (None if parts[0] is None else float(parts[0]) + dl["spe"],) + tuple(parts[1:])
        c["speed_parts"] = parts
        c["eff_speed"] = speed_parts_speed(parts, ab, forme_field(ab, (field_weather, None))[0])
        c["spe"] = parts[0]
    if view.get("weight"):
        c["weight"] = view["weight"]
    c.update(life_orb=False, choice=False, expert_belt=False, type_boost=None, scope_lens=False,
             loaded_dice=False, wide_lens=False)
    ga = ac.get("ground_args")
    if ga is not None:
        c["grounded"] = _is_grounded(list(view["types"]), ab, "", ga[1], ga[2])
    if ab in _AURA_ABILITIES:
        c["auras"] = field_auras(set(ac.get("auras") or ()) | {ab})
    if ab == "victorystar":
        c["victory_star"] = True
    return c


def write_move_part(vec: np.ndarray, pm: int, move_args, write_move, *, att_ctx, enemy_defenders,
                    enemy_mega, field_mods, mview, mac, ability_id, user_types) -> None:
    """The PER-MOVE part (NUM_MOVES × MEGA_PREVIEW_PER_MOVE at ``vec[pm:]``), shared by both encoders.

    ``write_move(scratch, arg, enemies, ctx, field_mods, ability_id, user_types)`` = the encoder's OWN move writer
    into a zeroed MOVE_FEATURES scratch row; ``move_args`` = [(slot, arg)]. Per enemy e that can mega: ONE write
    with e replaced by its mega profile (the PARTNER stays — Armor Tail / Dazzling still block priority), the field
    its mega sets, its aura, our speed re-weathered → [type-eff, band max, moves-first] into e. If THIS mon can
    mega: one write as its mega forme (its field, its aura; the enemies re-weathered) → [band max, moves-first]."""
    from v_dance.encoders.encoder_layout import (
        MEGA_PREVIEW_PER_MOVE, MOVE_FEATURES, _MV_BAND_MAX, _MV_FIRST, _MV_SIGNED,
    )
    from v_dance.encoders.battle_mechanics import field_auras
    if not move_args:
        return
    w0, t0 = field_mods or (None, None)
    my_ab = ability_id
    vs = []                                                     # (e, enemies, ctx, field) per mega-able enemy
    for e in range(2):
        em = (enemy_mega or [None, None])[e] if enemy_mega else None
        if em is None:
            continue
        enemies = list(enemy_defenders or [None, None]) + [None] * 2
        enemies = enemies[:2]
        enemies[e] = em
        fw, ft = forme_field(em.get("ability"), (w0, t0))
        ctx = dict(att_ctx or {})
        ctx["eff_speed"] = reweather_speed(ctx.get("eff_speed"), my_ab, w0, fw)
        if em.get("ability") in _AURA_ABILITIES:
            ctx["auras"] = field_auras(set(ctx.get("auras") or ()) | {em.get("ability")})
        vs.append((e, enemies, ctx, (fw, ft)))
    own = None
    if mac is not None and mview is not None:
        ow, ot = forme_field(mview["ability"], (w0, t0))
        enemies = []
        for d in list(enemy_defenders or [None, None])[:2]:
            if d is not None and ow != w0:
                d = dict(d)
                d["eff_speed"] = reweather_speed(d.get("eff_speed"), d.get("ability"), w0, ow)
            enemies.append(d)
        own = (enemies, (ow, ot))
    if not vs and own is None:
        return
    scr = np.zeros(MOVE_FEATURES, dtype=np.float32)
    for m_idx, arg in move_args:
        b = pm + m_idx * MEGA_PREVIEW_PER_MOVE
        for e, enemies, ctx, fm in vs:
            scr[:] = 0.0
            write_move(scr, arg, enemies, ctx, fm, my_ab, user_types)
            vec[b + 3 * e] = scr[_MV_SIGNED[e]]
            vec[b + 3 * e + 1] = scr[_MV_BAND_MAX[e]]
            vec[b + 3 * e + 2] = scr[_MV_FIRST[e]]
        if own is not None:
            scr[:] = 0.0
            write_move(scr, arg, own[0], mac, own[1], mview["ability"] or my_ab, tuple(mview["types"]))
            for e in range(2):
                vec[b + 6 + 2 * e] = scr[_MV_BAND_MAX[e]]
                vec[b + 6 + 2 * e + 1] = scr[_MV_FIRST[e]]


def side_has_megaed(mons) -> bool:
    """Offline: any mon of the side shows is_mega (a side megas once per game)."""
    return any(m and m.get("is_mega") for m in mons)
