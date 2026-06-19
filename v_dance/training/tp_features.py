"""15b-feat.1/.spread — shared Team-Preview feature extractor + curated mechanic-tag tables.

ONE source of truth for the per-mon feature vector, imported by BOTH the training dataset
(offline, from replays) and model_io._pack_side (serve, from live battle). That shared call —
not just a shared shape — is what makes train/serve parity real.

THE DESIGN (SBDA, sec 14 / the design panel):
  * OWN and OPP share a SYMMETRIC BASE block from (species, belief) ONLY, so a Type-B training
    example (own = species-only, like the opponent) is byte-identical to a belief-only serve:
    own_mon_features(s, b, known=None) == opp_mon_features(s, b).
  * Sharp OWN item/ability/moves ride an OVERLAY block gated by a has_own_detail bit.
  * Features are MECHANIC TAGS keyed on the MECHANIC, not raw ability identity (a raw-ability
    multihot was already trialled — teampreview_dataset "#A" — with NO lift). They pay off only
    CONSUMED BY the self-attention block (15b-arch.1): the two halves of a synergy must INTERACT.

PAIRWISE SYNERGY FAMILIES (each = an enabler tag <-> a beneficiary tag the attention layer pairs):
  * weather setter <-> weather abuser     (Sand Stream <-> Sand Rush)
  * terrain setter <-> terrain abuser     (Psychic Terrain <-> Expanding Force)
  * spread move    <-> ally immunity       (Earthquake <-> Flying/Levitate ally; Discharge <-> Ground ally)
  * role tags (redirect / speed_control / trick_room / tailwind / screens / fake_out / priority / intimidate)
  Long-tail synergies (Contrary<->ally-debuff, Beads of Ruin, ...) are left to the `teammates` prior +
  outcome training; the tags here are the high-value, mechanically-clean, EMERGING-tech-robust subset.

DATA vs MECHANICS (robustness — see mechanic_coverage.py): the pikalytics USAGE data is read LIVE from a
BeliefState and auto-updates on a swap. Only the ability/move -> mechanic-TAG tables are in code (reg-
INDEPENDENT game facts). A dex-grounded guard fails loud if a NEW reg's mechanic-bearer is unmapped.
Channel ORDER is asserted at import so a reorder can't silently corrupt a trained net.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from v_dance.parser.vod_parser.pokedex import norm_species, get_pokedex  # noqa: F401
from v_dance.training.teampreview_dataset import (
    mon_dex_features, MON_FEAT_DIM, NUM_TYPES, _TYPE_IDX, _canon_type,
)

FEATURE_SCHEMA_VERSION = "tpfeat-v6"   # v6: defensive type-effectiveness profile (complementarity)

# ── mechanic axes (ORDER IS LOAD-BEARING — asserted at import) ─────────────────
WEATHERS = ("sand", "rain", "sun", "snow")
TERRAINS = ("electric", "grassy", "psychic", "misty")
ROLE_TAGS = ("trick_room", "tailwind", "redirect", "fake_out", "screens",
             "speed_control", "priority", "intimidate")
GIMMICK_KINDS = ("none", "mega", "tera", "dynamax")
ORDER_FLAGS = ("illusion", "imposter")   # abilities where the team-ORDER/slot choice matters specially

# ── curated ability/move -> mechanic tag tables (grounded in the pinned M-A data) ─
SETTER_ABILITY = {"Sand Stream": "sand", "Drizzle": "rain", "Drought": "sun",
                  "Snow Warning": "snow", "Sand Spit": "sand"}
SETTER_MOVE = {"Sandstorm": "sand", "Rain Dance": "rain", "Sunny Day": "sun",
               "Snowscape": "snow", "Chilly Reception": "snow"}
ABUSER_ABILITY = {"Sand Rush": "sand", "Sand Force": "sand", "Swift Swim": "rain",
                  "Chlorophyll": "sun", "Solar Power": "sun", "Slush Rush": "snow"}
TERRAIN_SETTER_ABILITY = {"Electric Surge": "electric", "Grassy Surge": "grassy",
                          "Psychic Surge": "psychic", "Misty Surge": "misty", "Hadron Engine": "electric"}
TERRAIN_SETTER_MOVE = {"Electric Terrain": "electric", "Grassy Terrain": "grassy",
                       "Psychic Terrain": "psychic", "Misty Terrain": "misty"}
TERRAIN_ABUSER_ABILITY = {"Surge Surfer": "electric", "Quark Drive": "electric", "Grass Pelt": "grassy"}
TERRAIN_ABUSER_MOVE = {"Grassy Glide": "grassy", "Expanding Force": "psychic",
                       "Rising Voltage": "electric", "Psyblade": "electric", "Misty Explosion": "misty"}
ROLE_MOVE = {
    "trick_room": {"Trick Room"},
    "tailwind": {"Tailwind"},
    "redirect": {"Follow Me", "Rage Powder"},
    "fake_out": {"Fake Out"},
    "screens": {"Reflect", "Light Screen", "Aurora Veil"},
    "speed_control": {"Icy Wind", "Electroweb", "Thunder Wave", "Glaciate", "Bulldoze"},
    "priority": {"Extreme Speed", "Aqua Jet", "Bullet Punch", "Sucker Punch", "Grassy Glide",
                 "Ice Shard", "Shadow Sneak", "Quick Attack", "Mach Punch", "Jet Punch",
                 "Thunderclap", "Vacuum Wave", "Water Shuriken", "Feint", "Accelerock"},
}
ROLE_ABILITY = {"intimidate": {"Intimidate"}}

# 15b-feat.spread: ally-hitting (allAdjacent) spread moves -> type (the ones where ally-immunity is a
# synergy). Grounded in the dex `target: "allAdjacent"` set present in the M-A data; the guard catches new ones.
SPREAD_MOVE = {
    "Earthquake": "Ground", "Bulldoze": "Ground", "Magnitude": "Ground",
    "Discharge": "Electric", "Parabolic Charge": "Electric",
    "Surf": "Water", "Lava Plume": "Fire", "Petal Blizzard": "Grass", "Sludge Wave": "Poison",
    "Boomburst": "Normal", "Explosion": "Normal", "Self-Destruct": "Normal", "Brutal Swing": "Dark",
}
# Type-chart 0x immunities (defender type -> attacking types it is immune to) — reg-independent game rule.
TYPE_IMMUNITY = {
    "Flying": {"Ground"}, "Ground": {"Electric"}, "Ghost": {"Normal", "Fighting"},
    "Normal": {"Ghost"}, "Steel": {"Poison"}, "Dark": {"Psychic"}, "Fairy": {"Dragon"},
}
# Immunity ABILITIES (ability -> attacking type it nullifies on the holder).
IMMUNITY_ABILITY = {
    "Levitate": "Ground", "Earth Eater": "Ground",
    "Flash Fire": "Fire", "Well-Baked Body": "Fire",
    "Water Absorb": "Water", "Storm Drain": "Water", "Dry Skin": "Water",
    "Volt Absorb": "Electric", "Lightning Rod": "Electric", "Motor Drive": "Electric",
    "Sap Sipper": "Grass",
}
# 15b-feat.stat: a stat-change REVERSER (Contrary) pairs with an ally that applies a stat DROP, flipping
# it into a boost (Prankster Charm on a Contrary ally -> +2 Atk). Emerging M-A tech — a case where a
# mechanic tag BEATS the teammates co-occurrence prior (the combo isn't established in usage data yet).
STAT_REVERSER_ABILITY = {"Contrary"}
ALLY_DEBUFF_MOVE = {
    "Charm", "Fake Tears", "Baby-Doll Eyes", "Tearful Look", "Parting Shot", "Memento",
    "Feather Dance", "Screech", "Metal Sound", "Noble Roar", "Snarl", "Captivate",
    "Eerie Impulse", "Play Nice", "Confide", "Tickle",
}
# 15b-feat.order: abilities where the SLOT/ORDER you submit matters specially — Illusion (Zoroark /
# Hisuian Zoroark disguises as the LAST brought mon) and Imposter (Ditto transforms into who it faces;
# lead = instant copy vs bench = controlled). A prior FLAG that order is high-stakes for this mon; the
# optimal order is still learned from outcomes (a richer order-decode is a separate, larger change).
ORDER_ABILITY = {"Illusion": "illusion", "Imposter": "imposter"}

_W, _TR, _R, _G, _T = len(WEATHERS), len(TERRAINS), len(ROLE_TAGS), len(GIMMICK_KINDS), NUM_TYPES
_O = len(ORDER_FLAGS)

# ── fixed segment offsets ─────────────────────────────────────────────────────
OFF_DEX = 0                                   # mon_dex_features: types + base stats (46)
OFF_WSETS = OFF_DEX + MON_FEAT_DIM            # weather sets (4)
OFF_WABUSE = OFF_WSETS + _W                   # weather abuses (4)
OFF_TSETS = OFF_WABUSE + _W                   # terrain sets (4)
OFF_TABUSE = OFF_TSETS + _TR                  # terrain abuses (4)
OFF_ROLES = OFF_TABUSE + _TR                  # role tags (8)
OFF_SPREAD = OFF_ROLES + _R                   # spread_<type>: carries an ally-hitting spread move (NUM_TYPES)
OFF_IMMUNE = OFF_SPREAD + _T                  # immune_<type>: typing 0x + immunity-ability (NUM_TYPES)
OFF_REVERSER = OFF_IMMUNE + _T                # stat_reverser: benefits from stat drops (Contrary) (1)
OFF_DEBUFF = OFF_REVERSER + 1                 # ally_debuff: carries an ally-targetable stat-drop move (1)
OFF_ORDER = OFF_DEBUFF + 1                    # order-sensitivity flags illusion/imposter (len ORDER_FLAGS)
OFF_DEFEFF = OFF_ORDER + _O                   # def_eff[NUM_TYPES]: signed typing defensive effectiveness
OFF_HASDATA = OFF_DEFEFF + _T                 # belief.known(species) (1)
OFF_USAGE = OFF_HASDATA + 1                   # usage_pct/100 (1)
BASE_DIM = OFF_USAGE + 1                       # ── end of the SYMMETRIC base block
OFF_GK = BASE_DIM                              # gimmick_kind one-hot (4) — reserved, base prior
OFF_TERA = OFF_GK + _G                        # tera_type one-hot (NUM_TYPES) — reserved, zero in M-A
GIMMICK_END = OFF_TERA + _T
OFF_OWNBIT = GIMMICK_END                       # has_own_detail (1) — OVERLAY starts (own only)
OFF_KWSETS = OFF_OWNBIT + 1
OFF_KWABUSE = OFF_KWSETS + _W
OFF_KTSETS = OFF_KWABUSE + _W
OFF_KTABUSE = OFF_KTSETS + _TR
OFF_KROLES = OFF_KTABUSE + _TR
OFF_KSPREAD = OFF_KROLES + _R                  # hard known spread_<type> (NUM_TYPES)
OFF_KIMMUNE = OFF_KSPREAD + _T                 # hard known ability-immunity (NUM_TYPES)
OFF_KREVERSER = OFF_KIMMUNE + _T               # hard known stat_reverser (1)
OFF_KDEBUFF = OFF_KREVERSER + 1                # hard known ally_debuff (1)
OFF_KORDER = OFF_KDEBUFF + 1                   # hard known order flags (len ORDER_FLAGS)
OFF_KGK = OFF_KORDER + _O
OFF_KTERA = OFF_KGK + _G
FEAT_DIM = OFF_KTERA + _T


@dataclass
class OwnKnown:
    """The sharp OWN-side build, known at serve / self-play / Type-A only."""
    ability: Optional[str] = None
    moves: Sequence[str] = ()
    item: Optional[str] = None
    tera: Optional[str] = None
    will_mega: bool = False


# ── pure tag functions (no belief; unit-tested in isolation) ──────────────────────
def _idx_map(name_to_axis, axes):
    return {nm: axes.index(ax) for nm, ax in name_to_axis.items() if ax in axes}


_WSET_A = _idx_map(SETTER_ABILITY, WEATHERS)
_WABU_A = _idx_map(ABUSER_ABILITY, WEATHERS)
_WSET_M = _idx_map(SETTER_MOVE, WEATHERS)
_TSET_A = _idx_map(TERRAIN_SETTER_ABILITY, TERRAINS)
_TABU_A = _idx_map(TERRAIN_ABUSER_ABILITY, TERRAINS)
_TSET_M = _idx_map(TERRAIN_SETTER_MOVE, TERRAINS)
_TABU_M = _idx_map(TERRAIN_ABUSER_MOVE, TERRAINS)
# spread move / immunity index maps (over the TYPE axis)
_SPREAD_I = {nm: _TYPE_IDX[_canon_type(t)] for nm, t in SPREAD_MOVE.items() if _canon_type(t) in _TYPE_IDX}
_IMMABIL_I = {nm: _TYPE_IDX[_canon_type(t)] for nm, t in IMMUNITY_ABILITY.items() if _canon_type(t) in _TYPE_IDX}
_TYPE_IMMUNITY_C = {_canon_type(k): {_canon_type(a) for a in v} for k, v in TYPE_IMMUNITY.items()}


def _field_tags(ability_probs, move_probs, set_a, abu_a, set_m, abu_m, n):
    sets = np.zeros(n, np.float32)
    abuse = np.zeros(n, np.float32)
    for a in ability_probs:
        nm, p = a["name"], float(a["p"])
        if nm in set_a:
            sets[set_a[nm]] += p
        if nm in abu_a:
            abuse[abu_a[nm]] += p
    for m in move_probs:
        nm, p = m["name"], float(m["p"])
        if nm in set_m:
            sets[set_m[nm]] += p
        if nm in abu_m:
            abuse[abu_m[nm]] += p
    return np.clip(sets, 0.0, 1.0), np.clip(abuse, 0.0, 1.0)


def weather_tags(ability_probs, move_probs):
    return _field_tags(ability_probs, move_probs, _WSET_A, _WABU_A, _WSET_M, {}, _W)


def terrain_tags(ability_probs, move_probs):
    return _field_tags(ability_probs, move_probs, _TSET_A, _TABU_A, _TSET_M, _TABU_M, _TR)


def role_tags(ability_probs, move_probs):
    r = np.zeros(_R, np.float32)
    mp = {m["name"]: float(m["p"]) for m in move_probs}
    ap = {a["name"]: float(a["p"]) for a in ability_probs}
    for i, tag in enumerate(ROLE_TAGS):
        if tag in ROLE_MOVE:
            r[i] += sum(mp.get(mv, 0.0) for mv in ROLE_MOVE[tag])
        if tag in ROLE_ABILITY:
            r[i] += sum(ap.get(ab, 0.0) for ab in ROLE_ABILITY[tag])
    return np.clip(r, 0.0, 1.0)


def spread_tags(move_probs):
    """spread[NUM_TYPES]: carries an ally-hitting (allAdjacent) spread move of this type."""
    v = np.zeros(_T, np.float32)
    for m in move_probs:
        i = _SPREAD_I.get(m["name"])
        if i is not None:
            v[i] += float(m["p"])
    return np.clip(v, 0.0, 1.0)


def _species_types(species):
    dex = get_pokedex()
    e = dex.entry(species) if dex else None
    return [_canon_type(t) for t in ((e or {}).get("types") or [])]


def _typing_immune(species):
    v = np.zeros(_T, np.float32)
    for tc in _species_types(species):
        for atk in _TYPE_IMMUNITY_C.get(tc, ()):
            idx = _TYPE_IDX.get(atk)
            if idx is not None:
                v[idx] = 1.0
    return v


def _ability_immune(ability_probs):
    v = np.zeros(_T, np.float32)
    for a in ability_probs:
        i = _IMMABIL_I.get(a["name"])
        if i is not None:
            v[i] = max(v[i], float(a["p"]))
    return v


def immune_tags(species, ability_probs):
    """immune[NUM_TYPES]: immune to this attacking type — typing 0x (public) + immunity-ability prior."""
    return np.clip(_typing_immune(species) + _ability_immune(ability_probs), 0.0, 1.0)


_TYPE_CHART = None      # {DEFENDING_TYPE: {ATTACKING_TYPE: multiplier}}, lazy-loaded from poke-env


def _type_chart():
    global _TYPE_CHART
    if _TYPE_CHART is None:
        from poke_env.data import GenData
        _TYPE_CHART = GenData.from_gen(9).type_chart
    return _TYPE_CHART


def def_eff_profile(species):
    """def_eff[NUM_TYPES]: SIGNED defensive effectiveness vs each attacking type — log2(multiplier)/2
    clamped to [-1,1] (negative = resists incoming, positive = weak to it; immune -> -1). Pure typing
    (public, both sides) — the substrate for type complementarity (A covers B's weaknesses) and for
    "does my mon resist their threats". 15b-feat.defense."""
    v = np.zeros(_T, np.float32)
    types = _species_types(species)            # canonical (uppercase), matches poke-env chart keys
    if not types:
        return v
    tc = _type_chart()
    atk_types = {a for d in tc.values() for a in d}
    for atk in atk_types:
        idx = _TYPE_IDX.get(_canon_type(atk))
        if idx is None:
            continue
        mult = 1.0
        for t in types:
            mult *= tc.get(t, {}).get(atk, 1.0)
        v[idx] = -1.0 if mult <= 0 else float(np.clip(math.log2(mult) / 2.0, -1.0, 1.0))
    return v


def reverser_tag(ability_probs):
    """stat_reverser scalar: benefits from stat drops (Contrary) — the beneficiary half."""
    return float(min(1.0, sum(float(a["p"]) for a in ability_probs if a["name"] in STAT_REVERSER_ABILITY)))


def ally_debuff_tag(move_probs):
    """ally_debuff scalar: carries an ally-targetable stat-lowering move — the enabler half."""
    return float(min(1.0, sum(float(m["p"]) for m in move_probs if m["name"] in ALLY_DEBUFF_MOVE)))


_ORDER_I = {nm: ORDER_FLAGS.index(f) for nm, f in ORDER_ABILITY.items()}


def order_tags(ability_probs):
    """order[len(ORDER_FLAGS)]: ability makes the slot/order choice high-stakes (Illusion / Imposter)."""
    v = np.zeros(_O, np.float32)
    for a in ability_probs:
        i = _ORDER_I.get(a["name"])
        if i is not None:
            v[i] += float(a["p"])
    return np.clip(v, 0.0, 1.0)


def teammate_affinity_matrix(our_species, belief, n=6):
    """(n, n) SYMMETRIC co-occurrence affinity in [0,1] from Pikalytics `teammates`. A[i,j] = how often
    species i and j are teamed together (mean of the two directional pcts); diagonal 0; pad rows 0.

    15b-feat.1b: NOT a per-mon feature — it's a PAIRWISE prior fed as a bias into the self-attention
    (TeamPreviewModel use_teammate_bias). It encodes ESTABLISHED-meta pairings (so the attention starts
    focused on combos humans run together); the outcome training then corrects it toward what WINS
    (and mechanic tags catch the EMERGING combos co-occurrence hasn't caught up to yet)."""
    sp = list(our_species)[:n]
    tm = {s: {norm_species(t["name"]): float(t["p"]) for t in belief.teammates(s)} for s in sp}
    A = np.zeros((n, n), np.float32)
    for i in range(len(sp)):
        for j in range(len(sp)):
            if i == j:
                continue
            a = tm.get(sp[i], {}).get(norm_species(sp[j]), 0.0)
            b = tm.get(sp[j], {}).get(norm_species(sp[i]), 0.0)
            A[i, j] = 0.5 * (a + b)
    return A


def _hard(names):
    return [{"name": n, "p": 1.0} for n in names if n]


# ── the two public extractors (single source of truth for train + serve) ─────────
def _fill_base(f, species, belief):
    f[OFF_DEX:OFF_DEX + MON_FEAT_DIM] = mon_dex_features(species)
    has = bool(belief.known(species))
    ab = belief.ability_distribution(species, top_k=4) if has else []
    mv = belief.move_distribution(species, top_k=12) if has else []
    if has:
        ws, wa = weather_tags(ab, mv)
        ts, ta = terrain_tags(ab, mv)
        f[OFF_WSETS:OFF_WSETS + _W] = ws
        f[OFF_WABUSE:OFF_WABUSE + _W] = wa
        f[OFF_TSETS:OFF_TSETS + _TR] = ts
        f[OFF_TABUSE:OFF_TABUSE + _TR] = ta
        f[OFF_ROLES:OFF_ROLES + _R] = role_tags(ab, mv)
        f[OFF_SPREAD:OFF_SPREAD + _T] = spread_tags(mv)
        f[OFF_REVERSER] = reverser_tag(ab)
        f[OFF_DEBUFF] = ally_debuff_tag(mv)
        f[OFF_ORDER:OFF_ORDER + _O] = order_tags(ab)
        f[OFF_USAGE] = min(belief.usage(species), 100.0) / 100.0
    f[OFF_IMMUNE:OFF_IMMUNE + _T] = immune_tags(species, ab)   # typing always; ability prior if has
    f[OFF_DEFEFF:OFF_DEFEFF + _T] = def_eff_profile(species)   # typing matchup (public, both sides)
    f[OFF_HASDATA] = 1.0 if has else 0.0
    f[OFF_GK + GIMMICK_KINDS.index("none")] = 1.0


def _fill_overlay(f, known: OwnKnown):
    f[OFF_OWNBIT] = 1.0
    ab = _hard([known.ability])
    mv = _hard(known.moves)
    ws, wa = weather_tags(ab, mv)
    ts, ta = terrain_tags(ab, mv)
    f[OFF_KWSETS:OFF_KWSETS + _W] = ws
    f[OFF_KWABUSE:OFF_KWABUSE + _W] = wa
    f[OFF_KTSETS:OFF_KTSETS + _TR] = ts
    f[OFF_KTABUSE:OFF_KTABUSE + _TR] = ta
    f[OFF_KROLES:OFF_KROLES + _R] = role_tags(ab, mv)
    f[OFF_KSPREAD:OFF_KSPREAD + _T] = spread_tags(mv)
    f[OFF_KIMMUNE:OFF_KIMMUNE + _T] = _ability_immune(ab)      # sharp ability immunity (typing already in base)
    f[OFF_KREVERSER] = reverser_tag(ab)
    f[OFF_KDEBUFF] = ally_debuff_tag(mv)
    f[OFF_KORDER:OFF_KORDER + _O] = order_tags(ab)
    f[OFF_KGK + GIMMICK_KINDS.index("mega" if known.will_mega else "none")] = 1.0
    if known.tera:
        ti = _TYPE_IDX.get(_canon_type(known.tera))
        if ti is not None:
            f[OFF_KTERA + ti] = 1.0


def own_mon_features(species: str, belief, known: Optional[OwnKnown] = None) -> np.ndarray:
    """(FEAT_DIM,) OWN-side vector. With ``known=None`` (Type-B BC) the overlay is zero
    and the result is BYTE-IDENTICAL to opp_mon_features(species, belief)."""
    f = np.zeros(FEAT_DIM, dtype=np.float32)
    _fill_base(f, species, belief)
    if known is not None:
        _fill_overlay(f, known)
    return f


def opp_mon_features(species: str, belief) -> np.ndarray:
    """(FEAT_DIM,) OPP-side vector — species + belief PRIOR only. By signature it can
    never see a revealed opp item/ability/move (hidden at preview). Overlay stays zero."""
    f = np.zeros(FEAT_DIM, dtype=np.float32)
    _fill_base(f, species, belief)
    return f


# ── channel-order guard ───────────────────────────────────────────────────────
_EXPECTED = {
    "WEATHERS": ("sand", "rain", "sun", "snow"),
    "TERRAINS": ("electric", "grassy", "psychic", "misty"),
    "ROLE_TAGS": ("trick_room", "tailwind", "redirect", "fake_out", "screens",
                  "speed_control", "priority", "intimidate"),
    "GIMMICK_KINDS": ("none", "mega", "tera", "dynamax"),
    "ORDER_FLAGS": ("illusion", "imposter"),
}


def _assert_channel_order():
    for name, expected in _EXPECTED.items():
        got = globals()[name]
        assert got == expected, (
            f"{name} order changed ({got} != {expected}) — this shifts the OFF_* offsets and would "
            f"silently corrupt any TP net trained on schema {FEATURE_SCHEMA_VERSION}. Bump the schema "
            f"version + _EXPECTED deliberately if this is intended."
        )


_assert_channel_order()
