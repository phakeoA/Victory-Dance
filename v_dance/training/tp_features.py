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

NAME MATCHING IS CANONICAL (2026-07-23 fix): belief data mixes display names ("Drought") with
Showdown IDs ("drought" — the observed-meta rows), so every tag lookup canonicalises BOTH the
table key and the incoming name (lowercase, strip non-alnum). Before this fix 9 ability tags +
28 move tags were silently DEAD in the serving blend (Charizard's sun-setter among them — the
"TP won't bring Zard into rain" bug). The literal tables below KEEP display-case keys (they are
the human-audited registry mechanic_coverage.py checks); only the derived index maps are canon.

MEGA-STONE -> ABILITY (2026-07-23 fix): a held mega stone implies the mega forme's ability
(Charizardite Y => Drought), which the ability MARGINAL under-reports for two-mega species.
_fill_base max-merges dex-resolved stone-implied abilities into the ability distribution before
tagging, and writes the summed stone mass as the gimmick_kind mega prior (previously a constant
"none"). ⚠ Both fixes change feature VALUES (not the schema): serve only behind a TP retrain.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from v_dance.dex.pokedex import norm_species, get_pokedex  # noqa: F401
from v_dance.training.teampreview_dataset import (
    mon_dex_features, MON_FEAT_DIM, NUM_TYPES, _TYPE_IDX, _canon_type,
)

FEATURE_SCHEMA_VERSION = "tpfeat-v9"   # v9 (2026-10-02, mega audit gap 3): +67 base MEGA-FORME channels
                                       # (stone-share-weighted types / def-eff / immunities / base stats /
                                       # expected speed of the forme each stone makes) + 66 overlay twins
                                       # (a KNOWN stone, p=1). Every v8 channel keeps its VALUE; a v8
                                       # checkpoint serves through schema_columns("tpfeat-v8") (a column
                                       # subset of the v9 vector — no frozen copy needed).
# v8 (2026-07-23): +12 base channels (+10 overlay twins) —
                                       # intimidate punish/immune, priority-block, weather-negate,
                                       # sleep, phys-share, expected-speed, 6 item tags — plus the
                                       # canonical name matching + mega-stone ability fixes above.
                                       # ⚠ DIMS AND OFFSETS CHANGED vs v6/v7: older checkpoints
                                       # must serve through the frozen v7 extractor (model_io
                                       # dispatches on the checkpoint's feature_schema).

# ── mechanic axes (ORDER IS LOAD-BEARING — asserted at import) ─────────────────
WEATHERS = ("sand", "rain", "sun", "snow")
TERRAINS = ("electric", "grassy", "psychic", "misty")
ROLE_TAGS = ("trick_room", "tailwind", "redirect", "fake_out", "screens",
             "speed_control", "priority", "intimidate",
             # v8: guard family (Wide Guard = the DEFENSIVE half of the spread synergy —
             # Toxapex shielding Charizard from Rock Slide / itself from Earthquake;
             # Quick Guard = the fake_out/priority counter) + trapping (Infestation
             # pinning a weather setter so it can't re-set by switching).
             "wide_guard", "quick_guard", "trapping")
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
    "wide_guard": {"Wide Guard"},
    "quick_guard": {"Quick Guard"},
    "trapping": {"Infestation", "Fire Spin", "Whirlpool", "Sand Tomb", "Magma Storm",
                 "Thousand Waves", "Jaw Lock", "Anchor Shot", "Spirit Shackle",
                 "Block", "Mean Look", "Bind", "Wrap"},
}
ROLE_ABILITY = {"intimidate": {"Intimidate"},
                "trapping": {"Shadow Tag", "Arena Trap"}}

# ── v8 (2026-07-23): preview-relevant mechanic families the score head could not see before ──
# Intimidate interaction (the Incineroar-into-Kingambit bug): abilities that PUNISH an incoming
# Intimidate (gain from the drop) vs abilities that BLOCK/ignore it (Intimidate value is wasted).
INTIM_PUNISH_ABILITY = {"Defiant", "Competitive", "Guard Dog", "Mirror Armor", "Rattled"}
INTIM_IMMUNE_ABILITY = {"Hyper Cutter", "Clear Body", "White Smoke", "Full Metal Body",
                        "Inner Focus", "Own Tempo", "Oblivious", "Scrappy"}
# Priority/Fake-Out blockers (devalue the opponent's fake_out / priority role tags).
PRIO_BLOCK_ABILITY = {"Armor Tail", "Queenly Majesty", "Dazzling"}
# Weather nullifiers (devalue BOTH sides' weather setter/abuser tags while on field).
WEATHER_NEGATE_ABILITY = {"Cloud Nine", "Air Lock"}
# Sleep-inflicting moves (a preview-level threat axis; pairs vs safety_goggles/Misty setters).
SLEEP_MOVE = {"Spore", "Sleep Powder", "Hypnosis", "Yawn", "Dark Void", "Sing",
              "Grass Whistle", "Lovely Kiss"}
# Item tags — the first item channels in the TP schema. Order is load-bearing (channel axis).
ITEM_TAGS = ("focus_sash", "choice_lock", "safety_goggles", "covert_cloak",
             "booster_energy", "clear_amulet")

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
    "Levitate": "Ground", "Earth Eater": "Ground", "Eelevate": "Ground",   # Eelevate = Mega Eelektross
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
_I = len(ITEM_TAGS)

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
# ── v8 mechanic channels ──
OFF_INTIMP = OFF_ORDER + _O                   # intim_punish: Defiant/Competitive/Guard Dog/... (1)
OFF_INTIMI = OFF_INTIMP + 1                   # intim_immune: Hyper Cutter/Clear Body/Inner Focus/... (1)
OFF_PRIOB = OFF_INTIMI + 1                    # prio_block: Armor Tail/Queenly Majesty/Dazzling (1)
OFF_WNEG = OFF_PRIOB + 1                      # weather_negate: Cloud Nine/Air Lock (1)
OFF_SLEEP = OFF_WNEG + 1                      # sleep_move: carries a sleep-inflicting move (1)
OFF_PHYSSH = OFF_SLEEP + 1                    # phys_share: physical fraction of damaging moves (1)
OFF_EXPSPE = OFF_PHYSSH + 1                   # exp_speed: spread-weighted expected speed /255 (1)
OFF_ITEMS = OFF_EXPSPE + 1                    # item tags (len ITEM_TAGS)
OFF_DEFEFF = OFF_ITEMS + _I                   # def_eff[NUM_TYPES]: signed typing defensive effectiveness
OFF_HASDATA = OFF_DEFEFF + _T                 # belief.known(species) (1)
OFF_USAGE = OFF_HASDATA + 1                   # usage_pct/100 (1)
# ── v9 MEGA-FORME block (base, symmetric): Σ over the species' stones of share(stone) × the forme it makes ──
OFF_MTYPE = OFF_USAGE + 1                     # mega forme typing multi-hot (NUM_TYPES)
OFF_MDEFEFF = OFF_MTYPE + _T                  # mega forme signed defensive effectiveness (NUM_TYPES)
OFF_MIMMUNE = OFF_MDEFEFF + _T                # mega forme immunities: typing 0x + its fixed ability (NUM_TYPES)
OFF_MSTATS = OFF_MIMMUNE + _T                 # mega forme base stats /255 (6)
OFF_MEXPSPE = OFF_MSTATS + 6                  # mega forme spread-weighted expected speed /255 (1)
BASE_DIM = OFF_MEXPSPE + 1                     # ── end of the SYMMETRIC base block
OFF_GK = BASE_DIM                              # gimmick_kind one-hot (4) — mega prior since v8
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
# ── v8 hard-known twins (ability from the sheet incl. stone-implied mega ability) ──
OFF_KINTIMP = OFF_KORDER + _O
OFF_KINTIMI = OFF_KINTIMP + 1
OFF_KPRIOB = OFF_KINTIMI + 1
OFF_KWNEG = OFF_KPRIOB + 1
OFF_KSLEEP = OFF_KWNEG + 1
OFF_KPHYSSH = OFF_KSLEEP + 1
OFF_KEXPSPE = OFF_KPHYSSH + 1                  # exact own speed /255 (OwnKnown.spe), 0 if unknown
OFF_KITEMS = OFF_KEXPSPE + 1                   # hard known item tags (len ITEM_TAGS)
OFF_KGK = OFF_KITEMS + _I
OFF_KTERA = OFF_KGK + _G
# ── v9 hard-known MEGA-FORME twins (a sheet-revealed / own stone: the forme it makes, p=1) ──
OFF_KMTYPE = OFF_KTERA + _T
OFF_KMDEFEFF = OFF_KMTYPE + _T
OFF_KMIMMUNE = OFF_KMDEFEFF + _T
OFF_KMSTATS = OFF_KMIMMUNE + _T
FEAT_DIM = OFF_KMSTATS + 6

# ── older schemas served as a COLUMN SUBSET of the current vector (v9 only inserted blocks) ──
# v8 = v9 minus the base mega block [OFF_MTYPE, BASE_DIM) minus the overlay mega twins [OFF_KMTYPE, FEAT_DIM):
# every v8 channel is computed by the same code with the same value, so model_io / tp_val_report / the surgery
# feed a v8 checkpoint exactly the vector it trained on.
_V8_COLUMNS = np.r_[0:OFF_MTYPE, BASE_DIM:OFF_KMTYPE].astype(np.int64)
SCHEMA_DIMS = {"tpfeat-v8": int(len(_V8_COLUMNS)), FEATURE_SCHEMA_VERSION: FEAT_DIM}


def schema_columns(schema):
    """Index array that turns a CURRENT (v9) feature vector into ``schema``'s, or None for the current schema."""
    if schema == FEATURE_SCHEMA_VERSION:
        return None
    if schema == "tpfeat-v8":
        return _V8_COLUMNS
    raise ValueError(f"no column view from {FEATURE_SCHEMA_VERSION} to {schema!r}")


def view_for_schema(feats, schema):
    """``feats[..., cols]`` for an older schema (v8), ``feats`` unchanged for the current one."""
    cols = schema_columns(schema)
    return feats if cols is None else np.asarray(feats)[..., cols]


@dataclass
class OwnKnown:
    """The sharp OWN-side build, known at serve / self-play / Type-A only."""
    ability: Optional[str] = None
    moves: Sequence[str] = ()
    item: Optional[str] = None
    tera: Optional[str] = None
    will_mega: bool = False
    spe: Optional[float] = None   # v8: exact in-battle speed stat (from the own team's EVs)


# ── pure tag functions (no belief; unit-tested in isolation) ──────────────────────
import re as _re

_CANON_RE = _re.compile(r"[^a-z0-9]")


def _canon_name(s) -> str:
    """Canonical ability/move/item key: lowercase, strip non-alnum — collapses
    display names ("Drought") and Showdown IDs ("drought") to one key."""
    return _CANON_RE.sub("", str(s).lower()) if s else ""


def _idx_map(name_to_axis, axes):
    return {_canon_name(nm): axes.index(ax) for nm, ax in name_to_axis.items() if ax in axes}


def _canon_set(names):
    return {_canon_name(n) for n in names}


_WSET_A = _idx_map(SETTER_ABILITY, WEATHERS)
_WABU_A = _idx_map(ABUSER_ABILITY, WEATHERS)
_WSET_M = _idx_map(SETTER_MOVE, WEATHERS)
_TSET_A = _idx_map(TERRAIN_SETTER_ABILITY, TERRAINS)
_TABU_A = _idx_map(TERRAIN_ABUSER_ABILITY, TERRAINS)
_TSET_M = _idx_map(TERRAIN_SETTER_MOVE, TERRAINS)
_TABU_M = _idx_map(TERRAIN_ABUSER_MOVE, TERRAINS)
# spread move / immunity index maps (over the TYPE axis) — canon keys (see header)
_SPREAD_I = {_canon_name(nm): _TYPE_IDX[_canon_type(t)] for nm, t in SPREAD_MOVE.items()
             if _canon_type(t) in _TYPE_IDX}
_IMMABIL_I = {_canon_name(nm): _TYPE_IDX[_canon_type(t)] for nm, t in IMMUNITY_ABILITY.items()
              if _canon_type(t) in _TYPE_IDX}
_TYPE_IMMUNITY_C = {_canon_type(k): {_canon_type(a) for a in v} for k, v in TYPE_IMMUNITY.items()}


def _field_tags(ability_probs, move_probs, set_a, abu_a, set_m, abu_m, n):
    sets = np.zeros(n, np.float32)
    abuse = np.zeros(n, np.float32)
    for a in ability_probs:
        nm, p = _canon_name(a["name"]), float(a["p"])
        if nm in set_a:
            sets[set_a[nm]] += p
        if nm in abu_a:
            abuse[abu_a[nm]] += p
    for m in move_probs:
        nm, p = _canon_name(m["name"]), float(m["p"])
        if nm in set_m:
            sets[set_m[nm]] += p
        if nm in abu_m:
            abuse[abu_m[nm]] += p
    return np.clip(sets, 0.0, 1.0), np.clip(abuse, 0.0, 1.0)


def weather_tags(ability_probs, move_probs):
    return _field_tags(ability_probs, move_probs, _WSET_A, _WABU_A, _WSET_M, {}, _W)


def terrain_tags(ability_probs, move_probs):
    return _field_tags(ability_probs, move_probs, _TSET_A, _TABU_A, _TSET_M, _TABU_M, _TR)


_ROLE_MOVE_C = {tag: _canon_set(s) for tag, s in ROLE_MOVE.items()}
_ROLE_ABILITY_C = {tag: _canon_set(s) for tag, s in ROLE_ABILITY.items()}


def role_tags(ability_probs, move_probs):
    r = np.zeros(_R, np.float32)
    mp = {}
    for m in move_probs:                      # canon keys; keep MAX on collision (id + display dupes)
        k = _canon_name(m["name"])
        mp[k] = max(mp.get(k, 0.0), float(m["p"]))
    ap = {}
    for a in ability_probs:
        k = _canon_name(a["name"])
        ap[k] = max(ap.get(k, 0.0), float(a["p"]))
    for i, tag in enumerate(ROLE_TAGS):
        if tag in _ROLE_MOVE_C:
            r[i] += sum(mp.get(mv, 0.0) for mv in _ROLE_MOVE_C[tag])
        if tag in _ROLE_ABILITY_C:
            r[i] += sum(ap.get(ab, 0.0) for ab in _ROLE_ABILITY_C[tag])
    return np.clip(r, 0.0, 1.0)


def spread_tags(move_probs):
    """spread[NUM_TYPES]: carries an ally-hitting (allAdjacent) spread move of this type."""
    v = np.zeros(_T, np.float32)
    for m in move_probs:
        i = _SPREAD_I.get(_canon_name(m["name"]))
        if i is not None:
            v[i] += float(m["p"])
    return np.clip(v, 0.0, 1.0)


def _species_types(species):
    dex = get_pokedex()
    e = dex.entry(species) if dex else None
    return [_canon_type(t) for t in ((e or {}).get("types") or [])]


def _typing_immune(species):
    return _typing_immune_types(_species_types(species))


def _typing_immune_types(types):
    v = np.zeros(_T, np.float32)
    for tc in types:
        for atk in _TYPE_IMMUNITY_C.get(tc, ()):
            idx = _TYPE_IDX.get(atk)
            if idx is not None:
                v[idx] = 1.0
    return v


def _ability_immune(ability_probs):
    v = np.zeros(_T, np.float32)
    for a in ability_probs:
        i = _IMMABIL_I.get(_canon_name(a["name"]))
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
    return def_eff_from_types(_species_types(species))   # canonical (uppercase), matches poke-env chart keys


def def_eff_from_types(types):
    """def_eff_profile for an explicit TYPE LIST (v9: the mega forme's typing)."""
    v = np.zeros(_T, np.float32)
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


_REVERSER_C = _canon_set(STAT_REVERSER_ABILITY)
_DEBUFF_C = _canon_set(ALLY_DEBUFF_MOVE)


def reverser_tag(ability_probs):
    """stat_reverser scalar: benefits from stat drops (Contrary) — the beneficiary half."""
    return float(min(1.0, sum(float(a["p"]) for a in ability_probs
                              if _canon_name(a["name"]) in _REVERSER_C)))


def ally_debuff_tag(move_probs):
    """ally_debuff scalar: carries an ally-targetable stat-lowering move — the enabler half."""
    return float(min(1.0, sum(float(m["p"]) for m in move_probs
                              if _canon_name(m["name"]) in _DEBUFF_C)))


_ORDER_I = {_canon_name(nm): ORDER_FLAGS.index(f) for nm, f in ORDER_ABILITY.items()}


def order_tags(ability_probs):
    """order[len(ORDER_FLAGS)]: ability makes the slot/order choice high-stakes (Illusion / Imposter)."""
    v = np.zeros(_O, np.float32)
    for a in ability_probs:
        i = _ORDER_I.get(_canon_name(a["name"]))
        if i is not None:
            v[i] += float(a["p"])
    return np.clip(v, 0.0, 1.0)


# ── v8 tag functions ──────────────────────────────────────────────────────────
_INTIM_PUNISH_C = _canon_set(INTIM_PUNISH_ABILITY)
_INTIM_IMMUNE_C = _canon_set(INTIM_IMMUNE_ABILITY)
_PRIO_BLOCK_C = _canon_set(PRIO_BLOCK_ABILITY)
_WNEG_C = _canon_set(WEATHER_NEGATE_ABILITY)
_SLEEP_C = _canon_set(SLEEP_MOVE)


def ability_scalar_tag(ability_probs, canon_names) -> float:
    """Summed probability mass of the abilities whose canon name is in ``canon_names``, clipped."""
    return float(min(1.0, sum(float(a["p"]) for a in ability_probs
                              if _canon_name(a["name"]) in canon_names)))


def sleep_tag(move_probs) -> float:
    """Probability mass of carrying a sleep-inflicting move, clipped."""
    return float(min(1.0, sum(float(m["p"]) for m in move_probs
                              if _canon_name(m["name"]) in _SLEEP_C)))


def item_tags(item_probs) -> np.ndarray:
    """item[len(ITEM_TAGS)]: per-tag MAX item probability (choice_lock = any Choice item)."""
    v = np.zeros(_I, np.float32)
    for it in item_probs:
        ck, p = _canon_name(it["name"]), float(it["p"])
        if ck.startswith("choice"):
            v[ITEM_TAGS.index("choice_lock")] = max(v[ITEM_TAGS.index("choice_lock")], p)
        elif ck == "focussash":
            v[ITEM_TAGS.index("focus_sash")] = max(v[ITEM_TAGS.index("focus_sash")], p)
        elif ck == "safetygoggles":
            v[ITEM_TAGS.index("safety_goggles")] = max(v[ITEM_TAGS.index("safety_goggles")], p)
        elif ck == "covertcloak":
            v[ITEM_TAGS.index("covert_cloak")] = max(v[ITEM_TAGS.index("covert_cloak")], p)
        elif ck == "boosterenergy":
            v[ITEM_TAGS.index("booster_energy")] = max(v[ITEM_TAGS.index("booster_energy")], p)
        elif ck == "clearamulet":
            v[ITEM_TAGS.index("clear_amulet")] = max(v[ITEM_TAGS.index("clear_amulet")], p)
    return v


_MOVE_CATEGORY: Optional[dict] = None    # canon move name -> "physical" | "special" | "status"


def _move_category_map() -> dict:
    global _MOVE_CATEGORY
    if _MOVE_CATEGORY is None:
        import json
        from pathlib import Path
        path = Path(__file__).resolve().parents[2] / "data" / "moves.json"
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            _MOVE_CATEGORY = {_canon_name(k): str(v.get("category", "")).lower()
                              for k, v in raw.items()}
        except Exception:
            _MOVE_CATEGORY = {}    # missing data -> phys_share stays neutral
    return _MOVE_CATEGORY


def phys_share(move_probs) -> float:
    """Physical fraction of the DAMAGING move mass (0.5 = neutral / no damaging info) —
    the Intimidate/screens leverage axis."""
    cats = _move_category_map()
    phys = spec = 0.0
    for m in move_probs:
        c = cats.get(_canon_name(m["name"]))
        if c == "physical":
            phys += float(m["p"])
        elif c == "special":
            spec += float(m["p"])
    total = phys + spec
    return float(phys / total) if total > 0 else 0.5


def expected_speed(species, belief) -> float:
    """Spread-weighted expected in-battle speed /255 (base-stat fallback) — the
    Trick-Room / tailwind decision axis (base speed alone hides EV investment)."""
    from v_dance.parser.belief_state import dex_base_stats
    return expected_speed_from_base(species, dex_base_stats(species) or {}, belief)


def expected_speed_from_base(species, base, belief) -> float:
    """expected_speed with explicit BASE stats (v9: the species' EV/nature spreads on its mega forme's stats)."""
    fn = getattr(belief, "expected_stats_weighted", None)   # stub beliefs may omit it
    if callable(fn):
        try:
            st = fn(species, base)
            if st and st.get("spe"):
                return float(np.clip(float(st["spe"]) / 255.0, 0.0, 1.0))
        except Exception:
            pass
    return float(np.clip((base.get("spe", 0) or 0) / 255.0, 0.0, 1.0))


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


def _stone_augmented_abilities(species, ability_probs, item_probs):
    """Max-merge mega-stone-implied abilities into the ability marginal.

    A held mega stone fixes the mega forme's ability (Charizardite Y => Drought) — information
    the raw ability marginal under-reports for two-mega species (Pikalytics lists only BASE
    abilities; observed data smears X/Y). Dex-driven (requiredItem + the forme's single ability),
    no curated table. MAX-merge, not replace: when the observed marginal already carries the mega
    ability harder than the stone share (Zard `drought` 0.68 vs stone 0.24), the stronger signal
    wins. Returns ``(ab_aug, p_mega)`` — ``p_mega`` = summed stone mass (the gimmick prior)."""
    if not item_probs:
        return ability_probs, 0.0
    dex = get_pokedex()
    formes = dex.mega_formes_for(species) if dex else []
    if not formes:
        return ability_probs, 0.0
    ip = {}
    for i in item_probs:
        k = _canon_name(i["name"])
        ip[k] = max(ip.get(k, 0.0), float(i["p"]))
    ab_aug = list(ability_probs)
    p_mega = 0.0
    counted = set()
    for fm in formes:
        fe = dex.entry(fm["forme"]) or {}
        req, mega_ab = _canon_name(fe.get("requiredItem")), fm.get("ability")
        p = ip.get(req, 0.0)
        if not req or not mega_ab or p <= 0.0:
            continue
        if req not in counted:          # one stone, one share: Meowsticite serves M-Mega AND F-Mega (10-02)
            counted.add(req)
            p_mega += p
        ck, merged, out = _canon_name(mega_ab), False, []
        for e in ab_aug:
            if _canon_name(e["name"]) == ck:
                out.append({"name": e["name"], "p": max(float(e["p"]), p)})
                merged = True
            else:
                out.append(e)
        if not merged:
            out.append({"name": mega_ab, "p": p})
        ab_aug = out
    return ab_aug, float(min(p_mega, 1.0))


# ── v9: the MEGA FORME each stone makes (mega audit gap 3, 2026-10-02) ───────────────────────
def mega_forme_for_stone(species, stone):
    """``(forme_name, dex_entry, fixed_ability_or_None)`` of the mega forme ``stone`` turns ``species`` into, or
    None. One stone can serve several formes (Meowsticite → M-Mega / F-Mega; Tatsugirinite → Curly / Droopy /
    Stretchy): narrow by the species' own id ('meowsticf' → F-Mega), then the default (non-'F-') forme."""
    dex = get_pokedex()
    if not dex or not stone or not species:
        return None
    sid, st = norm_species(species), _canon_name(stone)
    cands = []
    from v_dance.encoders.mega_preview import mega_reachable   # Raichu-Alola never megas (review 10-02)
    for fm in dex.mega_formes_for(species):
        fe = dex.entry(fm["forme"]) or {}
        if _canon_name(fe.get("requiredItem")) == st and mega_reachable(sid, fe):
            cands.append((fm["forme"], fe, fm.get("ability")))
    if len(cands) > 1:
        pref = [c for c in cands if norm_species(c[0]).startswith(sid)]
        cands = pref or cands
    if len(cands) > 1:
        cands = [c for c in cands if not (c[1].get("forme") or "").startswith("F-")] or cands
    return sorted(cands, key=lambda c: norm_species(c[0]))[0] if cands else None


def species_stones(species):
    """The distinct mega stones of ``species`` (canon names), in dex order."""
    from v_dance.encoders.mega_preview import mega_reachable
    dex = get_pokedex()
    out = []
    for fm in (dex.mega_formes_for(species) if dex else []):
        fe = dex.entry(fm["forme"]) or {}
        st = _canon_name(fe.get("requiredItem"))
        if st and st not in out and mega_reachable(norm_species(species), fe):
            out.append(st)
    return out


def mega_profile(species, stone_probs, belief=None):
    """Stone-share-weighted MEGA-FORME channels: ``(mtype, mdefeff, mimmune, mstats, mexpspe)``.

    ``stone_probs`` = {canon stone: p} (belief item shares, or {stone: 1.0} for a known held stone). Each stone
    counts ONCE with its forme (Garchompite Z 0.45 → Dragon / Levitate / Spe 151; Garchompite 0.02 → Dragon /
    Ground / Sand Force). All zeros for a mon with no stone — a non-mega mon keeps exactly its v8 values."""
    mtype = np.zeros(_T, np.float32)
    mdef = np.zeros(_T, np.float32)
    mimm = np.zeros(_T, np.float32)
    mst = np.zeros(6, np.float32)
    mspe = 0.0
    if not stone_probs:
        return mtype, mdef, mimm, mst, mspe
    from v_dance.parser.belief_state import STAT_ORDER, dex_base_stats
    for st in species_stones(species):
        p = float(stone_probs.get(st, 0.0))
        if p <= 0.0:
            continue
        hit = mega_forme_for_stone(species, st)
        if hit is None:
            continue
        forme, fe, mega_ab = hit
        types = [_canon_type(t) for t in (fe.get("types") or [])]
        for t in types:
            if t in _TYPE_IDX:
                mtype[_TYPE_IDX[t]] += p
        mdef += p * def_eff_from_types(types)
        imm = _typing_immune_types(types)
        if mega_ab:
            imm = np.maximum(imm, _ability_immune([{"name": mega_ab, "p": 1.0}]))
        mimm += p * imm
        base = dex_base_stats(forme) or {}         # short keys (atk/def/…); the raw dex says 'attack'/'speed'
        mst += p * np.array([(base.get(k, 0) or 0) / 255.0 for k in STAT_ORDER], np.float32)
        mspe += p * expected_speed_from_base(species, base, belief)
    return (np.clip(mtype, 0.0, 1.0), np.clip(mdef, -1.0, 1.0), np.clip(mimm, 0.0, 1.0),
            np.clip(mst, 0.0, 1.0), float(np.clip(mspe, 0.0, 1.0)))


def _stone_probs(item_probs):
    ip = {}
    for i in item_probs or ():
        k = _canon_name(i["name"])
        ip[k] = max(ip.get(k, 0.0), float(i["p"]))
    return ip


# ── the two public extractors (single source of truth for train + serve) ─────────
def _fill_base(f, species, belief):
    f[OFF_DEX:OFF_DEX + MON_FEAT_DIM] = mon_dex_features(species)
    has = bool(belief.known(species))
    ab = belief.ability_distribution(species, top_k=4) if has else []
    mv = belief.move_distribution(species, top_k=12) if has else []
    p_mega = 0.0
    if has:
        item_fn = getattr(belief, "item_distribution", None)   # stub beliefs may omit it
        items = item_fn(species, top_k=6) if callable(item_fn) else []
        ab, p_mega = _stone_augmented_abilities(species, ab, items)
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
        # v8 mechanic channels (belief-prior side)
        f[OFF_INTIMP] = ability_scalar_tag(ab, _INTIM_PUNISH_C)
        f[OFF_INTIMI] = ability_scalar_tag(ab, _INTIM_IMMUNE_C)
        f[OFF_PRIOB] = ability_scalar_tag(ab, _PRIO_BLOCK_C)
        f[OFF_WNEG] = ability_scalar_tag(ab, _WNEG_C)
        f[OFF_SLEEP] = sleep_tag(mv)
        f[OFF_PHYSSH] = phys_share(mv)
        f[OFF_ITEMS:OFF_ITEMS + _I] = item_tags(items)
        f[OFF_USAGE] = min(belief.usage(species), 100.0) / 100.0
    f[OFF_EXPSPE] = expected_speed(species, belief)            # spread-weighted; base-stat fallback
    f[OFF_IMMUNE:OFF_IMMUNE + _T] = immune_tags(species, ab)   # typing always; ability prior if has
    f[OFF_DEFEFF:OFF_DEFEFF + _T] = def_eff_profile(species)   # typing matchup (public, both sides)
    f[OFF_HASDATA] = 1.0 if has else 0.0
    # gimmick prior: mega mass = summed mega-stone item share (0 when no stones / no data)
    f[OFF_GK + GIMMICK_KINDS.index("none")] = 1.0 - p_mega
    f[OFF_GK + GIMMICK_KINDS.index("mega")] = p_mega
    # v9: the forme(s) those stones make — typing / matchups / stats the picker scored as the BASE forme before
    if has and p_mega > 0.0:
        mt, md, mi, ms, mspe = mega_profile(species, _stone_probs(items), belief)
        f[OFF_MTYPE:OFF_MTYPE + _T] = mt
        f[OFF_MDEFEFF:OFF_MDEFEFF + _T] = md
        f[OFF_MIMMUNE:OFF_MIMMUNE + _T] = mi
        f[OFF_MSTATS:OFF_MSTATS + 6] = ms
        f[OFF_MEXPSPE] = mspe


def _stone_ability_for(species, item) -> Optional[str]:
    """The mega ability a KNOWN held stone locks in for ``species`` (None if not its stone)."""
    dex = get_pokedex()
    if not dex or not item:
        return None
    for fm in dex.mega_formes_for(species):
        fe = dex.entry(fm["forme"]) or {}
        if _canon_name(fe.get("requiredItem")) == _canon_name(item) and fm.get("ability"):
            return fm["ability"]
    return None


def _fill_overlay(f, species, known: OwnKnown):
    f[OFF_OWNBIT] = 1.0
    # v8: a known mega stone locks in the mega forme's ability (Charizardite Y => Drought = 1.0).
    # UNION with the sheet/base ability: entry-triggered base abilities (Intimidate) still fire
    # pre-mega, and the mega ability governs from the mega turn on — both are real this game.
    stone_ab = _stone_ability_for(species, known.item)
    ab = _hard([known.ability] + ([stone_ab] if stone_ab else []))
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
    # v8 hard-known twins
    f[OFF_KINTIMP] = ability_scalar_tag(ab, _INTIM_PUNISH_C)
    f[OFF_KINTIMI] = ability_scalar_tag(ab, _INTIM_IMMUNE_C)
    f[OFF_KPRIOB] = ability_scalar_tag(ab, _PRIO_BLOCK_C)
    f[OFF_KWNEG] = ability_scalar_tag(ab, _WNEG_C)
    f[OFF_KSLEEP] = sleep_tag(mv)
    f[OFF_KPHYSSH] = phys_share(mv)
    if known.spe is not None:
        f[OFF_KEXPSPE] = float(np.clip(float(known.spe) / 255.0, 0.0, 1.0))
    if known.item:
        f[OFF_KITEMS:OFF_KITEMS + _I] = item_tags([{"name": known.item, "p": 1.0}])
    will_mega = bool(known.will_mega or stone_ab)              # a held stone implies the mega
    f[OFF_KGK + GIMMICK_KINDS.index("mega" if will_mega else "none")] = 1.0
    if known.item:                                             # v9: the forme a KNOWN stone makes (p=1)
        mt, md, mi, ms, _spe = mega_profile(species, {_canon_name(known.item): 1.0})
        f[OFF_KMTYPE:OFF_KMTYPE + _T] = mt
        f[OFF_KMDEFEFF:OFF_KMDEFEFF + _T] = md
        f[OFF_KMIMMUNE:OFF_KMIMMUNE + _T] = mi
        f[OFF_KMSTATS:OFF_KMSTATS + 6] = ms
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
        _fill_overlay(f, species, known)
    return f


def opp_mon_features(species: str, belief,
                     revealed: Optional[OwnKnown] = None) -> np.ndarray:
    """(FEAT_DIM,) OPP-side vector — species + belief prior, PLUS the revealed
    sheet overlay when open team sheets make the opponent's build preview-visible
    (tpfeat-v7 / the OTS regimes).  ``revealed=None`` (closed sheets — ladder)
    keeps the overlay zero, byte-identical to v6: the ``has_own_detail`` bit acts
    as the per-mon open/closed regime marker, mirroring the battle encoder's
    known-vs-belief marks."""
    f = np.zeros(FEAT_DIM, dtype=np.float32)
    _fill_base(f, species, belief)
    if revealed is not None:
        _fill_overlay(f, species, revealed)
    return f


# ── channel-order guard ───────────────────────────────────────────────────────
_EXPECTED = {
    "WEATHERS": ("sand", "rain", "sun", "snow"),
    "TERRAINS": ("electric", "grassy", "psychic", "misty"),
    "ROLE_TAGS": ("trick_room", "tailwind", "redirect", "fake_out", "screens",
                  "speed_control", "priority", "intimidate",
                  "wide_guard", "quick_guard", "trapping"),          # extended in v8
    "GIMMICK_KINDS": ("none", "mega", "tera", "dynamax"),
    "ORDER_FLAGS": ("illusion", "imposter"),
    "ITEM_TAGS": ("focus_sash", "choice_lock", "safety_goggles", "covert_cloak",
                  "booster_energy", "clear_amulet"),                  # new in v8
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
