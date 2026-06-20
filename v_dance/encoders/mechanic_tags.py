"""v9 mechanic TAG taxonomy — multi-hot effect priors that ride ALONGSIDE the identity embeddings
(mechanic_vocab). This module is the v9 substrate; it does NOT touch STATE_DIM (consumed by the v9 encode
path at the layout cutover).

MOVE tags (this sub-step) are DERIVED from the pinned Showdown ``moves.ts`` structured fields — the exact
fields ``scratch/verify_mechanic_coverage.py`` confirmed: ``forceSwitch``/``selfSwitch``/``overrideOffensive
Stat``/``overrideDefensiveStat``/``overrideOffensivePokemon``/``basePowerCallback``/``mustrecharge``/
``flags.charge``/``status``/``secondary.status``/``boosts``/``sideCondition``/``volatileStatus`` — plus a few
small hand sets where single-field detection is unreliable (hazard_clear, redirect, ability/item-interact,
semi_invulnerable). Derived once at import; ``move_tags`` is a lookup. (Ability + item tags land in the next
sub-steps.)
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Set

from v_dance.encoders.mechanic_vocab import _DEX, _norm

# ── MOVE effect-tag taxonomy (29) ────────────────────────────────────────────────
MOVE_TAG_NAMES: List[str] = [
    "self_boost_off", "self_boost_def", "self_boost_spe", "stat_drop_target", "speed_control",
    "force_switch", "pivot", "status_brn", "status_par", "status_slp", "status_psn", "status_frz",
    "status_confuse", "flinch", "hazard_set", "hazard_clear", "redirect", "heal", "ability_interact",
    "item_interact", "move_restrict", "residual_chip", "two_turn_charge", "semi_invulnerable", "recharge",
    "dmg_uses_def", "dmg_uses_target_atk", "dmg_hits_def", "bp_variable",
]
_MTI = {n: i for i, n in enumerate(MOVE_TAG_NAMES)}
NUM_MOVE_TAGS = len(MOVE_TAG_NAMES)

# small hand sets (normalised ids) where a single data field doesn't cleanly identify the class
_HAZARD_CLEAR = {"defog", "rapidspin", "courtchange", "tidyup", "mortalspin", "gmaxwindrage"}
_REDIRECT = {"followme", "ragepowder", "spotlight"}
_ABILITY_INTERACT = {"skillswap", "worryseed", "gastroacid", "entrainment", "simplebeam", "doodle", "roleplay"}
_ITEM_INTERACT = {"knockoff", "trick", "switcheroo", "thief", "covet", "bugbite", "pluck", "fling",
                  "incinerate", "poltergeist", "corrosivegas"}
_SEMI_INVULN = {"fly", "dig", "dive", "bounce", "phantomforce", "shadowforce", "skydrop"}

_STATUS_TAG = {"brn": "status_brn", "par": "status_par", "slp": "status_slp",
               "psn": "status_psn", "tox": "status_psn", "frz": "status_frz"}


def _subobj(block: str, key: str) -> str:
    """Text inside ``key: { ... }`` (balanced braces), or '' if absent."""
    m = re.search(rf"\b{key}:\s*\{{", block)
    if not m:
        return ""
    depth = 0
    start = m.end() - 1
    for j in range(start, len(block)):
        c = block[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return block[start + 1:j]
    return ""


def _axes(boosts: str, neg: bool = False) -> Set[str]:
    """Stat axes (off/def/spe) present with the requested sign inside a flat boosts object string."""
    sign = r"-[1-6]" if neg else r"[1-6]"
    out: Set[str] = set()
    if re.search(rf"atk:\s*{sign}", boosts) or re.search(rf"spa:\s*{sign}", boosts):
        out.add("off")
    if re.search(rf"def:\s*{sign}", boosts) or re.search(rf"spd:\s*{sign}", boosts):
        out.add("def")
    if re.search(rf"spe:\s*{sign}", boosts):
        out.add("spe")
    return out


def _derive_move_tags(mid: str, block: str) -> Set[str]:
    t: Set[str] = set()
    is_status = bool(re.search(r'category:\s*"Status"', block))
    is_self = bool(re.search(r'target:\s*"(self|adjacentAllyOrSelf)"', block))

    self_b = _subobj(_subobj(block, "self"), "boosts")
    sec = _subobj(block, "secondary")
    sec_self_b = _subobj(_subobj(sec, "self"), "boosts")
    sec_b = _subobj(sec, "boosts")
    top_b = _subobj(block, "boosts")  # for a pure status move w/o self/secondary wrapper this is top-level

    # self boosts (positive only) — self:/secondary.self: always; top-level only for a self-target STATUS move
    pos = _axes(self_b) | _axes(sec_self_b) | (_axes(top_b) if (is_status and is_self) else set())
    for ax in pos:
        t.add(f"self_boost_{ax}")
    # target stat drops (negative) — top-level on a foe STATUS move, or any secondary boosts
    drop = (_axes(top_b, neg=True) if (is_status and not is_self) else set()) | _axes(sec_b, neg=True)
    if drop:
        t.add("stat_drop_target")

    for st in set(re.findall(r"status:\s*'(brn|par|slp|psn|tox|frz)'", block)):
        t.add(_STATUS_TAG[st])
    if re.search(r"volatileStatus:\s*'confusion'", block):
        t.add("status_confuse")
    if re.search(r"volatileStatus:\s*'flinch'", block):
        t.add("flinch")

    if re.search(r"forceSwitch:\s*true", block):
        t.add("force_switch")
    if re.search(r"\bselfSwitch:", block):
        t.add("pivot")
    if "mustrecharge" in block:
        t.add("recharge")
    if re.search(r"flags:\s*\{[^}]*\bcharge:\s*1", block):
        t.add("two_turn_charge")
    if mid in _SEMI_INVULN:
        t.add("semi_invulnerable")

    if re.search(r"overrideOffensiveStat:\s*'def'", block):
        t.add("dmg_uses_def")
    if re.search(r"overrideOffensivePokemon:\s*'target'", block):
        t.add("dmg_uses_target_atk")
    if re.search(r"overrideDefensiveStat:\s*'def'", block):
        t.add("dmg_hits_def")
    if "basePowerCallback" in block:
        t.add("bp_variable")

    if re.search(r"sideCondition:\s*'(spikes|stealthrock|toxicspikes|stickyweb|gmaxsteelsurge)'", block):
        t.add("hazard_set")
    if mid in _HAZARD_CLEAR:
        t.add("hazard_clear")
    if re.search(r"\bheal:\s*\[", block) or re.search(r"\bdrain:\s*\[", block):
        t.add("heal")
    if re.search(r"volatileStatus:\s*'(taunt|encore|disable|torment|healblock|imprison)'", block):
        t.add("move_restrict")
    if re.search(r"volatileStatus:\s*'(leechseed|partiallytrapped|saltcure|nightmare|curse|yawn)'", block):
        t.add("residual_chip")
    if mid in _REDIRECT:
        t.add("redirect")
    if mid in _ABILITY_INTERACT:
        t.add("ability_interact")
    if mid in _ITEM_INTERACT:
        t.add("item_interact")

    # speed control = tailwind/sticky web/trick room, paralysis induction, or a target Speed drop
    if (re.search(r"sideCondition:\s*'(tailwind|stickyweb)'", block)
            or re.search(r"pseudoWeather:\s*'trickroom'", block)
            or "status_par" in t
            or ("stat_drop_target" in t and "spe" in (_axes(top_b, neg=True) | _axes(sec_b, neg=True)))):
        t.add("speed_control")
    return t


def _build_move_tags() -> Dict[str, Set[str]]:
    p = _DEX / "moves.ts"
    out: Dict[str, Set[str]] = {}
    if not p.exists():
        return out
    text = p.read_text(encoding="utf-8", errors="replace")
    starts = [m.start() for m in re.finditer(r"(?m)^\t(\w+): \{", text)]
    starts.append(len(text))
    for i in range(len(starts) - 1):
        block = text[starts[i]:starts[i + 1]]
        mid = _norm(re.match(r"\t(\w+):", block).group(1))
        tags = _derive_move_tags(mid, block)
        if tags:
            out[mid] = tags
    return out


MOVE_TAGS: Dict[str, Set[str]] = _build_move_tags()


def move_tags(raw: Optional[str]) -> List[str]:
    """Sorted effect-tag names for a move id/name ([] for empty/unknown/pure-damage)."""
    return sorted(MOVE_TAGS.get(_norm(raw), set())) if raw else []


def move_tag_indices(raw: Optional[str]) -> List[int]:
    return sorted(_MTI[t] for t in MOVE_TAGS.get(_norm(raw), set())) if raw else []


# ── ABILITY effect-tag taxonomy (43; the last, ability_suppressed, is RUNTIME-set, never static) ──
ABILITY_TAG_NAMES: List[str] = [
    "intimidate", "weather_setter", "terrain_setter", "booster_ability", "weather_speed", "speed_control",
    "type_immunity", "damage_boost", "regenerator", "prankster", "magic_bounce", "defensive_reduce",
    "status_immunity", "reactive_boost", "mold_breaker", "guts_boost", "statdrop_boost",            # 17 v8
    "trapping", "unaware", "contrary", "parental_bond", "ko_snowball", "pinch_boost", "stat_drop_immune",
    "magic_guard", "disguise", "protect_bypass", "force_switch_block", "priority_block", "priority_grant",
    "ally_support", "contact_punish", "ability_change", "item_interaction", "contact_status",
    "info_on_entry", "crit_control", "accuracy_mod", "recoil_endure", "type_change", "weather_negate",
    "poison_heal", "ability_suppressed",                                                            # +26 v9
]
_ATI = {n: i for i, n in enumerate(ABILITY_TAG_NAMES)}
NUM_ABILITY_TAGS = len(ABILITY_TAG_NAMES)
ABILITY_SUPPRESSED_IDX = _ATI["ability_suppressed"]   # set by the encoder at runtime, not from membership


def _abilities_with_hook(hook: str) -> Set[str]:
    """Normalised ids of abilities whose dex block contains ``hook`` (data-grounded membership)."""
    ids: Set[str] = set()
    for rel in ("abilities.ts", "mods/champions/abilities.ts"):
        p = _DEX.joinpath(*rel.split("/"))
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        starts = [m.start() for m in re.finditer(r"(?m)^\t(\w+): \{", text)]
        starts.append(len(text))
        for i in range(len(starts) - 1):
            block = text[starts[i]:starts[i + 1]]
            if re.search(hook, block):
                ids.add(_norm(re.match(r"\t(\w+):", block).group(1)))
    return ids


# DATA-GROUNDED (parsed from abilities.ts hooks) — the user's specific interest
_DATA_TRAPPING = _abilities_with_hook(r"onFoeTrapPokemon")            # {shadowtag, arenatrap, magnetpull}
_DATA_MOVE_TYPECHANGE = _abilities_with_hook(r"onModifyType")         # the -ates + normalize/liquidvoice

_ABILITY_TAG_SETS: Dict[str, Set[str]] = {
    "intimidate": {"intimidate"},
    "weather_setter": {"drought", "drizzle", "sandstream", "snowwarning", "desolateland", "primordialsea",
                       "deltastream", "orichalcumpulse"},
    "terrain_setter": {"electricsurge", "grassysurge", "mistysurge", "psychicsurge", "hadronengine"},
    "booster_ability": {"protosynthesis", "quarkdrive"},
    "weather_speed": {"swiftswim", "chlorophyll", "sandrush", "slushrush"},
    "speed_control": {"speedboost", "unburden", "motordrive", "quickfeet"},
    "type_immunity": {"levitate", "voltabsorb", "waterabsorb", "flashfire", "sapsipper", "lightningrod",
                      "stormdrain", "motordrive", "dryskin", "eartheater", "wellbakedbody", "windrider",
                      "purifyingsalt", "soundproof", "bulletproof", "overcoat", "wonderguard"},
    "damage_boost": {"adaptability", "toughclaws", "sheerforce", "technician", "ironfist", "strongjaw",
                     "reckless", "pixilate", "refrigerate", "aerilate", "galvanize", "steelworker",
                     "punkrock", "sharpness", "rockypayload", "transistor", "dragonsmaw", "steelyspirit",
                     "hugepower", "purepower", "hustle", "guts", "tintedlens", "waterbubble", "megalauncher",
                     "firemane", "dragonize", "eelevate", "piercingdrill"},
    "regenerator": {"regenerator"},
    "prankster": {"prankster"},
    "magic_bounce": {"magicbounce", "goodasgold"},
    "defensive_reduce": {"multiscale", "shadowshield", "thickfat", "filter", "solidrock", "prismarmor",
                         "fluffy", "icescales", "furcoat", "heatproof", "wonderguard", "marvelscale"},
    "status_immunity": {"limber", "insomnia", "vitalspirit", "waterveil", "magmaarmor", "immunity",
                        "owntempo", "innerfocus", "sweetveil", "leafguard", "pastelveil", "thermalexchange",
                        "goodasgold", "shieldsdown", "purifyingsalt", "overcoat", "waterbubble", "oblivious",
                        "flowerveil"},
    "reactive_boost": {"justified", "angerpoint", "berserk", "weakarmor", "rattled", "stamina",
                       "watercompaction", "electromorphosis", "windpower", "angershell", "sandspit",
                       "seedsower", "toxicdebris", "steadfast", "opportunist"},
    "mold_breaker": {"moldbreaker", "teravolt", "turboblaze"},
    "guts_boost": {"guts", "quickfeet", "marvelscale", "flareboost", "toxicboost"},
    "statdrop_boost": {"defiant", "competitive"},
    # ── v9 new ──
    "trapping": set(_DATA_TRAPPING),
    "unaware": {"unaware"},
    "contrary": {"contrary", "simple"},
    "parental_bond": {"parentalbond"},
    "ko_snowball": {"moxie", "beastboost", "chillingneigh", "grimneigh", "soulheart", "eelevate"},
    "pinch_boost": {"blaze", "torrent", "overgrow", "swarm", "solarpower", "angershell", "supremeoverlord"},
    "stat_drop_immune": {"clearbody", "whitesmoke", "fullmetalbody", "hypercutter", "bigpecks", "keeneye",
                         "mirrorarmor"},
    "magic_guard": {"magicguard"},
    "disguise": {"disguise", "iceface"},
    "protect_bypass": {"unseenfist", "piercingdrill"},
    "force_switch_block": {"suctioncups", "guarddog"},
    "priority_block": {"queenlymajesty", "dazzling", "armortail"},
    "priority_grant": {"prankster", "galewings", "triage"},
    "ally_support": {"friendguard", "healer", "flowerveil", "sweetveil", "aromaveil", "telepathy",
                     "hospitality", "curiousmedicine", "victorystar", "powerspot", "battery", "costar",
                     "symbiosis"},
    "contact_punish": {"roughskin", "ironbarbs", "aftermath", "innardsout", "perishbody"},
    "ability_change": {"trace", "mummy", "lingeringaroma", "wanderingspirit", "receiver", "powerofalchemy",
                       "imposter"},
    "item_interaction": {"magician", "pickpocket", "stickyhold", "unburden", "symbiosis", "harvest",
                         "klutz", "ripen", "cheekpouch"},
    "contact_status": {"static", "flamebody", "poisonpoint", "effectspore", "cutecharm", "poisontouch",
                       "gooey", "tanglinghair"},
    "info_on_entry": {"frisk", "forewarn", "anticipation"},
    "crit_control": {"superluck", "sniper", "battlearmor", "shellarmor", "merciless", "angerpoint"},
    "accuracy_mod": {"compoundeyes", "noguard", "hustle", "victorystar", "keeneye", "illuminate",
                     "sandveil", "snowcloak", "tangledfeet", "wonderskin", "supersweetsyrup"},
    "recoil_endure": {"rockhead", "reckless", "sturdy"},
    "type_change": set(_DATA_MOVE_TYPECHANGE) | {"protean", "libero", "colorchange", "mimicry", "imposter"},
    "weather_negate": {"cloudnine", "airlock"},
    "poison_heal": {"poisonheal"},
}

# invert -> ability id -> set(tags)
_ABILITY_TAGS: Dict[str, Set[str]] = {}
for _tag, _ids in _ABILITY_TAG_SETS.items():
    for _i in _ids:
        _ABILITY_TAGS.setdefault(_i, set()).add(_tag)


def ability_tags(raw: Optional[str]) -> List[str]:
    """Sorted STATIC effect-tag names for an ability id/name ([] for empty/unknown). ``ability_suppressed``
    is NOT included here — the encoder sets it at runtime (Gastro Acid / Neutralizing Gas)."""
    return sorted(_ABILITY_TAGS.get(_norm(raw), set())) if raw else []


def ability_tag_indices(raw: Optional[str]) -> List[int]:
    return sorted(_ATI[t] for t in _ABILITY_TAGS.get(_norm(raw), set())) if raw else []


# ── ITEM effect-tag taxonomy (23 = the 16 v8 categories, replicated canonically here, + 7 v9 new) ──
ITEM_TAG_NAMES: List[str] = [
    "has_item", "choice", "choice_speed", "focus_sash", "life_orb", "assault_vest", "passive_recovery",
    "heal_berry", "resist_berry", "status_cure", "booster_energy", "contact_punish", "effect_shield",
    "drop_guard", "luck_item", "type_boost",                                                       # 16 v8
    "weather_rock", "terrain_extend", "screen_extend", "escape_trap", "se_boost", "category_boost",
    "mega_stone",                                                                                   # +7 v9
]
_ITI = {n: i for i, n in enumerate(ITEM_TAG_NAMES)}
NUM_ITEM_TAGS = len(ITEM_TAG_NAMES)

# v8 membership (replicated from state_encoder.py so mechanic_tags stays the canonical v9 source w/o a
# circular import; v8 state_encoder keeps its own copy until the layout cutover).
_ITEM_NONE_IDS = frozenset({"", "nothing", "none", "noitem", "unknownitem", "unknown_item"})
_RECOVERY_ITEMS = frozenset({"leftovers", "blacksludge", "shellbell"})
_HEAL_BERRIES = frozenset({"sitrusberry", "oranberry", "aguavberry", "figyberry", "wikiberry",
                           "magoberry", "iapapaberry", "enigmaberry"})
_RESIST_BERRIES = frozenset({"occaberry", "passhoberry", "wacanberry", "rindoberry", "yacheberry",
                             "chopleberry", "kebiaberry", "shucaberry", "cobaberry", "payapaberry",
                             "tangaberry", "chartiberry", "kasibberry", "habanberry", "colburberry",
                             "babiriberry", "chilanberry", "roseliberry"})
_STATUS_CURE_BERRIES = frozenset({"lumberry", "chestoberry", "cheriberry", "pechaberry", "rawstberry",
                                  "aspearberry", "persimberry", "mintberry"})
_CONTACT_PUNISH_ITEMS = frozenset({"rockyhelmet"})
_EFFECT_SHIELD_ITEMS = frozenset({"covertcloak", "safetygoggles"})
_DROP_GUARD_ITEMS = frozenset({"whiteherb", "mentalherb", "clearamulet"})
_LUCK_ITEMS = frozenset({"brightpowder", "quickclaw", "scopelens", "widelens", "zoomlens",
                         "laggingtail", "focusband", "kingsrock", "razorclaw", "razorfang"})
_TYPE_BOOST_ITEMS = frozenset({"charcoal", "mysticwater", "blackbelt", "magnet", "miracleseed",
                               "nevermeltice", "dragonfang", "blackglasses", "sharpbeak", "poisonbarb",
                               "softsand", "hardstone", "silkscarf", "spelltag", "twistedspoon",
                               "metalcoat", "silverpowder", "oddincense", "rockincense", "seaincense",
                               "roseincense", "waveincense", "fairyfeather", "pixieplate"})
# v9 new
_WEATHER_ROCKS = frozenset({"damprock", "heatrock", "icyrock", "smoothrock"})
_CATEGORY_BOOST = frozenset({"muscleband", "wiseglasses"})


def _items_with_field(field: str) -> Set[str]:
    p = _DEX / "items.ts"
    ids: Set[str] = set()
    if not p.exists():
        return ids
    text = p.read_text(encoding="utf-8", errors="replace")
    starts = [m.start() for m in re.finditer(r"(?m)^\t(\w+): \{", text)]
    starts.append(len(text))
    for i in range(len(starts) - 1):
        block = text[starts[i]:starts[i + 1]]
        if re.search(field, block):
            ids.add(_norm(re.match(r"\t(\w+):", block).group(1)))
    return ids


_DATA_MEGA_STONES = _items_with_field(r"\bmegaStone:")   # data-grounded (the ~93 mega stones)


def item_tags(raw: Optional[str]) -> List[str]:
    """Sorted effect-tag names for an item id/name ([] for empty/itemless/Knock-Off'd)."""
    iid = _norm(raw)
    if not iid or iid in _ITEM_NONE_IDS:
        return []
    t: Set[str] = {"has_item"}
    if iid.startswith("choice"):
        t.add("choice")
        if iid == "choicescarf":
            t.add("choice_speed")
    if iid == "focussash":
        t.add("focus_sash")
    if iid == "lifeorb":
        t.add("life_orb")
    if iid == "assaultvest":
        t.add("assault_vest")
    if iid in _RECOVERY_ITEMS:
        t.add("passive_recovery")
    if iid in _HEAL_BERRIES:
        t.add("heal_berry")
    if iid in _RESIST_BERRIES:
        t.add("resist_berry")
    if iid in _STATUS_CURE_BERRIES:
        t.add("status_cure")
    if iid == "boosterenergy":
        t.add("booster_energy")
    if iid in _CONTACT_PUNISH_ITEMS:
        t.add("contact_punish")
    if iid in _EFFECT_SHIELD_ITEMS:
        t.add("effect_shield")
    if iid in _DROP_GUARD_ITEMS:
        t.add("drop_guard")
    if iid in _LUCK_ITEMS:
        t.add("luck_item")
    if iid in _TYPE_BOOST_ITEMS or iid.endswith("plate") or iid.endswith("gem"):
        t.add("type_boost")
    # v9 new
    if iid in _WEATHER_ROCKS:
        t.add("weather_rock")
    if iid == "terrainextender":
        t.add("terrain_extend")
    if iid == "lightclay":
        t.add("screen_extend")
    if iid == "shedshell":
        t.add("escape_trap")
    if iid == "expertbelt":
        t.add("se_boost")
    if iid in _CATEGORY_BOOST:
        t.add("category_boost")
    if iid in _DATA_MEGA_STONES:
        t.add("mega_stone")
    return sorted(t)


def item_tag_indices(raw: Optional[str]) -> List[int]:
    return sorted(_ITI[t] for t in item_tags(raw))
