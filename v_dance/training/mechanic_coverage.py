"""15b-feat.robust.1 — dex-grounded, regulation-agnostic mechanic COVERAGE GUARD.

The curated mechanic-tag tables in tp_features.py are the SERVING source (regulation-
independent game facts). Their one risk is OMISSION: a NEW reg's ability/move that bears a
mechanic would be SILENTLY under-tagged (a zeroed channel, no error). This guard closes that
hole by grounding RECALL in the bundled, version-pinned Showdown dex (PINS.md): it enumerates
every ability/move the GAME defines as mechanic-bearing (setWeather/setTerrain in abilities;
weather:/terrain:/sideCondition:/volatileStatus:/pseudoWeather: fields in moves), and flags any
that is PRESENT in the loaded pikalytics data but ABSENT from the curated tables. So a data swap
that introduces an untagged setter/abuser/role fails a test instead of degrading silently — and
fixing it is a one-line table addition the guard names for you.

Two layers (severity):
  * HIGH (must-have): weather/terrain SETTERS + ABUSERS — these are the synergy carriers.
  * WARN: role heuristics (screens/tailwind/redirect/trick_room) — softer, dex pattern-based.
Anything reviewed-and-deliberately-untagged goes in _KNOWN_UNTAGGED.

Used by tests (a data swap goes red on an unmapped must-have) and importable as a runtime audit.
"""
from __future__ import annotations

import re
from pathlib import Path

from v_dance.training import tp_features as TF

_REPO = Path(__file__).resolve().parents[2]
DEX_DIR = _REPO / "pokemon-showdown" / "data"

_WEATHER_ID = {"sandstorm": "sand", "raindance": "rain", "sunnyday": "sun",
               "snowscape": "snow", "snow": "snow", "hail": "snow"}
_TERRAIN_ID = {"electricterrain": "electric", "grassyterrain": "grassy",
               "psychicterrain": "psychic", "mistyterrain": "misty"}

MUST_HAVE = {"weather_setter", "terrain_setter", "weather_abuser", "terrain_abuser", "spread"}

# Dex mechanic-bearers we have REVIEWED and deliberately do NOT tag (so they don't trip the guard).
_KNOWN_UNTAGGED = {
    # strong/illegal-in-format primal weathers (not in M-A; mapped only as future tripwires elsewhere)
    "Desolate Land", "Primordial Sea", "Delta Stream",
    # terrain-AMBIGUOUS moves (type/effect changes with the active terrain — no single axis)
    "Terrain Pulse", "Steel Roller",
    "Mimicry",
    # manual weather move that is also a pivot; weather handled via the setter table already — n/a
}


def _parse_blocks(path: Path):
    """Yield (display_name, block_text) for each top-level dex entry (``\\tid: { ... }``)."""
    text = path.read_text(encoding="utf-8")
    starts = [m.start() for m in re.finditer(r"(?m)^\t(\w+): \{", text)]
    starts.append(len(text))
    for i in range(len(starts) - 1):
        block = text[starts[i]:starts[i + 1]]
        nm = re.search(r'\n\t\tname: "([^"]+)"', block) or re.search(r'name: "([^"]+)"', block)
        if nm:
            yield nm.group(1), block


_STAT_HOOK = re.compile(r"onModifySpe|onModifyAtk|onModifySpA|onModifyDef|onBasePower")


def dex_mechanic_bearers(dex_dir: Path = DEX_DIR) -> dict:
    """{display_name: set(mechanic_class)} for every dex ability/move that bears a mechanic.
    Empty if the dex is unavailable (the guard then no-ops rather than crashing)."""
    out: dict = {}

    def add(name, cls):
        out.setdefault(name, set()).add(cls)

    abilities, moves = dex_dir / "abilities.ts", dex_dir / "moves.ts"
    if abilities.exists():
        for name, block in _parse_blocks(abilities):
            if any(w in _WEATHER_ID for w in re.findall(r"setWeather\('(\w+)'\)", block)):
                add(name, "weather_setter")
            if any(t in _TERRAIN_ID for t in re.findall(r"setTerrain\('(\w+)'\)", block)):
                add(name, "terrain_setter")
            if re.search(r"isWeather\(|effectiveWeather\(", block) and _STAT_HOOK.search(block):
                add(name, "weather_abuser")
            if re.search(r"isTerrain\(", block) and _STAT_HOOK.search(block):
                add(name, "terrain_abuser")
            if re.search(r"onChangeBoost", block) and re.search(r"\*=?\s*-1", block):
                add(name, "stat_reverser")     # Contrary-style (reverses stat drops into boosts); warn
    if moves.exists():
        for name, block in _parse_blocks(moves):
            if any(w in _WEATHER_ID for w in re.findall(r"weather: '(\w+)'", block)):
                add(name, "weather_setter")
            if any(t in _TERRAIN_ID for t in re.findall(r"terrain: '(\w+)'", block)):
                add(name, "terrain_setter")
            if re.search(r"sideCondition: '(reflect|lightscreen|auroraveil)'", block):
                add(name, "screens")
            if re.search(r"sideCondition: 'tailwind'", block):
                add(name, "tailwind")
            if re.search(r"volatileStatus: '(followme|ragepowder)'", block):
                add(name, "redirect")
            if re.search(r"pseudoWeather: 'trickroom'", block):
                add(name, "trick_room")
            if any(t == "allAdjacent" for t in re.findall(r'target: "(\w+)"', block)):
                add(name, "spread")            # ally-hitting spread move (15b-feat.spread)
    return out


def curated_names() -> set:
    """Every ability/move NAME the tp_features tag tables map (the serving source)."""
    names = (set(TF.SETTER_ABILITY) | set(TF.ABUSER_ABILITY) | set(TF.SETTER_MOVE)
             | set(TF.TERRAIN_SETTER_ABILITY) | set(TF.TERRAIN_ABUSER_ABILITY)
             | set(TF.TERRAIN_SETTER_MOVE) | set(TF.TERRAIN_ABUSER_MOVE)
             | set(TF.SPREAD_MOVE) | set(TF.IMMUNITY_ABILITY)
             | set(TF.STAT_REVERSER_ABILITY) | set(TF.ALLY_DEBUFF_MOVE) | set(TF.ORDER_ABILITY)
             # v8 (2026-07-23) families
             | set(TF.INTIM_PUNISH_ABILITY) | set(TF.INTIM_IMMUNE_ABILITY)
             | set(TF.PRIO_BLOCK_ABILITY) | set(TF.WEATHER_NEGATE_ABILITY)
             | set(TF.SLEEP_MOVE))
    for s in TF.ROLE_MOVE.values():
        names |= set(s)
    for s in TF.ROLE_ABILITY.values():
        names |= set(s)
    return names


def _diff(bearers: dict, present: set, curated: set) -> list:
    """Pure: mechanic-bearers PRESENT in the data but unmapped (and not deliberately untagged)."""
    findings = []
    for name, classes in bearers.items():
        if name not in present or name in curated or name in _KNOWN_UNTAGGED:
            continue
        findings.append({"name": name, "classes": sorted(classes),
                         "severity": "high" if classes & MUST_HAVE else "warn"})
    return sorted(findings, key=lambda f: (f["severity"] != "high", f["name"]))


def names_present_in_data(belief) -> set:
    """Every ability + move NAME any species runs in the loaded pikalytics data."""
    present = set()
    for sp in belief.all_pokemon():
        for a in belief.ability_distribution(sp, top_k=8):
            present.add(a["name"])
        for m in belief.move_distribution(sp, top_k=20):
            present.add(m["name"])
    return present


def audit_mechanic_coverage(belief, dex_dir: Path = DEX_DIR) -> list:
    """Findings for mechanic-bearing abilities/moves PRESENT in the loaded data but UNMAPPED.
    [] means full coverage. severity 'high' = a must-have setter/abuser is missing (FAIL the
    build); 'warn' = a softer role heuristic is missing (review)."""
    return _diff(dex_mechanic_bearers(dex_dir), names_present_in_data(belief), curated_names())


# ════════════════════════════════════════════════════════════════════════════════════════════════
# v9 BATTLE-ENCODER coverage guard (B1-mechanics step 7a)
# ════════════════════════════════════════════════════════════════════════════════════════════════
# The synergy guard above grounds the tp_features tables (TP net, DISPLAY-name space). This second
# layer grounds the v9 BATTLE-encoder substrate: the IDENTITY vocabs (mechanic_vocab) + the effect-TAG
# tables (mechanic_tags), both in normalised-id space. It scans the SAME pinned dex (PINS.md) for the
# must-have battle mechanics and asserts, fail-loud:
#   (1) RECALL — every move the dex marks mechanic-bearing (forceSwitch:true / selfSwitch / hazard
#       sideCondition / status:/secondary.status) gets its tag from ``mechanic_tags.move_tags``, and
#       every ``onFoeTrapPokemon`` ability gets ``trapping``. A regression in ``_derive_move_tags`` /
#       the ability-tag wiring (a renamed tag, a dropped class) flips a test red instead of silently
#       zeroing a channel the net relies on.
#   (2) VOCAB — every ability/move/item NAME the FORMAT (pikalytics) data uses resolves to a non-PAD
#       identity index. A name the usage data carries but the pinned dex doesn't (rename / alias / a
#       scraper artifact like a nature in the abilities field) would otherwise be silently embedded as
#       PAD; the guard names it so it's fixed (dex bump / data clean) before the retrain.
# These run as build-time tests AND are importable as a runtime audit. The two layers are independent
# (separate regexes from the tag derivers) so a refactor that breaks one is caught by the other.
from v_dance.encoders import mechanic_tags as _MT  # noqa: E402
from v_dance.encoders import mechanic_vocab as _MV  # noqa: E402

# Battle mechanics the encoder MUST represent — a dex bearer that loses its tag FAILS the build.
V9_MUST_HAVE = {"force_switch", "pivot", "hazard_set", "trapping",
                "status_brn", "status_par", "status_slp", "status_psn", "status_frz"}

_V9_HAZARD = r"sideCondition:\s*'(spikes|stealthrock|toxicspikes|stickyweb|gmaxsteelsurge)'"
# tox folds into status_psn (same as mechanic_tags._STATUS_TAG; kept independent here on purpose)
_V9_STATUS_TAG = {"brn": "status_brn", "par": "status_par", "slp": "status_slp",
                  "psn": "status_psn", "tox": "status_psn", "frz": "status_frz"}


def _norm_blocks(path: Path):
    """Yield (normalised_id, block_text) for every top-level dex entry in ``path`` ('' if absent)."""
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    starts = [m.start() for m in re.finditer(r"(?m)^\t(\w+): \{", text)]
    starts.append(len(text))
    for i in range(len(starts) - 1):
        block = text[starts[i]:starts[i + 1]]
        yield _MV._norm(re.match(r"\t(\w+):", block).group(1)), block


def required_move_tags(block: str) -> set:
    """Tags the DEX FIELDS require a move to carry — the must-have battle mechanics only. Derived
    independently of ``mechanic_tags._derive_move_tags`` so the two cross-check each other."""
    req: set = set()
    if re.search(r"forceSwitch:\s*true", block):
        req.add("force_switch")
    if re.search(r"\bselfSwitch:", block):
        req.add("pivot")
    if re.search(_V9_HAZARD, block):
        req.add("hazard_set")
    for st in set(re.findall(r"status:\s*'(brn|par|slp|psn|tox|frz)'", block)):
        req.add(_V9_STATUS_TAG[st])
    return req


def audit_v9_move_tag_coverage(dex_dir: Path = DEX_DIR) -> list:
    """Moves the dex marks mechanic-bearing whose ``mechanic_tags.move_tags`` MISSES a required tag.
    [] = full recall. Each finding: {id, missing, severity:'high'} — a fail-loud build gate."""
    findings = []
    for mid, block in _norm_blocks(dex_dir / "moves.ts"):
        missing = required_move_tags(block) - set(_MT.move_tags(mid))
        if missing:
            findings.append({"id": mid, "missing": sorted(missing), "severity": "high"})
    return sorted(findings, key=lambda f: f["id"])


def audit_v9_trapping_coverage(dex_dir: Path = DEX_DIR) -> list:
    """Abilities the dex defines as trapping (onFoeTrapPokemon) that ``mechanic_tags`` does NOT tag
    ``trapping``. [] = full recall (the user-flagged Shadow Tag / Arena Trap / Magnet Pull family)."""
    findings = []
    for rel in ("abilities.ts", "mods/champions/abilities.ts"):
        for aid, block in _norm_blocks(dex_dir.joinpath(*rel.split("/"))):
            if re.search(r"onFoeTrapPokemon", block) and "trapping" not in _MT.ability_tags(aid):
                findings.append({"id": aid, "missing": ["trapping"], "severity": "high"})
    return sorted(findings, key=lambda f: f["id"])


def v9_names_present_in_data(belief) -> dict:
    """Every ability/move/item NAME any species runs in the loaded pikalytics data, by kind. The
    no-item sentinels (Nothing/None/…) are dropped from items — they legitimately map to PAD."""
    abil, move, item = set(), set(), set()
    for sp in belief.all_pokemon():
        for a in belief.ability_distribution(sp, top_k=8):
            abil.add(a["name"])
        for m in belief.move_distribution(sp, top_k=20):
            move.add(m["name"])
        for it in belief.item_distribution(sp, top_k=8):
            item.add(it["name"])
    item = {n for n in item if _MV._norm(n) not in _MT._ITEM_NONE_IDS}
    return {"ability": abil, "move": move, "item": item}


def audit_v9_vocab_coverage(belief) -> dict:
    """Format-present ability/move/item names that do NOT resolve to a v9 identity vocab index.
    {'ability': [], 'move': [], 'item': []} (all empty) = full coverage. A non-empty list = a usage
    name with no embedding slot (rename / alias / scraper artifact) → would silently embed as PAD."""
    present = v9_names_present_in_data(belief)
    return {kind: _MV.missing_from_vocab(kind, names) for kind, names in present.items()}


def audit_v9_coverage(belief, dex_dir: Path = DEX_DIR) -> dict:
    """One-call v9 battle-encoder coverage audit. Fully clean iff every list below is empty:
    {'move_tags': [...], 'trapping': [...], 'vocab': {'ability':[], 'move':[], 'item':[]}}."""
    return {
        "move_tags": audit_v9_move_tag_coverage(dex_dir),
        "trapping": audit_v9_trapping_coverage(dex_dir),
        "vocab": audit_v9_vocab_coverage(belief),
    }
