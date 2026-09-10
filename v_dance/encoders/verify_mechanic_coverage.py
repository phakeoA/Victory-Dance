"""One-off coverage verification: parse the cloned Showdown data files and tally every structured
mechanic field, then check each maps to a planned tag / scalar / the identity embedding. Run:
  .venv/Scripts/python.exe -m v_dance.encoders.verify_mechanic_coverage
(moved from scratch/verify_mechanic_coverage.py in the 2026-09-10 refactor, Phase 1, and wrapped in main())
KEEP — this is the empirical basis for "every mechanic is tagged/encodable" before the B1-mechanics retrain.
"""
from __future__ import annotations
import re
from pathlib import Path
from collections import Counter

REPO = Path(__file__).resolve().parents[2]           # v_dance/encoders/ -> repo root
assert (REPO / "pyproject.toml").is_file(), REPO
DATA = REPO / "pokemon-showdown" / "data"


def parse_blocks(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    starts = [m.start() for m in re.finditer(r"(?m)^\t(\w+): \{", text)]
    starts.append(len(text))
    for i in range(len(starts) - 1):
        block = text[starts[i]:starts[i + 1]]
        idm = re.match(r"\t(\w+):", block)
        yield (idm.group(1) if idm else "?"), block


# ── MOVES: the canonical structured mechanic axes the engine defines ──────────────
# field-regex -> planned coverage (tag / scalar / embedding)
MOVE_AXES = {
    r"\bforceSwitch:\s*true": "tag:force_switch",
    r"\bselfSwitch:": "tag:pivot",
    r"volatileStatus:\s*'mustrecharge'|recharge:\s*true": "tag:recharge",
    r"flags:\s*\{[^}]*\bcharge:\s*1": "tag:two_turn_charge/semi_invulnerable",
    r"overrideOffensiveStat:\s*'def'": "tag:dmg_uses_def (BodyPress)",
    r"overrideOffensivePokemon:\s*'target'": "tag:dmg_uses_target_atk (FoulPlay)",
    r"overrideDefensiveStat:\s*'def'": "tag:dmg_hits_def (Psyshock)",
    r"\bbasePowerCallback": "tag:bp_variable (variable BP)",
    r"\bonModifyType": "tag:type_change (move-type change)",
    r"\bboosts:\s*\{": "tag:self_boost_*/stat_drop_target",
    r"self:\s*\{[^}]*boosts:": "tag:self_boost_*",
    r"\bstatus:\s*'(brn|par|slp|psn|tox|frz)'": "tag:status_* (primary)",
    r"secondary:[^}]*status:\s*'(brn|par|slp|psn|tox|frz)'": "tag:status_* (secondary)",
    r"volatileStatus:\s*'confusion'|secondary:[^}]*volatileStatus:\s*'confusion'": "tag:status_confuse",
    r"secondary:[^}]*volatileStatus:\s*'flinch'|\bvolatileStatus:\s*'flinch'": "tag:flinch",
    r"sideCondition:\s*'(spikes|stealthrock|toxicspikes|stickyweb|gmaxsteelsurge)'": "tag:hazard_set",
    r"sideCondition:\s*'tailwind'": "tag:speed_control",
    r"sideCondition:\s*'(reflect|lightscreen|auroraveil|safeguard|mist)'": "tag:screens (->global side-cond)",
    r"sideCondition:\s*'(wideguard|quickguard|matblock|craftyshield)'": "embed:doubles-protect",
    r"pseudoWeather:\s*'(trickroom|magicroom|wonderroom|gravity)'": "tag/global:field-room",
    r"\bweather:\s*'": "global:weather-setter-move",
    r"\bterrain:\s*'": "global:terrain-setter-move",
    r"\bheal:\s*\[|\bheal:\s*function|\bonHit:[^}]*\.heal\(": "tag:heal",
    r"\bdrain:\s*\[": "tag:heal (drain)",
    r"\bohko:": "embed:ohko",
    r"\bmultihit:": "scalar:multihit (existing)",
    r"\brecoil:\s*\[|hasCrashDamage:": "scalar:recoil (existing)+recoil_endure",
    r"volatileStatus:\s*'(taunt|encore|disable|torment|healblock|imprison)'": "tag:move_restrict",
    r"volatileStatus:\s*'(leechseed|partiallytrapped|saltcure|nightmare|curse|yawn|perishsong)'": "tag:residual_chip",
    r"volatileStatus:\s*'substitute'|\bvolatileStatus:\s*'substitute'": "volatile:has_substitute",
    r"\bonModifyMove": "embed/callback:onModifyMove",
    r"\bstealsBoosts:|\bonHit:[^}]*boosts": "embed:boost-steal/copy",
    r"slotCondition:\s*'(healingwish|lunardance|wish)'": "embed:slot-heal",
    r"\bselfdestruct:": "embed:selfdestruct",
    r"\bforceSwitch:": "tag:force_switch",
    r"priority:\s*-?\d": "scalar:priority (existing)",
    r"\baccuracy:\s*(true|\d)": "scalar:accuracy (existing)",
}

def main() -> int:
    moves = list(parse_blocks(DATA / "moves.ts"))
    tally = Counter()
    covered_moves = set()
    for mid, block in moves:
        for rx, cov in MOVE_AXES.items():
            if re.search(rx, block):
                tally[cov] += 1
                covered_moves.add(mid)

    # moves with NO recognised mechanic axis (pure damaging move, fine -> covered by scalars+embedding)
    status_or_special = [mid for mid, b in moves
                         if re.search(r'category:\s*"Status"', b) and mid not in covered_moves]

    print(f"== MOVES ==  parsed {len(moves)} entries")
    for cov, n in sorted(tally.items(), key=lambda x: -x[1]):
        print(f"  {n:4d}  {cov}")
    print(f"\n  STATUS moves with no recognised mechanic axis (review for a missing tag): {len(status_or_special)}")
    print("   ", ", ".join(sorted(status_or_special)[:60]))

    # ── ABILITIES: tally the callback hooks (mechanic surface) ────────────────────────
    abil = list(parse_blocks(DATA / "abilities.ts"))
    champ = list(parse_blocks(DATA / "mods" / "champions" / "abilities.ts")) if (DATA / "mods" / "champions" / "abilities.ts").exists() else []
    hook = Counter()
    for _id, b in abil:
        for h in re.findall(r"\bon[A-Z]\w+", b):
            hook[h] += 1
    print(f"\n== ABILITIES ==  base {len(abil)} entries, champions-mod {len(champ)} entries")
    print("  top callback hooks (mechanic surface):")
    for h, n in hook.most_common(30):
        print(f"  {n:4d}  {h}")
    trap = [i for i, b in abil if re.search(r"onFoeTrapPokemon|onTrapPokemon", b)]
    mtype = [i for i, b in abil if "onModifyType" in b]
    print(f"  trapping abilities (onFoeTrapPokemon): {trap}")
    print(f"  move-type-changing abilities (onModifyType): {mtype}")

    # ── ITEMS ────────────────────────────────────────────────────────────────────────
    items = list(parse_blocks(DATA / "items.ts"))
    mega = [i for i, b in items if "megaStone" in b or "megaEvolves" in b]
    isq = Counter()
    for _id, b in items:
        for h in re.findall(r"\bon[A-Z]\w+", b):
            isq[h] += 1
    print(f"\n== ITEMS ==  parsed {len(items)} entries  (mega stones: {len(mega)})")
    print("  top item callback hooks:")
    for h, n in isq.most_common(20):
        print(f"  {n:4d}  {h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
