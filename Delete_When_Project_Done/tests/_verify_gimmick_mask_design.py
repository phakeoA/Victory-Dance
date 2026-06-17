"""Validate the build_gimmick_mask DESIGN before implementing it.

For EVERY mega=True label in the exported corpus, the proposed mask must mark
bucket 1 (mega) LEGAL, else annotate_transition_actions would null the label.
Proposed legality:  mega-capable(base_species)  AND  no own mon is_mega yet.

Reports any label that would be marked illegal under this definition.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
SCRIPTS = HERE.parents[1]
sys.path.insert(0, str(SCRIPTS))
from vod_parser.pokedex import get_pokedex  # noqa: E402

PREP = SCRIPTS.parent / "vods" / "Prepared_training_data" / "Regulation_MA"
dirs = [PREP / "Jsonl_TypeB"] + [PREP / "Jsonl_TypeA" / f"Kronomono{n}" for n in (1, 2, 3)]

dex = get_pokedex()


def mega_capable(mon):
    base = mon.get("base_species") or mon.get("species")
    return bool(dex.mega_formes_for(base)) if dex else False


def own_already_megaed(snap):
    mons = list((snap.get("our_active") or {}).values()) + list(snap.get("our_bench") or [])
    return any(m.get("is_mega") for m in mons)


labels = legal = 0
fail_capable = []
fail_teammate = []
for d in dirs:
    for fp in sorted(d.glob("*.jsonl")):
        for line in fp.read_text(encoding="utf-8").splitlines():
            if '"mega": true' not in line:
                continue
            t = json.loads(line)
            snap = t.get("state_before_actions") or {}
            actives = snap.get("our_active") or {}
            team_megaed = own_already_megaed(snap)
            for act in t.get("our_actions") or []:
                if not act.get("mega"):
                    continue
                labels += 1
                mon = actives.get(act.get("slot")) or {}
                cap = mega_capable(mon)
                if not cap:
                    fail_capable.append((fp.name, t.get("turn"), mon.get("base_species")))
                if team_megaed:
                    fail_teammate.append((fp.name, t.get("turn"), act.get("slot")))
                if cap and not team_megaed:
                    legal += 1

print(f"mega labels                         : {labels}")
print(f"legal under proposed mask           : {legal}")
print(f"FAIL not-mega-capable               : {len(fail_capable)}")
print(f"FAIL teammate-already-megaed        : {len(fail_teammate)}")
for f in fail_capable[:10]:
    print("   not-capable:", f)
for f in fail_teammate[:10]:
    print("   teammate:", f)
ok = labels > 0 and legal == labels
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
