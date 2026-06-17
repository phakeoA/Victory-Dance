"""Verify mega labels in the EXPORTED JSONL corpus (post re-export).

Checks, over all Type B + Type A transitions:
  * mega=True appears ONLY on move actions (never switch).
  * every mega=True move carries a non-null action_index (=> a usable label).
  * dual-perspective symmetry: #(mega in our_actions) == #(mega in opp_actions).
  * mega labels actually exist (the blocker is cleared: was 0 before).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
SCRIPTS = HERE.parents[1]
REPO = SCRIPTS.parents[1]
PREP = REPO / "data" / "vods" / "Prepared_training_data" / "Regulation_MA"

groups = {
    "TypeB (dual-perspective)": [PREP / "Jsonl_TypeB"],
    "TypeA (single-perspective)": [PREP / "Jsonl_TypeA" / f"Kronomono{n}" for n in (1, 2, 3)],
}

mega_on_switch = mega_no_index = 0
typeb_our = typeb_opp = 0
for label, ds in groups.items():
    g_our = g_opp = g_files = 0
    for d in ds:
        for fp in sorted(d.glob("*.jsonl")):
            g_files += 1
            for line in fp.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                t = json.loads(line)
                for act in t.get("our_actions") or []:
                    if act.get("mega"):
                        g_our += 1
                        if act.get("action") != "move":
                            mega_on_switch += 1
                        if act.get("action_index") is None:
                            mega_no_index += 1
                for act in t.get("opp_actions_actual") or []:
                    if act.get("mega"):
                        g_opp += 1
                        if act.get("action") != "move":
                            mega_on_switch += 1
    print(f"{label}: files={g_files}  mega(our)={g_our}  mega(opp)={g_opp}")
    if label.startswith("TypeB"):
        typeb_our, typeb_opp = g_our, g_opp

print(f"mega=True on a SWITCH action : {mega_on_switch}  (must be 0)")
print(f"mega move w/ null index      : {mega_no_index}  (must be 0)")

ok = (
    typeb_our > 0
    and typeb_our == typeb_opp     # Type B is dual-perspective => exact symmetry
    and mega_on_switch == 0
    and mega_no_index == 0
)
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
