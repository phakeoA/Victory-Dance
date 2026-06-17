"""End-to-end corpus check of the IMPLEMENTED gimmick annotate (Task #2).

Re-runs state_encoder.annotate_transition_actions on every exported transition
and asserts:
  * every mega=True our_action ends up with gimmick_index == 1 (none dropped),
  * no non-mega our_action gets gimmick_index == 1,
  * every gimmick_index==1 is legal under the transition's gimmick_mask,
  * gimmick_mask rows are all length 2.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
SCRIPTS = HERE.parents[1]
sys.path.insert(0, str(SCRIPTS))
import state_encoder as se  # noqa: E402

PREP = SCRIPTS.parent / "vods" / "Prepared_training_data" / "Regulation_MA"
dirs = [PREP / "Jsonl_TypeB"] + [PREP / "Jsonl_TypeA" / f"Kronomono{n}" for n in (1, 2, 3)]

mega_labels = mega_kept = mega_dropped = 0
false_mega = 0          # non-mega action stamped gimmick 1
illegal_gimmick = 0     # gimmick_index 1 but mask says illegal
bad_mask_len = 0
tx = 0
for d in dirs:
    for fp in sorted(d.glob("*.jsonl")):
        for line in fp.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            t = json.loads(line)
            tx += 1
            se.annotate_transition_actions(t)
            gm = t.get("gimmick_mask") or {}
            for row in gm.values():
                if len(row) != 2:
                    bad_mask_len += 1
            for act in t.get("our_actions") or []:
                gi = act.get("gimmick_index")
                is_mega = bool(act.get("mega"))
                if is_mega:
                    mega_labels += 1
                    if gi == 1:
                        mega_kept += 1
                    else:
                        mega_dropped += 1
                else:
                    if gi == 1:
                        false_mega += 1
                if gi == 1:
                    row = gm.get(act.get("slot")) or []
                    if not (len(row) > 1 and row[1] == 1):
                        illegal_gimmick += 1

print(f"transitions re-annotated     : {tx}")
print(f"mega labels                  : {mega_labels}")
print(f"  kept   (gimmick_index==1)  : {mega_kept}")
print(f"  dropped(gimmick_index!=1)  : {mega_dropped}  (must be 0)")
print(f"false mega (non-mega -> 1)   : {false_mega}  (must be 0)")
print(f"illegal gimmick under mask   : {illegal_gimmick}  (must be 0)")
print(f"gimmick_mask rows != len 2   : {bad_mask_len}  (must be 0)")

ok = (mega_labels > 0 and mega_dropped == 0 and false_mega == 0
      and illegal_gimmick == 0 and bad_mask_len == 0)
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
