"""Real-replay corpus check of transitions.py gimmick wiring (Task #3).

Runs the full replay_to_transitions over real replays (belief=None) and asserts
the structural gimmick invariants on BOTH turn and replacement transitions:
  * every transition carries a gimmick_mask with our_a/our_b rows of length 2,
  * replacement transitions are switch-only: deciding slot == [1,0], the switch
    never gimmicks (gimmick_index in {0, None}, never 1), ally slot == [0,0],
  * any gimmick_index==1 sits on a mega-flagged move and is legal under the mask,
  * gimmick_index is always one of {0, 1, None}.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
SCRIPTS = HERE.parents[1]
sys.path.insert(0, str(SCRIPTS))
from vod_parser.transitions import replay_to_transitions  # noqa: E402

REPLAY_DIR = SCRIPTS.parent / "vods" / "Type_B" / "gen9championsvgc2026regma"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
files = sorted(REPLAY_DIR.glob("*.html"))[:N]

turn_tx = repl_tx = 0
bad_mask = bad_repl = bad_val = bad_legal = mega1 = 0
parsed = 0
for fp in files:
    try:
        tr = replay_to_transitions(fp, belief=None, players=["p1", "p2"], source_type="B")
    except Exception as e:  # noqa: BLE001
        print("PARSE ERROR", fp.name, e)
        continue
    parsed += 1
    for t in tr:
        gm = t.get("gimmick_mask")
        if not gm or set(gm) != {"our_a", "our_b"} or any(len(r) != 2 for r in gm.values()):
            bad_mask += 1
            continue
        if t["decision_type"] == "replacement":
            repl_tx += 1
            slot = t["our_actions"][0]["slot"] if t["our_actions"] else None
            ally = "our_b" if slot == "our_a" else "our_a"
            if gm.get(slot) != [1, 0] or gm.get(ally) != [0, 0]:
                bad_repl += 1
            for a in t["our_actions"]:
                if a.get("gimmick_index") == 1:
                    bad_repl += 1
        else:
            turn_tx += 1
        for a in t.get("our_actions") or []:
            gi = a.get("gimmick_index")
            if gi not in (0, 1, None):
                bad_val += 1
            if gi == 1:
                mega1 += 1
                if not a.get("mega") or gm.get(a.get("slot"), [0, 0])[1] != 1:
                    bad_legal += 1

print(f"parsed replays          : {parsed}/{len(files)}")
print(f"turn transitions        : {turn_tx}")
print(f"replacement transitions : {repl_tx}")
print(f"gimmick_index==1 (mega) : {mega1}")
print(f"bad gimmick_mask        : {bad_mask}  (must be 0)")
print(f"bad replacement mask    : {bad_repl}  (must be 0)")
print(f"bad gimmick value       : {bad_val}  (must be 0)")
print(f"mega1 not real/legal    : {bad_legal}  (must be 0)")

ok = bad_mask == bad_repl == bad_val == bad_legal == 0 and repl_tx > 0 and mega1 > 0
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
