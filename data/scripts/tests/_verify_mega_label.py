"""Adversarial verification of the parser mega-label join (Task #1).

Parses real replays and asserts that EVERY same-turn/same-slot mega_evolution
event is reflected as a mega=True flag on the matching move action, and that no
spurious mega flags appear.  Run:

    .venv/Scripts/python.exe data/scripts/tests/_verify_mega_label.py [N]
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
SCRIPTS = HERE.parents[1]            # data/scripts
REPO = SCRIPTS.parents[1]            # repo root
sys.path.insert(0, str(SCRIPTS))

from vod_parser.replay_parser import ShowdownReplayParser, extract_log_from_html  # noqa: E402

REPLAY_DIR = REPO / "data" / "vods" / "Type_B" / "gen9championsvgc2026regma"


def analyse(html_path: Path):
    log = extract_log_from_html(html_path.read_text(encoding="utf-8"))
    result = ShowdownReplayParser(log, our_player="p1").parse()

    mega_events = 0          # mega_evolution events in the raw action log
    mega_with_move = 0       # of those, a same-turn same-slot move exists
    mega_flags = 0           # move actions stamped mega=True
    forme_flags = 0          # sanity: forme_change must NEVER be stamped
    violations = []

    for turn in result.get("turns", []):
        # raw events for this turn
        evs = turn.get("actions", [])
        mega_slots = {e.get("slot") for e in evs if e.get("event") == "mega_evolution"}
        forme_slots = {e.get("slot") for e in evs if e.get("event") == "forme_change"}
        move_slots = {e.get("user_slot") for e in evs if e.get("event") == "move"}
        mega_events += len(mega_slots)
        mega_with_move += len(mega_slots & move_slots)

        # decision-level actions (both sides; these carry the join)
        for side_key in ("our_actions", "opp_actions_actual"):
            for act in turn.get(side_key, []):
                if act.get("action") != "move":
                    if act.get("mega"):
                        violations.append((turn["turn"], side_key, "non-move had mega", act))
                    continue
                slot = act.get("slot")
                if act.get("mega"):
                    mega_flags += 1
                    # every flag MUST correspond to a real mega_evolution on that slot
                    if slot not in mega_slots:
                        violations.append((turn["turn"], side_key, "flag without mega event", act))
                    if slot in forme_slots and slot not in mega_slots:
                        forme_flags += 1
                else:
                    # a move on a slot that mega'd this turn MUST be flagged
                    if slot in mega_slots:
                        violations.append((turn["turn"], side_key, "mega event but move unflagged", act))

    return {
        "mega_events": mega_events,
        "mega_with_move": mega_with_move,
        "mega_flags": mega_flags,
        "forme_flags": forme_flags,
        "violations": violations,
    }


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    files = sorted(REPLAY_DIR.glob("*.html"))[:n]
    tot = {"mega_events": 0, "mega_with_move": 0, "mega_flags": 0, "forme_flags": 0}
    all_viol = []
    parsed = 0
    for f in files:
        try:
            r = analyse(f)
        except Exception as e:  # noqa: BLE001
            print(f"  PARSE ERROR {f.name}: {e}")
            continue
        parsed += 1
        for k in tot:
            tot[k] += r[k]
        for v in r["violations"]:
            all_viol.append((f.name, *v))

    print(f"Parsed {parsed}/{len(files)} replays")
    print(f"  mega_evolution events          : {tot['mega_events']}")
    print(f"  ...with a same-slot move        : {tot['mega_with_move']}")
    print(f"  move actions flagged mega=True  : {tot['mega_flags']}")
    print(f"  forme_change wrongly flagged    : {tot['forme_flags']}")
    print(f"  violations                      : {len(all_viol)}")
    for v in all_viol[:20]:
        print("   VIOLATION:", v)

    ok = (
        len(all_viol) == 0
        and tot["forme_flags"] == 0
        and tot["mega_flags"] == tot["mega_with_move"]
        and tot["mega_flags"] > 0
    )
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
