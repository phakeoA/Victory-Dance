"""Gap #6 probe — compare LIVE vs OFFLINE opp active/bench COMPOSITION per turn.

For a replay + perspective, prints per turn:
  - offline opp_active (slot->species) + opp_bench (ordered species/seen)
  - live opp active (poke-env) + live opp bench (LiveStateEncoder._opp_bench_mons)
and dumps the raw |switch|/|drag|/|faint|/|replace| protocol lines for the opp
side so we can decide which path is CORRECT.

Run: PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe data/scripts/tests/_gap6_probe.py <fname-substr> <persp>
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "data" / "scripts"))
sys.path.insert(0, str(ROOT / "data" / "scripts" / "tests"))

import _parity_harness as H  # noqa: E402
from vod_parser.replay_parser import ShowdownReplayParser, extract_log_from_html  # noqa: E402
from live_state_encoder import LiveStateEncoder  # noqa: E402
from poke_env.battle import DoubleBattle  # noqa: E402
import logging  # noqa: E402

_LOG = logging.getLogger("g6probe"); _LOG.disabled = True
_SKIP = {"t:", "uhtmlchange", "win", "tie"}


def _bs(mon):
    return (getattr(mon, "base_species", None) or getattr(mon, "species", "") or "")


def offline_per_turn(log, persp):
    parser = ShowdownReplayParser(log, our_player=persp)
    res = parser.parse()
    out = {}
    for turn in res["turns"]:
        snap = turn["state_before_actions"][persp]
        oa = {k: v["species"] for k, v in snap["opp_active"].items()}
        ob = [(m["species"], m.get("seen"), m.get("is_fainted")) for m in snap["opp_bench"]]
        out[turn["turn"]] = (oa, ob)
    return out


def live_per_turn(log, user):
    enc = LiveStateEncoder()
    battle = DoubleBattle("g6", user, _LOG, gen=9)
    out = {}
    for ln in log.split("\n"):
        if not ln.startswith("|"):
            continue
        parts = ln.split("|")
        if len(parts) < 2 or parts[1] in _SKIP:
            continue
        try:
            battle.parse_message(parts)
        except Exception:
            pass
        if parts[1] == "turn":
            try:
                oa = battle.opponent_active_pokemon
            except ValueError:
                oa = [None, None]
            oa_set = {p for p in oa if p is not None}
            bench = enc._opp_bench_mons(battle, oa_set)
            out[int(parts[2])] = (
                [(_bs(p) if p else None, p.fainted if p else None) for p in oa],
                [(_bs(p), p.fainted, p.revealed) for p in bench],
            )
    return out


def main():
    sub = sys.argv[1] if len(sys.argv) > 1 else "orangeomar"
    persp = sys.argv[2] if len(sys.argv) > 2 else "p1"
    hit = next((ROOT / "data" / "vods").rglob(f"*{sub}*.html"), None)
    if hit is None:
        print("no replay matching", sub); return
    log = extract_log_from_html(hit.read_text(encoding="utf-8"))
    user = H.player_usernames(log)[persp]
    opp = "p2" if persp == "p1" else "p1"
    print(f"== {hit.name}  persp={persp}  opp={opp}  user={user} ==\n")

    print("--- raw opp protocol (switch/drag/faint/replace) ---")
    tn = 0
    for ln in log.split("\n"):
        if ln.startswith("|turn|"):
            tn = int(ln.split("|")[2])
        parts = ln.split("|")
        if len(parts) < 3:
            continue
        if parts[1] in ("switch", "drag", "faint", "replace", "-transform"):
            ident = parts[2]
            if ident.startswith(opp):
                extra = parts[3] if len(parts) > 3 else ""
                print(f"  t{tn:<2} |{parts[1]}|{ident}|{extra}")

    off = offline_per_turn(log, persp)
    live = live_per_turn(log, user)
    print("\n--- per-turn opp composition (OFFLINE vs LIVE) ---")
    for t in sorted(set(off) | set(live)):
        oa, ob = off.get(t, ({}, []))
        la, lb = live.get(t, ([], []))
        print(f"\n t{t}")
        print(f"   OFF active: {oa}")
        print(f"   LIVE active: {la}")
        print(f"   OFF bench : {ob}")
        print(f"   LIVE bench: {lb}")


if __name__ == "__main__":
    main()
