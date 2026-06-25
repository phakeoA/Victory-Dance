"""Human-vs-AI battle — play the PRODUCTION AI yourself instead of watching it on the ladder.

A MEASUREMENT harness: the AI (whatever ``model_io`` serves as production — currently the gen141 battle
net + SBDA team-preview) accepts YOUR challenge on the local Showdown server, so you can judge it by
playing it. Two team modes (both draw from ``teams/Champions/<reg>``, never the SAME team on both sides):

  --mode random          you + the AI each get a RANDOM DIFFERENT team from the pool.
  --mode choose          you pick BOTH teams individually:
                         --ai-team NAME --human-team NAME   (must differ).

Usage (from the repo root):
  python -m v_dance.play.play_vs_human --mode random
  python -m v_dance.play.play_vs_human --mode choose --ai-team WolfeGlick --human-team maw_zard
  python -m v_dance.play.play_vs_human --list-teams        # show the pool, then exit
  python -m v_dance.play.play_vs_human --mode random --n-battles 3   # play a best-of / series

Then follow the printed steps in your browser (open the local client, import YOUR team, challenge the AI).
"""
from __future__ import annotations

import argparse
import asyncio
import random
import sys
from pathlib import Path

from v_dance.play.run_local_battle import (
    BATTLE_FORMAT, SHOWDOWN_HOST, SHOWDOWN_PORT,
    discover_teams, load_team, make_player, resolve_team_path,
    start_showdown, stop_showdown,
)

AI_NAME = "VictoryDanceAI"     # the username you challenge in the client


def _pick_teams(mode: str, ai_team: str | None, human_team: str | None) -> tuple[str, str]:
    """Resolve (ai_team_name, human_team_name) for the chosen mode. Always two DIFFERENT pool teams."""
    pool = discover_teams(reg=BATTLE_FORMAT)            # repo-relative paths, the active reg's subfolder
    if mode == "random":
        if len(pool) < 2:
            raise SystemExit(f"[play] need >=2 teams in the {BATTLE_FORMAT} pool to draw two different "
                             f"ones (found {len(pool)}).")
        a, h = random.sample(pool, 2)                  # 2 distinct teams, never the same on both sides
        return a, h
    # choose
    if not (ai_team and human_team):
        raise SystemExit("[play] --mode choose needs BOTH --ai-team NAME and --human-team NAME "
                         "(run --list-teams to see the pool).")
    if Path(ai_team).name == Path(human_team).name:
        raise SystemExit("[play] --ai-team and --human-team must be DIFFERENT teams.")
    return ai_team, human_team


async def _run(ai_team_str: str, ai_team_name: str, human_team_str: str, human_team_name: str,
               human_name: str | None, n_battles: int) -> None:
    proc = start_showdown()
    ai = make_player(AI_NAME, ai_team_str)             # production checkpoints via model_io defaults
    url = f"http://{SHOWDOWN_HOST}:{SHOWDOWN_PORT}"
    try:
        print("\n" + "=" * 70)
        print("  HUMAN  vs  AI   (you pilot a real team against the production bot)")
        print("=" * 70)
        print(f"  AI ({AI_NAME}) team : {ai_team_name}")
        print(f"  YOUR team          : {human_team_name}")
        print(f"  format             : {BATTLE_FORMAT}")
        print(f"  battles to play    : {n_battles}")
        print("-" * 70)
        print("  STEPS (in your browser):")
        print(f"   1. Open  {url}")
        print(f"   2. Pick a username{f' — use exactly: {human_name}' if human_name else ' (any name; no password)'} and log in.")
        print("   3. Teambuilder -> Import -> paste YOUR team (printed below) -> Save.")
        print(f"   4. Search the user  {AI_NAME}  -> Challenge.")
        print(f"   5. In the dialog: format = {BATTLE_FORMAT}, pick your imported team -> Challenge.")
        print("   6. The AI auto-accepts. Play! (repeat the challenge for each battle in a series.)")
        print("-" * 70)
        print("  --- COPY YOUR TEAM BELOW INTO THE TEAMBUILDER (Import) ---\n")
        print(human_team_str.strip())
        print("\n" + "-" * 70)
        print(f"  Waiting for your challenge to {AI_NAME} ...  (Ctrl-C to stop)\n")
        sys.stdout.flush()                                  # show the instructions before the long wait
        won0, lost0, tied0 = ai.n_won_battles, ai.n_lost_battles, ai.n_tied_battles
        await ai.accept_challenges(human_name, n_battles)   # human_name=None -> accept from anyone
        ai_w = ai.n_won_battles - won0
        you_w = ai.n_lost_battles - lost0      # YOUR wins = the AI's losses — NOT (n_battles - ai_w),
        draws = ai.n_tied_battles - tied0      # which would wrongly credit DRAWS (a real VGC outcome) to you
        print("\n" + "=" * 70)
        print(f"  SESSION RESULT  —  AI {ai_w} / you {you_w} / draws {draws}  (of {ai_w + you_w + draws} finished)")
        print("=" * 70 + "\n")
    finally:
        stop_showdown(proc)


def main() -> None:
    ap = argparse.ArgumentParser(description="Play the production AI yourself (human vs AI, local Showdown).")
    ap.add_argument("--mode", choices=["random", "choose"], default="random",
                    help="random = both get a random different pool team; choose = pick both yourself.")
    ap.add_argument("--ai-team", default=None, help="(choose mode) team NAME for the AI side.")
    ap.add_argument("--human-team", default=None, help="(choose mode) team NAME for YOUR side (must differ).")
    ap.add_argument("--human-name", default=None,
                    help="restrict the AI to accept challenges ONLY from this Showdown name "
                         "(default: accept from anyone — simplest on a local server).")
    ap.add_argument("--n-battles", type=int, default=1, help="how many challenges to accept this session.")
    ap.add_argument("--list-teams", action="store_true", help="print the team pool and exit.")
    args = ap.parse_args()

    if args.list_teams:
        pool = discover_teams(reg=BATTLE_FORMAT)
        print(f"[play] {len(pool)} teams in the {BATTLE_FORMAT} pool:")
        for p in pool:
            print(f"   {Path(p).name}")
        return

    ai_name, human_name_team = _pick_teams(args.mode, args.ai_team, args.human_team)
    ai_team_str = load_team(resolve_team_path(ai_name))
    human_team_str = load_team(resolve_team_path(human_name_team))
    asyncio.run(_run(ai_team_str, Path(ai_name).name, human_team_str, Path(human_name_team).name,
                     args.human_name, args.n_battles))


if __name__ == "__main__":
    main()
