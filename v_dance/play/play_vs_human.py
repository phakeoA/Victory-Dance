"""Human-vs-AI battle — play the PRODUCTION AI yourself instead of watching it on the ladder.

A MEASUREMENT harness: the AI (whatever ``model_io`` serves as production — currently the gen141 battle
net + SBDA team-preview) accepts YOUR challenge on the local Showdown server, so you can judge it by
playing it. Two team modes (both draw from ``teams/Champions/<reg>``, never the SAME team on both sides):

  --mode random          you + the AI each get a RANDOM DIFFERENT team from the pool.
  --mode choose          you pick BOTH teams individually:
                         --ai-team NAME --human-team NAME   (must differ).

Usage (from the repo root):
  python -m v_dance.play.play_vs_human --mode random                 # serve until you press Ctrl-C
  python -m v_dance.play.play_vs_human --mode choose --ai-team WolfeGlick --human-team maw_zard
  python -m v_dance.play.play_vs_human --list-teams        # show the pool, then exit
  python -m v_dance.play.play_vs_human --mode random --n-battles 3   # auto-stop after 3 finished battles

By DEFAULT the local Showdown server stays up until you press Ctrl-C, so you can rematch as many times as
you like and the browser can finish its end-of-battle animation; ``--n-battles N`` auto-stops after N. The
local client opens in your browser automatically. Follow the printed steps (import YOUR team, challenge the AI).
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import random
import sys
import webbrowser
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


async def _serve(ai, human_name: str | None, n_battles: int | None) -> None:
    """Accept challenges ONE battle at a time until Ctrl-C (``n_battles=None``, the default) or until
    ``n_battles`` have finished. Accepting one at a time means we loop back to *waiting* after every
    battle, so the server stays up between matches — you can rematch, and the browser finishes its
    end-of-battle animation while we wait — instead of the old behaviour that tore the server down the
    instant the first result was decided.

    Each accept is driven as a short ``sleep`` loop, not a bare ``await``: poke-env runs its websocket
    I/O on the background ``POKE_LOOP`` thread, so THIS (main) loop has no timers of its own and a bare
    ``await`` would idle ``select()`` with no timeout — and on Windows a console Ctrl-C is NOT delivered
    to a parked ``select()`` (the wedge the user hit). The periodic wakeup makes the SIGINT land. The
    always-on ps_client listener queues any challenge that arrives between iterations, so none is missed."""
    accepted = 0
    while n_battles is None or accepted < n_battles:
        serve = asyncio.ensure_future(ai.accept_challenges(human_name, 1))
        try:
            while not serve.done():
                await asyncio.sleep(0.25)
        finally:
            if not serve.done():
                serve.cancel()                              # Ctrl-C / cancellation -> stop accepting
        serve.result()                                      # re-raise any real error from accept_challenges
        accepted += 1
        if n_battles is None or accepted < n_battles:
            print(f"  [battle {accepted} done]  tally — AI {ai.n_won_battles} / you {ai.n_lost_battles} "
                  f"/ draws {ai.n_tied_battles}.  Challenge {AI_NAME} again to keep playing, or Ctrl-C to stop.")
            sys.stdout.flush()


async def _run(ai_team_str: str, ai_team_name: str, human_team_str: str, human_team_name: str,
               human_name: str | None, n_battles: int | None, url: str) -> None:
    ai = make_player(AI_NAME, ai_team_str)             # production checkpoints via model_io defaults
    print("\n" + "=" * 70)
    print("  HUMAN  vs  AI   (you pilot a real team against the production bot)")
    print("=" * 70)
    print(f"  AI ({AI_NAME}) team : {ai_team_name}")
    print(f"  YOUR team          : {human_team_name}")
    print(f"  format             : {BATTLE_FORMAT}")
    print(f"  battles to play    : {n_battles}")
    print("-" * 70)
    print("  STEPS (the local client opens in your browser automatically):")
    print(f"   1. Client opening at  {url}   (if it doesn't, open that URL yourself)")
    print(f"   2. Pick a username{f' — use exactly: {human_name}' if human_name else ' (any name; no password)'} and log in.")
    print("   3. Teambuilder -> Import -> paste YOUR team (printed below) -> Save.")
    print(f"   4. Search the user  {AI_NAME}  -> Challenge.")
    print(f"   5. In the dialog: format = {BATTLE_FORMAT}, pick your imported team -> Challenge.")
    print("   6. The AI auto-accepts. Play! (repeat the challenge for each battle in a series.)")
    print("-" * 70)
    print("  --- COPY YOUR TEAM BELOW INTO THE TEAMBUILDER (Import) ---\n")
    print(human_team_str.strip())
    print("\n" + "-" * 70)
    print(f"  Waiting for your challenge to {AI_NAME} ...  (press Ctrl-C to stop)\n")
    sys.stdout.flush()                                  # show the instructions before the long wait
    try:
        webbrowser.open(url)                            # auto-open the local client
    except Exception:                                   # headless / no browser -> the printed URL still works
        pass
    won0, lost0, tied0 = ai.n_won_battles, ai.n_lost_battles, ai.n_tied_battles
    try:
        await _serve(ai, human_name, n_battles)         # human_name=None -> accept from anyone
    finally:
        # Print the tally on ANY exit — a clean end OR Ctrl-C (the unlimited default only returns via
        # cancellation), so you always see the session result before the server is torn down.
        ai_w = ai.n_won_battles - won0
        you_w = ai.n_lost_battles - lost0  # YOUR wins = the AI's losses — NOT (n_battles - ai_w),
        draws = ai.n_tied_battles - tied0  # which would wrongly credit DRAWS (a real VGC outcome) to you
        print("\n" + "=" * 70)
        print(f"  SESSION RESULT  —  AI {ai_w} / you {you_w} / draws {draws}  (of {ai_w + you_w + draws} finished)")
        print("=" * 70 + "\n")


def _parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Play the production AI yourself (human vs AI, local Showdown).")
    ap.add_argument("--mode", choices=["random", "choose"], default="random",
                    help="random = both get a random different pool team; choose = pick both yourself.")
    ap.add_argument("--ai-team", default=None, help="(choose mode) team NAME for the AI side.")
    ap.add_argument("--human-team", default=None, help="(choose mode) team NAME for YOUR side (must differ).")
    ap.add_argument("--human-name", default=None,
                    help="restrict the AI to accept challenges ONLY from this Showdown name "
                         "(default: accept from anyone — simplest on a local server).")
    ap.add_argument("--n-battles", type=int, default=None,
                    help="stop after this many FINISHED battles (default: keep the server up and accept "
                         "challenges until you press Ctrl-C).")
    ap.add_argument("--list-teams", action="store_true", help="print the team pool and exit.")
    return ap.parse_args(argv)


def main() -> None:
    args = _parse_args()

    if args.list_teams:
        pool = discover_teams(reg=BATTLE_FORMAT)
        print(f"[play] {len(pool)} teams in the {BATTLE_FORMAT} pool:")
        for p in pool:
            print(f"   {Path(p).name}")
        return

    ai_name, human_name_team = _pick_teams(args.mode, args.ai_team, args.human_team)
    ai_team_str = load_team(resolve_team_path(ai_name))
    human_team_str = load_team(resolve_team_path(human_name_team))
    url = f"http://{SHOWDOWN_HOST}:{SHOWDOWN_PORT}"

    # Server lifecycle lives HERE (main thread) so the tree-kill ALWAYS runs — including on a
    # Ctrl-C, which now unwinds cleanly (see _serve) instead of wedging the terminal.
    proc = start_showdown()
    try:
        asyncio.run(_run(ai_team_str, Path(ai_name).name, human_team_str, Path(human_name_team).name,
                         args.human_name, args.n_battles, url))
    except KeyboardInterrupt:
        print("\n[play] Ctrl-C received — stopping the AI and the Showdown server …")
    finally:
        # Killing the server resets the AI's still-open websocket; poke-env logs that as a noisy
        # ERROR + traceback on shutdown. Quiet those loggers first — the session is already over.
        logging.getLogger(AI_NAME).setLevel(logging.CRITICAL)
        logging.getLogger("poke_env").setLevel(logging.CRITICAL)
        stop_showdown(proc)
        print("[play] Showdown server stopped. Bye.")


if __name__ == "__main__":
    main()
