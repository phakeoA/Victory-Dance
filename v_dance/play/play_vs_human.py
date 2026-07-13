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
import json
import logging
import os
import random
import sys
import time
import webbrowser
from pathlib import Path

from v_dance.play.model_io import DEFAULT_BC_CHECKPOINT, DEFAULT_TP_CHECKPOINT
from v_dance.play.run_local_battle import (
    BATTLE_FORMAT, SHOWDOWN_HOST, SHOWDOWN_PORT,
    discover_teams, load_team, make_player, resolve_team_path,
    start_showdown, stop_showdown,
)

AI_NAME = "VictoryDanceAI"     # the username you challenge in the client

# Human-benchmark recording (docs/human_benchmark_design.md): one JSONL row per finished battle,
# appended+flushed immediately so a Ctrl-C mid-session loses nothing already played.
BENCH_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "human_benchmark"
BENCH_LOG = BENCH_DIR / "human_bench.jsonl"


class _BenchLog:
    """Append one row per finished battle; result from the player's counter deltas (draw-correct)."""

    def __init__(self, session_id: str, note: str, ai_team: str, human_team: str,
                 ckpt: Path, tp_ckpt: Path):
        self.session_id, self.note = session_id, note
        self.ai_team, self.human_team = ai_team, human_team
        self.ckpt, self.tp_ckpt = str(ckpt), str(tp_ckpt)
        self.game_idx = 0
        self._seen: set = set()                       # battle tags already logged
        BENCH_DIR.mkdir(parents=True, exist_ok=True)

    def record(self, ai, result: str) -> None:
        """Log the battle that just finished. ``result`` ∈ ai|human|draw (from counter deltas);
        tag/turns/opponent come from the newest finished battle object not yet seen."""
        tag, turns, opp = None, None, None
        try:
            fresh = [b for t, b in ai.battles.items() if b.finished and t not in self._seen]
            if fresh:
                b = fresh[-1]                          # one battle at a time → at most one fresh
                tag = b.battle_tag
                turns = getattr(b, "turn", None)
                opp = getattr(b, "opponent_username", None)
                self._seen.add(tag.lstrip(">") if tag else tag)
                self._seen.add(tag)
                # B-L2 dossier capture (passive; never raises) + a summary line for the USER.
                from v_dance.play.opponent_dossier import summary, update_from_battle
                if update_from_battle(b, result, our_team=self.ai_team, note=self.note):
                    print(f"  [dossier] {summary(opp)}")
        except Exception:
            pass                                       # a row with result-only still beats no row
        self.game_idx += 1
        row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "session_id": self.session_id, "note": self.note, "game_idx": self.game_idx,
               "battle_tag": tag, "ai_team": self.ai_team, "human_team": self.human_team,
               "opponent": opp, "result": result, "turns": turns,
               "ckpt": self.ckpt, "tp_ckpt": self.tp_ckpt}
        with BENCH_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


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


async def _serve(ai, human_name: str | None, n_battles: int | None,
                 bench: "_BenchLog | None" = None) -> None:
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
        w0, l0, t0 = ai.n_won_battles, ai.n_lost_battles, ai.n_tied_battles
        serve = asyncio.ensure_future(ai.accept_challenges(human_name, 1))
        try:
            while not serve.done():
                await asyncio.sleep(0.25)
        finally:
            if not serve.done():
                serve.cancel()                              # Ctrl-C / cancellation -> stop accepting
        serve.result()                                      # re-raise any real error from accept_challenges
        accepted += 1
        if bench is not None:                               # counter deltas → exact, draw-correct result
            result = ("ai" if ai.n_won_battles > w0 else
                      "human" if ai.n_lost_battles > l0 else
                      "draw" if ai.n_tied_battles > t0 else "unknown")
            bench.record(ai, result)
        if n_battles is None or accepted < n_battles:
            print(f"  [battle {accepted} done]  tally — AI {ai.n_won_battles} / you {ai.n_lost_battles} "
                  f"/ draws {ai.n_tied_battles}.  Challenge {AI_NAME} again to keep playing, or Ctrl-C to stop.")
            sys.stdout.flush()


async def _run(ai_team_str: str, ai_team_name: str, human_team_str: str, human_team_name: str,
               human_name: str | None, n_battles: int | None, url: str,
               ckpt: Path = DEFAULT_BC_CHECKPOINT, tp_ckpt: Path = DEFAULT_TP_CHECKPOINT,
               bench: _BenchLog | None = None, adapt_rules: bool = False,
               use_dossier: bool = False) -> None:
    # bench ON → the existing #18 plumbing saves a playable HTML replay per battle (embedded full
    # protocol log — re-parseable later for the Phase-3 adaptation loop).
    _rec = ({"save_replays": True, "live_dir": BENCH_DIR / "live",
             "replay_dir": BENCH_DIR / "replays" / bench.session_id, "replay_label": "bench"}
            if bench is not None else {})
    ai = make_player(AI_NAME, ai_team_str, model_path=ckpt, team_chooser_path=tp_ckpt,
                     adapt_rules=adapt_rules, use_dossier=use_dossier, **_rec)
    if adapt_rules:
        print("  adapt-rules ON (B-L1: Wide-Guard streak → spread-move tilt)")
    if use_dossier:
        print("  dossier ON (S1 L2b: cross-game opp item/ability/move warm-start)")
    print("\n" + "=" * 70)
    print("  HUMAN  vs  AI   (you pilot a real team against the production bot)")
    print("=" * 70)
    print(f"  AI ({AI_NAME}) team : {ai_team_name}")
    print(f"  AI battle ckpt     : {ckpt}")
    print(f"  AI TP ckpt         : {tp_ckpt}")
    if bench is not None:
        print(f"  bench session      : {bench.session_id}"
              + (f"  (note: {bench.note})" if bench.note else "")
              + f"  ->  {BENCH_LOG}")
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
        await _serve(ai, human_name, n_battles, bench)  # human_name=None -> accept from anyone
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
    # Serve refresh (docs/human_benchmark_design.md): benchmark ANY checkpoint without touching
    # the model_io production defaults (anchor promotion stays a separate USER decision).
    ap.add_argument("--ckpt", default=None,
                    help="battle-net checkpoint to serve (default: the model_io production default).")
    ap.add_argument("--tp-ckpt", default=None,
                    help="team-preview (SBDA) checkpoint to serve (default: production default).")
    ap.add_argument("--bench-note", default="",
                    help="free-text tag stamped on every benchmark row (e.g. hfdata_full).")
    ap.add_argument("--no-bench", action="store_true",
                    help="disable benchmark recording (rows + saved HTML replays). Default: ON.")
    ap.add_argument("--adapt-rules", action="store_true",
                    help="B-L1 serve-time pattern tilt (Wide-Guard streak → spread bias). Default OFF.")
    ap.add_argument("--dossier", action="store_true",
                    help="S1 L2b: warm-start unknown opp item/ability/moves from the per-opponent "
                         "dossier (cross-game; in-battle evidence always wins). Default OFF.")
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

    ckpt = Path(args.ckpt) if args.ckpt else DEFAULT_BC_CHECKPOINT
    tp_ckpt = Path(args.tp_ckpt) if args.tp_ckpt else DEFAULT_TP_CHECKPOINT
    for p in (ckpt, tp_ckpt):
        if not p.is_file():
            raise SystemExit(f"[play] checkpoint not found: {p}")
    bench = None
    if not args.no_bench:
        session_id = f"{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-{os.getpid()}"
        bench = _BenchLog(session_id, args.bench_note, Path(ai_name).name,
                          Path(human_name_team).name, ckpt, tp_ckpt)

    # Server lifecycle lives HERE (main thread) so the tree-kill ALWAYS runs — including on a
    # Ctrl-C, which now unwinds cleanly (see _serve) instead of wedging the terminal.
    proc = start_showdown()
    try:
        asyncio.run(_run(ai_team_str, Path(ai_name).name, human_team_str, Path(human_name_team).name,
                         args.human_name, args.n_battles, url, ckpt, tp_ckpt, bench,
                         args.adapt_rules, args.dossier))
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
