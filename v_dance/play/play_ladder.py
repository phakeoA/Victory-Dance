"""Ranked-ladder harness — the production AI searches ladder matches on ANY Showdown server.

The deployment path (goal-pillars G1: beat average players). Target-independent: the server /
account arrive as flags, so pointing at the real target later is a CLI change, not code.

  # local self-test (spawns a local server + an opponent that also searches; they match):
  python -m v_dance.play.play_ladder --self-test

  # real server (registered account; password via flag or the VD_LADDER_PASSWORD env var):
  python -m v_dance.play.play_ladder --server-url ws://<host>:<port>/showdown/websocket \
      --auth-url https://play.pokemonshowdown.com/action.php?  \
      --username MyBot --password ... --team maw_zard --games 10 \
      --ckpt ai_train_scripts/BC_model/checkpoints_attn_adv/battle_base.pt \
      --tp-ckpt ai_train_scripts/teamPreview_model/checkpoints/teampreview_sbda_v7.pt

Ladder play is CLOSED team sheets (poke-env default accept_open_team_sheet=False = the
production/training regime). Every finished game appends a bench JSONL row (note defaults to
"ladder") + rating fields when the server sends them, and saves an HTML replay — the same
recording contract as play_vs_human, so human_benchmark_report reads both.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

from poke_env.ps_client import AccountConfiguration
from poke_env.ps_client.server_configuration import ServerConfiguration

from v_dance.formats import DEFAULT_FORMAT
from v_dance.play.model_io import DEFAULT_BC_CHECKPOINT, DEFAULT_TP_CHECKPOINT
from v_dance.play.player import VGCPlayer
from v_dance.play.run_local_battle import (
    SHOWDOWN_HOST, SHOWDOWN_PORT, WS_OPEN_TIMEOUT, WS_PING_INTERVAL, WS_PING_TIMEOUT,
    discover_teams, load_team, resolve_team_path, start_showdown, stop_showdown,
)

_REPO = Path(__file__).resolve().parents[2]
BENCH_DIR = _REPO / "artifacts" / "human_benchmark"
BENCH_LOG = BENCH_DIR / "human_bench.jsonl"
_LOCAL_WS = f"ws://{SHOWDOWN_HOST}:{SHOWDOWN_PORT}/showdown/websocket"
# poke-env's LocalhostServerConfiguration auth URL (any reachable action.php works for guests).
_LOCAL_AUTH = "https://play.pokemonshowdown.com/action.php?"


def make_ladder_player(*, username: str, password: str | None, team_str: str,
                       server_url: str, auth_url: str, battle_format: str,
                       ckpt: Path | None, tp_ckpt: Path | None,
                       session_dir: str, save_replays: bool = True) -> VGCPlayer:
    """The production player (encoder + battle net + SBDA TP + gap-#6 splice) pointed at an
    ARBITRARY server/account. Mirrors run_local_battle.make_player's wiring; adds auth."""
    return VGCPlayer(
        model_path=ckpt,
        team_chooser_path=tp_ckpt,
        replay_path=_REPO / "artifacts" / "replay_buffer" / f"{username}.jsonl",
        device="cpu",
        account_configuration=AccountConfiguration(username, password),
        server_configuration=ServerConfiguration(server_url, auth_url),
        battle_format=battle_format,
        team=team_str,
        max_concurrent_battles=1,                 # ladder: one game at a time
        log_level=logging.WARNING,
        live_dir=BENCH_DIR / "live", save_replays=save_replays,
        replay_dir=BENCH_DIR / "replays" / session_dir, replay_label="ladder",
        ping_timeout=WS_PING_TIMEOUT, ping_interval=WS_PING_INTERVAL,
        open_timeout=WS_OPEN_TIMEOUT,
    )


def _record(ai, seen: set, session_id: str, note: str, team_name: str,
            ckpt: Path, tp_ckpt: Path, game_idx: int) -> None:
    """Append one bench row per newly-finished battle (same schema as play_vs_human +
    rating/opponent_rating). Append+flush per row -> a crash/Ctrl-C loses nothing."""
    for tag, b in list(ai.battles.items()):
        if not b.finished or tag in seen:
            continue
        seen.add(tag)
        won = getattr(b, "won", None)
        row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "session_id": session_id, "note": note, "game_idx": game_idx,
               "battle_tag": getattr(b, "battle_tag", tag), "ai_team": team_name,
               "human_team": None, "opponent": getattr(b, "opponent_username", None),
               "result": "ai" if won else ("draw" if won is None else "human"),
               "turns": getattr(b, "turn", None),
               "rating": getattr(b, "rating", None),
               "opponent_rating": getattr(b, "opponent_rating", None),
               "ckpt": str(ckpt), "tp_ckpt": str(tp_ckpt)}
        BENCH_DIR.mkdir(parents=True, exist_ok=True)
        with BENCH_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(f"  [game {game_idx}] {row['result'].upper():5s} vs {row['opponent']} "
              f"({row['turns']} turns, rating {row['rating']} vs {row['opponent_rating']})")
        sys.stdout.flush()


async def _ladder(ai, n_games: int, session_id: str, note: str, team_name: str,
                  ckpt: Path, tp_ckpt: Path) -> None:
    """n_games sequential ladder searches, recording after each. Games run one at a time
    (max_concurrent_battles=1), so per-game accounting is exact."""
    seen: set = set()
    for g in range(1, n_games + 1):
        print(f"[ladder] searching game {g}/{n_games} …"); sys.stdout.flush()
        task = asyncio.ensure_future(ai.ladder(1))
        try:
            while not task.done():                 # Windows Ctrl-C needs a waking loop
                await asyncio.sleep(0.25)
        finally:
            if not task.done():
                task.cancel()
        task.result()                              # surface real errors
        _record(ai, seen, session_id, note, team_name, ckpt, tp_ckpt, g)
    w, l, t = ai.n_won_battles, ai.n_lost_battles, ai.n_tied_battles
    print(f"[ladder] session done — AI {w} / opp {l} / draws {t}")


async def _self_test(args, ckpt: Path, tp_ckpt: Path, session_id: str) -> int:
    """Two players ladder-search the local server and get matched with each other —
    verifies the ENTIRE search->teampreview->play->record path with no human/server."""
    from v_dance.play.run_local_battle import make_player
    from v_dance.play.parallel_battles import close_players
    pool = discover_teams(reg=DEFAULT_FORMAT)
    if len(pool) < 2:
        raise SystemExit("[ladder] need >=2 pool teams for --self-test")
    t_ai, t_opp = pool[0], pool[1]
    ai = make_ladder_player(username="VDLadderAI", password=None,
                            team_str=load_team(resolve_team_path(t_ai)),
                            server_url=_LOCAL_WS, auth_url=_LOCAL_AUTH,
                            battle_format=args.format, ckpt=ckpt, tp_ckpt=tp_ckpt,
                            session_dir=session_id)
    opp = make_player("VDLadderOpp", load_team(resolve_team_path(t_opp)))
    both = asyncio.gather(ai.ladder(1), opp.ladder(1))
    try:
        await asyncio.wait_for(both, timeout=420.0)
    except asyncio.TimeoutError:
        print("[ladder] SELF-TEST TIMEOUT — the local server may not accept ladder "
              "searches for this format (challenge play is unaffected).")
        return 1
    finally:
        seen: set = set()
        _record(ai, seen, session_id, args.bench_note or "ladder-selftest",
                Path(t_ai).name, ckpt, tp_ckpt, 1)
        try:
            await close_players(ai, opp)
        except Exception:
            pass
    n = ai.n_finished_battles
    src = dict(getattr(ai, "_source_counts", {}) or {})
    print(f"[self-test] finished battles: {n}; decision sources: {src}")
    ok = n >= 1 and src.get("model", 0) > 0        # the MODEL must have driven decisions
    print(f"[self-test] VERDICT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser(description="Ranked-ladder play on any Showdown server.")
    ap.add_argument("--server-url", default=None,
                    help="websocket URL (default: the LOCAL server, which is auto-started).")
    ap.add_argument("--auth-url", default=_LOCAL_AUTH)
    ap.add_argument("--username", default="VictoryDanceAI")
    ap.add_argument("--password", default=None,
                    help="registered-account password (or set VD_LADDER_PASSWORD).")
    ap.add_argument("--team", default=None, help="pool team NAME (default: first pool team).")
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--format", default=DEFAULT_FORMAT)
    ap.add_argument("--ckpt", default=None, help="battle net (default: production).")
    ap.add_argument("--tp-ckpt", default=None, help="team-preview net (default: production).")
    ap.add_argument("--bench-note", default="ladder")
    ap.add_argument("--self-test", action="store_true",
                    help="local end-to-end check: two players ladder-search and get matched.")
    args = ap.parse_args()

    ckpt = Path(args.ckpt) if args.ckpt else DEFAULT_BC_CHECKPOINT
    tp_ckpt = Path(args.tp_ckpt) if args.tp_ckpt else DEFAULT_TP_CHECKPOINT
    for p in (ckpt, tp_ckpt):
        if not p.is_file():
            raise SystemExit(f"[ladder] checkpoint not found: {p}")
    password = args.password or os.environ.get("VD_LADDER_PASSWORD") or None
    session_id = f"{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-{os.getpid()}"

    local = args.server_url is None
    proc = start_showdown() if local else None     # remote server: nothing to manage
    try:
        if args.self_test:
            raise SystemExit(asyncio.run(_self_test(args, ckpt, tp_ckpt, session_id)))
        team_name = args.team or discover_teams(reg=DEFAULT_FORMAT)[0]
        ai = make_ladder_player(
            username=args.username, password=password,
            team_str=load_team(resolve_team_path(team_name)),
            server_url=args.server_url or _LOCAL_WS, auth_url=args.auth_url,
            battle_format=args.format, ckpt=ckpt, tp_ckpt=tp_ckpt, session_dir=session_id)
        print(f"[ladder] {args.username} @ {args.server_url or _LOCAL_WS} | format {args.format} "
              f"| team {Path(team_name).name}\n[ladder] ckpt {ckpt}\n[ladder] tp   {tp_ckpt}")
        asyncio.run(_ladder(ai, args.games, session_id, args.bench_note,
                            Path(team_name).name, ckpt, tp_ckpt))
    except KeyboardInterrupt:
        print("\n[ladder] Ctrl-C — stopping.")
    finally:
        logging.getLogger("poke_env").setLevel(logging.CRITICAL)
        if proc is not None:
            stop_showdown(proc)


if __name__ == "__main__":
    main()
