"""
run_local_battle.py  —  VGC Champions Reg M-A local battle harness
===================================================================
Automatically starts the local Pokémon Showdown server, runs battles
between two players, then shuts the server down cleanly on exit.

Both players load their team from `team1.txt` (Showdown paste format).

Usage
-----
    python run_local_battle.py              # 1 battle
    python run_local_battle.py -n 50        # 50 battles
    python run_local_battle.py --no-server  # if server is already running

Requirements
------------
    pip install poke-env
    pokemon-showdown/ cloned and npm-installed (done by setup.sh)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────
TEAM_FILE      = Path("team1.txt")
BATTLE_FORMAT  = "gen9championsvgc2026regma"
N_BATTLES_DEFAULT = 1

SHOWDOWN_DIR   = Path("pokemon-showdown")   # relative to this script's cwd
SHOWDOWN_HOST  = "localhost"
SHOWDOWN_PORT  = 8000
SHOWDOWN_READY_TIMEOUT = 30   # seconds to wait for server to accept connections


# ══════════════════════════════════════════════════════════════════════════════
# Showdown server lifecycle
# ══════════════════════════════════════════════════════════════════════════════

def _port_open(host: str, port: int) -> bool:
    """Return True if something is already listening on host:port."""
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def start_showdown() -> subprocess.Popen | None:
    """
    Launch `node pokemon-showdown start --no-security` as a background process.
    Returns the Popen handle, or None if the server was already running.
    Waits until the websocket port is accepting connections before returning.
    """
    if _port_open(SHOWDOWN_HOST, SHOWDOWN_PORT):
        log.info("Showdown server already running on port %d — skipping launch.", SHOWDOWN_PORT)
        return None

    if not SHOWDOWN_DIR.is_dir():
        log.error(
            "pokemon-showdown/ not found at %s\n"
            "  Run setup.sh first to clone and install it.",
            SHOWDOWN_DIR.resolve(),
        )
        sys.exit(1)

    log.info("Starting Pokémon Showdown server …")
    proc = subprocess.Popen(
        ["node", "pokemon-showdown", "start", "--no-security"],
        cwd=SHOWDOWN_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait until port is open (server is ready to accept websockets)
    deadline = time.monotonic() + SHOWDOWN_READY_TIMEOUT
    while time.monotonic() < deadline:
        if _port_open(SHOWDOWN_HOST, SHOWDOWN_PORT):
            log.info("Showdown server ready on port %d ✅", SHOWDOWN_PORT)
            return proc
        if proc.poll() is not None:
            log.error("Showdown server process exited unexpectedly (exit code %d).", proc.returncode)
            sys.exit(1)
        time.sleep(0.5)

    proc.terminate()
    log.error(
        "Showdown server did not open port %d within %ds.",
        SHOWDOWN_PORT,
        SHOWDOWN_READY_TIMEOUT,
    )
    sys.exit(1)


def stop_showdown(proc: subprocess.Popen | None) -> None:
    """Terminate the server process we launched (no-op if we didn't start it)."""
    if proc is None:
        return
    log.info("Stopping Showdown server …")
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    log.info("Showdown server stopped.")


# ══════════════════════════════════════════════════════════════════════════════
# Team loading
# ══════════════════════════════════════════════════════════════════════════════

def load_team(path: Path) -> str:
    if not path.exists():
        log.error("Team file not found: %s", path.resolve())
        sys.exit(1)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        log.error("Team file is empty: %s", path.resolve())
        sys.exit(1)
    log.info("Loaded team from %s", path.resolve())
    return text


# ══════════════════════════════════════════════════════════════════════════════
# Players  (swap RandomPlayer for your AlphaZero agent later)
# ══════════════════════════════════════════════════════════════════════════════

def make_player(username: str, team: str) -> RandomPlayer:
    return RandomPlayer(
        account_configuration=AccountConfiguration(username, None),
        battle_format=BATTLE_FORMAT,
        team=team,
        max_concurrent_battles=1,
        log_level=logging.WARNING,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

async def run(n_battles: int, manage_server: bool) -> None:
    server_proc = start_showdown() if manage_server else None

    # Register Ctrl-C / SIGTERM so the server always gets shut down
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: (stop_showdown(server_proc), sys.exit(0)))

    team_str = load_team(TEAM_FILE)
    player1  = make_player("TrainerRed",  team_str)
    player2  = make_player("TrainerBlue", team_str)

    log.info("Starting %d battle(s) — format: %s", n_battles, BATTLE_FORMAT)

    try:
        await player1.battle_against(player2, n_battles=n_battles)
    finally:
        await player1.ps_client.stop_listening()
        await player2.ps_client.stop_listening()
        stop_showdown(server_proc)

    # ── Results ───────────────────────────────────────────────────────────────
    total   = player1.n_finished_battles
    p1_wins = player1.n_won_battles
    p2_wins = player2.n_won_battles
    draws   = total - p1_wins - p2_wins

    print()
    print("━" * 50)
    print(f"  Battles played  : {total}")
    print(f"  TrainerRed  wins: {p1_wins}  ({p1_wins/total*100:.1f} %)")
    print(f"  TrainerBlue wins: {p2_wins}  ({p2_wins/total*100:.1f} %)")
    if draws:
        print(f"  Draws           : {draws}")
    print("━" * 50)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run local VGC Champions Reg M-A battles.")
    p.add_argument("--battles", "-n", type=int, default=N_BATTLES_DEFAULT,
                   help=f"Number of battles (default: {N_BATTLES_DEFAULT})")
    p.add_argument("--format", "-f", default=BATTLE_FORMAT, dest="battle_format",
                   help=f"Showdown format string (default: {BATTLE_FORMAT})")
    p.add_argument("--team", type=Path, default=TEAM_FILE,
                   help=f"Path to team paste file (default: {TEAM_FILE})")
    p.add_argument("--no-server", action="store_true",
                   help="Skip launching the server (use if it's already running)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    BATTLE_FORMAT = args.battle_format
    TEAM_FILE     = args.team
    asyncio.run(run(n_battles=args.battles, manage_server=not args.no_server))
