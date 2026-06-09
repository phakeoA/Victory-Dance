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
    python run_local_battle.py -v           # verbose (DEBUG) logging
 
Requirements
------------
    pip install poke-env
    pokemon-showdown/ cloned and npm-installed (done by setup.sh)
"""
 
from __future__ import annotations
 
import argparse
import asyncio
import logging
import os
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
 
from poke_env import AccountConfiguration
from player import VGCPlayer
 
# ── Logging ───────────────────────────────────────────────────────────────────
def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    # poke-env's own websocket logger is noisy at DEBUG; keep it at WARNING
    # unless the user explicitly wants it.
    if not verbose:
        logging.getLogger("poke_env").setLevel(logging.WARNING)
        logging.getLogger("websockets").setLevel(logging.WARNING)

log = logging.getLogger(__name__)
 
# ── Defaults ──────────────────────────────────────────────────────────────────
TEAM_FILE      = Path("teams/M-A/team1")
BATTLE_FORMAT  = "gen9championsvgc2026regma"
N_BATTLES_DEFAULT = 1
 
SHOWDOWN_DIR   = Path("pokemon-showdown")   # relative to this script's cwd
SHOWDOWN_HOST  = "localhost"
SHOWDOWN_PORT  = 8000
SHOWDOWN_READY_TIMEOUT = 120  # seconds — first launch builds Showdown (~30s)
 
# Venv-local node installed by setup.sh lives here (Windows exe or Unix bin)
_SCRIPT_DIR = Path(__file__).resolve().parent
_VENV_NODE_CANDIDATES = [
    _SCRIPT_DIR / ".venv" / "node" / "node.exe",   # Windows
    _SCRIPT_DIR / ".venv" / "node" / "node",        # Linux/macOS
    _SCRIPT_DIR / ".venv" / "bin"  / "node",        # symlink created by setup.sh
]
 
 
def _find_node() -> str:
    import shutil
    for candidate in _VENV_NODE_CANDIDATES:
        if candidate.exists():
            log.info("Using venv-local node: %s", candidate)
            return str(candidate)
    system_node = shutil.which("node")
    if system_node:
        log.info("Using system node: %s", system_node)
        return system_node
    log.error(
        "node executable not found.\n"
        "  Checked venv-local paths and system PATH.\n"
        "  Re-run setup.sh, or install Node.js and add it to PATH."
    )
    sys.exit(1)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# Showdown server lifecycle
# ══════════════════════════════════════════════════════════════════════════════
 
def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False
 
 
def start_showdown() -> subprocess.Popen | None:
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
 
    node = _find_node()
    node_dir = str(Path(node).parent)
    env = os.environ.copy()
    env["PATH"] = node_dir + os.pathsep + env.get("PATH", "")
 
    log.info("Starting Pokémon Showdown server …")
    log.info("(First launch will build Showdown — may take ~30s)")
    proc = subprocess.Popen(
        [node, "pokemon-showdown", "start", "--no-security"],
        cwd=SHOWDOWN_DIR,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
 
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
# Players
# ══════════════════════════════════════════════════════════════════════════════
 
def make_player(
    username: str,
    team: str,
    *,
    model=None,
    team_chooser=None,
) -> VGCPlayer:
    """
    Build a VGCPlayer.

    model        : trained battle nn.Module, or None (random fallback)
    team_chooser : trained teampreview nn.Module, or None (heuristic fallback)

    Swap in trained models when they are ready.
    """
    return VGCPlayer(
        model=model,
        team_chooser=team_chooser,
        replay_path=Path(f"replay_buffer/{username}.jsonl"),
        device="cpu",
        account_configuration=AccountConfiguration(username, None),
        battle_format=BATTLE_FORMAT,
        team=team,
        max_concurrent_battles=1,
        log_level=logging.WARNING,
    )
 
 
# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
 
async def run(n_battles: int, manage_server: bool, spectate: bool = True) -> None:
    server_proc = start_showdown() if manage_server else None
 
    team_str = load_team(TEAM_FILE)

    # ── Swap in trained models here when ready ────────────────────────────────
    battle_model  = None   # your trained nn.Module for in-battle decisions
    team_chooser  = None   # your trained nn.Module for teampreview selection
    # ─────────────────────────────────────────────────────────────────────────

    player1 = make_player("TrainerRed",  team_str, model=battle_model, team_chooser=team_chooser)
    player2 = make_player("TrainerBlue", team_str, model=battle_model, team_chooser=team_chooser)
 
    log.info("Starting %d battle(s) — format: %s", n_battles, BATTLE_FORMAT)

    # Hook: open the browser to spectate the first battle as soon as it starts
    _spectate_opened = False

    async def _maybe_open_spectator():
        if not spectate:
            return
        """Wait for the first battle tag to appear then open it in the browser."""
        nonlocal _spectate_opened
        for _ in range(60):          # wait up to 6 s for battle to start
            await asyncio.sleep(0.1)
            if player1.battles:
                tag = next(iter(player1.battles))
                url = f"http://{SHOWDOWN_HOST}:{SHOWDOWN_PORT}/{tag}"
                log.info("Opening spectator view: %s", url)
                webbrowser.open(url)
                _spectate_opened = True
                return
        log.warning("Could not find battle tag to open spectator view.")

    try:
        asyncio.ensure_future(_maybe_open_spectator())
        await player1.battle_against(player2, n_battles=n_battles)
    finally:
        await player1.ps_client.stop_listening()
        await player2.ps_client.stop_listening()
        player1.close()
        player2.close()
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
                   help=f"Showdown format ID (default: {BATTLE_FORMAT})")
    p.add_argument("--team", type=Path, default=TEAM_FILE,
                   help=f"Path to team paste file (default: {TEAM_FILE})")
    p.add_argument("--no-server", action="store_true",
                   help="Skip launching the server (use if it's already running)")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Enable DEBUG-level logging (very noisy)")
    p.add_argument("--no-spectate", action="store_true",
                   help="Do not open a browser tab to spectate")
    return p.parse_args()
 
 
if __name__ == "__main__":
    args = parse_args()
    _configure_logging(args.verbose)
    BATTLE_FORMAT = args.battle_format
    TEAM_FILE     = args.team
    asyncio.run(run(n_battles=args.battles, manage_server=not args.no_server, spectate=not args.no_spectate))
