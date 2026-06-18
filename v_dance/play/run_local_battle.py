"""
local_battle/run_local_battle.py  —  local battle harness (gap-#6 splice)
=========================================================================
Same as the repo-root ``run_local_battle.py`` but uses the gap-#6 SPLICING
players from this folder (local_battle/player.py + random_player.py), so the
opponent side is reconstructed via the training vod_parser at serve time.

Run from anywhere (paths are anchored to the repo root):

    python local_battle/run_local_battle.py              # 1 battle, opens browser
    python local_battle/run_local_battle.py -n 50        # 50 battles
    python local_battle/run_local_battle.py --no-server  # server already running
    python local_battle/run_local_battle.py -v           # DEBUG logging
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

# ── Path bootstrap: local_battle FIRST (so `import player`/`random_player`
# resolve to the spliced local versions), then repo root + data/scripts.
# remove-then-prepend so local_battle wins even over Python's auto-added
# script dir / any cached repo-root modules. ─────────────────────────────────
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]   # v_dance/play/ -> repo root
from poke_env import AccountConfiguration
from v_dance.play.player import VGCPlayer              # local_battle/player.py (spliced)
from v_dance.play.random_player import RandomVGCPlayer  # local_battle/random_player.py (spliced)


# ── Logging ───────────────────────────────────────────────────────────────────
def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    if not verbose:
        logging.getLogger("poke_env").setLevel(logging.WARNING)
        logging.getLogger("websockets").setLevel(logging.WARNING)


log = logging.getLogger(__name__)

# ── Defaults (anchored to repo root) ──────────────────────────────────────────
# Teams live under teams/Champions/<reg>/ (M-A now, M-B added over the coming week);
# the two players load DISTINCT teams by default. Override per-player with
# --team1/--team2, or force both onto one team with --team.
CHAMPIONS_DIR  = _REPO_ROOT / "teams" / "Champions"
TEAM_FILE      = CHAMPIONS_DIR / "M-A" / "team1"        # TrainerRed
TEAM_FILE_2    = CHAMPIONS_DIR / "M-A" / "WolfeGlick"   # TrainerBlue
BATTLE_FORMAT  = "gen9championsvgc2026regma"
N_BATTLES_DEFAULT = 1

SHOWDOWN_DIR   = _REPO_ROOT / "pokemon-showdown"
SHOWDOWN_HOST  = "localhost"
SHOWDOWN_PORT  = 8000
SHOWDOWN_READY_TIMEOUT = 120

_VENV_NODE_CANDIDATES = [
    _REPO_ROOT / ".venv" / "node" / "node.exe",   # Windows
    _REPO_ROOT / ".venv" / "node" / "node",        # Linux/macOS
    _REPO_ROOT / ".venv" / "bin"  / "node",        # symlink created by setup.sh
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
    log.error("node executable not found — re-run setup.sh or install Node.js.")
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
        log.error("pokemon-showdown/ not found at %s — run setup.sh first.", SHOWDOWN_DIR)
        sys.exit(1)

    node = _find_node()
    node_dir = str(Path(node).parent)
    env = os.environ.copy()
    env["PATH"] = node_dir + os.pathsep + env.get("PATH", "")

    log.info("Starting Pokémon Showdown server … (first launch builds it, ~30s)")
    # Isolate the server from the console's Ctrl-C: on Windows a Ctrl-C sends CTRL_C_EVENT
    # to the whole process group, which would KILL the server while our (soft-stop) collection
    # is still launching battles → a flood of ConnectionRefused. A new process group lets the
    # server survive until we stop it cleanly in the loop's finally (stop_showdown terminates
    # it explicitly regardless of group).
    _flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    proc = subprocess.Popen(
        [node, "pokemon-showdown", "start", "--no-security"],
        cwd=SHOWDOWN_DIR,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=_flags,
    )

    deadline = time.monotonic() + SHOWDOWN_READY_TIMEOUT
    while time.monotonic() < deadline:
        if _port_open(SHOWDOWN_HOST, SHOWDOWN_PORT):
            log.info("Showdown server ready on port %d ✅", SHOWDOWN_PORT)
            return proc
        if proc.poll() is not None:
            log.error("Showdown server exited unexpectedly (exit %d).", proc.returncode)
            sys.exit(1)
        time.sleep(0.5)

    proc.terminate()
    log.error("Showdown server did not open port %d within %ds.", SHOWDOWN_PORT, SHOWDOWN_READY_TIMEOUT)
    sys.exit(1)


def _terminate_tree(proc: subprocess.Popen) -> bool:
    """Terminate ``proc`` AND every descendant. ``node pokemon-showdown start`` forks a child
    server process (the one that actually holds the port), so killing only ``proc`` orphans it.
    Returns True if the tree was handled (psutil or taskkill), False if neither was usable
    (caller then falls back to a plain single-process terminate)."""
    # 1) psutil (cross-platform): SIGTERM the launcher + all descendants, then SIGKILL stragglers.
    try:
        import psutil
        try:
            parent = psutil.Process(proc.pid)
        except psutil.NoSuchProcess:
            return True                          # already gone
        tree = parent.children(recursive=True) + [parent]
        for p in tree:
            try:
                p.terminate()
            except psutil.NoSuchProcess:
                pass
        _gone, alive = psutil.wait_procs(tree, timeout=5)
        for p in alive:
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass
        return True
    except Exception:
        log.debug("psutil tree-kill unavailable/failed; trying taskkill", exc_info=True)

    # 2) Windows fallback: taskkill /T walks the tree without psutil.
    if sys.platform == "win32":
        try:
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/F", "/T"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            return True
        except Exception:
            log.debug("taskkill fallback failed", exc_info=True)
    return False


def stop_showdown(proc: subprocess.Popen | None) -> None:
    """Stop the Showdown server, killing the WHOLE process tree.

    ``node pokemon-showdown start`` forks a CHILD server process that holds port 8000; the old
    ``proc.terminate()`` killed only the launcher, ORPHANING that child on Windows — and the next
    ``start_showdown`` then silently REUSES the stale server (its port-open check passes). So
    terminate the launcher AND every descendant (``_terminate_tree``), with a plain terminate/kill
    as the last resort for the no-fork case."""
    if proc is None:
        return
    log.info("Stopping Showdown server …")
    if proc.poll() is not None:
        return                                   # launcher already exited
    if _terminate_tree(proc):
        try:
            proc.wait(timeout=5)                 # reap the launcher handle
        except Exception:
            pass
        log.info("Showdown server stopped.")
        return
    # last resort (psutil + taskkill both unusable): the original single-process terminate/kill.
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    log.info("Showdown server stopped.")


# ══════════════════════════════════════════════════════════════════════════════
# Team loading
# ══════════════════════════════════════════════════════════════════════════════

# Files under teams/Champions/ that are NOT team pastes (skipped by discover_teams).
_NON_TEAM_SUFFIXES = {".json", ".md", ".py", ".gitkeep", ".gitignore", ".lock",
                      ".yml", ".yaml", ".txt"}


def discover_teams(root: Path | None = None) -> list[str]:
    """Every team paste under ``teams/Champions/`` (recursively), so the training pool
    auto-grows as M-A/M-B (or any future reg) folders are filled in — drop a file in,
    no code change. Returns repo-relative POSIX paths, sorted; READMEs / json / hidden
    files are skipped. Pass ``root`` to scan a different directory (used by tests)."""
    base = Path(root) if root is not None else CHAMPIONS_DIR
    out: list[str] = []
    if not base.exists():
        return out
    for p in sorted(base.rglob("*")):
        if not p.is_file() or p.name.startswith(".") \
                or p.suffix.lower() in _NON_TEAM_SUFFIXES \
                or p.name.lower().startswith("readme"):
            continue
        try:
            out.append(p.relative_to(_REPO_ROOT).as_posix())
        except ValueError:                      # outside the repo (test tmp dir) -> absolute
            out.append(str(p))
    return out


def resolve_team_path(arg) -> Path:
    """Accept either a team NAME (found anywhere under teams/Champions/, e.g. 'WolfeGlick')
    or a path to a team file. Exits with a clear error if neither exists."""
    p = Path(arg)
    if p.exists():
        return p
    rp = _REPO_ROOT / arg                       # repo-relative (e.g. from discover_teams)
    if rp.exists():
        return rp
    if CHAMPIONS_DIR.exists():                   # search by name under any reg subfolder
        matches = sorted(q for q in CHAMPIONS_DIR.rglob(str(arg)) if q.is_file())
        if matches:
            return matches[0]
    log.error("Team '%s' not found (looked for the file, %s, and by name under "
              "teams/Champions/).\nAvailable teams under teams/Champions/: %s",
              arg, rp, ", ".join(discover_teams()))
    sys.exit(1)


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
    model_path: Path = _REPO_ROOT / "ai_train_scripts" / "BC_model" / "checkpoints" / "bc_best.pt",
    team_chooser_path: Path = _REPO_ROOT / "ai_train_scripts" / "teamPreview_model" / "checkpoints" / "teampreview_best.pt",
    max_concurrent_battles: int = 1,
) -> VGCPlayer:
    """Build a (gap-#6 spliced) player.  model_path=None → random fallback.
    ``max_concurrent_battles`` > 1 lets poke-env run that many battles of this player
    in parallel (3c.8c CPU-parallel collection)."""
    replay_path = _REPO_ROOT / "artifacts" / "replay_buffer" / f"{username}.jsonl"
    if model_path is None:
        return RandomVGCPlayer(
            replay_path=replay_path,
            account_configuration=AccountConfiguration(username, None),
            battle_format=BATTLE_FORMAT,
            team=team,
            max_concurrent_battles=max_concurrent_battles,
            log_level=logging.WARNING,
        )
    return VGCPlayer(
        model_path=model_path,
        team_chooser_path=team_chooser_path,
        replay_path=replay_path,
        device="cpu",
        account_configuration=AccountConfiguration(username, None),
        battle_format=BATTLE_FORMAT,
        team=team,
        max_concurrent_battles=max_concurrent_battles,
        log_level=logging.WARNING,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

async def run(
    n_battles: int,
    manage_server: bool,
    spectate: bool = True,
    team1_path: Path = TEAM_FILE,
    team2_path: Path = TEAM_FILE_2,
) -> None:
    server_proc = start_showdown() if manage_server else None

    # Each player gets its OWN team (load_team validates existence + non-empty).
    team1_str = load_team(team1_path)
    team2_str = load_team(team2_path)
    log.info("TrainerRed  team → %s", team1_path)
    log.info("TrainerBlue team → %s", team2_path)
    if Path(team1_path).resolve() == Path(team2_path).resolve():
        log.warning("Both players are using the SAME team file (%s) — pass "
                    "--team1/--team2 for distinct teams.", team1_path)

    # ── Trained checkpoints (set to None to fall back to a random player) ─────
    battle_model_path = _REPO_ROOT / "ai_train_scripts" / "BC_model" / "checkpoints" / "bc_best.pt"
    team_chooser_path = _REPO_ROOT / "ai_train_scripts" / "teamPreview_model" / "checkpoints" / "teampreview_best.pt"
    for _p in (battle_model_path, team_chooser_path):
        if not _p.exists():
            log.warning("checkpoint missing: %s — that player will use the random fallback.", _p)
    # ─────────────────────────────────────────────────────────────────────────

    player1 = make_player("TrainerRed",  team1_str, model_path=battle_model_path, team_chooser_path=team_chooser_path)
    player2 = make_player("TrainerBlue", team2_str, model_path=battle_model_path, team_chooser_path=team_chooser_path)

    log.info("Starting %d battle(s) — format: %s (gap-#6 opponent splice ON)", n_battles, BATTLE_FORMAT)

    _spectate_opened = False

    async def _maybe_open_spectator():
        if not spectate:
            return
        nonlocal _spectate_opened
        for _ in range(60):
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
        # stop_showdown(server_proc)

    total   = player1.n_finished_battles
    p1_wins = player1.n_won_battles
    p2_wins = player2.n_won_battles
    draws   = total - p1_wins - p2_wins

    def _src(p):
        # how each player's decisions were made — proves the AI (not randomness)
        # is driving: expect mostly 'model' (+ 'forced_switch' for post-faint
        # replacements).  'retry' > 0 means Showdown rejected an order (a bug);
        # 'model_error'/'no_model' mean the policy failed — none should appear.
        c = getattr(p, "_source_counts", {})
        return dict(c) if c else {}

    print()
    print("━" * 50)
    print(f"  Battles played  : {total}")
    if total:
        print(f"  TrainerRed  wins: {p1_wins}  ({p1_wins/total*100:.1f} %)")
        print(f"  TrainerBlue wins: {p2_wins}  ({p2_wins/total*100:.1f} %)")
    if draws:
        print(f"  Draws           : {draws}")
    print("─" * 50)
    print(f"  TrainerRed  decisions by source: {_src(player1)}")
    print(f"  TrainerBlue decisions by source: {_src(player2)}")
    print("  (want mostly 'model'; 'retry'/'model_error'/'no_model' = a problem)")

    def _tp(p):
        c = getattr(p, "_tp_source", {})
        return dict(c) if c else {}

    print("-" * 50)
    print(f"  TrainerRed  team-preview by source: {_tp(player1)}")
    print(f"  TrainerBlue team-preview by source: {_tp(player2)}")
    print("  (#4: want all 'model' = TP NET drove team preview; 'heuristic' = fell back)")
    print("━" * 50)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run local VGC Reg M-A battles (gap-#6 splice).")
    p.add_argument("--battles", "-n", type=int, default=N_BATTLES_DEFAULT)
    p.add_argument("--format", "-f", default=BATTLE_FORMAT, dest="battle_format")
    p.add_argument("--team1", default=TEAM_FILE,
                   help="TrainerRed's team — a name under teams/M-A/ (e.g. WolfeGlick) "
                        "or a path (default: team1)")
    p.add_argument("--team2", default=TEAM_FILE_2,
                   help="TrainerBlue's team — a name under teams/M-A/ or a path "
                        "(default: WolfeGlick)")
    p.add_argument("--team", default=None,
                   help="shortcut: use this ONE team (name or path) for BOTH players "
                        "— handy for mirror testing")
    p.add_argument("--no-server", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--no-spectate", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _configure_logging(args.verbose)
    BATTLE_FORMAT = args.battle_format
    # --team (if given) forces both players onto one team; otherwise each player
    # uses its own --team1 / --team2 (distinct by default).  Each accepts a team
    # NAME (resolved under teams/M-A/) or a path.
    team1_path = resolve_team_path(args.team or args.team1)
    team2_path = resolve_team_path(args.team or args.team2)
    asyncio.run(run(
        n_battles=args.battles,
        manage_server=not args.no_server,
        spectate=not args.no_spectate,
        team1_path=team1_path,
        team2_path=team2_path,
    ))
