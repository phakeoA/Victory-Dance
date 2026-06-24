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
# Force the SelectorEventLoop BEFORE poke_env is imported (it builds its background
# loop at import time) — on Windows the default Proactor loop's _poll races under
# Ctrl-C and wedges the terminal. See v_dance/__init__.py for the full rationale;
# this duplicate guard covers running this file as a script (where poke_env is
# imported before v_dance.__init__ runs).
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
from poke_env import AccountConfiguration
from v_dance.play.player import VGCPlayer              # local_battle/player.py (spliced)
from v_dance.play.random_player import RandomVGCPlayer  # local_battle/random_player.py (spliced)
from v_dance import formats                            # single source of truth for the active format
from v_dance.play.model_io import (DEFAULT_BC_CHECKPOINT,   # single source of truth for the prod checkpoints
                                   DEFAULT_TP_CHECKPOINT)   # (shared with generation.py / gauntlet.py)


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
BATTLE_FORMAT  = formats.DEFAULT_FORMAT   # active Champions-doubles reg (M-B); env/registry-driven, see v_dance/formats.py
N_BATTLES_DEFAULT = 1

SHOWDOWN_DIR   = _REPO_ROOT / "pokemon-showdown"
SHOWDOWN_HOST  = "localhost"
SHOWDOWN_PORT  = 8000
SHOWDOWN_READY_TIMEOUT = 120

# #22: poke-env websocket grace. Under heavy parallel eval (the 8-proc mirror) the single local
# server briefly stalls for tens of seconds — at poke-env's defaults (ping_timeout 20s,
# open_timeout 10s) that drops connections (`sent 1011 ... keepalive ping timeout`) and fails
# logins (`Expected BC… to be logged in`). Wider timeouts let the connection ride out the stall.
WS_PING_TIMEOUT  = 60.0
WS_PING_INTERVAL = 30.0
WS_OPEN_TIMEOUT  = 30.0

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


def start_showdown_on(port: int = SHOWDOWN_PORT) -> subprocess.Popen | None:
    """Start a Showdown server on ``port``, returning the launcher proc (or None if a server is
    ALREADY listening there — someone else manages it). 22f: the ``ServerPool`` runs several of
    these on consecutive ports. The port is passed as Showdown's positional ``[PORT]`` arg
    (``server/sockets.ts`` scans argv for the first all-numeric token)."""
    if _port_open(SHOWDOWN_HOST, port):
        log.info("Showdown server already running on port %d — skipping launch.", port)
        return None

    if not SHOWDOWN_DIR.is_dir():
        log.error("pokemon-showdown/ not found at %s — run setup.sh first.", SHOWDOWN_DIR)
        sys.exit(1)

    node = _find_node()
    node_dir = str(Path(node).parent)
    env = os.environ.copy()
    env["PATH"] = node_dir + os.pathsep + env.get("PATH", "")

    log.info("Starting Pokémon Showdown server on port %d … (first launch builds it, ~30s)", port)
    # Isolate the server from the console's Ctrl-C: on Windows a Ctrl-C sends CTRL_C_EVENT
    # to the whole process group, which would KILL the server while our (soft-stop) collection
    # is still launching battles → a flood of ConnectionRefused. A new process group lets the
    # server survive until we stop it cleanly in the loop's finally (stop_showdown terminates
    # it explicitly regardless of group).
    _flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    proc = subprocess.Popen(
        [node, "pokemon-showdown", "start", str(int(port)), "--no-security"],
        cwd=SHOWDOWN_DIR,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=_flags,
    )

    deadline = time.monotonic() + SHOWDOWN_READY_TIMEOUT
    while time.monotonic() < deadline:
        if _port_open(SHOWDOWN_HOST, port):
            log.info("Showdown server ready on port %d ✅", port)
            return proc
        if proc.poll() is not None:
            log.error("Showdown server exited unexpectedly (exit %d).", proc.returncode)
            sys.exit(1)
        time.sleep(0.5)

    proc.terminate()
    log.error("Showdown server did not open port %d within %ds.", port, SHOWDOWN_READY_TIMEOUT)
    sys.exit(1)


def start_showdown() -> subprocess.Popen | None:
    """Start the default single server on port 8000 — a back-compat wrapper around
    ``start_showdown_on`` (every existing caller passes no port)."""
    return start_showdown_on(SHOWDOWN_PORT)


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


class ServerPool:
    """A pool of K Showdown servers on consecutive ports (22f, the multi-server architecture).

    The single shared server is the throughput bottleneck — it SATURATES under high worker
    concurrency (#22, the eval deadlock) AND BLOATS over a long run (#22c, the gen-54 crawl).
    Spreading the worker processes across K servers (each on its own port, ``base_port`` ..
    ``base_port+K-1``) makes every server handle ~1/K the battles, so neither failure mode bites
    as hard, and one server can be recycled without stalling the others. ``n_servers <= 1`` (or
    ``manage=False``) reduces to the existing single-server behaviour with ZERO change.

    ``start_fn`` / ``stop_fn`` are injectable so the start / recycle / stop orchestration
    unit-tests offline WITHOUT spawning Node (the real spawn is ``start_showdown_on`` /
    ``stop_showdown``, exercised by the live smoke)."""

    def __init__(self, n_servers: int = 1, *, base_port: int = SHOWDOWN_PORT, manage: bool = True,
                 start_fn=None, stop_fn=None):
        self.n = max(1, int(n_servers))
        self.base_port = int(base_port)
        self.manage = bool(manage)
        self.ports = [self.base_port + i for i in range(self.n)]
        self._start = start_fn or start_showdown_on
        self._stop = stop_fn or stop_showdown
        self._procs: dict = {}                 # port -> launcher proc (None = pre-existing/unmanaged)

    def start_all(self) -> "ServerPool":
        """Start every server in the pool (no-op when ``manage=False``)."""
        if self.manage:
            for port in self.ports:
                self._procs[port] = self._start(port)
        return self

    def port_for_worker(self, worker_index: int) -> int:
        """Round-robin worker→server assignment (balanced: worker ``i`` → ``ports[i % n]``)."""
        return self.ports[int(worker_index) % self.n]

    def recycle(self, port: int) -> None:
        """Stop+restart ONE server (anti-bloat at a gen boundary), leaving the others up. The
        port stays the same, so the next gen's workers reconnect to a fresh server transparently."""
        if not self.manage:
            return
        self._stop(self._procs.get(port))
        self._procs[port] = self._start(port)

    def stop_all(self) -> None:
        """Tree-kill every managed server (each via ``stop_showdown``)."""
        if not self.manage:
            return
        for port in list(self._procs):
            self._stop(self._procs.pop(port))

    def __enter__(self) -> "ServerPool":
        return self.start_all()

    def __exit__(self, *exc) -> None:
        self.stop_all()


# ══════════════════════════════════════════════════════════════════════════════
# Team loading
# ══════════════════════════════════════════════════════════════════════════════

# Files under teams/Champions/ that are NOT team pastes (skipped by discover_teams).
_NON_TEAM_SUFFIXES = {".json", ".md", ".py", ".gitkeep", ".gitignore", ".lock",
                      ".yml", ".yaml", ".txt"}


def _reg_team_subdir(base: Path, fmt: str | None) -> Path | None:
    """The teams/Champions/<reg> subfolder matching ``fmt`` (regmb -> 'M-B'), or
    None if there's no matching subfolder."""
    from v_dance.formats import reg_token
    tok = reg_token(fmt)
    if not tok:
        return None
    want = tok[3:]                              # 'regmb' -> 'mb'
    for d in base.iterdir() if base.exists() else []:
        if d.is_dir() and "".join(c for c in d.name.lower() if c.isalnum()) == want:
            return d
    return None


def discover_teams(root: Path | None = None, reg: str | None = None) -> list[str]:
    """Every team paste under ``teams/Champions/`` (recursively), so the training pool
    auto-grows as M-A/M-B (or any future reg) folders are filled in — drop a file in,
    no code change. Returns repo-relative POSIX paths, sorted; READMEs / json / hidden
    files are skipped. Pass ``root`` to scan a different directory (used by tests).
    Pass ``reg`` (a format id) to RESTRICT to that reg's subfolder when one exists —
    so a backwards-compat (older-reg) run never draws a newer-reg team illegal in it."""
    base = Path(root) if root is not None else CHAMPIONS_DIR
    if reg and root is None:                    # filter the default pool to the reg's subdir
        sub = _reg_team_subdir(base, reg)
        if sub is not None:
            base = sub
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


def _normalize_team_paste(text: str) -> str:
    """Make a Showdown paste robust to the Champions bucket-scale export quirk where EV/IV
    stat groups are joined by a BARE '/' with no surrounding spaces (``EVs: 32 HP/15 Def/19
    Spe``). poke-env's strict teambuilder splits on whitespace and reads ``HP/15`` as a stat
    name -> ``KeyError: 'hp/15'``. Insert ' / ' between a stat letter and the next value on
    EVs:/IVs: lines only; already-spaced pastes are unchanged (idempotent). Showdown accepts
    the spaced form too, so the team sent to the server is unaffected."""
    import re
    out = []
    for line in text.splitlines():
        if line.lstrip()[:4].lower() in ("evs:", "ivs:"):
            line = re.sub(r"(?<=[A-Za-z])\s*/\s*(?=\d)", " / ", line)
        out.append(line)
    return "\n".join(out)


def load_team(path: Path) -> str:
    if not path.exists():
        log.error("Team file not found: %s", path.resolve())
        sys.exit(1)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        log.error("Team file is empty: %s", path.resolve())
        sys.exit(1)
    text = _normalize_team_paste(text)   # tolerate the no-space EV/IV separator export quirk
    log.info("Loaded team from %s", path.resolve())
    return text


def _normalize_team_selection(paths, repo_root) -> list[str]:
    """Dedup + repo-relative-POSIX-ify picked team paths, ORDER-PRESERVED (files outside
    the repo stay absolute). Pure (no GUI) so the cross-folder accumulation is unit-testable."""
    out: list[str] = []
    seen: set = set()
    for p in paths:
        pp = Path(p)
        try:
            rel = pp.relative_to(repo_root).as_posix()
        except ValueError:                           # outside the repo -> absolute
            rel = str(pp)
        if rel not in seen:
            seen.add(rel)
            out.append(rel)
    return out


def pick_team_files(initialdir=None) -> list[str]:
    """Native OS multi-select file picker for team pastes (Ctrl/Shift-click for several).
    ACCUMULATES ACROSS FOLDERS: a native dialog only multi-selects within ONE folder, so
    after each batch it asks whether to add more from another folder — letting you pick some
    teams from M-A and some from M-B in one session. Returns repo-relative POSIX paths
    (deduped, order-preserved), or [] on cancel / unavailable so the caller can fall back.
    SHARED by the gauntlet (``--pick-teams``) and the self-play eval (``--pick-eval-teams``).
    The tkinter import is lazy (no GUI cost unless used)."""
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        idir = str(initialdir or CHAMPIONS_DIR)
        if not Path(idir).exists():
            idir = None
        collected: list[str] = []
        while True:
            sel = filedialog.askopenfilenames(
                title="Select team files (Ctrl/Shift-click for several; add more folders next)",
                initialdir=idir)
            collected.extend(sel)
            if sel:
                idir = str(Path(sel[-1]).parent)     # reopen at the last-used folder
            n = len(_normalize_team_selection(collected, _REPO_ROOT))
            if not messagebox.askyesno(
                    "Add more team files?",
                    f"{n} team(s) selected so far.\n\n"
                    "Add more from ANOTHER folder (e.g. switch between M-A and M-B)?",
                    default=messagebox.NO):
                break
        root.destroy()
        return _normalize_team_selection(collected, _REPO_ROOT)
    except Exception as exc:
        print(f"[warn] team file picker unavailable ({exc}); falling back.")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# Players
# ══════════════════════════════════════════════════════════════════════════════

def localhost_server_config(port: int = SHOWDOWN_PORT):
    """A poke-env ``ServerConfiguration`` for a LOCAL Showdown server on ``port`` (22f: each worker
    binds the server it was assigned). Mirrors poke-env's ``LocalhostServerConfiguration`` with the
    port swapped; the auth endpoint is unused under ``--no-security`` (local play) but kept so the
    config is otherwise identical."""
    from poke_env.ps_client.server_configuration import (LocalhostServerConfiguration,
                                                         ServerConfiguration)
    return ServerConfiguration(
        f"ws://localhost:{int(port)}/showdown/websocket",
        LocalhostServerConfiguration.authentication_url,
    )


def make_player(
    username: str,
    team: str,
    *,
    model_path: Path = DEFAULT_BC_CHECKPOINT,
    team_chooser_path: Path = DEFAULT_TP_CHECKPOINT,
    max_concurrent_battles: int = 1,
    live_dir=None, save_replays: bool = False,
    replay_dir=None, replay_label=None,
    port: int | None = None,
) -> VGCPlayer:
    """Build a (gap-#6 spliced) player.  model_path=None → random fallback.
    ``max_concurrent_battles`` > 1 lets poke-env run that many battles of this player
    in parallel (3c.8c CPU-parallel collection). ``live_dir`` (#18b) wires the spectate feed so
    this player's battles show on the dashboard (used for EVAL match spectate); ``save_replays``
    keeps them on finish. ``replay_dir`` / ``replay_label`` (task E) route the SAVED html into a
    sub-folder with a descriptive name (e.g. ``eval/heuristic/gen3_vs_heuristic_<tag>.html``).
    ``port`` (22f) binds this player to the Showdown server on that port (a specific pool member);
    ``None`` keeps poke-env's default (localhost:8000), so the single-server path is unchanged."""
    replay_path = _REPO_ROOT / "artifacts" / "replay_buffer" / f"{username}.jsonl"
    # 22f: bind to the assigned pool server when a port is given (else poke-env's localhost default).
    _server = {"server_configuration": localhost_server_config(port)} if port is not None else {}
    if model_path is None:
        return RandomVGCPlayer(
            replay_path=replay_path,
            account_configuration=AccountConfiguration(username, None),
            battle_format=BATTLE_FORMAT,
            team=team,
            max_concurrent_battles=max_concurrent_battles,
            log_level=logging.WARNING,
            live_dir=live_dir, save_replays=save_replays,
            replay_dir=replay_dir, replay_label=replay_label,
            ping_timeout=WS_PING_TIMEOUT, ping_interval=WS_PING_INTERVAL,
            open_timeout=WS_OPEN_TIMEOUT,
            **_server,
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
        live_dir=live_dir, save_replays=save_replays,
        replay_dir=replay_dir, replay_label=replay_label,
        ping_timeout=WS_PING_TIMEOUT, ping_interval=WS_PING_INTERVAL,
        open_timeout=WS_OPEN_TIMEOUT,
        **_server,
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
    battle_model_path = DEFAULT_BC_CHECKPOINT
    team_chooser_path = DEFAULT_TP_CHECKPOINT
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
    p = argparse.ArgumentParser(description="Run local VGC Reg M-B battles (gap-#6 splice).")
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
    # Select the format for this process (parity with generation/gauntlet) so the
    # belief resolver follows --format too.
    if not formats.is_champions_doubles(args.battle_format):
        raise SystemExit(f"--format {args.battle_format!r} is not a Champions-doubles id "
                         f"(known: {formats.known_formats()})")
    _newest = formats.DEFAULT_FORMAT
    formats.set_active_format(args.battle_format)
    BATTLE_FORMAT = formats.DEFAULT_FORMAT
    if args.battle_format != _newest:
        log.warning("Running NON-default format %s (newest=%s) — the default team pool "
                    "spans all regs; pin --team/--team1/--team2 to a %s-legal team.",
                    args.battle_format, _newest, args.battle_format)
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
