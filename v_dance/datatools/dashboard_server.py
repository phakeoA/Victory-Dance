"""Flask server for the self-play dashboard (3c.6e-2).

Read-only: it just serves the dashboard's static files plus the two live JSON
feeds that the self-play loop writes to ``artifacts/self_play_archive/``:

    GET /                 -> dashboard.html
    GET /<file>           -> dashboard.css / dashboard.js / demo_manifest.json
    GET /manifest.json    -> artifacts/self_play_archive/manifest.json  (live, no-cache)
    GET /status.json      -> artifacts/self_play_archive/status.json    (live, no-cache)

The dashboard (served here) polls ``/status.json`` + ``/manifest.json`` every
couple of seconds. Decoupled from training on purpose: the self-play process
WRITES the files (via LiveStatus / write_manifest), this server only READS them,
so starting/stopping/crashing either side never corrupts the other. The Spectate
tab embeds the local Showdown client (a different origin, ``localhost:8000``)
directly, so this server doesn't proxy battles.

Run:  .venv/Scripts/python.exe -m v_dance.datatools.dashboard_server [--port 5175]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory

_REPO = Path(__file__).resolve().parents[2]
_DASH_DIR = _REPO / "data" / "scripts" / "dashboard"
_ARCHIVE_DIR = _REPO / "artifacts" / "self_play_archive"

# the self-play launcher (UI) drives this module: python -m v_dance.selfplay.generation --config ...
_GEN_MODULE = "v_dance.selfplay.generation"

# Console tab: serve at most this many trailing bytes on the first (offset-less) /launch_log poll, so
# a multi-MB run log doesn't ship whole; subsequent polls fetch only the NEW bytes since the offset.
_LOG_TAIL_BYTES = 200_000

# only these are served from the dashboard dir (defense-in-depth alongside
# send_from_directory's own traversal protection)
_ALLOWED = {"dashboard.html", "dashboard.css", "dashboard.js", "demo_manifest.json",
            "launcher.css", "launcher.js",   # launcher is a dashboard TAB, loaded by dashboard.html
            "console.css", "console.js"}     # console is a dashboard TAB (live run stdout/stderr)

# eval/ sub-folders whose saved .html replays the browser may serve (#H); whitelist guards the route.
_EVAL_SECTIONS = {"league", "random", "max_damage", "heuristic", "replays", "eval"}

_EMPTY_MANIFEST = {"n_generations": 0, "best_path": None, "best_generation": None,
                   "best_win_rate": None, "best_elo": None, "n_promotions": 0,
                   "champion_path": None, "champion_generation": None, "champion_elo": None,
                   "best_scripted_generation": None, "best_scripted_win_rate": None,
                   "league": [], "generations": []}
_IDLE_STATUS = {"live": False, "updated_at": None, "showdown_url": "http://localhost:8000",
                "run": {"phase": "idle", "generation": 0, "n_generations": None,
                        "games_done": 0, "games_total": 0, "running_p1_winrate": None,
                        "started_at": None, "hours_budget": None, "last_verdict": None},
                "update": {}, "active_battles": []}
_EMPTY_LOG = {"tag": None, "turn": 0, "updated_at": None, "n_lines": 0, "log": []}


def _serve_json(path: Path, default: dict):
    """Serve a live JSON file no-cache, or a sane default if it doesn't exist yet."""
    if path.exists():
        resp = Response(path.read_text(encoding="utf-8"), mimetype="application/json")
    else:
        resp = jsonify(default)
    resp.headers["Cache-Control"] = "no-store, max-age=0"
    return resp


_SCHEMA_CACHE: dict = {}


def _config_schema() -> dict:
    """The launcher UI's field list: shell out to ``generation --dump-config`` (keeps torch OUT of
    this light server), annotate each key with an inferred input type, and cache it (defaults are
    static). Shape: ``{"sections": {sec: {field: {default, type, nullable}}}}``."""
    if _SCHEMA_CACHE.get("data"):
        return _SCHEMA_CACHE["data"]
    import subprocess
    import sys
    import json as _json
    try:
        out = subprocess.run([sys.executable, "-m", _GEN_MODULE, "--dump-config"],
                             cwd=str(_REPO), capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=180,
                             env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        cfg = _json.loads(out.stdout)
    except Exception as exc:                              # surface, don't crash the server
        return {"error": f"could not load config schema: {exc}", "sections": {}}

    def _typ(v):
        if isinstance(v, bool):
            return "bool"
        if isinstance(v, int):
            return "int"
        if isinstance(v, float):
            return "float"
        if isinstance(v, list):
            return "list"
        return "str"

    sections = {sec: {k: {"default": v, "type": _typ(v), "nullable": v is None}
                      for k, v in body.items()}
                for sec, body in cfg.items() if isinstance(body, dict)}
    data = {"sections": sections}
    _SCHEMA_CACHE["data"] = data
    return data


def _spawn_run(cfg_path: Path, log_path: Path):
    """Spawn ``python -m generation --config <cfg>`` FULLY DETACHED from the launching console,
    stdout+stderr -> a log file. Detached so that closing the terminal / the dashboard that started
    it does NOT kill the run (Windows sends CTRL_CLOSE to console children). Returns the Popen
    handle. Injectable — tests monkeypatch this to avoid a real launch."""
    import subprocess
    import sys
    logf = open(log_path, "w", encoding="utf-8")
    kwargs = {}
    if os.name == "nt":
        # CREATE_NO_WINDOW: the run gets its OWN HIDDEN console -> (1) no popup terminal, and its
        # children (the mp workers + the Node Showdown server) inherit that hidden console so THEY
        # don't pop windows either; (2) it's not attached to the launching console, so closing the
        # terminal/dashboard doesn't kill it. (DETACHED_PROCESS gave NO console, which made those
        # console children each spawn a NEW visible window — the cmd popups.) CREATE_NEW_PROCESS_GROUP
        # keeps the whole tree killable for /stop.
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True                  # POSIX: detach from the controlling tty
    # PYTHONUNBUFFERED: stdout to a FILE is block-buffered by default, so a run's output would only
    # reach the log in ~4-8 KB chunks and the Console tab would lag far behind. Force unbuffered so the
    # live tail streams promptly, "like a terminal". (Overhead is negligible — the loop prints rarely.)
    return subprocess.Popen(
        [sys.executable, "-m", _GEN_MODULE, "--config", str(cfg_path)],
        cwd=str(_REPO), stdout=logf, stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}, **kwargs)


def _pid_alive(pid) -> bool:
    """True if process ``pid`` is currently running (no psutil dependency)."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if os.name == "nt":
        import subprocess
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                             capture_output=True, text=True)
        return str(pid) in (out.stdout or "")
    try:
        os.kill(pid, 0)                                      # POSIX: signal 0 = liveness probe
        return True
    except OSError:
        return False


def _kill_pid_tree(pid) -> None:
    """Terminate ``pid`` and its child tree (the detached run + its mp workers + Showdown server)."""
    if os.name == "nt":
        import subprocess
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(int(pid))], capture_output=True)
    else:
        import signal
        os.kill(int(pid), signal.SIGTERM)


# A run that crashed / was killed leaves status.json ``live:true`` forever, which would deadlock
# the launcher (can't start: "a run is already in progress"). Treat a live flag as STALE — i.e. the
# writing process is gone — when status.json hasn't been updated in this many seconds. Generous so a
# genuinely-running run (which writes status every phase / collection batch) is never falsely cleared.
_STALE_LIVE_SECONDS = 1800.0


def _run_is_live(archive_dir: Path) -> bool:
    """True only if status.json reports a live run AND it's still fresh (updated within
    ``_STALE_LIVE_SECONDS``). A crashed run's stale ``live:true`` reads as NOT live, so it can't
    block /launch forever."""
    st = archive_dir / "status.json"
    if not st.exists():
        return False
    try:
        import json as _json
        import time as _time
        s = _json.loads(st.read_text(encoding="utf-8"))
        if not s.get("live"):
            return False
        ua = s.get("updated_at")
        try:
            return (_time.time() - float(ua)) < _STALE_LIVE_SECONDS    # live AND fresh
        except (TypeError, ValueError):
            return True                                                # live, no/odd timestamp -> assume live
    except Exception:
        return False


def _clear_live_flag(archive_dir: Path) -> bool:
    """Set status.json ``live:false`` (recover from a crashed run's stale flag). Returns True if it
    WAS live (so we cleared something). A still-running process will just re-write live:true on its
    next status write, so clearing a genuinely-live run is harmless (no-op in effect)."""
    st = archive_dir / "status.json"
    if not st.exists():
        return False
    try:
        import json as _json
        s = _json.loads(st.read_text(encoding="utf-8"))
        if not s.get("live"):
            return False
        s["live"] = False
        if isinstance(s.get("run"), dict):
            s["run"]["phase"] = "idle"
        st.write_text(_json.dumps(s), encoding="utf-8")
        return True
    except Exception:
        return False


def _launch_running(app, archive_dir: Path) -> bool:
    """True if a launcher-spawned run is currently alive — the in-memory Popen if we still hold it,
    else the persisted PID (survives a dashboard restart). Mirrors /launch_state's liveness check."""
    proc = app.config.get("_LAUNCHED")
    if proc is not None:
        return proc.poll() is None
    pidf = archive_dir / "launched.pid"
    if pidf.exists():
        try:
            pid = int((pidf.read_text(encoding="utf-8") or "0").strip()) or None
        except Exception:
            pid = None
        return bool(pid and _pid_alive(pid))
    return False


def _resolve_launch_log(app, archive_dir: Path):
    """Resolve the active run's stdout/stderr log Path (or None). Prefer the in-memory path of the
    run we launched; else the NEWEST logs/launch_*.log so a dashboard restarted AFTER the launch can
    still tail the detached run."""
    cur = app.config.get("_LAUNCH_LOG")
    if cur and Path(cur).exists():
        return Path(cur)
    logs = archive_dir / "logs"
    if logs.is_dir():
        cands = sorted(logs.glob("launch_*.log"), key=lambda p: p.stat().st_mtime)
        if cands:
            return cands[-1]
    return None


def create_app(dash_dir=_DASH_DIR, archive_dir=_ARCHIVE_DIR) -> Flask:
    # resolve to ABSOLUTE: send_from_directory (the /eval_replay file server, #H) 404s on a relative
    # directory, so a relative --archive would otherwise serve nothing.
    dash_dir, archive_dir = Path(dash_dir).resolve(), Path(archive_dir).resolve()
    app = Flask(__name__, static_folder=None)
    app.config["DASH_DIR"] = dash_dir
    app.config["ARCHIVE_DIR"] = archive_dir
    app.config["_LAUNCHED"] = None        # the Popen of a run started via POST /launch
    app.config["_LAUNCH_LOG"] = None

    @app.route("/")
    def index():
        return send_from_directory(dash_dir, "dashboard.html")

    @app.route("/manifest.json")
    def manifest():
        return _serve_json(archive_dir / "manifest.json", _EMPTY_MANIFEST)

    @app.route("/status.json")
    def status():
        return _serve_json(archive_dir / "status.json", _IDLE_STATUS)

    @app.route("/live_log.json")
    def live_log():
        return _serve_json(archive_dir / "live_log.json", _EMPTY_LOG)

    @app.route("/live_battles.json")
    def live_battles():
        # #18 multi-battle spectate: aggregate the file-per-battle feed, written by each recorder
        # (incl. separate MP collection/eval workers). Scope to the CURRENT run's latest gen
        # (#18b review) so we don't rglob every saved replay ever; drop finished/stale ones.
        from v_dance.selfplay.status import read_live_battles, current_live_dir
        try:
            battles = read_live_battles(current_live_dir(archive_dir / "live"))
        except Exception:
            battles = []
        resp = jsonify({"battles": battles, "n": len(battles)})
        resp.headers["Cache-Control"] = "no-store, max-age=0"
        return resp

    @app.route("/eval_replays.json")
    def eval_replays():
        # #H: list the SAVED eval replay HTMLs (task E) for a generation, grouped by section
        # (league/random/max_damage/heuristic), so the Spectate tab can browse them per gen.
        from v_dance.selfplay.status import list_saved_eval_replays
        gen = request.args.get("gen", type=int)
        try:
            sections = list_saved_eval_replays(archive_dir / "live", gen) if gen is not None else []
        except Exception:
            sections = []
        resp = jsonify({"gen": gen, "sections": sections})
        resp.headers["Cache-Control"] = "no-store, max-age=0"
        return resp

    @app.route("/eval_replay/<int:gen>/<kind>/<path:name>")
    def eval_replay(gen, kind, name):
        # #H: serve one saved replay HTML so clicking it opens the animated Showdown replay. Scoped to
        # the current run's gen_<N>/eval/<kind>/; send_from_directory rejects path traversal.
        from v_dance.selfplay.status import latest_run_dir
        if kind not in _EVAL_SECTIONS or not name.endswith(".html"):
            return ("not found", 404)
        run = latest_run_dir(archive_dir / "live")
        if run is None:
            return ("not found", 404)
        d = run / f"gen_{gen}" / "eval" / kind
        if not d.is_dir():
            return ("not found", 404)
        return send_from_directory(d, name)

    # ── Self-play launcher (a dashboard TAB): config-driven, generation.py as the backend ──
    @app.route("/config-schema")
    def config_schema():
        resp = jsonify(_config_schema())
        resp.headers["Cache-Control"] = "no-store, max-age=0"
        return resp

    @app.route("/launch", methods=["POST"])
    def launch():
        cfg = request.get_json(silent=True)
        if not isinstance(cfg, dict):
            return jsonify({"ok": False, "error": "body must be a JSON object (the run config)"}), 400
        if _run_is_live(archive_dir):
            return jsonify({"ok": False,
                            "error": "a run is already in progress (status.json live)"}), 409
        prev = app.config.get("_LAUNCHED")
        if prev is not None and prev.poll() is None:
            return jsonify({"ok": False,
                            "error": "a launched run is still active — stop it first"}), 409
        import json as _json
        import time as _time
        stamp = _time.strftime("%Y%m%d_%H%M%S")
        cfg_dir = archive_dir / "launch_configs"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = cfg_dir / f"config_{stamp}.json"
        cfg_path.write_text(_json.dumps(cfg, indent=2), encoding="utf-8")
        log_dir = archive_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"launch_{stamp}.log"
        try:
            proc = _spawn_run(cfg_path, log_path)
        except Exception as exc:
            return jsonify({"ok": False, "error": f"spawn failed: {exc}"}), 500
        app.config["_LAUNCHED"] = proc
        app.config["_LAUNCH_LOG"] = str(log_path)
        # Persist the PID so a RESTARTED dashboard (its own terminal closed) can still stop the
        # detached run + read its state — the run outlives the launcher now.
        try:
            (archive_dir / "launched.pid").write_text(str(proc.pid), encoding="utf-8")
        except Exception:
            pass
        return jsonify({"ok": True, "pid": proc.pid,
                        "config": str(cfg_path), "log": str(log_path)})

    @app.route("/stop", methods=["POST"])
    def stop():
        proc = app.config.get("_LAUNCHED")
        if proc is not None and proc.poll() is None:
            try:
                _kill_pid_tree(proc.pid)                     # UI-launched run is alive -> tree-kill
            except Exception as exc:
                return jsonify({"ok": False, "error": str(exc)}), 500
            _clear_live_flag(archive_dir)
            return jsonify({"ok": True, "stopped_pid": proc.pid})
        # No in-memory handle (e.g. the dashboard was restarted while a DETACHED run keeps going).
        # Recover via the persisted PID: kill it if still alive, then clear the live flag.
        pidf = archive_dir / "launched.pid"
        if pidf.exists():
            try:
                pid = int((pidf.read_text(encoding="utf-8") or "0").strip())
            except Exception:
                pid = 0
            if pid and _pid_alive(pid):
                try:
                    _kill_pid_tree(pid)
                except Exception as exc:
                    return jsonify({"ok": False, "error": str(exc)}), 500
                _clear_live_flag(archive_dir)
                return jsonify({"ok": True, "stopped_pid": pid, "via": "pidfile"})
        # Nothing alive to kill — clear a stale live flag if present so the launcher can recover.
        if _clear_live_flag(archive_dir):
            return jsonify({"ok": True, "cleared_live_flag": True,
                            "note": "no running launched process found; cleared a stale live flag."})
        return jsonify({"ok": False, "error": "no active run to stop"}), 409

    @app.route("/launch_state")
    def launch_state():
        proc = app.config.get("_LAUNCHED")
        if proc is not None:
            running, pid, rc = (proc.poll() is None), proc.pid, proc.returncode
        else:
            # dashboard restarted: fall back to the persisted PID of a (detached) run
            pid, running, rc = None, False, None
            pidf = archive_dir / "launched.pid"
            if pidf.exists():
                try:
                    pid = int((pidf.read_text(encoding="utf-8") or "0").strip()) or None
                except Exception:
                    pid = None
                running = bool(pid and _pid_alive(pid))
        tail = []
        lp = app.config.get("_LAUNCH_LOG")
        if lp and Path(lp).exists():
            try:
                tail = Path(lp).read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
            except Exception:
                tail = []
        resp = jsonify({"running": running, "pid": pid, "returncode": rc, "log_tail": tail})
        resp.headers["Cache-Control"] = "no-store, max-age=0"
        return resp

    @app.route("/launch_log")
    def launch_log():
        # Console TAB: incremental tail of the launched run's stdout/stderr log (it goes to a HIDDEN
        # console, so this is the only way to watch a UI-launched run "like a terminal"). The client
        # passes the ?path it currently holds + the byte ?offset it has consumed; we return only the
        # NEW bytes since that offset. First poll (no offset), a run change (path mismatch), or a
        # rotated/truncated file (offset past EOF) -> resync from the last _LOG_TAIL_BYTES with
        # reset=true so the UI clears its buffer.
        lp = _resolve_launch_log(app, archive_dir)
        running = _launch_running(app, archive_dir)
        if lp is None:
            resp = jsonify({"available": False, "running": running, "path": None,
                            "offset": 0, "size": 0, "data": "", "reset": False})
            resp.headers["Cache-Control"] = "no-store, max-age=0"
            return resp
        try:
            size = lp.stat().st_size
        except OSError:
            size = 0
        off = request.args.get("offset", type=int)
        req_path = request.args.get("path")
        reset = off is None or off > size or (req_path is not None and req_path != lp.name)
        start = max(0, size - _LOG_TAIL_BYTES) if reset else off
        data = ""
        try:
            with open(lp, "rb") as f:
                f.seek(start)
                data = f.read(max(0, size - start)).decode("utf-8", errors="replace")
        except Exception:
            data = ""
        resp = jsonify({"available": True, "running": running, "path": lp.name,
                        "offset": size, "size": size, "data": data, "reset": reset})
        resp.headers["Cache-Control"] = "no-store, max-age=0"
        return resp

    @app.route("/<path:fname>")
    def asset(fname):
        if fname not in _ALLOWED:
            return ("not found", 404)
        return send_from_directory(dash_dir, fname)

    return app


def main(argv=None):
    ap = argparse.ArgumentParser(description="Self-play dashboard server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5175)
    ap.add_argument("--archive", default=str(_ARCHIVE_DIR),
                    help="dir holding the live manifest.json + status.json")
    args = ap.parse_args(argv)
    app = create_app(archive_dir=args.archive)
    url = f"http://{args.host}:{args.port}/"
    print(f"[dashboard] serving {url}")
    print(f"[dashboard]   dashboard files: {_DASH_DIR}")
    print(f"[dashboard]   live feeds from: {Path(args.archive)} (manifest.json + status.json)")
    print(f"[dashboard]   spectate embeds the Showdown client at http://localhost:8000")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
