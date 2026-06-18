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
from pathlib import Path

from flask import Flask, Response, jsonify, send_from_directory

_REPO = Path(__file__).resolve().parents[2]
_DASH_DIR = _REPO / "data" / "scripts" / "dashboard"
_ARCHIVE_DIR = _REPO / "artifacts" / "self_play_archive"

# only these are served from the dashboard dir (defense-in-depth alongside
# send_from_directory's own traversal protection)
_ALLOWED = {"dashboard.html", "dashboard.css", "dashboard.js", "demo_manifest.json"}

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


def create_app(dash_dir=_DASH_DIR, archive_dir=_ARCHIVE_DIR) -> Flask:
    dash_dir, archive_dir = Path(dash_dir), Path(archive_dir)
    app = Flask(__name__, static_folder=None)
    app.config["DASH_DIR"] = dash_dir
    app.config["ARCHIVE_DIR"] = archive_dir

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
