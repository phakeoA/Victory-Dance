"""
server.py
=========
Local Flask backend for the Victory-Dance team builder UI.

Lives at data/scripts/; front-end files live in data/scripts/team_builder/

Run from anywhere:
  python data/scripts/server.py

Then open:
  http://localhost:5174/

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ENDPOINTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GET  /
    Serves team_builder.html

GET  /data/<path:filename>
    Serves any file under data/ (pokedex.json, moves.json, etc.)

POST /parse
    Body: multipart/form-data
      replay_html : the uploaded .html replay file
      known_teams : (optional) JSON string of what the user has
                    filled in so far for this battle, same shape
                    as a single battle entry in known_teams.json:
                    {
                      "p1": { "Kingambit": { ... } },
                      "p2": { ... },
                      "_meta": { "yourSide": "p1", "winner": "p1",
                                 "p1name": "...", "p2name": "..." }
                    }
    Returns: JSON preview dict from parse_replay_for_preview()

POST /export
    Body: JSON
      {
        "battle_id": "...",
        "known_teams_entry": { ... },   ← full annotated entry (user-approved)
        "replay_html": "...",           ← raw HTML string
      }
    Runs replay_to_transitions() and returns:
      { "transitions": [ ... ] }        ← list of JSONL-ready dicts
      Content-Disposition: attachment   ← browser triggers download

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import json
import sys
import tempfile
import os
from pathlib import Path

# ── Resolve directories from this file's location ───────────────────────────
# server.py lives at data/scripts/
# Front-end files live at data/scripts/team_builder/
_SCRIPTS_DIR  = Path(__file__).resolve().parent
_UI_DIR       = _SCRIPTS_DIR / "team_builder"   # HTML + CSS + JS modules
_PROJECT_ROOT = _SCRIPTS_DIR.parents[1]           # used for data/ file paths

if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# ── Flask ─────────────────────────────────────────────────────────────────────
try:
    from flask import Flask, request, jsonify, send_file, Response
    from flask_cors import CORS
except ImportError:
    raise ImportError(
        "Flask and flask-cors are required.\n"
        "Install with:  pip install flask flask-cors --break-system-packages"
    )

# ── Project imports (siblings in data/scripts/) ──────────────────────────────
# vod_parser is always required
from vod_parser import parse_replay_for_preview, replay_to_transitions

# belief_state and state_encoder are optional — they may not exist yet
try:
    from belief_state import BeliefState as _BeliefState
except ImportError:
    _BeliefState = None  # type: ignore

try:
    from state_encoder import StateEncoder as _StateEncoder
except ImportError:
    _StateEncoder = None  # type: ignore

# ── Globals (loaded once at startup) ─────────────────────────────────────────
_BELIEF_PATH = _PROJECT_ROOT / "data" / "pikalytics_regma.json"

_belief = None
_encoder = None

if _BeliefState is None:
    print("[warn] belief_state.py not found — Pikalytics inference unavailable.")
elif not _BELIEF_PATH.exists():
    print(
        f"[warn] Belief state file not found at {_BELIEF_PATH}.\n"
        "       Pikalytics inference will be unavailable until it exists."
    )
else:
    print(f"[startup] Loading belief state from {_BELIEF_PATH} ...")
    _belief = _BeliefState(_BELIEF_PATH)
    print("[startup] Belief state ready.")

if _StateEncoder is not None:
    _encoder = _StateEncoder()
else:
    print("[warn] state_encoder.py not found — encoding will be skipped.")

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=str(_UI_DIR), static_url_path="")
CORS(app)  # allow requests from file:// during local dev


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    """Serve the UI entry point from data/scripts/team_builder/."""
    return send_file(_UI_DIR / "team_builder.html")


@app.get("/data/<path:filename>")
def serve_data(filename):
    """
    Serve any file from the project's data/ directory.
    Allows team_builder.html to fetch data/pokedex.json, data/moves.json, etc.
    using the same relative paths regardless of how the page was opened.
    """
    target = (_PROJECT_ROOT / "data" / filename).resolve()
    # Safety: ensure the resolved path is still inside data/
    if not str(target).startswith(str((_PROJECT_ROOT / "data").resolve())):
        return jsonify({"error": "Forbidden"}), 403
    if not target.exists():
        return jsonify({"error": f"Not found: data/{filename}"}), 404
    return send_file(target)


@app.get("/health")
def health():
    """Simple liveness check used by the UI status dot."""
    return jsonify({
        "ok": True,
        "belief_loaded": _belief is not None,
        "encoder_loaded": _encoder is not None,
    })


@app.post("/parse")
def parse():
    """
    Accept an uploaded replay HTML + optional partial known_teams entry.
    Return the annotated preview dict so the UI can populate itself.
    """

    # ── HTML file ─────────────────────────────────────────────────────────────
    if "replay_html" not in request.files:
        return jsonify({"error": "No replay_html file in request."}), 400

    html_file = request.files["replay_html"]
    html_content = html_file.read().decode("utf-8", errors="replace")

    # ── Optional partial known_teams entry ────────────────────────────────────
    known_entry = None
    raw_known = request.form.get("known_teams")
    if raw_known:
        try:
            known_entry = json.loads(raw_known)
        except json.JSONDecodeError as exc:
            return jsonify({"error": f"known_teams JSON parse error: {exc}"}), 400

    # ── Run parser ────────────────────────────────────────────────────────────
    try:
        preview = parse_replay_for_preview(html_content, _belief, known_entry)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify(preview)


@app.post("/export")
def export():
    """
    Accept the fully annotated, user-approved battle entry + raw replay HTML.
    Run replay_to_transitions() and return the JSONL as a downloadable file.
    """

    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Expected JSON body."}), 400

    battle_id        = body.get("battle_id", "unknown")
    known_entry      = body.get("known_teams_entry", {})
    html_content     = body.get("replay_html", "")

    if not html_content:
        return jsonify({"error": "replay_html is empty."}), 400

    # Wrap known_entry under battle_id so replay_to_transitions can find it
    known_teams = {battle_id: known_entry}

    # Write HTML to temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", encoding="utf-8", delete=False
    ) as tmp:
        tmp.write(html_content)
        tmp_path = Path(tmp.name)

    try:
        # Determine which player perspective(s) to encode
        meta       = known_entry.get("_meta", {})
        your_side  = meta.get("yourSide")
        players    = [your_side] if your_side else ["p1", "p2"]

        transitions = replay_to_transitions(
            tmp_path, _belief, _encoder, players, known_teams
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        os.unlink(tmp_path)

    # ── Return as downloadable JSONL ──────────────────────────────────────────
    jsonl_bytes = "\n".join(json.dumps(t) for t in transitions).encode("utf-8")
    filename    = f"{battle_id}.jsonl"

    return Response(
        jsonl_bytes,
        mimetype="application/x-ndjson",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Transition-Count":  str(len(transitions)),
        },
    )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[server] Project root : {_PROJECT_ROOT}")
    print(f"[server] Scripts dir  : {_SCRIPTS_DIR}")
    print(f"[server] UI dir       : {_UI_DIR}")
    print(f"[server] Serving      : http://localhost:5174/")

    # Warn about any missing front-end files
    _REQUIRED_FILES = [
        "team_builder.html",
        "team_builder.css",
        "tb_constants.js",
        "tb_parser.js",
        "tb_api.js",
        "tb_render.js",
        "tb_actions.js",
    ]
    for _f in _REQUIRED_FILES:
        if not (_UI_DIR / _f).exists():
            print(f"[warn]   missing front-end file: team_builder/{_f}")

    app.run(host="127.0.0.1", port=5174, debug=True)
