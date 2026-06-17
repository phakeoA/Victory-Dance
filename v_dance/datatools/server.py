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
        "source_type": "own_vod",       ← UI Type A/B/C/D selector (optional,
                                          defaults to Type B ranked_player_vod)
      }
    Type B exports BOTH perspectives (two ranked players = two BC targets
    per replay); other types export only _meta.yourSide.
    Runs replay_to_transitions() and returns:
      { "transitions": [ ... ] }        ← list of JSONL-ready dicts
      Content-Disposition: attachment   ← browser triggers download

POST /fill-beliefs
    Body: JSON
      {
        "vod_type": "B" | "ranked_player_vod" | "own_vod" | "bot_vod" | "self_play",
        "players": { "our_side": "p1", "p1": {"roster": [...]}, "p2": {...} },
        "revealed_info": { "p1:Kingambit": {...}, ... }   ← optional
      }
    Returns per-species inject-panel suggestions from Pikalytics
    (belief_state.ui_fill_suggestions).  Only invoked by the UI's
    "Use Belief Integration" button — never automatically.

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
# Parser package lives at v_dance/parser/vod_parser/   ← Python package
# Front-end files live at data/scripts/team_builder/
_SCRIPTS_DIR    = Path(__file__).resolve().parent          # v_dance/datatools/
_PROJECT_ROOT   = _SCRIPTS_DIR.parents[1]                  # repo root (v_dance/.. )
_VOD_PARSER_DIR = _PROJECT_ROOT / "v_dance" / "parser" / "vod_parser"  # path checks/warnings
_UI_DIR         = _PROJECT_ROOT / "data" / "scripts" / "team_builder"  # HTML + CSS + JS modules

# Add scripts/ to sys.path so `import vod_parser` resolves to the package,
# and sibling modules (belief_state, state_encoder, …) are also importable.
# ── Flask ─────────────────────────────────────────────────────────────────────
try:
    from flask import Flask, request, jsonify, send_file, Response
    from flask_cors import CORS
except ImportError:
    raise ImportError(
        "Flask and flask-cors are required.\n"
        "Install with:  pip install flask flask-cors --break-system-packages"
    )

# ── Project imports ───────────────────────────────────────────────────────────
# vod_parser is a package at data/scripts/vod_parser/__init__.py
from v_dance.parser.vod_parser import parse_replay_for_preview, replay_to_transitions

# belief_state and state_encoder are optional siblings in data/scripts/
try:
    from v_dance.parser.belief_state import BeliefState as _BeliefState
except ImportError:
    _BeliefState = None  # type: ignore

try:
    from v_dance.encoders.state_encoder import StateEncoder as _StateEncoder
except ImportError:
    _StateEncoder = None  # type: ignore

# ── Globals (loaded once at startup) ─────────────────────────────────────────
_BELIEF_PATH = _PROJECT_ROOT / "data" / "pikalytics_regma.json"

_belief = None
_belief_mtime = None
_encoder = None


def _get_belief():
    """Return the BeliefState, (re)loading it when the JSON changes on disk.

    A long-running server otherwise keeps serving beliefs scraped before
    pikalytics_regma.json was regenerated (made worse by Flask's debug
    reloader leaving orphan children alive across restarts) — mons missing
    from the stale data silently get no auto-fill suggestions.
    """
    global _belief, _belief_mtime
    if _BeliefState is None or not _BELIEF_PATH.exists():
        return _belief
    try:
        mtime = _BELIEF_PATH.stat().st_mtime
        if _belief is None or mtime != _belief_mtime:
            print(f"[belief] loading {_BELIEF_PATH} ...")
            _belief = _BeliefState(_BELIEF_PATH)
            _belief_mtime = mtime
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[warn] belief reload failed ({exc}) — keeping previous data.")
    return _belief


if _BeliefState is None:
    print("[warn] belief_state.py not found — Pikalytics inference unavailable.")
elif not _BELIEF_PATH.exists():
    print(
        f"[warn] Belief state file not found at {_BELIEF_PATH}.\n"
        "       Pikalytics inference will be unavailable until it exists."
    )
else:
    print(f"[startup] Loading belief state from {_BELIEF_PATH} ...")
    _get_belief()
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
    from v_dance.parser.vod_parser import get_pokedex
    return jsonify({
        "ok": True,
        "belief_loaded": _get_belief() is not None,
        "encoder_loaded": _encoder is not None,
        "pokedex_loaded": get_pokedex() is not None,
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
        preview = parse_replay_for_preview(html_content, _get_belief(), known_entry)
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
    source_type      = body.get("source_type")  # UI Type A/B/C/D selector

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
        # Determine which player perspective(s) to encode.  Type B (ranked
        # player VOD) always exports BOTH perspectives: each side is a ranked
        # player whose decisions are valid behavioural-cloning targets, so a
        # single replay yields double the transitions.  Other types keep the
        # yourSide restriction — e.g. a Type A opponent's perspective would
        # carry inverted stats quality (their side distribution, ours exact).
        from v_dance.parser.belief_state import VodType

        meta       = known_entry.get("_meta", {})
        your_side  = meta.get("yourSide")
        if VodType.coerce(source_type or "B") is VodType.B:
            players = ["p1", "p2"]
        else:
            players = [your_side] if your_side else ["p1", "p2"]

        transitions = replay_to_transitions(
            tmp_path, _get_belief(), _encoder, players, known_teams,
            source_type=source_type,
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


@app.post("/fill-beliefs")
def fill_beliefs():
    """
    Build Pikalytics auto-fill suggestions for the inject panel.
    Triggered manually by the UI's "Use Belief Integration" button.
    """
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Expected JSON body."}), 400
    belief = _get_belief()
    if belief is None:
        return jsonify({
            "error": "Belief data unavailable — belief_state.py or "
                     "data/pikalytics_regma.json missing on the server."
        }), 503

    try:
        from v_dance.parser.belief_state import ui_fill_suggestions
        result = ui_fill_suggestions(
            belief,
            vod_type=body.get("vod_type") or body.get("source_type") or "B",
            players=body.get("players") or {},
            revealed_info=body.get("revealed_info") or {},
            top_k=int(body.get("top_k") or 5),
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify(result)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[server] Project root   : {_PROJECT_ROOT}")
    print(f"[server] Scripts dir    : {_SCRIPTS_DIR}")
    print(f"[server] VOD parser pkg : {_VOD_PARSER_DIR}")
    print(f"[server] UI dir         : {_UI_DIR}")
    print(f"[server] Serving        : http://localhost:5174/")

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
