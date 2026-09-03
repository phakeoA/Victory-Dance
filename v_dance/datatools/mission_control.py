"""Mission Control — ONE local page over the whole Victory-Dance pipeline.

Design: docs/mission_control_ui_design.md (2026-07-12).

    python -m v_dance.datatools.mission_control            # serve + open the page
    python -m v_dance.datatools.mission_control --no-browser --port 8990

Deliberately stdlib-only and torch-free (same pattern as play/bot_control_ui.py): the
server starts instantly, never fights the Flask debug reloader, and can run alongside
anything. It knows the project's entry points through a typed COMMAND REGISTRY. Every entry is a
launchable managed job (argv list, never shell=True; logs under artifacts/logs/mc_*.log;
stop = Windows tree-kill `taskkill /T /F`); the page ALSO renders the exact `PYTHONUTF8=1
.venv/Scripts/python.exe -X utf8 -u -m … | tee …` one-liner for a terminal ("Copy command").

  · heavy entries  → launchable from the UI (the USER clicks = the USER launches), but
    ONE heavy run at a time — _JOBS.start refuses a 2nd heavy launch while one is alive
    (RTX 3070 Ti 8GB / 32GB RAM). Training / exploiter stream live progress (epoch/loss/
    metric or games-trained/WR), parsed from the log tail by _job_progress.
  · light entries  → launch freely; concurrency-capped only by the box.

Security: binds 127.0.0.1 only; /api/status exposes a strict .env WHITELIST (PS_PASSWORD
never leaves the process); /api/env writes only whitelisted deploy keys via the same
atomic tmp-replace as bot_control_ui.save_format; job options are validated server-side
(teams/ckpts must exist on disk, numbers clamped, free text is a single argv element).
"""
from __future__ import annotations

import argparse
import json
import re
import os
import socket
import subprocess
import threading
import time
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

_REPO = Path(__file__).resolve().parents[2]
_ENV_PATH = _REPO / ".env"
_LOGS_DIR = _REPO / "artifacts" / "logs"
_HTML_PATH = _REPO / "data" / "scripts" / "mission_control" / "mission_control.html"
_PY = str(_REPO / ".venv" / "Scripts" / "python.exe")
_FORMATS_REGISTRY = _REPO / "data" / "regulations" / "vod_parser_format_names.json"
_MODEL_IO = _REPO / "v_dance" / "play" / "model_io.py"
_BC_CKPT_ROOT = _REPO / "ai_train_scripts" / "BC_model"
_TP_CKPT_ROOT = _REPO / "ai_train_scripts" / "teamPreview_model"
_CHAMPIONS_DIR = _REPO / "teams" / "Champions"

# .env keys the browser may SEE (never PS_PASSWORD) and the subset it may WRITE.
_ENV_READ_KEYS = ("PS_USERNAME", "PS_AVATAR", "PS_CLIENT_URL", "VDANCE_BATTLE_FORMAT",
                  "VD_BATTLE_CKPT", "VD_TP_CKPT", "VD_DEFAULT_TEAM", "VD_AUTO_CLOSE_ROOMS",
                  "VD_SERVE_TAU", "VD_SERVE_TOP_P",
                  # 2026-09-02 serve mode: bandit on/off + the launch-time pin (frozen arm)
                  "VD_BANDIT", "VD_BANDIT_PIN",
                  # 2026-09-02 ladder lanes: rated games at once at launch (1..5)
                  "VD_LADDER_LANES",
                  # 2026-09-02 (USER): battle-timer mode at launch (1 = /timer on at the first frame)
                  "VD_TIMER_IMMEDIATE",
                  # 2026-09-03 (USER): open team sheets at launch (1 = accept the offer)
                  "VD_OTS_ACCEPT")
_ENV_WRITE_KEYS = ("VDANCE_BATTLE_FORMAT", "VD_BATTLE_CKPT", "VD_TP_CKPT",
                   "VD_DEFAULT_TEAM", "VD_AUTO_CLOSE_ROOMS",
                   "VD_SERVE_TAU", "VD_SERVE_TOP_P",
                   "VD_BANDIT", "VD_BANDIT_PIN", "VD_LADDER_LANES", "VD_TIMER_IMMEDIATE", "VD_OTS_ACCEPT")

def _bandit_arm_names() -> list:
    """Arm names from config/serve_bandit.json (names only — the launch-default pin picker; the
    bot validates checkpoints itself at launch). [] when the config is absent/unreadable."""
    try:
        cfg = json.loads((_REPO / "config" / "serve_bandit.json").read_text(encoding="utf-8"))
        return [str(a["name"]) for a in (cfg.get("arms") or []) if a.get("name")]
    except Exception:
        return []


# well-known local services (display + start/stop for the two datatools servers)
_SERVICES = {
    "showdown":     {"port": 8000, "label": "Local Showdown server",
                     "note": "battle harnesses start/stop it themselves"},
    "team_builder": {"port": 5174, "label": "Team builder", "job": "svc_team_builder"},
    "dashboard":    {"port": 5175, "label": "Self-play dashboard", "job": "svc_dashboard"},
    "bot_panel":    {"port": 8777, "label": "Bot control panel",
                     "note": "lives inside the online bot process"},
}


# ── .env ──────────────────────────────────────────────────────────────────────
def _env_read() -> dict:
    out: dict = {}
    if _ENV_PATH.is_file():
        for ln in _ENV_PATH.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if ln and not ln.startswith("#") and "=" in ln:
                k, v = ln.split("=", 1)
                out[k.strip()] = v.strip()
    return out


def _env_write(key: str, value: str) -> None:
    """Atomic single-key .env update (mirrors bot_control_ui.save_format)."""
    if key not in _ENV_WRITE_KEYS:
        raise ValueError(f"key {key!r} is not editable from the panel")
    value = (value or "").strip()
    if not value or any(c in value for c in "\r\n"):
        raise ValueError("empty/invalid value")
    lines = (_ENV_PATH.read_text(encoding="utf-8").splitlines()
             if _ENV_PATH.is_file() else [])
    for i, ln in enumerate(lines):
        if ln.split("=", 1)[0].strip() == key:
            lines[i] = f"{key}={value}"
            break
    else:
        lines.append(f"{key}={value}")
    tmp = _ENV_PATH.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(_ENV_PATH)


# ── inventories (pure filesystem — no v_dance imports) ───────────────────────
def _known_formats() -> list:
    try:
        js = json.loads(_FORMATS_REGISTRY.read_text(encoding="utf-8"))
        return [f["id"] for f in js.get("formats", []) if f.get("id")]
    except Exception:
        return []


def _known_formats_full() -> list:
    """[{id, name}] from the registry — for a readable format picker (future-proof: add a format to
    data/regulations/vod_parser_format_names.json and it appears automatically)."""
    try:
        js = json.loads(_FORMATS_REGISTRY.read_text(encoding="utf-8"))
        return [{"id": f["id"], "name": f.get("name") or f["id"]}
                for f in js.get("formats", []) if f.get("id")]
    except Exception:
        return []


def _active_format(env: dict) -> str:
    return env.get("VDANCE_BATTLE_FORMAT") or (_known_formats() or ["gen9championsvgc2026regmb"])[0]


def _reg_subdir(fmt: str):
    """teams/Champions/<REG>/ whose letters match the format's reg suffix (regmb → M-B)."""
    m = re.search(r"reg([a-z]+)$", (fmt or "").lower())
    want = m.group(1) if m else None
    if want and _CHAMPIONS_DIR.is_dir():
        for d in _CHAMPIONS_DIR.iterdir():
            if d.is_dir() and re.sub(r"[^a-z]", "", d.name.lower()) == want:
                return d
    return None


def _discover_teams(fmt: str) -> list:
    """Team NAMES for the format (reg subfolder when present, else all of Champions/) —
    mirrors run_local_battle.discover_teams semantics without importing the play stack."""
    base = _reg_subdir(fmt) or _CHAMPIONS_DIR
    if not base.is_dir():
        return []
    out = []
    for p in sorted(base.rglob("*")):
        if (p.is_file() and not p.name.startswith(".")
                and p.suffix.lower() not in (".json", ".md")
                and not p.name.lower().startswith("readme")):
            out.append(p.name)
    return out


def _ckpt_inventory() -> dict:
    """{'battle': [{path,dir,file,mtime,mb}], 'tp': [...]} — repo-relative POSIX paths."""
    def scan(root: Path) -> list:
        rows = []
        if root.is_dir():
            for pt in sorted(root.glob("checkpoints*/**/*.pt")):
                st = pt.stat()
                rows.append({"path": pt.relative_to(_REPO).as_posix(),
                             "dir": pt.parent.name, "file": pt.name,
                             "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime)),
                             "mb": round(st.st_size / 1e6, 1)})
        return rows
    return {"battle": scan(_BC_CKPT_ROOT), "tp": scan(_TP_CKPT_ROOT)}


def _model_io_defaults() -> dict:
    """Parse DEFAULT_BC_CHECKPOINT / DEFAULT_TP_CHECKPOINT from model_io.py SOURCE (no import)."""
    out = {"battle": None, "tp": None}
    try:
        src = _MODEL_IO.read_text(encoding="utf-8")
        for key, name in (("battle", "DEFAULT_BC_CHECKPOINT"), ("tp", "DEFAULT_TP_CHECKPOINT")):
            m = re.search(rf'^{name}\s*=\s*(.+)$', src, re.M)
            if m:
                parts = re.findall(r'"([^"]+)"', m.group(1))
                if parts:
                    out[key] = "ai_train_scripts/" + "/".join(parts)
    except Exception:
        pass
    return out


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.15)
        return s.connect_ex(("127.0.0.1", port)) == 0


# ── online bot control-panel proxy (front the :8777 panel from the master UI) ──
# The panel (play/bot_control_ui.py) lives INSIDE the running online bot process — it needs the
# live Playwright page. Mission Control can't hold that page, so it PROXIES the panel's HTTP API:
# the Online tab drives ladder runs / auto-accept / challenges through here, so the USER never
# opens :8777 separately. Panel binds the first free port in [8777, 8786].
_PANEL_PORTS = range(8777, 8787)
_PANEL_POST_OK = {"/api/ladder/start", "/api/ladder/stop", "/api/challenge",
                  "/api/challenge/cancel", "/api/options", "/api/team", "/api/format"}
_panel_cache = {"port": None}


def _panel_status() -> dict:
    """The live online-bot panel status (its own /api/status), or {'up': False}. Probes the
    cached port first, then the range; identifies the panel by its status shape (run + tally)."""
    order = ([_panel_cache["port"]] if _panel_cache["port"] else [])
    order += [p for p in _PANEL_PORTS if p != _panel_cache["port"]]
    for p in order:
        if not p or not _port_open(p):
            continue
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{p}/api/status", timeout=0.6) as r:
                js = json.loads(r.read().decode("utf-8"))
            if isinstance(js, dict) and "run" in js and "tally" in js:
                _panel_cache["port"] = p
                return {"up": True, "port": p, **js}
        except Exception:
            continue
    _panel_cache["port"] = None
    return {"up": False}


def _dashboard_status() -> dict:
    """2026-09-03 (USER: "embed the dashboard so I can see it better"): the self-play dashboard's run
    summary — its ``/run_info.json`` (the run its feeds follow + that run's ``status.json``) — for the
    Dashboard tab's run strip, or ``{'up': False}`` when :5175 is not answering."""
    port = _SERVICES["dashboard"]["port"]
    if not _port_open(port):
        return {"up": False}
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/run_info.json", timeout=0.8) as r:
            js = json.loads(r.read().decode("utf-8"))
        return {"up": True, **js} if isinstance(js, dict) else {"up": False}
    except Exception:
        return {"up": False}


def _panel_post(path: str, body: dict) -> dict:
    if path not in _PANEL_POST_OK:
        raise ValueError(f"not a panel endpoint: {path}")
    port = _panel_cache["port"] or _panel_status().get("port")
    if not port:
        raise ValueError("online bot panel not found — is the online bot running?")
    data = json.dumps(body or {}).encode("utf-8")
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=35) as r:
        return json.loads(r.read().decode("utf-8"))


# ── team-builder (:5174) bridge: species list (for autocomplete), generate proxy, save ─────
_TB_PORT = 5174
_species_cache = {"names": None, "mtime": None}


def _species_list() -> list:
    """Canonical belief species (display names) for the active reg — the exact set the generator
    accepts. Read straight from the belief JSON (torch-free), mtime-cached."""
    env = _env_read()
    m = re.search(r"reg([a-z]+)$", _active_format(env).lower())
    path = _REPO / "data" / f"pikalytics_reg{m.group(1)}.json" if m else None
    if path is None or not path.is_file():
        path = _REPO / "data" / "pikalytics_regmb.json"
    try:
        mt = path.stat().st_mtime
    except OSError:
        return []
    if _species_cache["names"] is None or _species_cache["mtime"] != mt:
        try:
            pk = (json.loads(path.read_text(encoding="utf-8")) or {}).get("pokemon") or {}
            _species_cache["names"] = sorted(pk.keys()) if isinstance(pk, dict) else []
            _species_cache["mtime"] = mt
        except Exception:
            _species_cache["names"] = []
    return _species_cache["names"] or []


def _tb_post(path: str, body: dict) -> dict:
    """Proxy a POST to the team-builder server (:5174), which owns the belief + validator + TP model."""
    if not _port_open(_TB_PORT):
        raise ValueError("the team builder (:5174) is not running — start it from the Parser tab")
    data = json.dumps(body or {}).encode("utf-8")
    req = urllib.request.Request(f"http://127.0.0.1:{_TB_PORT}{path}", data=data,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as r:   # generate validates each team via node
        return json.loads(r.read().decode("utf-8"))


def _save_team(name: str, paste: str) -> str:
    """Write a paste to teams/Champions/<active-reg>/<name> (name sanitized, no traversal)."""
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", (name or "").strip())
    if not safe:
        raise ValueError("team name required")
    if not (paste or "").strip():
        raise ValueError("paste required")
    champ = _REPO / "teams" / "Champions"
    m = re.search(r"reg([a-z]+)$", _active_format(_env_read()).lower())
    want = m.group(1) if m else None
    subdir = champ
    if want and champ.is_dir():
        for dd in champ.iterdir():
            if dd.is_dir() and re.sub(r"[^a-z]", "", dd.name.lower()) == want:
                subdir = dd
                break
    target = (subdir / safe).resolve()
    if not str(target).startswith(str(champ.resolve())):
        raise ValueError("bad team name")
    subdir.mkdir(parents=True, exist_ok=True)
    target.write_text(paste, encoding="utf-8")
    return target.relative_to(_REPO).as_posix()


def _exploit_summary() -> dict:
    """Newest exploit_curve.json under artifacts/logs/exploit_*/ (the 4a G3 meter)."""
    best, best_mtime = None, -1.0
    for p in _LOGS_DIR.glob("exploit_*/exploit_curve.json"):
        mt = p.stat().st_mtime
        if mt > best_mtime:
            best, best_mtime = p, mt
    if best is None:
        return {"found": False}
    try:
        return {"found": True, "path": best.relative_to(_REPO).as_posix(),
                "updated": time.strftime("%Y-%m-%d %H:%M", time.localtime(best_mtime)),
                "curve": json.loads(best.read_text(encoding="utf-8"))}
    except Exception as exc:
        return {"found": True, "path": best.relative_to(_REPO).as_posix(), "error": str(exc)}


def _recent_logs(n: int = 40) -> list:
    if not _LOGS_DIR.is_dir():
        return []
    files = [p for p in _LOGS_DIR.iterdir() if p.is_file() and p.suffix in (".log", ".json")]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [{"name": p.name,
             "mtime": time.strftime("%m-%d %H:%M", time.localtime(p.stat().st_mtime)),
             "kb": round(p.stat().st_size / 1024, 1)} for p in files[:n]]


# ── command registry ──────────────────────────────────────────────────────────
# opt types: team | ckpt | tpckpt | int | float | text | flag | choice
# heavy=True → copy-only (server refuses to launch). envopts → optional env toggles.
REGISTRY = [
    # ---- PLAY ----
    dict(id="play_online", cat="play", heavy=False, single=True, title="Online bot (browser)",
         module="v_dance.play.play_online_browser",
         desc="Logs into play.pokemonshowdown.com. Drive it live from the Online bot tab — set the "
              "format + launch there (format writes .env), then ladder runs, auto-accept, private "
              "challenges — the control panel is embedded there (no separate window). Bo3 venues: "
              "set TP ckpt to checkpoints_setctx.",
         opts=[dict(name="ai-team", type="team", label="Team pin (optional)"),
               dict(name="dossier", type="flag", label="--dossier (S1 belief warm-start)", default=True),
               dict(name="adapt-rules", type="flag", label="--adapt-rules", default=True),
               dict(name="dry-run", type="flag", label="--dry-run (login + teams, no battles)"),
               dict(name="bench-note", type="text", label="--bench-note", default="online"),
               dict(name="ckpt", type="ckpt", label="Battle ckpt override (optional)"),
               dict(name="tp-ckpt", type="tpckpt", label="TP ckpt override (optional)")],
         envopts=[dict(key="VD_ROUTE_TEAMS", label="VD_ROUTE_TEAMS=1 (4b router, challenge-accepts)"),
                  dict(key="VD_TP_OTS_OVERLAY", label="VD_TP_OTS_OVERLAY=1 (force the TP sheet overlay ON; unset = AUTO — a certified OTS-trained TP ckpt already uses the sheets)")]),
    dict(id="play_vs_human_browser", cat="play", heavy=False, title="Play vs human (browser, local)",
         module="v_dance.play.play_vs_human_browser",
         desc="Local Showdown in a headed browser: you in one tab, the bot in the other.",
         opts=[dict(name="ai-team", type="team", label="AI team (optional)"),
               dict(name="human-name", type="text", label="--human-name", default="Challenger"),
               dict(name="dossier", type="flag", label="--dossier"),
               dict(name="adapt-rules", type="flag", label="--adapt-rules"),
               dict(name="bench-note", type="text", label="--bench-note", default="local-browser"),
               dict(name="ckpt", type="ckpt", label="Battle ckpt override (optional)"),
               dict(name="tp-ckpt", type="tpckpt", label="TP ckpt override (optional)")]),
    dict(id="play_vs_human", cat="play", heavy=False, title="Play vs human (terminal serve)",
         module="v_dance.play.play_vs_human",
         desc="Serves battles on the local Showdown server until Ctrl-C (you join from any browser).",
         opts=[dict(name="mode", type="choice", label="--mode", choices=["random", "choose"], default="random"),
               dict(name="ai-team", type="team", label="AI team (choose mode)"),
               dict(name="human-team", type="team", label="Your team (choose mode)"),
               dict(name="n-battles", type="int", label="--n-battles (blank = until Ctrl-C)", min=1, max=10000),
               dict(name="dossier", type="flag", label="--dossier"),
               dict(name="bench-note", type="text", label="--bench-note", default="")]),
    dict(id="run_local_battle", cat="play", heavy=False, title="AI vs AI local battle (spectate)",
         module="v_dance.play.run_local_battle",
         desc="Bot vs bot on the local server; a spectator tab opens so you can watch.",
         opts=[dict(name="battles", type="int", label="--battles", default=1, min=1, max=100),
               dict(name="team1", type="team", label="--team1 (optional)"),
               dict(name="team2", type="team", label="--team2 (optional)"),
               dict(name="no-spectate", type="flag", label="--no-spectate")]),
    dict(id="play_ladder", cat="play", heavy=False, title="Ladder (direct websocket, no browser)",
         module="v_dance.play.play_ladder",
         desc="Autonomous ladder mode over the raw websocket (no Playwright, no panel).",
         opts=[dict(name="games", type="int", label="--games", default=1, min=1, max=200),
               dict(name="team", type="team", label="--team (optional)"),
               dict(name="bench-note", type="text", label="--bench-note", default="ladder")]),

    # ---- TRAIN (heavy → copy-only) ----
    dict(id="train_bc", cat="train", heavy=True, title="Battle net retrain (era N+1)",
         module="v_dance.training.train_bc",
         desc="Era-chain rule: warm-start/value base = the GATED era-N net. Era-2 recipe = _adv "
              "recipe + Type_C winner-only + --rating-weight + blend belief. Gates after: suite, "
              "ruler vs pinned val, type-eff probe, ladder block.",
         opts=[dict(name="warm-start", type="ckpt", label="--warm-start (gated era-N net)",
                    default="ai_train_scripts/BC_model/checkpoints_attn_era2/battle_base.pt"),
               dict(name="epochs", type="int", label="--epochs", default=30, min=1, max=200),
               dict(name="rating-weight", type="flag", label="--rating-weight", default=True),
               dict(name="mmap-cache", type="flag", label="--mmap-cache"),
               dict(name="loader-workers", type="int", label="--loader-workers (M6)", default=4, min=0, max=16),
               dict(name="patience", type="int", label="--patience", default=5, min=0, max=50),
               dict(name="out", type="text", label="--out",
                    default="ai_train_scripts/BC_model/checkpoints_attn_era3")],
         note="Data folders default to the full corpus inside train_bc; add --data … to override."),
    dict(id="tp_set_head", cat="train", heavy=True, title="TP set-head retrain",
         module="v_dance.training.train_teampreview",
         desc="The deployed contrastive set head (artifacts/tp_set_head_train.sh, verified).",
         fixed=["--features", "sbda", "--self-attn", "--cross-attn", "--teammate-bias", "--set-head",
                "--warm-start", "ai_train_scripts/teamPreview_model/checkpoints/teampreview_sbda.pt",
                "--belief", "data/pikalytics_regmb_blend_2026-07-11.json",
                "--data", "data/vods/Prepared_training_data/Regulation_MA/Jsonl_TypeA",
                "data/vods/Prepared_training_data/Regulation_MA/Jsonl_TypeB",
                "data/vods/Prepared_training_data/Regulation_MB/Jsonl_TypeB",
                "--epochs", "60", "--patience", "5",
                "--out", "ai_train_scripts/teamPreview_model/checkpoints_set"],
         opts=[],
         note="⚠ TP gate confound: gate on the trainer's OWN split. Watch lead-exact separately "
              "if the data mix changes (HF shifts lead conventions)."),
    dict(id="tp_set_ctx", cat="train", heavy=True, title="TP set-ctx retrain (Bo3)",
         module="v_dance.training.train_teampreview",
         desc="Bo3 previous-game conditioning (artifacts/tp_set_ctx_train.sh, verified). "
              "Deployed to Bo3 venues only (--tp-ckpt checkpoints_setctx).",
         fixed=["--features", "sbda", "--self-attn", "--cross-attn", "--teammate-bias", "--set-head",
                "--set-ctx",
                "--warm-start", "ai_train_scripts/teamPreview_model/checkpoints_set/teampreview_sbda.pt",
                "--belief", "data/pikalytics_regmb_blend_2026-07-11.json",
                "--data", "data/vods/Prepared_training_data/Regulation_MA/Jsonl_TypeA",
                "data/vods/Prepared_training_data/Regulation_MA/Jsonl_TypeB",
                "data/vods/Prepared_training_data/Regulation_MB/Jsonl_TypeB",
                "data/vods/Prepared_training_data/Regulation_MB_Bo3/Jsonl_HF_OTS",
                "--epochs", "60", "--patience", "5",
                "--out", "ai_train_scripts/teamPreview_model/checkpoints_setctx"],
         opts=[]),
    dict(id="exploiter", cat="train", heavy=True, title="4a exploiter meter (~8h)",
         module="v_dance.selfplay.exploiter",
         desc="Worst-case G3 meter vs a FROZEN deploy net. Curve → artifacts/logs/exploit_<ckpt>/"
              "exploit_curve.json. NEVER feed exploiter games into training. League stays frozen "
              "(no gate/HoF; pfsp_decay=1.0).",
         opts=[dict(name="target-ckpt", type="ckpt", label="--target-ckpt (the frozen prey)",
                    default="ai_train_scripts/BC_model/checkpoints_attn_era2/battle_base.pt", required=True),
               dict(name="team", type="team", label="--team (mirror)", default="maw_zard"),
               dict(name="budget-hours", type="float", label="--budget-hours", default=8.0, min=0.1, max=48),
               dict(name="workers", type="int", label="--workers", default=8, min=1, max=16)]),
    dict(id="exploiter_smoke", cat="train", heavy=False, title="Exploiter --smoke (quick check)",
         module="v_dance.selfplay.exploiter",
         desc="Tiny smoke run of the exploiter plumbing. Needs port 8000 free (no other run).",
         fixed=["--smoke"],
         opts=[dict(name="target-ckpt", type="ckpt", label="--target-ckpt",
                    default="ai_train_scripts/BC_model/checkpoints_attn_era2/battle_base.pt", required=True)]),
    dict(id="selfplay_gen", cat="train", heavy=True, title="Self-play generations (PPO loop)",
         module="v_dance.selfplay.generation",
         desc="The full self-play loop. Prefer the dashboard's launcher tab (:5175) — it has the "
              "config wizard and live console. ⚠ always set save_replays:true in run configs.",
         opts=[dict(name="wizard", type="flag", label="--wizard (interactive config)", default=True)]),

    # ---- DATA ----
    dict(id="scrape_pika", cat="data", heavy=True, title="Scrape Pikalytics (MANUAL only)",
         script="data/scripts/scrapers/scrape_pikalytics.py",
         desc="⚠ Never run automatically (standing rule). A belief change ⇒ TP/battle RE-EXPORT "
              "of affected corpora before any retrain.",
         opts=[]),
    dict(id="observed_meta", cat="data", heavy=False, title="Rebuild observed meta (from dossiers)",
         module="v_dance.datatools.observed_meta", opts=[]),
    dict(id="belief_blend", cat="data", heavy=False, title="Blend belief (scrape + observed)",
         module="v_dance.datatools.belief_blend",
         desc="Writes data/pikalytics_<reg>_blend_<date>.json. Deploying a new blend = copy over "
              "pikalytics_regmb.json (keep parity pinned) — a belief change ⇒ re-export.",
         opts=[dict(name="k", type="float", label="--k (blend strength)", default=20.0, min=0, max=1000)]),
    dict(id="bulk_parse", cat="data", heavy=True, title="Bulk-parse replays → JSONL",
         module="v_dance.datatools.bulk_parse_replays",
         desc="Re-export path for parser/belief changes. Type B exports BOTH perspectives.",
         opts=[dict(name="input", type="text", label="--input (folder)"),
               dict(name="output", type="text", label="--output (folder)"),
               dict(name="type", type="choice", label="--type", choices=["A", "B", "C", "D"], default="B"),
               dict(name="winner-only", type="flag", label="--winner-only"),
               dict(name="recursive", type="flag", label="--recursive", default=True)]),
    dict(id="ingest_hf", cat="data", heavy=True, title="Ingest HF VGC-Bench logs",
         module="v_dance.datatools.ingest_hf_logs",
         desc="Champions MA/MB subsets only (Gen9-standard/tera is a different game — HELD).",
         opts=[dict(name="input", type="text", label="--input", required=True),
               dict(name="output", type="text", label="--output", required=True),
               dict(name="ots-mode", type="choice", label="--ots-mode", choices=["open", "closed"], default="open"),
               dict(name="tp-only", type="flag", label="--tp-only"),
               dict(name="workers", type="int", label="--workers", default=1, min=1, max=8)]),
    dict(id="filter_type_c", cat="data", heavy=False, title="Type-C filter (dry-run)",
         module="v_dance.datatools.filter_type_c",
         desc="Report-only scan (no --apply). Run the --apply/--quarantine variant from a terminal.",
         opts=[]),
    dict(id="corpus_qa", cat="data", heavy=False, title="Corpus QA",
         module="v_dance.datatools.corpus_qa",
         opts=[dict(name="strict", type="flag", label="--strict")]),
    dict(id="seed_router", cat="data", heavy=False, title="Seed router priors (4b + matrix sidecar)",
         script="scratch/seed_router_priors.py",
         desc="Re-seeds data/router_priors.json AND writes data/router_matrix.json (M4 sidecar).",
         opts=[]),

    # ---- EVAL ----
    dict(id="pytest", cat="eval", heavy=False, title="Full pytest suite",
         raw=["-m", "pytest", "tests", "-q", "-rs"],
         desc="The suite gate. Expect EXIT 0 with 4 stale-v17 test_model_io SKIPs.", opts=[]),
    dict(id="tp_val", cat="eval", heavy=False, title="TP val report",
         module="v_dance.eval.tp_val_report",
         opts=[dict(name="ckpt", type="tpckpt", label="--ckpt", required=True,
                    default="ai_train_scripts/teamPreview_model/checkpoints_set/teampreview_sbda.pt"),
               dict(name="set-ab", type="flag", label="--set-ab (greedy vs set-decode gate)")]),
    dict(id="bc_val", cat="eval", heavy=False, title="BC val report (ruler)",
         module="v_dance.eval.bc_val_report",
         opts=[dict(name="ckpt", type="ckpt", label="--ckpt", required=True,
                    default="ai_train_scripts/BC_model/checkpoints_attn_era2/battle_base.pt")]),
    dict(id="bench_report", cat="eval", heavy=False, title="Human benchmark report",
         module="v_dance.eval.human_benchmark_report",
         desc="Elo-adjusted performance per --bench-note block (the goal metric).",
         opts=[dict(name="note", type="text", label="--note (block tag, optional)"),
               dict(name="min-games", type="int", label="--min-games", default=1, min=1, max=1000)]),
    dict(id="gauntlet", cat="eval", heavy=False, title="Gauntlet (scripted opponents)",
         module="v_dance.eval.gauntlet",
         desc="Short CPU eval vs scripted opponents. Keep it small from the panel (≤50).",
         opts=[dict(name="battles", type="int", label="--battles", default=20, min=1, max=50),
               dict(name="ckpt", type="ckpt", label="--ckpt (optional)")]),
    dict(id="type_eff", cat="eval", heavy=False, title="Type-effectiveness probe",
         script="scratch/type_eff_probe.py",
         desc="Reusable acceptance gate: the battle net must RESPECT type effectiveness (graded).",
         opts=[]),

    # ---- SERVICES ----
    dict(id="svc_team_builder", cat="svc", heavy=False, title="Team builder server (:5174)",
         module="v_dance.datatools.server", opts=[]),
    dict(id="svc_dashboard", cat="svc", heavy=False, title="Self-play dashboard (:5175)",
         module="v_dance.datatools.dashboard_server", opts=[]),
]
# Every PLAY command gets a Format picker (blank = the deployed .env default). Done in one place so
# a new play command inherits it. type="format" is injected as the VDANCE_BATTLE_FORMAT env var at
# launch (spawn-safe — every harness reads it at import), NOT as an argv flag.
# EXCEPT play_online: its format is chosen in the Online-bot tab, which writes VDANCE_BATTLE_FORMAT
# to .env (play_online_browser binds the format from .env at import) — so no per-launch format field
# on its Play card (2026-07-14).
for _e in REGISTRY:
    if _e.get("cat") == "play" and _e["id"] != "play_online":
        _e.setdefault("opts", []).insert(0, dict(
            name="format", type="format", label="Format (blank = deployed default)"))

_REG_BY_ID = {e["id"]: e for e in REGISTRY}

# ── live progress parsers (per command, from the streamed log) ─────────────────
_EPOCH_RE = r"epoch\s+(\d+)\s*\|"
_PROGRESS = {
    "train_bc": {"epoch_re": _EPOCH_RE, "total_arg": "--epochs",
                 "metrics": [("val top1", r"val loss [\d.]+ top1 ([\d.]+)"),
                             ("train loss", r"train loss ([\d.]+)"),
                             ("gim recall", r"gim recall ([\d.]+)")]},
    "tp_set_head": {"epoch_re": _EPOCH_RE, "total_arg": "--epochs",
                    "metrics": [("set exact", r"set exact ([\d.]+)"),
                                ("bring exact", r"bring exact ([\d.]+)"),
                                ("lead exact", r"lead exact ([\d.]+)")]},
    "tp_set_ctx": {"epoch_re": _EPOCH_RE, "total_arg": "--epochs",
                   "metrics": [("set exact", r"set exact ([\d.]+)"),
                               ("bring exact", r"bring exact ([\d.]+)"),
                               ("lead exact", r"lead exact ([\d.]+)")]},
    "exploiter": {"mode": "exploiter"},
}


def _read_tail(path: Path, nbytes: int = 90_000) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - nbytes))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def _arg_val(argv: list, flag: str) -> Optional[str]:
    try:
        return argv[argv.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def _job_progress(entry_id: str, argv: list, log_path: Path) -> Optional[dict]:
    """Structured live progress for a running/finished job, parsed from its log tail.
    None when the command has no known progress format (still shows the raw tail)."""
    spec = _PROGRESS.get(entry_id)
    if not spec:
        return None
    text = _read_tail(log_path)
    if not text:
        return None
    if spec.get("mode") == "exploiter":
        tot = re.findall(r"\(total (\d+)\)", text)
        wr = re.findall(r"trained: WR ([\d.]+)", text)
        wall = re.findall(r"([\d.]+)h\)", text)
        budget = _arg_val(argv, "--budget-hours")
        pct = None
        if budget and wall:
            try:
                pct = min(100, round(100 * float(wall[-1]) / float(budget)))
            except ValueError:
                pass
        return {"label": "games trained", "value": tot[-1] if tot else "0", "pct": pct,
                "metrics": {"exploit WR": wr[-1] if wr else "—",
                            "wall h": wall[-1] if wall else "—"}}
    ep = re.findall(spec["epoch_re"], text)
    cur = int(ep[-1]) if ep else 0
    total = None
    tv = _arg_val(argv, spec.get("total_arg", "--epochs"))
    if tv:
        try:
            total = int(tv)
        except ValueError:
            total = None
    metrics = {}
    for label, rx in spec["metrics"]:
        m = re.findall(rx, text)
        if m:
            metrics[label] = m[-1]
    return {"label": "epoch", "value": f"{cur}/{total}" if total else str(cur),
            "pct": round(100 * cur / total) if total else None, "metrics": metrics}


# ── option validation + argv build ────────────────────────────────────────────
def _build_argv(entry: dict, options: dict, teams: list, ckpts: dict) -> list:
    argv = [_PY, "-X", "utf8", "-u"]
    if entry.get("module"):
        argv += ["-m", entry["module"]]
    elif entry.get("script"):
        sp = (_REPO / entry["script"])
        if not sp.is_file():
            raise ValueError(f"script missing: {entry['script']}")
        argv.append(str(sp))
    elif entry.get("raw"):
        argv += list(entry["raw"])
    argv += [str(a) for a in entry.get("fixed", [])]

    battle_paths = {r["path"] for r in ckpts["battle"]}
    tp_paths = {r["path"] for r in ckpts["tp"]}
    for opt in entry.get("opts", []):
        name, typ = opt["name"], opt["type"]
        if typ == "format":
            continue                                # handled as the VDANCE_BATTLE_FORMAT env var, not argv
        raw = options.get(name, None)
        if raw is None and "default" in opt:      # apply the registry default SERVER-SIDE so a
            raw = opt["default"]                    # launch is robust even if the client omits it
        raw = raw.strip() if isinstance(raw, str) else raw
        if raw in ("", None, False):
            if opt.get("required"):
                raise ValueError(f"--{name} is required")
            continue                                # flag default True falls through to the flag branch
        if typ == "flag":
            argv.append(f"--{name}")
        elif typ == "int":
            v = max(opt.get("min", 0), min(int(raw), opt.get("max", 10 ** 9)))
            argv += [f"--{name}", str(v)]
        elif typ == "float":
            v = max(opt.get("min", 0.0), min(float(raw), opt.get("max", 1e9)))
            argv += [f"--{name}", str(v)]
        elif typ == "choice":
            if raw not in opt["choices"]:
                raise ValueError(f"--{name}: invalid choice {raw!r}")
            argv += [f"--{name}", raw]
        elif typ == "team":
            if raw not in teams:
                raise ValueError(f"--{name}: unknown team {raw!r}")
            argv += [f"--{name}", raw]
        elif typ in ("ckpt", "tpckpt"):
            pool = battle_paths if typ == "ckpt" else tp_paths
            if raw not in pool:
                raise ValueError(f"--{name}: {raw!r} is not an on-disk checkpoint")
            argv += [f"--{name}", raw]
        elif typ == "text":
            val = str(raw).replace("\n", " ").replace("\r", " ")[:300]
            argv += [f"--{name}", val]
        else:
            raise ValueError(f"unknown option type {typ!r}")
    return argv


# ── managed jobs ──────────────────────────────────────────────────────────────
class _Job:
    def __init__(self, jid: str, entry_id: str, argv: list, env_extra: dict,
                 heavy: bool = False):
        self.jid, self.entry_id = jid, entry_id
        self.argv, self.env_extra = argv, env_extra
        self.heavy = heavy
        self.started = time.time()
        self.log_path = _LOGS_DIR / f"mc_{entry_id}_{time.strftime('%Y%m%d-%H%M%S')}.log"
        env = {**os.environ, "PYTHONUTF8": "1", **env_extra}
        _LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self._log_f = open(self.log_path, "wb")
        self.proc = subprocess.Popen(
            argv, cwd=str(_REPO), env=env, stdout=self._log_f, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))

    @property
    def alive(self) -> bool:
        return self.proc.poll() is None

    def stop(self) -> None:
        if self.alive:  # Windows: kill the whole tree (poke-env/showdown children)
            subprocess.run(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"],
                           capture_output=True)

    def info(self) -> dict:
        rc = self.proc.poll()
        return {"job": self.jid, "id": self.entry_id, "alive": rc is None, "returncode": rc,
                "heavy": self.heavy,
                "started": time.strftime("%H:%M:%S", time.localtime(self.started)),
                "elapsed_s": int(time.time() - self.started),
                "log": self.log_path.name,
                "progress": _job_progress(self.entry_id, self.argv, self.log_path),
                "cmd": " ".join(Path(self.argv[0]).name if i == 0 else a
                                for i, a in enumerate(self.argv))}


class _Jobs:
    def __init__(self):
        self._jobs: dict = {}
        self._lock = threading.Lock()
        self._n = 0

    def start(self, entry: dict, options: dict, env_extra: dict) -> dict:
        env_fmt = _env_read()
        teams = _discover_teams(_active_format(env_fmt))
        argv = _build_argv(entry, options, teams, _ckpt_inventory())
        heavy = bool(entry.get("heavy"))
        with self._lock:
            if entry.get("single"):
                # 2026-09-03: ONE online bot at a time. Two launches 5 min apart played the SAME
                # battles on one account (13 rejections in 14 games, a game sealed under arm None,
                # the bandit state written by both). The bot itself also refuses (DuplicateBotError).
                dup = next((j for j in self._jobs.values() if j.entry_id == entry["id"] and j.alive), None)
                if dup is not None:
                    raise ValueError(
                        f"{entry.get('title', entry['id'])} is already running ({dup.jid}). One bot per "
                        "account — stop it in the Jobs tab (or wait for it to finish) before launching again.")
            if heavy:
                # ONE heavy run at a time (RTX 3070 Ti 8GB / 32GB RAM). A 2nd heavy launch is
                # refused with the offender named — stop it (or wait) before starting another.
                busy = next((j for j in self._jobs.values() if j.heavy and j.alive), None)
                if busy is not None:
                    raise ValueError(
                        f"a heavy run is already active — {busy.entry_id} ({busy.jid}). "
                        "One heavy run at a time (GPU/RAM). Stop it first or let it finish.")
            self._n += 1
            jid = f"j{self._n}"
            job = _Job(jid, entry["id"], argv, env_extra, heavy=heavy)
            self._jobs[jid] = job
        return job.info()

    def stop(self, jid: str) -> dict:
        with self._lock:
            job = self._jobs.get(jid)
        if job is None:
            raise ValueError(f"unknown job {jid!r}")
        job.stop()
        return job.info()

    def list(self) -> list:
        with self._lock:
            return [j.info() for j in self._jobs.values()]


_JOBS = _Jobs()


# ── status assembly ───────────────────────────────────────────────────────────
def _status() -> dict:
    env = _env_read()
    fmt = _active_format(env)
    inv = _ckpt_inventory()
    mio = _model_io_defaults()
    env_pub = {k: env.get(k, "") for k in _ENV_READ_KEYS}
    parity = {
        "battle": {"env": env.get("VD_BATTLE_CKPT", ""), "model_io": mio["battle"],
                   "match": bool(mio["battle"]) and env.get("VD_BATTLE_CKPT", "") == mio["battle"]},
        "tp": {"env": env.get("VD_TP_CKPT", ""), "model_io": mio["tp"],
               "match": bool(mio["tp"]) and env.get("VD_TP_CKPT", "") == mio["tp"]},
    }
    services = {}
    for key, svc in _SERVICES.items():
        services[key] = {**{k: v for k, v in svc.items() if k != "job"},
                         "up": _port_open(svc["port"]), "job": svc.get("job")}
    return {"repo": str(_REPO), "env": env_pub, "format": fmt, "formats": _known_formats(),
            "teams": _discover_teams(fmt), "ckpts": inv, "parity": parity,
            "services": services, "exploit": _exploit_summary(), "logs": _recent_logs(),
            "jobs": _JOBS.list(), "formats_full": _known_formats_full(),
            "editable_env": list(_ENV_WRITE_KEYS),
            "bandit_arms": _bandit_arm_names()}


def _registry_public() -> list:
    out = []
    for e in REGISTRY:
        out.append({k: e.get(k) for k in
                    ("id", "cat", "heavy", "title", "desc", "module", "script", "raw",
                     "fixed", "opts", "envopts", "note")})
    return out


# ── HTTP layer ────────────────────────────────────────────────────────────────
def _make_handler():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # noqa: D102 — silence request noise
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            u = urlparse(self.path)
            try:
                if u.path == "/" or u.path.startswith("/index"):
                    body = _HTML_PATH.read_bytes() if _HTML_PATH.is_file() else \
                        b"<h1>mission_control.html missing</h1>"
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif u.path == "/api/status":
                    self._json(_status())
                elif u.path == "/api/commands":
                    self._json({"commands": _registry_public(), "python": ".venv/Scripts/python.exe"})
                elif u.path == "/api/exploit_curve":
                    self._json(_exploit_summary())
                elif u.path == "/api/online/status":
                    self._json(_panel_status())
                elif u.path == "/api/dashboard/status":     # 2026-09-03: the Dashboard tab's run strip
                    self._json(_dashboard_status())
                elif u.path == "/api/species":
                    self._json({"species": _species_list()})
                elif u.path == "/api/log":
                    q = parse_qs(u.query)
                    name = (q.get("name") or [""])[0]
                    offset = int((q.get("offset") or ["0"])[0])
                    target = (_LOGS_DIR / name).resolve()
                    if not str(target).startswith(str(_LOGS_DIR.resolve())) or not target.is_file():
                        self._json({"ok": False, "error": "bad log name"}, 404)
                        return
                    size = target.stat().st_size
                    start = max(0, size - 100_000) if offset == 0 and size > 100_000 else offset
                    with open(target, "rb") as f:
                        f.seek(start)
                        data = f.read(200_000)
                    self._json({"ok": True, "size": size, "offset": start + len(data),
                                "data": data.decode("utf-8", errors="replace")})
                else:
                    self._json({"ok": False, "error": "not found"}, 404)
            except Exception as exc:  # noqa: BLE001 — surface, don't kill the server
                self._json({"ok": False, "error": str(exc)}, 500)

        def do_POST(self):  # noqa: N802
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}") if n else {}
            except Exception:
                self._json({"ok": False, "error": "bad JSON"}, 400)
                return
            try:
                if self.path == "/api/env":
                    _env_write(str(body.get("key") or ""), str(body.get("value") or ""))
                    self._json({"ok": True})
                elif self.path == "/api/job/start":
                    entry = _REG_BY_ID.get(str(body.get("id") or ""))
                    if entry is None:
                        raise ValueError("unknown command id")
                    # Heavy runs are launchable from the UI (the USER clicks = the USER launches);
                    # _JOBS.start enforces the one-heavy-run-at-a-time lock.
                    env_extra = {}
                    for eo in entry.get("envopts", []) or []:
                        if (body.get("env") or {}).get(eo["key"]):
                            env_extra[eo["key"]] = "1"
                    # Format picker → VDANCE_BATTLE_FORMAT (spawn-safe; every harness reads it at
                    # import). Blank = the deployed .env default. Validated to a Showdown format id.
                    if any(o.get("type") == "format" for o in entry.get("opts", [])):
                        fv = str((body.get("options") or {}).get("format") or "").strip()
                        if fv:
                            if not re.match(r"^[a-z0-9]+$", fv):
                                raise ValueError("format must be a Showdown format id — lowercase "
                                                 "letters/digits only, e.g. gen9championsvgc2026regmb")
                            env_extra["VDANCE_BATTLE_FORMAT"] = fv
                    info = _JOBS.start(entry, body.get("options") or {}, env_extra)
                    self._json({"ok": True, "job": info})
                elif self.path == "/api/job/stop":
                    self._json({"ok": True, "job": _JOBS.stop(str(body.get("job") or ""))})
                elif self.path.startswith("/api/online/"):
                    # proxy to the live bot panel: /api/online/ladder/start -> panel /api/ladder/start
                    target = "/api" + self.path[len("/api/online"):]
                    self._json({"ok": True, "panel": _panel_post(target, body)})
                elif self.path == "/api/tb/generate":
                    self._json(_tb_post("/api/teams/generate", body))
                elif self.path == "/api/tb/score":
                    self._json(_tb_post("/api/teams/score", body))
                elif self.path == "/api/teams/save":
                    self._json({"ok": True, "path": _save_team(body.get("name"), body.get("paste"))})
                else:
                    self._json({"ok": False, "error": "not found"}, 404)
            except ValueError as exc:
                self._json({"ok": False, "error": str(exc)}, 400)
            except Exception as exc:  # noqa: BLE001
                self._json({"ok": False, "error": str(exc)}, 500)

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description="Victory-Dance Mission Control")
    ap.add_argument("--port", type=int, default=8990)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    handler = _make_handler()
    server, last_exc = None, None
    for p in range(args.port, args.port + 10):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", p), handler)
            break
        except OSError as exc:
            last_exc = exc
    if server is None:
        raise OSError(f"no free port in {args.port}-{args.port + 9}: {last_exc}")
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"[mission-control] {url}   (repo: {_REPO})")
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[mission-control] bye.")


if __name__ == "__main__":
    main()
