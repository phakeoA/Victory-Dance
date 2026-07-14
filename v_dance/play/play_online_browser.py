"""Online browser play — the AI battles on the REAL Showdown site under YOUR control.

Design: docs/online_browser_play_design.md. The play_vs_human_browser transport pointed at
play.pokemonshowdown.com: a Playwright tab logs into the ``.env`` account, YOU find the matches
(ladder Battle! / challenges — incoming challenges in our format are AUTO-ACCEPTED), and the AI
plays every battle room that opens via the connection-less ``BattleHost`` (frames in → /choose
back into the tab). Closed team sheets; every finished game appends a bench-JSONL row.

  python -m v_dance.play.play_online_browser --dry-run     # connect+login+teams, NO battles
  python -m v_dance.play.play_online_browser               # the real thing (Ctrl-C to stop)

.env keys: PS_USERNAME/PS_PASSWORD (login), PS_CLIENT_URL, PS_AVATAR, VDANCE_BATTLE_FORMAT
(exported BEFORE v_dance imports so the whole stack runs that format), VD_BATTLE_CKPT/VD_TP_CKPT/
VD_DEFAULT_TEAM (deploy defaults; --ckpt/--tp-ckpt/--ai-team override). Password never printed.
"""
from __future__ import annotations

import os
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]


def _load_env(path: Path = _REPO / ".env") -> dict:
    env: dict = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip()
    return env


_ENV = _load_env()
# ⚠ Deliberate pre-import side effect: formats.py resolves DEFAULT_FORMAT from the
# VDANCE_BATTLE_FORMAT env var AT IMPORT, and every play module binds it from there — so the
# .env override must land in os.environ before any v_dance.play import (same spawn-safe pattern
# formats.py documents for mp workers).
if _ENV.get("VDANCE_BATTLE_FORMAT"):
    os.environ.setdefault("VDANCE_BATTLE_FORMAT", _ENV["VDANCE_BATTLE_FORMAT"])

import argparse                                    # noqa: E402
import asyncio                                     # noqa: E402
import json                                        # noqa: E402
import logging                                     # noqa: E402
import sys                                         # noqa: E402
import time                                        # noqa: E402

import v_dance                                     # noqa: F401,E402  (Selector policy for POKE_LOOP)
import v_dance.play.play_vs_human_browser as _pvhb  # noqa: E402  (the reused local transport)
from v_dance.play.browser.battle_host import BattleHost          # noqa: E402
from v_dance.play.model_io import DEFAULT_BC_CHECKPOINT, DEFAULT_TP_CHECKPOINT  # noqa: E402
from v_dance.play.play_vs_human_browser import (                 # noqa: E402
    _ai_consumer, _default_format, _load_pool, _use_proactor_loop,
)
from v_dance.play.run_local_battle import BATTLE_FORMAT          # noqa: E402

BENCH_DIR = _REPO / "artifacts" / "human_benchmark"
BENCH_LOG = BENCH_DIR / "human_bench.jsonl"
# 2026-07-10 (USER): artifacts/logs = THE home for all future logs (see artifacts/README.md).
# The online session logger writes online_<session>.log there — per-game detail lines + rating
# updates as they happen, and on exit the era benchmark report + an observed-meta refresh, so
# the grind-to-plateau loop needs no manual report commands.
LOG_DIR = _REPO / "artifacts" / "logs"


def _slog(path: Path, text: str) -> None:
    """Append one line to the session log, crash-safe; logging must never break play."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(text.rstrip("\n") + "\n")
    except Exception:
        pass

# 2026-07-10 (USER): default client settings, applied EVERY launch — the Playwright context is
# fresh so saved prefs never persist. Pref keys + live side-effects verified against the served
# client source (js/oldclient/client-topbar.js OptionsPopup + storage.js Storage.prefs): theme
# needs the html.dark class toggle, onepanel needs app.updateLayout(), sprite prefs need
# Dex.loadSpriteData — the same calls the popup's own change handlers make, so no page reload.
# (Block PMs/Challenges + Language are SERVER-side app.user.updateSetting settings whose defaults
# already match the wanted state — intentionally not touched.)
_CLIENT_PREFS_JS = (
    "(prefs) => {"
    "  for (const k in prefs) Storage.prefs(k, prefs[k]);"
    "  let th = prefs.theme;"
    "  if (th === 'system') th = (window.matchMedia && matchMedia('(prefers-color-scheme: dark)').matches) ? 'dark' : 'light';"
    "  if (prefs.theme) $('html').toggleClass('dark', th === 'dark');"
    "  if ('onepanel' in prefs) { app.singlePanelMode = !!prefs.onepanel; app.updateLayout(); }"
    "  if (window.Dex && Dex.loadSpriteData) Dex.loadSpriteData((prefs.noanim || prefs.bwgfx) ? 'bw' : 'xy');"
    "  return Object.keys(prefs).length; }"
)


# Same team-import JS as the local _setup_client (see its ⚠ packTeam/capacity:6 comments —
# verified live in scratch/browser_team_import_probe.py / browser_capacity_probe.py).
_IMPORT_TEAMS_JS = (
    "(args) => { const [fmt, pastes] = args; const failed = [];"
    "  for (const [nm, paste] of pastes) {"
    "    if (Storage.teams.some(t => t.name === nm && t.format === fmt)) continue;"  # re-runs: no dupes
    "    const sets = Storage.importTeam(paste);"
    "    if (!sets || !sets.length) { failed.push(nm); continue; }"
    "    Storage.teams.push({ name: nm, format: fmt, folder: '', capacity: 6, team: Storage.packTeam(sets) }); }"
    "  Storage.saveTeams && Storage.saveTeams(); return failed; }"
)


def _sockjs_unwrap(payload: str) -> list:
    """play.pokemonshowdown.com (psim.us) delivers protocol messages SockJS-framed: ``o`` (open),
    ``h`` (heartbeat), ``c[…]`` (close) and ``a["msg", …]`` — each msg a JSON string holding the
    raw ``|``-protocol text (newlines escaped). The LOCAL server's endpoint delivers BARE protocol
    text, which is why the local harness never needed this — online, nothing parsed until 2026-07-09
    (no auto-accept, no teampreview: every frame silently missed the ``>battle``/challenge branches).
    Unwrap ``a`` frames into their messages, drop control frames, pass bare frames through."""
    if not payload:
        return []
    if payload.startswith("a["):
        try:
            return [m for m in json.loads(payload[1:]) if isinstance(m, str)]
        except Exception:
            return [payload]                       # not actually SockJS → pass through untouched
    if payload in ("o", "h") or payload.startswith("c["):
        return []
    return [payload]


def _write_avatar(avatar: str, env_path: Path = _REPO / ".env") -> None:
    """Persist an avatar choice to ``.env`` PS_AVATAR (no-op when unchanged). Atomic write
    (temp + replace): .env holds credentials, a torn write is never acceptable."""
    avatar = (avatar or "").strip()
    if not avatar or avatar == (_ENV.get("PS_AVATAR") or "").strip():
        return
    lines = env_path.read_text(encoding="utf-8").splitlines()
    for i, ln in enumerate(lines):
        if ln.split("=", 1)[0].strip() == "PS_AVATAR":
            lines[i] = f"PS_AVATAR={avatar}"
            break
    else:
        lines.append(f"PS_AVATAR={avatar}")
    tmp = env_path.with_name(env_path.name + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, env_path)
    _ENV["PS_AVATAR"] = avatar
    print(f"[online] avatar changed in the browser → saved to .env (PS_AVATAR={avatar})")


_AVATAR_CMD = "/avatar "


def _sync_avatar_from_send(payload: str, env_path: Path = _REPO / ".env") -> None:
    """Persist browser-side avatar picks by watching the OUTGOING frames (2026-07-10 v2): the
    client sends ``room|/avatar <x>`` when the USER picks an avatar in the UI. v1 watched the
    incoming ``|updateuser|`` — but the server only sends that on LOGIN/RENAME, never on an
    avatar change, so real picks were never saved and the next session's startup ``/avatar``
    (from the stale .env) snapped the avatar back — the observed "refuses to save, defaults
    back to 2". The harness's own startup send passes through here too = a same-value no-op.
    Client→server frames arrive JSON-array-wrapped (``["msg", …]``) or bare."""
    try:
        if _AVATAR_CMD not in payload:
            return
        msgs = [payload]
        if payload.startswith("["):
            try:
                msgs = [m for m in json.loads(payload) if isinstance(m, str)]
            except Exception:
                msgs = [payload]
        for m in msgs:
            text = m.split("|", 1)[1] if "|" in m else m   # strip the room prefix
            if text.startswith(_AVATAR_CMD):
                _write_avatar(text[len(_AVATAR_CMD):], env_path)
                return
    except Exception as exc:                       # a sync failure must never break play
        print(f"[online] avatar .env sync failed (non-fatal): {exc!r}")


async def _login(page, username: str, password: str) -> bool:
    """Log the tab into the registered account. Scripted via the client's own rename flow (the
    challstr/assertion dance is the client's job); on any failure fall back to MANUAL login —
    print instructions and wait until the tab is named as ``username``. Never raises on UI drift."""
    named_js = ("(u) => window.app && app.user && app.user.get('named') && "
                "app.user.get('name').toLowerCase().replace(/[^a-z0-9]/g,'') === u")
    uid = "".join(c for c in username.lower() if c.isalnum())
    try:
        # app.user.rename triggers the challstr flow; a registered name pops the password form.
        await page.evaluate("(n) => app.user.rename(n)", username)
        await page.fill("input[name=password]", password, timeout=8000)
        await page.click("button[type=submit]", timeout=4000)
        await page.wait_for_function(named_js, arg=uid, timeout=15000)
        print(f"[online] logged in as {username}")
        return True
    except Exception as exc:
        print(f"[online] scripted login failed ({type(exc).__name__}) — log in MANUALLY in the "
              f"browser window (Choose name → {username} → password). Waiting …")
        try:
            await page.wait_for_function(named_js, arg=uid, timeout=300000)
            print(f"[online] logged in as {username} (manual)")
            return True
        except Exception:
            return False


def _write_session_summary(session_log: Path, note: str, tally: dict) -> None:
    """On session exit (2026-07-10, USER): append the era benchmark report + refresh the
    observed-meta aggregate — the grind loop's bookkeeping, automated. Never raises."""
    try:
        _slog(session_log, f"=== session end — tally AI {tally.get('ai', 0)} / "
                           f"opp {tally.get('you', 0)} / draws {tally.get('draw', 0)}")
        from v_dance.eval.human_benchmark_report import load_rows, report_text
        if BENCH_LOG.is_file():
            rows = [r for r in load_rows(BENCH_LOG) if r.get("note") == note]
            if rows:
                _slog(session_log, report_text(rows))
        import re as _re
        from v_dance.datatools.observed_meta import DOSSIER_DIR, _display_names, aggregate
        m = _re.search(r"(reg[a-z0-9]+)", BATTLE_FORMAT or "")
        reg = m.group(1) if m else "regma"
        data = aggregate(DOSSIER_DIR, fmt=BATTLE_FORMAT, names=_display_names(reg))
        out = _REPO / "data" / f"observed_meta_{reg}.json"
        out.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
        _slog(session_log, f"observed-meta refreshed: {data['n_opponents']} opponents / "
                           f"{data['n_games']} games / {len(data['pokemon'])} species -> {out}")
    except Exception as exc:                       # bookkeeping must never break shutdown
        _slog(session_log, f"(session summary failed: {exc!r})")


def _wrap_bench_recording(host: BattleHost, session_id: str, note: str,
                          ckpt: Path, tp_ckpt: Path) -> None:
    """Append a bench-JSONL row for every finished battle. Hook = host.end_battle (the consumer
    calls it exactly once per battle, while the battle object is still in host.player._battles),
    so recording needs zero changes to the reused consumer."""
    orig = host.end_battle
    state = {"n": 0}

    def end_battle(tag: str) -> None:
        try:
            b = host.player._battles.get((tag or "").lstrip(">"))
            if b is not None and getattr(b, "finished", False):
                state["n"] += 1
                won = getattr(b, "won", None)
                row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "session_id": session_id, "note": note, "game_idx": state["n"],
                       "battle_tag": getattr(b, "battle_tag", tag),
                       "ai_team": getattr(host.player, "_team_name", None),
                       "human_team": None,
                       "opponent": getattr(b, "opponent_username", None),
                       "result": "ai" if won else ("draw" if won is None else "human"),
                       "turns": getattr(b, "turn", None),
                       "rating": getattr(b, "rating", None),
                       "opponent_rating": getattr(b, "opponent_rating", None),
                       "ckpt": str(ckpt), "tp_ckpt": str(tp_ckpt)}
                BENCH_DIR.mkdir(parents=True, exist_ok=True)
                with BENCH_LOG.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                # session detail log (2026-07-10): one human-readable line per game as it lands
                _slog(LOG_DIR / f"online_{session_id}.log",
                      f"{row['ts']}  game {row['game_idx']:>3}  {row['result'].upper():<5} "
                      f"vs {(row['opponent'] or '?'):<20} {row['turns']}t  "
                      f"team {row['ai_team'] or '?'}  {row['battle_tag']}")
                # B-L2 dossier capture (passive; never raises) — key data for online opponents.
                from v_dance.play.opponent_dossier import summary, update_from_battle
                if update_from_battle(b, row["result"],
                                      our_team=row["ai_team"], note=note):
                    s = summary(row["opponent"] or "")
                    print(f"[online] dossier: {s}")
                    _slog(LOG_DIR / f"online_{session_id}.log", f"    dossier: {s}")
        except Exception as exc:                       # recording must never break play
            print(f"[online] bench-record failed (non-fatal): {exc!r}")
        orig(tag)

    host.end_battle = end_battle  # type: ignore[method-assign]


async def run(args, username: str, password: str, ckpt: Path, tp_ckpt: Path) -> dict:
    from playwright.async_api import async_playwright

    teams = _load_pool()
    if not teams:
        raise SystemExit(f"[online] no teams in the {BATTLE_FORMAT} pool.")
    ai_pool = [n for n, _ in teams]
    if args.ai_team and args.ai_team not in ai_pool:
        raise SystemExit(f"[online] --ai-team {args.ai_team!r} not in the pool ({ai_pool[:10]} …)")

    session_id = f"{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-{os.getpid()}"
    host = BattleHost(team=teams[0][1], username=username,       # username → correct battle side
                      model_path=ckpt, team_chooser_path=tp_ckpt,
                      adapt_rules=args.adapt_rules,
                      use_dossier=args.dossier,              # S1 L2b (default OFF)
                      # HTML replays (S4 online protocol: the stop-loss review + Phase-3 data)
                      live_dir=BENCH_DIR / "live", save_replays=True,
                      replay_dir=BENCH_DIR / "replays" / session_id, replay_label="online",
                      # Type_C training copy (2026-07-10, USER): real games → corpus folder
                      replay_copy_dir=_REPO / "data" / "vods" / "Type_C")
    _wrap_bench_recording(host, session_id, args.bench_note, ckpt, tp_ckpt)
    print(f"[online] replays -> {BENCH_DIR / 'replays' / session_id}")

    # Ladder-rating capture (2026-07-10): the post-battle |raw| rating lines arrive after the
    # game row is written, so they land as separate crash-safe "rating_update" rows keyed by
    # battle_tag; human_benchmark_report joins them onto the game rows at read time.
    def _rating_row(tag: str, user: str, rating: int) -> None:
        key = "rating" if _pvhb._toid(user) == _pvhb._toid(username) else "opponent_rating"
        BENCH_DIR.mkdir(parents=True, exist_ok=True)
        with BENCH_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                "type": "rating_update", "session_id": session_id,
                                "battle_tag": tag, key: rating}) + "\n")
        _slog(LOG_DIR / f"online_{session_id}.log",
              f"    rating: {user} {rating} ({'us' if key == 'rating' else 'them'})  {tag}")

    _pvhb.RATING_HOOK = _rating_row
    session_log = LOG_DIR / f"online_{session_id}.log"
    _slog(session_log, f"=== ONLINE session {session_id} — {username} @ {args.client_url}")
    _slog(session_log, f"    format {BATTLE_FORMAT}  note {args.bench_note!r}")
    _slog(session_log, f"    ckpt {ckpt}\n    tp   {tp_ckpt}")
    print(f"[online] session log -> {session_log}")
    # The reused consumer/_credit print + fall back on the module's _AI_NAME; online, the AI IS
    # the .env account. (The local harness isn't running in this process — safe to repoint.)
    _pvhb._AI_NAME = username

    # Team-pick wrapper: remember the picked name (for the bench row), and make .env
    # VD_DEFAULT_TEAM the RANDOM-fallback — an explicitly OPENED Teambuilder team still wins.
    # 2026-07-10 control panel: a team pinned in the panel OUTRANKS everything (the panel's
    # dropdown is the user's live 'use this team' gesture; challenge-accepts honour it too).
    default_team = _ENV.get("VD_DEFAULT_TEAM")
    _orig_pick = _pvhb._pick_ai_team
    ctrl_ref: dict = {"c": None}                   # filled after login (the panel needs the page)

    async def _pick(page, pool, pin, route_for=None):
        # ⚠ MUST accept route_for: _ai_consumer's challenge-accept path calls
        # _pick_ai_team(..., route_for=challenger) (the 4b router param). Without it this
        # wrapper raised TypeError on every incoming challenge accept (caught + swallowed as
        # "accept failed"), so the online bot silently declined all direct challenges.
        c = ctrl_ref.get("c")
        if c is not None and c.team_pin and c.team_pin in pool:
            host.player._team_name = c.team_pin
            return c.team_pin, "control panel"
        nm, src = await _orig_pick(page, pool, pin, route_for=route_for)
        if src == "random" and default_team in pool:
            nm, src = default_team, ".env default"
        host.player._team_name = nm
        return nm, src

    _pvhb._pick_ai_team = _pick

    tally = {"ai": 0, "you": 0, "draw": 0}
    stop = asyncio.Event()
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=False, args=["--window-size=1280,900"])
        try:
            frame_q: asyncio.Queue = asyncio.Queue()
            ctx = await browser.new_context(no_viewport=True)
            page = await ctx.new_page()

            def _on_frame(p) -> None:              # incoming: SockJS unwrap → consumer feed
                for msg in _sockjs_unwrap(p):
                    frame_q.put_nowait(msg)
                    c = ctrl_ref.get("c")          # control panel's read-only frame tap
                    if c is not None:
                        c.tap_frame(msg)

            def _on_ws(ws) -> None:
                ws.on("framereceived", _on_frame)
                # outgoing: the USER's avatar picks send `/avatar <x>` — the only reliable
                # signal (the server sends no updateuser on avatar changes; see the sync's doc).
                ws.on("framesent", _sync_avatar_from_send)

            page.on("websocket", _on_ws)
            await page.goto(args.client_url, wait_until="domcontentloaded")
            await page.wait_for_function(
                "() => window.app && app.socket && app.socket.readyState === 1", timeout=30000)
            if not await _login(page, username, password):
                raise SystemExit("[online] login did not complete — aborting.")
            if _ENV.get("PS_AVATAR"):
                await page.evaluate("(a) => app.socket.send('|/avatar ' + a)", _ENV["PS_AVATAR"])
            if _ENV.get("PS_CLIENT_PREFS"):        # 2026-07-10: USER's default client settings
                try:
                    _n = await page.evaluate(_CLIENT_PREFS_JS, json.loads(_ENV["PS_CLIENT_PREFS"]))
                    print(f"[online] applied {_n} client prefs from .env PS_CLIENT_PREFS")
                except Exception as exc:
                    print(f"[online] PS_CLIENT_PREFS failed (non-fatal): {exc!r}")
            if not args.no_import:
                failed = await page.evaluate(_IMPORT_TEAMS_JS, [BATTLE_FORMAT, teams])
                if failed:
                    print(f"[online] WARNING: {len(failed)} team import(s) failed: {failed[:8]}")
                print(f"[online] {len(teams) - len(failed)} pool teams available in the Teambuilder")
            await _default_format(page, "online")

            # 2026-07-10 (USER): local control server — ladder-run count / team pin / private
            # challenges / auto-accept, all without touching the bot window. Its HTTP API must keep
            # running (Mission Control's 'Online bot' tab is now the UI — it PROXIES this server;
            # it can't drive the Playwright page itself). 2026-07-13 (USER): the standalone panel
            # WINDOW no longer auto-opens (redundant with Mission Control) — pass --panel-window to
            # restore the popup. NOT in --dry-run: the consumer isn't serving there.
            if args.control_port and not args.dry_run:
                try:
                    from v_dance.play.bot_control_ui import start_control_ui
                    ctrl_ref["c"] = start_control_ui(
                        page=page, host=host, tally=tally, ai_pool=ai_pool,
                        fmt=BATTLE_FORMAT, username=username,
                        loop=asyncio.get_running_loop(), env_path=_REPO / ".env",
                        port=args.control_port, open_browser=args.panel_window,
                        team_pin_default=args.ai_team or default_team,
                        auto_close_default=(_ENV.get("VD_AUTO_CLOSE_ROOMS", "0").strip() == "1"),
                        log_line=lambda t: _slog(session_log, t))
                    _ctrl = ctrl_ref["c"]
                    print(f"[online] control server on {_ctrl.url}  —  drive it from Mission Control's "
                          f"'Online bot' tab (or open that URL for the standalone panel).")
                except Exception as exc:
                    print(f"[online] control panel failed to start (non-fatal): {exc!r}")

            print("\n" + "=" * 64)
            print(f"  ONLINE — {username} @ {args.client_url}   format {BATTLE_FORMAT}")
            print(f"  battle ckpt : {ckpt}")
            print(f"  TP ckpt     : {tp_ckpt}")
            print(f"  bench       : session {session_id} (note {args.bench_note!r}) -> {BENCH_LOG}")
            if ctrl_ref["c"] is not None:
                print(f"  control     : {ctrl_ref['c'].url}  (ladder runs / challenges / auto-accept)")
            print("  YOU find the matches in this window (ladder Battle! / send a challenge).")
            print(f"  Incoming challenges in {BATTLE_FORMAT} are AUTO-ACCEPTED (toggle in the panel).")
            print(f"  AI team per battle = control-panel pin, else --ai-team, else the team OPEN "
                  f"in the Teambuilder, else random. Ctrl-C here to stop.")
            print("=" * 64 + "\n")
            sys.stdout.flush()
            if args.dry_run:
                print("[online] DRY RUN — no battles will be played; Ctrl-C (or close the window) to exit.")
                while not page.is_closed():        # window closed = auto-Ctrl-C, same as the consumer
                    await asyncio.sleep(1.0)
                print("[online] browser window closed — exiting dry run.")
                return tally
            await _ai_consumer(page, host, frame_q, ai_pool, tally, stop, args.ai_team)
        finally:
            if ctrl_ref.get("c") is not None:      # stop the panel's HTTP thread (best-effort)
                ctrl_ref["c"].stop()
            try:
                await browser.close()
            except Exception:
                # window-close auto-stop: the browser (and its driver pipe) are already gone —
                # close() then raises "Connection closed while reading from the driver". Benign.
                pass
            _write_session_summary(session_log, args.bench_note, tally)
            print(f"[online] session summary + era report -> {session_log}")
            # 2026-07-10 (USER): keep Type_C training-clean automatically — DELETE any junk
            # replays this session recorded (turn-1 forfeit/timeout/disconnect: nothing to learn,
            # and the elo they granted is already captured in the bench rows regardless).
            try:
                _tc = _REPO / "data" / "vods" / "Type_C"
                if _tc.is_dir():
                    from v_dance.datatools.filter_type_c import run as _filter_type_c
                    _filter_type_c(_tc, apply=True)
            except Exception as exc:
                print(f"[online] Type_C junk filter failed (non-fatal): {exc!r}")
    return tally


def main() -> None:
    ap = argparse.ArgumentParser(description="AI plays online through your browser (you control matchmaking).")
    ap.add_argument("--client-url", default=_ENV.get("PS_CLIENT_URL", "https://play.pokemonshowdown.com"))
    ap.add_argument("--ckpt", default=None, help="battle net (default: .env VD_BATTLE_CKPT, else prod).")
    ap.add_argument("--tp-ckpt", default=None, help="TP net (default: .env VD_TP_CKPT, else prod).")
    ap.add_argument("--ai-team", default=None,
                    help="pin the AI team (default: .env VD_DEFAULT_TEAM as the random-fallback pin; "
                         "the team OPEN in the Teambuilder still wins when no pin is given).")
    ap.add_argument("--no-import", action="store_true", help="don't import pool teams into the tab.")
    ap.add_argument("--bench-note", default="online")
    ap.add_argument("--dry-run", action="store_true",
                    help="connect + login + import teams, then idle — the safe first live test.")
    ap.add_argument("--adapt-rules", action="store_true",
                    help="B-L1 serve-time pattern tilt (Wide-Guard streak → spread bias). Default OFF.")
    ap.add_argument("--dossier", action="store_true",
                    help="S1 L2b: warm-start unknown opp item/ability/moves from the per-opponent "
                         "dossier (cross-game; in-battle evidence always wins). Default OFF.")
    ap.add_argument("--control-port", type=int, default=8777,
                    help="local control-panel port (ladder runs / challenges / auto-accept); 0 = off.")
    ap.add_argument("--panel-window", action="store_true",
                    help="also POP OPEN the standalone control-panel page in a browser. Default OFF: "
                         "Mission Control's 'Online bot' tab is the control UI (it proxies this "
                         "server), so the panel window is redundant. The panel URL is still logged.")
    args = ap.parse_args()

    username = _ENV.get("PS_USERNAME")
    password = _ENV.get("PS_PASSWORD")
    if not (username and password):
        raise SystemExit("[online] PS_USERNAME / PS_PASSWORD missing from .env")
    ckpt = Path(args.ckpt or _ENV.get("VD_BATTLE_CKPT") or DEFAULT_BC_CHECKPOINT)
    tp_ckpt = Path(args.tp_ckpt or _ENV.get("VD_TP_CKPT") or DEFAULT_TP_CHECKPOINT)
    for p in (ckpt, tp_ckpt):
        if not p.is_file():
            raise SystemExit(f"[online] checkpoint not found: {p}")

    _use_proactor_loop()
    tally = None
    try:
        tally = asyncio.run(run(args, username, password, ckpt, tp_ckpt))
    except KeyboardInterrupt:
        print("\n[online] Ctrl-C — shutting down.")
    finally:
        logging.getLogger("poke_env").setLevel(logging.CRITICAL)
    if tally is not None:
        print(f"[online] session tally — AI {tally['ai']} / opp {tally['you']} / draws {tally['draw']}")


if __name__ == "__main__":
    main()
