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
VD_DEFAULT_TEAM (deploy defaults; --ckpt/--tp-ckpt/--ai-team override), VD_SERVE_TAU/VD_SERVE_TOP_P
(N2b serve sampling; 0/unset = argmax, byte-identical). Password never printed.
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
# N2b serve sampling + era-4 TP near-tie + 2b pair-decode kill switch: model_io/BattleHost
# read these from os.environ, so a .env entry must be exported too (setdefault → a real
# environment variable still wins).
for _k in ("VD_SERVE_TAU", "VD_SERVE_TOP_P", "VD_TP_TIE_EPS", "VD_PAIR_DECODE",
           "VD_FUTILITY_MASK",
           # 2026-09-01 link watchdog knobs (LinkWatch): probe-after / dead-after seconds + kill switch
           "VD_LINK_PROBE_S", "VD_LINK_DEAD_S", "VD_LINK_RECONNECT",
           # 2026-09-01 serve-side bandit (era-5 W0) + the site's official-rating poll
           "VD_BANDIT", "VD_BANDIT_CONFIG", "VD_SITE_POLL",
           # 2026-09-02 serve-mode pin at launch (arm name = frozen; explore/0 = clear; unset = keep)
           "VD_BANDIT_PIN",
           # 2026-09-02 ladder lanes: rated games at once at launch (1..5)
           "VD_LADDER_LANES",
           # 2026-09-02 W3b-0: record ladder games as RL trajectories (default ON; 0 = off)
           "VD_LADDER_RECORD",
           # 2026-09-02 (USER): battle-timer mode at launch (1 = /timer on at every game's first frame)
           "VD_TIMER_IMMEDIATE",
           # 2026-09-03 (USER): open team sheets at launch (1 = accept the server's team-preview offer)
           "VD_OTS_ACCEPT"):
    if _ENV.get(_k):
        os.environ.setdefault(_k, _ENV[_k])


def _timer_immediate_env() -> bool:
    """Launch default for the battle-timer MODE (USER 2026-09-02): ``VD_TIMER_IMMEDIATE=1`` → /timer on
    at the FIRST frame of every game (team preview included); unset / 0 → the per-room grace (the
    consumer's ``_OPP_TIMER_S`` of opponent silence after our decision). Runtime toggle in the panel."""
    return str(os.environ.get("VD_TIMER_IMMEDIATE") or _ENV.get("VD_TIMER_IMMEDIATE") or "0").strip() == "1"


def _timer_banner(immediate: bool, grace_s: float) -> str:
    """The launch echo for the timer mode (verify-flag-uptake rule: every launch knob echoes)."""
    if immediate:
        return ("[online] battle timer: IMMEDIATE — /timer on at the first frame of every game "
                "(VD_TIMER_IMMEDIATE=1; toggle live in the panel / Mission Control)")
    return (f"[online] battle timer: GRACE — /timer on once our decision sat unanswered {grace_s:.0f}s, "
            f"judged per room (VD_TIMER_IMMEDIATE=1 = at the first frame; toggle live in the panel)")


def _ots_accept_env() -> bool:
    """Launch default for OPEN TEAM SHEETS (USER 2026-09-03): ``VD_OTS_ACCEPT=1`` → the consumer answers the
    server's team-preview offer with /acceptopenteamsheets; unset / 0 → reject (closed sheets, the old
    behaviour). Runtime toggle in the panel / Mission Control."""
    return str(os.environ.get("VD_OTS_ACCEPT") or _ENV.get("VD_OTS_ACCEPT") or "0").strip() == "1"


def _ots_banner(accept: bool) -> str:
    """The launch echo for the open-team-sheet mode (verify-flag-uptake rule: every launch knob echoes)."""
    if accept:
        return ("[online] open team sheets: ACCEPT — when the opponent accepts too, their sheet is stamped into "
                "the battle net's opponent view (VD_OTS_ACCEPT=1; toggle live in the panel / Mission Control)")
    return ("[online] open team sheets: REJECT — closed sheets (VD_OTS_ACCEPT=1 = accept the offer; toggle live "
            "in the panel / Mission Control)")


def _int_env(key: str, default: int) -> int:
    try:
        return int(str(os.environ.get(key) or _ENV.get(key) or default).strip())
    except (TypeError, ValueError):
        return default

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
# 2026-09-02 (USER): the panel's matchup tables seed from the opponent dossiers
from v_dance.play.opponent_dossier import DOSSIER_DIR as _DOSSIER_DIR   # noqa: E402


def _panel_belief():
    """The served Pikalytics belief (the encoder's singleton) for the matchup tables' item /
    ability defaults; None when unavailable (the rows then read '—')."""
    try:
        from v_dance.play.vgc_base import _default_belief
        return _default_belief()
    except Exception:
        return None


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


_LINK_NAMED_JS = ("(u) => !!(window.app && app.user && app.user.get('named') && "
                  "app.user.get('name').toLowerCase().replace(/[^a-z0-9]/g,'') === u)")
_LINK_STATE_JS = "() => (window.app && app.socket) ? app.socket.readyState : -1"
_LINK_OPEN_JS = "() => !!(window.app && app.socket && app.socket.readyState === 1)"
_LINK_PROBE_JS = "(u) => app.socket.send('|/cmd userdetails ' + u)"
_LINK_REJOIN_JS = "(r) => { if (window.app && app.joinRoom && !app.rooms[r]) app.joinRoom(r, 'battle'); }"


class LinkWatch:
    """Dead-socket detector + auto-reconnect for the online tab (2026-09-01).

    USER report: on long sessions "the server keeps disconnecting the bot" — MID-BATTLE, so a slow
    recovery is a timer loss. Root causes (memory ``ladder-disconnect-diagnosis-2026-09-01``): the
    legacy client only shows a ReconnectPopup (a CLICK reconnects, nothing automatic), and the
    consumer could not tell a dead socket from a slow opponent (no close handler; the SockJS
    heartbeats were dropped before the queue). This watches the RAW frame clock and the socket:

      * ``ws.on("close")`` / readyState != 1  → reconnect on the next idle tick (~1 s).
      * no raw frame (SockJS ``h`` counts) for ``probe_s`` → send ``/cmd userdetails`` (a reply IS
        a frame); still nothing at ``dead_s`` → half-open socket → reconnect.

    Reconnect = the client's own recovery (a page reload: cookies keep the login, localStorage
    keeps teams/prefs, the server re-lists live games and the room is re-joined), THEN the live
    battle is rebuilt from the server's replayed log — the host FORGETS its battle object first so
    the replay cannot double-count (``BattleHost.forget_battle``; the tag sits in
    ``play_vs_human_browser.REJOINING`` until its live ``|request|`` lands). Worst case ≈ ``dead_s``
    + ~8 s, inside Showdown's disconnect grace. ``VD_LINK_RECONNECT=0`` = detect + log only.
    """

    def __init__(self, *, page, host, username: str, password: str, loop, log,
                 probe_s: float = 15.0, dead_s: float = 25.0, cooldown_s: float = 20.0,
                 reconnect: bool = True, prefs: dict | None = None, ctrl_ref: dict | None = None,
                 now=None):
        self.page, self.host = page, host
        self.username, self.password = username, password
        self._uid = "".join(c for c in username.lower() if c.isalnum())
        self.log = log
        self._now = now or (loop.time if loop is not None else time.monotonic)
        self.probe_s = max(5.0, float(probe_s))
        self.dead_s = max(self.probe_s + 5.0, float(dead_s))
        self.cooldown_s = float(cooldown_s)
        self.reconnect_enabled = bool(reconnect)
        self.prefs, self.ctrl_ref = prefs, ctrl_ref
        self.ws = None                          # the CURRENT websocket (a stale socket's close is noise)
        self.last_rx = self._now()              # raw-frame clock (heartbeats included)
        self.closed_at = None                   # set by a close event / failed reconnect → down
        self.probe_sent_at = None
        self.last_reconnect_at = -1e9
        self.reconnects = 0
        self.probes = 0
        self.frames = 0
        self._busy = False
        self._last_off_warn = -1e9

    # ── transport callbacks (Playwright event thread = the bot loop) ─────────
    def on_ws_open(self, ws) -> None:
        self.ws = ws
        self.closed_at = None
        self.last_rx = self._now()
        self.log(f"[online] websocket opened{' (reconnect #%d)' % self.reconnects if self.reconnects else ''}")

    def on_raw_frame(self, _payload) -> None:
        self.frames += 1
        self.last_rx = self._now()

    def on_ws_close(self, ws=None) -> None:
        if ws is not None and self.ws is not None and ws is not self.ws:
            return                              # an OLD socket closing after we already reconnected
        if self.closed_at is None:
            self.closed_at = self._now()
            self.log("[online] websocket CLOSED — reconnect pending (the client only shows a popup)")

    def on_ws_error(self, err=None) -> None:
        self.log(f"[online] websocket error: {str(err)[:120]}")

    # ── the tick (runs from the consumer's idle branch, ~1 s cadence) ─────────
    async def tick(self) -> None:
        if self._busy:
            return
        now = self._now()
        if self.closed_at is not None:
            await self._reconnect("socket closed")
            return
        idle = now - self.last_rx
        if idle < self.probe_s:
            return
        if self.probe_sent_at is not None and self.probe_sent_at > self.last_rx:
            if idle >= self.dead_s:             # probed and STILL silent → half-open socket
                await self._reconnect(f"no frames for {idle:.0f}s, probe unanswered")
            return
        # first silence past probe_s: ask the socket itself, then ping the server
        try:
            state = await self.page.evaluate(_LINK_STATE_JS)
        except Exception as exc:
            self.log(f"[online] link probe: page.evaluate failed ({type(exc).__name__}) — waiting")
            self.probe_sent_at = now
            return
        if state != 1:
            await self._reconnect(f"socket readyState={state}")
            return
        try:
            await self.page.evaluate(_LINK_PROBE_JS, self.username)
        except Exception as exc:
            self.log(f"[online] link probe send failed ({type(exc).__name__}) — waiting")
        self.probe_sent_at = now
        self.probes += 1

    def _live_tags(self) -> list:
        ended = getattr(self.host, "_ended", set())
        return [t for t, b in list(getattr(self.host, "battles", {}).items())
                if t not in ended and not getattr(b, "finished", False)]

    async def _ensure_named(self) -> bool:
        try:
            await self.page.wait_for_function(_LINK_NAMED_JS, arg=self._uid, timeout=8000)
            return True
        except Exception:
            pass
        try:                                    # scripted re-login: same steps as _login, no 5-min wait
            await self.page.evaluate("(n) => app.user.rename(n)", self.username)
            await self.page.fill("input[name=password]", self.password, timeout=8000)
            await self.page.click("button[type=submit]", timeout=4000)
            await self.page.wait_for_function(_LINK_NAMED_JS, arg=self._uid, timeout=15000)
            return True
        except Exception as exc:
            self.log(f"[online] re-login FAILED ({type(exc).__name__}) — log in MANUALLY in the bot window")
            return False

    async def _reconnect(self, reason: str) -> None:
        now = self._now()
        if not self.reconnect_enabled:
            if now - self._last_off_warn >= 30.0:
                self.log(f"[online] LINK DOWN ({reason}) — auto-reconnect is OFF (VD_LINK_RECONNECT=0)")
                self._last_off_warn = now
            return
        if now - self.last_reconnect_at < self.cooldown_s:
            return                              # a reconnect just ran — let it settle, retry after cooldown
        self._busy = True
        self.reconnects += 1
        t0 = now
        live = self._live_tags()
        self.log(f"[online] LINK DOWN ({reason}) — reconnecting (#{self.reconnects}); "
                 f"live battles: {', '.join(live) if live else 'none'}")
        for t in live:                          # forget BEFORE the replay lands (see class doc)
            try:
                self.host.forget_battle(t)
            except Exception as exc:
                self.log(f"[online] forget_battle({t}) failed (non-fatal): {exc!r}")
            _pvhb.REJOINING.add(t)
        ok, named = False, False
        try:
            await self.page.reload(wait_until="domcontentloaded", timeout=25000)
            await self.page.wait_for_function(_LINK_OPEN_JS, timeout=25000)
            self.last_rx = self._now()
            self.closed_at = None
            self.probe_sent_at = None
            named = await self._ensure_named()
            if self.prefs:
                try:
                    await self.page.evaluate(_CLIENT_PREFS_JS, self.prefs)
                except Exception:
                    pass
            try:
                await _default_format(self.page, "online")
            except Exception:
                pass
            for t in live:                      # the client re-joins live games itself; this is the belt
                try:
                    await self.page.evaluate(_LINK_REJOIN_JS, t)
                except Exception as exc:
                    self.log(f"[online] rejoin {t} failed (non-fatal): {exc!r}")
            ok = True
            self.log(f"[online] LINK RECONNECTED in {self._now() - t0:.1f}s"
                     f"{'' if named else ' (NOT logged in!)'}; rejoining {len(live)} battle(s)")
        except Exception as exc:
            self.log(f"[online] reconnect FAILED ({exc!r}) — retrying after {self.cooldown_s:.0f}s")
            self.closed_at = self._now()        # stay 'down' → the next tick retries after the cooldown
        finally:
            self._busy = False
            self.last_reconnect_at = self._now()
        if ok:
            c = (self.ctrl_ref or {}).get("c")
            if c is not None and hasattr(c, "on_reconnected"):
                try:
                    c.on_reconnected(live)
                except Exception as exc:
                    self.log(f"[online] panel on_reconnected failed (non-fatal): {exc!r}")

    def room_gone(self, tag: str) -> None:
        """Consumer hook: a rejoined room no longer exists → tell the panel (run bookkeeping)."""
        c = (self.ctrl_ref or {}).get("c")
        if c is not None and hasattr(c, "room_gone"):
            c.room_gone(tag)

    def banner(self) -> str:
        return (f"[online] link watchdog ACTIVE — probe after {self.probe_s:.0f}s idle, reconnect on "
                f"socket close or after {self.dead_s:.0f}s silence; auto-reconnect "
                f"{'ON' if self.reconnect_enabled else 'OFF (detect + log only, VD_LINK_RECONNECT=0)'}")

    def status(self) -> dict:
        return {"enabled": self.reconnect_enabled, "down": self.closed_at is not None,
                "idle_s": round(max(0.0, self._now() - self.last_rx), 1),
                "reconnects": self.reconnects, "probes": self.probes, "frames": self.frames}


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
                          ckpt: Path, tp_ckpt: Path, arm_of=None, recorder=None) -> None:
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
                if recorder is not None:           # W3b-0: seal this game's RL trajectory (±1 terminal)
                    try:
                        lost = getattr(b, "lost", None)
                        recorder.finish(tag, b, won=(None if won is None else bool(won)),
                                        lost=(None if lost is None else bool(lost)),
                                        opponent=getattr(b, "opponent_username", None),
                                        rating_before=getattr(b, "rating", None),
                                        opp_rating_before=getattr(b, "opponent_rating", None),
                                        turn=getattr(b, "turn", None))
                    except Exception as exc:
                        print(f"[online] ladder record finish failed (non-fatal): {exc!r}")
                # 2026-09-02 (lanes): the arm BOUND to this tag, not whatever the player's default
                # stack happens to be while several games run at once.
                arm_name, arm_pinned = (arm_of(tag) if arm_of is not None else (None, False))
                if arm_name is None:
                    arm_name = getattr(host.player, "_arm_name", None)
                    arm_pinned = bool(getattr(host.player, "_arm_pinned", False))
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
                       "ckpt": str(ckpt), "tp_ckpt": str(tp_ckpt),
                       # 2026-09-01 (era-5 W0): which bandit arm played this game (None = no bandit)
                       "arm": arm_name}
                if arm_pinned:
                    row["pinned"] = True           # 2026-09-02 serve-mode pin: a frozen-block game
                BENCH_DIR.mkdir(parents=True, exist_ok=True)
                with BENCH_LOG.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                # session detail log (2026-07-10): one human-readable line per game as it lands
                _slog(LOG_DIR / f"online_{session_id}.log",
                      f"{row['ts']}  game {row['game_idx']:>3}  {row['result'].upper():<5} "
                      f"vs {(row['opponent'] or '?'):<20} {row['turns']}t  "
                      f"team {row['ai_team'] or '?'}  {row['battle_tag']}"
                      + (f"  arm {row['arm']}" if row.get("arm") else ""))
                # B-L2 dossier capture (passive; never raises) — key data for online opponents.
                from v_dance.play.opponent_dossier import summary, update_from_battle
                if update_from_battle(b, row["result"],
                                      our_team=row["ai_team"], note=note):
                    s = summary(row["opponent"] or "")
                    print(f"[online] dossier: {s}")
                    _slog(LOG_DIR / f"online_{session_id}.log", f"    dossier: {s}")
                # 2026-09-02 (USER): the panel's game-done tap — matchup tables + thought feed.
                if getattr(_pvhb, "GAME_DONE_HOOK", None) is not None:
                    try:
                        _pvhb.GAME_DONE_HOOK(tag, b, row)
                    except Exception as exc:
                        print(f"[online] game-done hook failed (non-fatal): {exc!r}")
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
    def _arm_of(tag):                              # lanes (2026-09-02): the arm BOUND to this tag
        b = bandit                                 # assigned below, before any game can end
        if b is None:
            return None, False
        return b.arm_for(tag), b.pinned_for(tag)

    def _arm_info(tag):                            # W3b-0: what the recorder stamps per game
        b = bandit
        player = host.player
        if b is None:
            return {"arm": None, "pinned": False,
                    "tau": float(getattr(player, "_temperature", 0.0) or 0.0),
                    "top_p": float(getattr(player, "_top_p", 1.0) or 1.0),
                    "pair_decode": bool(getattr(getattr(player, "_model", None), "_pair_decode", False))}
        name = b.arm_for(tag)
        arm = b.by_name.get(name) if name else None
        model = None
        try:                                       # the arm's own model (lanes: not the default stack)
            model = (_bundles.get(name) or {}).get("model") if name else None
        except Exception:
            model = None
        if model is None:
            model = getattr(player, "_model", None)
        return {"arm": name, "pinned": b.pinned_for(tag),
                "tau": float(arm.tau) if arm else 0.0, "top_p": float(arm.top_p) if arm else 1.0,
                "pair_decode": bool(getattr(model, "_pair_decode", False)),
                "adapt_rules": (arm.adapt_rules_for(bool(args.adapt_rules)) if arm
                                else bool(args.adapt_rules))}

    # W3b-0 (2026-09-02): every ladder game as an RL trajectory (docs/w3b_ladder_ppo_design.md §4).
    recorder = None
    if os.environ.get("VD_LADDER_RECORD", "1").strip() != "0" and not args.dry_run:
        try:
            from v_dance.play.ladder_recorder import LADDER_RL_DIR, LadderRecorder
            recorder = LadderRecorder(host.player, LADDER_RL_DIR / BATTLE_FORMAT / f"{session_id}.jsonl",
                                      session_id=session_id, fmt=BATTLE_FORMAT, arm_info=_arm_info,
                                      adapt_rules=bool(args.adapt_rules))
            print(recorder.banner())
            _slog(LOG_DIR / f"online_{session_id}.log", "    " + recorder.banner())
        except Exception as exc:
            print(f"[online] ladder recorder failed to start (non-fatal): {exc!r}")
            recorder = None
    else:
        print("[online] ladder recorder OFF")

    _wrap_bench_recording(host, session_id, args.bench_note, ckpt, tp_ckpt, arm_of=_arm_of,
                          recorder=recorder)
    print(f"[online] replays -> {BENCH_DIR / 'replays' / session_id}")

    # Ladder-rating capture (2026-07-10): the post-battle |raw| rating lines arrive after the
    # game row is written, so they land as separate crash-safe "rating_update" rows keyed by
    # battle_tag; human_benchmark_report joins them onto the game rows at read time.
    # 2026-09-01 (rating reflector): the line reads "OLD &rarr; NEW". ``rating`` /
    # ``opponent_rating`` keep the PRE-battle number (poke-env's semantics, what play_ladder rows
    # hold, what the report joins on); ``*_after`` is the POST-battle number the profile page
    # shows — the panel's current rating, the per-regulation peaks and the bandit reward use it.
    def _rating_change_row(tag: str, user: str, old: int, new: int) -> None:
        us = _pvhb._toid(user) == _pvhb._toid(username)
        key = "rating" if us else "opponent_rating"
        arm = ((bandit.arm_for(tag) if bandit is not None else getattr(host.player, "_arm_name", None))
               if us else None)                    # lanes (2026-09-02): the arm BOUND to this tag
        row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "type": "rating_update", "session_id": session_id,
               "battle_tag": tag, key: old, key + "_after": new}
        pinned = bool(arm) and (bandit.pinned_for(tag) if bandit is not None
                                else bool(getattr(host.player, "_arm_pinned", False)))
        if arm:
            row["arm"] = arm
        if pinned:
            row["pinned"] = True                   # 2026-09-02 serve-mode pin: a frozen-block game
        BENCH_DIR.mkdir(parents=True, exist_ok=True)
        with BENCH_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        _slog(LOG_DIR / f"online_{session_id}.log",
              f"    rating: {user} {old} ({'us' if us else 'them'}) → {new}  {tag}"
              + (f"  arm {arm}{' (pinned)' if pinned else ''}" if arm else ""))

    _pvhb.RATING_HOOK = None                        # the panel chains its confirm logic here
    _pvhb.RATING_CHANGE_HOOK = _rating_change_row
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

    # 2026-09-01 (era-5 W0): the serve-side bandit — the ladder picks between candidate
    # checkpoints / serve knobs per game. OFF when VD_BANDIT=0 or no config file exists
    # (then the deployed stack plays every game, byte-identical to before).
    bandit = None
    if os.environ.get("VD_BANDIT", "1").strip() != "0":
        try:
            from v_dance.play import serve_bandit as _SB
            _cfg = Path(os.environ.get("VD_BANDIT_CONFIG") or _SB.DEFAULT_CONFIG)
            if _cfg.is_file():
                _arms = _SB.load_arms(_cfg)
                if _arms:
                    _arm_cache: dict = {}
                    _bundles: dict = {}

                    def _bundle_for(arm):
                        if arm.name not in _bundles:
                            _bundles[arm.name] = _SB.load_bundle(
                                arm, _arm_cache, default_battle=ckpt, default_tp=tp_ckpt,
                                adapt_rules_default=bool(args.adapt_rules))   # τ arms: forced OFF
                        return _bundles[arm.name]

                    bandit = _SB.ServeBandit(
                        _arms, fmt=BATTLE_FORMAT,
                        applier=lambda arm: _SB.apply_bundle(host.player, _bundle_for(arm)),
                        **_SB.config_params(_cfg))
                    # 2026-09-02 (lanes): each battle decides under the arm BOUND to its tag, so
                    # several games under different arms can run at once (serve_bandit.arm_scope).
                    _b = bandit

                    def _resolve_bundle(tag):
                        name = _b.arm_for(tag)
                        return _bundle_for(_b.by_name[name]) if name in _b.by_name else None

                    host.player._arm_resolver = _resolve_bundle
                    _pin_note = _SB.apply_env_pin(bandit, os.environ.get("VD_BANDIT_PIN"))
                    print(bandit.banner())
                    _slog(session_log, "    " + bandit.banner())
                    if _pin_note:                  # launch echo for the frozen mode
                        print(_pin_note)
                        _slog(session_log, "    " + _pin_note)
            else:
                print(f"[online] serve bandit OFF (no config at {_cfg})")
        except Exception as exc:
            print(f"[online] serve bandit failed to start (non-fatal — incumbent only): {exc!r}")
            bandit = None

    async def _pick(page, pool, pin, route_for=None):
        # ⚠ MUST accept route_for: _ai_consumer's challenge-accept path calls
        # _pick_ai_team(..., route_for=challenger) (the 4b router param). Without it this
        # wrapper raised TypeError on every incoming challenge accept (caught + swallowed as
        # "accept failed"), so the online bot silently declined all direct challenges.
        c = ctrl_ref.get("c")
        if c is not None and getattr(c, "bandit", None) is not None:
            c.apply_next_arm("challenge")          # idempotent: the panel's search path already did
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
            # 2026-09-01 link watchdog (USER: mid-battle disconnects) — see LinkWatch.
            _prefs = None
            if _ENV.get("PS_CLIENT_PREFS"):
                try:
                    _prefs = json.loads(_ENV["PS_CLIENT_PREFS"])
                except Exception:
                    _prefs = None
            link = LinkWatch(page=page, host=host, username=username, password=password,
                             loop=asyncio.get_running_loop(),
                             log=lambda t: (print(t), _slog(session_log, "    " + t)),
                             probe_s=float(os.environ.get("VD_LINK_PROBE_S") or 15.0),
                             dead_s=float(os.environ.get("VD_LINK_DEAD_S") or 25.0),
                             reconnect=(os.environ.get("VD_LINK_RECONNECT", "1").strip() != "0"),
                             prefs=_prefs, ctrl_ref=ctrl_ref)

            def _on_frame(p) -> None:              # incoming: SockJS unwrap → consumer feed
                link.on_raw_frame(p)               # RAW clock first: SockJS heartbeats count here
                for msg in _sockjs_unwrap(p):
                    frame_q.put_nowait(msg)
                    c = ctrl_ref.get("c")          # control panel's read-only frame tap
                    if c is not None:
                        c.tap_frame(msg)

            def _on_ws(ws) -> None:
                link.on_ws_open(ws)
                ws.on("framereceived", _on_frame)
                ws.on("close", link.on_ws_close)   # a server/network close → reconnect within ~1 s
                ws.on("socketerror", link.on_ws_error)
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
            # arm the watchdog only once the tab is logged in + set up (its tick runs from the
            # consumer's idle branch; verify-flag-uptake rule: the banner is the launch echo).
            _pvhb.LINK_TICK = link.tick
            _pvhb.ROOM_GONE_HOOK = link.room_gone
            print(link.banner())
            _slog(session_log, "    " + link.banner())
            # 2026-09-02 (USER): battle-timer mode — immediate vs the per-room grace (launch echo)
            _pvhb.TIMER_IMMEDIATE = _timer_immediate_env()
            _tb = _timer_banner(_pvhb.TIMER_IMMEDIATE, _pvhb._OPP_TIMER_S)
            print(_tb)
            _slog(session_log, "    " + _tb)
            # 2026-09-03 (USER): open team sheets — accept the offer / reject (launch echo)
            _pvhb.OTS_ACCEPT = _ots_accept_env()
            _ob = _ots_banner(_pvhb.OTS_ACCEPT)
            print(_ob)
            _slog(session_log, "    " + _ob)

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
                        log_line=lambda t: _slog(session_log, t),
                        bench_path=BENCH_LOG,          # all-time peak elo per regulation (2026-09-01)
                        bandit=bandit,                 # era-5 W0 serve-side bandit (None = off)
                        lanes_default=_int_env("VD_LADDER_LANES", 1),   # 2026-09-02 games at once
                        # 2026-09-02 (USER): matchup tables (session + all-time per team, seeded
                        # from the dossiers; VD_MATCHUPS=0 = off) + the "what the nets are
                        # thinking" feed (VD_THOUGHT_FEED=0 = off)
                        session_id=session_id, belief=_panel_belief(),
                        dossier_dir=_DOSSIER_DIR,
                        matchups=os.environ.get("VD_MATCHUPS", "1").strip() != "0",
                        thoughts=os.environ.get("VD_THOUGHT_FEED", "1").strip() != "0")
                    _ctrl = ctrl_ref["c"]
                    _ctrl.recorder = recorder      # W3b-0: status shows games/steps recorded
                    _ctrl.link = link                  # link status in the panel / Mission Control
                    print(f"[online] control server on {_ctrl.url}  —  drive it from Mission Control's "
                          f"'Online bot' tab (or open that URL for the standalone panel).")
                except Exception as exc:
                    from v_dance.play.bot_control_ui import DuplicateBotError
                    if isinstance(exc, DuplicateBotError):
                        # 2026-09-03: a second bot on this account would play the SAME battles as
                        # the running one and lose every order — refuse to serve, loudly.
                        msg = f"[online] REFUSED TO START — {exc}"
                        print(msg)
                        _slog(session_log, "    " + msg)
                        raise SystemExit(3)
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
            _pvhb.LINK_TICK = None                 # disarm the watchdog hooks (process-global)
            _pvhb.ROOM_GONE_HOOK = None
            _pvhb.REJOINING.clear()
            _pvhb.RATING_CHANGE_HOOK = None
            if bandit is not None:
                bandit.save()
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
    _tau = float(os.environ.get("VD_SERVE_TAU") or 0.0)
    if _tau > 0.0:
        print(f"[online] serve SAMPLING active: tau={_tau:g} "
              f"top_p={float(os.environ.get('VD_SERVE_TOP_P') or 1.0):g} (argmax when unset/0)")

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
