"""Bot control panel — a local, Showdown-styled web UI that INDIRECTLY drives the online bot.

Runs INSIDE the ``play_online_browser`` process (it needs the live Playwright ``page`` to inject
protocol commands) but is its OWN module by design (USER, 2026-07-10): the play file only calls
``start_control_ui(...)`` and taps frames into ``BotController.tap_frame``.

What it controls (http://127.0.0.1:<port>, default 8777 — local-only bind, no auth needed):
  · LADDER RUN — pick a team + how many rated games; the panel sends the same ``/utm <packed>`` +
    ``/search <format>`` pair the official client's Battle! button sends, then re-queues after each
    finished game until the target is reached (Stop = end the run; Cancel search = ``/cancelsearch``).
  · PRIVATE BATTLE — type a username → ``/utm`` + ``/challenge <user>, <format>`` (+ cancel).
  · AUTO-ACCEPT toggle — flips ``play_vs_human_browser.AUTO_ACCEPT`` (declines are surfaced, not
    silently dropped).
  · TEAM pin — the same pin the consumer's team pick honours for accepted challenges.
  · FORMAT — the stack binds the battle format AT IMPORT (formats.py), so the dropdown writes
    ``VDANCE_BATTLE_FORMAT`` to ``.env`` and honestly reports "applies on the next launch".

Threading model: the stdlib ``ThreadingHTTPServer`` runs on a daemon thread; every action that
touches the page/host is a coroutine shipped onto the bot's asyncio loop with
``run_coroutine_threadsafe`` (the page is loop-bound). ``tap_frame`` is called synchronously from
the websocket callback on that same loop thread — it must never raise (guarded).

Games counted toward a run's target = battles COMPLETED while the run is active (a mid-run accepted
challenge counts too — turn auto-accept off for an exact rated-only count; stated in the panel).
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import time
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Optional

import v_dance.play.play_vs_human_browser as _pvhb
from v_dance.formats import known_formats, reg_token
from v_dance.play.run_local_battle import discover_teams, load_team, resolve_team_path

_ENV_FORMAT_KEY = "VDANCE_BATTLE_FORMAT"
_RESEARCH_DELAY_S = 4.0      # settle time between a CONFIRMED-done game and the next /search
_LIVE_RETRY_S = 5.0          # re-check cadence while a battle is still live / queue is busy
_SEARCH_ACK_TIMEOUT_S = 30.0  # /search sent but no updatesearch ack → treat as lost, retry
_LIVE_STALE_S = 300.0        # a 'live' battle with NO frames this long = ghost tracking → sweep
_CONFIRM_TIMEOUT_S = 60.0    # |win| seen but no rating exchange → unrated/missed, proceed anyway

# Showdown roomid = battle-<formatid>-<battlenum>[-<private-access-suffix>]. The suffix appears on
# the room's OWN frames but ``updatesearch.games`` announces the found game under the BARE id —
# both must normalise to one canonical tag or the bare alias haunts ``_live`` forever.
_TAG_RE = re.compile(r"^(battle-[a-z0-9]+-\d+)")


def _base_tag(tag: str) -> str:
    """Canonical battle id (2026-07-10 pt-9c stall #2, game 4 of the first 10-run: the panel logged
    'battle started' for …-2647021200 AND …-2647021200-6x833…pw; only the suffixed one got the
    |win| → the bare ghost kept the resume watcher deferring forever)."""
    m = _TAG_RE.match(tag or "")
    return m.group(1) if m else (tag or "")


def _reg_label(fmt: str) -> str:
    """'gen9championsvgc2026regmb' -> 'M-B' (the regulation, the way the USER names it)."""
    tok = reg_token(fmt) or ""
    body = tok[3:].upper() if tok.startswith("reg") else tok.upper()
    if len(body) == 2:
        return f"{body[0]}-{body[1]}"
    return body or (fmt or "?")


class RatingBook:
    """Per-REGULATION rating bookkeeping for the panel (USER 2026-09-01: "peak elo in the panel,
    sorted by regulation"). Ladder ratings are per format, so every number is keyed by the
    battle's format id (parsed from its room tag). Session numbers come from the live rating
    lines; ALL-TIME peaks are seeded from the bench JSONL (every past session's rows) at startup.
    Before this, both UIs only ever showed the LAST rating — a peak field never existed."""

    def __init__(self, bench_path: Optional[Path] = None):
        self.fmts: dict = {}
        self.loaded_rows = 0
        if bench_path is not None:
            try:
                self.load_all_time(Path(bench_path))
            except Exception:
                pass                               # a bad/missing bench file = no all-time seed

    @staticmethod
    def fmt_of(tag: Optional[str]) -> Optional[str]:
        m = _TAG_RE.match(tag or "")
        return m.group(1).split("-")[1] if m else None

    def _entry(self, fmt: str) -> dict:
        return self.fmts.setdefault(fmt, {"current": None, "session_start": None,
                                          "session_peak": None, "all_time_peak": None,
                                          "all_time_games": 0})

    def load_all_time(self, path: Path) -> None:
        """Seed all-time peaks from the bench JSONL. ``rating`` is OUR rating on every row kind
        (game rows carry it directly; the online transport writes it as ``rating_update`` rows)."""
        if not path.is_file():
            return
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                fmt = self.fmt_of(r.get("battle_tag"))
                if not fmt:
                    continue
                e = self._entry(fmt)
                if r.get("type") != "rating_update":
                    e["all_time_games"] += 1
                # POST-battle value when the row has it (2026-09-01 rows), else the pre-battle one
                for rt in (r.get("rating"), r.get("rating_after")):
                    if isinstance(rt, (int, float)) and not isinstance(rt, bool):
                        e["all_time_peak"] = max(e["all_time_peak"] or 0, int(rt))
                self.loaded_rows += 1

    def record(self, tag: str, rating) -> None:
        """A live rating line for OUR account (RATING_HOOK) — session start/peak + all-time peak."""
        fmt = self.fmt_of(tag)
        if not fmt or not isinstance(rating, (int, float)) or isinstance(rating, bool):
            return
        rating = int(rating)
        e = self._entry(fmt)
        e["current"] = rating
        if e["session_start"] is None:
            e["session_start"] = rating
        e["session_peak"] = max(e["session_peak"] or 0, rating)
        e["all_time_peak"] = max(e["all_time_peak"] or 0, rating)

    def summary(self) -> list:
        """One dict per format, sorted by regulation (M-A, M-B, M-C, …)."""
        out = [{"format": fmt, "reg": _reg_label(fmt), **e} for fmt, e in self.fmts.items()]
        out.sort(key=lambda d: (reg_token(d["format"]) or d["format"]))
        return out


class BotController:
    """State + actions behind the panel. All ``page``-touching methods are coroutines that must run
    on ``self.loop`` (the bot's asyncio loop); the HTTP thread ships them there."""

    def __init__(self, *, page, host, tally: dict, ai_pool: list, fmt: str, username: str,
                 loop: asyncio.AbstractEventLoop, env_path: Path,
                 team_pin_default: Optional[str] = None,
                 log_line: Optional[Callable[[str], None]] = None,
                 auto_close_default: bool = False,
                 bench_path: Optional[Path] = None,
                 bandit=None) -> None:
        self.page, self.host, self.tally = page, host, tally
        self.ai_pool, self.fmt, self.username = list(ai_pool), fmt, username
        self.loop, self.env_path = loop, env_path
        self._log_line = log_line
        # 2026-09-01: per-regulation rating book (session start/peak + all-time peak) and the
        # online link watchdog (set by play_online_browser; None for other harnesses).
        self.ratings = RatingBook(bench_path)
        self.link = None
        # 2026-09-01 (era-5 W0): the serve-side bandit (None = off) and the site's official
        # numbers (Elo / GXE / Glicko / W-L from pokemonshowdown.com/users/<id>.json — the true
        # "elo reflector"; the in-game rating line lags it by one game).
        self.bandit = bandit
        self.official: dict = {}
        self._official_at = -1e9
        self._site_poll_enabled = os.environ.get("VD_SITE_POLL", "1").strip() != "0"
        self.team_pin: Optional[str] = team_pin_default if team_pin_default in ai_pool else None
        # ladder-run state
        self.run_active = False
        self.run_target = 0
        self.run_done = 0
        self.searching = False           # per the latest |updatesearch| frame
        self._search_outstanding = False  # bridge: /search sent, server ack (updatesearch) not seen yet
        self._search_sent_at = 0.0       # loop.time() of the last /search (stale-ack recovery)
        self._resume_scheduled = False   # single-flight guard for the run's resume timer
        self._live: set = set()          # BASE battle tags seen live, no |win|/|tie| yet (_base_tag)
        self._counted: set = set()       # BASE tags already credited (|win| re-delivery guard)
        self._last_battle_frame_at = 0.0  # loop.time() of the last battle frame (ghost-live sweep)
        self.auto_close = bool(auto_close_default)  # close each battle tab once confirmed done
        self._full_tags: dict = {}       # base → FULL roomid (private suffix; app.leaveRoom needs it)
        self._ended_unconfirmed: dict = {}  # base → {"seen": set of userids with a rating line}
        self.challenge_out: Optional[str] = None
        self.last_rating: Optional[int] = None
        self.env_fmt_saved: Optional[str] = None
        self.events: deque = deque(maxlen=200)
        self._server: Optional[ThreadingHTTPServer] = None
        self.url: Optional[str] = None

    # ── logging ──────────────────────────────────────────────────────────────
    def log(self, msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')}  {msg}"
        self.events.append(line)
        print(f"[panel] {msg}")
        if self._log_line:
            try:
                self._log_line(f"    panel: {msg}")
            except Exception:
                pass

    # ── frame tap (sync, websocket callback on the bot loop thread) ─────────
    def tap_frame(self, payload: str) -> None:
        try:
            self._tap(payload)
        except Exception:
            pass                                   # a tap must NEVER break the frame pipeline

    def _tap(self, payload: str) -> None:
        if payload.startswith(">battle"):
            tag = payload.split("\n", 1)[0].lstrip(">").strip()
            if not tag:
                return
            self._last_battle_frame_at = self.loop.time()
            self._battle_seen(tag)
            if "\n|win|" in payload or "\n|tie|" in payload:
                self._battle_ended(tag, payload)
            return
        for line in payload.split("\n"):
            parts = line.split("|")
            if len(parts) >= 3 and parts[1] == "updatesearch":
                try:
                    js = json.loads(parts[2]) or {}
                except Exception:
                    continue
                self.searching = self.fmt in (js.get("searching") or [])
                # updatesearch IS the server's authoritative queue state — it always supersedes
                # the send-time bridge flag (searching=True carries a live queue from here on).
                self._search_outstanding = False
                for g in (js.get("games") or {}):  # authoritative active-room list
                    if g.startswith("battle-"):
                        self._battle_seen(g)
            elif len(parts) >= 3 and parts[1] == "popup":
                # team-validation / search rejections surface here — the panel must show them
                self.log("popup: " + parts[2].replace("||", " ")[:200])
            elif (len(parts) >= 5 and parts[1] == "pm"
                    and parts[4].startswith("/challenge")
                    and _pvhb._toid(parts[2]) != _pvhb._toid(self.username)):
                fmt_c = parts[4][len("/challenge"):].strip()
                if fmt_c:
                    note = "" if _pvhb.AUTO_ACCEPT else " — auto-accept is OFF"
                    self.log(f"incoming challenge from {parts[2].strip()} ({fmt_c}){note}")

    def _battle_seen(self, tag: str) -> None:
        """First sighting of a battle room — via its own frames OR the ``updatesearch.games`` list.
        For a found ladder game the server sends the updatesearch FIRST; handling both paths in one
        place is the 2026-07-10 run-stall fix (the updatesearch-side add used to leave
        ``_search_outstanding`` latched True, so the +4s resume silently bailed and the run died
        after game 1 — the missing 'battle started' log line was the tell). Tags are normalised to
        the BASE roomid: the two paths name the SAME room bare vs private-suffixed (stall #2)."""
        full, tag = tag, _base_tag(tag)
        if len(full) > len(self._full_tags.get(tag, "")):
            self._full_tags[tag] = full            # remember the SUFFIXED id (leaveRoom needs it)
        if tag in self._live or tag in self._counted:
            return
        self._live.add(tag)
        self._last_battle_frame_at = self.loop.time()   # fresh room = fresh grace for the sweep
        self._search_outstanding = False           # a room appeared — the search resolved
        arm = None
        if self.bandit is not None:                # attribute this game to the arm that was applied
            try:
                arm = self.bandit.bind(tag)
            except Exception:
                arm = None
        self.log(f"battle started: {tag}" + (f"  [arm {arm}]" if arm else ""))
        # 2026-09-01: keep ONE tick pending while a battle is live. Before this, nothing was
        # scheduled between a resolved search and the battle's end, so when a dead link stopped
        # the frames the ghost-live sweep never ran and the run hung silently (09-01: 2.5 h).
        if self.run_active:
            self._schedule_resume(_LIVE_RETRY_S)

    def _battle_ended(self, tag: str, payload: str) -> None:
        tag = _base_tag(tag)                       # |win| arrives on the SUFFIXED room id (stall #2)
        if tag in self._counted:
            return                                 # |win| re-delivered on rejoin — already credited
        self._counted.add(tag)
        self._live.discard(tag)
        res = "draw"
        for line in payload.split("\n"):
            if line.startswith("|win|"):
                res = "WON" if _pvhb._toid(line[5:]) == _pvhb._toid(self.username) else "LOST"
                break
        self.log(f"battle ended — bot {res} ({tag})")
        # USER (2026-07-10): 'done' = the RATING EXCHANGE is confirmed (both |raw| rating lines seen),
        # not just |win| — the next run search AND the auto-close both wait for it. Unrated games
        # (challenges) never send rating lines → the timeout path proceeds after _CONFIRM_TIMEOUT_S.
        self._ended_unconfirmed[tag] = {"seen": set()}
        self.loop.call_later(_CONFIRM_TIMEOUT_S, lambda b=tag: self._confirm(b, timeout=True))
        if not self.run_active:
            return
        self.run_done += 1
        if self.run_done >= self.run_target:
            self.run_active = False
            self.log(f"ladder run COMPLETE — {self.run_done}/{self.run_target} games played.")
        else:
            self.log(f"ladder run {self.run_done}/{self.run_target} — next search once the "
                     f"rating exchange confirms…")
            self._schedule_resume(_RESEARCH_DELAY_S)

    def _on_rating(self, tag: str, user: str, rating) -> None:
        """Rating-hook tap (chained in start_control_ui): a |raw| rating line for an ended battle.
        Two distinct users with a rating line = elo genuinely exchanged = the battle is DONE."""
        e = self._ended_unconfirmed.get(_base_tag(tag))
        if e is None:
            return
        e["seen"].add(_pvhb._toid(user or ""))
        if len(e["seen"]) >= 2:
            self._confirm(_base_tag(tag))

    def _confirm(self, base: str, timeout: bool = False) -> None:
        """The battle's rating exchange confirmed (or timed out: unrated / lines missed) —
        release the run's next search and auto-close the room if enabled. Idempotent."""
        if self._ended_unconfirmed.pop(base, None) is None:
            return                                 # already confirmed (rating beat the timeout)
        if timeout:
            self.log(f"no rating exchange seen for {base} (unrated or missed) — proceeding.")
        else:
            self.log(f"rating exchange confirmed — {base} is done.")
        if self.auto_close:
            self._close_room(base)
        self.schedule_site_poll()                  # refresh the site's official numbers
        if self.run_active:
            self._schedule_resume(_RESEARCH_DELAY_S)

    # ── 2026-09-01: serve-side bandit + the site's official rating ───────────
    def apply_next_arm(self, reason: str = "pre-game"):
        """Pick + apply the bandit arm for the NEXT battle (idempotent until that battle starts —
        the search path and the challenge path both call this). Skipped while a battle is live:
        a model swap mid-game would split one game across two arms."""
        if self.bandit is None:
            return None
        if self._live:
            return None
        try:
            before = self.bandit.pending
            arm = self.bandit.apply_pending()
        except Exception as exc:
            self.log(f"bandit: apply failed ({exc!r}) — playing the current model")
            return None
        if arm is not None and before != arm.name:
            extra = f" τ={arm.tau:g}" if arm.tau else ""
            self.log(f"arm → {arm.name}{extra} ({reason})")
        return arm

    def schedule_site_poll(self, min_interval: float = 15.0) -> None:
        """Fetch pokemonshowdown.com/users/<id>.json (Elo, GXE, Glicko-1, W-L per format) off
        the bot loop; rate-limited; never raises. Result lands in ``self.official``."""
        if not self._site_poll_enabled or self.loop is None:
            return
        try:
            now = self.loop.time()
        except Exception:
            return
        if now - self._official_at < min_interval:
            return
        self._official_at = now

        async def _go():
            from v_dance.play.serve_bandit import fetch_official_ratings
            uid = _pvhb._toid(self.username)
            try:
                data = await self.loop.run_in_executor(None, fetch_official_ratings, uid)
                self.official = {"ratings": data, "fetched": time.strftime("%H:%M:%S")}
            except Exception as exc:
                if not self.official.get("error"):
                    self.log(f"site rating poll failed (non-fatal): {type(exc).__name__}")
                self.official = {**self.official, "error": str(exc)[:80]}

        try:
            self.loop.create_task(_go())
        except Exception:
            pass

    def _close_room(self, base: str) -> None:
        full = self._full_tags.get(base, base)

        async def _leave():
            await self.page.evaluate("(r) => { if (app.rooms[r]) app.leaveRoom(r); }", full)
            self.log(f"closed battle tab {full}")

        self.loop.create_task(self._guarded(_leave()))

    def room_gone(self, tag: str) -> None:
        """The consumer got ``|noinit|`` for a room we thought was live (a reconnect rejoin after
        the game ended while the link was down). Drop it from the live set, count it toward the
        run (it WAS a game — decided by the timer), and get the run moving again."""
        base = _base_tag(tag)
        was_live = base in self._live
        self._live.discard(base)
        self._full_tags.pop(base, None)
        self.log(f"room gone — {base} no longer exists (decided while the link was down); dropped as live")
        if was_live and base not in self._counted and self.run_active:
            self._counted.add(base)
            self.run_done += 1
            if self.run_done >= self.run_target:
                self.run_active = False
                self.log(f"ladder run COMPLETE — {self.run_done}/{self.run_target} games played.")
                return
            self.log(f"ladder run {self.run_done}/{self.run_target} (room-gone game counted)")
        if self.run_active:
            self._schedule_resume(_RESEARCH_DELAY_S)

    def on_reconnected(self, rejoining=None) -> None:
        """The online link came back (LinkWatch). The server re-states the queue via
        ``|updatesearch|`` after login, so drop the send-time bridge flag; an active run gets its
        next search out once nothing is live (a rejoined battle keeps the tick deferring)."""
        self._search_outstanding = False
        self.searching = False
        rj = [_base_tag(t) for t in (rejoining or [])]
        self.log("link reconnected — " + (f"rejoining {', '.join(rj)}" if rj else "no live battle")
                 + ("; run resumes" if self.run_active else ""))
        if self.run_active:
            self._schedule_resume(_RESEARCH_DELAY_S)

    def _schedule_resume(self, delay: float) -> None:
        """Single-flight resume timer: while a run is active there is always EXACTLY ONE pending
        tick that keeps re-checking until the next search actually goes out — a busy/queued state
        can delay the run but can never silently kill it (the 2026-07-10 stall hardening)."""
        if self._resume_scheduled:
            return
        self._resume_scheduled = True
        self.loop.call_later(delay, self._resume_tick)

    def _resume_tick(self) -> None:
        self._resume_scheduled = False
        if not self.run_active:
            return
        if self._search_outstanding \
                and self.loop.time() - self._search_sent_at > _SEARCH_ACK_TIMEOUT_S:
            self._search_outstanding = False       # /search never acked (lost/rejected) → retry
            self.log("search was never acknowledged — retrying.")
        # Ghost-live sweep: a 'live' battle whose frames stopped _LIVE_STALE_S ago is tracking
        # debris, not a game (defense-in-depth behind the _base_tag aliasing fix) — clear it so
        # the run resumes instead of deferring forever. A real battle always has frames flowing
        # well inside the window (turn timer alone forces action every ~2-3 min).
        if self._live and self.loop.time() - self._last_battle_frame_at > _LIVE_STALE_S:
            self.log(f"live-battle tracking looked stale — clearing {sorted(self._live)}.")
            self._live.clear()
        # _ended_unconfirmed: an ended game whose rating exchange isn't confirmed yet — its own
        # _CONFIRM_TIMEOUT_S timer guarantees it always clears, so this defer is bounded.
        if self.searching or self._search_outstanding or self._live or self._ended_unconfirmed:
            self._schedule_resume(_LIVE_RETRY_S)   # queued / live / awaiting Δelo — keep watching
            return
        self.loop.create_task(self._guarded(self._do_search()))

    async def _guarded(self, coro) -> None:
        try:
            await coro
        except Exception as exc:                   # actions must never kill the bot loop
            self.log(f"action failed: {exc!r}")

    # ── page actions (bot loop only) ─────────────────────────────────────────
    async def _utm_current_team(self) -> tuple:
        """Pick the battle team (panel pin → Teambuilder-open → random), bind it to the host's
        decision core, and ``/utm`` it — the exact sequence of the consumer's challenge-accept
        path, so the server-side team and the encoder's own-side team can never diverge."""
        self.apply_next_arm("search")              # era-5 W0: which checkpoint/knobs play next
        name, src = await _pvhb._pick_ai_team(self.page, self.ai_pool, self.team_pin)
        scoped = next((p for p in discover_teams(reg=self.fmt) if Path(p).name == name), name)
        team = self._load_scoped_team(scoped)
        self.host.player.update_team(team)
        self.host.player._team_name = name
        packed = self.host.player._team.yield_team()
        await self.page.evaluate("(t) => app.socket.send('|/utm ' + t)", packed)
        return name, src

    def _load_scoped_team(self, scoped):           # seam: tests stub the filesystem load
        return load_team(resolve_team_path(scoped))

    async def _do_search(self) -> None:
        if self.searching or self._search_outstanding:
            self.log("already searching — ignored.")
            if self.run_active:                    # a guarded skip must not kill an active run
                self._schedule_resume(_LIVE_RETRY_S)
            return
        name, src = await self._utm_current_team()
        await self.page.evaluate("(f) => app.socket.send('|/search ' + f)", self.fmt)
        self._search_outstanding = True
        self._search_sent_at = self.loop.time()
        self.log(f"searching ladder ({self.fmt}) with team {name!r} [{src}]")
        if self.run_active:                        # the tick-always-pending invariant (2026-09-01)
            self._schedule_resume(_LIVE_RETRY_S)

    async def start_ladder(self, games: int, team: Optional[str]) -> None:
        self.set_team(team)
        self.run_target = max(1, min(int(games), 500))
        self.run_done = 0
        self.run_active = True
        self.log(f"ladder run started — target {self.run_target} game(s), "
                 f"team {self.team_pin or '(Teambuilder/default)'}")
        if self._live:
            self.log("a battle is live — the run queues after it ends.")
            self._schedule_resume(_LIVE_RETRY_S)
        else:
            await self._do_search()

    async def stop_ladder(self) -> None:
        was = self.run_active
        self.run_active = False
        if self.searching or self._search_outstanding:
            await self.page.evaluate("() => app.socket.send('|/cancelsearch')")
            self._search_outstanding = False
        self.log("ladder run stopped." if was else "search cancelled.")

    async def send_challenge(self, user: str, team: Optional[str]) -> None:
        user = (user or "").strip()
        if not _pvhb._toid(user):
            raise ValueError("empty username")
        self.set_team(team)
        name, src = await self._utm_current_team()
        await self.page.evaluate("(a) => app.socket.send('|/challenge ' + a)",
                                 f"{user}, {self.fmt}")
        self.challenge_out = user
        self.log(f"challenge sent to {user!r} ({self.fmt}) with team {name!r} [{src}]")

    async def cancel_challenge(self) -> None:
        if not self.challenge_out:
            self.log("no outgoing challenge to cancel.")
            return
        await self.page.evaluate("(u) => app.socket.send('|/cancelchallenge ' + u)",
                                 _pvhb._toid(self.challenge_out))
        self.log(f"challenge to {self.challenge_out!r} cancelled.")
        self.challenge_out = None

    # ── sync setters (safe from the HTTP thread) ─────────────────────────────
    def set_auto_accept(self, on: bool) -> None:
        _pvhb.AUTO_ACCEPT = bool(on)
        self.log(f"auto-accept challenges: {'ON' if on else 'OFF'}")

    def set_auto_close(self, on: bool) -> None:
        self.auto_close = bool(on)
        self.log(f"auto-close finished battle tabs: {'ON' if on else 'OFF'}")

    def set_team(self, team: Optional[str]) -> None:
        team = (team or "").strip() or None
        if team is not None and team not in self.ai_pool:
            raise ValueError(f"unknown team {team!r}")
        if team != self.team_pin:
            self.team_pin = team
            self.log(f"team pin → {team or '(Teambuilder/default)'}")

    def save_format(self, fmt: str) -> str:
        """Persist a format choice to .env (atomic — mirrors play_online_browser._write_avatar).
        The stack binds the format at import, so this HONESTLY applies on the next launch."""
        fmt = (fmt or "").strip()
        if not fmt:
            raise ValueError("empty format")
        if fmt == self.fmt:
            self.env_fmt_saved = None
            return f"{fmt} is already the active format."
        lines = (self.env_path.read_text(encoding="utf-8").splitlines()
                 if self.env_path.is_file() else [])
        for i, ln in enumerate(lines):
            if ln.split("=", 1)[0].strip() == _ENV_FORMAT_KEY:
                lines[i] = f"{_ENV_FORMAT_KEY}={fmt}"
                break
        else:
            lines.append(f"{_ENV_FORMAT_KEY}={fmt}")
        tmp = self.env_path.with_suffix(".tmp")
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        tmp.replace(self.env_path)
        self.env_fmt_saved = fmt
        self.log(f"format {fmt} saved to .env — applies on the NEXT bot launch "
                 f"(this session stays on {self.fmt}).")
        return f"{fmt} saved — restart the bot to play it."

    # ── status for the panel ──────────────────────────────────────────────────
    def status(self) -> dict:
        return {
            "username": self.username, "format": self.fmt,
            "formats": (known_formats() or [self.fmt]),
            "teams": self.ai_pool, "team_pin": self.team_pin or "",
            "auto_accept": bool(_pvhb.AUTO_ACCEPT),
            "auto_close": bool(self.auto_close),
            "awaiting_confirm": sorted(self._ended_unconfirmed),
            "run": {"active": self.run_active, "target": self.run_target, "done": self.run_done},
            "searching": bool(self.searching or self._search_outstanding),
            "live": sorted(self._live), "tally": dict(self.tally),
            "rating": self.last_rating, "challenge_out": self.challenge_out or "",
            "env_fmt_saved": self.env_fmt_saved or "",
            "events": list(self.events)[-80:],
            # 2026-09-01: peak elo per regulation + the online link watchdog's health
            "peaks": self.ratings.summary(),
            "link": (self.link.status() if self.link is not None else None),
            # 2026-09-01: the site's official numbers + the serve-side bandit's evidence
            "official": self.official,
            "bandit": (self.bandit.summary() if self.bandit is not None else None),
        }

    def stop(self) -> None:
        srv, self._server = self._server, None
        if srv is not None:
            try:
                srv.shutdown()
            except Exception:
                pass


# ── HTTP layer ────────────────────────────────────────────────────────────────
def _make_handler(ctrl: BotController):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a) -> None:        # silence per-request stderr noise
            pass

        def _json(self, obj, code=200) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:                 # noqa: N802 (stdlib name)
            if self.path == "/" or self.path.startswith("/index"):
                body = _PANEL_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/status":
                self._json(ctrl.status())
            else:
                self._json({"ok": False, "error": "not found"}, 404)

        def do_POST(self) -> None:                # noqa: N802
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}") if n else {}
            except Exception:
                self._json({"ok": False, "error": "bad JSON"}, 400)
                return

            def run_async(coro):
                fut = asyncio.run_coroutine_threadsafe(ctrl._guarded(coro), ctrl.loop)
                fut.result(timeout=30)            # _guarded logs failures; result = completion only

            try:
                p = self.path
                note = ""
                if p == "/api/ladder/start":
                    run_async(ctrl.start_ladder(int(body.get("games") or 1), body.get("team")))
                elif p == "/api/ladder/stop":
                    run_async(ctrl.stop_ladder())
                elif p == "/api/challenge":
                    run_async(ctrl.send_challenge(body.get("user") or "", body.get("team")))
                elif p == "/api/challenge/cancel":
                    run_async(ctrl.cancel_challenge())
                elif p == "/api/options":
                    if "auto_accept" in body:
                        ctrl.set_auto_accept(bool(body.get("auto_accept")))
                    if "auto_close" in body:
                        ctrl.set_auto_close(bool(body.get("auto_close")))
                elif p == "/api/team":
                    ctrl.set_team(body.get("team"))
                elif p == "/api/format":
                    note = ctrl.save_format(body.get("format") or "")
                else:
                    self._json({"ok": False, "error": "not found"}, 404)
                    return
                self._json({"ok": True, "note": note})
            except Exception as exc:
                self._json({"ok": False, "error": str(exc)}, 400)

    return Handler


def start_control_ui(*, page, host, tally: dict, ai_pool: list, fmt: str, username: str,
                     loop: asyncio.AbstractEventLoop, env_path: Path, port: int = 8777,
                     team_pin_default: Optional[str] = None,
                     log_line: Optional[Callable[[str], None]] = None,
                     auto_close_default: bool = False,
                     open_browser: bool = True,
                     bench_path: Optional[Path] = None,
                     bandit=None) -> BotController:
    """Build the controller + serve the panel on 127.0.0.1 (first free port in [port, port+9]).
    Also chains itself onto ``RATING_HOOK`` (the Δelo battle-done confirm) and
    ``RATING_CHANGE_HOOK`` (post-battle rating → display, peaks, bandit reward) — call AFTER the
    harness set them. ``bench_path`` seeds the all-time peak per regulation."""
    ctrl = BotController(page=page, host=host, tally=tally, ai_pool=ai_pool, fmt=fmt,
                         username=username, loop=loop, env_path=env_path,
                         team_pin_default=team_pin_default, log_line=log_line,
                         auto_close_default=auto_close_default, bench_path=bench_path,
                         bandit=bandit)
    prev_hook = _pvhb.RATING_HOOK

    def _hook(tag, user, rating):                  # PRE-battle number (poke-env semantics)
        if prev_hook is not None:
            prev_hook(tag, user, rating)
        if _pvhb._toid(user) == _pvhb._toid(username):
            if ctrl.last_rating is None:
                ctrl.last_rating = rating
            ctrl.ratings.record(tag, rating)       # session START = the rating before game 1
        ctrl._on_rating(tag, user, rating)         # the Δelo-exchanged battle-done confirm

    _pvhb.RATING_HOOK = _hook
    prev_change = _pvhb.RATING_CHANGE_HOOK

    def _change_hook(tag, user, old, new):         # POST-battle number (what the site shows)
        if prev_change is not None:
            prev_change(tag, user, old, new)
        if _pvhb._toid(user) != _pvhb._toid(username):
            return
        ctrl.last_rating = new
        ctrl.ratings.record(tag, new)
        delta = int(new) - int(old)
        line = f"rating: {old} → {new} ({delta:+d})"
        if ctrl.bandit is not None:
            try:
                name = ctrl.bandit.observe(tag, delta)
            except Exception as exc:
                name = None
                ctrl.log(f"bandit: observe failed (non-fatal): {exc!r}")
            if name:
                s = ctrl.bandit.stats[name]
                line += f"  → arm {name}: {s.wins}W-{s.losses}L, mean Δ {s.mean_delta():+.1f} over {s.n}"
                if s.retired:
                    line += "  ⛔ RETIRED"
        ctrl.log(line)

    _pvhb.RATING_CHANGE_HOOK = _change_hook
    ctrl.schedule_site_poll(min_interval=0.0)      # the site's numbers at startup

    handler = _make_handler(ctrl)
    last_exc = None
    for p in range(port, port + 10):
        try:
            ctrl._server = ThreadingHTTPServer(("127.0.0.1", p), handler)
            break
        except OSError as exc:
            last_exc = exc
    if ctrl._server is None:
        raise OSError(f"control panel: no free port in {port}-{port + 9}: {last_exc}")
    threading.Thread(target=ctrl._server.serve_forever, name="bot-control-ui", daemon=True).start()
    ctrl.url = f"http://127.0.0.1:{ctrl._server.server_address[1]}/"
    ctrl.log(f"control panel → {ctrl.url}")
    if open_browser:
        try:
            webbrowser.open(ctrl.url)
        except Exception:
            pass
    return ctrl


# ── The panel (Showdown-styled, self-contained: no external assets) ───────────
_PANEL_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Victory Dance — Bot Control</title>
<style>
  * { box-sizing: border-box; }
  html, body { margin:0; padding:0; font: 10pt Verdana, sans-serif; color:#222;
               background: linear-gradient(180deg, #4a6785 0%, #34506e 55%, #2b4257 100%) fixed; }
  .header { background: linear-gradient(#32506e, #1b3a5b); color:#fff; padding:9px 16px;
            border-bottom: 2px solid #12283d; display:flex; align-items:center; gap:10px; }
  .header b { font-size: 12pt; letter-spacing:.2px; }
  .dot { width:10px; height:10px; border-radius:50%; background:#5cab3c; display:inline-block;
         box-shadow: 0 0 4px #5cab3c; }
  .dot.busy { background:#e8a33d; box-shadow:0 0 4px #e8a33d; }
  .hdr-right { margin-left:auto; font-size:8.5pt; opacity:.9; }
  .wrap { max-width: 920px; margin: 18px auto; padding: 0 12px; display:grid;
          grid-template-columns: 1fr 1fr; gap: 14px; }
  .panel { background:#f8fbfd; border:1px solid #8b9dad; border-radius:8px; overflow:hidden;
           box-shadow: 0 2px 6px rgba(0,0,0,.35); }
  .panel h2 { margin:0; padding:6px 10px; font: bold 10.5pt Verdana; background:#e0e7ea;
              border-bottom:1px solid #b5c3ce; color:#36485a; }
  .panel .pad { padding: 12px; }
  label.fld { display:block; font: bold 8.5pt Verdana; color:#4a5a68; margin: 8px 0 3px; }
  select, input[type=text], input[type=number] {
      width:100%; padding:5px 7px; font:10pt Verdana; border:1px solid #91a3b2;
      border-radius:4px; background:#fff; }
  .battlebtn { display:block; width:100%; margin-top:12px; padding:11px 0; cursor:pointer;
      font: bold 13pt Verdana; color:#fff; text-shadow: 0 1px 0 #16324f;
      background: linear-gradient(#4d81b5, #2c5a8c); border:1px solid #16324f; border-radius:9px; }
  .battlebtn:hover { background: linear-gradient(#5a8ec2, #336598); }
  .battlebtn:disabled { opacity:.55; cursor:default; }
  .btn { padding:5px 12px; cursor:pointer; font: bold 9pt Verdana; color:#222;
      background: linear-gradient(#f6f9fb, #d4dbe0); border:1px solid #91a3b2; border-radius:5px; }
  .btn:hover { background: linear-gradient(#fff, #e2e8ec); }
  .row { display:flex; gap:8px; margin-top:10px; }
  .muted { color:#5c6c7a; font-size:8.5pt; }
  .prog { margin-top:10px; font: bold 10pt Verdana; color:#2c5a8c; }
  .toggle { display:flex; align-items:center; gap:8px; margin:6px 0; font:10pt Verdana; }
  .toggle input { width:16px; height:16px; }
  .full { grid-column: 1 / -1; }
  #log { height: 190px; overflow-y:auto; background:#eef2f5; border-top:1px solid #b5c3ce;
         padding:8px 10px; font: 8.5pt Consolas, monospace; white-space:pre-wrap; color:#33424f; }
  .stat { display:flex; gap:16px; flex-wrap:wrap; font:10pt Verdana; }
  .stat b { color:#2c5a8c; }
  .note { color:#8a5a1f; font-size:8.5pt; margin-top:4px; min-height:1.2em; }
</style></head><body>
<div class="header"><span class="dot" id="dot"></span><b>Victory Dance</b>
  <span id="who">connecting…</span>
  <span class="hdr-right" id="fmtLabel"></span></div>
<div class="wrap">

  <div class="panel"><h2>Ladder</h2><div class="pad">
    <label class="fld">Format</label>
    <select id="fmtSel"></select>
    <div class="note" id="fmtNote"></div>
    <label class="fld">Team</label>
    <select id="teamSel"></select>
    <label class="fld">Rated games to play</label>
    <input type="number" id="games" min="1" max="500" value="10">
    <button class="battlebtn" id="startBtn">Battle!</button>
    <div class="row">
      <button class="btn" id="stopBtn">Stop run</button>
      <button class="btn" id="cancelSearchBtn">Cancel search</button>
    </div>
    <div class="prog" id="prog"></div>
  </div></div>

  <div>
    <div class="panel"><h2>Private battle</h2><div class="pad">
      <label class="fld">Opponent username</label>
      <input type="text" id="challengeUser" placeholder="exact Showdown username">
      <div class="row">
        <button class="btn" id="challengeBtn" style="flex:1">Challenge</button>
        <button class="btn" id="cancelChallengeBtn">Cancel</button>
      </div>
      <div class="muted" id="challengeOut" style="margin-top:6px"></div>
    </div></div>

    <div class="panel" style="margin-top:14px"><h2>Options &amp; status</h2><div class="pad">
      <div class="toggle"><input type="checkbox" id="autoAccept">
        <label for="autoAccept">Auto-accept incoming challenges</label></div>
      <div class="toggle"><input type="checkbox" id="autoClose">
        <label for="autoClose">Auto-close finished battle tabs</label></div>
      <div class="muted">Games finished while a run is active count toward its target —
        turn auto-accept off for an exact rated-only count. Tabs close (and the next search
        starts) only after the rating exchange confirms the battle is done.</div>
      <div class="stat" style="margin-top:10px">
        <span>Rating: <b id="rating">—</b></span>
        <span>Session peak: <b id="peak">—</b></span>
        <span>Session: <b id="tally">0-0</b></span>
        <span>Live battles: <b id="live">0</b></span>
        <span>Link: <b id="link">—</b></span>
      </div>
      <div class="muted" id="peaks" style="margin-top:6px"></div>
      <div class="muted" id="site" style="margin-top:4px"></div>
      <div class="muted" id="bandit" style="margin-top:4px;white-space:pre-line"></div>
    </div></div>
  </div>

  <div class="panel full"><h2>Activity</h2><div id="log"></div></div>
</div>
<script>
const $ = (id) => document.getElementById(id);
let filled = false, lastNote = "";

async function api(path, body) {
  const r = await fetch(path, {method:'POST', headers:{'Content-Type':'application/json'},
                               body: JSON.stringify(body || {})});
  const js = await r.json();
  if (!js.ok) alert(js.error || 'action failed');
  else if (js.note) { lastNote = js.note; $('fmtNote').textContent = js.note; }
  refresh();
  return js;
}

function fillOnce(s) {
  if (filled) return;
  filled = true;
  $('teamSel').innerHTML = '<option value="">— team open in Teambuilder / default —</option>' +
    s.teams.map(t => `<option>${t}</option>`).join('');
  const fmts = s.formats.includes(s.format) ? s.formats : [s.format, ...s.formats];
  $('fmtSel').innerHTML = fmts.map(f =>
    `<option value="${f}">${f}${f === s.format ? ' (active)' : ''}</option>`).join('');
  $('fmtSel').value = s.format;
  $('teamSel').value = s.team_pin || '';
}

function render(s) {
  fillOnce(s);
  $('who').textContent = s.username;
  $('fmtLabel').textContent = s.format;
  $('dot').className = 'dot' + ((s.live.length || s.searching) ? ' busy' : '');
  $('autoAccept').checked = s.auto_accept;
  $('autoClose').checked = s.auto_close;
  $('rating').textContent = s.rating ?? '—';
  $('tally').textContent = `${s.tally.ai || 0}W-${s.tally.you || 0}L` +
                           (s.tally.draw ? `-${s.tally.draw}D` : '');
  $('live').textContent = s.live.length;
  // 2026-09-01: session peak for the active regulation + all-time peaks by regulation + link health
  const peaks = s.peaks || [];
  const cur = peaks.find(p => p.format === s.format) || peaks[0];
  $('peak').textContent = (cur && cur.session_peak != null)
      ? `${cur.session_peak} (start ${cur.session_start})` : '—';
  $('peaks').textContent = peaks.length
      ? 'All-time peak by regulation: ' + peaks.map(p =>
          `${p.reg} ${p.all_time_peak ?? '—'} (${p.all_time_games} games)`).join(' · ')
      : '';
  $('link').textContent = s.link
      ? (s.link.down ? 'DOWN — reconnecting'
                     : `ok · ${s.link.idle_s}s since last frame` +
                       (s.link.reconnects ? ` · ${s.link.reconnects} reconnect(s)` : ''))
      : '—';
  // the site's official numbers (pokemonshowdown.com/users/<id>.json) + the bandit's evidence
  const of = (s.official && s.official.ratings) ? s.official.ratings[s.format] : null;
  $('site').textContent = of
      ? `Site: Elo ${of.elo} · GXE ${of.gxe ?? '—'}% · Glicko ${of.glicko ?? '—'}±${of.glicko_dev ?? '—'}` +
        ` · ${of.w}W-${of.l}L lifetime (as of ${s.official.fetched})`
      : (s.official && s.official.error ? 'Site: rating poll failed' : '');
  const arms = s.bandit || [];
  $('bandit').textContent = arms.length
      ? 'Bandit (◀ = playing, * = incumbent): ' + arms.map(a =>
          `${a.name}${a.incumbent ? '*' : ''}${a.current ? ' ◀' : ''} ${a.wins}W-${a.losses}L` +
          ` Δ${a.mean_delta >= 0 ? '+' : ''}${a.mean_delta}${a.retired ? ' RETIRED' : ''}`).join(' · ')
      : '';
  $('challengeOut').textContent = s.challenge_out ? ('outgoing challenge: ' + s.challenge_out) : '';
  const wait = s.awaiting_confirm.length ? ' — confirming result…' : '';
  $('prog').textContent = s.run.active
      ? `Run: ${s.run.done}/${s.run.target} played` + (s.searching ? ' — searching…' : wait)
      : (s.searching ? 'Searching…' : '');
  $('startBtn').disabled = s.run.active;
  if (s.env_fmt_saved && !lastNote)
    $('fmtNote').textContent = s.env_fmt_saved + ' saved — restart the bot to play it.';
  const log = $('log'), atEnd = log.scrollTop + log.clientHeight >= log.scrollHeight - 8;
  log.textContent = s.events.join('\\n');
  if (atEnd) log.scrollTop = log.scrollHeight;
}

async function refresh() {
  try {
    const s = await (await fetch('/api/status')).json();
    render(s);
  } catch (e) { $('who').textContent = 'bot offline?'; $('dot').className = 'dot busy'; }
}

$('startBtn').onclick = () => api('/api/ladder/start',
    {games: +$('games').value || 1, team: $('teamSel').value});
$('stopBtn').onclick = () => api('/api/ladder/stop');
$('cancelSearchBtn').onclick = () => api('/api/ladder/stop');
$('challengeBtn').onclick = () => api('/api/challenge',
    {user: $('challengeUser').value, team: $('teamSel').value});
$('cancelChallengeBtn').onclick = () => api('/api/challenge/cancel');
$('autoAccept').onchange = () => api('/api/options', {auto_accept: $('autoAccept').checked});
$('autoClose').onchange = () => api('/api/options', {auto_close: $('autoClose').checked});
$('teamSel').onchange = () => api('/api/team', {team: $('teamSel').value});
$('fmtSel').onchange = () => api('/api/format', {format: $('fmtSel').value});

setInterval(refresh, 2000);
refresh();
</script></body></html>
"""
