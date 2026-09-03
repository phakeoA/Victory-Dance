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
from v_dance.play.matchup_book import ALL_TEAMS, MatchupBook, display_species

_ENV_FORMAT_KEY = "VDANCE_BATTLE_FORMAT"
_RESEARCH_DELAY_S = 4.0      # settle time between a CONFIRMED-done game and the next /search
_LIVE_RETRY_S = 5.0          # re-check cadence while a battle is still live / queue is busy
_SEARCH_ACK_TIMEOUT_S = 30.0  # /search sent but no updatesearch ack → treat as lost, retry
_LIVE_STALE_S = 300.0        # a 'live' battle with NO frames this long = ghost tracking → sweep
# 2026-09-02 (USER): ladder LANES — several rated games at once. Showdown's own limits (pinned
# server/monitor.ts): "limited to 5 games at the same time" per account, "12 battles and team
# validations every 3 minutes" per IP, and ONE ladder search per format at a time.
_MAX_LANES = 5
_LANE_FILL_S = 3.0           # a found game frees the single search slot → fill the next lane soon
_PREP_WINDOW_S = 180.0
_PREP_MAX = 10               # our ceiling under the server's 12 (challenges share the budget)
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
                 bandit=None, lanes_default: int = 1,
                 session_id: Optional[str] = None, belief=None,
                 dossier_dir: Optional[Path] = None, matchups: bool = True) -> None:
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
        # 2026-09-02 (USER): lanes = rated games at once (1..5, Showdown's per-account cap)
        self.lanes = max(1, min(int(lanes_default or 1), _MAX_LANES))
        self._search_times: list = []    # loop.time() of every /search sent (prep-rate guard)
        self._last_frame_at: dict = {}   # base tag -> loop.time() of ITS last frame (per-room sweep)
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
        # 2026-09-02 (USER): matchup tables — SESSION (this process) + ALL-TIME per (format, OUR
        # team), seeded from the opponent dossiers; None = off. ``belief`` supplies the item /
        # ability defaults for unrevealed mons ("(belief)" in the UI).
        self.session_id = session_id
        self.belief = belief
        self.matchup_team: str = ""          # "" = follow the ladder-run team; "*" = all teams
        self.matchups: Optional[MatchupBook] = None
        if matchups:
            try:
                self.matchups = MatchupBook()
                if dossier_dir is not None:
                    self.matchups.load_dossiers(dossier_dir)
                self.log(self.matchups.banner()[len("[panel] "):])
            except Exception as exc:
                self.matchups = None
                self.log(f"matchup book failed to load (non-fatal): {exc!r}")
        self.thoughts = None                 # "what the nets are thinking" feed (start_control_ui)

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
            self._last_frame_at[_base_tag(tag)] = self._last_battle_frame_at
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
        pinned = bool(arm) and self.bandit is not None and self.bandit.pinned_for(tag)
        self._last_frame_at.setdefault(tag, self.loop.time())
        lane = f"  (lane {len(self._live)}/{self.lanes})" if self.lanes > 1 else ""
        self.log(f"battle started: {tag}"
                 + (f"  [arm {arm}{' pinned' if pinned else ''}]" if arm else "") + lane)
        # 2026-09-01: keep ONE tick pending while a battle is live. Before this, nothing was
        # scheduled between a resolved search and the battle's end, so when a dead link stopped
        # the frames the ghost-live sweep never ran and the run hung silently (09-01: 2.5 h).
        # 2026-09-02 (lanes): a found game also frees the single search slot — fill the next lane.
        if self.run_active:
            self._schedule_resume(_LANE_FILL_S if (self.lanes > 1 and self._lane_free())
                                  else _LIVE_RETRY_S)

    def _battle_ended(self, tag: str, payload: str) -> None:
        tag = _base_tag(tag)                       # |win| arrives on the SUFFIXED room id (stall #2)
        if tag in self._counted:
            return                                 # |win| re-delivered on rejoin — already credited
        self._counted.add(tag)
        self._live.discard(tag)
        self._last_frame_at.pop(tag, None)
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
        the search path and the challenge path both call this). Skipped while a battle is live
        UNLESS the player resolves its arm per battle tag (lanes, 2026-09-02) — without that, a
        model swap mid-game would split one game across two arms."""
        if self.bandit is None:
            return None
        if self._live and getattr(self.host.player, "_arm_resolver", None) is None:
            return None
        try:
            before = self.bandit.pending
            arm = self.bandit.apply_pending()
        except Exception as exc:
            self.log(f"bandit: apply failed ({exc!r}) — playing the current model")
            return None
        if arm is not None:
            try:                                   # bench rows read this: "pinned": True on frozen games
                self.host.player._arm_pinned = bool(self.bandit.pinned) and self.bandit.pinned == arm.name
            except Exception:
                pass
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
        self._last_frame_at.pop(base, None)
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
        # 2026-09-02 (lanes): judged PER ROOM — one dead room must not hide behind the others'
        # frames; global silence (no frames from any room) still sweeps every room.
        now = self.loop.time()
        silent_all = now - self._last_battle_frame_at > _LIVE_STALE_S
        stale = [t for t in sorted(self._live)
                 if silent_all or now - self._last_frame_at.get(t, self._last_battle_frame_at) > _LIVE_STALE_S]
        if stale:
            self.log(f"live-battle tracking looked stale — clearing {stale}.")
            for t in stale:
                self._live.discard(t)
                self._last_frame_at.pop(t, None)
        if self.searching or self._search_outstanding:
            self._schedule_resume(_LIVE_RETRY_S)   # ONE ladder search per format at a time (server)
            return
        if self.lanes <= 1:
            # One lane = the 2026-07-10 contract: the next search waits for the live game AND its
            # rating exchange (_ended_unconfirmed clears by its own _CONFIRM_TIMEOUT_S timer, so
            # this defer is bounded).
            if self._live or self._ended_unconfirmed:
                self._schedule_resume(_LIVE_RETRY_S)   # live / awaiting Δelo — keep watching
                return
        elif not self._lane_free():
            self._schedule_resume(_LIVE_RETRY_S)   # lanes full / target covered — keep watching
            return
        elif not self._prep_budget_ok():
            self.log(f"search deferred — {_PREP_MAX} searches in the last {_PREP_WINDOW_S:.0f}s "
                     f"(the server caps battle preps at 12 per 3 min)")
            self._schedule_resume(_LIVE_RETRY_S)
            return
        self.loop.create_task(self._guarded(self._do_search()))

    def _lane_free(self) -> bool:
        """Capacity for another rated game: a lane is open AND the run's target is not already
        covered by finished + live games (lanes, 2026-09-02)."""
        if len(self._live) >= self.lanes:
            return False
        return (self.run_done + len(self._live)) < self.run_target

    def _prep_budget_ok(self) -> bool:
        now = self.loop.time()
        self._search_times = [t for t in self._search_times if now - t < _PREP_WINDOW_S]
        return len(self._search_times) < _PREP_MAX

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
        self._search_times.append(self._search_sent_at)   # prep-rate guard (lanes)
        self.log(f"searching ladder ({self.fmt}) with team {name!r} [{src}]")
        if self.run_active:                        # the tick-always-pending invariant (2026-09-01)
            self._schedule_resume(_LIVE_RETRY_S)

    async def start_ladder(self, games: int, team: Optional[str]) -> None:
        self.set_team(team)
        self.run_target = max(1, min(int(games), 10000))
        self.run_done = 0
        self.run_active = True
        self.log(f"ladder run started — target {self.run_target} game(s), "
                 f"team {self.team_pin or '(Teambuilder/default)'}"
                 + (f", {self.lanes} games at once" if self.lanes > 1 else ""))
        if not self._lane_free():
            self.log("a battle is live — the run queues after it ends." if self.lanes <= 1 else
                     f"lanes full ({len(self._live)}/{self.lanes}) — the run fills the next free lane.")
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

    def set_timer_immediate(self, on: bool) -> None:
        """USER 2026-09-02: battle-timer mode. ON = /timer on at the first frame of every game (team
        preview included); OFF = the consumer's per-room grace (/timer on once our decision sat
        unanswered ``_OPP_TIMER_S``). Applies from the next battle frame — no restart."""
        _pvhb.TIMER_IMMEDIATE = bool(on)
        self.log("battle timer: " + ("IMMEDIATE — /timer on at every game's first frame" if on else
                                     f"GRACE — /timer on after {_pvhb._OPP_TIMER_S:.0f}s of opponent "
                                     f"silence, per room"))

    def set_bandit_pin(self, name: Optional[str]) -> Optional[str]:
        """Serve mode from the panel (2026-09-02): pin one arm (FROZEN — it plays every game) or
        unpin (explore). Takes effect at the NEXT battle: applied right away on the bot loop when
        no battle is live, else deferred to the pre-search apply (a model swap mid-game would
        split one game across two arms)."""
        if self.bandit is None:
            raise ValueError("serve bandit is OFF (VD_BANDIT=0 or no config) — the deployed .env "
                             "stack already plays every game")
        before = self.bandit.pinned
        pinned = self.bandit.pin(name)
        if pinned == before:
            return pinned
        if pinned:
            self.log(f"bandit: PINNED → {pinned} (frozen — every game until unpinned)")
        else:
            self.log(f"bandit: UNPINNED (was {before}) — exploring again")
        if self._live and getattr(self.host.player, "_arm_resolver", None) is None:
            self.log("bandit: a battle is live — the pin applies from the next game")
        elif self.loop is not None:
            try:                                   # swap models on the bot loop, never from the HTTP thread
                self.loop.call_soon_threadsafe(self.apply_next_arm, "pin")
            except Exception:
                pass
        return pinned

    def set_lanes(self, n) -> int:
        """USER 2026-09-02: rated games at once (1..5 — Showdown's per-account cap). Takes effect
        at the next search; an active run fills newly opened lanes on its next tick."""
        n = max(1, min(int(n or 1), _MAX_LANES))
        if n != self.lanes:
            self.lanes = n
            self.log(f"lanes → {n} rated game(s) at once")
            if self.run_active:
                self._schedule_resume(_LANE_FILL_S)
        return self.lanes

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
    # ── 2026-09-02 (USER): matchup tables + the thought feed ────────────────
    def on_game_done(self, tag: str, battle, row: Optional[dict]) -> None:
        """GAME_DONE_HOOK tap: a finished battle + its bench row → the matchup books (session +
        all-time for THIS team) and the thought feed's finished mark. Never raises."""
        row = row or {}
        if not self.session_id and row.get("session_id"):
            self.session_id = str(row["session_id"])
        try:
            if self.matchups is not None and self.matchups.record_battle(
                    battle, row, session_id=self.session_id):
                seen = [display_species(getattr(m, "species", "?") or "?", self.belief)
                        for m in (getattr(battle, "opponent_team", None) or {}).values()]
                res = {"ai": "WON", "human": "LOST"}.get(row.get("result"), "DRAW")
                self.log(f"matchups: {res} vs {', '.join(seen) or '?'} "
                         f"(team {row.get('ai_team') or '?'}) — tables updated")
        except Exception:
            pass
        try:
            if self.thoughts is not None:
                self.thoughts.mark_finished(_base_tag(tag))
        except Exception:
            pass

    def set_matchup_team(self, team) -> str:
        """Regulation-table team: "" = follow the ladder-run team, "*" = all teams, else a team
        name (any team with recorded games counts, not only the current pool)."""
        team = (team or "").strip()
        if team and team != ALL_TEAMS and self.matchups is not None:
            known = {d["team"] for d in self.matchups.teams(self.fmt)} | set(self.ai_pool)
            if team not in known:
                raise ValueError(f"unknown team {team!r}")
        if team != self.matchup_team:
            self.matchup_team = team
            self.log("matchup table → " + ("follows the ladder-run team" if not team else
                                            "all teams" if team == ALL_TEAMS else team))
        return self.matchup_team

    def matchup_status(self) -> Optional[dict]:
        """Both tables for the panel: SESSION (this process, every team it played) and ALL-TIME
        for the selected team (the ladder-run pin by default) — always for the active format."""
        if self.matchups is None:
            return None
        sel = self.matchup_team
        team = None if sel == ALL_TEAMS else (sel or self.team_pin or None)
        mode = "all" if team is None else ("select" if sel else "pin")
        try:
            return {"format": self.fmt, "reg": _reg_label(self.fmt), "team": team or "",
                    "mode": mode, "selected": sel, "teams": self.matchups.teams(self.fmt),
                    "session": self.matchups.summary(self.fmt, session_id=self.session_id or "",
                                                     belief=self.belief),
                    "alltime": self.matchups.summary(self.fmt, team=(team or ALL_TEAMS),
                                                     belief=self.belief),
                    "loaded": {"games": self.matchups.loaded_games,
                               "files": self.matchups.loaded_files,
                               "live": self.matchups.live_games}}
        except Exception as exc:
            return {"format": self.fmt, "error": str(exc)[:120]}

    # ── status for the panel ──────────────────────────────────────────────────
    def status(self) -> dict:
        return {
            "username": self.username, "format": self.fmt,
            "formats": (known_formats() or [self.fmt]),
            "teams": self.ai_pool, "team_pin": self.team_pin or "",
            "auto_accept": bool(_pvhb.AUTO_ACCEPT),
            "auto_close": bool(self.auto_close),
            # 2026-09-02 (USER): battle-timer mode (immediate vs per-room grace) + the grace length
            "timer_immediate": bool(_pvhb.TIMER_IMMEDIATE),
            "timer_grace_s": float(_pvhb._OPP_TIMER_S),
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
            # 2026-09-02: serve mode (frozen pin vs explore); bandit_on = the panel control is live
            "bandit_on": self.bandit is not None,
            "bandit_pin": (self.bandit.pinned if self.bandit is not None else None),
            # 2026-09-03 (USER): the arms PANEL — the allocation / retire parameters + the learning arms
            "bandit_rule": (self.bandit.rule() if self.bandit is not None else None),
            # 2026-09-02: lanes = rated games at once (server cap 5)
            "lanes": self.lanes, "max_lanes": _MAX_LANES,
            # 2026-09-02 W3b-0: the ladder trajectory recorder (None = off)
            "recorder": (self.recorder.summary() if getattr(self, "recorder", None) is not None else None),
            # 2026-09-02 (USER): matchup tables (session + all-time per team) and the thought feed
            "matchups": self.matchup_status(),
            "thoughts": (self.thoughts.summary(live_tags=self._live)
                         if self.thoughts is not None else None),
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
                    if "timer_immediate" in body:  # battle timer at the first frame vs the grace
                        ctrl.set_timer_immediate(bool(body.get("timer_immediate")))
                    if "bandit_pin" in body:       # "" / null = unpin (explore)
                        ctrl.set_bandit_pin(body.get("bandit_pin"))
                    if "lanes" in body:            # rated games at once (1..5)
                        ctrl.set_lanes(body.get("lanes"))
                    if "matchup_team" in body:     # "" = follow the pin, "*" = all teams
                        ctrl.set_matchup_team(body.get("matchup_team"))
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
                     bandit=None, lanes_default: int = 1,
                     session_id: Optional[str] = None, belief=None,
                     dossier_dir: Optional[Path] = None, matchups: bool = True,
                     thoughts: bool = True) -> BotController:
    """Build the controller + serve the panel on 127.0.0.1 (first free port in [port, port+9]).
    Also chains itself onto ``RATING_HOOK`` (the Δelo battle-done confirm) and
    ``RATING_CHANGE_HOOK`` (post-battle rating → display, peaks, bandit reward) — call AFTER the
    harness set them. ``bench_path`` seeds the all-time peak per regulation."""
    ctrl = BotController(page=page, host=host, tally=tally, ai_pool=ai_pool, fmt=fmt,
                         username=username, loop=loop, env_path=env_path,
                         team_pin_default=team_pin_default, log_line=log_line,
                         auto_close_default=auto_close_default, bench_path=bench_path,
                         bandit=bandit, lanes_default=lanes_default,
                         session_id=session_id, belief=belief, dossier_dir=dossier_dir,
                         matchups=matchups)
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
                pin = " [pinned]" if ctrl.bandit.pinned == name else ""
                line += f"  → arm {name}{pin}: {s.wins}W-{s.losses}L, mean Δ {s.mean_delta():+.1f} over {s.n}"
                if s.retired:
                    line += "  ⛔ RETIRED"
        ctrl.log(line)

    _pvhb.RATING_CHANGE_HOOK = _change_hook
    # 2026-09-02 (USER): finished-game tap → matchup tables + the thought feed's finished mark
    prev_done = getattr(_pvhb, "GAME_DONE_HOOK", None)

    def _done_hook(tag, battle, row):
        if prev_done is not None:
            prev_done(tag, battle, row)
        ctrl.on_game_done(tag, battle, row)

    _pvhb.GAME_DONE_HOOK = _done_hook
    if thoughts:                                   # "what the nets are thinking" (VD_THOUGHT_FEED=0 off)
        try:
            from v_dance.play.thought_feed import ThoughtFeed
            ctrl.thoughts = ThoughtFeed().install(host.player)
            ctrl.log(ctrl.thoughts.banner())
        except Exception as exc:
            ctrl.log(f"thought feed failed to start (non-fatal): {exc!r}")
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
  /* 2026-09-02 (USER): matchup tables + the "what the nets are thinking" feed */
  .mu { border-collapse:collapse; width:100%; font:8.5pt Verdana; }
  .mu th, .mu td { border:1px solid #c7d3dc; padding:3px 6px; text-align:left; white-space:nowrap; }
  .mu th { background:#e0e7ea; color:#36485a; }
  .mu td.n { text-align:right; }
  .mu td:nth-child(5), .mu td:nth-child(6) { white-space:normal; }
  .mu .seen { color:#1f6b3a; font-weight:bold; }
  .mu .belief { color:#6b7a88; font-style:italic; }
  /* 2026-09-03 (USER): the bandit arms panel — badges + row states */
  .mu.arms td:nth-child(5), .mu.arms td:nth-child(6) { white-space:nowrap; }
  .mu.arms td:nth-child(8) { white-space:normal; max-width:340px; color:#5c6c7a; }
  .mu.arms tr.retired td { color:#8b9aa8; }
  .mu.arms tr.retired td b { text-decoration:line-through; }
  .mu.arms tr.learning td { background:#e6f2fb; }
  .abs { display:flex; flex-wrap:wrap; gap:4px; margin-top:3px; }
  .ab { font:bold 7.5pt Verdana; padding:1px 7px; border-radius:9px; background:#e0e7ea; color:#36485a;
        border:1px solid #b5c3ce; white-space:nowrap; }
  .ab.play { background:#d8f3de; color:#1f6b3a; border-color:#7fc491; }
  .ab.inc { background:#fff1c2; color:#7a5a00; border-color:#d9b84a; }
  .ab.pin { background:#eedcf7; color:#5a2a7a; border-color:#b58ad0; }
  .ab.learn { background:#d9ecfa; color:#134a73; border-color:#7fb5e6; }
  .ab.ret { background:#f8d7d7; color:#8a2020; border-color:#d98a8a; }
  .muwrap { overflow-x:auto; }
  .twocol { display:grid; grid-template-columns:1fr 1fr; gap:14px; padding:12px; }
  #think { height:320px; overflow-y:auto; background:#eef2f5; border-top:1px solid #b5c3ce;
           padding:8px 10px; font: 8.5pt Consolas, monospace; white-space:pre-wrap; color:#33424f; }
  #think .bh { color:#2c5a8c; font-weight:bold; margin-top:8px; }
  #think .te { margin-top:4px; }
  #think .ts { color:#7a8a98; }
  #think .warn { color:#a0461f; }
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
    <input type="number" id="games" min="1" max="10000" value="10">
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
      <div class="toggle"><input type="checkbox" id="timerNow">
        <label for="timerNow">Start the battle timer immediately <span class="muted" id="timerNote">(off = after 30 s of opponent silence, per room)</span></label></div>
      <label class="fld" style="margin-top:8px">Serve mode</label>
      <select id="banditPin" disabled><option value="">Bandit — explore (per-game arm choice)</option></select>
      <div class="muted" id="banditPinNote"></div>
      <label class="fld" style="margin-top:8px">Games at once <span class="muted">(lanes — Showdown allows 5 per account)</span></label>
      <select id="lanes"><option value="1">1</option><option value="2">2</option><option value="3">3</option>
        <option value="4">4</option><option value="5">5</option></select>
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
      <div class="muted" id="recorder" style="margin-top:4px"></div>
    </div></div>
  </div>

  <div class="panel full"><h2>Bandit arms <span class="muted" id="armsHdr"></span></h2><div class="pad">
    <div class="muted">One row per serve arm (a checkpoint / serve-knob candidate the ladder judges); all-time
      evidence from artifacts/bandit/&lt;format&gt;.json. ◀ playing · ★ incumbent (the retire rule's reference,
      never retired by it) · 📌 pinned (frozen mode) · 🧠 learning = the adaptive τ arm whose games feed the
      nightly ladder-PPO update (retire-exempt) · ⛔ retired (hover for the reason). P(worse) = the retire
      rule's number.</div>
    <div class="muwrap" style="margin-top:6px"><table class="mu arms" id="arms"><thead><tr><th>arm</th><th>games</th>
      <th>W-L</th><th>win %</th><th>Δ Elo/g</th><th>P(worse)</th><th>τ</th><th>note</th></tr></thead><tbody></tbody></table></div>
    <div class="muted" id="armsNote" style="margin-top:6px"></div>
  </div></div>

  <div class="panel full"><h2>Matchups <span class="muted" id="muHdr"></span></h2>
    <div class="twocol">
      <div><b>Session</b> <span class="muted" id="muSHdr"></span>
        <div class="muwrap" style="margin-top:6px"><table class="mu" id="muS"><thead><tr><th>Pokémon</th>
          <th>games</th><th>W-L</th><th>win %</th><th>item</th><th>ability</th></tr></thead><tbody></tbody></table></div>
        <div class="muted" id="muSFoot" style="margin-top:4px"></div></div>
      <div><b>Regulation, all games</b> <span class="muted" id="muAHdr"></span>
        <select id="muTeam" style="margin-top:6px"></select>
        <div class="muwrap" style="margin-top:6px"><table class="mu" id="muA"><thead><tr><th>Pokémon</th>
          <th>games</th><th>W-L</th><th>win %</th><th>item</th><th>ability</th></tr></thead><tbody></tbody></table></div>
        <div class="muted" id="muAFoot" style="margin-top:4px"></div></div>
    </div></div>

  <div class="panel full"><h2>What the nets are thinking
      <span class="muted">(TP net · battle net · gimmick · value head — newest first)</span></h2>
    <div style="padding:8px 12px"><div class="row" style="margin-top:0;align-items:center">
      <select id="thinkTag" style="width:auto;min-width:260px"><option value="">all live battles</option></select>
      <span class="muted" id="thinkStat"></span></div></div>
    <div id="think"></div></div>

  <div class="panel full"><h2>Activity</h2><div id="log"></div></div>
</div>
<script>
const $ = (id) => document.getElementById(id);
const esc = (v) => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let filled = false, lastNote = "", lastS = null;

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
  // 2026-09-02: serve-mode pin (frozen = one arm every game) — options come from the arm list
  const pinSel = $('banditPin');
  if (arms.length && pinSel.options.length < arms.length + 1) {
    pinSel.innerHTML = '<option value="">Bandit — explore (per-game arm choice)</option>' +
      arms.map(a => `<option value="${a.name}">Frozen: ${a.name}${a.incumbent ? ' (incumbent)' : ''}</option>`).join('');
  }
  pinSel.disabled = !s.bandit_on;
  if (document.activeElement !== pinSel) pinSel.value = s.bandit_pin || '';
  $('banditPinNote').textContent = !s.bandit_on
      ? 'Bandit OFF (VD_BANDIT=0 / no config) — the deployed .env stack plays every game.'
      : (s.bandit_pin ? `FROZEN: ${s.bandit_pin} plays every game from the next battle (still credited to its arm; bench rows carry pinned).`
                      : 'Exploring: warm-up, then Thompson picks the arm per game.');
  // 2026-09-03 (USER): the arms live in their own panel (renderArms); this line is the pointer
  $('bandit').textContent = arms.length
      ? `Bandit: ${arms.length} arm(s), ${arms.filter(a => !a.retired).length} active` +
        (arms.some(a => a.current) ? ` · playing ${arms.filter(a => a.current).map(a => a.name).join(', ')}` : '') +
        ' — see the Bandit arms panel below'
      : '';
  renderArms(s);
  $('challengeOut').textContent = s.challenge_out ? ('outgoing challenge: ' + s.challenge_out) : '';
  const rc = s.recorder;
  $('recorder').textContent = rc ? `RL recorder: ${rc.games} game(s) / ${rc.steps} decision(s) this session` +
      (rc.failed ? ` · ${rc.failed} failed` : '') : '';
  if (document.activeElement !== $('lanes')) $('lanes').value = String(s.lanes || 1);
  // 2026-09-02 (USER): battle-timer mode — immediate vs the per-room grace
  if (document.activeElement !== $('timerNow')) $('timerNow').checked = !!s.timer_immediate;
  $('timerNote').textContent = `(off = after ${Math.round(s.timer_grace_s || 30)} s of opponent silence, per room)`;
  const wait = s.awaiting_confirm.length ? ' — confirming result…' : '';
  const lanesTxt = (s.lanes || 1) > 1 ? ` · live ${s.live.length}/${s.lanes}` : '';
  $('prog').textContent = s.run.active
      ? `Run: ${s.run.done}/${s.run.target} played${lanesTxt}` + (s.searching ? ' — searching…' : wait)
      : (s.searching ? 'Searching…' : '');
  $('startBtn').disabled = s.run.active;
  if (s.env_fmt_saved && !lastNote)
    $('fmtNote').textContent = s.env_fmt_saved + ' saved — restart the bot to play it.';
  // 2026-09-02 (USER): matchup tables (session / regulation per team) + the thought feed
  lastS = s;
  renderMatchups(s.matchups, s.team_pin || '');
  renderThoughts(s.thoughts);
  const log = $('log'), atEnd = log.scrollTop + log.clientHeight >= log.scrollHeight - 8;
  log.textContent = s.events.join('\\n');
  if (atEnd) log.scrollTop = log.scrollHeight;
}

function muRows(tbody, rows) {
  if (!tbody) return;
  if (!rows || !rows.length) { tbody.innerHTML = '<tr><td colspan="6" class="muted">no games yet</td></tr>'; return; }
  const dist = (list) => (list && list.length) ? list.map(e =>
      `<span class="${e.seen ? 'seen' : 'belief'}">${e.name === '?' ? 'unknown' : esc(e.name)}` +
      `${e.mega ? ' (mega)' : ''} ${e.p}%${e.seen ? ` (${e.seen} seen)` : ''}</span>`).join(' · ') : '—';
  const legacy = (name, src, n, pct) => src === 'seen' ? `<span class="seen">${esc(name)} ×${n} seen</span>`
               : src === 'belief' ? `<span class="belief">${esc(name)}${pct != null ? ' ' + pct + '%' : ''} (belief)</span>` : '—';
  tbody.innerHTML = rows.map(r => {
    const item = r.items ? dist(r.items) : legacy(r.item, r.item_src, r.item_n, r.item_pct);
    const ab = r.abilities ? dist(r.abilities) : legacy(r.ability, r.ability_src, r.ability_n, null);
    return `<tr><td>${esc(r.display || r.species)}</td><td class="n">${r.games}</td>` +
           `<td class="n">${r.wins}-${r.losses}${r.draws ? '-' + r.draws + 'D' : ''}</td>` +
           `<td class="n">${r.win_pct == null ? '—' : r.win_pct + '%'}</td><td>${item}</td><td>${ab}</td></tr>`;
  }).join('');
}
/* 2026-09-03 (USER): the bandit ARMS PANEL — win rate, retired / active, the LEARNING label */
function armBadges(a) {
  const b = [];
  if (a.current) b.push('<span class="ab play" title="bound to a live game / the next game">◀ playing</span>');
  if (a.incumbent) b.push('<span class="ab inc" title="the incumbent: the retire rule reference; never retired by the rule">★ incumbent</span>');
  if (a.pinned) b.push('<span class="ab pin" title="frozen mode: this arm plays every game until unpinned">📌 pinned</span>');
  if (a.learning) b.push('<span class="ab learn" title="the adaptive arm: its games feed the nightly ladder-PPO update; exempt from the retire rule">🧠 learning</span>');
  b.push(a.retired ? `<span class="ab ret" title="${esc(a.reason || '')}">⛔ retired</span>`
                   : '<span class="ab" title="in the Thompson rotation">active</span>');
  if (a.in_flight) b.push(`<span class="ab" title="games bound but not yet rewarded">${a.in_flight} in flight</span>`);
  return b.join(' ');
}
function renderArms(s) {
  const tb = $('arms'); if (!tb) return;
  const arms = s.bandit || [], rule = s.bandit_rule || {};
  const hdr = $('armsHdr'), note = $('armsNote'), body = tb.querySelector('tbody');
  if (!s.bandit_on) {
    hdr.textContent = '(bandit OFF — VD_BANDIT=0 / no config: the .env stack plays every game)';
    body.innerHTML = ''; note.textContent = ''; return;
  }
  hdr.textContent = `· ${arms.length} arm(s) · ${arms.filter(a => !a.retired).length} active` +
      (s.bandit_pin ? ` · FROZEN on ${esc(s.bandit_pin)}` : ' · exploring');
  body.innerHTML = arms.map(a => {
    const d = (a.wins || 0) + (a.losses || 0), wr = d ? (100 * a.wins / d) : null;
    const cls = a.retired ? 'retired' : (a.learning ? 'learning' : '');
    return `<tr class="${cls}"><td><b>${esc(a.name)}</b><div class="abs">${armBadges(a)}</div></td>` +
      `<td class="n">${a.n}</td><td class="n">${a.wins}-${a.losses}${a.draws ? '-' + a.draws + 'D' : ''}</td>` +
      `<td class="n">${wr == null ? '—' : wr.toFixed(1) + '%'}</td>` +
      `<td class="n">${a.mean_delta >= 0 ? '+' : ''}${a.mean_delta}</td>` +
      `<td class="n">${a.p_worse == null ? '—' : (100 * a.p_worse).toFixed(0) + '%'}</td>` +
      `<td class="n">${a.tau}</td><td title="${esc(a.note || '')}">${esc(a.note || '')}</td></tr>`;
  }).join('');
  note.textContent = rule.retire_min_games
    ? `Warm-up ${rule.min_games} game(s) per arm, then Thompson sampling over the mean Δ Elo/game. ` +
      `Retire at ≥ ${rule.retire_min_games} games when P(worse than the incumbent ${rule.incumbent}) ≥ ` +
      `${(100 * rule.retire_prob).toFixed(0)}%` +
      (rule.retire_margin_elo ? ` (by ≥ ${rule.retire_margin_elo} Elo/game)` : '') +
      ((rule.learning || []).length ? ` · learning arm(s) exempt: ${rule.learning.join(', ')}` : ' · no learning arm') +
      ' · the incumbent is never retired by the rule; promotion is a human call at ≥ 200 games.'
    : '';
}
let muKey = '', thinkKey = '';
function renderMatchups(m, pin) {
  if (!$('muS')) return;
  if (!m) { $('muHdr').textContent = '(off — VD_MATCHUPS=0)'; return; }
  if (m.error) { $('muHdr').textContent = 'error: ' + m.error; return; }
  $('muHdr').textContent = `· ${m.reg} · opponent Pokémon SEEN in battle · item / ability = revealed sightings (green) blended with the belief for the rest`;
  $('muSHdr').textContent = 'this bot session';
  muRows($('muS').querySelector('tbody'), (m.session || {}).rows);
  const sf = (m.session || {}).footer || {};
  $('muSFoot').textContent = `${sf.games || 0} game(s) · ${sf.opponents || 0} opponent(s) · ${sf.species || 0} species`;
  $('muAHdr').textContent = m.mode === 'all' ? 'all teams' : `team ${m.team}`;
  const sel = $('muTeam'), key = (m.teams || []).map(t => t.team + ':' + t.games).join('|') + '#' + pin;
  if (key !== muKey) {
    muKey = key;
    sel.innerHTML = `<option value="">— follow the ladder-run team (${esc(pin || 'none → all teams')}) —</option>` +
      '<option value="*">All teams</option>' +
      (m.teams || []).map(t => `<option value="${esc(t.team)}">${esc(t.team)} (${t.games} games)</option>`).join('');
  }
  if (document.activeElement !== sel) sel.value = m.selected || '';
  muRows($('muA').querySelector('tbody'), (m.alltime || {}).rows);
  const af = (m.alltime || {}).footer || {};
  $('muAFoot').textContent = `${af.games || 0} game(s) · ${af.opponents || 0} opponent(s) · ${af.species || 0} species` +
      (m.mode === 'all' && af.teams && af.teams.length ? ` · teams: ${af.teams.join(', ')}` : '') +
      (m.loaded ? ` · seeded from ${m.loaded.games} dossier game(s), +${m.loaded.live} live` : '');
}
function renderThoughts(t) {
  const box = $('think'); if (!box) return;
  if (!t) { box.textContent = 'thought feed off (VD_THOUGHT_FEED=0) or not started'; return; }
  const sel = $('thinkTag'), battles = t.battles || [];
  const key = battles.map(b => b.tag + (b.live ? 'L' : b.finished ? 'F' : '')).join('|');
  if (key !== thinkKey) {
    thinkKey = key;
    const cur = sel.value;
    sel.innerHTML = '<option value="">all live battles</option>' + battles.map(b =>
      `<option value="${esc(b.tag)}">…${esc(b.tag.split('-').pop())} · ${b.live ? 'LIVE' : b.finished ? 'done' : 'idle'}` +
      ` · ${esc(b.arm || '?')} · vs ${esc(b.opponent || '?')}</option>`).join('');
    if ([...sel.options].some(o => o.value === cur)) sel.value = cur;
  }
  $('thinkStat').textContent = `${t.total || 0} decision(s) narrated` + (t.failed ? ` · ${t.failed} failed` : '');
  const want = sel.value, anyLive = battles.some(b => b.live);
  const shown = battles.filter(b => want ? b.tag === want : (anyLive ? b.live : true));
  const atTop = box.scrollTop <= 4;               // newest first: stay pinned to the top unless scrolled
  let html = '';
  for (const b of shown) {
    html += `<div class="bh">── ${esc(b.tag)} · ${b.live ? 'LIVE' : b.finished ? 'finished' : 'idle'}` +
            ` · arm ${esc(b.arm || '?')} · vs ${esc(b.opponent || '?')} · turn ${b.turn ?? '?'}</div>`;
    for (const e of (t.entries || []).filter(e => e.tag === b.tag))
      html += `<div class="te${e.kind === 'note' ? ' warn' : ''}"><span class="ts">${esc(e.ts)}</span> ` +
              `${esc(e.text).replace(/\\n/g, '\\n    ')}</div>`;
  }
  box.innerHTML = html || '<span class="muted">no decisions yet — the first team preview writes the first block</span>';
  if (atTop) box.scrollTop = 0;
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
$('timerNow').onchange = () => api('/api/options', {timer_immediate: $('timerNow').checked});
$('banditPin').onchange = () => api('/api/options', {bandit_pin: $('banditPin').value});
$('lanes').onchange = () => api('/api/options', {lanes: +$('lanes').value});
$('teamSel').onchange = () => api('/api/team', {team: $('teamSel').value});
$('fmtSel').onchange = () => api('/api/format', {format: $('fmtSel').value});
$('muTeam').onchange = () => api('/api/options', {matchup_team: $('muTeam').value});
$('thinkTag').onchange = () => renderThoughts(lastS && lastS.thoughts);

setInterval(refresh, 2000);
refresh();
</script></body></html>
"""
