"""play_vs_human_browser.py — "AI plays THROUGH the browser" (LOCAL human-vs-AI).

Opens a real Chromium with TWO tabs: YOURS (you pick a team and challenge the AI) and the AI's
(``VictoryDanceAI``, driven by the production gen141 + SBDA via ``BattleHost``). Both tabs get all your
champion teams auto-imported and are auto-logged-in. The AI tab accepts your challenges indefinitely; on
each it uses a RANDOM team from the list, plays the whole battle by shipping the model's ``/choose`` into
the tab, and keeps a running tally. Runs until you press Ctrl-C.

    python -m v_dance.play.play_vs_human_browser                 # play (headed, two tabs)
    python -m v_dance.play.play_vs_human_browser --self-test     # headless smoke vs a poke-env opponent

The decision core is REUSED unchanged (BattleHost). The browser is only a TRANSPORT: capture the AI tab's
websocket frames → feed the host (on POKE_LOOP) → send its /choose|/team back into the tab. The same
transport will point at play.pokemonshowdown.com for the online mode (see online_multitransport_plan.md).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import v_dance  # noqa: F401  (Selector policy for poke-env's POKE_LOOP, on its own thread)
from poke_env.concurrency import POKE_LOOP
from v_dance.play.browser.battle_host import BattleHost
from v_dance.play.run_local_battle import (
    BATTLE_FORMAT, SHOWDOWN_HOST, SHOWDOWN_PORT, discover_teams, load_team,
    resolve_team_path, start_showdown, stop_showdown,
)

_AI_NAME = "VictoryDanceAI"
URL = f"http://{SHOWDOWN_HOST}:{SHOWDOWN_PORT}"
# Per-battle inactivity watchdog (the consumer frees the AI if one battle runs absurdly long / the tab
# went silent). Module-level so the self-test budget can be sized ABOVE it (#16) from a single source.
_MAX_BATTLE_S = 600.0
_OPP_TIMER_S = 30.0    # opponent think-time before the consumer sends /timer on (USER, 2026-07-10)

# The local client redirects to https://localhost.psim.us then opens insecure ws://localhost:8000;
# fresh Chromium blocks that (mixed-content / private-network). These flags allow it (LOCAL dev only).
_CHROMIUM_ARGS = [
    "--disable-web-security",
    "--allow-running-insecure-content",
    "--disable-features=BlockInsecurePrivateNetworkRequests,PrivateNetworkAccessSendPreflights,PrivateNetworkAccessForWorkers",
    f"--unsafely-treat-insecure-origin-as-secure={URL},ws://{SHOWDOWN_HOST}:{SHOWDOWN_PORT}",
    "--window-size=1280,900",          # initial window size (with no_viewport the page then follows resizes)
]


def _use_proactor_loop() -> None:
    """Playwright launches the browser via an asyncio subprocess → needs the Windows PROACTOR loop, but
    importing v_dance set SELECTOR (poke-env's POKE_LOOP, on its own thread, is unaffected). Flip the
    main-thread policy so Playwright's loop can spawn Chromium; reach the host on POKE_LOOP via the bridge.
    NOTE: v_dance/__init__.py forces Selector precisely because Proactor has a Ctrl-C `_poll` race; that
    mitigation only protects POKE_LOOP. This main-thread Proactor loop is exposed to that race, but it is
    acceptable for a local interactive session (Ctrl-C is caught in main(); worst case is a cosmetic
    shutdown traceback). A long-lived ONLINE Proactor session would want a cleaner SIGINT-driven unwind."""
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


def _toid(s: str) -> str:
    return "".join(c for c in (s or "").lower() if c.isalnum())


def _load_pool() -> list[tuple[str, str]]:
    """Every champion team for the active format, as (name, paste). #14: resolve the FULL reg-scoped
    repo-relative path from discover_teams (resolve_team_path returns it verbatim when it exists),
    NOT just Path(p).name — a bare name goes through resolve_team_path's cross-reg rglob and would
    return the alphabetically-first match if a filename ever existed in two regulations (wrong-reg team)."""
    return [(Path(p).name, load_team(resolve_team_path(p)))
            for p in discover_teams(reg=BATTLE_FORMAT)]


# The local client opens on Random Battle; a challenge issued in that format is IGNORED by the AI (it
# only accepts BATTLE_FORMAT) — which is why "you couldn't challenge the AI". This JS points the main-menu
# at our Champions format and wraps challenge() so a user-initiated challenge defaults to it.
# ⚠ It must re-render BOTH the format AND the team button (and set curTeamFormat/curTeamIndex): swapping
# only the format button leaves the TEAM button stuck on "Random team" (the user then has to toggle the
# format off+on to free it). Mirror the client's own change (it reads format from the format button, picks
# a capacity-6 team of that format, re-renders both). All verified live (scratch/browser_format_change_probe.py).
_DEFAULT_FORMAT_JS = """(fmt) => {
    const home = app.rooms[''] || app.rooms['home'];
    if (!home) return 'no-home-room';
    try {
        const f = window.BattleFormats && window.BattleFormats[fmt];
        const teamFormat = f ? (f.teambuilderFormat || (f.isTeambuilderFormat ? fmt : false)) : fmt;
        let idx = -1;                                             // first capacity-6 team of this format
        for (let i = 0; i < Storage.teams.length; i++) {
            if (Storage.teams[i].format === teamFormat && Storage.teams[i].capacity === 6) { idx = i; break; }
        }
        home.curFormat = fmt;
        if (idx >= 0) { home.curTeamFormat = teamFormat; home.curTeamIndex = idx; }
        if (typeof home.renderFormats === 'function' && home.$el) {
            const fb = home.$el.find('button[name=format]');
            if (fb && fb.length) fb.replaceWith(home.renderFormats(fmt));   // also sets home.curFormat
            const tb = home.$el.find('button[name=team]');
            if (tb && tb.length) tb.replaceWith(home.renderTeams(fmt, idx >= 0 ? idx : undefined));
        }
        if (!home._vdChallengePatched && typeof home.challenge === 'function') {
            const _orig = home.challenge;
            home.challenge = function (name, format, team) { return _orig.call(this, name, format || fmt, team); };
            home._vdChallengePatched = true;
        }
    } catch (e) { return 'err: ' + e; }
    return home.curFormat;
}"""


async def _default_format(page, name: str) -> None:
    """Default a tab to the Champions format so the user's teams show + challenges go out in the format
    the AI accepts. Non-fatal: warn and continue if the client isn't in the expected shape."""
    try:
        await page.wait_for_function(
            "(fmt) => !!(window.BattleFormats && window.BattleFormats[fmt])",
            arg=BATTLE_FORMAT, timeout=15000)
        res = await page.evaluate(_DEFAULT_FORMAT_JS, BATTLE_FORMAT)
        if res != BATTLE_FORMAT:
            print(f"[play] WARNING: could not default the {name} tab to {BATTLE_FORMAT} (got {res!r}); "
                  f"pick the format manually in the challenge box.")
    except Exception as exc:
        print(f"[play] WARNING: format-default step failed on the {name} tab ({exc!r}); "
              f"pick the format manually in the challenge box.")


async def _setup_client(context, *, name: str, teams: list[tuple[str, str]], frame_q=None):
    """Open a client tab, inject all teams into its Teambuilder, log in as ``name``, return the page.
    ``frame_q`` (if given) receives every websocket frame the tab gets (the AI tab uses this)."""
    page = await context.new_page()
    if frame_q is not None:
        page.on("websocket", lambda ws: ws.on("framereceived", lambda p: frame_q.put_nowait(p)))
    await page.goto(URL, wait_until="domcontentloaded")
    await page.wait_for_function("() => window.app && app.socket && app.socket.readyState === 1", timeout=25000)
    # ⚠ Storage.importTeam(paste) returns an ARRAY of parsed set objects — NOT a {name,format,team}
    # wrapper. The old code tacked .name/.format onto that array and pushed it; saveTeams() then wrote an
    # EMPTY packed `team` string, so the Teambuilder showed the team NAME but no Pokémon. Build a real
    # wrapper whose `team` is the PACKED string (Storage.packTeam(sets)). Verified live in
    # scratch/browser_team_import_probe.py (pokemon_count_seen_by_client: 0 → 6).
    # ⚠ `capacity: 6` is REQUIRED: the team selector's auto-pick (MainMenuRoom.renderTeams) only selects a
    # team where `format === teamFormat && capacity === 6`. Without it the picker shows "Select a team"
    # (you can't choose a team) and a stale index renders as "Error: Corrupted team". Verified live in
    # scratch/browser_capacity_probe.py (no-capacity → "Select a team"; capacity:6 → the team + 6 mons).
    failed = await page.evaluate(
        "(args) => { const [fmt, pastes] = args; const failed = [];"
        "  for (const [nm, paste] of pastes) {"
        "    const sets = Storage.importTeam(paste);"
        "    if (!sets || !sets.length) { failed.push(nm); continue; }"
        "    Storage.teams.push({ name: nm, format: fmt, folder: '', capacity: 6, team: Storage.packTeam(sets) }); }"
        "  Storage.saveTeams && Storage.saveTeams(); return failed; }",
        [BATTLE_FORMAT, teams])
    if failed:                                          # a malformed paste would silently vanish from the picker
        print(f"[play] WARNING: {len(failed)} team(s) failed to import into the {name} tab: {failed[:8]}")
    await page.evaluate("(n) => app.socket.send('|/trn ' + n + ',0,')", name)
    await page.wait_for_function("() => window.app && app.user && app.user.get('named')", timeout=15000)
    await _default_format(page, name)
    return page


def _parse_challenge(payload: str):
    """Return (challenger_name, format) for an incoming challenge, or None. Locally a challenge arrives as
    ``|pm| FROM| TO|/challenge FORMAT|...`` (an empty FORMAT = the challenge was cancelled); some servers
    also send ``|updatechallenges|{json}``. Handle both."""
    for line in payload.split("\n"):
        parts = line.split("|")
        # skip pms WE sent (an outgoing challenge lists us as the sender) — the receiver's
        # /accept of its own challenge is a server error. Only matters online, where the AI
        # window can also be used to send challenges.
        if (len(parts) >= 5 and parts[1] == "pm" and parts[4].startswith("/challenge")
                and _toid(parts[2]) != _toid(_AI_NAME)):
            fmt = parts[4][len("/challenge"):].strip()
            if fmt:
                return parts[2].strip(), fmt
        if len(parts) >= 3 and parts[1] == "updatechallenges":
            try:
                cf = (json.loads(parts[2]) or {}).get("challengesFrom") or {}
            except Exception:
                cf = {}
            for user, fmt in cf.items():                 # prefer a challenge in OUR format …
                if fmt == BATTLE_FORMAT:
                    return user, fmt
            for user, fmt in cf.items():                 # … else the first (caller ignores other formats)
                return user, fmt
    return None


def _current_challengers(payload: str):
    """The set of user-ids currently challenging us, per the latest ``updatechallenges`` frame in
    ``payload`` (the AUTHORITATIVE list), or ``None`` if this payload carries no such frame. A
    challenger absent from this set has CANCELLED/withdrawn — the consumer uses it to clear a stale
    ``pending`` so the AI never tries to /accept a challenge that no longer exists."""
    found = None
    for line in payload.split("\n"):
        parts = line.split("|")
        if len(parts) >= 3 and parts[1] == "updatechallenges":
            try:
                cf = (json.loads(parts[2]) or {}).get("challengesFrom") or {}
            except Exception:
                cf = {}
            found = {_toid(u) for u in cf.keys()}
    return found


def _result_line(payload: str):
    """The terminal ``|win|<name>`` / ``|tie|`` line of a battle frame, or None. PREFIX-matched per line
    (not a substring scan of the whole frame) so battle-chat text like ``|win|`` can't false-trigger."""
    for line in payload.split("\n"):
        if line.startswith(("|win|", "|tie|")):
            return line
    return None


# Ladder-rating capture (2026-07-10): the server's post-battle ``|raw|<user>'s rating: N …``
# ladder updates arrive AFTER the consumer already recorded + tore the battle down (end_battle →
# the ended-tag guard then drops those frames before poke-env), so ``battle.rating`` can never be
# set on the browser transport. The consumer instead captures them TEXTUALLY and calls
# ``RATING_HOOK(battle_tag, username, rating)`` — the online harness points it at a crash-safe
# ``rating_update`` bench row that ``human_benchmark_report`` joins by battle_tag. None = off
# (local play: challenges are unrated, the lines never occur).
RATING_HOOK = None

# 2026-07-10 (control panel): gate for the consumer's incoming-challenge auto-accept. The online
# control UI (bot_control_ui) flips it at the user's request; default True preserves the local
# harness + prior online behaviour exactly. A declined challenge is SURFACED, never silent-dropped.
AUTO_ACCEPT = True

# 2026-09-01 (link reconnect — USER: "the server keeps disconnecting the bot", mid-battle): three
# hooks the ONLINE harness sets; all None/empty = the local harness is byte-identical.
#   LINK_TICK      — async callable run on every idle tick (~1 s) of the consumer; the online
#                    LinkWatch's dead-socket detector + auto-reconnect lives behind it.
#   ROOM_GONE_HOOK — callable(tag) when a battle room answers |noinit| (it no longer exists — the
#                    game was decided while the link was down); the control panel drops it as live.
#   REJOINING      — battle tags the host FORGOT for a reconnect rejoin: the server's replayed log
#                    makes poke-env re-emit the open-team-sheets answer, which is stale mid-battle,
#                    so it is not shipped; the tag leaves the set once its live |request| arrived.
LINK_TICK = None
ROOM_GONE_HOOK = None
REJOINING: set = set()

# 2026-09-01 (rating reflector): the server's rating line reads "NAME's rating: OLD &rarr;
# <strong>NEW</strong>". RATING_HOOK receives the PRE-battle number (poke-env's parse, which
# play_ladder's rows also record). RATING_CHANGE_HOOK(tag, user, old, new) additionally
# receives the POST-battle number — what the profile page shows — for the live display,
# the per-regulation peaks and the serve-side bandit's per-game reward (new − old).
RATING_CHANGE_HOOK = None
_RATING_CHANGE_RE = re.compile(r"'s rating: (\d+)\D+?(\d+)")


def _parse_rating_changes(payload: str) -> list:
    """[(username, old, new)] for every ``|raw|…'s rating: OLD &rarr; <strong>NEW</strong>…``
    line in the frame. A line with only one number (unexpected) is skipped."""
    out = []
    if "'s rating: " not in payload:
        return out
    for line in payload.split("\n"):
        parts = line.split("|")
        if len(parts) >= 3 and parts[1] == "raw" and "'s rating: " in parts[2]:
            user = parts[2].split("'s rating: ", 1)[0].strip()
            m = _RATING_CHANGE_RE.search(parts[2])
            if m:
                out.append((user, int(m.group(1)), int(m.group(2))))
    return out


def _parse_rating_lines(payload: str) -> list:
    """[(username, rating)] from a frame's ``|raw|…'s rating: NNNN…`` lines — the same parse
    poke-env's abstract_battle uses (split on "'s rating: ", int of the first 4 chars = the
    PRE-battle rating, matching what play_ladder's poke-env-native rows record)."""
    out = []
    if "'s rating: " not in payload:
        return out
    for line in payload.split("\n"):
        parts = line.split("|")
        if len(parts) >= 3 and parts[1] == "raw" and "'s rating: " in parts[2]:
            try:
                user, rest = parts[2].split("'s rating: ", 1)
                out.append((user.strip(), int(rest[:4])))
            except (ValueError, IndexError):
                pass
    return out


def _credit(host: BattleHost, tag: str, result_line: str, tally: dict) -> None:
    """Credit a finished battle. Prefer the host's AUTHORITATIVE outcome (poke-env ``battle.won``) —
    collision-proof regardless of the players' names — and fall back to the ``|win|<name>`` only if the
    battle object is unavailable (e.g. already evicted or a feed error)."""
    if result_line.startswith("|tie|"):
        tally["draw"] += 1
        return
    battle = host.player._battles.get((tag or "").lstrip(">"))
    won = getattr(battle, "won", None) if battle is not None else None
    if won is True:
        tally["ai"] += 1
    elif won is False:
        tally["you"] += 1
    else:                                              # battle object gone → fall back to the win name
        tally["ai" if _toid(result_line[5:]) == _toid(_AI_NAME) else "you"] += 1


async def _ai_selected_team(page) -> str | None:
    """The team name you have OPEN in the AI tab's Teambuilder (the one you clicked into to edit), or
    None if none is open. ``app.rooms.teambuilder.curTeam`` is null at the team list, and the opened
    team once you click in — so opening a team IS the 'use this team' gesture."""
    try:
        return await page.evaluate(
            "() => { const r = app.rooms && app.rooms['teambuilder'];"
            "  return (r && r.curTeam && r.curTeam.name) ? r.curTeam.name : null; }")
    except Exception:
        return None


async def _pick_ai_team(page, ai_pool: list[str], pin: str | None,
                        route_for: str | None = None) -> tuple[str, str]:
    """Choose the AI's team for the next battle, returning (name, source): a PINNED ``--ai-team`` →
    the Phase-4b router vs a KNOWN opponent (``route_for``, flag-gated VD_ROUTE_TEAMS=1, default
    OFF = byte-identical) → the team you have OPEN in the AI Teambuilder → a random team."""
    if pin:
        return pin, "pinned"
    if route_for:
        from v_dance.play import team_router
        if team_router.ROUTE_TEAMS:
            routed, why = team_router.route(route_for, ai_pool)
            if routed:
                print(f"[ai] team router: {why}")
                return routed, "routed"
    picked = await _ai_selected_team(page)
    if picked and picked in ai_pool:
        return picked, "your pick"
    return random.choice(ai_pool), "random"


async def _ai_consumer(page, host: BattleHost, frame_q: asyncio.Queue,
                       ai_pool: list[str], tally: dict, stop: asyncio.Event,
                       ai_team_pin: str | None = None) -> None:
    """Indefinite loop: accept incoming challenges (random AI team each), drive every battle via the host,
    keep the tally. One battle at a time (the host serves sequentially)."""
    loop = asyncio.get_running_loop()
    busy = False                 # serving a battle right now (one at a time)
    busy_since = 0.0             # loop.time() when we accepted — for the watchdog
    hb_next = 0.0                # loop.time() of the next "still serving" heartbeat (stall visibility)
    active_tag = None            # the battle room we're currently serving (None until its first frame)
    pending = None               # a challenger (our format) seen while busy — accepted once free
    seen_wrong_fmt: set = set()  # (challenger, fmt) pairs already warned about (dedup the repeated frames)
    errors = 0                   # consecutive frame-handling errors → stop if the tab is dead
    last_ship = 0.0              # loop.time() of our last shipped /choose|/team — from then until our
                                 # NEXT ship, the opponent (or server) is the one we're waiting on
    timer_sent: set = set()      # battles we already sent /timer on for (once per battle)
    t0_tag = None                # ladder battles ARRIVE without an accept (the Battle! queue), so
                                 # busy_since was never set → stamp it on a battle's FIRST frame
    # _MAX_BATTLE_S is module-level (#16) so run()'s self-test budget can be sized above it.
    while not stop.is_set():
        # watchdog: free up if a battle ran absurdly long / went silent (tab closed, desync)
        if busy and (loop.time() - busy_since) > _MAX_BATTLE_S:
            print(f"[ai] watchdog: battle exceeded {_MAX_BATTLE_S:.0f}s — freeing up to accept again")
            if active_tag:
                host.end_battle(active_tag)
            busy, active_tag = False, None
        # accept a pending challenge as soon as we're free (checked every loop, incl. idle ticks, so a
        # challenge issued mid-battle is honoured once the current one ends — not silently dropped).
        if not busy and pending is not None:
            challenger, pending = pending, None
            try:
                busy, busy_since, active_tag = True, loop.time(), None
                hb_next = busy_since + 15.0                           # first heartbeat 15s after accept
                ai_name, src = await _pick_ai_team(page, ai_pool, ai_team_pin,
                                                   route_for=challenger)
                # audit: resolve the AI team WITHIN the active reg. A bare name through resolve_team_path
                # cross-reg rglobs and returns the alphabetically-first match (M-A sorts before M-B), so a
                # same-named paste in two reg folders would load the wrong-reg (possibly illegal) team. Map
                # the name back to its BATTLE_FORMAT-scoped path first; fall back to the bare name.
                _scoped = next((p for p in discover_teams(reg=BATTLE_FORMAT)
                                if Path(p).name == ai_name), ai_name)
                ai_team = load_team(resolve_team_path(_scoped))
                host.player.update_team(ai_team)                     # decision core uses this team
                host.player._team_name = ai_name                     # bench-row ai_team stamp
                packed = host.player._team.yield_team()
                await page.evaluate("(t) => app.socket.send('|/utm ' + t)", packed)
                await page.evaluate("(u) => app.socket.send('|/accept ' + u)", _toid(challenger))
                print(f"[ai] accepted challenge from {challenger} (AI team: {ai_name} [{src}])")
            except Exception as exc:                                 # accept failed → don't latch busy
                busy, active_tag = False, None
                print(f"[ai] accept failed (continuing): {exc!r}")
                if page.is_closed():
                    print("[ai] AI tab is closed — stopping consumer."); stop.set(); return
        try:
            payload = await asyncio.wait_for(frame_q.get(), timeout=1.0)
        except asyncio.TimeoutError:
            # Window closed → auto-Ctrl-C (USER request 2026-07-09): a closed tab sends no frames,
            # so this timeout branch is guaranteed to run within ~1s of the close — even mid-battle.
            # Without this, an IDLE consumer would spin on timeouts forever (is_closed was only
            # checked on exceptions), leaving the process running headless after the window died.
            if page.is_closed():
                print("[ai] browser window closed — stopping (auto Ctrl-C).")
                stop.set()
                return
            # Opponent-stall timer (USER request 2026-07-10): our decision shipped >30s ago and the
            # battle hasn't advanced (no frames since → we're on idle ticks) → /timer on, once per
            # battle. A rare false positive (e.g. an extremely long turn resolution) is harmless —
            # the timer is a legitimate tool either way.
            if (active_tag and last_ship and active_tag not in timer_sent
                    and active_tag not in host._ended
                    and loop.time() - last_ship > _OPP_TIMER_S):
                timer_sent.add(active_tag)
                try:
                    await page.evaluate("(d) => app.socket.send(d.r + '|' + d.m)",
                                        {"r": active_tag, "m": "/timer on"})
                    print(f"[ai] opponent slow (>{_OPP_TIMER_S:.0f}s) — /timer on ({active_tag})")
                except Exception as exc:
                    print(f"[ai] timer-on failed (non-fatal): {exc!r}")
            # Heartbeat while serving: if we accepted a battle but no frames are arriving, surface the
            # stall (battle never started / frames not captured / desync) instead of failing silently.
            if busy and loop.time() >= hb_next:
                print(f"[ai] …still serving battle (tag={active_tag}); "
                      f"{loop.time() - busy_since:.0f}s since accept, no frames arriving")
                hb_next = loop.time() + 15.0
            # 2026-09-01: online dead-socket detection + auto-reconnect rides the idle tick — a
            # dead link produces exactly this branch (no frames), so it is the right clock.
            if LINK_TICK is not None:
                try:
                    await LINK_TICK()
                except Exception as exc:                             # the watchdog must never kill play
                    print(f"[ai] link watchdog error (non-fatal): {exc!r}")
            continue
        try:
            # 1) note any incoming challenge in OUR format (remember it even while busy)
            if not payload.startswith(">battle"):
                ch = _parse_challenge(payload)
                if ch and ch[1] == BATTLE_FORMAT:
                    if AUTO_ACCEPT:
                        pending = ch[0]
                    else:                                            # control-panel toggle (2026-07-10):
                        print(f"[ai] auto-accept is OFF — challenge from {ch[0]} NOT accepted "
                              f"(toggle it in the control panel or accept manually in the window).")
                elif ch and ch not in seen_wrong_fmt:                # surface (don't silently drop) a
                    seen_wrong_fmt.add(ch)                            # wrong-format challenge — it's why a
                    print(f"[ai] ignoring challenge from {ch[0]} in format {ch[1]!r} — "
                          f"challenge in {BATTLE_FORMAT} (the challenge box should default to it).")
                # audit: clear a stale `pending` if its challenger has CANCELLED. A cancellation sends an
                # authoritative updatechallenges frame with the challenger gone; without this, a busy AI
                # would later /accept a withdrawn challenge and then sit latched-busy until the ~600s
                # watchdog, skipping genuinely-new challenges in the meantime.
                cur = _current_challengers(payload)
                if cur is not None and pending is not None and _toid(pending) not in cur:
                    print(f"[ai] challenge from {pending} was cancelled — clearing pending.")
                    pending = None
            # 2) battle frame → host decides → ship its commands back into the tab
            elif payload.startswith(">battle"):
                active_tag = payload.split("\n", 1)[0].lstrip(">")
                # 2026-09-01: the room is GONE server-side — ``|noinit|nonexistent|`` answers a
                # reconnect rejoin after the game was decided while the link was down (or a stale
                # /join). poke-env raises on the event (or, for a forgotten tag, would block forever
                # in _get_battle), so reclaim the battle HERE and never feed the frame.
                _second = payload.split("\n", 2)[1] if "\n" in payload else ""
                if _second.startswith("|noinit|"):
                    why = _second[len("|noinit|"):].split("|", 1)[0] or "?"
                    print(f"[ai] room gone ({why}): {active_tag} — abandoning it")
                    abandon = getattr(host, "abandon_battle", None)
                    if callable(abandon) and active_tag not in host._ended:
                        abandon(active_tag)
                    REJOINING.discard(active_tag)
                    if ROOM_GONE_HOOK is not None:
                        try:
                            ROOM_GONE_HOOK(active_tag)
                        except Exception as exc:
                            print(f"[ai] room-gone hook failed (non-fatal): {exc!r}")
                    timer_sent.discard(active_tag)
                    busy, active_tag, last_ship = False, None, 0.0
                    errors = 0
                    continue
                # ladder-queue battles arrive with NO accept step → busy_since was never stamped
                # (the "battle done in 686352s" print) — stamp it on the battle's first frame.
                if not busy and active_tag != t0_tag and active_tag not in host._ended:
                    busy_since, t0_tag = loop.time(), active_tag
                # ladder-rating capture: these |raw| lines belong to an already-ENDED battle
                # (recorded + torn down), so scan the text BEFORE the host's ended-tag guard
                # drops the frame. No-op unless a harness installed RATING_HOOK.
                if RATING_HOOK is not None:
                    for _u, _rt in _parse_rating_lines(payload):
                        try:
                            RATING_HOOK(active_tag, _u, _rt)
                        except Exception as exc:                     # capture must never break play
                            print(f"[ai] rating-capture failed (non-fatal): {exc!r}")
                if RATING_CHANGE_HOOK is not None:                   # post-battle number (2026-09-01)
                    for _u, _old, _new in _parse_rating_changes(payload):
                        try:
                            RATING_CHANGE_HOOK(active_tag, _u, _old, _new)
                        except Exception as exc:
                            print(f"[ai] rating-change hook failed (non-fatal): {exc!r}")
                # M5 (DS-M5): OTS |showteam| capture → player._ots_sheets, keyed by the room's
                # base tag, so team preview can (flag-gated VD_TP_OTS_OVERLAY) feed the
                # opponent's revealed builds to the TP net. Closed play: no such frames.
                if "|showteam|" in payload:
                    try:
                        from v_dance.parser.vod_parser.team_sheet import parse_showteam_sides
                        from v_dance.play.player import room_base_tag
                        _sides = parse_showteam_sides(payload)
                        if _sides:
                            _store = host.player.__dict__.setdefault("_ots_sheets", {})
                            _store.setdefault(room_base_tag(active_tag), {}).update(_sides)
                            print(f"[ai] OTS sheets captured for {active_tag} "
                                  f"(sides: {sorted(_sides)})")
                    except Exception as exc:                         # capture must never break play
                        print(f"[ai] showteam capture failed (non-fatal): {exc!r}")
                # DS-4c stage 2: a Bo3 game room announces itself via |uhtml|bestof| —
                # register (parent set, game idx) so game-2/3 previews can see the
                # previous game (bo3_state; capture must never break play).
                if "|uhtml|bestof|" in payload:
                    from v_dance.play import bo3_state
                    bo3_state.note_bestof_frame(host.player, active_tag, payload)
                result = _result_line(payload)                       # prefix-matched terminal line, or None
                try:
                    decisions = await asyncio.wrap_future(
                        asyncio.run_coroutine_threadsafe(host.feed_async(payload), POKE_LOOP))
                    for r, msg in decisions:
                        # ship every ROOM-scoped decision: /choose, /team, /forfeit (the forced-switch
                        # backstop — dropping it would hang the battle) AND the OTS answer + timer.
                        # ⚠ the old "the client auto-handles the OTS command" assumption is FALSE on
                        # the official server (2026-07-10: /rejectopenteamsheets sat unanswered every
                        # game, so the closed-sheets production regime was never actually enforced).
                        if msg.startswith(("/choose", "/team", "/forfeit", "/timer",
                                           "/rejectopenteamsheets", "/acceptopenteamsheets")):
                            if r in REJOINING and msg.startswith(("/rejectopenteamsheets",
                                                                    "/acceptopenteamsheets")):
                                # reconnect rejoin: poke-env re-answers open team sheets from the
                                # replayed |init| — stale mid-battle, the server would only error.
                                print(f"[ai] (rejoin: stale {msg} not shipped for {r})")
                                continue
                            await page.evaluate("(d) => app.socket.send(d.r + '|' + d.m)", {"r": r, "m": msg})
                            if msg.startswith(("/choose", "/team")):
                                last_ship = loop.time()          # the opponent's think-clock starts now
                        elif r:
                            print(f"[ai] (battle command not shipped: {msg[:30]})")
                    if REJOINING and active_tag in REJOINING and "\n|request|" in payload:
                        REJOINING.discard(active_tag)            # live request received: rejoin complete
                finally:
                    # Battle ended → tally + free up + reclaim state, EVEN IF the feed/ship raised.
                    # Guard on host._ended: Showdown re-sends a room's full log (incl. |win|) on
                    # reconnect/rejoin, so a RE-DELIVERED terminal frame for an already-credited room must
                    # NOT double-count the tally or fall back to the collision-prone win-name (feed_async
                    # already drops the stray frame; mirror that here).
                    if result is not None and active_tag and active_tag not in host._ended:
                        # DS-4c stage 2: harvest the finished game into its set state
                        # BEFORE end_battle tears the battle object down (no-op off-set).
                        from v_dance.play import bo3_state
                        bo3_state.record_game_end(host.player, active_tag, result)
                        _credit(host, active_tag, result, tally)
                        host.end_battle(active_tag)
                        timer_sent.discard(active_tag)
                        busy, active_tag, last_ship = False, None, 0.0
                        print(f"[ai] battle done in {loop.time() - busy_since:.0f}s — "
                              f"tally: AI {tally['ai']} / you {tally['you']} / draws {tally['draw']}")
            errors = 0                                               # a clean iteration resets the failure streak
        except Exception as exc:                                     # a transport hiccup must not kill the loop
            errors += 1
            print(f"[ai] frame-handling error (continuing): {exc!r}")
            if page.is_closed() or errors >= 30:                     # tab dead / persistent failure → stop, don't spin
                print("[ai] AI tab closed or too many consecutive errors — stopping consumer.")
                stop.set()
                return


async def run(headed: bool, human_name: str, self_test: bool, ai_team_pin: str | None = None,
              self_test_battles: int = 1, ckpt=None, tp_ckpt=None,
              bench_note: str = "local-browser", adapt_rules: bool = False,
              use_dossier: bool = False) -> dict:
    from playwright.async_api import async_playwright

    teams = _load_pool()
    if len(teams) < 2:
        raise SystemExit(f"[play] need >=2 teams in the {BATTLE_FORMAT} pool (found {len(teams)}).")
    ai_pool = [n for n, _ in teams]
    if ai_team_pin and ai_team_pin not in ai_pool:
        raise SystemExit(f"[play] --ai-team {ai_team_pin!r} is not in the pool. Available: {ai_pool[:12]} …")
    # Serve refresh: benchmark ANY ckpt without touching the model_io defaults (None → defaults).
    _ck = {}
    if ckpt is not None:
        _ck["model_path"] = ckpt
    if tp_ckpt is not None:
        _ck["team_chooser_path"] = tp_ckpt
    # HTML-replay recording (docs/human_benchmark_design.md): same wiring as play_vs_human /
    # play_ladder, so USER-set games are reviewable + become the Phase-3 adaptation data.
    # ON for --self-test too — the self-test then proves the whole record path end-to-end.
    from v_dance.play.play_vs_human import BENCH_DIR, BENCH_LOG
    session_id = f"{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-{os.getpid()}"
    # Type_C training copy (2026-07-10, USER): every REAL game's replay also lands in the
    # corpus folder for later ingest — self-test games are junk data and stay out.
    _type_c = (None if self_test
               else Path(__file__).resolve().parents[2] / "data" / "vods" / "Type_C")
    host = BattleHost(team=teams[0][1], username=_AI_NAME, adapt_rules=adapt_rules,
                      use_dossier=use_dossier,
                      live_dir=BENCH_DIR / "live", save_replays=True,
                      replay_dir=BENCH_DIR / "replays" / session_id, replay_label="bench",
                      replay_copy_dir=_type_c,
                      **_ck)                                         # team replaced per battle
    if adapt_rules:
        print("[play] adapt-rules ON (B-L1: Wide-Guard streak → spread-move tilt)")
    if use_dossier:
        print("[play] dossier ON (S1 L2b: cross-game opp item/ability/move warm-start)")

    # Bench recording (docs/human_benchmark_design.md), same end_battle hook as the online
    # harness: one JSONL row per finished battle, appended+flushed (skipped for --self-test).
    if not self_test:
        _row_ckpt = str(ckpt or "model_io-default")
        _row_tp = str(tp_ckpt or "model_io-default")
        _orig_end, _n = host.end_battle, {"n": 0}

        def _rec_end_battle(tag: str) -> None:
            try:
                b = host.player._battles.get((tag or "").lstrip(">"))
                if b is not None and getattr(b, "finished", False):
                    _n["n"] += 1
                    won = getattr(b, "won", None)
                    BENCH_DIR.mkdir(parents=True, exist_ok=True)
                    with BENCH_LOG.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "session_id": session_id, "note": bench_note, "game_idx": _n["n"],
                            "battle_tag": getattr(b, "battle_tag", tag),
                            "ai_team": getattr(host.player, "_team_name", None),
                            "human_team": None,
                            "opponent": getattr(b, "opponent_username", None),
                            "result": "ai" if won else ("draw" if won is None else "human"),
                            "turns": getattr(b, "turn", None),
                            "ckpt": _row_ckpt, "tp_ckpt": _row_tp}) + "\n")
                    # B-L2 dossier capture (passive; never raises).
                    from v_dance.play.opponent_dossier import summary, update_from_battle
                    if update_from_battle(b, "ai" if won else ("draw" if won is None else "human"),
                                          our_team=getattr(host.player, "_team_name", None),
                                          note=bench_note):
                        print(f"[play] dossier: {summary(getattr(b, 'opponent_username', '') or '')}")
            except Exception as exc:                   # recording must never break play
                print(f"[play] bench-record failed (non-fatal): {exc!r}")
            _orig_end(tag)

        host.end_battle = _rec_end_battle  # type: ignore[method-assign]
        print(f"[play] bench recording ON — session {session_id} (note {bench_note!r}) -> {BENCH_LOG}\n"
              f"[play] replays -> {BENCH_DIR / 'replays' / session_id}")
    tally = {"ai": 0, "you": 0, "draw": 0}
    stop = asyncio.Event()

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=not headed, args=_CHROMIUM_ARGS)
        consumer = None            # created mid-try; the finally must not mask a setup
        try:                       # failure with an UnboundLocalError (bug found 2026-07-09)
            frame_q: asyncio.Queue = asyncio.Queue()
            # no_viewport=True → the page tracks the real OS window size and resizes with it (the default
            # fixed 1280x720 viewport stays static when you resize the window).
            ctx_ai = await browser.new_context(no_viewport=True)
            page_ai = await _setup_client(ctx_ai, name=_AI_NAME, teams=teams, frame_q=frame_q)
            print(f"[play] AI tab ready ({_AI_NAME}); {len(teams)} teams loaded")
            consumer = asyncio.ensure_future(
                _ai_consumer(page_ai, host, frame_q, ai_pool, tally, stop, ai_team_pin))

            if self_test:
                from v_dance.play.run_local_battle import make_player
                from v_dance.play.parallel_battles import close_players
                opp = make_player("BrowserOpp", teams[1][1], max_concurrent_battles=1)
                # Re-challenge regression: send_test_battles SEQUENTIAL challenges (each waits for the prior
                # battle to finish), exercising the consumer's accept→play→free→accept-again path that a
                # single-battle smoke never covered.
                ch = asyncio.run_coroutine_threadsafe(
                    opp.send_challenges(_AI_NAME, self_test_battles), POKE_LOOP)
                try:
                    # #16: budget must exceed the per-battle watchdog so the consumer's watchdog fires +
                    # logs FIRST (450*N alone is < 600 for the default N=1). Single-sourced off _MAX_BATTLE_S.
                    await asyncio.wait_for(_wait_total(tally, self_test_battles),
                                           timeout=max(_MAX_BATTLE_S + 100.0, 450.0 * self_test_battles))
                except asyncio.TimeoutError:
                    print("[play] self-test TIMEOUT")
                    if ch.done() and ch.exception():               # the CHALLENGE failed, not the AI
                        print(f"[play] (challenge itself errored: {ch.exception()!r})")
                stop.set()
                await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(close_players(opp), POKE_LOOP))
            else:
                ctx_you = await browser.new_context(no_viewport=True)
                await _setup_client(ctx_you, name=human_name, teams=teams)
                ai_src = (f"pinned to '{ai_team_pin}'" if ai_team_pin else
                          f"the team you OPEN in the {_AI_NAME} window's Teambuilder (random if none open)")
                print("\n" + "=" * 64)
                print(f"  In YOUR window: search '{_AI_NAME}' → click their name → Challenge.")
                print(f"  The challenge box is pre-set to {BATTLE_FORMAT} + your teams — just pick one.")
                print(f"  AI team for each battle = {ai_src}.")
                print("  The AI auto-accepts and plays; challenge again for each battle. Ctrl-C to stop.")
                print("=" * 64 + "\n")
                # Wait until Ctrl-C OR the consumer dies — so a crashed consumer ends the session with its
                # error instead of leaving the AI silently unresponsive.
                await asyncio.wait({consumer, asyncio.ensure_future(stop.wait())},
                                   return_when=asyncio.FIRST_COMPLETED)
                if consumer.done() and not consumer.cancelled() and consumer.exception():
                    print(f"[play] AI consumer crashed: {consumer.exception()!r}")
        finally:
            stop.set()
            if consumer is not None:
                consumer.cancel()
                try:
                    await consumer
                except BaseException:                              # CancelledError or a propagated error
                    pass
            await browser.close()
    return tally


async def _wait_total(tally: dict, n: int) -> None:
    while tally["ai"] + tally["you"] + tally["draw"] < n:
        await asyncio.sleep(0.5)


def main() -> None:
    ap = argparse.ArgumentParser(description="Play the production AI through the browser (local human-vs-AI).")
    ap.add_argument("--human-name", default="Challenger", help="your Showdown name in the play tab.")
    ap.add_argument("--ai-team", default=None,
                    help="pin the AI to this team NAME every battle (default: the team you OPEN in the AI "
                         "window's Teambuilder, or random if none is open).")
    ap.add_argument("--self-test", action="store_true", help="headless smoke vs a poke-env opponent (no human).")
    ap.add_argument("--self-test-battles", type=int, default=1,
                    help="number of SEQUENTIAL self-test battles (>=2 exercises the re-challenge path).")
    ap.add_argument("--headed", dest="headed", action="store_true", default=True, help="show the browser (default).")
    ap.add_argument("--headless", dest="headed", action="store_false", help="run headless.")
    ap.add_argument("--ckpt", default=None,
                    help="battle-net checkpoint to serve (default: the model_io production default).")
    ap.add_argument("--tp-ckpt", default=None,
                    help="team-preview (SBDA) checkpoint to serve (default: production default).")
    ap.add_argument("--bench-note", default="local-browser",
                    help="tag stamped on every benchmark row (e.g. adv_v7).")
    ap.add_argument("--adapt-rules", action="store_true",
                    help="B-L1 serve-time pattern tilt (Wide-Guard streak → spread bias). Default OFF.")
    ap.add_argument("--dossier", action="store_true",
                    help="S1 L2b: warm-start unknown opp item/ability/moves from the per-opponent "
                         "dossier (cross-game knowledge; in-battle evidence always wins). Default OFF.")
    args = ap.parse_args()

    for _v in (args.ckpt, args.tp_ckpt):
        if _v is not None and not Path(_v).is_file():
            raise SystemExit(f"[play] checkpoint not found: {_v}")

    _use_proactor_loop()
    headed = args.headed and not args.self_test
    proc = start_showdown()
    try:
        tally = asyncio.run(run(headed=headed, human_name=args.human_name, self_test=args.self_test,
                                ai_team_pin=args.ai_team, self_test_battles=args.self_test_battles,
                                ckpt=Path(args.ckpt) if args.ckpt else None,
                                tp_ckpt=Path(args.tp_ckpt) if args.tp_ckpt else None,
                                bench_note=args.bench_note, adapt_rules=args.adapt_rules,
                                use_dossier=args.dossier))
    except KeyboardInterrupt:
        print("\n[play] Ctrl-C — shutting down …")
        tally = None
    finally:
        stop_showdown(proc)
        # proc None = the port was already serving when we started (someone else's server —
        # possibly a STALE leaked one). Say so instead of claiming we stopped it (2026-07-09:
        # an unconditional "stopped." here masked a leaked-cluster reuse for a whole session).
        print("[play] Showdown server stopped." if proc is not None else
              "[play] ⚠ Showdown server was already running (not ours) — left running. "
              "If unexpected, check for a stale server: Get-Process node")
    if args.self_test:
        n = args.self_test_battles
        ok = tally is not None and (tally["ai"] + tally["you"] + tally["draw"]) >= n
        print(f"[self-test] VERDICT: {f'PASS — AI played {n} full battle(s) through the browser' if ok else 'FAIL'} "
              f"(tally {tally}, wanted {n})")
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
