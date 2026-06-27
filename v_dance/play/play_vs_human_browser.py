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
import random
import sys
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
        if len(parts) >= 5 and parts[1] == "pm" and parts[4].startswith("/challenge"):
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


def _result_line(payload: str):
    """The terminal ``|win|<name>`` / ``|tie|`` line of a battle frame, or None. PREFIX-matched per line
    (not a substring scan of the whole frame) so battle-chat text like ``|win|`` can't false-trigger."""
    for line in payload.split("\n"):
        if line.startswith(("|win|", "|tie|")):
            return line
    return None


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


async def _pick_ai_team(page, ai_pool: list[str], pin: str | None) -> tuple[str, str]:
    """Choose the AI's team for the next battle, returning (name, source): a PINNED ``--ai-team`` →
    the team you have OPEN in the AI Teambuilder → a random team from the list."""
    if pin:
        return pin, "pinned"
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
                ai_name, src = await _pick_ai_team(page, ai_pool, ai_team_pin)
                ai_team = load_team(resolve_team_path(ai_name))
                host.player.update_team(ai_team)                     # decision core uses this team
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
            # Heartbeat while serving: if we accepted a battle but no frames are arriving, surface the
            # stall (battle never started / frames not captured / desync) instead of failing silently.
            if busy and loop.time() >= hb_next:
                print(f"[ai] …still serving battle (tag={active_tag}); "
                      f"{loop.time() - busy_since:.0f}s since accept, no frames arriving")
                hb_next = loop.time() + 15.0
            continue
        try:
            # 1) note any incoming challenge in OUR format (remember it even while busy)
            if not payload.startswith(">battle"):
                ch = _parse_challenge(payload)
                if ch and ch[1] == BATTLE_FORMAT:
                    pending = ch[0]
                elif ch and ch not in seen_wrong_fmt:                # surface (don't silently drop) a
                    seen_wrong_fmt.add(ch)                            # wrong-format challenge — it's why a
                    print(f"[ai] ignoring challenge from {ch[0]} in format {ch[1]!r} — "
                          f"challenge in {BATTLE_FORMAT} (the challenge box should default to it).")
            # 2) battle frame → host decides → ship its commands back into the tab
            elif payload.startswith(">battle"):
                active_tag = payload.split("\n", 1)[0].lstrip(">")
                result = _result_line(payload)                       # prefix-matched terminal line, or None
                try:
                    decisions = await asyncio.wrap_future(
                        asyncio.run_coroutine_threadsafe(host.feed_async(payload), POKE_LOOP))
                    for r, msg in decisions:
                        # ship every ROOM-scoped decision: /choose, /team AND /forfeit (the forced-switch
                        # backstop — dropping it would hang the battle). The client auto-handles the
                        # roomless OTS/timer commands, so we don't ship those.
                        if msg.startswith(("/choose", "/team", "/forfeit")):
                            await page.evaluate("(d) => app.socket.send(d.r + '|' + d.m)", {"r": r, "m": msg})
                        elif r:
                            print(f"[ai] (battle command not shipped: {msg[:30]})")
                finally:
                    # Battle ended → tally + free up + reclaim state, EVEN IF the feed/ship raised.
                    if result is not None:
                        _credit(host, active_tag, result, tally)
                        host.end_battle(active_tag)
                        busy, active_tag = False, None
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
              self_test_battles: int = 1) -> dict:
    from playwright.async_api import async_playwright

    teams = _load_pool()
    if len(teams) < 2:
        raise SystemExit(f"[play] need >=2 teams in the {BATTLE_FORMAT} pool (found {len(teams)}).")
    ai_pool = [n for n, _ in teams]
    if ai_team_pin and ai_team_pin not in ai_pool:
        raise SystemExit(f"[play] --ai-team {ai_team_pin!r} is not in the pool. Available: {ai_pool[:12]} …")
    host = BattleHost(team=teams[0][1], username=_AI_NAME)           # team replaced per battle
    tally = {"ai": 0, "you": 0, "draw": 0}
    stop = asyncio.Event()

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=not headed, args=_CHROMIUM_ARGS)
        try:
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
            consumer.cancel()
            try:
                await consumer
            except BaseException:                                  # CancelledError or a propagated error
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
    args = ap.parse_args()

    _use_proactor_loop()
    headed = args.headed and not args.self_test
    proc = start_showdown()
    try:
        tally = asyncio.run(run(headed=headed, human_name=args.human_name, self_test=args.self_test,
                                ai_team_pin=args.ai_team, self_test_battles=args.self_test_battles))
    except KeyboardInterrupt:
        print("\n[play] Ctrl-C — shutting down …")
        tally = None
    finally:
        stop_showdown(proc)
        print("[play] Showdown server stopped.")
    if args.self_test:
        n = args.self_test_battles
        ok = tally is not None and (tally["ai"] + tally["you"] + tally["draw"]) >= n
        print(f"[self-test] VERDICT: {f'PASS — AI played {n} full battle(s) through the browser' if ok else 'FAIL'} "
              f"(tally {tally}, wanted {n})")
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
