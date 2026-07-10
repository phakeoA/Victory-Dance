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


def _sync_avatar_from_frame(payload: str, env_path: Path = _REPO / ".env") -> None:
    """Persist a browser-side avatar change to ``.env`` (USER request 2026-07-09): the server
    confirms every avatar change with ``|updateuser|USER|NAMED|AVATAR|{settings}``, so watching
    the incoming frames catches changes made through the client UI. Only NAMED (post-login)
    updates for OUR account are applied — the guest frames of the login dance never touch .env.
    Atomic write (temp + replace): .env holds credentials, a torn write is never acceptable."""
    try:
        if "|updateuser|" not in payload:
            return
        uid = "".join(c for c in (_ENV.get("PS_USERNAME") or "").lower() if c.isalnum())
        for line in payload.split("\n"):
            parts = line.split("|")
            if len(parts) < 5 or parts[1] != "updateuser":
                continue
            user, named, avatar = parts[2], parts[3], parts[4].strip()
            # rank chars (space/+/*/#/…) are non-alnum → the filter normalizes them away
            if named != "1" or not avatar or "".join(c for c in user.lower() if c.isalnum()) != uid:
                continue
            if avatar == (_ENV.get("PS_AVATAR") or "").strip():
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
                # B-L2 dossier capture (passive; never raises) — key data for online opponents.
                from v_dance.play.opponent_dossier import summary, update_from_battle
                if update_from_battle(b, row["result"],
                                      our_team=row["ai_team"], note=note):
                    print(f"[online] dossier: {summary(row['opponent'] or '')}")
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

    _pvhb.RATING_HOOK = _rating_row
    # The reused consumer/_credit print + fall back on the module's _AI_NAME; online, the AI IS
    # the .env account. (The local harness isn't running in this process — safe to repoint.)
    _pvhb._AI_NAME = username

    # Team-pick wrapper: remember the picked name (for the bench row), and make .env
    # VD_DEFAULT_TEAM the RANDOM-fallback — an explicitly OPENED Teambuilder team still wins.
    default_team = _ENV.get("VD_DEFAULT_TEAM")
    _orig_pick = _pvhb._pick_ai_team

    async def _pick(page, pool, pin):
        nm, src = await _orig_pick(page, pool, pin)
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

            def _on_frame(p) -> None:              # SockJS unwrap → consumer feed + avatar watch
                for msg in _sockjs_unwrap(p):
                    frame_q.put_nowait(msg)
                    _sync_avatar_from_frame(msg)

            page.on("websocket", lambda ws: ws.on("framereceived", _on_frame))
            await page.goto(args.client_url, wait_until="domcontentloaded")
            await page.wait_for_function(
                "() => window.app && app.socket && app.socket.readyState === 1", timeout=30000)
            if not await _login(page, username, password):
                raise SystemExit("[online] login did not complete — aborting.")
            if _ENV.get("PS_AVATAR"):
                await page.evaluate("(a) => app.socket.send('|/avatar ' + a)", _ENV["PS_AVATAR"])
            if not args.no_import:
                failed = await page.evaluate(_IMPORT_TEAMS_JS, [BATTLE_FORMAT, teams])
                if failed:
                    print(f"[online] WARNING: {len(failed)} team import(s) failed: {failed[:8]}")
                print(f"[online] {len(teams) - len(failed)} pool teams available in the Teambuilder")
            await _default_format(page, "online")

            print("\n" + "=" * 64)
            print(f"  ONLINE — {username} @ {args.client_url}   format {BATTLE_FORMAT}")
            print(f"  battle ckpt : {ckpt}")
            print(f"  TP ckpt     : {tp_ckpt}")
            print(f"  bench       : session {session_id} (note {args.bench_note!r}) -> {BENCH_LOG}")
            print("  YOU find the matches in this window (ladder Battle! / send a challenge).")
            print(f"  Incoming challenges in {BATTLE_FORMAT} are AUTO-ACCEPTED.")
            print(f"  AI team per battle = pinned --ai-team, else the team OPEN in the "
                  f"Teambuilder, else random. Ctrl-C here to stop.")
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
            try:
                await browser.close()
            except Exception:
                # window-close auto-stop: the browser (and its driver pipe) are already gone —
                # close() then raises "Connection closed while reading from the driver". Benign.
                pass
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
