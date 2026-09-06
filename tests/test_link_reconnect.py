"""Link reconnect (2026-09-01) — the online tab's dead-socket detector + auto-reconnect, and the
pieces around it: the consumer's |noinit| handling, the host's forget/abandon, and the panel's
run bookkeeping across a reconnect.

USER report: on long sessions the bot "keeps getting disconnected", MID-BATTLE (a slow recovery
is a timer loss). Root causes locked here: the legacy client only shows a popup (a click reconnects),
the consumer could not tell a dead socket from a slow opponent, ladder runs had no pending tick
while a battle was live, and a |noinit| frame either raised inside poke-env or (for a forgotten
tag) would block forever. Everything is driven with fakes — no Playwright, no server."""
from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("poke_env")

import v_dance.play.play_vs_human_browser as _pvhb
from v_dance.play import play_online_browser as pob
from v_dance.play.bot_control_ui import BotController

TAG = "battle-gen9championsvgc2026regmb-2673980258"


# ── fakes ──────────────────────────────────────────────────────────────────────
class FakePage:
    def __init__(self, state=1, fail_reloads=0, named=True, hang=False):
        self.state = state            # app.socket.readyState the page reports
        self.evals = []               # (js, arg) of every evaluate()
        self.reloads = 0
        self.gotos = []               # 2026-09-06: navigations (a crashed renderer needs goto, not reload)
        self.fail_reloads = fail_reloads
        self.named = named
        self.hang = hang              # 2026-09-06: a crashed renderer's evaluate never answers

    async def evaluate(self, js, arg=None):
        self.evals.append((js, arg))
        if self.hang:
            await asyncio.Event().wait()                  # never resolves — the 2026-09-06 "Aw, Snap!" symptom
        if "readyState" in js and "===" not in js:
            return self.state
        if "named" in js:                                 # the login check (_LINK_NAMED_JS)
            return self.named
        return None

    async def reload(self, **_kw):
        if self.fail_reloads > 0:
            self.fail_reloads -= 1
            raise RuntimeError("navigation failed")
        self.reloads += 1
        self.state = 1
        self.hang = False

    async def goto(self, url, **_kw):
        self.gotos.append(url)
        self.state = 1
        self.hang = False

    async def wait_for_function(self, js, arg=None, timeout=None):
        if "named" in js and not self.named:
            raise TimeoutError("not named")
        return True

    async def fill(self, *_a, **_k):
        pass

    async def click(self, *_a, **_k):
        pass

    def is_closed(self):
        return False


class FakeBattle:
    finished = False


class FakeHost:
    def __init__(self, live=()):
        self.battles = {t: FakeBattle() for t in live}
        self._ended = set()
        self.forgotten = []
        self.abandoned = []
        self.fed = []
        self.player = type("P", (), {"_battles": self.battles})()

    def forget_battle(self, t):
        self.forgotten.append(t)
        self.battles.pop(t, None)

    def abandon_battle(self, t):
        self.abandoned.append(t)
        self._ended.add(t)
        self.battles.pop(t, None)

    def end_battle(self, t):
        self._ended.add(t)

    async def feed_async(self, payload):
        self.fed.append(payload)
        return []


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class Ctrl:
    def __init__(self):
        self.reconnected = []
        self.gone = []

    def on_reconnected(self, live):
        self.reconnected.append(list(live))

    def room_gone(self, tag):
        self.gone.append(tag)


def _watch(page=None, host=None, clock=None, ctrl=None, **kw):
    clock = clock or Clock()
    page = page or FakePage()
    host = host or FakeHost()
    logs = []
    ref = {"c": ctrl}
    w = pob.LinkWatch(page=page, host=host, username="VictoriousDancing", password="pw", loop=None,
                      log=logs.append, now=clock, probe_s=15, dead_s=25, cooldown_s=20,
                      ctrl_ref=ref, **kw)
    return w, page, host, clock, logs


def _sends(page, needle):
    return [a for js, a in page.evals if needle in js]


@pytest.fixture(autouse=True)
def _clean_rejoining():
    _pvhb.REJOINING.clear()
    _pvhb.REJOIN_REFUSED.clear()
    yield
    _pvhb.REJOINING.clear()
    _pvhb.REJOIN_REFUSED.clear()


# ── LinkWatch: detection ──────────────────────────────────────────────────────
def test_socket_close_reconnects_on_the_next_tick():
    w, page, host, clock, logs = _watch()
    ws = object()
    w.on_ws_open(ws)
    w.on_ws_close(ws)
    asyncio.run(w.tick())
    assert page.reloads == 1
    assert any("LINK DOWN (socket closed)" in line for line in logs)
    assert any("LINK RECONNECTED" in line for line in logs)
    assert w.closed_at is None and w.status()["down"] is False


def test_stale_socket_close_is_ignored_after_a_new_socket_opened():
    w, page, *_ = _watch()
    old, new = object(), object()
    w.on_ws_open(old)
    w.on_ws_open(new)
    w.on_ws_close(old)                       # the OLD socket's close arriving late must be noise
    asyncio.run(w.tick())
    assert page.reloads == 0 and w.closed_at is None


def test_silence_probes_first_then_reconnects_when_the_probe_goes_unanswered():
    w, page, host, clock, logs = _watch()
    clock.t += 16                            # past probe_s, socket says open → probe, no reload
    asyncio.run(w.tick())
    assert page.reloads == 0
    assert len(_sends(page, "/cmd userdetails")) == 1
    clock.t += 5                             # 21 s: probed, still waiting → nothing new
    asyncio.run(w.tick())
    assert page.reloads == 0 and len(_sends(page, "/cmd userdetails")) == 1
    clock.t += 5                             # 26 s ≥ dead_s with the probe unanswered → half-open
    asyncio.run(w.tick())
    assert page.reloads == 1
    assert any("probe unanswered" in line for line in logs)


def test_a_frame_after_the_probe_resets_the_cycle():
    w, page, host, clock, _ = _watch()
    clock.t += 16
    asyncio.run(w.tick())                    # probe #1
    clock.t += 2
    w.on_raw_frame("h")                      # the reply / a heartbeat → link alive
    clock.t += 12                            # 12 s since that frame: below probe_s
    asyncio.run(w.tick())
    assert page.reloads == 0 and len(_sends(page, "/cmd userdetails")) == 1
    clock.t += 4                             # 16 s since the frame → a NEW probe, not a reload
    asyncio.run(w.tick())
    assert page.reloads == 0 and len(_sends(page, "/cmd userdetails")) == 2


def test_socket_not_open_at_probe_time_reconnects_without_probing():
    w, page, host, clock, logs = _watch(page=FakePage(state=3))
    clock.t += 16
    asyncio.run(w.tick())
    assert page.reloads == 1
    assert _sends(page, "/cmd userdetails") == []
    assert any("readyState=3" in line for line in logs)


def test_kill_switch_detects_and_logs_but_never_reconnects():
    w, page, host, clock, logs = _watch(reconnect=False)
    ws = object()
    w.on_ws_open(ws)
    w.on_ws_close(ws)
    asyncio.run(w.tick())
    asyncio.run(w.tick())
    assert page.reloads == 0
    assert sum("auto-reconnect is OFF" in line for line in logs) == 1   # warned once, not every tick
    assert w.status()["down"] is True


# ── LinkWatch: recovery ───────────────────────────────────────────────────────
def test_reconnect_forgets_live_battles_marks_them_rejoining_and_rejoins():
    ctrl = Ctrl()
    w, page, host, clock, logs = _watch(host=FakeHost(live=[TAG]), ctrl=ctrl)
    ws = object()
    w.on_ws_open(ws)
    w.on_ws_close(ws)
    asyncio.run(w.tick())
    assert host.forgotten == [TAG]                       # forgotten BEFORE the replay can land
    assert TAG in _pvhb.REJOINING                        # stale OTS answers suppressed until |request|
    assert _sends(page, "joinRoom") == [TAG]             # explicit rejoin (belt to the client's own)
    assert ctrl.reconnected == [[TAG]]                   # panel told, with the rejoining tag


def test_finished_or_ended_battles_are_not_rejoined():
    host = FakeHost(live=[TAG, "battle-gen9championsvgc2026regmb-1"])
    host.battles[TAG].finished = True
    host._ended.add("battle-gen9championsvgc2026regmb-1")
    w, page, *_ = _watch(host=host)
    ws = object()
    w.on_ws_open(ws)
    w.on_ws_close(ws)
    asyncio.run(w.tick())
    assert host.forgotten == [] and _sends(page, "joinRoom") == []


def test_cooldown_prevents_a_reload_loop():
    w, page, host, clock, _ = _watch()
    ws = object()
    w.on_ws_open(ws)
    w.on_ws_close(ws)
    asyncio.run(w.tick())
    assert page.reloads == 1
    w.on_ws_close(ws)                        # closes again right away (server still restarting)
    asyncio.run(w.tick())
    assert page.reloads == 1                 # inside the cooldown: wait
    clock.t += 21
    asyncio.run(w.tick())
    assert page.reloads == 2


def test_failed_reconnect_keeps_the_down_state_and_retries_after_cooldown():
    w, page, host, clock, logs = _watch(page=FakePage(fail_reloads=1))
    ws = object()
    w.on_ws_open(ws)
    w.on_ws_close(ws)
    asyncio.run(w.tick())
    assert page.reloads == 0 and w.closed_at is not None
    assert any("reconnect FAILED" in line for line in logs)
    clock.t += 21
    asyncio.run(w.tick())
    assert page.reloads == 1 and w.closed_at is None


def test_relogin_runs_when_the_reloaded_tab_is_not_named():
    w, page, host, clock, logs = _watch(page=FakePage(named=False))
    ws = object()
    w.on_ws_open(ws)
    w.on_ws_close(ws)
    asyncio.run(w.tick())
    assert page.reloads == 1
    assert any("app.user.rename" in js for js, _ in page.evals)       # scripted re-login attempted
    assert any("NOT logged in" in line or "re-login FAILED" in line for line in logs)


def test_banner_and_status_shape():
    w, *_ = _watch()
    assert "link watchdog ACTIVE" in w.banner() and "auto-reconnect ON" in w.banner()
    s = w.status()
    assert set(s) >= {"enabled", "down", "idle_s", "reconnects", "probes", "frames"}


# ── consumer: |noinit| room-gone handling ─────────────────────────────────────
def test_consumer_abandons_a_gone_room_without_feeding_pokeenv():
    gone_frame = f">{TAG}\n|noinit|nonexistent|The room \"{TAG}\" does not exist."
    gone = []

    async def main():
        q: asyncio.Queue = asyncio.Queue()
        q.put_nowait(gone_frame)
        stop = asyncio.Event()
        host = FakeHost(live=[TAG])
        _pvhb.ROOM_GONE_HOOK = gone.append

        async def stopper():
            await asyncio.sleep(1.4)
            stop.set()

        asyncio.create_task(stopper())
        await _pvhb._ai_consumer(FakePage(), host, q, ["a", "b"],
                                 {"ai": 0, "you": 0, "draw": 0}, stop)
        return host

    try:
        host = asyncio.run(main())
    finally:
        _pvhb.ROOM_GONE_HOOK = None
    assert host.abandoned == [TAG]           # reclaimed via abandon (no bench row), gated in _ended
    assert host.fed == []                    # the frame never reached poke-env
    assert gone == [TAG]                     # the panel hook fired


def test_consumer_runs_link_tick_on_idle_ticks():
    ticks = []

    async def tick():
        ticks.append(1)

    async def main():
        q: asyncio.Queue = asyncio.Queue()
        stop = asyncio.Event()
        _pvhb.LINK_TICK = tick

        async def stopper():
            await asyncio.sleep(2.3)
            stop.set()

        asyncio.create_task(stopper())
        await _pvhb._ai_consumer(FakePage(), FakeHost(), q, ["a", "b"],
                                 {"ai": 0, "you": 0, "draw": 0}, stop)

    try:
        asyncio.run(main())
    finally:
        _pvhb.LINK_TICK = None
    assert len(ticks) >= 2                   # ~1 s cadence while idle


# ── panel: run bookkeeping across a reconnect ─────────────────────────────────
class _PanelPage:
    def __init__(self):
        self.sent = []

    async def evaluate(self, js, arg=None):
        self.sent.append((js, arg))


class _PanelHost:
    def __init__(self):
        class _T:
            def yield_team(self):
                return "PACKED"

        self.player = type("P", (), {"_team": _T(), "_team_name": None,
                                     "update_team": lambda self, t: None})()


def _panel(loop):
    from pathlib import Path
    c = BotController(page=_PanelPage(), host=_PanelHost(), tally={"ai": 0, "you": 0, "draw": 0},
                      ai_pool=["alpha"], fmt="gen9championsvgc2026regmb", username="VictoriousDancing",
                      loop=loop, env_path=Path("unused.env"))
    c._load_scoped_team = lambda scoped: "LOADED"
    return c


def test_run_keeps_a_tick_pending_after_search_and_after_battle_start(monkeypatch):
    from v_dance.play import bot_control_ui as bcu
    monkeypatch.setattr(bcu, "discover_teams", lambda reg=None: [])

    async def main():
        c = _panel(asyncio.get_running_loop())
        await c.start_ladder(5, "alpha")
        assert c._resume_scheduled                      # after /search: a tick is pending
        c._resume_scheduled = False
        c._battle_seen(TAG)                             # the game was found
        assert c._resume_scheduled                      # still one pending while it is live
        return c

    asyncio.run(main())


def test_room_gone_drops_live_counts_the_game_and_resumes(monkeypatch):
    from v_dance.play import bot_control_ui as bcu
    monkeypatch.setattr(bcu, "discover_teams", lambda reg=None: [])

    async def main():
        c = _panel(asyncio.get_running_loop())
        await c.start_ladder(3, "alpha")
        c._battle_seen(TAG)
        c._resume_scheduled = False
        c.room_gone(TAG)
        assert not c._live and c.run_done == 1 and c.run_active
        assert c._resume_scheduled                      # the run moves on
        c.room_gone(TAG)                                # idempotent: no double count
        assert c.run_done == 1
        return c

    asyncio.run(main())


def test_on_reconnected_clears_bridge_flags_and_reschedules(monkeypatch):
    from v_dance.play import bot_control_ui as bcu
    monkeypatch.setattr(bcu, "discover_teams", lambda reg=None: [])

    async def main():
        c = _panel(asyncio.get_running_loop())
        await c.start_ladder(3, "alpha")
        assert c._search_outstanding
        c._resume_scheduled = False
        c.on_reconnected([])
        assert not c._search_outstanding and not c.searching
        assert c._resume_scheduled
        assert any("link reconnected" in e for e in c.events)
        return c

    asyncio.run(main())


# ── 2026-09-06: the "Aw, Snap!" pause (USER: "fix it so it doesn't happen again") ─────────────────────────────
# A renderer crash made page.evaluate HANG (no exception) for 38 min: the consumer and the panel sat inside it, and
# the watchdog — driven only from the consumer's idle branch — never ticked. Now: page calls time out, Playwright's
# crash event marks the link down, recovery NAVIGATES (a crashed tab cannot reload), and the watchdog is its own task.
CLIENT = "https://play.pokemonshowdown.com/"


def test_page_crash_event_navigates_back_instead_of_reloading():
    async def main():
        w, page, host, clock, logs = _watch(client_url=CLIENT)
        w.on_page_crash()                                              # Playwright's page.on("crash")
        assert w.crashed and w.closed_at is not None and w.page_crashes == 1
        assert any("PAGE CRASHED" in s for s in logs)
        await w.tick()
        assert page.gotos == [CLIENT] and page.reloads == 0            # a NAVIGATION, never a reload
        assert w.crashed is False and w.closed_at is None and w.reconnects == 1
        assert any("LINK DOWN (page crashed)" in s for s in logs) and any("LINK RECONNECTED" in s for s in logs)
        assert w.status()["page_crashes"] == 1 and "page crash" in w.banner()
    asyncio.run(main())


def test_reload_failure_falls_back_to_navigating_when_the_client_url_is_known():
    async def main():
        w, page, host, clock, logs = _watch(page=FakePage(fail_reloads=1), client_url=CLIENT)
        ws = object()
        w.on_ws_open(ws)
        w.on_ws_close(ws)
        await w.tick()
        assert page.reloads == 0 and page.gotos == [CLIENT] and w.closed_at is None
        assert any("reload failed" in s and "navigating" in s for s in logs)
        # the old contract without a client_url: the reload's failure still propagates as a failed reconnect
        w2, page2, _, clock2, logs2 = _watch(page=FakePage(fail_reloads=1))
        w2.on_ws_open(ws)
        w2.on_ws_close(ws)
        await w2.tick()
        assert page2.gotos == [] and w2.closed_at is not None and any("reconnect FAILED" in s for s in logs2)
    asyncio.run(main())


def test_a_hung_page_call_times_out_and_the_silence_path_reconnects(monkeypatch):
    monkeypatch.setattr(_pvhb, "PAGE_CALL_TIMEOUT_S", 0.05)

    async def main():
        w, page, host, clock, logs = _watch(page=FakePage(hang=True), client_url=CLIENT)
        clock.t += 16                                                  # past probe_s with no frames
        await w.tick()                                                 # the state probe hangs → TimeoutError in 50 ms
        assert any("page.evaluate failed (TimeoutError)" in s for s in logs) and w.probe_sent_at is not None
        clock.t += 10                                                  # past dead_s → reconnect
        await w.tick()
        assert w.reconnects == 1 and page.reloads == 1 and w.closed_at is None
        assert any("probe unanswered" in s for s in logs)
    asyncio.run(main())


def test_the_standalone_watchdog_task_ticks_without_the_consumer():
    async def main():
        w, page, host, clock, logs = _watch(client_url=CLIENT)
        clock.t += 16                                                  # silence past probe_s, nobody else ticking
        stop = asyncio.Event()
        task = asyncio.get_running_loop().create_task(w.run(interval_s=0.01, stop=stop))
        for _ in range(50):
            await asyncio.sleep(0.01)
            if w.probes:
                break
        stop.set()
        await task
        assert w.probes == 1 and _sends(page, "userdetails") == ["VictoriousDancing"]   # the probe went out
        # start() creates the task once; stop() cancels it
        t = w.start(asyncio.get_running_loop(), interval_s=0.01)
        assert w.start(asyncio.get_running_loop()) is t
        w.stop()
        with pytest.raises(asyncio.CancelledError):
            await t
    asyncio.run(main())


def test_page_eval_helper_times_out_instead_of_hanging(monkeypatch):
    monkeypatch.setattr(_pvhb, "PAGE_CALL_TIMEOUT_S", 0.05)

    async def main():
        page = FakePage(hang=True)
        with pytest.raises(asyncio.TimeoutError):
            await _pvhb.page_eval(page, "() => 1")
        ok = FakePage()
        assert await _pvhb.page_eval(ok, "() => app.socket.readyState") == 1
    asyncio.run(main())


# ── 2026-09-06: the Rena Valentine game — a REFUSED rejoin is not a gone room ────────────────────────────────────
# That battle started seconds before reconnect #6; the watchdog's rejoin was answered |noinit|namerequired| (the join
# reached the server before the reload's login did). The consumer treated every |noinit| as "room gone": the host
# abandoned the battle, the panel dropped the tag and counted the game — then the client rejoined the room on its own,
# the host ignored the frames of an abandoned tag, the timer chose every turn, and the tab was never closed.
def test_consumer_keeps_a_room_whose_join_was_refused_and_only_abandons_a_nonexistent_one():
    refused = f">{TAG}\n|noinit|namerequired|You must choose a name before you can join a battle."
    gone = []

    async def main():
        q: asyncio.Queue = asyncio.Queue()
        q.put_nowait(refused)
        stop = asyncio.Event()
        host = FakeHost(live=[TAG])
        _pvhb.REJOINING.add(TAG)
        _pvhb.ROOM_GONE_HOOK = gone.append

        async def stopper():
            await asyncio.sleep(1.4)
            stop.set()

        asyncio.create_task(stopper())
        await _pvhb._ai_consumer(FakePage(), host, q, ["a", "b"], {"ai": 0, "you": 0, "draw": 0}, stop)
        return host

    try:
        host = asyncio.run(main())
        assert host.abandoned == [] and host.fed == [] and gone == []       # kept, not fed, panel not told
        assert TAG in _pvhb.REJOINING and _pvhb.REJOIN_REFUSED.get(TAG) == "namerequired"
    finally:
        _pvhb.ROOM_GONE_HOOK = None
        _pvhb.REJOIN_REFUSED.clear()


def test_watchdog_retries_a_refused_rejoin_once_logged_in_then_gives_up():
    async def main():
        ctrl = Ctrl()
        host = FakeHost(live=[TAG])
        w, page, host, clock, logs = _watch(host=host, ctrl=ctrl, client_url=CLIENT)
        w.rejoin_retry_s, w.rejoin_retries = 8.0, 2
        _pvhb.REJOINING.add(TAG)
        _pvhb.REJOIN_REFUSED[TAG] = "namerequired"
        page.named = False                                                  # the reload's login not settled yet

        async def tick_alive():                                              # frames keep flowing: no silence path
            w.on_raw_frame("h")
            await w.tick()

        clock.t += 10
        await tick_alive()
        assert _sends(page, "joinRoom") == [] and any("not logged in yet" in s for s in logs)
        page.named = True
        clock.t += 10
        await tick_alive()                                                   # retry #1
        assert _sends(page, "joinRoom") == [TAG] and w.status()["rejoin_retries"] == 1
        clock.t += 3
        await tick_alive()                                                   # inside rejoin_retry_s: nothing new
        assert _sends(page, "joinRoom") == [TAG]
        clock.t += 10
        await tick_alive()                                                   # retry #2 (the last allowed)
        assert _sends(page, "joinRoom") == [TAG, TAG]
        clock.t += 10
        await tick_alive()                                                   # #3 > rejoin_retries → give up
        assert host.abandoned == [TAG] and ctrl.gone == [TAG]
        assert TAG not in _pvhb.REJOINING and TAG not in _pvhb.REJOIN_REFUSED
        assert any("giving up" in s for s in logs) and w.status()["rejoin_refused"] == []
        # a live |request| (REJOINING cleared by the consumer) drops the retry bookkeeping without a join
        _pvhb.REJOIN_REFUSED["other"] = "namerequired"
        clock.t += 10
        await tick_alive()
        assert "other" not in _pvhb.REJOIN_REFUSED and _sends(page, "joinRoom") == [TAG, TAG]
    try:
        asyncio.run(main())
    finally:
        _pvhb.REJOIN_REFUSED.clear()


def test_close_room_leaves_every_id_form_and_reports_a_missing_tab(monkeypatch):
    from v_dance.play import bot_control_ui as bcu
    monkeypatch.setattr(bcu, "discover_teams", lambda reg=None: [])

    class CountingPage(_PanelPage):
        def __init__(self, n):
            super().__init__()
            self.n, self.args = n, []

        async def evaluate(self, js, arg=None):
            if "leaveRoom" in js:
                self.args.append(arg)
                return self.n
            return await super().evaluate(js, arg)

    async def main():
        base = TAG
        full = TAG + "-privatesuffix0000000000000000000"
        c = _panel(asyncio.get_running_loop())
        c.page = CountingPage(2)
        c._full_tags[base] = full
        c._close_room(base)
        await asyncio.sleep(0.05)
        assert c.page.args == [{"base": base, "full": full}]                 # both id forms go to the client
        assert any("closed battle tab" in e for e in c.events)
        c.page = CountingPage(0)
        c._close_room(base)
        await asyncio.sleep(0.05)
        assert any("not found in the client" in e for e in c.events)         # no silent "closed" any more
        return c

    asyncio.run(main())
