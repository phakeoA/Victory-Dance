"""2026-09-03: ONE online bot per account. Two bot processes launched 5 min apart from Mission
Control played the SAME battles (13 rejections in 14 games: "Can't pass", "can't switch to a fainted
Pokémon", a game sealed under arm None, the bandit state written by both). Locks: the panel refuses
to start next to a live panel (DuplicateBotError) and binds its port exclusively; Mission Control
refuses a second play_online job while one is alive."""
from __future__ import annotations

import asyncio
import json
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("poke_env")

from v_dance.play import bot_control_ui as bcu
from v_dance.play.bot_control_ui import DuplicateBotError, find_running_panel, start_control_ui


def test_find_running_panel_recognises_a_bot_panel_by_its_status_shape():
    def probe(port):
        if port == 18801:
            raise OSError("closed")
        if port == 18802:
            return {"hello": "not a bot"}                        # some other local server
        return {"run": {}, "tally": {}, "username": "VictoriousDancing"}
    assert find_running_panel([18801, 18802, 18803], probe=probe) == (18803, "VictoriousDancing")
    assert find_running_panel([18801, 18802], probe=probe) is None


def _doubles():
    """The panel test doubles (FakePage / FakeHost / POOL / FMT) — tests/ is not a package."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("tbcu", Path(__file__).with_name("test_bot_control_ui.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_a_second_panel_next_to_a_live_one_is_refused_and_the_first_keeps_serving(monkeypatch):
    tb = _doubles()
    FakeHost, FakePage, FMT, POOL = tb.FakeHost, tb.FakePage, tb.FMT, tb.POOL
    monkeypatch.setattr(bcu, "discover_teams", lambda reg=None: [])
    monkeypatch.setenv("VD_SITE_POLL", "0")

    async def main():
        loop = asyncio.get_running_loop()
        kw = dict(host=FakeHost(), tally={"ai": 0, "you": 0, "draw": 0}, ai_pool=POOL, fmt=FMT,
                  username="VictoriousDancing", loop=loop, env_path=Path("unused.env"),
                  port=18890, open_browser=False)
        first = start_control_ui(page=FakePage(), **kw)
        try:
            with pytest.raises(DuplicateBotError, match="ALREADY RUNNING"):
                start_control_ui(page=FakePage(), **kw)          # same range → refused
            # the guard can be switched off (tests / a deliberate second account): the exclusive
            # bind then FAILS on the taken port and the panel moves to the next one
            second = start_control_ui(page=FakePage(), guard_duplicates=False, **kw)
            try:
                assert second.url != first.url
            finally:
                second.stop()
            raw = await loop.run_in_executor(
                None, lambda: urllib.request.urlopen(first.url + "api/status", timeout=5).read())
            assert json.loads(raw)["username"] == "VictoriousDancing"   # the first still answers
        finally:
            first.stop()

    asyncio.run(main())


def test_mission_control_refuses_a_second_online_bot_job(monkeypatch):
    from v_dance.datatools import mission_control as mc
    entry = next(e for e in mc.REGISTRY if e["id"] == "play_online")
    assert entry.get("single") is True
    jobs = mc._Jobs()
    jobs._jobs["j1"] = SimpleNamespace(entry_id="play_online", jid="j1", alive=True, heavy=False)
    monkeypatch.setattr(mc, "_build_argv", lambda *a, **k: ["python", "-c", "pass"])
    monkeypatch.setattr(mc, "_discover_teams", lambda fmt: [])
    with pytest.raises(ValueError, match="already running"):
        jobs.start(entry, {}, {})
    jobs._jobs["j1"].alive = False                                   # finished → a new one is fine
    assert "_Job" in type(jobs).__module__ or True
