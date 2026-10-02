"""Mission Control with two ladder accounts (2026-10-02, USER: "separating the elos on the graph, having the ui support
it" + "control both accounts with different teams").

Under test: the play_online launch carries ``--account N`` and its own log name; one bot PER ACCOUNT (account 2 may
start next to account 1, never a second account 1); the Online tab's proxy reaches THAT account's panel range;
the status poll lists the configured accounts with their bot state; the port probes run concurrently; the rating
chart draws one chart per (format, account) and older rows stay account 1's.
"""
from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from v_dance.ui import mission_control as mc
from v_dance.ui import rating_chart as RC


class _FakeJob:
    made: list = []

    def __init__(self, jid, entry_id, argv, env_extra, heavy=False):
        self.jid, self.entry_id, self.argv, self.heavy, self.alive = jid, entry_id, argv, heavy, True
        a = mc._arg_val(argv, "--account")
        self.account = mc._slot(a) if a else None
        _FakeJob.made.append(self)

    def info(self):
        return {"job": self.jid, "id": self.entry_id, "account": self.account}


@pytest.fixture
def jobs(monkeypatch):
    _FakeJob.made = []
    monkeypatch.setattr(mc, "_Job", _FakeJob)
    monkeypatch.setattr(mc, "_panel_status", lambda *a: {"up": False})
    monkeypatch.setattr(mc, "_discover_teams", lambda fmt: ["Baltimore_Sand_Psy", "The_Big_6_v2"])
    monkeypatch.setattr(mc, "_ckpt_inventory", lambda: {"battle": [], "tp": []})
    return mc._Jobs()


def test_the_launch_carries_the_account_and_one_bot_per_account_may_run(jobs):
    bot = mc._REG_BY_ID["play_online"]
    one = jobs.start(bot, {"account": "1", "ai-team": "Baltimore_Sand_Psy"}, {})
    two = jobs.start(bot, {"account": "2", "ai-team": "The_Big_6_v2"}, {})      # a different team per account
    assert (one["account"], two["account"]) == (1, 2)
    a1, a2 = (j.argv for j in _FakeJob.made)
    assert a1[a1.index("--account") + 1] == "1" and a2[a2.index("--account") + 1] == "2"
    assert a1[a1.index("--ai-team") + 1] == "Baltimore_Sand_Psy" and a2[a2.index("--ai-team") + 1] == "The_Big_6_v2"
    with pytest.raises(ValueError, match="already running for account 2"):
        jobs.start(bot, {"account": "2"}, {})
    with pytest.raises(ValueError, match="already running for account 1"):
        jobs.start(bot, {}, {})                                                  # no option = account 1


def test_the_chain_update_is_refused_while_either_accounts_bot_is_up(jobs, monkeypatch):
    monkeypatch.setattr(mc, "_panel_status",
                        lambda slot=1: {"up": slot == 2, "port": 8787 if slot == 2 else None})
    with pytest.raises(ValueError, match="needs the online bot DOWN.*8787"):
        jobs.start(mc._REG_BY_ID["ladder_ppo"], {}, {})


def test_account_two_gets_its_own_log_name(tmp_path, monkeypatch):
    monkeypatch.setattr(mc, "_LOGS_DIR", tmp_path)
    monkeypatch.setattr(mc.subprocess, "Popen", lambda *a, **k: SimpleNamespace(poll=lambda: None, pid=0))
    j2 = mc._Job("j1", "play_online", ["py", "-m", "v_dance.online.bot", "--account", "2"], {})
    j1 = mc._Job("j2", "play_online", ["py", "-m", "v_dance.online.bot", "--account", "1"], {})
    other = mc._Job("j3", "ladder_ppo", ["py"], {})
    assert j2.log_path.name.startswith("mc_play_online_acct2_") and j2.account == 2
    assert j1.log_path.name.startswith("mc_play_online_2") and j1.account == 1    # account 1: the old name
    assert other.account is None and other.log_path.name.startswith("mc_ladder_ppo_")
    for j in (j1, j2, other):
        j._log_f.close()


def test_the_online_proxy_looks_in_the_accounts_own_port_range(monkeypatch):
    probed = []
    monkeypatch.setattr(mc, "_port_open", lambda p: (probed.append(p), False)[1])
    mc._panel_cache.clear()
    assert mc._panel_status(2) == {"up": False, "account": 2}
    assert sorted(probed) == list(range(8787, 8797))
    probed.clear()
    mc._panel_status(1)
    assert sorted(probed) == list(range(8777, 8787))
    with pytest.raises(ValueError, match="account 2"):
        mc._panel_post("/api/team", {}, 2)
    assert mc._slot("2") == 2 and mc._slot("9") == 1 and mc._slot(None) == 1


def test_the_status_lists_each_configured_account_and_whose_bot_is_up(monkeypatch):
    monkeypatch.setattr(mc, "_env_read", lambda: {"PS_USERNAME": "VictoriousDancing", "PS_PASSWORD": "x",
                                                  "PS2_USERNAME": "EncoreFN", "PS2_PASSWORD": "y"})
    mc._panel_cache.clear()
    accts = mc._accounts_up({8777: False, 8787: True})
    assert [(a["slot"], a["username"], a["up"], a["port"]) for a in accts] == \
        [(1, "VictoriousDancing", False, None), (2, "EncoreFN", True, 8787)]
    assert mc._bot_panel_up({8777: False, 8787: True}) is True
    assert mc._bot_panel_up({8777: False, 8787: False}) is False
    assert "VD_BANDIT_PIN_2" in mc._ENV_WRITE_KEYS and "VD_DEFAULT_TEAM_2" in mc._ENV_READ_KEYS


def test_port_probes_run_concurrently(monkeypatch):
    def slow(p):
        time.sleep(0.15)
        return p == 5175
    monkeypatch.setattr(mc, "_port_open", slow)
    t0 = time.perf_counter()
    got = mc._ports_open([8000, 5174, 5175, 8777, 8787, 8787])
    assert got == {8000: False, 5174: False, 5175: True, 8777: False, 8787: False}
    assert time.perf_counter() - t0 < 0.45                     # serial would be 5 × 0.15 = 0.75 s


def test_the_rating_chart_draws_one_chart_per_account(tmp_path):
    env = tmp_path / ".env"
    env.write_text("PS_USERNAME=VictoriousDancing\nPS_PASSWORD=x\nPS2_USERNAME=EncoreFN\nPS2_PASSWORD=y\n")
    rows = []
    for i in range(30):
        t1, t2 = f"battle-gen9championsvgc2026regmc-{100 + i}", f"battle-gen9championsvgc2026regmc-{500 + i}"
        rows.append({"type": "rating_update", "battle_tag": t1, "rating": 1200 + i, "rating_after": 1201 + i})  # old row
        rows.append({"type": "rating_update", "battle_tag": t2, "rating": 1000 + 5 * i, "rating_after": 1005 + 5 * i,
                     "account": "encorefn"})
    rows.append({"type": "site_rating", "format": "gen9championsvgc2026regmc", "elo": 1150, "w": 20, "l": 10, "ts": "2026-10-02T05:00:00Z",
                 "account": "encorefn"})
    b = tmp_path / "bench.jsonl"
    b.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    page, counts = RC.build_page(b, window=100, theme="dark", env_path=env)
    assert counts == {"gen9championsvgc2026regmc": 30, "gen9championsvgc2026regmc · EncoreFN": 30}
    assert page.count("<svg") == 2
    assert page.index("· VictoriousDancing —") < page.index("· EncoreFN —")       # the primary first
    assert "SITE (truth) 1150, 20W-10L" in page                                   # on EncoreFN's chart only
    h2s = [ln for ln in page.split("<h2>")[1:]]
    assert "SITE" not in h2s[0].split("</h2>")[0]                                # VictoriousDancing: no site row


def test_a_browser_that_hangs_up_mid_reply_is_not_a_traceback(monkeypatch):
    """USER 2026-10-02: every Mission Control launch printed WinError 10053 twice — a poll's client hung up while the
    reply (the Online tab's panel status, ~0.5 s) was being prepared. The handler now drops such replies quietly."""
    import socket
    import struct
    import threading
    from http.server import ThreadingHTTPServer

    def slow_status(slot=1):
        time.sleep(0.4)
        return {"up": False, "account": slot}
    monkeypatch.setattr(mc, "_panel_status", slow_status)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), mc._make_handler())
    errors = []
    srv.handle_error = lambda request, addr: errors.append(addr)    # what printed the traceback
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        for _ in range(3):
            c = socket.create_connection(srv.server_address)
            c.sendall(b"GET /api/online/status?account=2 HTTP/1.1\r\nHost: x\r\n\r\n")
            c.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            c.close()                                               # RST: the reload / tab close
        time.sleep(1.0)
        with socket.create_connection(srv.server_address) as c:     # and it still serves a live client
            c.sendall(b"GET /api/online/status?account=2 HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
            data = b""
            while chunk := c.recv(4096):
                data += chunk
        assert b"200 OK" in data and b'"account": 2' in data
        assert errors == []
    finally:
        srv.shutdown()
