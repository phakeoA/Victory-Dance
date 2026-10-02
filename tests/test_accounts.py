"""Two ladder accounts on one box (2026-10-02, USER: "make a 2nd account for the bot to gather data twice as fast").

The 10-01 overnight log showed the 5-games-per-ACCOUNT cap is the bottleneck on M-C (1096 of 1245 games started at
lane 5/5), so a second account adds games — but both bot processes share the box's IP, and Showdown's prep cap (12
battles per 3 min) is PER IP. Under test: the account slots (.env keys, panel ports, per-account knobs, bandit
config + state paths), the bench-row account stamp (old rows = account 1), the locked JSONL append, the box-wide
search ledger and the panel using it, the duplicate guard now refusing only the SAME account, and per-account peaks.
"""
from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import pytest

from v_dance.online import accounts as A

ENV = {"PS_USERNAME": "VictoriousDancing", "PS_PASSWORD": "pw1", "PS_AVATAR": "lucas",
       "PS2_USERNAME": "Victory Dance Two", "PS2_PASSWORD": "pw2", "PS2_AVATAR": "dawn"}   # the USER's spelling


# ── slots ────────────────────────────────────────────────────────────────────
def test_slot_one_is_the_original_keys_and_slot_two_the_ps2_ones():
    a1, a2 = A.load_account(ENV, 1), A.load_account(ENV, 2)
    assert (a1.username, a1.password, a1.avatar, a1.avatar_key) == ("VictoriousDancing", "pw1", "lucas", "PS_AVATAR")
    assert (a2.username, a2.password, a2.avatar, a2.avatar_key) == ("Victory Dance Two", "pw2", "dawn", "PS2_AVATAR")
    assert a2.userid == "victorydancetwo"
    assert "pw2" not in repr(a2)                                   # the password never prints
    assert A.configured_slots(ENV) == [1, 2]
    assert A.configured_slots({k: v for k, v in ENV.items() if not k.startswith("PS2_")}) == [1]


def test_the_suffixed_spelling_works_too_and_a_new_avatar_follows_the_usernames_spelling():
    env = {"PS_USERNAME_2": "Two", "PS_PASSWORD_2": "pw"}
    a = A.load_account(env, 2)
    assert (a.username, a.avatar, a.avatar_key) == ("Two", "", "PS_AVATAR_2")
    assert A.load_account({"PS2_USERNAME": "Two", "PS2_PASSWORD": "pw"}, 2).avatar_key == "PS2_AVATAR"


def test_a_missing_or_out_of_range_account_is_refused_by_name():
    with pytest.raises(ValueError, match="PS2_USERNAME / PS2_PASSWORD or PS_USERNAME_2 / PS_PASSWORD_2"):
        A.load_account({"PS_USERNAME": "a", "PS_PASSWORD": "b"}, 2)
    with pytest.raises(ValueError, match="1..2"):
        A.load_account(ENV, 3)


def test_each_account_has_its_own_panel_port_range():
    assert A.panel_ports(1) == range(8777, 8787)                   # unchanged for account 1
    assert A.panel_ports(2) == range(8787, 8797)
    assert A.load_account(ENV, 2).panel_port == 8787
    assert A.all_panel_ports() == range(8777, 8797)
    assert (A.slot_of_port(8780), A.slot_of_port(8790), A.slot_of_port(8800)) == (1, 2, None)


def test_per_account_knobs_override_only_for_that_account():
    env = {"VD_BANDIT_PIN": "explore", "VD_BANDIT_PIN_2": "era2", "VD_LADDER_LANES_2": "3"}
    assert A.knob_overrides(1, env.get) == {}
    assert A.knob_overrides(2, env.get) == {"VD_BANDIT_PIN": "era2", "VD_LADDER_LANES": "3"}


def test_bandit_config_per_account_falls_back_to_the_shared_one(tmp_path):
    shared = tmp_path / "serve_bandit.json"
    shared.write_text("{}")
    look = {}.get
    assert A.bandit_config_path(1, look, shared, tmp_path) == shared
    assert A.bandit_config_path(2, look, shared, tmp_path) == shared          # no own config → the same arms
    own = tmp_path / "serve_bandit_2.json"
    own.write_text("{}")
    assert A.bandit_config_path(2, look, shared, tmp_path) == own
    assert A.bandit_config_path(1, look, shared, tmp_path) == shared          # account 1 never picks it up
    explicit = tmp_path / "exp.json"
    assert A.bandit_config_path(2, {"VD_BANDIT_CONFIG_2": str(explicit)}.get, shared, tmp_path) == explicit


def test_bandit_state_is_per_account_and_account_one_keeps_its_file(tmp_path):
    fmt = "gen9championsvgc2026regmc"
    assert A.bandit_state_path(A.load_account(ENV, 1), fmt, tmp_path) == tmp_path / f"{fmt}.json"
    assert A.bandit_state_path(A.load_account(ENV, 2), fmt, tmp_path) == tmp_path / f"{fmt}_victorydancetwo.json"


def test_rows_without_an_account_belong_to_account_one():
    assert A.row_account({"battle_tag": "x"}, "victoriousdancing") == "victoriousdancing"
    assert A.row_account({"account": "victorydancetwo"}, "victoriousdancing") == "victorydancetwo"


# ── the locked append ────────────────────────────────────────────────────────
def _append_many(path, who, n):
    for i in range(n):
        A.append_jsonl(Path(path), {"who": who, "i": i, "pad": "x" * 200})


def test_two_processes_appending_never_lose_or_tear_a_row(tmp_path):
    p = tmp_path / "bench.jsonl"
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_append_many, args=(str(p), w, 150)) for w in ("a", "b", "c")]
    for pr in procs:
        pr.start()
    for pr in procs:
        pr.join(60)
    rows = [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 450
    assert {(r["who"], r["i"]) for r in rows} == {(w, i) for w in "abc" for i in range(150)}


# ── the box-wide search ledger ───────────────────────────────────────────────
def test_the_ledger_counts_the_other_accounts_recent_searches_only(tmp_path):
    now = [1_000_000.0]
    me = A.PrepLedger("VictoriousDancing", tmp_path, clock=lambda: now[0])
    other = A.PrepLedger("Victory Dance Two", tmp_path, clock=lambda: now[0])
    for dt in (-200.0, -100.0, -10.0, -1.0):                       # one outside the 180 s window
        other.record(now[0] + dt)
    me.record()
    me.record()
    assert me.others(180.0) == 3                                    # own searches are not "others"
    assert other.others(180.0) == 2
    with other.path.open("a") as f:
        f.write("garbage\n99999")                                   # a torn last line is skipped
    assert me.others(180.0) == 3


def test_a_ledger_untouched_for_a_whole_window_is_skipped_without_reading(tmp_path):
    other = A.PrepLedger("two", tmp_path)
    other.record(time.time())
    old = time.time() - 600
    os.utime(other.path, (old, old))
    assert A.PrepLedger("one", tmp_path).others(180.0) == 0


class _Ledger:
    def __init__(self, others):
        self.n, self.recorded = others, 0

    def others(self, window_s):
        return self.n

    def record(self, t=None):
        self.recorded += 1


def test_the_panel_defers_a_search_when_the_other_account_spent_the_budget():
    pytest.importorskip("poke_env")
    import importlib.util
    spec = importlib.util.spec_from_file_location("tll", Path(__file__).with_name("test_ladder_lanes.py"))
    tll = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tll)
    from v_dance.online import panel as bcu

    async def main():
        page = tll.FakePage()
        ledger = _Ledger(bcu._PREP_MAX - 1)
        c = tll._ctrl(asyncio.get_running_loop(), page, lanes_default=5, prep_ledger=ledger)
        await c.start_ladder(50, "alpha")                           # 1 own + 9 others = the ceiling
        assert tll._searches(page) == 1 and ledger.recorded == 1
        c.tap_frame(f">{tll._tag(1)}\n|init|battle")
        await tll._tick(c)
        assert tll._searches(page) == 1
        assert any("search deferred" in e and "other account" in e for e in c.events)
        assert c.status()["prep"] == {"own": 1, "others": bcu._PREP_MAX - 1, "max": bcu._PREP_MAX,
                                      "window_s": bcu._PREP_WINDOW_S}
        ledger.n = 0                                                # their window rolled over
        await tll._tick(c)
        assert tll._searches(page) == 2 and ledger.recorded == 2

    with pytest.MonkeyPatch.context() as mpatch:
        mpatch.setattr(bcu, "discover_teams", lambda reg=None: [])
        mpatch.setenv("VD_SITE_POLL", "0")
        asyncio.run(main())


# ── the duplicate guard + per-account peaks ──────────────────────────────────
def test_a_panel_for_a_different_account_may_start_but_the_same_account_may_not(monkeypatch):
    pytest.importorskip("poke_env")
    import importlib.util
    import urllib.request
    from v_dance.online import panel as bcu
    from v_dance.online.panel import DuplicateBotError, start_control_ui
    spec = importlib.util.spec_from_file_location("tbcu", Path(__file__).with_name("test_bot_control_ui.py"))
    tb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tb)
    monkeypatch.setattr(bcu, "discover_teams", lambda reg=None: [])
    monkeypatch.setenv("VD_SITE_POLL", "0")

    async def main():
        loop = asyncio.get_running_loop()
        kw = dict(host=tb.FakeHost(), tally={"ai": 0, "you": 0, "draw": 0}, ai_pool=tb.POOL, fmt=tb.FMT,
                  loop=loop, env_path=Path("unused.env"), open_browser=False, guard_ports=range(18900, 18920))
        first = start_control_ui(page=tb.FakePage(), username="VictoriousDancing", port=18900,
                                 account_slot=1, **kw)
        try:
            with pytest.raises(DuplicateBotError, match="ALREADY RUNNING"):
                start_control_ui(page=tb.FakePage(), username="Victorious Dancing", port=18910, **kw)
            second = start_control_ui(page=tb.FakePage(), username="Victory Dance Two", port=18910,
                                      account_slot=2, account_id="victorydancetwo", **kw)
            try:
                raw = await loop.run_in_executor(
                    None, lambda: urllib.request.urlopen(second.url + "api/status", timeout=5).read())
                st = json.loads(raw)
                assert (st["username"], st["account_slot"], st["account_id"]) == \
                    ("Victory Dance Two", 2, "victorydancetwo")
                assert second.url.endswith(":18910/")
            finally:
                second.stop()
        finally:
            first.stop()

    asyncio.run(main())


def test_all_time_peaks_are_seeded_from_this_accounts_rows_only(tmp_path):
    pytest.importorskip("poke_env")
    from v_dance.online.panel import RatingBook
    tag = "battle-gen9championsvgc2026regmc-{}"
    p = tmp_path / "bench.jsonl"
    rows = [{"type": "rating_update", "battle_tag": tag.format(1), "rating": 1300, "rating_after": 1320},  # old = acct 1
            {"type": "rating_update", "battle_tag": tag.format(2), "rating": 1500, "rating_after": 1510,
             "account": "victorydancetwo"},
            {"battle_tag": tag.format(2), "result": "ai", "account": "victorydancetwo"}]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    one = RatingBook(p, account="victoriousdancing", primary="victoriousdancing").summary()[0]
    two = RatingBook(p, account="victorydancetwo", primary="victoriousdancing").summary()[0]
    both = RatingBook(p).summary()[0]
    assert (one["all_time_peak"], one["all_time_games"]) == (1320, 0)
    assert (two["all_time_peak"], two["all_time_games"]) == (1510, 1)
    assert both["all_time_peak"] == 1510                              # account=None: every row, as before


def test_the_second_accounts_avatar_is_saved_under_its_own_key(tmp_path, monkeypatch):
    pytest.importorskip("poke_env")
    from v_dance.online import bot as O
    monkeypatch.setattr(O, "_ENV", {})                              # don't touch the real .env mirror
    p = tmp_path / ".env"
    p.write_text("PS_USERNAME=a\nPS_AVATAR=lucas\nPS_USERNAME_2=b\n", encoding="utf-8")
    O._write_avatar("dawn", env_path=p, key="PS_AVATAR_2")
    text = p.read_text(encoding="utf-8")
    assert "PS_AVATAR=lucas" in text and "PS_AVATAR_2=dawn" in text


# ── the self-match tripwire ──────────────────────────────────────────────────
def test_a_game_against_our_own_other_account_is_flagged_never_counted_silently(tmp_path, monkeypatch):
    """USER 2026-10-02: "both accounts at around the same elo will fight each other". Showdown never pairs two
    same-IP users on the ladder, so this must never fire live — but if it ever does, the row says so."""
    pytest.importorskip("poke_env")
    from types import SimpleNamespace
    from v_dance.online import bot as O
    from v_dance.play import opponent_dossier as OD
    monkeypatch.setattr(O, "BENCH_LOG", tmp_path / "bench.jsonl")
    monkeypatch.setattr(O, "LOG_DIR", tmp_path)
    monkeypatch.setattr(OD, "update_from_battle", lambda *a, **k: None)      # no dossier writes from a test
    monkeypatch.setattr(O._pvhb, "GAME_DONE_HOOK", None)
    monkeypatch.setattr(O, "_ACCOUNT_ID", "victoriousdancing")
    monkeypatch.setattr(O, "_OWN_ACCOUNTS", {"victoriousdancing", "encorefn"})
    battles = {}
    host = SimpleNamespace(player=SimpleNamespace(_battles=battles, username="VictoriousDancing",
                                                  _team_name="Baltimore_Sand_Psy"),
                           end_battle=lambda tag: None)
    O._wrap_bench_recording(host, "s1", "online", Path("a.pt"), Path("b.pt"))
    for i, opp in enumerate(("EncoreFN", "SomeHuman")):
        tag = f"battle-gen9championsvgc2026regmc-{900 + i}"
        battles[tag] = SimpleNamespace(finished=True, won=True, lost=False, battle_tag=tag, opponent_username=opp,
                                       turn=6, _players=[])
        host.end_battle(tag)
    rows = [json.loads(ln) for ln in (tmp_path / "bench.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(r["opponent"], r.get("self_match", False), r["account"]) for r in rows] == \
        [("EncoreFN", True, "victoriousdancing"), ("SomeHuman", False, "victoriousdancing")]
    assert "SELF-MATCH" in (tmp_path / "online_s1.log").read_text(encoding="utf-8")
