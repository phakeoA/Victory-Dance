"""Ladder LANES (2026-09-02) — several rated games at once, designed for Showdown's 5-per-account cap.

USER: "can the bot play multiple ranked games at once? ... design for the 5-game limit so the live
ladder sessions are quicker." Server facts (pinned pokemon-showdown, server/monitor.ts): 5 games
at the same time per account, 12 battle preps + validations per 3 min per IP, ONE ladder search
per format at a time (re-searching while a game is live is allowed).

Pieces under test: the panel's lane-aware run loop (capacity, target coverage, prep-rate guard,
per-room stale sweep, one-lane behaviour unchanged), the bandit's in-flight accounting (five
searches before any reward still rotate the warm-up), pinned-tag memory, and the player's
per-battle arm scope (each decision under the arm BOUND to its tag).
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from v_dance.play import bot_control_ui as bcu
from v_dance.play import serve_bandit as SB
from v_dance.play.bot_control_ui import BotController

FMT = "gen9championsvgc2026regmb"


class FakePage:
    def __init__(self):
        self.sent = []

    async def evaluate(self, js, arg=None):
        self.sent.append((js, arg))
        return None


class _FakeTeam:
    def yield_team(self):
        return "PACKED"


class FakePlayer:
    def __init__(self):
        self._team = _FakeTeam()
        self._team_name = None

    def update_team(self, team):
        pass


class FakeHost:
    def __init__(self):
        self.player = FakePlayer()


@pytest.fixture(autouse=True)
def _no_fs(monkeypatch):
    monkeypatch.setattr(bcu, "discover_teams", lambda reg=None: [])
    monkeypatch.setenv("VD_SITE_POLL", "0")


def _ctrl(loop, page=None, **kw):
    c = BotController(page=page or FakePage(), host=FakeHost(), tally={"ai": 0, "you": 0, "draw": 0},
                      ai_pool=["alpha"], fmt=FMT, username="VictoriousDancing", loop=loop,
                      env_path=Path("unused.env"), **kw)
    c._load_scoped_team = lambda scoped: "LOADED"
    return c


def _searches(page):
    return sum(1 for js, _ in page.sent if "/search" in js)


def _tag(n):
    return f"battle-{FMT}-{7000 + n}"


async def _tick(c):
    c._resume_scheduled = False
    c._resume_tick()
    await asyncio.sleep(0.05)


def _rate(c, tag):
    c._on_rating(tag, "VictoriousDancing", 1200)
    c._on_rating(tag, "Opponent", 1190)


# ── the run loop ─────────────────────────────────────────────────────────────
def test_three_lanes_fill_as_games_are_found_and_stop_at_capacity():
    async def main():
        page = FakePage()
        c = _ctrl(asyncio.get_running_loop(), page, lanes_default=3)
        await c.start_ladder(6, "alpha")
        assert _searches(page) == 1 and c._search_outstanding
        c.tap_frame(f">{_tag(1)}\n|init|battle")           # game A found → the search slot frees
        assert not c._search_outstanding and len(c._live) == 1
        await _tick(c)
        assert _searches(page) == 2                           # a second search while A is live
        c.tap_frame(f">{_tag(2)}\n|init|battle")
        await _tick(c)
        assert _searches(page) == 3
        c.tap_frame(f">{_tag(3)}\n|init|battle")
        await _tick(c)
        assert _searches(page) == 3 and len(c._live) == 3     # lanes full → no 4th search
        assert any("lane 3/3" in e for e in c.events)
        # A ends: a lane opens; with lanes > 1 the next search does NOT wait for A's rating exchange
        c.tap_frame(f">{_tag(1)}\n|win|VictoriousDancing")
        assert c.run_done == 1 and len(c._live) == 2 and c.run_active
        await _tick(c)
        assert _searches(page) == 4
        _rate(c, _tag(1))                                     # confirm arrives later — harmless
        return c

    asyncio.run(main())


def test_target_coverage_counts_live_games_so_lanes_never_overshoot():
    async def main():
        page = FakePage()
        c = _ctrl(asyncio.get_running_loop(), page, lanes_default=5)
        await c.start_ladder(3, "alpha")
        for i in (1, 2, 3):
            c.tap_frame(f">{_tag(i)}\n|init|battle")
            await _tick(c)
        assert _searches(page) == 3 and len(c._live) == 3     # target 3 = 3 live → no 4th
        c.tap_frame(f">{_tag(1)}\n|win|Opponent")
        await _tick(c)
        assert _searches(page) == 3                           # 1 done + 2 live still covers 3
        c.tap_frame(f">{_tag(2)}\n|win|Opponent")
        c.tap_frame(f">{_tag(3)}\n|win|VictoriousDancing")
        assert c.run_done == 3 and not c.run_active
        assert any("COMPLETE" in e for e in c.events)

    asyncio.run(main())


def test_one_lane_keeps_the_wait_for_rating_exchange_contract():
    async def main():
        page = FakePage()
        c = _ctrl(asyncio.get_running_loop(), page)           # default = 1 lane
        assert c.lanes == 1
        await c.start_ladder(3, "alpha")
        c.tap_frame(f">{_tag(1)}\n|init|battle")
        await _tick(c)
        assert _searches(page) == 1                           # one lane: no search while live
        c.tap_frame(f">{_tag(1)}\n|win|Opponent")
        await _tick(c)
        assert _searches(page) == 1                           # ...nor before the Δelo confirm
        _rate(c, _tag(1))
        await _tick(c)
        assert _searches(page) == 2

    asyncio.run(main())


def test_prep_rate_guard_defers_the_eleventh_search_in_three_minutes():
    async def main():
        page = FakePage()
        c = _ctrl(asyncio.get_running_loop(), page, lanes_default=5)
        await c.start_ladder(50, "alpha")
        c.tap_frame(f">{_tag(1)}\n|init|battle")
        c._search_times = [c.loop.time()] * bcu._PREP_MAX     # budget spent this window
        await _tick(c)
        assert _searches(page) == 1
        assert any("search deferred" in e for e in c.events)
        c._search_times = []                                  # window rolled over
        await _tick(c)
        assert _searches(page) == 2

    asyncio.run(main())


def test_stale_sweep_is_per_room_under_lanes():
    async def main():
        page = FakePage()
        c = _ctrl(asyncio.get_running_loop(), page, lanes_default=2)
        await c.start_ladder(10, "alpha")
        c.tap_frame(f">{_tag(1)}\n|init|battle")
        c.tap_frame(f">{_tag(2)}\n|init|battle")
        assert len(c._live) == 2
        c._last_frame_at[_tag(1)] = c.loop.time() - bcu._LIVE_STALE_S - 5   # room 1 went silent
        await _tick(c)
        assert c._live == {_tag(2)}                           # only the silent room is swept
        assert any("looked stale" in e and _tag(1) in e and _tag(2) not in e for e in c.events)
        # global silence (no frames from ANY room) still sweeps everything
        c._last_battle_frame_at = c.loop.time() - bcu._LIVE_STALE_S - 5
        await _tick(c)
        assert not c._live

    asyncio.run(main())


def test_set_lanes_clamps_to_the_server_cap_and_reports_in_status():
    async def main():
        page = FakePage()
        c = _ctrl(asyncio.get_running_loop(), page)
        assert c.set_lanes(9) == 5 and c.status()["lanes"] == 5 and c.status()["max_lanes"] == 5
        assert c.set_lanes(0) == 1
        assert c.set_lanes("3") == 3
        assert any("lanes → 3" in e for e in c.events)
        c2 = _ctrl(asyncio.get_running_loop(), FakePage(), lanes_default=7)
        assert c2.lanes == 5

    asyncio.run(main())


# ── the bandit under lanes ───────────────────────────────────────────────────
def _bandit(tmp_path, names=("inc", "a", "b"), **kw):
    arms = [SB.Arm(name=n, incumbent=(i == 0)) for i, n in enumerate(names)]
    kw.setdefault("min_games", 2)
    return SB.ServeBandit(arms, fmt=FMT, state_path=tmp_path / "state.json", seed=1,
                          now=lambda: 1000.0, **kw)


def test_warm_up_rotates_arms_across_in_flight_games(tmp_path: Path):
    b = _bandit(tmp_path)
    chosen = []
    for i in range(3):                                        # three searches, NO reward yet
        chosen.append(b.choose().name)
        b.bind(_tag(i))
    assert sorted(chosen) == ["a", "b", "inc"]                # not "inc, inc, inc"
    assert b.in_flight == {"inc": 1, "a": 1, "b": 1}
    assert [d["in_flight"] for d in b.summary()] == [1, 1, 1]
    for i in range(3):                                        # rewards land in any order
        b.observe(_tag(2 - i), +1)
    assert b.in_flight == {"inc": 0, "a": 0, "b": 0}
    assert all(b.stats[n].n == 1 for n in ("inc", "a", "b"))


def test_pinned_for_remembers_frozen_games_and_persists(tmp_path: Path):
    b = _bandit(tmp_path)
    b.choose()
    b.bind(_tag(1))                                           # exploring
    b.pin("b")
    b.choose()
    b.bind(_tag(2))                                           # frozen
    assert b.pinned_for(_tag(2)) and not b.pinned_for(_tag(1))
    assert b.pinned_for(_tag(2) + "-privpw")                  # base-tag normalised
    b2 = _bandit(tmp_path)
    assert b2.pinned_for(_tag(2)) and not b2.pinned_for(_tag(1))


# ── per-battle arm scope on the player ───────────────────────────────────────
def _bundle(name, tau=0.0, eps=None):
    return {"name": name, "model": f"M-{name}", "heads": f"H-{name}", "chooser": f"C-{name}",
            "vocab": {name: 1}, "cfg": {"arm": name}, "tau": tau, "top_p": 0.9, "tp_tie_eps": eps,
            "rng": None}


def _player():
    return SimpleNamespace(_model="M-default", _model_heads="H-default", _team_chooser="C-default",
                           _tc_vocab={}, _tc_cfg={}, _temperature=0.0, _top_p=1.0, _rng=None,
                           _arm_name="default", _arm_resolver=None)


def test_arm_scope_swaps_the_bound_bundle_in_and_restores_after(monkeypatch):
    monkeypatch.delenv("VD_TP_TIE_EPS", raising=False)
    p = _player()
    bundles = {_tag(1): _bundle("tau", tau=0.3, eps=0.5), _tag(2): _bundle("argmax")}
    p._arm_resolver = lambda tag: bundles.get(SB.ServeBandit.base_tag(tag))
    with SB.arm_scope(p, _tag(1) + "-privpw") as b:            # suffixed tag resolves too
        assert b["name"] == "tau" and p._model == "M-tau" and p._temperature == 0.3
        assert p._tc_cfg == {"arm": "tau"} and os.environ["VD_TP_TIE_EPS"] == "0.5"
    assert p._model == "M-default" and p._temperature == 0.0 and p._arm_name == "default"
    assert "VD_TP_TIE_EPS" not in os.environ                  # env restored (was unset)
    with SB.arm_scope(p, _tag(2)) as b:
        assert b["name"] == "argmax" and p._model == "M-argmax"
    with SB.arm_scope(p, _tag(99)) as b:                      # unknown tag → no-op
        assert b is None and p._model == "M-default"


def test_arm_scope_is_a_no_op_without_a_resolver_and_survives_a_broken_one():
    p = _player()
    with SB.arm_scope(p, _tag(1)) as b:
        assert b is None and p._model == "M-default"

    def boom(tag):
        raise RuntimeError("resolver broke")
    p._arm_resolver = boom
    with SB.arm_scope(p, _tag(1)) as b:
        assert b is None and p._model == "M-default"


def test_apply_bundle_matches_apply_arm(tmp_path: Path):
    loads = []

    def loader(kind, path):
        loads.append((kind, str(path)))
        return ("model", "heads") if kind == "battle" else ("chooser", {"v": 1}, {"c": 1})
    arm = SB.Arm(name="x", battle_ckpt="default", tp_ckpt="default", tau=0.2, top_p=0.9, tp_tie_eps=1.0)
    cache = {}
    b = SB.load_bundle(arm, cache, default_battle=tmp_path / "b.pt", default_tp=tmp_path / "t.pt",
                       loader=loader)
    SB.load_bundle(arm, cache, default_battle=tmp_path / "b.pt", default_tp=tmp_path / "t.pt",
                   loader=loader)
    assert len(loads) == 2                                    # cached per path
    assert b["model"] == "model" and b["chooser"] == "chooser" and b["rng"] is not None
    host = SimpleNamespace(player=_player())
    SB.apply_arm(host, arm, cache, default_battle=tmp_path / "b.pt", default_tp=tmp_path / "t.pt",
                 loader=loader)
    assert host.player._model == "model" and host.player._temperature == 0.2 \
        and host.player._arm_name == "x" and os.environ.get("VD_TP_TIE_EPS") == "1.0"


# ── the panel with a per-battle resolver: swaps are allowed under live games ─
class _Page:
    async def evaluate(self, js, arg=None):
        return None


def test_panel_applies_arms_and_pins_under_live_games_when_the_player_resolves_per_tag(tmp_path):
    applied = []
    arms = [SB.Arm(name="inc", incumbent=True), SB.Arm(name="a")]
    b = SB.ServeBandit(arms, fmt=FMT, state_path=tmp_path / "s.json", seed=0, min_games=1,
                       applier=lambda arm: applied.append(arm.name))

    async def main():
        c = BotController(page=_Page(), host=FakeHost(), tally={"ai": 0, "you": 0, "draw": 0},
                          ai_pool=["alpha"], fmt=FMT, username="VictoriousDancing",
                          loop=asyncio.get_running_loop(), env_path=tmp_path / "u.env", bandit=b,
                          lanes_default=3)
        c._load_scoped_team = lambda scoped: "LOADED"
        c.host.player._arm_resolver = lambda tag: None        # per-tag resolution installed
        await c.start_ladder(6, "alpha")
        first = applied[-1]
        c._battle_seen(_tag(1))
        assert b.arm_for(_tag(1)) == first
        assert c.apply_next_arm("search") is not None         # allowed while a game is live
        c.set_bandit_pin("a")
        await asyncio.sleep(0)
        assert b.pinned == "a" and applied[-1] == "a"          # applied right away, no deferral
        assert not any("applies from the next game" in e for e in c.events)
        c._battle_seen(_tag(2))
        assert b.arm_for(_tag(2)) == "a" and any("[arm a pinned]" in e for e in c.events)
        assert b.arm_for(_tag(1)) == first                     # the live game keeps ITS arm

    asyncio.run(main())


def test_mission_control_exposes_the_lanes_env_key():
    from v_dance.datatools import mission_control as mc
    assert "VD_LADDER_LANES" in mc._ENV_WRITE_KEYS and "VD_LADDER_LANES" in mc._ENV_READ_KEYS


# ── W3b decision 1 (2026-09-02): adapt-rules per arm, forced OFF for τ arms ──
def test_adapt_rules_per_arm_defaults_off_for_tau_arms_and_rides_the_bundle(tmp_path: Path):
    argmax = SB.Arm(name="inc", incumbent=True)
    tau = SB.Arm(name="tau", tau=0.3)
    forced_on = SB.Arm(name="tau_on", tau=0.3, adapt_rules=True)
    assert argmax.adapt_rules_for(True) is True and argmax.adapt_rules_for(False) is False
    assert tau.adapt_rules_for(True) is False                 # τ > 0 → clean unless the config says so
    assert forced_on.adapt_rules_for(False) is True

    def loader(kind, path):
        return ("m", "h") if kind == "battle" else ("c", {}, {})
    cache = {}
    kw = dict(default_battle=tmp_path / "b.pt", default_tp=tmp_path / "t.pt", loader=loader)
    assert SB.load_bundle(argmax, cache, adapt_rules_default=True, **kw)["adapt_rules"] is True
    assert SB.load_bundle(tau, cache, adapt_rules_default=True, **kw)["adapt_rules"] is False
    p = _player()
    p._adapt_rules = True                                      # the launch flag on the player
    with SB.arm_scope(p, _tag(1)) as b:                        # no resolver → untouched
        assert b is None and p._adapt_rules is True
    bundles = {_tag(1): dict(_bundle("tau", tau=0.3), adapt_rules=False)}
    p._arm_resolver = lambda tag: bundles.get(SB.ServeBandit.base_tag(tag))
    with SB.arm_scope(p, _tag(1)):
        assert p._adapt_rules is False                         # the τ arm decides WITHOUT the tilt
    assert p._adapt_rules is True                              # restored


def test_load_arms_reads_adapt_rules_from_the_config(tmp_path: Path):
    import json
    cfg = tmp_path / "arms.json"
    cfg.write_text(json.dumps({"arms": [
        {"name": "inc", "incumbent": True},
        {"name": "tau", "tau": 0.3, "top_p": 1.0, "adapt_rules": False},
        {"name": "tau_on", "tau": 0.3, "adapt_rules": True},
    ]}), encoding="utf-8")
    arms = {a.name: a for a in SB.load_arms(cfg, exists=lambda p: True)}
    assert arms["inc"].adapt_rules is None and arms["tau"].adapt_rules is False
    assert arms["tau_on"].adapt_rules is True

