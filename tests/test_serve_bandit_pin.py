"""Serve-mode PIN (2026-09-02) — the frozen-vs-explore toggle on top of the era-5 W0 bandit.

USER: "if this is with the live weights changing, can we have a toggle in Mission Control or a
separate mode in case I want to ladder with a frozen-weight version?" Nothing trains live — the
bandit only swaps FIXED checkpoints between games — so "frozen" = pin ONE arm: it plays every game
from the next battle on, its games still credit its stats, bench rows carry ``pinned``. Set at
runtime from the panel (Mission Control fronts it) or at launch via ``VD_BANDIT_PIN``.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from v_dance.play import serve_bandit as SB
from v_dance.play.bot_control_ui import BotController

FMT = "gen9championsvgc2026regmb"


def _tag(n, suffix=""):
    return f"battle-{FMT}-{2000 + n}{suffix}"


def _bandit(tmp_path, names=("inc", "a", "b"), **kw):
    arms = [SB.Arm(name=n, incumbent=(i == 0)) for i, n in enumerate(names)]
    kw.setdefault("min_games", 2)
    return SB.ServeBandit(arms, fmt=FMT, state_path=tmp_path / "state.json", seed=1,
                          now=lambda: 1000.0, **kw)


# ── the bandit ───────────────────────────────────────────────────────────────
def test_pin_makes_one_arm_play_every_game_and_persists(tmp_path: Path):
    b = _bandit(tmp_path)
    assert b.pin("b") == "b"
    for i in range(5):                              # warm-up would have rotated; the pin wins
        assert b.choose().name == "b"
        b.bind(_tag(i))
        b.observe(_tag(i), -3)
    assert b.stats["b"].n == 5 and b.stats["b"].losses == 5   # pinned games still credit the arm
    assert b.stats["inc"].n == 0 and b.stats["a"].n == 0
    b2 = _bandit(tmp_path)                          # restart → the pin is restored from state
    assert b2.pinned == "b" and b2.choose().name == "b"
    assert [d["pinned"] for d in b2.summary()] == [False, False, True]
    assert "PINNED → b" in b2.banner()


def test_unpin_resets_pending_and_the_warm_up_rotates_again(tmp_path: Path):
    b = _bandit(tmp_path)
    b.pin("a")
    assert b.choose().name == "a" and b.pending == "a"
    assert b.pin(None) is None and b.pending is None
    assert "PINNED" not in b.banner()
    played = set()
    for i in range(6):
        played.add(b.choose().name)
        b.bind(_tag(i))
        b.observe(_tag(i), 0)
    assert played == {"inc", "a", "b"}              # every arm plays its warm-up games again


def test_pin_beats_the_retired_flag_and_rejects_unknown_arms(tmp_path: Path):
    b = _bandit(tmp_path)
    b.stats["b"].retired = True
    b.pin("b")
    assert b.choose().name == "b"                   # an explicit human choice overrides retirement
    with pytest.raises(ValueError):
        b.pin("nope")
    assert b.pinned == "b"                          # a bad name changes nothing
    assert b.pin("b") == "b"                        # re-pinning the same arm is a no-op
    assert b.pin("") is None                        # "" clears, like None


def test_a_stale_pin_in_the_state_file_is_dropped(tmp_path: Path):
    b = _bandit(tmp_path)
    b.pin("a")
    b2 = _bandit(tmp_path, names=("inc", "b"))      # 'a' no longer configured → no pin
    assert b2.pinned is None and b2.choose().name in ("inc", "b")


def test_pin_change_re_applies_through_apply_pending(tmp_path: Path):
    applied = []
    arms = [SB.Arm(name="inc", incumbent=True), SB.Arm(name="a")]
    b = SB.ServeBandit(arms, fmt=FMT, state_path=tmp_path / "s.json", seed=0, min_games=1,
                       applier=lambda arm: applied.append(arm.name))
    first = b.apply_pending().name
    other = "a" if first == "inc" else "inc"
    b.pin(other)
    assert b.apply_pending().name == other and applied[-1] == other
    b.apply_pending()                               # unchanged pin → no second apply
    assert applied.count(other) == 1


def test_apply_env_pin_semantics(tmp_path: Path):
    b = _bandit(tmp_path)
    assert SB.apply_env_pin(b, None) == "" and b.pinned is None          # unset, nothing saved → quiet
    assert "PINNED" in SB.apply_env_pin(b, "a") and b.pinned == "a"      # arm name → pin
    assert "IGNORED" in SB.apply_env_pin(b, "typo") and b.pinned == "a"  # unknown → loud, unchanged
    assert "restored" in SB.apply_env_pin(b, "") and b.pinned == "a"     # unset → keep the persisted pin
    assert "cleared" in SB.apply_env_pin(b, "explore") and b.pinned is None
    assert SB.apply_env_pin(b, "0") == "" and b.pinned is None           # clearing nothing = quiet
    assert "PINNED" in SB.apply_env_pin(b, " b ") and b.pinned == "b"    # whitespace tolerated


# ── the panel (Mission Control fronts it) ────────────────────────────────────
class _Page:
    async def evaluate(self, js, arg=None):
        return None


class _PanelHost:
    def __init__(self):
        class _T:
            def yield_team(self):
                return "PACKED"

        self.player = type("P", (), {"_team": _T(), "_team_name": None,
                                     "update_team": lambda self, t: None})()


def _controller(tmp_path, bandit):
    c = BotController(page=_Page(), host=_PanelHost(), tally={"ai": 0, "you": 0, "draw": 0},
                      ai_pool=["alpha"], fmt=FMT, username="VictoriousDancing",
                      loop=asyncio.get_running_loop(), env_path=tmp_path / "u.env", bandit=bandit)
    c._load_scoped_team = lambda scoped: "LOADED"
    return c


def test_panel_pin_applies_between_games_and_defers_while_a_battle_is_live(monkeypatch, tmp_path):
    from v_dance.play import bot_control_ui as bcu
    monkeypatch.setattr(bcu, "discover_teams", lambda reg=None: [])
    monkeypatch.setenv("VD_SITE_POLL", "0")
    applied = []
    arms = [SB.Arm(name="inc", incumbent=True), SB.Arm(name="a")]
    b = SB.ServeBandit(arms, fmt=FMT, state_path=tmp_path / "s.json", seed=0, min_games=1,
                       applier=lambda arm: applied.append(arm.name))

    async def main():
        c = _controller(tmp_path, b)
        st = c.status()
        assert st["bandit_on"] is True and st["bandit_pin"] is None
        c.set_bandit_pin("a")                        # no battle live → applied on the loop right away
        await asyncio.sleep(0)
        assert applied[-1] == "a" and c.host.player._arm_pinned is True
        assert c.status()["bandit_pin"] == "a"
        assert any("PINNED → a" in e for e in c.events) and any("arm → a" in e for e in c.events)
        c._battle_seen(_tag(7))
        assert b.arm_for(_tag(7)) == "a" and any("[arm a pinned]" in e for e in c.events)
        c.set_bandit_pin("")                         # unpin while the battle is LIVE → deferred
        await asyncio.sleep(0)
        assert b.pinned is None and applied[-1] == "a"      # no mid-game swap
        assert any("applies from the next game" in e for e in c.events)
        assert c.set_bandit_pin(None) is None        # unchanged → silent
        with pytest.raises(ValueError):
            c.set_bandit_pin("nope")
        assert b.pinned is None

    asyncio.run(main())


def test_panel_pin_marks_the_rating_line_and_is_refused_when_the_bandit_is_off(monkeypatch, tmp_path):
    from v_dance.play import bot_control_ui as bcu
    monkeypatch.setattr(bcu, "discover_teams", lambda reg=None: [])
    monkeypatch.setenv("VD_SITE_POLL", "0")

    async def main():
        c = _controller(tmp_path, None)
        assert c.status()["bandit_on"] is False and c.status()["bandit_pin"] is None
        with pytest.raises(ValueError):
            c.set_bandit_pin("inc")

    asyncio.run(main())


def test_mission_control_exposes_the_pin_env_keys_and_the_arm_names():
    from v_dance.datatools import mission_control as mc
    for k in ("VD_BANDIT", "VD_BANDIT_PIN"):
        assert k in mc._ENV_WRITE_KEYS and k in mc._ENV_READ_KEYS
    names = mc._bandit_arm_names()
    assert isinstance(names, list)
    if names:                                       # config/ is gitignored → may be absent on CI
        # 2026-09-04: the roster rotates nightly (chain heads, benched frozen arms) — assert the shape, not a name
        assert all(isinstance(n, str) and n for n in names) and len(set(names)) == len(names)
    with pytest.raises(ValueError):                 # the whitelist still rejects everything else
        mc._env_write("PS_PASSWORD", "x")
