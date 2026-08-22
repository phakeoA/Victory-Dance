"""Unit tests for the pure helpers of play_vs_human_browser (the browser human-vs-AI transport).

The full flow needs a real browser + server (covered by `--self-test`); here we lock the parsing logic
that actually had a bug: challenges arrive as a |pm|/challenge frame (not |updatechallenges|), and the
tally must credit |win| to the right side."""
from __future__ import annotations

import pytest

pytest.importorskip("poke_env")

import v_dance.play.play_vs_human_browser as B

_FMT = B.BATTLE_FORMAT


def test_parse_challenge_from_pm_frame():
    frame = f"|pm| BrowserOpp| VictoryDanceAI|/challenge {_FMT}|{_FMT}|||"
    assert B._parse_challenge(frame) == ("BrowserOpp", _FMT)


def test_parse_challenge_cancelled_is_none():
    # an empty FORMAT after /challenge means the challenge was withdrawn
    assert B._parse_challenge("|pm| BrowserOpp| VictoryDanceAI|/challenge||") is None


def test_parse_challenge_from_updatechallenges_json():
    frame = '|updatechallenges|{"challengesFrom":{"someone":"%s"},"challengeTo":null}' % _FMT
    assert B._parse_challenge(frame) == ("someone", _FMT)


def test_parse_challenge_prefers_our_format_when_multiple_pending():
    frame = ('|updatechallenges|{"challengesFrom":{"alice":"gen9ou","bob":"%s"},"challengeTo":null}' % _FMT)
    assert B._parse_challenge(frame) == ("bob", _FMT)


def test_parse_challenge_ignores_non_challenge_frames():
    assert B._parse_challenge(">battle-x\n|turn|1") is None
    assert B._parse_challenge("|pm| A| B|hi there") is None


def test_result_line_is_prefix_matched_not_substring():
    assert B._result_line(">battle-x\n|win|VictoryDanceAI") == "|win|VictoryDanceAI"
    assert B._result_line(">battle-x\n|tie|") == "|tie|"
    # battle CHAT containing the literal '|win|' must NOT be read as a result (the old substring bug)
    assert B._result_line(">battle-x\n|c|user|gg |win| nice") is None
    assert B._result_line(">battle-x\n|turn|3") is None


class _FakeHost:
    def __init__(self):
        p = type("P", (), {})()
        p._battles = {}
        self.player = p


class _FakeBattle:
    def __init__(self, won):
        self.won = won


def test_credit_prefers_authoritative_battle_won_over_name():
    host = _FakeHost()
    t = {"ai": 0, "you": 0, "draw": 0}
    host.player._battles["battle-x"] = _FakeBattle(won=True)
    B._credit(host, "battle-x", "|win|SomeoneElse", t)      # name lies; battle.won=True → AI
    assert t == {"ai": 1, "you": 0, "draw": 0}
    host.player._battles["battle-y"] = _FakeBattle(won=False)
    B._credit(host, "battle-y", "|win|VictoryDanceAI", t)   # name collides; battle.won=False → you
    assert t == {"ai": 1, "you": 1, "draw": 0}


def test_credit_tie_and_name_fallback_when_no_battle_object():
    host = _FakeHost()
    t = {"ai": 0, "you": 0, "draw": 0}
    B._credit(host, "battle-x", "|tie|", t)
    assert t["draw"] == 1
    B._credit(host, "battle-z", "|win|VictoryDanceAI", t)   # no battle obj → fall back to the win name
    assert t["ai"] == 1
    B._credit(host, "battle-z", "|win|Challenger", t)
    assert t["you"] == 1


def test_toid_normalises():
    assert B._toid(" Victory Dance AI ") == "victorydanceai"
    assert B._toid("BrowserOpp") == "browseropp"


# ── AI team selection: pinned > your Teambuilder pick > random ─────────────────────
import asyncio


class _FakePage:
    """Stand-in for the AI tab: evaluate() returns whatever team is 'open' in the Teambuilder."""
    def __init__(self, open_team):
        self._open = open_team

    async def evaluate(self, *_a, **_k):
        return self._open


def test_pick_ai_team_pinned_wins():
    name, src = asyncio.run(B._pick_ai_team(_FakePage("whatever"), ["a", "b"], "a"))
    assert (name, src) == ("a", "pinned")


def test_pick_ai_team_uses_your_open_teambuilder_team():
    name, src = asyncio.run(B._pick_ai_team(_FakePage("b"), ["a", "b"], None))
    assert (name, src) == ("b", "your pick")


def test_pick_ai_team_random_when_nothing_open():
    name, src = asyncio.run(B._pick_ai_team(_FakePage(None), ["a", "b"], None))
    assert name in ("a", "b") and src == "random"


def test_pick_ai_team_ignores_open_team_not_in_pool():
    name, src = asyncio.run(B._pick_ai_team(_FakePage("not-a-loaded-team"), ["a", "b"], None))
    assert name in ("a", "b") and src == "random"


# ── BUG 2 (teams imported empty) + BUG 1 (can't challenge) regression locks ────────
import inspect


def test_team_import_uses_packTeam_not_bare_array():
    """BUG 2 lock: Storage.importTeam(paste) returns an ARRAY of sets, NOT a {name,format,team} wrapper.
    The stored team's `team` field must be the PACKED string via Storage.packTeam(sets); the old code
    tacked .name/.format onto the array → saveTeams wrote an EMPTY team → only the name imported."""
    src = inspect.getsource(B._setup_client)
    assert "Storage.packTeam(sets)" in src, "team must be packed, not the bare importTeam array"
    assert "t.format = fmt" not in src, "the old bare-array mutation must be gone"


def test_team_import_sets_capacity_6():
    """BUG 3/4 lock: the team selector's auto-pick needs `capacity === 6` — without it the picker shows
    'Select a team' (can't choose) and a stale index renders as 'Error: Corrupted team'."""
    src = inspect.getsource(B._setup_client)
    assert "capacity: 6" in src, "imported teams must carry capacity:6 to be selectable"


def test_default_format_js_defaults_format_and_wraps_challenge():
    """BUG 1 lock: the client opens on Random Battle (which the AI ignores). The fix points the format
    at ours AND wraps challenge() so a user-initiated challenge defaults to our format."""
    js = B._DEFAULT_FORMAT_JS
    assert "renderFormats(fmt)" in js                  # sets curFormat + the main-menu format button
    assert "renderTeams(fmt" in js                     # re-render the TEAM button too (else stuck "Random team")
    assert "curTeamFormat" in js and "curTeamIndex" in js  # set the internal team-format state
    assert "_vdChallengePatched" in js                 # idempotent wrap guard
    assert "format || fmt" in js                       # default a no-format challenge to ours


class _FmtPage:
    """Stand-in tab for _default_format: records evaluate() args; can raise to exercise the guards."""
    def __init__(self, eval_return, raise_on=None):
        self._eval_return = eval_return
        self._raise_on = raise_on
        self.evaluated = []

    async def wait_for_function(self, *_a, **_k):
        if self._raise_on == "wait":
            raise RuntimeError("formats never loaded")
        return True

    async def evaluate(self, js, arg=None):
        self.evaluated.append((js, arg))
        if self._raise_on == "eval":
            raise RuntimeError("evaluate boom")
        return self._eval_return


def test_default_format_success_is_quiet(capsys):
    page = _FmtPage(B.BATTLE_FORMAT)
    asyncio.run(B._default_format(page, "AI"))
    assert "WARNING" not in capsys.readouterr().out
    assert any(arg == B.BATTLE_FORMAT for _js, arg in page.evaluated)   # ran the format JS with our format


def test_default_format_warns_on_unexpected_return(capsys):
    asyncio.run(B._default_format(_FmtPage("no-home-room"), "AI"))
    assert "WARNING" in capsys.readouterr().out        # non-fatal: tells the user to pick manually


def test_default_format_warns_on_exception(capsys):
    asyncio.run(B._default_format(_FmtPage(None, raise_on="eval"), "AI"))
    assert "WARNING" in capsys.readouterr().out


def test_default_format_warns_when_formats_never_load(capsys):
    asyncio.run(B._default_format(_FmtPage(None, raise_on="wait"), "AI"))
    assert "WARNING" in capsys.readouterr().out
