"""Tests for the human-vs-AI serve loop (play_vs_human._serve).

The bug these guard: the server used to tear down the instant the FIRST battle's result was
decided (n_battles defaulted to 1), cutting off the browser animation. The fix serves ONE battle
at a time and, by default, keeps going until Ctrl-C — so the server stays up between matches.
We exercise _serve with a fake player (no Showdown server / no websocket needed)."""
from __future__ import annotations

import asyncio

import pytest

import v_dance.play.play_vs_human as P


class _FakeAI:
    """Minimal stand-in for a poke-env player: accept_challenges 'plays' one battle per call."""
    def __init__(self, raise_after: int | None = None):
        self.calls = 0
        self.n_won_battles = self.n_lost_battles = self.n_tied_battles = 0
        self._raise_after = raise_after

    async def accept_challenges(self, opponent, n):       # noqa: D401 (signature mirrors poke-env)
        self.calls += 1
        self.n_lost_battles += n                          # pretend YOU won (the AI lost) one battle
        if self._raise_after is not None and self.calls >= self._raise_after:
            raise _Stop()


class _Stop(Exception):
    pass


def test_serve_stops_after_n_finished_battles():
    # --n-battles N => accept exactly N, one battle per accept, then return.
    ai = _FakeAI()
    asyncio.run(P._serve(ai, None, 3))
    assert ai.calls == 3


def test_serve_unlimited_keeps_going_past_the_first_battle():
    # The regression: with the unlimited default the loop must NOT stop after battle 1 — it keeps
    # accepting until interrupted. We interrupt on the 4th accept to bound the test.
    ai = _FakeAI(raise_after=4)
    with pytest.raises(_Stop):
        asyncio.run(P._serve(ai, None, None))             # n_battles=None => serve until Ctrl-C
    assert ai.calls == 4                                  # looped well past 1 (the old bug stopped at 1)


def test_default_n_battles_is_unlimited():
    # The fix flips the REAL default from 1 (auto-stop after one result) to None (serve until Ctrl-C);
    # this checks the actual parser so a silent revert is caught.
    assert P._parse_args([]).n_battles is None
    assert P._parse_args(["--n-battles", "3"]).n_battles == 3
