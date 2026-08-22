"""B-L1 adapt-rules unit tests (Phase-3 design L1): streak parser, bias vector layout,
bc_action_indices bias path + None-bias byte-identity."""
import numpy as np
import pytest

from v_dance.encoders.encoder_layout import ACTIONS_PER_SLOT
from v_dance.play.adapt_rules import (
    SPREAD_BIAS, spread_bias_for_kinds, wide_guard_streak,
)


def _turn(n):
    return f"|turn|{n}"


def _mv(role_slot, move):
    return f"|move|{role_slot}: Mon|{move}|p1a: Target"


class TestWideGuardStreak:
    def test_two_consecutive_turns(self):
        # turns 1,2 completed with p2 Wide Guard; |turn|3 = deciding now
        lines = [_turn(1), _mv("p2a", "Wide Guard"),
                 _turn(2), _mv("p2b", "Wide Guard"),
                 _turn(3)]
        assert wide_guard_streak(lines, "p2") == 2

    def test_broken_streak_resets(self):
        lines = [_turn(1), _mv("p2a", "Wide Guard"),
                 _turn(2), _mv("p2a", "Protect"),          # plain Protect ≠ wide blocker
                 _turn(3), _mv("p2a", "Wide Guard"),
                 _turn(4)]
        assert wide_guard_streak(lines, "p2") == 1

    def test_own_side_ignored(self):
        lines = [_turn(1), _mv("p1a", "Wide Guard"),
                 _turn(2), _mv("p1a", "Wide Guard"),
                 _turn(3)]
        assert wide_guard_streak(lines, "p2") == 0

    def test_current_turn_excluded_and_empty(self):
        # Wide Guard only on the CURRENT (still-deciding) turn's |turn| marker turn → streak 0
        assert wide_guard_streak([_turn(1)], "p2") == 0
        assert wide_guard_streak(None, "p2") == 0
        assert wide_guard_streak([], "p2") == 0

    def test_mat_block_counts(self):
        lines = [_turn(1), _mv("p2a", "Mat Block"),
                 _turn(2), _mv("p2a", "Wide Guard"),
                 _turn(3)]
        assert wide_guard_streak(lines, "p2") == 2


class TestSpreadBias:
    def test_spread_bucket_indices(self):
        kinds = ["allAdjacentFoes", "normal", "allAdjacent", "self"]
        arr = spread_bias_for_kinds(kinds)
        assert arr is not None and arr.shape == (ACTIONS_PER_SLOT,)
        assert arr[0 * 3] == pytest.approx(SPREAD_BIAS)     # move 0 spread bucket
        assert arr[2 * 3] == pytest.approx(SPREAD_BIAS)     # move 2 spread bucket
        assert np.count_nonzero(arr) == 2                    # nothing else touched

    def test_no_spread_moves_is_none(self):
        assert spread_bias_for_kinds(["normal", "self", "adjacentAlly", None]) is None


class TestBiasPath:
    def _stub(self, monkeypatch, l0, l1):
        import v_dance.play.model_io as m
        monkeypatch.setattr(m, "head_logits", lambda *a, **k: (np.array(l0, dtype=np.float32),
                                                               np.array(l1, dtype=np.float32)))
        return m

    def test_none_bias_identity_and_tilt_flips(self, monkeypatch):
        # Two near-tie legal actions: 0 (spread bucket) barely beats 1. The tilt must flip
        # the argmax to 1; bias=None must reproduce the original pick exactly.
        l = [0.0] * ACTIONS_PER_SLOT
        l[0], l[1] = 1.0, 0.9
        m = self._stub(monkeypatch, l, l)
        mask = [True, True] + [False] * (ACTIONS_PER_SLOT - 2)
        a0, a1 = m.bc_action_indices(None, ("our_a", "our_b"), np.zeros(4), mask, mask)
        assert (a0, a1) == (0, 0)                            # None-bias = original argmax
        bias = np.zeros(ACTIONS_PER_SLOT, dtype=np.float32)
        bias[0] = SPREAD_BIAS
        b0, b1 = m.bc_action_indices(None, ("our_a", "our_b"), np.zeros(4), mask, mask,
                                     bias0=bias, bias1=None)
        assert b0 == 1                                       # tilted slot flips off the spread pick
        assert b1 == 0                                       # un-tilted slot unchanged

    def test_overwhelming_preference_survives_tilt(self, monkeypatch):
        # Design contract: a tilt, not a mask — a strongly-preferred spread move still wins.
        l = [0.0] * ACTIONS_PER_SLOT
        l[0], l[1] = 5.0, 0.5                                # spread hugely preferred
        m = self._stub(monkeypatch, l, l)
        mask = [True, True] + [False] * (ACTIONS_PER_SLOT - 2)
        bias = np.zeros(ACTIONS_PER_SLOT, dtype=np.float32)
        bias[0] = SPREAD_BIAS
        a0, _ = m.bc_action_indices(None, ("our_a", "our_b"), np.zeros(4), mask, mask,
                                    bias0=bias, bias1=bias)
        assert a0 == 0
