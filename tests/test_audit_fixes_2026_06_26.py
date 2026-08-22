"""Regression tests for the 2026-06-26 project audit fixes (the behavioural ones).

Locks: #21 (model_driven_fraction must NOT count bookkeeping `rejected_resample`/`abandon_forfeit`
in its denominator) and #01 (discount_forfeits subtracts backstop-forfeits, clamped non-negative).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from v_dance.selfplay.reward import model_driven_fraction, assert_model_driven
from v_dance.play.parallel_battles import discount_forfeits


# ── #21: rejected_resample / abandon_forfeit are bookkeeping, not non-model decisions ──
def test_resample_burst_does_not_deflate_model_driven():
    # `model` is double-counted on a resample (rejected + accepted call), and rejected_resample
    # tallies the rejection. A pure-resample burst is 100% model-driven.
    M, N = 90, 50
    assert model_driven_fraction({"model": M + 2 * N, "rejected_resample": N}) == 1.0
    # abandon_forfeit (watchdog bookkeeping) is likewise excluded
    assert model_driven_fraction({"model": 100, "abandon_forfeit": 7}) == 1.0


def test_real_non_model_sources_still_deflate_and_trip_the_guard():
    # a genuine fallback/escape decision still counts in the denominator
    assert model_driven_fraction({"model": 95, "forced_switch": 5}) == pytest.approx(0.95)
    assert model_driven_fraction({"model": 90, "forced_default": 10}) == pytest.approx(0.90)
    with pytest.raises(AssertionError, match="MODEL-DRIVEN"):
        assert_model_driven({"model": 95, "forced_switch": 5}, 0.99)


# ── #01: shared backstop-forfeit discount ─────────────────────────────────────
def _player(tags):
    return SimpleNamespace(_forfeited_tags=set(tags))


def test_discount_forfeits_subtracts_opponent_and_own():
    model = _player(["a"])          # we forfeited 1 (loop-guard/abandon)
    opp = _player(["b", "c"])       # opponent forfeited 2 -> 2 spurious "wins" for us
    # raw chunk: 5 wins / 10 finished
    w, f = discount_forfeits(5, 10, model, opp)
    assert w == 5 - 2          # opp forfeits removed from wins
    assert f == 10 - 2 - 1     # opp + own forfeits removed from finished


def test_discount_forfeits_clamps_non_negative():
    model = _player([])
    opp = _player(["x", "y", "z"])  # more opp forfeits than reflected wins (stall-abandon window)
    w, f = discount_forfeits(1, 2, model, opp)
    assert w == 0 and f == 0       # clamped, never negative into the gate


def test_discount_forfeits_handles_missing_attr():
    bare = SimpleNamespace()        # no _forfeited_tags at all
    assert discount_forfeits(4, 6, bare, bare) == (4, 6)


def test_discount_forfeits_shared_tag_counts_once():
    # #0 regression: a watchdog-abandoned battle is tagged on BOTH players' sets. It must be subtracted
    # ONCE from finished (cancelling the +abandoned_n fold), and NEVER from wins (an abandon is no win).
    model = _player(["t1", "t2"])
    opp = _player(["t1", "t2"])
    w, f = discount_forfeits(6, 12, model, opp)
    assert w == 6      # opp-exclusive = {} -> wins untouched
    assert f == 10     # union = {t1,t2} -> finished -2 (NOT -4 as the old len+len did)


def test_discount_forfeits_mixed_shared_and_exclusive():
    model = _player(["t1"])            # t1 abandoned (also on opp)
    opp = _player(["t1", "x"])         # t1 (both) + x = opp loop-guard forfeit (our spurious win)
    w, f = discount_forfeits(5, 9, model, opp)
    assert w == 4      # opp-exclusive = {x} -> w-1
    assert f == 7      # union = {t1, x} -> f-2


# ── #3: build_train_configs precedence (CLI > --config > launcher default) ─────
def test_build_train_configs_kl_precedence():
    pytest.importorskip("torch")
    from v_dance.selfplay.generation import build_train_configs
    # bare run -> LAUNCHER defaults (kl 0.5 / target_kl 0.15), NOT the PPOConfig/TrainConfig defaults (0.0/None)
    ppo, tr = build_train_configs()
    assert ppo.kl_coef == 0.5 and tr.target_kl_from_bc == 0.15
    # a --config value is HONORED when the CLI flag is unset (the #3 bug: it used to be clobbered)
    ppo, tr = build_train_configs(ppo_overrides={"kl_coef": 0.2}, train_overrides={"target_kl_from_bc": 0.3})
    assert ppo.kl_coef == 0.2 and tr.target_kl_from_bc == 0.3
    # explicit CLI value beats the --config value
    ppo, tr = build_train_configs(kl_coef=0.7, target_kl_bc=0.25,
                                  ppo_overrides={"kl_coef": 0.2}, train_overrides={"target_kl_from_bc": 0.3})
    assert ppo.kl_coef == 0.7 and tr.target_kl_from_bc == 0.25
    # an explicit non-positive target_kl_bc disables the guard
    _, tr = build_train_configs(target_kl_bc=0.0)
    assert tr.target_kl_from_bc is None
