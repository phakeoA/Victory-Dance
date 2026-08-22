"""Tests for the forced-switch infinite-loop fix (the 'no available switches —
sending Pass' flood that hangs a battle).

The fix: when a required slot has no replacement, OR poke-env re-sends the
identical forceSwitch request (because Showdown rejected our last order), send a
DefaultBattleOrder ('/choose default') which Showdown resolves and always accepts,
so a forced switch can never loop forever.
"""

from __future__ import annotations

import sys
import types
from collections import Counter
from pathlib import Path

import pytest

pytest.importorskip("poke_env")

_REPO = Path(__file__).resolve().parents[1]
from poke_env.player.battle_order import (  # noqa: E402
    DefaultBattleOrder, DoubleBattleOrder, ForfeitBattleOrder,
)
from v_dance.play.vgc_base import VGCPlayerBase  # noqa: E402


class _Concrete(VGCPlayerBase):
    def _select_actions(self, battle, state_vec):
        return 0, 0, "test"


def _root():
    obj = _Concrete.__new__(_Concrete)      # skip the networked __init__
    obj._fs_attempts = {}
    obj._fs_battle_count = {}
    return obj


def _battle(force, switches, tag="b1", turn=1):
    return types.SimpleNamespace(
        force_switch=list(force),
        available_switches=[list(s) for s in switches],
        battle_tag=tag, turn=turn,
    )


def _mon(species="incineroar"):
    return types.SimpleNamespace(species=species)


# ── no-candidate required slot → default escape ───────────────────────────────
def test_no_replacement_for_required_slot_returns_default():
    # slot 0 MUST switch but has no available replacement → a Pass would loop.
    order = _root()._handle_force_switch(_battle([True, False], [[], []]))
    assert isinstance(order, DefaultBattleOrder)


def test_double_faint_with_one_mon_escapes_to_default():
    # both slots must switch, only ONE bench mon (offered to both): slot 0 takes
    # it, slot 1 has no candidate after dedup → default (Showdown legally resolves
    # the switch + pass) instead of a hand-built order.
    m = _mon()
    order = _root()._handle_force_switch(_battle([True, True], [[m], [m]]))
    assert isinstance(order, DefaultBattleOrder)


def test_normal_force_switch_builds_double_order():
    order = _root()._handle_force_switch(_battle([True, False], [[_mon()], []]))
    assert isinstance(order, DoubleBattleOrder)
    assert not isinstance(order, DefaultBattleOrder)


# ── loop guard: identical re-request → default ────────────────────────────────
def test_loop_guard_defaults_after_repeated_identical_request():
    p = _root()
    b = _battle([True, False], [[_mon()], []])
    o1 = p._handle_force_switch(b)           # attempt 1 → build
    o2 = p._handle_force_switch(b)           # attempt 2 → build
    o3 = p._handle_force_switch(b)           # attempt 3 → was rejected → default
    assert isinstance(o1, DoubleBattleOrder)
    assert isinstance(o2, DoubleBattleOrder)
    assert isinstance(o3, DefaultBattleOrder)


def test_loop_guard_keyed_by_turn_so_a_new_turn_builds_again():
    p = _root()
    for _ in range(3):                        # trip the guard on turn 1
        p._handle_force_switch(_battle([True, False], [[_mon()], []], turn=1))
    fresh = p._handle_force_switch(_battle([True, False], [[_mon()], []], turn=2))
    assert isinstance(fresh, DoubleBattleOrder)   # different turn → not stuck


def test_forfeit_backstop_breaks_a_turn_shifting_loop():
    # A loop where the turn shifts each iteration defeats the per-request guard;
    # the per-BATTLE counter still trips and FORFEITS so the battle can't hang.
    p = _root()
    order = None
    for t in range(1, 40):                    # distinct turn each call (same battle)
        order = p._handle_force_switch(_battle([True, False], [[_mon()], []], turn=t))
        if isinstance(order, ForfeitBattleOrder):
            break
    assert isinstance(order, ForfeitBattleOrder)


# ── spliced (model-player) handler shares the same guard ──────────────────────
def test_spliced_handler_guard_short_circuits_before_encoding():
    from v_dance.play.live_vgc_base import SplicingVGCPlayerBase

    class _C(SplicingVGCPlayerBase):
        def _select_actions(self, battle, state_vec):
            return 0, 0, "test"

    obj = _C.__new__(_C)
    b = _battle([True, False], [[_mon()], []], turn=1)
    key = (b.battle_tag, b.turn, (True, False))
    obj._fs_attempts = {key: 2}               # pre-trip the per-request guard (n>=2)
    obj._fs_battle_count = {}
    obj._source_counts = Counter()
    order = obj._handle_force_switch(b)        # must NOT reach the encoder
    assert isinstance(order, DefaultBattleOrder)
    assert obj._source_counts["forced_switch_escape"] == 1


# ── fs-monitor: forfeit is counted SEPARATELY from a /choose default escape ────
def _spliced_player():
    from v_dance.play.live_vgc_base import SplicingVGCPlayerBase

    class _C(SplicingVGCPlayerBase):
        def _select_actions(self, battle, state_vec):
            return 0, 0, "test"

    obj = _C.__new__(_C)
    obj._fs_attempts = {}
    obj._fs_battle_count = {}
    obj._source_counts = Counter()
    return obj


def test_forfeit_escape_counts_forfeit_not_switch_escape():
    obj = _spliced_player()
    obj._force_switch_escape = lambda battle: ForfeitBattleOrder()   # force the FORFEIT outcome
    order = obj._handle_force_switch(_battle([True, False], [[_mon()], []]))
    assert isinstance(order, ForfeitBattleOrder)
    assert obj._source_counts["forfeit"] == 1
    assert obj._source_counts["forced_switch_escape"] == 0           # not double-counted


def test_default_escape_counts_switch_escape_not_forfeit():
    obj = _spliced_player()
    obj._force_switch_escape = lambda battle: DefaultBattleOrder()   # the /choose default outcome
    order = obj._handle_force_switch(_battle([True, False], [[_mon()], []]))
    assert isinstance(order, DefaultBattleOrder)
    assert obj._source_counts["forced_switch_escape"] == 1
    assert obj._source_counts["forfeit"] == 0


def test_forfeit_counted_once_per_battle_even_on_re_request():
    # A server can re-request the SAME battle's forceSwitch before it processes our /forfeit, re-
    # entering the escape. The forfeit tally is per-BATTLE, so 3 re-requests of one tag = 1 forfeit.
    obj = _spliced_player()
    obj._force_switch_escape = lambda battle: ForfeitBattleOrder()
    b = _battle([True, False], [[_mon()], []], tag="bX")
    for _ in range(3):
        assert isinstance(obj._handle_force_switch(b), ForfeitBattleOrder)
    assert obj._source_counts["forfeit"] == 1                       # one forfeited battle, not 3
    # a DIFFERENT battle tag forfeits independently
    obj._handle_force_switch(_battle([True, False], [[_mon()], []], tag="bY"))
    assert obj._source_counts["forfeit"] == 2


# ── normal-turn retry-storm exhaustion detection ──────────────────────────────
def test_has_fresh_legal_detects_exhaustion(monkeypatch):
    import v_dance.play.live_vgc_base as L
    # mask: actions 0 and 2 legal, 1 illegal
    monkeypatch.setattr(L, "build_legal_action_mask", lambda b, s: [True, False, True])
    SP = L.SplicingVGCPlayerBase
    assert SP._has_fresh_legal(None, 0, set()) is True       # nothing tried yet
    assert SP._has_fresh_legal(None, 0, {0}) is True         # action 2 still fresh
    assert SP._has_fresh_legal(None, 0, {0, 2}) is False      # both legal tried → exhausted


def test_active_empty_mask_forced_move_vs_normal(monkeypatch):
    """An active, non-fainted slot with an EMPTY mask (its only usable order is a
    non-representable forced move — Struggle / recharge / 2-turn continuation) must be
    detected so the turn goes to /choose default instead of an illegal Pass; a normal
    (non-empty) mask must NOT trigger it, and a fainted slot is ignored."""
    from v_dance.play import vgc_base as VB
    mon0 = types.SimpleNamespace(fainted=False, species="sylveon")
    battle = types.SimpleNamespace(active_pokemon=[mon0, None])

    monkeypatch.setattr(VB, "build_legal_action_mask", lambda b, s: [False] * 16)
    assert VB.VGCPlayerBase._active_empty_mask(battle) is True             # empty -> forced-move

    monkeypatch.setattr(VB, "build_legal_action_mask", lambda b, s: [True] + [False] * 15)
    assert VB.VGCPlayerBase._active_empty_mask(battle) is False            # has a legal action

    mon0.fainted = True                                                    # fainted slot ignored
    monkeypatch.setattr(VB, "build_legal_action_mask", lambda b, s: [False] * 16)
    assert VB.VGCPlayerBase._active_empty_mask(battle) is False


# ── #4-ext: ROOT-lineage (scripted/random) opponents record backstop-forfeits ─────────────────
def test_root_handle_force_switch_records_forfeit_tag():
    """A ROOT-lineage player's _handle_force_switch records the battle tag in _forfeited_tags when the
    loop-guard forfeits, so the self-play recorder can FALLBACK-discard a battle the OPPONENT forfeited
    (scripted max_damage/heuristic opps use this root path; the Splicing handler already did this)."""
    p = _Concrete.__new__(_Concrete)
    p._force_switch_escape = lambda b: ForfeitBattleOrder()
    battle = types.SimpleNamespace(battle_tag=">battle-x")
    order = p._handle_force_switch(battle)
    assert isinstance(order, ForfeitBattleOrder)
    assert getattr(p, "_forfeited_tags", None) == {"battle-x"}      # leading '>' stripped (== _norm_tag)
    p._handle_force_switch(battle)                                  # re-request for the SAME battle
    assert p._forfeited_tags == {"battle-x"}                        # idempotent — no double-add


def test_root_handle_force_switch_non_forfeit_records_nothing():
    """A non-forfeit escape (/choose default) must NOT populate _forfeited_tags."""
    p = _Concrete.__new__(_Concrete)
    p._force_switch_escape = lambda b: DefaultBattleOrder()
    order = p._handle_force_switch(types.SimpleNamespace(battle_tag="battle-y"))
    assert isinstance(order, DefaultBattleOrder)
    assert getattr(p, "_forfeited_tags", None) in (None, set())
