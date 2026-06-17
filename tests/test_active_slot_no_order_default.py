"""Regression: the live "Can't pass: Your <mon> must make a move (or switch)"
rejection — an ACTIVE, non-fainted slot whose ONLY codec-expressible legal action
is a switch the OTHER slot is already taking.

THE BUG (user's overnight run, gen 1, Sneasler):
  Sneasler's only Showdown-usable move was one NOT in ``mon.moves`` (a forced move
  — Struggle / recharge), so ``available_moves[slot]`` was non-empty but the
  16-action MOVE mask was empty.  Its sole legal action was the single bench switch.
  Slot 0 took that same switch first, so the cross-slot dedup refused slot 1's switch
  (``action_to_order`` → None) and ``_safe_order``'s fallback scan skipped it (the only
  legal mask action *is* the rejected action) → it fell through to ``PassBattleOrder``.
  Passing an ACTIVE mon is illegal, so Showdown rejected the order → retry-storm +
  "the model did NOT drive this slot".

  ``_active_empty_mask`` did NOT catch it: the mask was NON-empty (it held the switch);
  the unusability only emerges once slot 0's order is known.

THE FIX:
  ``_safe_order`` returns ``None`` (not a Pass) for an active non-fainted slot with no
  representable legal order; both ``choose_move`` bodies then resolve the whole turn
  with ``/choose default`` — the cross-slot variant of the empty-mask escape.
"""

from __future__ import annotations

import types
from collections import Counter

import pytest

pytest.importorskip("numpy")
pytest.importorskip("poke_env")


# ── Fake board: one bench mon both slots want; slot 1 cannot move ──────────────
def _make_collision_battle():
    """Two living actives + one visible foe.  Slot 0 has a usable move; slot 1 has
    NO usable move (forced-Struggle: its move-mask is empty) so its only legal action
    is the single bench switch — which slot 0 is also taking.  ``own_bench_mons`` /
    ``own_switch_slot`` are monkeypatched by the caller to a SINGLE bench mon → slot 3."""
    from poke_env.battle import Move
    cc = Move("closecombat", gen=9)

    class _Mon:
        def __init__(self, tag, moves):
            self.tag = tag
            self.fainted = False
            self.species = tag
            self.base_species = tag
            self._mv = moves
        @property
        def moves(self):
            return self._mv

    actor0 = _Mon("actor0", {"closecombat": cc})
    actor1 = _Mon("actor1", {})            # no usable move (forced move not in mon.moves)
    foe = _Mon("foe_a", {})
    bench0 = _Mon("bench0", {})

    class _Battle:
        OPPONENT_1_POSITION = 1
        OPPONENT_2_POSITION = 2
        POKEMON_1_POSITION = -1
        POKEMON_2_POSITION = -2
        def __init__(self):
            self.active_pokemon = [actor0, actor1]
            self.opponent_active_pokemon = [foe, None]
            self.available_moves = [[cc], []]          # slot 1: no usable move
            self.available_switches = [[], []]
            self.team = {}
            self.can_mega_evolve = [False, False]
            self.reviving = False
            self.trapped = [False, False]
            self.force_switch = [False, False]
            self.player_role = None
            self.battle_tag = "battle-collision-1"
            self.turn = 5
        def to_showdown_target(self, mv, tgt):
            return 1 if getattr(tgt, "tag", None) == "foe_a" else 2

    return _Battle(), bench0


def _patch_single_bench(monkeypatch, bench0):
    """own_bench_mons → [bench0] (ONE mon); own_switch_slot → team slot 3."""
    import v_dance.play.vgc_base as vb
    monkeypatch.setattr(vb, "own_bench_mons", lambda battle: [bench0])
    monkeypatch.setattr(vb, "own_switch_slot",
                        lambda battle, mon: 3 if mon is bench0 else None)


# ══════════════════════════════════════════════════════════════════════════════
# CORE — _safe_order returns None (not a Pass) for the active-mon collision
# ══════════════════════════════════════════════════════════════════════════════
def test_safe_order_returns_none_when_only_switch_collides_and_no_move(monkeypatch):
    from v_dance.play.vgc_base import (
        VGCPlayerBase, action_to_order, build_legal_action_mask, _switch_order_target,
    )
    battle, bench0 = _make_collision_battle()
    _patch_single_bench(monkeypatch, bench0)

    # Slot 1's ONLY legal action is switch index 12 (no usable move, one bench mon).
    mask1 = build_legal_action_mask(battle, 1)
    assert [i for i, ok in enumerate(mask1) if ok] == [12]

    # Slot 0 takes that single switch ("/choose switch 3").
    order_s0 = action_to_order(12, battle, 0)
    assert _switch_order_target(order_s0) == "/choose switch 3"
    taken = {_switch_order_target(order_s0)}

    # Slot 1's only action collides → no representable order → None (NOT a Pass).
    order_s1 = VGCPlayerBase._safe_order(None, 12, battle, 1, taken_switch_targets=taken)
    assert order_s1 is None


def test_safe_order_still_passes_a_fainted_slot(monkeypatch):
    """The None path is for an ACTIVE non-fainted slot ONLY.  A fainted/empty slot
    with no decodable action is a legitimate no-op → still a PassBattleOrder, so a
    genuine no-op slot never escalates the whole turn to /choose default."""
    from poke_env.player.battle_order import PassBattleOrder
    from v_dance.play.vgc_base import VGCPlayerBase
    battle, bench0 = _make_collision_battle()
    _patch_single_bench(monkeypatch, bench0)
    battle.active_pokemon[1].fainted = True        # slot 1 is gone

    order_s1 = VGCPlayerBase._safe_order(None, 12, battle, 1,
                                         taken_switch_targets={"/choose switch 3"})
    assert isinstance(order_s1, PassBattleOrder)


def test_safe_order_returns_order_when_an_alternative_exists(monkeypatch):
    """Negative control: the None path must NOT over-fire.  If slot 1 has ANY other
    legal action (here: a usable move), _safe_order returns it, never None."""
    from v_dance.play.vgc_base import VGCPlayerBase
    battle, bench0 = _make_collision_battle()
    _patch_single_bench(monkeypatch, bench0)
    # give slot 1 a usable move so it has an alternative to the colliding switch
    from poke_env.battle import Move
    cc = Move("closecombat", gen=9)
    battle.active_pokemon[1]._mv = {"closecombat": cc}
    battle.available_moves[1] = [cc]

    order_s1 = VGCPlayerBase._safe_order(None, 12, battle, 1,
                                         taken_switch_targets={"/choose switch 3"})
    assert order_s1 is not None
    assert "closecombat" in order_s1.message.lower()


# ══════════════════════════════════════════════════════════════════════════════
# INTEGRATION — choose_move resolves the whole turn with /choose default
# ══════════════════════════════════════════════════════════════════════════════
def test_root_choose_move_defaults_on_active_slot_collision(monkeypatch):
    """Root path (scripted / random players): both slots pick the single bench switch,
    slot 1 has no other legal action → choose_move returns DefaultBattleOrder (NOT a
    DoubleBattleOrder that would carry an illegal Pass for the active slot 1)."""
    import v_dance.play.vgc_base as VB
    from poke_env.player.battle_order import DefaultBattleOrder, DoubleBattleOrder
    battle, bench0 = _make_collision_battle()
    _patch_single_bench(monkeypatch, bench0)
    monkeypatch.setattr(VB, "DoubleBattle", type(battle))   # fake passes the isinstance guard

    class _Concrete(VB.VGCPlayerBase):
        def _select_actions(self, b, state_vec):
            return 12, 12, "model"                          # both want switch index 12
    p = _Concrete.__new__(_Concrete)
    p._encoder = types.SimpleNamespace(encode=lambda b: None)
    p._replay = types.SimpleNamespace(record=lambda **kw: None)

    order = p.choose_move(battle)
    assert isinstance(order, DefaultBattleOrder)
    assert not isinstance(order, DoubleBattleOrder)


def test_live_choose_move_defaults_and_discards_on_active_slot_collision(monkeypatch):
    """Live (model / self-play) path: same collision → DefaultBattleOrder, the
    'forced_default' source is counted, and the rejected step is NOT recorded as a
    model decision (no retry-storm, no illegal Pass)."""
    import v_dance.play.live_vgc_base as LV
    from poke_env.player.battle_order import DefaultBattleOrder, DoubleBattleOrder
    battle, bench0 = _make_collision_battle()
    _patch_single_bench(monkeypatch, bench0)
    monkeypatch.setattr(LV, "DoubleBattle", type(battle))   # fake passes the isinstance guard

    discarded = []

    class _C(LV.SplicingVGCPlayerBase):
        def _select_actions(self, b, state_vec):
            return 12, 12, "model"
        def _discard_rl_decision(self, b, decision_type):
            discarded.append(decision_type)
    p = _C.__new__(_C)
    p._encoder = types.SimpleNamespace(encode=lambda b, opp_snapshot=None: None)
    p._replay = types.SimpleNamespace(record=lambda **kw: None)
    p._proto_log = {}
    p._tried_actions = {}
    p._last_error = {}
    p._source_counts = Counter()

    order = p.choose_move(battle)
    assert isinstance(order, DefaultBattleOrder)
    assert not isinstance(order, DoubleBattleOrder)
    assert p._source_counts["forced_default"] == 1
    assert discarded == ["turn"]                            # rejected step dropped, not recorded
