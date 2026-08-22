"""Mask-2 (the _safe_order codec root-cause, candidate 3): when ONE active slot is forced into a
SYNTHETIC move the 16-action codec can't express (recharge after Hyper Beam/Giga Impact/…, or a trapped
Struggle) WHILE the OTHER slot has a genuine choice, the turn must NOT collapse to /choose default (which
plays the free slot RANDOMLY = policy-destructive). The forced slot plays its forced move; the partner
slot keeps the model's decision. Only when EVERY active slot is forced is /choose default correct.

Verified gap: adversarial workflow wf_63c1ab6a-04d (refuted the "no gap" hypothesis on exactly this case).
"""
from __future__ import annotations

import types
from collections import Counter

import pytest

pytest.importorskip("numpy")
pytest.importorskip("poke_env")


def _make_partial_forced_battle(both_forced=False):
    """Slot 0 is recharging (mon.moves holds the real move Hyper Beam, but available_moves[0] holds only
    the synthetic 'recharge' → empty codec mask; recharge traps → no switch). Slot 1 has a usable move
    (Close Combat) → a real choice. both_forced=True makes slot 1 recharge too (no free slot)."""
    from poke_env.battle import Move
    rc = Move("recharge", gen=9)
    hb = Move("hyperbeam", gen=9)
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

    actor0 = _Mon("rech0", {"hyperbeam": hb})                       # recharging (real move in mon.moves)
    actor1 = (_Mon("rech1", {"hyperbeam": hb}) if both_forced
              else _Mon("free1", {"closecombat": cc}))
    foe = _Mon("foe_a", {})

    class _Battle:
        OPPONENT_1_POSITION = 1
        OPPONENT_2_POSITION = 2
        POKEMON_1_POSITION = -1
        POKEMON_2_POSITION = -2
        def __init__(self):
            self.active_pokemon = [actor0, actor1]
            self.opponent_active_pokemon = [foe, None]
            self.available_moves = [[rc], [rc] if both_forced else [cc]]
            self.available_switches = [[], []]
            self.team = {}
            self.can_mega_evolve = [False, False]
            self.reviving = False
            self.trapped = [True, True if both_forced else False]   # recharge traps
            self.force_switch = [False, False]
            self.player_role = None
            self.battle_tag = "battle-partial-forced"
            self.turn = 7
        def to_showdown_target(self, mv, tgt):
            return 1 if getattr(tgt, "tag", None) == "foe_a" else 2

    return _Battle()


def _no_bench(monkeypatch):
    import v_dance.play.vgc_base as vb
    monkeypatch.setattr(vb, "own_bench_mons", lambda battle: [])


# ── helper ──────────────────────────────────────────────────────────────────
def test_forced_single_order_builds_recharge():
    from v_dance.play.vgc_base import VGCPlayerBase
    battle = _make_partial_forced_battle()
    o = VGCPlayerBase._forced_single_order(battle, 0)
    assert o is not None and "move" in o.message.lower()        # the recharge synthetic move
    # an empty available_moves slot (Commander / no action) → None (caller defaults)
    battle.available_moves[0] = []
    assert VGCPlayerBase._forced_single_order(battle, 0) is None


# ── root (scripted/gauntlet model) path ───────────────────────────────────────
def test_root_choose_move_preserves_free_partner(monkeypatch):
    import v_dance.play.vgc_base as VB
    from poke_env.player.battle_order import DefaultBattleOrder, DoubleBattleOrder
    battle = _make_partial_forced_battle()
    _no_bench(monkeypatch)
    monkeypatch.setattr(VB, "DoubleBattle", type(battle))

    class _Concrete(VB.VGCPlayerBase):
        def _select_actions(self, b, state_vec):
            return 0, 0, "model"                               # slot1 → Close Combat on the foe
        def _select_gimmicks(self, b, state_vec, a0, a1):
            return 0, 0
    p = _Concrete.__new__(_Concrete)
    p._encoder = types.SimpleNamespace(encode=lambda b: None)
    p._replay = types.SimpleNamespace(record=lambda **kw: None)

    order = p.choose_move(battle)
    assert isinstance(order, DoubleBattleOrder)                 # NOT a whole-turn /choose default
    assert not isinstance(order, DefaultBattleOrder)
    assert "closecombat" in order.message.lower()              # the free partner's model move survives


def test_root_both_forced_still_defaults(monkeypatch):
    import v_dance.play.vgc_base as VB
    from poke_env.player.battle_order import DefaultBattleOrder
    battle = _make_partial_forced_battle(both_forced=True)
    _no_bench(monkeypatch)
    monkeypatch.setattr(VB, "DoubleBattle", type(battle))

    class _Concrete(VB.VGCPlayerBase):
        def _select_actions(self, b, state_vec):
            return 0, 0, "model"
        def _select_gimmicks(self, b, state_vec, a0, a1):
            return 0, 0
    p = _Concrete.__new__(_Concrete)
    p._encoder = types.SimpleNamespace(encode=lambda b: None)
    p._replay = types.SimpleNamespace(record=lambda **kw: None)

    order = p.choose_move(battle)
    assert isinstance(order, DefaultBattleOrder)                # no free slot → proven escape unchanged


# ── live (model / self-play) path ─────────────────────────────────────────────
def test_live_choose_move_preserves_free_partner(monkeypatch):
    import v_dance.play.live_vgc_base as LV
    from poke_env.player.battle_order import DefaultBattleOrder, DoubleBattleOrder
    battle = _make_partial_forced_battle()
    _no_bench(monkeypatch)
    monkeypatch.setattr(LV, "DoubleBattle", type(battle))

    discarded = []

    class _C(LV.SplicingVGCPlayerBase):
        def _select_actions(self, b, state_vec):
            return 0, 0, "model"
        def _select_gimmicks(self, b, state_vec, a0, a1):
            return 0, 0
        def _discard_rl_decision(self, b, decision_type):
            discarded.append(decision_type)
    p = _C.__new__(_C)
    p._encoder = types.SimpleNamespace(encode=lambda b, opp_snapshot=None: None)
    p._replay = types.SimpleNamespace(record=lambda **kw: None)
    p._proto_log = {}
    p._tried_actions = {}
    p._forced_aware_attempts = {}
    p._last_error = {}
    p._source_counts = Counter()

    order = p.choose_move(battle)
    assert isinstance(order, DoubleBattleOrder)                 # free partner preserved
    assert not isinstance(order, DefaultBattleOrder)
    assert "closecombat" in order.message.lower()
    assert p._source_counts["forced_partial"] == 1             # new telemetry: model drove the free slot
    assert p._source_counts["forced_default"] == 0
    assert discarded == ["turn"]                               # forced slot unrecordable → step dropped


def test_live_both_forced_still_defaults(monkeypatch):
    import v_dance.play.live_vgc_base as LV
    from poke_env.player.battle_order import DefaultBattleOrder
    battle = _make_partial_forced_battle(both_forced=True)
    _no_bench(monkeypatch)
    monkeypatch.setattr(LV, "DoubleBattle", type(battle))

    class _C(LV.SplicingVGCPlayerBase):
        def _select_actions(self, b, state_vec):
            return 0, 0, "model"
        def _select_gimmicks(self, b, state_vec, a0, a1):
            return 0, 0
        def _discard_rl_decision(self, b, decision_type):
            pass
    p = _C.__new__(_C)
    p._encoder = types.SimpleNamespace(encode=lambda b, opp_snapshot=None: None)
    p._replay = types.SimpleNamespace(record=lambda **kw: None)
    p._proto_log = {}
    p._tried_actions = {}
    p._forced_aware_attempts = {}
    p._last_error = {}
    p._source_counts = Counter()

    order = p.choose_move(battle)
    assert isinstance(order, DefaultBattleOrder)                # both forced → /choose default
    assert p._source_counts["forced_default"] == 1
    assert p._source_counts["forced_partial"] == 0
