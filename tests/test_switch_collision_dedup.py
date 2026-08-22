"""Regression: the doubles "[Invalid choice] Can't switch: The Pokémon in slot N
can only switch in once" rejection on a NORMAL turn.

own_bench_mons is slot-INDEPENDENT, so switch index 12+i decodes to the SAME
benched mon for BOTH active slots.  Two slots can therefore emit the same
``/choose switch`` order via TWO independent routes that the action-level
``_select_actions`` dedup (a0==a1) cannot both catch:

  1. PERTURBATION — the live retry branch re-picks each slot independently
     (``_fresh_legal``), so both can land on the same switch index.
  2. UNDER-ILLUSION FALLBACK — both slots pick MOVE actions that decode to None
     (disabled / choice-locked / no-visible-foe), so each ``_safe_order`` runs its
     level-2 scan independently and both substitute the FIRST legal switch
     (index 12 → bench[0]) → the same team slot.

The fix is an ORDER-LEVEL dedup: slot-0's resolved ``/choose switch`` command is
threaded into slot-1's ``_safe_order`` (both its primary decode AND its fallback
scan), so slot 1 can never re-use slot 0's switch.  Robust to the Illusion
name-vs-slot alias because the collision is judged on the resolved switch command.
"""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")
pytest.importorskip("poke_env")


# ── Fake board ────────────────────────────────────────────────────────────────
def _make_switch_battle(slot1_can_move: bool):
    """Two living actives + a single visible foe.  Slot 0 always has a usable move;
    slot 1's usability is controlled by ``slot1_can_move`` (False → its only orders
    are switches, the under-illusion case).  Bench mons + their team slots are
    supplied by monkeypatching own_bench_mons / own_switch_slot in the test."""
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
    actor1 = _Mon("actor1", {"closecombat": cc} if slot1_can_move else {})
    foe = _Mon("foe_a", {})
    bench0, bench1 = _Mon("bench0", {}), _Mon("bench1", {})

    class _Battle:
        OPPONENT_1_POSITION = 1
        OPPONENT_2_POSITION = 2
        POKEMON_1_POSITION = -1
        POKEMON_2_POSITION = -2
        def __init__(self):
            self.active_pokemon = [actor0, actor1]
            self.opponent_active_pokemon = [foe, None]
            self.available_moves = [[cc], [cc] if slot1_can_move else []]
            self.available_switches = [[], []]
            self.team = {}
            self.can_mega_evolve = [False, False]
            self.reviving = False
            self.trapped = [False, False]
        def to_showdown_target(self, mv, tgt):
            return 1 if getattr(tgt, "tag", None) == "foe_a" else 2

    return _Battle(), bench0, bench1


def _patch_bench(monkeypatch, bench0, bench1, slot_map):
    """own_bench_mons → [bench0, bench1]; own_switch_slot → slot_map[id(mon)]."""
    import v_dance.play.vgc_base as vb
    monkeypatch.setattr(vb, "own_bench_mons", lambda battle: [bench0, bench1])
    monkeypatch.setattr(vb, "own_switch_slot",
                        lambda battle, mon: slot_map.get(id(mon)))


# ══════════════════════════════════════════════════════════════════════════════
# _switch_order_target
# ══════════════════════════════════════════════════════════════════════════════
def test_switch_order_target_extracts_switch_command():
    from poke_env.player.battle_order import (
        SingleBattleOrder, PassBattleOrder, DefaultBattleOrder,
    )
    from v_dance.play.vgc_base import _switch_order_target
    assert _switch_order_target(SingleBattleOrder("/choose switch 3")) == "/choose switch 3"
    assert _switch_order_target(SingleBattleOrder("/choose move closecombat 1")) is None
    assert _switch_order_target(PassBattleOrder()) is None      # "/choose pass"
    assert _switch_order_target(DefaultBattleOrder()) is None   # "/choose default"
    assert _switch_order_target(None) is None


# ══════════════════════════════════════════════════════════════════════════════
# action_to_order — refuses a taken switch, allows a distinct one, legacy when None
# ══════════════════════════════════════════════════════════════════════════════
def test_action_to_order_refuses_taken_switch(monkeypatch):
    from v_dance.play.vgc_base import action_to_order, _switch_order_target
    battle, b0, b1 = _make_switch_battle(slot1_can_move=True)
    _patch_bench(monkeypatch, b0, b1, {id(b0): 3, id(b1): 4})

    # switch index 12 → bench0 → "/choose switch 3"
    o = action_to_order(12, battle, 0)
    assert _switch_order_target(o) == "/choose switch 3"

    # REFUSED when that exact target is already taken by the other slot
    assert action_to_order(12, battle, 1, taken_switch_targets={"/choose switch 3"}) is None
    # a DIFFERENT taken target does not block it
    assert _switch_order_target(
        action_to_order(12, battle, 1, taken_switch_targets={"/choose switch 4"})
    ) == "/choose switch 3"
    # legacy: no taken set → unchanged behaviour
    assert _switch_order_target(action_to_order(12, battle, 1)) == "/choose switch 3"


# ══════════════════════════════════════════════════════════════════════════════
# CAUSE #1 — perturbation: both slots pick the SAME switch index
# ══════════════════════════════════════════════════════════════════════════════
def test_safe_order_dedups_same_switch_index(monkeypatch):
    """Both slots perturb to switch index 12 → without the dedup both emit
    '/choose switch 3' (collision); with it, slot 1's colliding switch is refused
    and it takes a legal NON-colliding action instead.  A real double-switch would
    use two DIFFERENT bench indices, so a same-index collision is only ever a
    perturbation/illusion artifact — any non-colliding legal order is correct."""
    from v_dance.play.vgc_base import VGCPlayerBase, _switch_order_target
    battle, b0, b1 = _make_switch_battle(slot1_can_move=True)
    _patch_bench(monkeypatch, b0, b1, {id(b0): 3, id(b1): 4})

    order_s0 = VGCPlayerBase._safe_order(None, 12, battle, 0)
    taken = {_switch_order_target(order_s0)}
    order_s1 = VGCPlayerBase._safe_order(None, 12, battle, 1, taken_switch_targets=taken)

    assert _switch_order_target(order_s0) == "/choose switch 3"
    assert _switch_order_target(order_s1) != "/choose switch 3"      # NO collision
    assert order_s0.message != order_s1.message

    # Demonstrate the BUG without the fix: slot 1 re-uses slot 0's switch.
    bug = VGCPlayerBase._safe_order(None, 12, battle, 1, taken_switch_targets=None)
    assert _switch_order_target(bug) == "/choose switch 3"


# ══════════════════════════════════════════════════════════════════════════════
# CAUSE #2 — under-illusion: two MOVE actions both fall back to the first switch
# ══════════════════════════════════════════════════════════════════════════════
def test_safe_order_fallback_avoids_taken_switch_under_illusion(monkeypatch):
    """Slot 1's move action decodes to None (illusion: no usable move), so
    _safe_order's level-2 scan substitutes a switch.  WITHOUT the fix it grabs the
    first legal switch (index 12 → slot 3) = slot 0's switch → collision; WITH it,
    the scan skips the taken switch and picks bench1 (slot 4)."""
    from v_dance.play.vgc_base import VGCPlayerBase, action_to_order, _switch_order_target
    battle, b0, b1 = _make_switch_battle(slot1_can_move=False)
    _patch_bench(monkeypatch, b0, b1, {id(b0): 3, id(b1): 4})

    order_s0 = action_to_order(12, battle, 0)                 # slot 0 switches bench0
    assert _switch_order_target(order_s0) == "/choose switch 3"
    taken = {_switch_order_target(order_s0)}

    # slot 1 picks a MOVE (action 0) that cannot decode (no usable move)
    assert action_to_order(0, battle, 1) is None
    order_s1 = VGCPlayerBase._safe_order(None, 0, battle, 1, taken_switch_targets=taken)
    assert _switch_order_target(order_s1) == "/choose switch 4"   # NOT slot 3

    # BUG without the fix: the scan grabs slot 0's switch.
    bug = VGCPlayerBase._safe_order(None, 0, battle, 1, taken_switch_targets=None)
    assert _switch_order_target(bug) == "/choose switch 3"


# ══════════════════════════════════════════════════════════════════════════════
# ILLUSION ALIAS — two DIFFERENT bench indices resolve to the SAME team slot
# ══════════════════════════════════════════════════════════════════════════════
def test_dedup_catches_two_indices_aliasing_one_team_slot(monkeypatch):
    """A same-species Illusion could alias two bench indices onto one /choose-switch
    command.  Comparing the RESOLVED command (not the codec index) still catches it:
    slot 1's action 13 resolves to slot 0's '/choose switch 3' → refused → slot 1
    falls back to its legal move instead of colliding."""
    from v_dance.play.vgc_base import VGCPlayerBase, action_to_order, _switch_order_target
    battle, b0, b1 = _make_switch_battle(slot1_can_move=True)
    _patch_bench(monkeypatch, b0, b1, {id(b0): 3, id(b1): 3})   # BOTH → slot 3

    order_s0 = action_to_order(12, battle, 0)
    assert _switch_order_target(order_s0) == "/choose switch 3"
    taken = {_switch_order_target(order_s0)}

    # slot 1 wanted switch index 13, which ALSO resolves to "/choose switch 3"
    assert action_to_order(13, battle, 1, taken_switch_targets=taken) is None
    order_s1 = VGCPlayerBase._safe_order(None, 13, battle, 1, taken_switch_targets=taken)
    # no non-colliding switch exists → fall back to slot 1's legal move (not a switch)
    assert _switch_order_target(order_s1) is None
    assert "closecombat" in order_s1.message.lower()


# ══════════════════════════════════════════════════════════════════════════════
# NEGATIVE CONTROL — a legitimate DISTINCT double-switch must NOT be rewritten
# ══════════════════════════════════════════════════════════════════════════════
def test_distinct_double_switch_is_preserved(monkeypatch):
    """The dedup must only fire on a genuine collision: two slots switching to
    DIFFERENT bench mons (distinct team slots) keep their orders unchanged."""
    from v_dance.play.vgc_base import VGCPlayerBase, action_to_order, _switch_order_target
    battle, b0, b1 = _make_switch_battle(slot1_can_move=True)
    _patch_bench(monkeypatch, b0, b1, {id(b0): 3, id(b1): 4})

    order_s0 = action_to_order(12, battle, 0)                 # bench0 → slot 3
    taken = {_switch_order_target(order_s0)}
    order_s1 = VGCPlayerBase._safe_order(None, 13, battle, 1, taken_switch_targets=taken)

    assert _switch_order_target(order_s0) == "/choose switch 3"
    assert _switch_order_target(order_s1) == "/choose switch 4"   # preserved, not over-fired
