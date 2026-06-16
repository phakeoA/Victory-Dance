"""Task #8 (#15): deliberate opp_a/opp_b targeting during a same-species Zoroark
illusion, using the gap-#6 reconstructed opponent occupancy.

When poke-env MERGES two same-species foes it loses a target slot; the codec
then can't aim the model's chosen opp_a/opp_b foe and (legacy) collapses to the
only visible foe.  With the reconstruction confirming the slot is occupied, the
codec instead targets that slot's FIXED Showdown position (opp_a→1, opp_b→2).
"""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")
pytest.importorskip("poke_env")


def _repo_on_path():
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parents[2]
    for p in (str(repo), str(repo / "data" / "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)


def _make_battle(opp_active):
    """A single-actor board with a usable single-target move; ``opp_active`` is
    the (possibly slot-lost) opponent_active_pokemon list.  to_showdown_target
    maps the two real foes to positions 1 / 2 by identity."""
    from poke_env.battle import Move
    cc = Move("closecombat", gen=9)

    class _Mon:
        def __init__(self, tag):
            self.tag = tag
            self.fainted = False
            self.species = "x"
            self.base_species = "x"
            self._mv = {"closecombat": cc}
        @property
        def moves(self):
            return self._mv

    actor = _Mon("actor")
    foe_a, foe_b = _Mon("foe_a"), _Mon("foe_b")

    class _Battle:
        active_pokemon = [actor, None]
        available_moves = [list(actor.moves.values()), []]
        available_switches = [[], []]
        team = {}
        can_mega_evolve = [False, False]
        OPPONENT_1_POSITION = 1
        OPPONENT_2_POSITION = 2
        POKEMON_1_POSITION = -1
        POKEMON_2_POSITION = -2
        def __init__(self, oa):
            self.opponent_active_pokemon = oa
        def to_showdown_target(self, mv, tgt):
            return 1 if getattr(tgt, "tag", None) == "foe_a" else 2

    return _Battle(opp_active), foe_a, foe_b


# action indices: move slot 0, target bucket → 0=opp_a, 1=opp_b
A_OPP_A, A_OPP_B = 0, 1


def _target_pos(order):
    return order.message.strip().split()[-1]      # "/choose move closecombat <pos>"


def test_direct_targets_both_foes_when_pokeenv_sees_them():
    _repo_on_path()
    from vgc_base import action_to_order
    battle, foe_a, foe_b = _make_battle(None)
    battle.opponent_active_pokemon = [foe_a, foe_b]
    assert _target_pos(action_to_order(A_OPP_A, battle, 0)) == "1"   # opp_a → foe_a
    assert _target_pos(action_to_order(A_OPP_B, battle, 0)) == "2"   # opp_b → foe_b


def test_illusion_lost_slot_b_targets_position_2_with_recon():
    """THE FIX: poke-env lost opp slot b to the illusion merge, but the
    reconstruction confirms it is occupied → opp_b is ordered at position 2."""
    _repo_on_path()
    from vgc_base import action_to_order
    battle, foe_a, _ = _make_battle(None)
    battle.opponent_active_pokemon = [foe_a, None]          # slot b merged/lost
    order = action_to_order(A_OPP_B, battle, 0, opp_present_recon={0: True, 1: True})
    assert _target_pos(order) == "2", f"expected deliberate opp_b @ pos2: {order.message!r}"
    # and opp_a still resolves to position 1
    assert _target_pos(action_to_order(A_OPP_A, battle, 0,
                                       opp_present_recon={0: True, 1: True})) == "1"


def test_illusion_lost_slot_b_legacy_collapses_without_recon():
    """No reconstruction (legacy): opp_b collapses to the only visible foe (pos1)
    — the documented pre-#15 behaviour, preserved when recon is unavailable."""
    _repo_on_path()
    from vgc_base import action_to_order
    battle, foe_a, _ = _make_battle(None)
    battle.opponent_active_pokemon = [foe_a, None]
    assert _target_pos(action_to_order(A_OPP_B, battle, 0)) == "1"          # recon=None
    assert _target_pos(action_to_order(A_OPP_B, battle, 0,
                                       opp_present_recon={0: True, 1: False})) == "1"


def test_illusion_lost_slot_a_targets_position_1_with_recon():
    _repo_on_path()
    from vgc_base import action_to_order
    battle, _, foe_b = _make_battle(None)
    battle.opponent_active_pokemon = [None, foe_b]          # slot a merged/lost
    order = action_to_order(A_OPP_A, battle, 0, opp_present_recon={0: True, 1: True})
    assert _target_pos(order) == "1", f"expected deliberate opp_a @ pos1: {order.message!r}"
