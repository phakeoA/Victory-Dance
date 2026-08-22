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
    return  # imports are absolute v_dance now; path bootstrap no longer needed
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
    from v_dance.play.vgc_base import action_to_order
    battle, foe_a, foe_b = _make_battle(None)
    battle.opponent_active_pokemon = [foe_a, foe_b]
    assert _target_pos(action_to_order(A_OPP_A, battle, 0)) == "1"   # opp_a → foe_a
    assert _target_pos(action_to_order(A_OPP_B, battle, 0)) == "2"   # opp_b → foe_b


def test_illusion_lost_slot_b_targets_position_2_with_recon():
    """THE FIX: poke-env lost opp slot b to the illusion merge, but the
    reconstruction confirms it is occupied → opp_b is ordered at position 2."""
    _repo_on_path()
    from v_dance.play.vgc_base import action_to_order
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
    from v_dance.play.vgc_base import action_to_order
    battle, foe_a, _ = _make_battle(None)
    battle.opponent_active_pokemon = [foe_a, None]
    assert _target_pos(action_to_order(A_OPP_B, battle, 0)) == "1"          # recon=None
    assert _target_pos(action_to_order(A_OPP_B, battle, 0,
                                       opp_present_recon={0: True, 1: False})) == "1"


def test_illusion_lost_slot_a_targets_position_1_with_recon():
    _repo_on_path()
    from v_dance.play.vgc_base import action_to_order
    battle, _, foe_b = _make_battle(None)
    battle.opponent_active_pokemon = [None, foe_b]          # slot a merged/lost
    order = action_to_order(A_OPP_A, battle, 0, opp_present_recon={0: True, 1: True})
    assert _target_pos(order) == "1", f"expected deliberate opp_a @ pos1: {order.message!r}"


# ── #1b: own ACTIVE mon under illusion has empty moves → recover from |request| ──
def _make_illusion_active_battle():
    """Own active mon whose poke-env object has EMPTY moves (a freshly sent-in
    Zoroark disguised as a brought teammate, before it has used a move), while the
    private |request| (``battle.available_moves[0]``) holds its real move.  No bench
    and not trapped, so WITHOUT the #1b recovery the action mask is ALL-ZERO →
    ``/choose default`` (a wasted turn)."""
    from poke_env.battle import Move
    cc = Move("closecombat", gen=9)

    class _Mon:
        def __init__(self, tag, moves):
            self.tag = tag
            self.fainted = False
            self.species = "x"
            self.base_species = "x"
            self._mv = moves
        @property
        def moves(self):
            return self._mv

    disguise = _Mon("actor", {})                       # EMPTY — illusion-stale
    foe_a, foe_b = _Mon("foe_a", {}), _Mon("foe_b", {})

    class _Battle:
        OPPONENT_1_POSITION = 1
        OPPONENT_2_POSITION = 2
        POKEMON_1_POSITION = -1
        POKEMON_2_POSITION = -2
        def __init__(self):
            self.active_pokemon = [disguise, None]
            self.opponent_active_pokemon = [foe_a, foe_b]
            self.available_moves = [[cc], []]          # request-authoritative real move
            self.available_switches = [[], []]
            self.team = {}
            self.can_mega_evolve = [False, False]
            self.reviving = False
        def to_showdown_target(self, mv, tgt):
            return 1 if getattr(tgt, "tag", None) == "foe_a" else 2

    return _Battle(), cc


def test_own_active_move_list_recovers_request_moves_under_illusion():
    _repo_on_path()
    from v_dance.encoders.live_state_encoder import own_active_move_list
    battle, _cc = _make_illusion_active_battle()
    got = own_active_move_list(battle, 0, battle.active_pokemon[0])
    assert [m.id for m in got] == ["closecombat"], \
        "empty mon.moves must fall back to the request's available_moves"


def test_own_active_move_list_prefers_mon_moves_when_present():
    _repo_on_path()
    from v_dance.encoders.live_state_encoder import own_active_move_list
    from poke_env.battle import Move
    battle, _cc = _make_illusion_active_battle()
    battle.active_pokemon[0]._mv = {"tackle": Move("tackle", gen=9)}
    got = own_active_move_list(battle, 0, battle.active_pokemon[0])
    assert [m.id for m in got] == ["tackle"], \
        "request fallback must trigger ONLY when mon.moves is empty"


def test_own_active_move_list_empty_when_no_moves_anywhere():
    _repo_on_path()
    from v_dance.encoders.live_state_encoder import own_active_move_list
    battle, _cc = _make_illusion_active_battle()
    battle.available_moves = [[], []]
    assert own_active_move_list(battle, 0, battle.active_pokemon[0]) == []


def test_illusion_empty_moves_mask_offers_request_move_not_default():
    """WITHOUT #1b the disguise's empty moves yield an all-zero mask → /choose
    default; WITH it the mask offers the request move's foe buckets (switch-free)."""
    _repo_on_path()
    from v_dance.play.vgc_base import build_legal_action_mask
    from v_dance.encoders.state_encoder import SWITCH_OFFSET
    battle, _cc = _make_illusion_active_battle()
    mask = build_legal_action_mask(battle, 0)
    assert mask[0] is True and mask[1] is True, "move slot 0 opp_a/opp_b must be legal"
    assert mask[2] is False                       # no ally present → no ally bucket
    assert not any(mask[SWITCH_OFFSET:])          # no bench → no switches
    assert any(mask), "mask must NOT be all-zero (that would force /choose default)"


def test_illusion_empty_moves_codec_orders_real_move_not_none():
    """The chosen move decodes to a real /choose order instead of None → Pass →
    'must make a move' retry → default."""
    _repo_on_path()
    from v_dance.play.vgc_base import action_to_order
    battle, _cc = _make_illusion_active_battle()
    order = action_to_order(A_OPP_A, battle, 0)   # move slot 0, target opp_a
    assert order is not None, "empty-moves illusion slot must still order a real move"
    assert "closecombat" in order.message.lower()
    assert _target_pos(order) == "1"              # opp_a → foe_a → position 1
