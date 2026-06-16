"""Task #5e: the live player threads the model's gimmick (mega) decision into the
order.  Tests the base hook (never megas) and VGCPlayer._select_gimmicks (megas a
capable MOVE slot when the trained gimmick head favours it; never on a switch or
an untrained head).  Calls the methods unbound with a lightweight fake self so we
don't open a Showdown connection."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("poke_env")


def _setup_path():
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parents[3]
    for p in (str(repo / "local_battle"), str(repo / "data" / "scripts"),
              str(repo / "ai_train_scripts" / "BC_model"), str(repo)):
        if p not in sys.path:
            sys.path.insert(0, p)


class _DMon:
    def __init__(self, species, fainted=False):
        self.species = species
        self.base_species = species
        self.fainted = fainted


class _DBattle:
    def __init__(self, active, team=None):
        self.active_pokemon = active
        self.team = team or {}


def _mega_favoring_model():
    """A gimmick-trained BCPolicy whose gimmick heads always prefer mega (bias
    [0, 10]) so masked-argmax picks mega wherever the mask allows it."""
    import torch
    from bc_model import BCPolicy
    m = BCPolicy(hidden_dims=(8,), dropout=0.0)
    with torch.no_grad():
        for h in m.gimmick_heads:
            m.gimmick_heads[h].weight.zero_()
            m.gimmick_heads[h].bias.copy_(torch.tensor([0.0, 10.0]))
    m._gimmick_trained = True
    m.eval()
    return m


def test_base_select_gimmicks_never_megas():
    _setup_path()
    from vgc_base import VGCPlayerBase, GIMMICK_NONE
    # The base method ignores self/battle/state — it simply never gimmicks.
    assert VGCPlayerBase._select_gimmicks(None, None, None, 0, 1) == (GIMMICK_NONE, GIMMICK_NONE)


def test_select_actions_dedups_cross_slot_switch(monkeypatch):
    """Both heads picking the SAME bench mon to switch into is rejected by Showdown
    ('can only switch in once'); _select_actions must re-decode slot 1 with the
    colliding switch masked → a non-colliding action (#8 follow-up bug)."""
    _setup_path()
    import types
    import numpy as np
    import player as P
    from state_encoder import SWITCH_OFFSET

    def fake_mask(battle, slot):
        row = [False] * 16
        row[SWITCH_OFFSET] = True            # bench-0 switch legal for both slots
        if slot == 1:
            row[0] = True                     # slot 1 also has move-0 available
        return row

    def fake_indices(model, heads, sv, m0, m1, device):
        # both heads want the same switch; once it's masked for slot 1, slot 1
        # falls to move 0.
        return (SWITCH_OFFSET, SWITCH_OFFSET) if m1[SWITCH_OFFSET] else (SWITCH_OFFSET, 0)

    monkeypatch.setattr(P, "build_legal_action_mask", fake_mask)
    monkeypatch.setattr(P._M, "bc_action_indices", fake_indices)

    fake = types.SimpleNamespace(_model=object(), _model_heads=("our_a", "our_b"),
                                 _device="cpu")
    a0, a1, src = P.VGCPlayer._select_actions(fake, object(), np.zeros(4, np.float32))
    assert a0 == SWITCH_OFFSET
    assert a1 == 0 and a1 != a0               # deduped to a non-colliding action
    assert src == "model"


def test_model_select_gimmicks_megas_capable_move_slot_only():
    _setup_path()
    import types
    import numpy as np
    from player import VGCPlayer
    from state_encoder import STATE_DIM, GIMMICK_MEGA, GIMMICK_NONE, SWITCH_OFFSET

    fake = types.SimpleNamespace(_model=_mega_favoring_model(),
                                 _model_heads=("our_a", "our_b"), _device="cpu")
    battle = _DBattle([_DMon("charizard"), _DMon("incineroar")])
    sv = np.zeros(STATE_DIM, np.float32)

    # slot 0 move + capable Charizard → mega; slot 1 move but Incineroar not
    # capable → gimmick mask [1,0] → none.
    g0, g1 = VGCPlayer._select_gimmicks(fake, battle, sv, 0, 0)
    assert g0 == GIMMICK_MEGA
    assert g1 == GIMMICK_NONE

    # a switch action never megas, even for the capable slot.
    gsw, _ = VGCPlayer._select_gimmicks(fake, battle, sv, SWITCH_OFFSET, 0)
    assert gsw == GIMMICK_NONE


def test_model_select_gimmicks_untrained_head_never_megas():
    _setup_path()
    import types
    import numpy as np
    from player import VGCPlayer
    from state_encoder import STATE_DIM, GIMMICK_NONE

    model = _mega_favoring_model()
    model._gimmick_trained = False        # a pre-gimmick checkpoint
    fake = types.SimpleNamespace(_model=model, _model_heads=("our_a", "our_b"),
                                 _device="cpu")
    battle = _DBattle([_DMon("charizard"), None])
    g = VGCPlayer._select_gimmicks(fake, battle, np.zeros(STATE_DIM, np.float32), 0, 0)
    assert g == (GIMMICK_NONE, GIMMICK_NONE)


def test_end_to_end_gimmick_head_to_mega_order():
    """Full serve chain: trained gimmick head → _select_gimmicks picks mega →
    _safe_order(gimmick) → action_to_order emits a real mega order."""
    _setup_path()
    import types
    import numpy as np
    from poke_env.battle import Move
    from player import VGCPlayer
    from vgc_base import VGCPlayerBase
    from state_encoder import STATE_DIM, GIMMICK_MEGA

    fth = Move("flamethrower", gen=9)

    class _Mon:
        def __init__(self, species, moves):
            self.species = species
            self.base_species = species
            self.fainted = False
            self._mv = moves
        @property
        def moves(self):
            return self._mv

    actor = _Mon("charizard", {"flamethrower": fth})
    foe = _Mon("incineroar", {"flamethrower": fth})

    class _Battle:
        active_pokemon = [actor, None]
        opponent_active_pokemon = [foe, None]
        available_moves = [list(actor.moves.values()), []]
        available_switches = [[], []]
        team = {}
        can_mega_evolve = [True, False]
        def to_showdown_target(self, mv, tgt):
            return 1

    battle = _Battle()
    sv = np.zeros(STATE_DIM, np.float32)
    fake = types.SimpleNamespace(_model=_mega_favoring_model(),
                                 _model_heads=("our_a", "our_b"), _device="cpu")

    # 1) the gimmick head selects mega for the capable move slot…
    g0, _g1 = VGCPlayer._select_gimmicks(fake, battle, sv, 0, 0)
    assert g0 == GIMMICK_MEGA
    # 2) …and _safe_order threads it into a real mega order.
    order = VGCPlayerBase._safe_order(None, 0, battle, 0, g0)
    assert "mega" in order.message, f"expected a mega order: {order.message!r}"
