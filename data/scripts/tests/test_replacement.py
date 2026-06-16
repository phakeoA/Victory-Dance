"""Model-driven post-faint replacement tests.

The forced-replacement path encodes the post-faint board (gap-#6 opp splice) and
lets the BC policy choose the replacement: for each fainted slot, masked-argmax the
MATCHING head (slot 0 → our_a, slot 1 → our_b) over a switch-only replacement mask,
deduping the bench mon across slots.  This mirrors how training's
``decision_type='replacement'`` transitions are labelled, so the heads are
in-distribution for it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("poke_env")
import numpy as np

_REPO = Path(__file__).resolve().parents[3]
for _p in (str(_REPO / "data" / "scripts"), str(_REPO), str(_REPO / "local_battle")):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

from poke_env.battle import Pokemon
from vgc_base import build_replacement_mask
from state_encoder import SWITCH_OFFSET, STATE_DIM


def _mon(species, hp="100/100", fainted=False, revealed=False):
    p = Pokemon(gen=9, species=species)
    p.set_hp_status("0 fnt" if fainted else hp)
    p._revealed = revealed
    return p


class _Battle:
    """Minimal post-faint board for the replacement helpers."""
    def __init__(self, active, available_switches, team, force_switch, tag="t-1"):
        self._active = active
        self.available_switches = available_switches
        self.team = {f"p1: {m.species}{i}": m for i, m in enumerate(team)}
        self.force_switch = force_switch
        self.battle_tag = tag
        self.turn = 3
    @property
    def active_pokemon(self):
        return self._active


def test_build_replacement_mask_is_switch_only_and_available():
    survivor = _mon("kingambit", revealed=True)
    b1 = _mon("sneasler")
    b2 = _mon("sylveon")
    unbrought = _mon("charizard")           # in team, NOT switchable
    battle = _Battle(
        active=[survivor, None],            # slot 1 fainted
        available_switches=[[], [b1, b2]],  # slot 1 may bring b1/b2
        team=[survivor, b1, b2, unbrought],
        force_switch=[False, True],
    )
    mask = build_replacement_mask(battle, 1)
    # ONLY switch indices, and only for the two available bench mons.
    assert mask[SWITCH_OFFSET + 0] is True and mask[SWITCH_OFFSET + 1] is True
    assert sum(1 for x in mask if x) == 2          # exactly two switches
    assert not any(mask[:SWITCH_OFFSET])           # no move actions
    # The un-brought mon is never offered.
    assert mask[SWITCH_OFFSET + 2] is False and mask[SWITCH_OFFSET + 3] is False
    # A trapped / non-switching slot exposes nothing.
    assert not any(build_replacement_mask(battle, 0))


def test_replacement_order_decodes_switch_and_pass():
    from live_vgc_base import SplicingVGCPlayerBase as S
    from poke_env.player.battle_order import PassBattleOrder

    survivor = _mon("kingambit", revealed=True)
    b1 = _mon("sneasler")
    b2 = _mon("sylveon")
    battle = _Battle(
        active=[survivor, None],
        available_switches=[[], [b1, b2]],
        team=[survivor, b1, b2],
        force_switch=[False, True],
    )
    # switch index 12 → first bench mon (own_bench_mons order = team order: b1).
    order = S._replacement_order(battle, 1, SWITCH_OFFSET + 0)
    assert "sneasler" in order.message.lower()
    # None (non-switching slot) → Pass.
    assert isinstance(S._replacement_order(battle, 0, None), PassBattleOrder)


@pytest.fixture(scope="module")
def bc_model():
    pytest.importorskip("torch")
    import model_io as M
    ckpt = _REPO / "ai_train_scripts" / "BC_model" / "checkpoints" / "bc_best.pt"
    if not ckpt.exists():
        pytest.skip("bc_best.pt checkpoint missing")
    return M.load_bc_policy(ckpt, "cpu")


def _dummy_player(bc_model):
    """A stand-in carrying just what VGCPlayer._select_replacement_actions reads,
    so we can exercise it without building a full poke-env Player."""
    from player import VGCPlayer

    class _P:
        pass
    p = _P()
    p._model, p._model_heads = bc_model
    p._device = "cpu"
    p._select_replacement_actions = VGCPlayer._select_replacement_actions.__get__(p, _P)
    return p


def test_model_replacement_picks_a_legal_switch(bc_model):
    p = _dummy_player(bc_model)
    survivor = _mon("kingambit", revealed=True)
    b1 = _mon("sneasler")
    b2 = _mon("sylveon")
    battle = _Battle(
        active=[survivor, None],
        available_switches=[[], [b1, b2]],
        team=[survivor, b1, b2],
        force_switch=[False, True],
    )
    state = np.zeros(STATE_DIM, dtype=np.float32)
    res = p._select_replacement_actions(battle, state)
    assert res is not None, "model returned no replacement for a legal board"
    a0, a1, source = res
    assert a0 is None                              # slot 0 not switching
    assert a1 in (SWITCH_OFFSET + 0, SWITCH_OFFSET + 1)   # a legal switch
    assert source == "forced_switch_model"


def test_model_replacement_dedupes_double_faint(bc_model):
    p = _dummy_player(bc_model)
    b1 = _mon("sneasler")
    b2 = _mon("sylveon")
    b3 = _mon("kingambit")
    battle = _Battle(
        active=[None, None],                        # BOTH fainted
        available_switches=[[b1, b2, b3], [b1, b2, b3]],
        team=[b1, b2, b3],
        force_switch=[True, True],
    )
    state = np.zeros(STATE_DIM, dtype=np.float32)
    res = p._select_replacement_actions(battle, state)
    assert res is not None
    a0, a1, _ = res
    assert a0 is not None and a1 is not None
    assert a0 != a1, "both slots brought the SAME mon (dedupe failed)"


def test_model_replacement_falls_back_when_no_legal_switch(bc_model):
    p = _dummy_player(bc_model)
    survivor = _mon("kingambit", revealed=True)
    battle = _Battle(
        active=[survivor, None],
        available_switches=[[], []],                # nothing to switch to
        team=[survivor],
        force_switch=[False, True],
    )
    state = np.zeros(STATE_DIM, dtype=np.float32)
    # No legal replacement → defer to the random picker (None).
    assert p._select_replacement_actions(battle, state) is None
