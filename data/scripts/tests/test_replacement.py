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


# ── poke-env available_switches desync recovery (#11 endgame forced-switch) ───
def test_replacement_mask_recovers_from_available_switches_desync():
    """poke-env can DROP a healthy, brought bench mon from available_switches after
    an Imposter/Illusion turn → the model gets no legal replacement and the turn
    collapses to /choose default.  A post-faint replacement is never trapped, so
    every living brought bench mon (our own_bench_mons reconstruction) is legal:
    when poke-env offers NOTHING for a must-switch slot but our reconstruction sees
    a living bench mon, the mask offers it by POSITION so the model can still drive."""
    survivor = _mon("kingambit", revealed=True)
    dropped = _mon("sneasler", revealed=True)     # healthy, was on field, now benched
    battle = _Battle(
        active=[survivor, None],                  # slot 1 fainted
        available_switches=[[], []],              # DESYNC: poke-env drops `dropped`
        team=[survivor, dropped],
        force_switch=[False, True],
    )
    mask = build_replacement_mask(battle, 1)
    assert mask[SWITCH_OFFSET + 0] is True, "desync recovery failed to offer the bench mon"
    assert sum(1 for x in mask if x) == 1
    assert not any(mask[:SWITCH_OFFSET])          # still switch-only
    # The non-switching slot stays empty even under the desync fallback.
    assert not any(build_replacement_mask(battle, 0))


def test_recovers_mon_wrongly_flagged_active_by_illusion():
    """Faithful to the upstream mechanism: poke-env's available_switches excludes a
    mon when its per-mon ``active`` FLAG is True (double_battle.py), but an Illusion
    can leave a healthy BENCHED teammate's ``active`` flag set (the disguise's switch
    -in was applied to the real same-name teammate's object).  own_bench_mons keys
    off ``battle.active_pokemon`` (the resolved slot list), NOT the per-mon flag, so
    it still sees the mon — and the replacement mask recovers it."""
    survivor = _mon("kingambit", revealed=True)
    survivor._active = True
    dropped = _mon("sneasler", revealed=True)
    dropped._active = True                          # CORRUPTED flag (illusion desync)

    class _B:
        team = {"p1: Kingambit": survivor, "p1: Sneasler": dropped}
        available_switches = [[], []]               # poke-env drops `dropped` (active flag)
        force_switch = [False, True]
        battle_tag = "t-illusion"
        turn = 12
        @property
        def active_pokemon(self):
            return [survivor, None]                 # only `survivor` is truly on a slot

    from live_state_encoder import own_bench_mons
    assert "sneasler" in [m.species for m in own_bench_mons(_B())], \
        "own_bench_mons should see the wrongly-active mon (it reads active_pokemon, not the flag)"
    mask = build_replacement_mask(_B(), 1)
    assert mask[SWITCH_OFFSET + 0] is True


# ── #11c: normal-turn voluntary-switch mask robust to the desync ──────────────
def _normal_turn_battle(trapped, available_switches, reviving=False):
    from poke_env.battle import Status  # noqa: F401 (kept for parity with other tests)
    king = _mon("kingambit", revealed=True)
    sneasler = _mon("sneasler", revealed=True)

    class _B:
        team = {"p1: Kingambit": king, "p1: Sneasler": sneasler}
        last_request = {"side": {"pokemon": [
            {"ident": "p1: Kingambit", "active": True, "condition": "100/100"},
            {"ident": "p1: Sneasler", "active": False, "condition": "150/150"},
        ]}}
        available_moves = [[], []]
        @property
        def active_pokemon(self):
            return [king, None]
        @property
        def opponent_active_pokemon(self):
            return [None, None]

    b = _B()
    b.available_switches = available_switches
    b.trapped = trapped
    b.reviving = reviving
    return b


def test_normal_switch_mask_recovers_from_corrupted_available_switches():
    """#11c: on a NORMAL turn (not a forced replacement), the switch mask must
    offer a healthy bench mon even when an Illusion has corrupted available_switches
    to drop it — gated only on the (uncorruptible, request-derived) ``trapped``
    signal, not on available_switches membership."""
    from vgc_base import build_legal_action_mask
    b = _normal_turn_battle(trapped=[False, False], available_switches=[[], []])
    mask = build_legal_action_mask(b, 0)
    assert mask[SWITCH_OFFSET + 0] is True, \
        "healthy bench mon dropped by corrupted available_switches not recovered"


def test_normal_switch_mask_respects_real_trapping():
    """A genuinely trapped active (Arena Trap / Shadow Tag — battle.trapped[slot])
    must expose NO switch, even though the bench mon is healthy and available."""
    from vgc_base import build_legal_action_mask
    b = _normal_turn_battle(trapped=[True, False], available_switches=[[], []])
    assert not any(build_legal_action_mask(b, 0)[SWITCH_OFFSET:SWITCH_OFFSET + 4]), \
        "trapped slot must expose no switch"


def test_normal_switch_mask_reviving_offers_no_living_switch():
    """Revival Blessing switches in a FAINTED mon, which the living-bench codec
    can't express — so a reviving request offers no switch (prior behaviour), never
    a living-mon /switch the server would reject."""
    from vgc_base import build_legal_action_mask
    b = _normal_turn_battle(trapped=[False, False], available_switches=[[], []],
                            reviving=True)
    assert not any(build_legal_action_mask(b, 0)[SWITCH_OFFSET:SWITCH_OFFSET + 4])


# ── #11b: request-authoritative own bench (immune to Illusion flag corruption) ─
def test_own_bench_uses_request_over_corrupted_fainted_flag():
    """#11b: an Illusion can leave a healthy bench mon's poke-env ``fainted`` FLAG
    set, dropping it from available_switches AND the flag-based bench.  The in-battle
    |request| (brought-only, authoritative) says it is healthy and benched, so the
    request-aware bench recovers it — and the replacement mask offers it."""
    from live_state_encoder import own_bench_mons
    from poke_env.battle import Status
    king = _mon("kingambit", revealed=True)
    sneasler = _mon("sneasler", revealed=True)
    sneasler._status = Status.FNT                  # CORRUPTED flag; max_hp stays intact
    assert sneasler.fainted is True

    class _B:
        team = {"p1: Kingambit": king, "p1: Sneasler": sneasler}
        last_request = {"side": {"pokemon": [
            {"ident": "p1: Kingambit", "active": True, "condition": "100/100"},
            {"ident": "p1: Sneasler", "active": False, "condition": "150/150"},  # truth
        ]}}
        available_switches = [[], []]               # poke-env offers nothing (desync)
        force_switch = [False, True]
        battle_tag = "t-11b"; turn = 14
        @property
        def active_pokemon(self):
            return [king, None]

    b = _B()
    assert "sneasler" in [m.species for m in own_bench_mons(b)], \
        "request-aware bench should recover the wrongly-fainted mon"
    assert build_replacement_mask(b, 1)[SWITCH_OFFSET + 0] is True


def test_request_aware_bench_excludes_unbrought_mon():
    """The in-battle request lists ONLY the brought team, so a mon in battle.team
    but ABSENT from the request (an un-brought 2-of-6) is never offered, even though
    poke-env reports it healthy."""
    from live_state_encoder import own_bench_mons
    king = _mon("kingambit", revealed=True)
    bench = _mon("sneasler", revealed=True)
    unbrought = _mon("charizard")                  # healthy in poke-env, NOT in request

    class _B:
        team = {"p1: Kingambit": king, "p1: Sneasler": bench, "p1: Charizard": unbrought}
        last_request = {"side": {"pokemon": [
            {"ident": "p1: Kingambit", "active": True, "condition": "100/100"},
            {"ident": "p1: Sneasler", "active": False, "condition": "150/150"},
        ]}}
        available_switches = [[], []]
        force_switch = [False, True]
        battle_tag = "t-11b2"; turn = 14
        @property
        def active_pokemon(self):
            return [king, None]

    species = [m.species for m in own_bench_mons(_B())]
    assert "sneasler" in species and "charizard" not in species


def test_no_request_falls_back_to_flag_heuristic():
    """Absent a usable request (the replay-driven parity path), behaviour is
    unchanged — _request_own_state returns None and the flag/reveal heuristic drives
    the bench, so the offline parity tests are unaffected."""
    from live_state_encoder import _request_own_state, own_bench_mons
    king = _mon("kingambit", revealed=True)
    bench = _mon("sneasler", revealed=True)

    class _B:
        team = {"p1: Kingambit": king, "p1: Sneasler": bench}
        last_request = {}                          # no request yet
        available_switches = [[], [bench]]
        force_switch = [False, True]
        battle_tag = "t-11b3"; turn = 3
        @property
        def active_pokemon(self):
            return [king, None]

    b = _B()
    assert _request_own_state(b) is None
    assert "sneasler" in [m.species for m in own_bench_mons(b)]


def test_desync_recovery_does_not_offer_unbrought_mon():
    """The recovery must NOT surface an un-brought roster mon (the un-brought
    2-of-6 are un-revealed and absent from available_switches): own_bench_mons
    already excludes them, so an empty available_switches with ONLY an un-brought
    mon present still yields no replacement (genuine /choose default)."""
    survivor = _mon("kingambit", revealed=True)
    unbrought = _mon("charizard", revealed=False)  # never brought / never seen
    battle = _Battle(
        active=[survivor, None],
        available_switches=[[], []],               # empty for both slots
        team=[survivor, unbrought],
        force_switch=[False, True],
    )
    assert not any(build_replacement_mask(battle, 1)), \
        "un-brought mon wrongly offered by the desync fallback"


def test_desync_recovery_orders_the_real_bench_mon():
    """End-to-end: the recovered switch index decodes (via _replacement_order, which
    builds from the SAME own_bench_mons list) to a /switch at the real healthy mon —
    the order Showdown accepts, NOT /choose default."""
    from live_vgc_base import SplicingVGCPlayerBase as S
    survivor = _mon("kingambit", revealed=True)
    dropped = _mon("basculegion", revealed=True)
    battle = _Battle(
        active=[survivor, None],
        available_switches=[[], []],
        team=[survivor, dropped],
        force_switch=[False, True],
    )
    mask = build_replacement_mask(battle, 1)
    idx = next(i for i, ok in enumerate(mask) if ok)
    order = S._replacement_order(battle, 1, idx)
    assert "basculegion" in order.message.lower()


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
    try:
        return M.load_bc_policy(ckpt, "cpu")
    except ValueError as e:
        # stale-layout checkpoint (predates the gap-#5 layout) → retrain pending
        if "layout" in str(e) or "state_dim" in str(e):
            pytest.skip(f"bc_best.pt is a stale-layout checkpoint (retrain pending): {e}")
        raise


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
