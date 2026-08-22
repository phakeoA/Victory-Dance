"""v11 Phase D (Tera) — the bot ordering its own Terastallization (forward-compat; tera is mod-disabled in
Champions so this is INERT in production but fully wired).

Covers: D1 constants (GIMMICK_DIM 2->3), D9 layout bump (STATE_DIM 4673->4913, POKEMON_FEATURES 381->401,
layout v17) + the per-mon tera_type one-hot (revealed-only → all-zero today, lit when populated), and the D5
serve dedup matrix (both-tera dedups; mega+tera cross-slot KEPT). The mask/codec/parser pieces are covered in
test_gimmick_codec / _serve / _player / _transitions (extended to 3-wide).
"""
from __future__ import annotations

import numpy as np

from v_dance.encoders.state_encoder import (
    StateEncoder, get_state_dim, get_state_layout_version, POKEMON_FEATURES, NUM_TYPES,
    GIMMICK_DIM, GIMMICK_TERA, GIMMICK_MEGA, GIMMICK_NONE, _TYPE_IDX,
)

# tera_type one-hot block = the NUM_TYPES slot right before the 4 trailing flags (per-mon).
_TERA_BLOCK = POKEMON_FEATURES - 4 - NUM_TYPES


def _mon(species, *, is_terastallized=False, known_tera_type=None):
    return {"species": species, "base_species": species, "hp_pct": 100.0, "seen": True,
            "is_fainted": False, "known_moves": [], "revealed_moves": [], "boosts": {}, "status": None,
            "known_ability": "Pressure", "volatiles": {},
            "is_terastallized": is_terastallized, "known_tera_type": known_tera_type,
            "stats_estimate": {"mode": "exact",
                               "stats": {"atk": 100, "spa": 100, "def": 100, "spd": 100, "hp": 200, "spe": 100}}}


def _vec(our_a):
    snap = {"our_active": {"our_a": our_a, "our_b": None},
            "opp_active": {"opp_a": _mon("Blissey"), "opp_b": None},
            "our_bench": [], "opp_bench": [], "field": {}, "side_conditions": {}}
    return StateEncoder().encode_snapshot(snap, turn=1)


# ════════════════════════════ D1/D9 layout ════════════════════════════
def test_phase_d_layout():
    assert GIMMICK_DIM == 3 and GIMMICK_TERA == 2
    assert NUM_TYPES == 20
    assert POKEMON_FEATURES == 413
    assert get_state_dim() == 5057
    assert get_state_layout_version() == 19


# ════════════════════════════ D9 tera_type one-hot ════════════════════════════
def test_tera_type_onehot_zero_when_not_teraed():
    # the production case: no mon teras (mod-disabled) → the whole tera block is zero
    vec = _vec(_mon("Garchomp"))
    assert float(vec[_TERA_BLOCK: _TERA_BLOCK + NUM_TYPES].sum()) == 0.0


def test_tera_type_onehot_lit_when_revealed():
    # forward-compat: if a mon HAS tera'd (is_terastallized + known_tera_type), the one-hot lights that type
    vec = _vec(_mon("Garchomp", is_terastallized=True, known_tera_type="Fire"))
    block = vec[_TERA_BLOCK: _TERA_BLOCK + NUM_TYPES]
    assert float(block.sum()) == 1.0
    assert block[_TYPE_IDX["FIRE"]] == 1.0


def test_tera_type_onehot_needs_both_flags():
    # is_terastallized without a known type, or a known type without the flag → still zero (revealed-only)
    assert float(_vec(_mon("Garchomp", is_terastallized=True))[_TERA_BLOCK:_TERA_BLOCK + NUM_TYPES].sum()) == 0.0
    assert float(_vec(_mon("Garchomp", known_tera_type="Fire"))[_TERA_BLOCK:_TERA_BLOCK + NUM_TYPES].sum()) == 0.0


# ════════════════════════════ D5 serve dedup matrix ════════════════════════════
def _select_with_logits(monkeypatch, glog):
    """Drive VGCPlayer._select_gimmicks with controlled per-slot gimmick logits + an all-legal mask."""
    import types
    import v_dance.play.player as P
    monkeypatch.setattr(P._M, "gimmick_trained", lambda m: True)
    monkeypatch.setattr(P._M, "gimmick_logits", lambda *a, **k: glog)
    monkeypatch.setattr(P, "build_gimmick_legal_mask", lambda b, s: [True, True, True])
    fake = types.SimpleNamespace(_model=object(), _model_heads=("our_a", "our_b"), _device="cpu")
    return P.VGCPlayer._select_gimmicks(fake, object(), np.zeros(get_state_dim(), np.float32), 0, 0)


def test_dedup_both_tera_keeps_one(monkeypatch):
    # both slots want tera (once-per-battle) → drop the weaker-margin slot to NONE
    g0, g1 = _select_with_logits(monkeypatch, [[0.0, 0.0, 10.0], [0.0, 0.0, 5.0]])
    assert (g0, g1) == (GIMMICK_TERA, GIMMICK_NONE)   # slot 0 stronger margin kept


def test_dedup_mega_plus_tera_both_kept(monkeypatch):
    # slot0 mega + slot1 tera = DIFFERENT gimmicks → BOTH legal (Champions allows both across the battle)
    g0, g1 = _select_with_logits(monkeypatch, [[0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
    assert (g0, g1) == (GIMMICK_MEGA, GIMMICK_TERA)


def test_dedup_both_mega_still_dedups(monkeypatch):
    # regression: the original both-mega dedup still works under the generalized rule
    g0, g1 = _select_with_logits(monkeypatch, [[0.0, 4.0, 0.0], [0.0, 10.0, 0.0]])
    assert (g0, g1) == (GIMMICK_NONE, GIMMICK_MEGA)   # slot 1 stronger margin kept
