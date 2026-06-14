"""Tests for the state_encoder OFFLINE path, focused on the layout-v2 additions:
opponent-bench slots, the per-mon is_fainted flag, and the team-count globals.
"""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")
import numpy as np

from state_encoder import (
    StateEncoder, get_state_dim,
    POKEMON_FEATURES, ACTIVE_SLOTS, BENCH_SLOTS, OPP_BENCH_SLOTS, GLOBAL_FEATURES,
)

# Per-mon feature offsets (within a 110-float slot, layout-v2)
_HP = 0
_TYPE1 = 1                             # type1 one-hot starts here (20 wide)
_BASE = 1 + 20 + 20                    # base stats start (6 wide)
_IS_ACTIVE = POKEMON_FEATURES - 4      # 106
_IS_REVEALED = POKEMON_FEATURES - 3    # 107
_IS_FAINTED = POKEMON_FEATURES - 2     # 108
_IS_TRANSFORMED = POKEMON_FEATURES - 1 # 109
_OPP_BENCH0 = (ACTIVE_SLOTS + BENCH_SLOTS) * POKEMON_FEATURES   # slot 8 base


def _mon(species, *, hp=100.0, seen=True, fainted=False, moves=("Fake Out",)):
    return {
        "species": species, "base_species": species,
        "hp_pct": (None if (not seen and not fainted) else (0.0 if fainted else hp)),
        "seen": seen, "is_fainted": fainted,
        "known_moves": list(moves), "revealed_moves": [],
        "boosts": {}, "status": None,
    }


def _snap():
    return {
        "our_active": {"our_a": _mon("Incineroar"), "our_b": _mon("Kingambit")},
        "opp_active": {"opp_a": _mon("Calyrex-Shadow"), "opp_b": _mon("Miraidon")},
        "our_bench": [_mon("Rillaboom"), _mon("Urshifu", fainted=True)],
        "opp_bench": [
            _mon("Chien-Pao", seen=True),                  # seen, alive
            _mon("Ogerpon", seen=True, fainted=True),      # seen, fainted
            _mon("Landorus", seen=False),                  # unseen stub
        ],
        "field": {}, "side_conditions": {},
    }


# ── Dimension / structure ───────────────────────────────────────────────────
def test_state_dim_is_layout_v2():
    assert POKEMON_FEATURES == 110
    assert (ACTIVE_SLOTS, BENCH_SLOTS, OPP_BENCH_SLOTS) == (4, 4, 4)
    assert GLOBAL_FEATURES == 78
    assert get_state_dim() == 1398


def test_encode_shape_and_finite():
    vec = StateEncoder().encode_snapshot(_snap(), turn=3)
    assert vec.shape == (get_state_dim(),)
    assert np.isfinite(vec).all()


# ── Opponent bench slots ─────────────────────────────────────────────────────
def test_opp_bench_slots_populated_in_priority_order():
    vec = StateEncoder().encode_snapshot(_snap(), turn=3)

    def slot(j):  # opp bench slot j → absolute base
        return _OPP_BENCH0 + j * POKEMON_FEATURES

    s0, s1, s2, s3 = (slot(0), slot(1), slot(2), slot(3))
    # slot 0 = seen alive (Chien-Pao): revealed, not fainted, hp>0
    assert vec[s0 + _IS_REVEALED] == 1.0
    assert vec[s0 + _IS_FAINTED] == 0.0
    assert vec[s0 + _HP] > 0.0
    # slot 1 = seen fainted (Ogerpon): revealed AND fainted flag set
    assert vec[s1 + _IS_REVEALED] == 1.0
    assert vec[s1 + _IS_FAINTED] == 1.0
    # slot 2 = unseen stub (Landorus): not revealed, not fainted
    assert vec[s2 + _IS_REVEALED] == 0.0
    assert vec[s2 + _IS_FAINTED] == 0.0
    # slot 3 = empty → all zeros
    assert np.all(vec[s3:s3 + POKEMON_FEATURES] == 0.0)


def test_active_and_own_bench_are_never_flagged_fainted():
    vec = StateEncoder().encode_snapshot(_snap(), turn=3)
    # active slots 0..3 and own-bench slots 4..7 all carry is_fainted = 0
    for slot_idx in range(ACTIVE_SLOTS + BENCH_SLOTS):
        base = slot_idx * POKEMON_FEATURES
        assert vec[base + _IS_FAINTED] == 0.0


# ── Team-count globals (last 4 features) ────────────────────────────────────
def test_team_count_globals():
    vec = StateEncoder().encode_snapshot(_snap(), turn=3)
    own_live, opp_live, own_fnt, opp_fnt = vec[-4:]
    # own bench: Rillaboom alive, Urshifu fainted
    assert own_live == pytest.approx(1 / 4)
    assert own_fnt == pytest.approx(1 / 4)
    # opp: 1 seen-alive (Chien-Pao), 1 seen-fainted (Ogerpon); unseen not counted
    assert opp_live == pytest.approx(1 / 4)
    assert opp_fnt == pytest.approx(1 / 4)


def test_unseen_opp_bench_not_counted_as_alive():
    snap = _snap()
    # make ALL opp bench unseen stubs → opp_live and opp_fnt both 0
    snap["opp_bench"] = [_mon("Chien-Pao", seen=False),
                         _mon("Ogerpon", seen=False)]
    vec = StateEncoder().encode_snapshot(snap, turn=1)
    _, opp_live, _, opp_fnt = vec[-4:]
    assert opp_live == 0.0 and opp_fnt == 0.0


# ── Transform / Ditto (Solution A): encode the copied forme + is_transformed ──
def test_transformed_mon_encodes_as_copied_forme():
    ditto = _mon("Ditto")
    ditto.update({"is_transformed": True, "transformed_into": "Garchomp"})
    snap = {
        "our_active": {"our_a": ditto, "our_b": _mon("Garchomp")},
        "opp_active": {}, "our_bench": [], "opp_bench": [],
        "field": {}, "side_conditions": {},
    }
    vec = StateEncoder().encode_snapshot(snap, turn=1)
    a, b = 0, POKEMON_FEATURES          # our_a (Ditto→Garchomp) vs our_b (real Garchomp)
    # types one-hot (40 wide) and base stats (6 wide) match the copied forme
    assert np.array_equal(vec[a + _TYPE1:a + _TYPE1 + 40], vec[b + _TYPE1:b + _TYPE1 + 40])
    assert np.allclose(vec[a + _BASE:a + _BASE + 6], vec[b + _BASE:b + _BASE + 6])
    assert vec[a + _BASE:a + _BASE + 6].sum() > 0     # Garchomp dex present
    # flag set on the transformed Ditto, not on the genuine Garchomp
    assert vec[a + _IS_TRANSFORMED] == 1.0
    assert vec[b + _IS_TRANSFORMED] == 0.0


def test_untransformed_ditto_encodes_as_ditto():
    snap = {
        "our_active": {"our_a": _mon("Ditto"), "our_b": _mon("Garchomp")},
        "opp_active": {}, "our_bench": [], "opp_bench": [],
        "field": {}, "side_conditions": {},
    }
    vec = StateEncoder().encode_snapshot(snap, turn=1)
    a, b = 0, POKEMON_FEATURES
    # a plain Ditto's base stats differ from Garchomp's, and no transform flag
    assert not np.allclose(vec[a + _BASE:a + _BASE + 6], vec[b + _BASE:b + _BASE + 6])
    assert vec[a + _IS_TRANSFORMED] == 0.0
