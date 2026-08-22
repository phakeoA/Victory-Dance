"""v11 E4-fix: the GLOBAL turn-order effective-speed block must suppress Choice Scarf under Magic Room
(parity with the per-move who-moves-first eff_speed, which already passed magic_room).

Before the fix the global B1.4 turn-order block called _effective_speed WITHOUT magic_room (on BOTH
encoders → parity-clean but a correctness gap): a Choice-Scarf holder's turn-order speed showed the
×1.5 even under Magic Room, disagreeing with the att_ctx eff_speed. The fix threads _magic_room into
both turn-order blocks (Klutz/Embargo were already gated INSIDE _effective_speed from the mon).
"""
from __future__ import annotations

from v_dance.encoders.state_encoder import (
    StateEncoder, POKEMON_FEATURES, NUM_WEATHER, NUM_FIELDS, NUM_SIDE_CONDS,
)

# Absolute index of the our_a effective-speed channel (first of the 12-wide turn-order block):
#   12 mon-slots × POKEMON_FEATURES, then weather + fields + own-SC + opp-SC + turn + trick_room.
_GLOBAL_BASE = 12 * POKEMON_FEATURES
_OUR_A_SPEED = _GLOBAL_BASE + NUM_WEATHER + NUM_FIELDS + 2 * NUM_SIDE_CONDS + 1 + 1


def _scarf_mon(spe=200):
    return {"species": "Dragapult", "base_species": "Dragapult", "hp_pct": 100.0, "seen": True,
            "is_fainted": False, "known_moves": ["Dragon Darts"], "revealed_moves": [], "boosts": {},
            "status": None, "known_ability": "Clear Body", "known_item": "Choice Scarf", "volatiles": {},
            "stats_estimate": {"mode": "exact",
                               "stats": {"atk": 120, "spa": 100, "def": 80, "spd": 80, "hp": 280, "spe": spe}}}


def _our_a_turnorder_speed(magic_room: bool):
    field = {"magic_room_turns_remaining": 5} if magic_room else {}
    snap = {"our_active": {"our_a": _scarf_mon(), "our_b": None},
            "opp_active": {"opp_a": None, "opp_b": None},
            "our_bench": [], "opp_bench": [], "field": field, "side_conditions": {}}
    vec = StateEncoder().encode_snapshot(snap, turn=1)
    return float(vec[_OUR_A_SPEED])


def test_magic_room_suppresses_scarf_in_turn_order_speed():
    no_mr = _our_a_turnorder_speed(magic_room=False)   # Scarf active → ×1.5
    mr = _our_a_turnorder_speed(magic_room=True)        # Scarf suppressed → base speed
    assert no_mr > 0.0
    assert mr < no_mr                                   # Magic Room drops the ×1.5 Scarf boost
    # spe 200: no-MR = 200×1.5/600 = 0.5 ; MR = 200/600 ≈ 0.333
    assert abs(no_mr - 200 * 1.5 / 600.0) < 1e-6
    assert abs(mr - 200 / 600.0) < 1e-6
