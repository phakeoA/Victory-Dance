"""v11 Phase C.2e — GRAVITY.

Gravity was previously excluded as a parity shortcut (the offline parser did not track it). It IS a real
strategy (Gravity Hypnosis = no-miss sleep; Gravity grounds Flying/Levitate so EQ/Spikes/Arena Trap all
hit). Trick Room is also a pseudo-weather tracked offline via the same -fieldstart hook, so Gravity is
trackable byte-identically. C.2e: (1) parser tracks gravity (base 5 / 7 persistent); (2) the live FIELD
block is RESTRICTED to offline-tracked channels (fixes a latent train/serve drift where live emitted
GRAVITY/Magic-Room/etc. that offline never set); (3) Gravity grounds every active (Arena Trap, terrain,
Ground-immunity); (4) type-immunity negation (Ground-vs-Flying / Levitate, via C.2d); (5) accuracy ×6840/4096
(numeric only, capped at 1.0); (6) the GRAVITY presence channel (already in FIELD_NAMES) is lit. Value-only.
"""
from __future__ import annotations

from v_dance.parser.vod_parser.replay_parser import ShowdownReplayParser
from v_dance.encoders.state_encoder import (
    StateEncoder, POKEMON_FEATURES, VOLATILE_FEATURES, NUM_TYPES, NUM_WEATHER, _FIELD_IDX,
    MOVE_FEATURES, _MOVE_BLOCK_REL, move_slots_for_mon, norm_species, _GRAVITY_ACC_MULT,
)
from v_dance.encoders.live_state_encoder import LiveStateEncoder

# GRAVITY presence channel: field block sits after 12 mon blocks + the weather block.
_GRAVITY_CH = 12 * POKEMON_FEATURES + NUM_WEATHER + _FIELD_IDX["GRAVITY"]
# can_switch = 5th slot of the volatile block (mirror test_ability_trap_b3).
CAN_SWITCH = POKEMON_FEATURES - 4 - NUM_TYPES - VOLATILE_FEATURES + 4   # v11 Phase D D9: tera one-hot before flags

HEADER = """|player|p1|Alice|1|
|player|p2|Bob|1|
|poke|p1|Garchomp, L50, M|
|poke|p2|Talonflame, L50, M|
|teamsize|p1|1
|teamsize|p2|1
|start
|switch|p1a: Garchomp|Garchomp, L50, M|100/100
|switch|p2a: Talonflame|Talonflame, L50, M|100/100
"""


def _parse(body):
    return ShowdownReplayParser(HEADER + body + "|turn|999\n", our_player="p1").parse()


def _field_at(result, turn):
    for t in result["turns"]:
        if t["turn"] == turn:
            return t["state_before_actions"]["p1"]["field"]
    raise AssertionError(f"turn {turn} not found")


# ════════════════════════════ parser tracking ════════════════════════════
def test_parser_tracks_gravity():
    r = _parse("|-fieldstart|move: Gravity\n|turn|2\n")
    assert _field_at(r, 2)["gravity_turns_remaining"] == 5


def test_parser_gravity_persistent_is_7():
    r = _parse("|-fieldstart|move: Gravity|[persistent]\n|turn|2\n")
    assert _field_at(r, 2)["gravity_turns_remaining"] == 7


def test_parser_gravity_decrements_on_upkeep():
    # one |upkeep| between set and the snapshot → 5 → 4.
    r = _parse("|-fieldstart|move: Gravity\n|upkeep\n|turn|2\n")
    assert _field_at(r, 2)["gravity_turns_remaining"] == 4


def test_parser_gravity_ends():
    r = _parse("|-fieldstart|move: Gravity\n|turn|2\n|-fieldend|move: Gravity\n|turn|3\n")
    assert _field_at(r, 2)["gravity_turns_remaining"] == 5
    assert _field_at(r, 3)["gravity_turns_remaining"] == 0


# ════════════════════════════ live field-block restriction (latent-bug fix) ════════════════════════════
def test_live_offline_field_idx_restricts_to_tracked():
    enc = LiveStateEncoder()
    if not hasattr(enc, "_offline_field_idx"):
        import pytest
        pytest.skip("poke-env not importable — live field index set not built")
    allowed = enc._offline_field_idx
    # v11 P5: MAGIC_ROOM / WONDER_ROOM are now offline-tracked → they JOINED the live allow-list.
    for n in ("ELECTRIC_TERRAIN", "GRASSY_TERRAIN", "MISTY_TERRAIN", "PSYCHIC_TERRAIN",
              "TRICK_ROOM", "GRAVITY", "MAGIC_ROOM", "WONDER_ROOM"):
        assert _FIELD_IDX[n] in allowed, f"{n} should be live-emitted (offline tracks it)"
    # The channels offline never sets must NOT be emitted live (the latent-drift guard).
    for n in ("HEAL_BLOCK", "FAIRY_LOCK", "NEUTRALIZING_GAS", "UNKNOWN"):
        if n in _FIELD_IDX:
            assert _FIELD_IDX[n] not in allowed, f"{n} must NOT be live-emitted (offline never sets it)"


# ════════════════════════════ end-to-end through the encoder ════════════════════════════
def _mon(species, ability, *, moves=(), volatiles=None, item=None):
    m = {
        "species": species, "base_species": species, "hp_pct": 100.0, "seen": True,
        "is_fainted": False, "known_moves": list(moves), "revealed_moves": [], "boosts": {},
        "status": None, "known_ability": ability, "volatiles": volatiles or {},
        "stats_estimate": {"mode": "exact",
                           "stats": {"atk": 130, "spa": 130, "def": 100, "spd": 100,
                                     "hp": 260, "spe": 80}},
    }
    if item is not None:
        m["known_item"] = item
    return m


def _vec(att, dfn, field, *, swap=False):
    a, b = (dfn, att) if swap else (att, dfn)
    snap = {"our_active": {"our_a": a, "our_b": None}, "opp_active": {"opp_a": b, "opp_b": None},
            "our_bench": [], "opp_bench": [], "field": field, "side_conditions": {}}
    return StateEncoder().encode_snapshot(snap, turn=3)


def _chan(att, dfn, move, field):
    """(signed, immune, dmax, accuracy) for `move` of our_a vs opp_a."""
    vec = _vec(att, dfn, field)
    for m_idx, (mv, _c) in enumerate(move_slots_for_mon(att)):
        if norm_species(mv) == norm_species(move):
            b = _MOVE_BLOCK_REL + m_idx * MOVE_FEATURES
            return (float(vec[b + 9]), float(vec[b + 10]), float(vec[b + 13 + 1]), float(vec[b + 4]))
    raise AssertionError(move)


GRAV = {"gravity_turns_remaining": 5}


def test_presence_channel():
    v_off = _vec(_mon("Garchomp", "Rough Skin"), _mon("Talonflame", "Flame Body"), {})
    v_on = _vec(_mon("Garchomp", "Rough Skin"), _mon("Talonflame", "Flame Body"), GRAV)
    assert v_off[_GRAVITY_CH] == 0.0
    assert v_on[_GRAVITY_CH] == 1.0


def test_grounds_flyer_eq_hits():
    chomp = _mon("Garchomp", "Rough Skin", moves=["Earthquake"])
    off = _chan(chomp, _mon("Talonflame", "Flame Body"), "Earthquake", {})
    on = _chan(chomp, _mon("Talonflame", "Flame Body"), "Earthquake", GRAV)
    assert off[1] == 1.0 and off[2] == 0.0            # Flying immune to Ground
    assert on[1] == 0.0 and on[2] > 0.0               # grounded → hits
    assert on[0] == 0.5                               # Fire ×2 preserved (Flying 0×→1×, not blanket)


def test_grounds_levitator_eq_hits():
    chomp = _mon("Garchomp", "Rough Skin", moves=["Earthquake"])
    off = _chan(chomp, _mon("Bronzong", "Levitate"), "Earthquake", {})
    on = _chan(chomp, _mon("Bronzong", "Levitate"), "Earthquake", GRAV)
    assert off[1] == 1.0 and off[2] == 0.0            # Levitate immune
    assert on[1] == 0.0 and on[2] > 0.0               # Gravity negates Levitate


def test_gravity_hypnosis_accuracy_caps():
    chomp = _mon("Garchomp", "Rough Skin", moves=["Hypnosis"])
    off = _chan(chomp, _mon("Talonflame", "Flame Body"), "Hypnosis", {})
    on = _chan(chomp, _mon("Talonflame", "Flame Body"), "Hypnosis", GRAV)
    assert abs(off[3] - 0.60) < 1e-6                  # Hypnosis base 60%
    assert on[3] == 1.0                               # 0.60 × 1.6699 → capped 1.0 (the no-miss payoff)


def test_gravity_accuracy_non_cap_scales():
    # Sing (55%) × 6840/4096 = 0.9182… (does not cap) — proves the exact rational, not a flat 1.0.
    chomp = _mon("Garchomp", "Rough Skin", moves=["Sing"])
    on = _chan(chomp, _mon("Talonflame", "Flame Body"), "Sing", GRAV)
    assert abs(on[3] - min(0.55 * _GRAVITY_ACC_MULT, 1.0)) < 1e-6


def test_gravity_always_hit_untouched():
    # Swift (accuracy: true) must stay 1.0 under Gravity — no spurious multiply.
    chomp = _mon("Garchomp", "Rough Skin", moves=["Swift"])
    off = _chan(chomp, _mon("Talonflame", "Flame Body"), "Swift", {})
    on = _chan(chomp, _mon("Talonflame", "Flame Body"), "Swift", GRAV)
    assert off[3] == 1.0 and on[3] == 1.0


def test_arena_trap_under_gravity():
    # Our Flyer vs opposing Arena Trap: free to switch normally (ungrounded), trapped under Gravity.
    flyer = _mon("Talonflame", "Flame Body", moves=["Brave Bird"])
    trapper = _mon("Dugtrio", "Arena Trap")
    off = _vec(flyer, trapper, {})
    on = _vec(flyer, trapper, GRAV)
    assert off[CAN_SWITCH] == 1.0                     # Flying ungrounded → Arena Trap can't trap
    assert on[CAN_SWITCH] == 0.0                      # Gravity grounds → trapped


def test_iron_ball_idempotent():
    # An Iron-Ball (already-grounded) Flyer is unaffected by Gravity — grounding channels identical.
    chomp = _mon("Garchomp", "Rough Skin", moves=["Earthquake"])
    flyer_off = _chan(chomp, _mon("Talonflame", "Flame Body", item="Iron Ball"), "Earthquake", {})
    flyer_on = _chan(chomp, _mon("Talonflame", "Flame Body", item="Iron Ball"), "Earthquake", GRAV)
    assert flyer_off == flyer_on                      # Iron Ball already grounded → no double-count


def test_legacy_invariance_no_gravity():
    # A snapshot with no gravity field → GRAVITY channel 0.0 and all the move channels unchanged.
    chomp = _mon("Garchomp", "Rough Skin", moves=["Earthquake"])
    flyer = _mon("Talonflame", "Flame Body")
    assert _chan(chomp, flyer, "Earthquake", {}) == _chan(chomp, flyer, "Earthquake",
                                                          {"gravity_turns_remaining": 0})
