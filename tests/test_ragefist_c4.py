"""v11 Phase C.4 — Rage Fist base power = min(350, 50 + 50 × times_attacked).

times_attacked = a per-stint counter mirroring Showdown Pokemon.timesAttacked: +1 per DIRECT damaging-move
|-damage| hit (chip/recoil/confusion/Stealth-Rock/Future-Sight carry [from] → excluded; bare self-cost lines
guarded), reset on switch-in (offline twin of clearVolatile). VALUE-ONLY (fed through the existing
variable_base_power → damage band, like Last Respects' fainted_allies — NO new channel, NO layout change).
LIVE parity = Architecture B: the own-side count rides the SAME offline parse that builds opp_snapshot (no
custom battle), so both sides come from ONE filter (design workflow wf_9c7f4f4a-1bd).
"""
from __future__ import annotations

from v_dance.parser.vod_parser.replay_parser import ShowdownReplayParser
from v_dance.encoders import damage_mechanics as _DMG
from v_dance.encoders.state_encoder import (
    StateEncoder, get_state_dim, get_state_layout_version, POKEMON_FEATURES,
    MOVE_FEATURES, _MOVE_BLOCK_REL, move_slots_for_mon, norm_species,
)

HEADER = """|player|p1|A|1|
|player|p2|B|1|
|poke|p1|Annihilape, L50, M|
|poke|p1|Garchomp, L50, M|
|poke|p2|Garchomp, L50, M|
|teamsize|p1|2
|teamsize|p2|1
|start
|switch|p1a: Annihilape|Annihilape, L50, M|100/100
|switch|p2a: Garchomp|Garchomp, L50, M|100/100
"""


def _parse(body):
    return ShowdownReplayParser(HEADER + body + "|turn|999\n", our_player="p1").parse()


def _ta(result, n, rel="our_a"):
    for t in result["turns"]:
        if t["turn"] == n:
            return t["state_before_actions"]["p1"]["our_active"][rel]["times_attacked"]
    raise AssertionError(f"turn {n} not found")


# ════════════════════════════ parser counter ════════════════════════════
def test_single_hit_increments():
    r = _parse("|turn|1\n|move|p2a: Garchomp|Earthquake|p1a: Annihilape\n"
               "|-damage|p1a: Annihilape|70/100\n|turn|2\n")
    assert _ta(r, 1) == 0
    assert _ta(r, 2) == 1


def test_multihit_counts_per_hit():
    # a 3-hit move emits one bare |-damage| per hit → +3
    r = _parse("|turn|1\n|move|p2a: Garchomp|Bullet Seed|p1a: Annihilape\n"
               "|-damage|p1a: Annihilape|90/100\n|-damage|p1a: Annihilape|80/100\n"
               "|-damage|p1a: Annihilape|70/100\n|-hitcount|p1a: Annihilape|3\n|turn|2\n")
    assert _ta(r, 2) == 3


def test_chip_and_indirect_do_not_count():
    # all of these carry [from] → excluded
    r = _parse("|turn|1\n"
               "|move|p2a: Garchomp|Stealth Rock|p2a: Garchomp\n"
               "|-damage|p1a: Annihilape|88/100|[from] Stealth Rock\n"
               "|-damage|p1a: Annihilape|82/100|[from] psn\n"
               "|-damage|p1a: Annihilape|76/100|[from] item: Rocky Helmet|[of] p2a: Garchomp\n"
               "|-damage|p1a: Annihilape|70/100|[from] confusion\n"
               "|-damage|p1a: Annihilape|60/100|[from] move: Future Sight\n|turn|2\n")
    assert _ta(r, 2) == 0


def test_substitute_self_cost_guarded():
    # Substitute creation damages the USER with a bare |-damage| (no [from]) → must NOT count
    r = _parse("|turn|1\n|move|p1a: Annihilape|Substitute|p1a: Annihilape\n"
               "|-damage|p1a: Annihilape|75/100\n|turn|2\n")
    assert _ta(r, 2) == 0


def test_reset_on_switch_in():
    r = _parse("|turn|1\n|move|p2a: Garchomp|Earthquake|p1a: Annihilape\n"
               "|-damage|p1a: Annihilape|70/100\n"
               "|switch|p1a: Garchomp|Garchomp, L50, M|100/100\n"   # Annihilape out
               "|turn|2\n|switch|p1a: Annihilape|Annihilape, L50, M|100/100\n"  # back in → reset
               "|turn|3\n")
    assert _ta(r, 3) == 0


def test_drag_resets():
    r = _parse("|turn|1\n|move|p2a: Garchomp|Earthquake|p1a: Annihilape\n"
               "|-damage|p1a: Annihilape|70/100\n"
               "|drag|p1a: Garchomp|Garchomp, L50, M|100/100\n|turn|2\n"
               "|drag|p1a: Annihilape|Annihilape, L50, M|100/100\n|turn|3\n")
    assert _ta(r, 3) == 0


def test_ally_hit_counts():
    # a spread move hitting your OWN ally still increments the ally (target != user) — not self-cost.
    # (singles header has no ally; assert the guard is user-specific, not target-specific, via direct hit.)
    r = _parse("|turn|1\n|move|p2a: Garchomp|Rock Slide|p1a: Annihilape\n"
               "|-damage|p1a: Annihilape|80/100\n|turn|2\n")
    assert _ta(r, 2) == 1


# ════════════════════════════ BP formula (damage_mechanics) ════════════════════════════
def test_rage_fist_bp_formula():
    assert _DMG.variable_base_power("ragefist", 50, times_hit=0) == 50.0
    assert _DMG.variable_base_power("ragefist", 50, times_hit=1) == 100.0
    assert _DMG.variable_base_power("ragefist", 50, times_hit=3) == 200.0
    assert _DMG.variable_base_power("ragefist", 50, times_hit=6) == 350.0
    assert _DMG.variable_base_power("ragefist", 50, times_hit=9) == 350.0   # cap


# ════════════════════════════ encoder band (value-only, no layout) ════════════════════════════
def _mon(species, ability, *, moves=(), times_attacked=0):
    return {"species": species, "base_species": species, "hp_pct": 100.0, "seen": True,
            "is_fainted": False, "known_moves": list(moves), "revealed_moves": [], "boosts": {},
            "status": None, "known_ability": ability, "volatiles": {}, "times_attacked": times_attacked,
            "stats_estimate": {"mode": "exact",
                               "stats": {"atk": 120, "spa": 90, "def": 110, "spd": 110,
                                         "hp": 280, "spe": 60}}}


def _rage_dmax(times_attacked):
    att = _mon("Annihilape", "Defiant", moves=["Rage Fist"], times_attacked=times_attacked)
    snap = {"our_active": {"our_a": att, "our_b": None},
            "opp_active": {"opp_a": _mon("Garchomp", "Rough Skin", moves=[]), "opp_b": None},
            "our_bench": [], "opp_bench": [], "field": {}, "side_conditions": {}}
    vec = StateEncoder().encode_snapshot(snap, turn=1)
    for m_idx, (mv, _c) in enumerate(move_slots_for_mon(att)):
        if norm_species(mv) == "ragefist":
            return float(vec[_MOVE_BLOCK_REL + m_idx * MOVE_FEATURES + 13 + 1])
    raise AssertionError("Rage Fist")


def test_encoder_band_scales_with_times_attacked():
    assert _rage_dmax(0) < _rage_dmax(3) < _rage_dmax(6)


def test_layout_unchanged_value_only():
    # C.4 adds NO channel; layout later bumped to v16 by B2b (+2 per-move hit-chance channels).
    assert get_state_dim() == 5057
    assert get_state_layout_version() == 19
    assert POKEMON_FEATURES == 413
