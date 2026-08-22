"""v11 B1 (Mimikyu Disguise first-hit band-zero) + P1 (-formechange parser handler).

P1: the parser had no `-formechange` case → Aegislash Stance Change (Blade↔Shield base-stat swap) was dropped
    offline while live (poke-env) tracked it. Fix: route `-formechange` to `_handle_mega`'s non-mega species
    mutation. (No switch-in species reset: poke-env's `_update_from_details` early-returns on identical switch
    details so it KEEPS the forme stats on a silent revert — a reset would CREATE drift, verified empirically.)
B1: an intact Mimikyu Disguise blocks the first damaging hit (band → 0). Keyed on ability + full HP (the
    busted species is invisible to poke-env, so the species flip is offline-only and would break parity). Mold
    Breaker bypasses. Ice Face/Eiscue NOT modelled (HP-free bust breaks the hp-frac proxy; absent from the meta).
Design: scratch/v11_b1p1_*; the §1b switch-reset + Ice Face were dropped after empirical poke-env verification.
"""
from __future__ import annotations

from v_dance.encoders.state_encoder import (
    StateEncoder, MOVE_FEATURES, _MOVE_BLOCK_REL, move_slots_for_mon, norm_species, _disguise_intact,
)
from v_dance.parser.vod_parser.replay_parser import ShowdownReplayParser


# ════════════════════════════ B1 unit: _disguise_intact ════════════════════════════
def test_disguise_intact_predicate():
    assert _disguise_intact("disguise", 1.0) is True
    assert _disguise_intact("disguise", 0.875) is False    # busted (lost 1/8) → not intact
    assert _disguise_intact("disguise", None) is False
    assert _disguise_intact("levitate", 1.0) is False       # wrong ability
    assert _disguise_intact(None, 1.0) is False


# ════════════════════════════ B1 e2e: band-zero vs intact Disguise ════════════════════════════
def _att(ability, *, moves):
    return {"species": "Iron Valiant", "base_species": "Iron Valiant", "hp_pct": 100.0, "seen": True,
            "is_fainted": False, "known_moves": list(moves), "revealed_moves": [], "boosts": {},
            "status": None, "known_ability": ability, "volatiles": {},
            "stats_estimate": {"mode": "exact",
                               "stats": {"atk": 130, "spa": 130, "def": 90, "spd": 90, "hp": 200, "spe": 130}}}


def _mimikyu(hp_pct):
    return {"species": "Mimikyu", "base_species": "Mimikyu", "hp_pct": hp_pct, "seen": True,
            "is_fainted": False, "known_moves": [], "revealed_moves": [], "boosts": {}, "status": None,
            "known_ability": "Disguise", "volatiles": {},
            "stats_estimate": {"mode": "exact",
                               "stats": {"atk": 90, "spa": 50, "def": 80, "spd": 105, "hp": 165, "spe": 96}}}


def _move_block(att, move, mimikyu_hp):
    snap = {"our_active": {"our_a": att, "our_b": None},
            "opp_active": {"opp_a": _mimikyu(mimikyu_hp), "opp_b": None},
            "our_bench": [], "opp_bench": [], "field": {}, "side_conditions": {}}
    vec = StateEncoder().encode_snapshot(snap, turn=1)
    for m_idx, (mv, _c) in enumerate(move_slots_for_mon(att)):
        if norm_species(mv) == norm_species(move):
            b = _MOVE_BLOCK_REL + m_idx * MOVE_FEATURES
            # signed/immune type-eff vs opp_a at b+9/b+10; damage band vs opp_a at b+13/b+14
            return {"signed": float(vec[b + 9]), "immune": float(vec[b + 10]),
                    "dmin": float(vec[b + 13]), "dmax": float(vec[b + 14])}
    raise AssertionError(move)


def test_b1_intact_disguise_zeros_band():
    # Shadow Ball (Ghost, super-effective vs Ghost/Fairy Mimikyu) vs a FULL-HP intact Disguise → band 0.
    r = _move_block(_att("Quark Drive", moves=["Shadow Ball"]), "Shadow Ball", 100.0)
    assert r["dmin"] == 0.0 and r["dmax"] == 0.0


def test_b1_busted_disguise_takes_real_damage():
    # a chipped/busted Mimikyu (<1.0 HP) → real band (the disguise no longer blocks).
    r = _move_block(_att("Quark Drive", moves=["Shadow Ball"]), "Shadow Ball", 87.0)
    assert r["dmax"] > 0.0


def test_b1_mold_breaker_bypasses_disguise():
    # Mold Breaker attacker → Disguise is suppressed → real band even vs a full-HP Mimikyu.
    r = _move_block(_att("Mold Breaker", moves=["Shadow Ball"]), "Shadow Ball", 100.0)
    assert r["dmax"] > 0.0


def test_b1_type_eff_channels_untouched():
    # B1 zeros only the BAND — the move still connects type-wise (super-effective), NOT a type immunity.
    r = _move_block(_att("Quark Drive", moves=["Shadow Ball"]), "Shadow Ball", 100.0)
    assert r["immune"] == 0.0          # NOT flagged immune
    assert r["signed"] > 0.0           # Ghost vs Ghost/Fairy reads super-effective
    assert r["dmax"] == 0.0            # …but the band is zeroed (intact Disguise blocks this hit)


# ════════════════════════════ P1: -formechange parser handler ════════════════════════════
_HDR = """\
|player|p1|alice|101|1500
|player|p2|bob|102|1500
|gen|9
|tier|[Gen 9] Test Doubles
|poke|p1|Aegislash, L50, M|
|poke|p1|Garchomp, L50, M|
|poke|p2|Kingambit, L50, M|
|poke|p2|Rotom-Wash, L50|
|teamsize|p1|2
|teamsize|p2|2
|start
|switch|p1a: Aegislash|Aegislash, L50, M|100/100
|switch|p1b: Garchomp|Garchomp, L50, M|100/100
|switch|p2a: Kingambit|Kingambit, L50, M|100/100
|switch|p2b: Rotom|Rotom-Wash, L50|100/100
|turn|1
"""


def _parse(body):
    p = ShowdownReplayParser(_HDR + body, our_player="p1")
    p.parse()
    return p


def test_p1_formechange_mutates_species():
    p = _parse("|-formechange|p1a: Aegislash|Aegislash-Blade\n|turn|2\n|win|alice\n")
    aeg = next(m for k, m in p.active_slots.items() if m and m.species.startswith("Aegislash"))
    assert aeg.species == "Aegislash-Blade"     # Stance Change tracked offline now
    assert aeg.base_species == "Aegislash"      # base frozen (Pikalytics spread key)
    assert aeg.is_mega is False                 # NOT a mega


def test_p1_formechange_reversion():
    p = _parse("|-formechange|p1a: Aegislash|Aegislash-Blade\n|turn|2\n"
               "|-formechange|p1a: Aegislash|Aegislash\n|turn|3\n|win|alice\n")
    aeg = next(m for k, m in p.active_slots.items() if m and m.species.startswith("Aegislash"))
    assert aeg.species == "Aegislash"           # explicit revert restores Shield forme
