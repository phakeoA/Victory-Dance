"""
Level C / B0d — variance-aware faint bracket (2026-06-30, fork-B reframe).

The stepper now carries per-mon distribution shadows (_hp_hi = luckiest/min-damage, _hp_lo = max non-crit,
_hp_floor = max + crit) so the B0d gate can judge KO-probability calibration + reachability instead of a binary
faint match against one realized roll. Tests: bracket present/monotonic, deterministic lockstep, heal lockstep,
undamaged mons stay shadow-free, mean HP unchanged (inside the bracket), and the probe P(KO) formula.
"""
from __future__ import annotations

import sys
from pathlib import Path

from v_dance.encoders import white_box_sim as W

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# scratch/ is a local-only dev-harness dir (gitignored) — skip cleanly where it's absent (CI)
pytest.importorskip("scratch.levelC_b0_validation_probe",
                    reason="scratch/ probe harness is local-only, not in the published repo")
from scratch.levelC_b0_validation_probe import _p_ko  # noqa: E402


def _mon(species, hp=100.0, spa=130, atk=130, spe=100, hp_stat=300, defn=130, item=None):
    return {
        "species": species, "base_species": species, "hp_pct": hp, "status": None, "is_fainted": False,
        "boosts": {}, "known_item": item, "known_ability": None, "mega_ability": None, "is_mega": False,
        "volatiles": {}, "revealed_moves": [], "times_attacked": 0,
        "stats_estimate": {"mode": "distribution",
                           "stats": {"hp": hp_stat, "atk": atk, "def": defn, "spa": spa, "spd": defn, "spe": spe}},
    }


def _state(our_a=None, opp_a=None, opp_b=None, field=None):
    return {
        "field": field or {"weather": None, "terrain": None, "trick_room_turns_remaining": 0},
        "side_conditions": {"our_side": {"tailwind_turns_remaining": 0, "screens": {}},
                            "opp_side": {"tailwind_turns_remaining": 0, "screens": {}}},
        "our_active": {"our_a": our_a, "our_b": None}, "opp_active": {"opp_a": opp_a, "opp_b": opp_b},
        "our_bench": [], "opp_bench": [],
    }


def test_bracket_present_and_monotonic():
    s = _state(_mon("Pikachu", spa=120), _mon("Snorlax", hp_stat=320, defn=120))
    nxt, _ = W.white_box_sim(s, {"our_a": {"kind": "move", "move": "Thunderbolt", "target": "opp_a"}})
    d = nxt["opp_active"]["opp_a"]
    assert "_hp_lo" in d and "_hp_hi" in d and "_hp_floor" in d
    # _hp_floor (max+crit) ≤ _hp_lo (max non-crit) ≤ mean hp_pct ≤ _hp_hi (min dmg) ≤ 100
    assert d["_hp_floor"] <= d["_hp_lo"] <= d["hp_pct"] <= d["_hp_hi"] <= 100.0
    assert d["_hp_hi"] < 100.0                    # took at least the minimum roll


def test_deterministic_damage_shifts_shadows_lockstep():
    m = {"hp_pct": 80.0, "is_fainted": False, "_hp_hi": 80.0, "_hp_lo": 60.0, "_hp_floor": 50.0}
    W.apply_damage_pct(m, 10.0)
    assert (m["hp_pct"], m["_hp_hi"], m["_hp_lo"], m["_hp_floor"]) == (70.0, 70.0, 50.0, 40.0)


def test_heal_shifts_shadows_up():
    m = {"hp_pct": 50.0, "is_fainted": False, "_hp_hi": 50.0, "_hp_lo": 30.0, "_hp_floor": 20.0}
    W.apply_heal_pct(m, 10.0)
    assert (m["hp_pct"], m["_hp_hi"], m["_hp_lo"], m["_hp_floor"]) == (60.0, 60.0, 40.0, 30.0)


def test_undamaged_mon_stays_shadow_free():
    s = _state(_mon("Pikachu"), _mon("Snorlax"), _mon("Garchomp"))   # opp_b never targeted
    nxt, _ = W.white_box_sim(s, {"our_a": {"kind": "move", "move": "Thunderbolt", "target": "opp_a"}})
    assert "_hp_lo" not in nxt["opp_active"]["opp_b"]                  # probe will default its bracket to hp_pct


def test_apply_damage_rolled_mean_matches_plain():
    # the mean path is byte-identical to a plain deterministic subtract of the same mean
    a = {"hp_pct": 100.0, "is_fainted": False}
    b = {"hp_pct": 100.0, "is_fainted": False}
    W.apply_damage_rolled(a, 30.0, 25.0, 35.0, 52.5)
    W.apply_damage_pct(b, 30.0)
    assert a["hp_pct"] == b["hp_pct"] == 70.0


def test_p_ko_formula():
    assert _p_ko(5.0, 20.0) == 0.0           # unluckiest roll survives → never KO
    assert _p_ko(-5.0, -1.0) == 1.0          # luckiest roll still faints → certain KO
    assert abs(_p_ko(-10.0, 20.0) - (10.0 / 30.0)) < 1e-9
    assert _p_ko(0.0, 0.0) == 1.0            # exactly 0 HP at best case → KO
    assert _p_ko(None, 50.0) == 0.0
