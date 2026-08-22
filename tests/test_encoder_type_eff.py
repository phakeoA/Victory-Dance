"""B1/#25: per-move type-effectiveness channels in the OFFLINE encoder (B1.1a).

The encoder now writes, in each move block, the resolved type multiplier vs each of the 2 ENEMY
actives — a signed log2(mult)/2 channel + a separate `immune` flag so 0× (no effect) stays DISTINCT
from 0.25× (both clamp to -1 on the continuous axis). Tera-aware (inactive in Reg M-A; prepared-for).
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pytest

from v_dance.encoders.state_encoder import (  # noqa: E402
    StateEncoder, _type_eff_signed_immune, _effective_types,
    POKEMON_FEATURES, MOVE_FEATURES, _get_moves_data, norm_species, move_slots_for_mon,
    _MOVE_BLOCK_REL,
)

# within-mon offset of the move block — canonical from the encoder (v9: 71, after the weight feature);
# type-eff sits after base_power..is_spread (0..8).
OFF_MOVES = _MOVE_BLOCK_REL
OFF_TYPEEFF = 9


def test_type_eff_known_matchups():
    assert _type_eff_signed_immune("electric", ["WATER"]) == (0.5, 0.0)          # 2× super
    assert _type_eff_signed_immune("electric", ["GROUND"]) == (-1.0, 1.0)        # 0× immune
    assert _type_eff_signed_immune("electric", ["GRASS"]) == (-0.5, 0.0)         # 0.5× resist
    assert _type_eff_signed_immune("fire", ["GRASS"]) == (0.5, 0.0)              # 2×
    assert _type_eff_signed_immune("ice", ["DRAGON", "FLYING"]) == (1.0, 0.0)    # 4×
    assert _type_eff_signed_immune("normal", ["GHOST"]) == (-1.0, 1.0)           # 0× immune
    assert _type_eff_signed_immune("water", ["NORMAL"]) == (0.0, 0.0)            # 1× neutral
    assert _type_eff_signed_immune("electric", []) == (0.0, 0.0)                 # no defender → neutral


def test_025x_distinct_from_immune():
    """The user's exact concern: 0.25× (still chips) must NOT collapse onto 0× (no effect)."""
    quarter = _type_eff_signed_immune("fire", ["WATER", "ROCK"])   # 0.5×0.5 = 0.25×
    immune = _type_eff_signed_immune("ground", ["FLYING"])         # 0×
    assert quarter == (-1.0, 0.0)        # signed -1, NOT immune
    assert immune == (-1.0, 1.0)         # signed -1, immune flag set → distinct from 0.25×


def _first_transition():
    folder = (Path(__file__).resolve().parents[1] / "data" / "vods" /
              "Prepared_training_data" / "Regulation_MA" / "Jsonl_TypeB")
    for fp in sorted(glob.glob(str(folder / "**" / "*.jsonl"), recursive=True))[:30]:
        with open(fp, encoding="utf-8") as fh:
            for line in fh:
                t = json.loads(line)
                sba = t.get("state_before_actions") or {}
                oa = (sba.get("our_active") or {}).get("our_a")
                opp = (sba.get("opp_active") or {}).get("opp_a")
                if oa and not oa.get("is_fainted") and opp and not opp.get("is_fainted"):
                    return t
    return None


def test_offline_encodes_type_eff_vs_opp_active():
    """Integration: encode a REAL state; our_a's move type-eff channels (vs opp_a = enemy 0) must
    match the chart computed INDEPENDENTLY from poke-env (not via the encoder's own helper)."""
    t = _first_transition()
    if t is None:
        pytest.skip("no Reg-MA TypeB corpus")
    from poke_env.data import GenData
    chart = GenData.from_gen(9).type_chart

    def expected(mtype, dtypes):
        mult = 1.0
        for dt in dtypes:
            mult *= float(chart.get(dt.upper(), {}).get(mtype.upper(), 1.0))
        return (-1.0, 1.0) if mult <= 0 else (round(float(np.clip(np.log2(mult) / 2.0, -1, 1)), 5), 0.0)

    sba = t["state_before_actions"]
    x = StateEncoder().encode_snapshot(sba, turn=t.get("turn") or 0)
    opp_a_types = _effective_types(sba["opp_active"]["opp_a"])
    our_a = sba["our_active"]["our_a"]
    checked = 0
    for m_idx, (mv, _c) in enumerate(move_slots_for_mon(our_a)):
        data = _get_moves_data().get(norm_species(mv))
        if not data or not data.get("type"):
            continue
        base = OFF_MOVES + m_idx * MOVE_FEATURES + OFF_TYPEEFF   # slot 0 (our_a) starts at index 0
        got = (round(float(x[base]), 5), float(x[base + 1]))     # signed + immune vs enemy 0 (opp_a)
        assert got == expected(data["type"], opp_a_types), (mv, got)
        checked += 1
    assert checked > 0


# ── B1.2 damage band ──────────────────────────────────────────────────────────
OFF_DAMAGE = 13   # within a move block: after type-eff (9-12); is_known is last (17)


def test_damage_band_formula():
    from v_dance.encoders.state_encoder import _damage_band
    # bp100, A150, D100, hp_stat200, full HP, neutral type, STAB, single-target:
    # base = ((2*50/5+2)*100*1.5)/50 + 2 = 68; ×1.5 STAB = 102; /200 cur_hp = 0.51 max, 0.4335 min.
    lo, hi = _damage_band(100, 150, 100, 200, 1.0, 1.0, True, False)
    assert hi == pytest.approx(0.51, abs=1e-3) and lo == pytest.approx(0.4335, abs=1e-3)
    assert _damage_band(0, 150, 100, 200, 1.0, 1.0, True, False) == (0.0, 0.0)      # status (bp 0)
    assert _damage_band(100, None, 100, 200, 1.0, 1.0, True, False) == (0.0, 0.0)   # missing belief
    assert _damage_band(100, 150, 100, 200, 1.0, 0.0, True, False) == (0.0, 0.0)    # hp_frac 0
    assert _damage_band(100, 150, 100, 200, 0.0, 1.0, True, False) == (0.0, 0.0)    # type-immune (mult 0)
    spread = _damage_band(100, 150, 100, 200, 1.0, 1.0, False, True)                # spread 0.75×, no STAB
    assert spread[1] == pytest.approx(68 * 0.75 / 200, abs=1e-3)


def test_offline_damage_band_invariants():
    t = _first_transition()
    if t is None:
        pytest.skip("no Reg-MA TypeB corpus")
    sba = t["state_before_actions"]
    x = StateEncoder().encode_snapshot(sba, turn=t.get("turn") or 0)
    our_a = sba["our_active"]["our_a"]
    for m_idx, (mv, _c) in enumerate(move_slots_for_mon(our_a)):
        data = _get_moves_data().get(norm_species(mv))
        if not data:
            continue
        base = OFF_MOVES + m_idx * MOVE_FEATURES + OFF_DAMAGE
        for off in (0, 2):                                   # vs enemy0, enemy1
            dmin, dmax = float(x[base + off]), float(x[base + off + 1])
            assert 0.0 <= dmin <= dmax <= 1.0               # band ordered + clamped
        if (data.get("category") or "") == "status" or (data.get("basePower") or 0) <= 0:
            assert float(x[base]) == 0.0 and float(x[base + 1]) == 0.0   # status → no damage


# ── B1.3 move intrinsics ──────────────────────────────────────────────────────
OFF_INTRINSICS = 19   # within a move block: after damage (13-16) + v11 B.1b moves-first (17-18); is_known last (53)


def test_offline_move_intrinsics():
    t = _first_transition()
    if t is None:
        pytest.skip("no Reg-MA TypeB corpus")
    sba = t["state_before_actions"]
    x = StateEncoder().encode_snapshot(sba, turn=t.get("turn") or 0)
    our_a = sba["our_active"]["our_a"]
    checked = 0
    for m_idx, (mv, _c) in enumerate(move_slots_for_mon(our_a)):
        data = _get_moves_data().get(norm_species(mv))
        if not data:
            continue
        base = OFF_MOVES + m_idx * MOVE_FEATURES + OFF_INTRINSICS
        flags = data.get("flags") or {}
        assert float(x[base]) == (1.0 if flags.get("contact") else 0.0)        # contact
        assert float(x[base + 1]) == (1.0 if data.get("recoil") else 0.0)      # recoil
        assert float(x[base + 2]) == (1.0 if data.get("drain") else 0.0)       # drain
        assert 0.0 <= float(x[base + 3]) <= 1.0                                # multihit-count/5
        checked += 1
    assert checked > 0


# ── B1.4 turn-order (effective speed + moves-first margin) ────────────────────
def test_moves_first_and_effective_speed():
    from v_dance.encoders.state_encoder import _moves_first, _effective_speed
    # moves-first: faster 'a' → positive; Trick Room flips the sign; a tie / missing speed → 0.
    assert _moves_first(200, 100, False) == pytest.approx(np.tanh(np.log(2)), abs=1e-4)
    assert _moves_first(200, 100, True) == pytest.approx(-np.tanh(np.log(2)), abs=1e-4)
    assert _moves_first(100, 100, False) == 0.0
    assert _moves_first(0, 100, False) == 0.0
    # effective speed: base; Choice Scarf ×1.5; Tailwind ×2; paralysis ×0.5; +1 boost ×1.5.
    base = {"stats_estimate": {"mode": "exact", "stats": {"spe": 100}}, "boosts": {}, "status": None}
    assert _effective_speed(base, False) == (100.0, 1.0)
    assert _effective_speed(base, True)[0] == pytest.approx(200.0)                     # Tailwind
    scarf = {"stats_estimate": {"mode": "exact", "stats": {"spe": 100}},
             "known_item": "Choice Scarf", "boosts": {}}
    assert _effective_speed(scarf, False)[0] == pytest.approx(150.0)                   # Scarf
    par = {"stats_estimate": {"mode": "exact", "stats": {"spe": 100}}, "status": "par", "boosts": {}}
    assert _effective_speed(par, False)[0] == pytest.approx(50.0)                      # paralysis
    boosted = {"stats_estimate": {"mode": "exact", "stats": {"spe": 100}}, "boosts": {"spe": 1}}
    assert _effective_speed(boosted, False)[0] == pytest.approx(150.0)                 # +1 boost
    assert _effective_speed(None, False) == (0.0, 0.0)


# ── B1.2b situational damage modifiers + weather-speed ability ────────────────
def test_situational_damage_mult():
    from v_dance.encoders.state_encoder import _situational_damage_mult as S
    g = {"grounded": True, "screen_phys": False, "screen_spec": False}
    assert S("FIRE", False, "SUNNYDAY", None, g, False, False, False) == pytest.approx(1.5)   # sun→fire
    assert S("WATER", False, "SUNNYDAY", None, g, False, False, False) == pytest.approx(0.5)  # sun↓water
    assert S("WATER", False, "RAINDANCE", None, g, False, False, False) == pytest.approx(1.5) # rain→water
    assert S("ELECTRIC", False, None, "ELECTRIC_TERRAIN", g, False, False, False) == pytest.approx(1.3)
    assert S("ELECTRIC", False, None, "ELECTRIC_TERRAIN", {"grounded": False}, False, False, False) == 1.0
    phys_scr = {"grounded": True, "screen_phys": True, "screen_spec": False}
    assert S("NORMAL", True, None, None, phys_scr, False, False, False) == pytest.approx(0.667)
    assert S("NORMAL", False, None, None, phys_scr, False, False, False) == 1.0                # spec unaffected
    assert S("NORMAL", True, None, None, g, False, True, False) == pytest.approx(1.3)          # Life Orb
    assert S("NORMAL", True, None, None, g, False, False, True) == pytest.approx(1.5)          # Choice
    assert S("NORMAL", True, None, None, g, True, False, False) == pytest.approx(0.5)          # burn (phys)
    assert S("NORMAL", False, None, None, g, True, False, False) == 1.0                        # burn no-op (spec)


def test_weather_speed_ability():
    from v_dance.encoders.state_encoder import _effective_speed
    swift = {"stats_estimate": {"mode": "exact", "stats": {"spe": 100}},
             "known_ability": "Swift Swim", "boosts": {}}
    assert _effective_speed(swift, False, "RAINDANCE")[0] == pytest.approx(200.0)   # ×2 in rain
    assert _effective_speed(swift, False, "SUNNYDAY")[0] == pytest.approx(100.0)    # no boost off-weather
    assert _effective_speed(swift, False, None)[0] == pytest.approx(100.0)          # no weather
