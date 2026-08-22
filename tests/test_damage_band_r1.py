"""v11 Phase B.2 — damage-band R1: variable base power (#6) + stat-override (#7).

variable_base_power (Low Kick/Heavy Slam/Gyro Ball/Eruption/Stored Power/Punishment) is now fed into the
band (was never called → static 0/1 BP); the 5 override moves use the right stat (Body Press = user Def,
Foul Play = target Atk, Psyshock/Psystrike/Secret Sword = target Def). Value-only, no layout change.
Verified vs damage_mechanics.py + moves.ts (design workflow wf_9c6ed6a6-82a).
"""
from __future__ import annotations

import math

from v_dance.encoders.state_encoder import (
    StateEncoder, MOVE_FEATURES, _MOVE_BLOCK_REL, move_slots_for_mon, norm_species,
    _situational_damage_mult, _DEF_HIT_MOVES,
)
from v_dance.encoders import damage_mechanics as _DMG

OFF_DAMAGE = 13   # band min/max vs enemy0 within a move block

# a high-bulk defender keeps bands under the [0,1] clamp so ratios/inequalities are meaningful
_TANK = {"atk": 80, "spa": 80, "def": 300, "spd": 300, "hp": 700, "spe": 40}


def _mon(species, ability, *, moves=(), hp_pct=100.0, stats=None, boosts=None):
    return {
        "species": species, "base_species": species, "hp_pct": hp_pct,
        "seen": True, "is_fainted": False, "known_moves": list(moves), "revealed_moves": [],
        "boosts": boosts or {}, "status": None, "known_ability": ability,
        "stats_estimate": {"mode": "exact",
                           "stats": stats or {"atk": 100, "spa": 100, "def": 100,
                                              "spd": 100, "hp": 200, "spe": 100}},
    }


def _dmax(attacker, defender, move):
    snap = {"our_active": {"our_a": attacker, "our_b": None},
            "opp_active": {"opp_a": defender, "opp_b": None},
            "our_bench": [], "opp_bench": [], "field": {}, "side_conditions": {}}
    vec = StateEncoder().encode_snapshot(snap, turn=1)
    for m_idx, (mv, _c) in enumerate(move_slots_for_mon(attacker)):
        if norm_species(mv) == norm_species(move):
            b = _MOVE_BLOCK_REL + m_idx * MOVE_FEATURES + OFF_DAMAGE
            return float(vec[b + 1])
    raise AssertionError(f"{move} not in attacker slots")


def _close(a, b):
    return math.isclose(a, b, rel_tol=2e-3, abs_tol=1e-6)


# ── variable BP is now actually called ────────────────────────────────────────
def test_variable_base_power_is_wired():
    # sanity that the function still no-ops a fixed-BP move and scales a variable one
    assert _DMG.variable_base_power("earthquake", 100) == 100.0
    assert _DMG.variable_base_power("eruption", 150, attacker_hp_frac=0.5) == 75.0


# ── B.2 gap-fix: Last Respects BP = 50 + 50×(side faints) — was floored at 50 (un-deferred 2026-06-21) ──
def test_last_respects_bp_formula():
    assert _DMG.variable_base_power("lastrespects", 50, fainted_allies=0) == 50.0
    assert _DMG.variable_base_power("lastrespects", 50, fainted_allies=2) == 150.0
    assert _DMG.variable_base_power("lastrespects", 50, fainted_allies=3) == 200.0


def test_e2e_last_respects_scales_with_side_faints():
    def _faint(n):
        return [_mon(f"Filler{i}", "Levitate") | {"is_fainted": True} for i in range(n)]

    def _lr_dmax(n_faint):
        snap = {"our_active": {"our_a": _mon("Houndstone", "Sand Rush", moves=["Last Respects"]),
                               "our_b": None},
                "opp_active": {"opp_a": _mon("Garchomp", "Rough Skin", stats=_TANK), "opp_b": None},
                "our_bench": _faint(n_faint), "opp_bench": [], "field": {}, "side_conditions": {}}
        vec = StateEncoder().encode_snapshot(snap, turn=1)
        for m_idx, (mv, _c) in enumerate(move_slots_for_mon(snap["our_active"]["our_a"])):
            if norm_species(mv) == "lastrespects":
                return float(vec[_MOVE_BLOCK_REL + m_idx * MOVE_FEATURES + OFF_DAMAGE + 1])
        raise AssertionError("Last Respects")

    d = [_lr_dmax(n) for n in range(4)]
    assert d[0] < d[1] < d[2] < d[3]                       # 50 → 100 → 150 → 200 BP (monotone in faints)


def test_eruption_scales_with_hp():
    base = {"atk": 100, "spa": 145, "def": 70, "spd": 105, "hp": 180, "spe": 40}
    full = _mon("Camerupt", "Magma Armor", moves=["Eruption"], stats=base, hp_pct=100.0)
    half = _mon("Camerupt", "Magma Armor", moves=["Eruption"], stats=base, hp_pct=50.0)
    f, h = _dmax(full, _mon("Toxapex", "Regenerator", stats=_TANK), "Eruption"), \
           _dmax(half, _mon("Toxapex", "Regenerator", stats=_TANK), "Eruption")
    assert f > 0 and 0 < h < f
    assert f * 0.45 < h < f * 0.6                   # BP ~halves with HP (not exact: the +2 L50 term)


def test_heavy_slam_and_low_kick_now_nonzero():
    # both have basePowerCallback (static 0) → before B.2 the band was 0.
    heavy = _mon("Hariyama", "Thick Fat", moves=["Heavy Slam"],
                 stats={"atk": 120, "spa": 60, "def": 100, "spd": 90, "hp": 254, "spe": 50})
    assert _dmax(heavy, _mon("Flutter Mane", "Protosynthesis", stats=_TANK), "Heavy Slam") > 0.0
    lk = _mon("Hariyama", "Thick Fat", moves=["Low Kick"],
              stats={"atk": 120, "spa": 60, "def": 100, "spd": 90, "hp": 254, "spe": 50})
    # both Normal-type (same 2× Fighting eff) so ONLY the weight differs (Low Kick scales with target wt).
    heavy_tgt = _mon("Snorlax", "Thick Fat", stats=_TANK)        # 460 kg → max Low Kick BP
    light_tgt = _mon("Rattata", "Run Away", stats=_TANK)         # 3.5 kg → min BP
    assert _dmax(lk, heavy_tgt, "Low Kick") > _dmax(lk, light_tgt, "Low Kick") > 0.0


def test_stored_power_scales_with_attacker_boosts():
    st = {"atk": 80, "spa": 130, "def": 90, "spd": 110, "hp": 160, "spe": 100}
    no_boost = _mon("Hatterene", "Magic Bounce", moves=["Stored Power"], stats=st)
    boosted = _mon("Hatterene", "Magic Bounce", moves=["Stored Power"], stats=st,
                   boosts={"spa": 2, "spe": 1})           # +3 positive stages → BP 20+60 = 80
    tank = _mon("Toxapex", "Regenerator", stats=_TANK)
    assert _dmax(boosted, tank, "Stored Power") > _dmax(no_boost, tank, "Stored Power")


def test_gyro_ball_slow_attacker_and_zero_speed_fallback():
    slow = _mon("Ferrothorn", "Iron Barbs", moves=["Gyro Ball"],
                stats={"atk": 110, "spa": 60, "def": 130, "spd": 120, "hp": 200, "spe": 20})
    fast = _mon("Dragapult", "Clear Body", stats={**_TANK, "spe": 142})
    assert _dmax(slow, fast, "Gyro Ball") > 0.0          # slow vs fast → high BP
    # a defender with unknown speed (no stats) → eff_speed 0 → falls back to static (0) → band 0
    no_spd = {"species": "Mew", "base_species": "Mew", "hp_pct": 100.0, "seen": False,
              "is_fainted": False, "known_moves": [], "revealed_moves": [], "boosts": {},
              "status": None, "known_ability": None, "stats_estimate": {"mode": "none", "stats": {}}}
    assert _dmax(slow, no_spd, "Gyro Ball") == 0.0


def test_rage_fist_floor_is_nonzero():
    # Rage Fist static BP is 0; with no times_hit tracked it floors at 50 BP → band nonzero (was 0).
    rf = _mon("Annihilape", "Defiant", moves=["Rage Fist"],
              stats={"atk": 115, "spa": 70, "def": 80, "spd": 90, "hp": 200, "spe": 90})
    assert _dmax(rf, _mon("Toxapex", "Regenerator", stats=_TANK), "Rage Fist") > 0.0


# ── stat-override ─────────────────────────────────────────────────────────────
def test_body_press_uses_user_def():
    tank = _mon("Toxapex", "Regenerator", stats=_TANK)
    hi_def = _mon("Stakataka", "Beast Boost", moves=["Body Press"],
                  stats={"atk": 60, "spa": 50, "def": 250, "spd": 100, "hp": 160, "spe": 10})
    lo_def = _mon("Stakataka", "Beast Boost", moves=["Body Press"],
                  stats={"atk": 60, "spa": 50, "def": 80, "spd": 100, "hp": 160, "spe": 10})
    assert _dmax(hi_def, tank, "Body Press") > _dmax(lo_def, tank, "Body Press") > 0.0


def test_foul_play_uses_target_atk():
    user = _mon("Spectrier", "Grim Neigh", moves=["Foul Play"],
                stats={"atk": 65, "spa": 145, "def": 60, "spd": 80, "hp": 180, "spe": 130})
    hi_atk = _mon("Rampardos", "Mold Breaker", stats={**_TANK, "atk": 300})
    lo_atk = _mon("Rampardos", "Mold Breaker", stats={**_TANK, "atk": 40})
    assert _dmax(user, hi_atk, "Foul Play") > _dmax(user, lo_atk, "Foul Play") > 0.0


def test_psyshock_hits_physical_def():
    user = _mon("Hatterene", "Magic Bounce", moves=["Psyshock"],
                stats={"atk": 90, "spa": 136, "def": 95, "spd": 103, "hp": 170, "spe": 29})
    # target physically frail / specially bulky → Psyshock (hits Def) does MORE than vs the inverse.
    frail_def = _mon("Blissey", "Natural Cure", stats={"atk": 10, "spa": 75, "def": 50, "spd": 300, "hp": 700, "spe": 55})
    bulky_def = _mon("Blissey", "Natural Cure", stats={"atk": 10, "spa": 75, "def": 300, "spd": 50, "hp": 700, "spe": 55})
    assert _dmax(user, frail_def, "Psyshock") > _dmax(user, bulky_def, "Psyshock") > 0.0


# ── _situational_damage_mult hits_def screen side ─────────────────────────────
def test_situational_hits_def_selects_physical_screen():
    reflect = {"grounded": True, "screen_phys": True, "screen_spec": False}
    lightscr = {"grounded": True, "screen_phys": False, "screen_spec": True}
    # a SPECIAL move that hits Def (Psyshock): Reflect reduces it, Light Screen does not.
    assert _situational_damage_mult("PSYCHIC", False, None, None, reflect, False, False, False, hits_def=True) == 0.667
    assert _situational_damage_mult("PSYCHIC", False, None, None, lightscr, False, False, False, hits_def=True) == 1.0
    # legacy (hits_def=None): a special move is reduced by Light Screen, not Reflect.
    assert _situational_damage_mult("PSYCHIC", False, None, None, lightscr, False, False, False) == 0.667
    assert _situational_damage_mult("PSYCHIC", False, None, None, reflect, False, False, False) == 1.0


def test_override_and_variable_sets_are_disjoint():
    variable_ids = {"lowkick", "grassknot", "heavyslam", "heatcrash", "gyroball", "electroball",
                    "eruption", "waterspout", "dragonenergy", "storedpower", "powertrip", "punishment",
                    "lastrespects", "ragefist"}
    override_ids = {"bodypress", "foulplay"} | set(_DEF_HIT_MOVES)
    assert variable_ids.isdisjoint(override_ids)
