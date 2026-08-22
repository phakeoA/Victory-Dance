"""v11 B2 (attacker-side accuracy fold) + N3 (defensive weather) + the gap-scan fixes G4/G5/G6.

N3: Sandstorm +50% SpD for ROCK defenders (incoming special ×2/3) · Snowscape +50% Def for ICE defenders
    (incoming physical ×2/3) — in the shared _situational_damage_mult. ⚠ The band weather token is the
    PRE-alias "SNOWSCAPE"/"SANDSTORM" on BOTH paths (never "SNOW" — that alias is only at the global one-hot).
G4: Misty Terrain halves Dragon vs a grounded defender. G5: Sand Force ×1.3 Rock/Ground/Steel in sand.
G6: Solar Power ×1.5 special in sun (both in _ability_damage_mult via a threaded `weather` kwarg).
B2: attacker accuracy abilities (Compound Eyes/Hustle-physical/No Guard) + accuracy boost STAGE + Wide Lens,
    folded into the single accuracy channel (OHKO + always-hit skipped). Defender-side evasion DEFERRED (per-enemy
    layout). Design + gap-scan: scratch / wf_b4e5c4e3.
"""
from __future__ import annotations

from v_dance.encoders.state_encoder import (
    _situational_damage_mult, _ability_damage_mult, _accuracy_modifiers, _move_always_hit, _move_is_ohko,
    _DMG_BOOST_AB,
)

_TT = 2.0 / 3.0


def _D(types, grounded=True):
    return {"types": list(types), "grounded": grounded, "screen_phys": False, "screen_spec": False}


# ════════════════════════════ N3 defensive weather ════════════════════════════
def test_n3_sand_boosts_spd_for_rock():
    rock = _D(["ROCK", "GROUND"])
    assert _situational_damage_mult("WATER", False, "SANDSTORM", None, rock, False, False, False) == _TT  # special
    assert _situational_damage_mult("WATER", True, "SANDSTORM", None, rock, False, False, False) == 1.0   # physical unaffected


def test_n3_snow_boosts_def_for_ice():
    ice = _D(["ICE"])
    assert _situational_damage_mult("FIGHTING", True, "SNOWSCAPE", None, ice, False, False, False) == _TT  # physical
    assert _situational_damage_mult("FIGHTING", False, "SNOWSCAPE", None, ice, False, False, False) == 1.0  # special unaffected


def test_n3_snow_token_is_snowscape_not_snow():
    # regression guard for G1: the band never sees "SNOW" (only the global one-hot aliases it) → must not fire.
    ice = _D(["ICE"])
    assert _situational_damage_mult("FIGHTING", True, "SNOW", None, ice, False, False, False) == 1.0
    assert _situational_damage_mult("FIGHTING", True, "HAIL", None, ice, False, False, False) == 1.0   # hail = no Def boost


def test_n3_type_gated():
    normal = _D(["NORMAL"])
    assert _situational_damage_mult("WATER", False, "SANDSTORM", None, normal, False, False, False) == 1.0
    assert _situational_damage_mult("FIGHTING", True, "SNOWSCAPE", None, normal, False, False, False) == 1.0


def test_n3_psyshock_uses_def_side():
    # a special move with hits_def=True (Psyshock) hits Def → Snow side, NOT Sand side.
    ice, rock = _D(["ICE"]), _D(["ROCK"])
    assert _situational_damage_mult("PSYCHIC", False, "SNOWSCAPE", None, ice, False, False, False, hits_def=True) == _TT
    assert _situational_damage_mult("PSYCHIC", False, "SANDSTORM", None, rock, False, False, False, hits_def=True) == 1.0


# ════════════════════════════ G4 Misty Terrain × Dragon ════════════════════════════
def test_g4_misty_halves_dragon_grounded():
    assert _situational_damage_mult("DRAGON", False, None, "MISTY_TERRAIN", _D(["STEEL"]), False, False, False) == 0.5
    assert _situational_damage_mult("DRAGON", False, None, "MISTY_TERRAIN", _D(["FLYING"], grounded=False),
                                    False, False, False) == 1.0   # ungrounded
    assert _situational_damage_mult("FIRE", False, None, "MISTY_TERRAIN", _D(["STEEL"]), False, False, False) == 1.0  # non-Dragon


# ════════════════════════════ G5 Sand Force / G6 Solar Power ════════════════════════════
def _am(ab, mt, phys, weather):
    return _ability_damage_mult("x", ab, None, eff_move_type=mt, is_stab=False, is_physical=phys,
                                type_mult=1.0, hp_frac=1.0, att_burned=False, att_statused=False, weather=weather)


def test_g5_sand_force():
    assert "sandforce" in _DMG_BOOST_AB
    assert abs(_am("sandforce", "GROUND", True, "SANDSTORM") - 5325 / 4096) < 1e-9   # ×1.3
    assert abs(_am("sandforce", "ROCK", True, "SANDSTORM") - 5325 / 4096) < 1e-9
    assert _am("sandforce", "WATER", True, "SANDSTORM") == 1.0      # non-boosted type
    assert _am("sandforce", "GROUND", True, None) == 1.0           # no sand


def test_g6_solar_power():
    assert "solarpower" in _DMG_BOOST_AB
    assert _am("solarpower", "FIRE", False, "SUNNYDAY") == 1.5      # special in sun
    assert _am("solarpower", "FIRE", True, "SUNNYDAY") == 1.0       # physical unaffected
    assert _am("solarpower", "FIRE", False, None) == 1.0           # no sun


# ════════════════════════════ B2 attacker accuracy fold ════════════════════════════
def _acc(base, ability_id=None, is_physical=False, is_ohko=False, acc_stage=0, wide_lens=False):
    return _accuracy_modifiers(base, ability_id=ability_id, is_physical=is_physical, is_ohko=is_ohko,
                               acc_stage=acc_stage, wide_lens=wide_lens)


def test_b2_compound_eyes():
    assert abs(_acc(0.75, ability_id="compoundeyes") - 0.75 * 5325 / 4096) < 1e-9
    assert _acc(1.0, ability_id="compoundeyes") == 1.0             # capped


def test_b2_hustle_physical_only():
    assert abs(_acc(1.0, ability_id="hustle", is_physical=True) - 3277 / 4096) < 1e-9
    assert _acc(1.0, ability_id="hustle", is_physical=False) == 1.0   # special/status unaffected


def test_b2_no_guard_and_wide_lens():
    assert _acc(0.5, ability_id="noguard") == 1.0
    assert abs(_acc(0.85, wide_lens=True) - min(0.85 * 4505 / 4096, 1.0)) < 1e-9


def test_b2_accuracy_stage():
    assert abs(_acc(0.6, acc_stage=1) - 0.6 * 4 / 3) < 1e-9        # +1 → ×4/3
    assert abs(_acc(0.9, acc_stage=-2) - 0.9 * 3 / 5) < 1e-9       # -2 → ×3/5
    assert _acc(0.5, acc_stage=99) == min(0.5 * 9 / 3, 1.0)        # clamp +6 → ×3


def test_b2_ohko_and_always_hit_untouched():
    assert _acc(0.30, ability_id="compoundeyes", is_ohko=True) == 0.30   # G3: OHKO untouched
    assert _move_always_hit("aerialace") is True and _move_always_hit("earthquake") is False
    assert _move_is_ohko("fissure") is True and _move_is_ohko("earthquake") is False
