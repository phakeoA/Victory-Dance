"""v11 G7: Grassy Terrain halves Earthquake/Bulldoze/Magnitude vs a GROUNDED defender.

Source: pokemon-showdown/data/moves.ts grassyterrain onBasePower —
``['earthquake','bulldoze','magnitude']`` chainModify(0.5) when ``defender.isGrounded()``.
The ``grassy_eq`` flag is computed by BOTH encoders (move_id in _GRASSY_WEAKENED) and passed to the
shared ``_situational_damage_mult`` helper, so the offline/live band is byte-parity by construction.
"""
from v_dance.encoders.state_encoder import _situational_damage_mult, _GRASSY_WEAKENED


def _grounded(types=("NORMAL",)):
    return {"grounded": True, "types": list(types)}


def test_grassy_weakened_set():
    assert _GRASSY_WEAKENED == frozenset({"earthquake", "bulldoze", "magnitude"})


def test_grassy_halves_eq_vs_grounded():
    # GROUND move, Grassy Terrain, grounded defender, no other mods → ×0.5
    m = _situational_damage_mult("GROUND", True, None, "GRASSY_TERRAIN", _grounded(),
                                 False, False, False, grassy_eq=True)
    assert abs(m - 0.5) < 1e-9


def test_grassy_no_halve_when_not_grassy():
    # Same move flagged grassy_eq but the terrain is Electric → no halving.
    m = _situational_damage_mult("GROUND", True, None, "ELECTRIC_TERRAIN", _grounded(),
                                 False, False, False, grassy_eq=True)
    assert abs(m - 1.0) < 1e-9


def test_grassy_no_halve_when_airborne():
    # Grassy Terrain but the defender is NOT grounded → no halving (matches isGrounded()).
    d = {"grounded": False, "types": ["FLYING"]}
    m = _situational_damage_mult("GROUND", True, None, "GRASSY_TERRAIN", d,
                                 False, False, False, grassy_eq=True)
    assert abs(m - 1.0) < 1e-9


def test_grassy_no_halve_for_non_eq_move():
    # A GRASS move under Grassy Terrain gets the +1.3 boost, NOT the EQ ×0.5 (grassy_eq=False).
    m = _situational_damage_mult("GRASS", False, None, "GRASSY_TERRAIN", _grounded(),
                                 False, False, False, grassy_eq=False)
    assert abs(m - 1.3) < 1e-9


def test_grassy_default_arg_is_false():
    # Back-compat: omitting grassy_eq must behave exactly as before (no halving).
    m = _situational_damage_mult("GROUND", True, None, "GRASSY_TERRAIN", _grounded(),
                                 False, False, False)
    assert abs(m - 1.0) < 1e-9
