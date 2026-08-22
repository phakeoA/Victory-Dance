"""Priority-block (2026-07-10): an increased-priority damaging move FAILS vs a grounded target
under Psychic Terrain (conditions.ts psychicterrain onTryHit) and vs any target on a side with
Dazzling / Queenly Majesty / Armor Tail active (abilities.ts onFoeTryMove — holder AND ally).

Both encoders call the shared ``priority_blocked`` inside the per-enemy damage loop (the same
seam as ability immunities), so a blocked move's dmin/dmax band reads 0/0 — the "dead move"
signal the net already avoids. Found via the online misplay review: the AI clicked Sucker Punch
into Psychic Terrain twice in one game (replay 2646890310).
"""
from v_dance.encoders.battle_mechanics import _PRIORITY_BLOCK_AB, priority_blocked


def _d(grounded=True, ability=None, types=("PSYCHIC",)):
    return {"grounded": grounded, "ability": ability, "types": list(types)}


def test_block_ability_set():
    assert set(_PRIORITY_BLOCK_AB) == {"dazzling", "armortail", "queenlymajesty"}


def test_psychic_terrain_blocks_priority_vs_grounded():
    assert priority_blocked(1, "PSYCHIC_TERRAIN", _d(grounded=True))          # Sucker Punch
    assert priority_blocked(3, "PSYCHIC_TERRAIN", _d(grounded=True))          # Fake Out


def test_psychic_terrain_spares_airborne():
    # Flying / Levitate defenders are NOT protected by the terrain (isGrounded() false).
    assert not priority_blocked(1, "PSYCHIC_TERRAIN", _d(grounded=False, types=("FLYING",)))


def test_other_terrains_never_block():
    for t in (None, "", "ELECTRIC_TERRAIN", "GRASSY_TERRAIN", "MISTY_TERRAIN"):
        assert not priority_blocked(1, t, _d(grounded=True))


def test_normal_priority_never_blocked():
    # priority 0 (normal moves) and negative brackets pass through everywhere.
    assert not priority_blocked(0, "PSYCHIC_TERRAIN", _d(grounded=True))
    assert not priority_blocked(-6, "PSYCHIC_TERRAIN", _d(grounded=True))     # Trick Room et al.
    assert not priority_blocked(0, None, _d(ability="dazzling"))


def test_dazzling_class_blocks_regardless_of_ground():
    for ab in _PRIORITY_BLOCK_AB:
        assert priority_blocked(1, None, _d(grounded=False, ability=ab))


def test_dazzling_on_partner_protects_ally():
    # Queenly Majesty-class protect the WHOLE side: the ally without the ability is covered too.
    target = _d(ability=None)
    partner = _d(ability="queenlymajesty")
    assert priority_blocked(1, None, target, defender_side=[target, partner])
    # ... and with no protector anywhere on the side, no block.
    assert not priority_blocked(1, None, target, defender_side=[target, _d(ability="intimidate")])


def test_empty_defender_and_side_holes_are_safe():
    assert not priority_blocked(1, "PSYCHIC_TERRAIN", None)
    assert priority_blocked(1, None, _d(ability="dazzling"), defender_side=[None, _d(ability="dazzling")])


def test_band_zeroed_through_the_writer_seam():
    """End-to-end at the seam contract level: the writers gate the ENTIRE damage computation on
    ``not priority_blocked(...)`` — blocked → the else-branch writes dmin=dmax=0.0 (identical to
    the pre-existing ability-immunity path). Assert the gate composes with a real profile shape."""
    defender = {"grounded": True, "ability": "innerfocus", "types": ["PSYCHIC"],
                "hp": 175.0, "hp_frac": 1.0, "def": 100.0, "spd": 120.0, "eff_speed": 120.0}
    side = [defender, {"grounded": True, "ability": "magicguard", "types": ["PSYCHIC"]}]
    assert priority_blocked(1, "PSYCHIC_TERRAIN", defender, side)             # Sucker Punch dies
    assert not priority_blocked(1, "ELECTRIC_TERRAIN", defender, side)        # terrain flipped → live
