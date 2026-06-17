"""State-rep #B: Defiant / Competitive split into a dedicated ``statdrop_boost``
ability-effect category (separate from the buff-on-being-HIT ``reactive_boost``).

Before: defiant + competitive collapsed into ``reactive_boost`` (idx 13), so the
model could not tell an Intimidate-punishing ability apart from Justified/Berserk.
After: a new ``statdrop_boost`` category (NUM_ABILITY_EFFECTS 16→17, layout v4).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("poke_env")

from v_dance.encoders.state_encoder import (  # noqa: E402
    ability_effect_indices, ABILITY_EFFECT_NAMES, NUM_ABILITY_EFFECTS,
    get_state_dim, get_state_layout_version,
)


def _cats(ability_id: str) -> set:
    return {ABILITY_EFFECT_NAMES[i] for i in ability_effect_indices(ability_id)}


# ── the split itself ───────────────────────────────────────────────────────────
def test_defiant_and_competitive_map_to_statdrop_not_reactive():
    assert _cats("defiant") == {"statdrop_boost"}
    assert _cats("competitive") == {"statdrop_boost"}


def test_on_hit_reactives_stay_reactive_not_statdrop():
    # the buff-on-being-HIT abilities keep reactive_boost and must NOT leak into
    # the new stat-drop category.
    for ab in ("justified", "angerpoint", "berserk", "weakarmor", "stamina"):
        cats = _cats(ab)
        assert "reactive_boost" in cats, ab
        assert "statdrop_boost" not in cats, ab


def test_statdrop_and_reactive_are_distinct_categories():
    i_react = ABILITY_EFFECT_NAMES.index("reactive_boost")
    i_drop = ABILITY_EFFECT_NAMES.index("statdrop_boost")
    assert i_react != i_drop


def test_layout_v4_dims():
    assert NUM_ABILITY_EFFECTS == 17
    assert get_state_dim() == 1866
    assert get_state_layout_version() == 4


# ── end-to-end: the encoder writes the right ability bit ───────────────────────
# Ability block base, mirroring test_state_encoder.py (_ITEM0 + ITEM_FEATURES).
from v_dance.encoders.state_encoder import NUM_MOVES, MOVE_FEATURES, ITEM_FEATURES  # noqa: E402

_ITEM0 = 1 + 20 + 20 + 6 + 6 + 1 + 1 + 1 + 7 + 7 + NUM_MOVES * MOVE_FEATURES
_ABIL0 = _ITEM0 + ITEM_FEATURES


def _mon(species, ability, *, moves=("Fake Out",)):
    return {
        "species": species, "base_species": species,
        "hp_pct": 100.0, "seen": True, "is_fainted": False,
        "known_moves": list(moves), "revealed_moves": [],
        "boosts": {}, "status": None, "known_ability": ability,
    }


def _abil_bits_for(ability):
    """Encode a single own-active mon with ``ability`` and return the set of ON
    ability-effect category names in its slot."""
    from v_dance.encoders.state_encoder import StateEncoder
    snap = {"our_active": {"our_a": _mon("Kingambit", ability), "our_b": None},
            "opp_active": {}, "our_bench": [], "opp_bench": [],
            "field": {}, "side_conditions": {}}
    vec = StateEncoder().encode_snapshot(snap, turn=1)
    return {ABILITY_EFFECT_NAMES[k] for k in range(NUM_ABILITY_EFFECTS)
            if vec[_ABIL0 + k] == 1.0}


def test_encoder_writes_statdrop_bit_for_defiant():
    on = _abil_bits_for("Defiant")
    assert "statdrop_boost" in on
    assert "reactive_boost" not in on


def test_encoder_writes_reactive_bit_for_justified():
    on = _abil_bits_for("Justified")
    assert "reactive_boost" in on
    assert "statdrop_boost" not in on
