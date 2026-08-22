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


def test_layout_v14_dims():
    assert NUM_ABILITY_EFFECTS == 17     # the v8 effect helper is unchanged (still used internally)
    assert get_state_dim() == 5057       # layout v16 (B2b: +2 per-move hit-chance channels)
    assert get_state_layout_version() == 19


# ── end-to-end: the encoder writes the right ability TAG bit (v9) ──────────────
# v9 ability block within a mon = [identity index] + [NUM_ABILITY_TAGS tags] + [known]; tags start
# right after the identity index (ABILITY_ID_REL is the absolute offset for slot-0 own_a).
from v_dance.encoders.state_encoder import ABILITY_ID_REL, NUM_ABILITY_TAGS  # noqa: E402
from v_dance.encoders.mechanic_tags import ABILITY_TAG_NAMES  # noqa: E402

_ABIL_TAGS0 = ABILITY_ID_REL + 1


def _mon(species, ability, *, moves=("Fake Out",)):
    return {
        "species": species, "base_species": species,
        "hp_pct": 100.0, "seen": True, "is_fainted": False,
        "known_moves": list(moves), "revealed_moves": [],
        "boosts": {}, "status": None, "known_ability": ability,
    }


def _abil_bits_for(ability):
    """Encode a single own-active mon with ``ability`` and return the set of ON ability-TAG names (v9)."""
    from v_dance.encoders.state_encoder import StateEncoder
    snap = {"our_active": {"our_a": _mon("Kingambit", ability), "our_b": None},
            "opp_active": {}, "our_bench": [], "opp_bench": [],
            "field": {}, "side_conditions": {}}
    vec = StateEncoder().encode_snapshot(snap, turn=1)
    return {ABILITY_TAG_NAMES[k] for k in range(NUM_ABILITY_TAGS)
            if vec[_ABIL_TAGS0 + k] == 1.0}


def test_encoder_writes_statdrop_bit_for_defiant():
    on = _abil_bits_for("Defiant")
    assert "statdrop_boost" in on
    assert "reactive_boost" not in on


def test_encoder_writes_reactive_bit_for_justified():
    on = _abil_bits_for("Justified")
    assert "reactive_boost" in on
    assert "statdrop_boost" not in on
