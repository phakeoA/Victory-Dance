"""Tests for vod_parser/battle_models.py — PokemonSlot serialisation,
including the Bug 8 split-ability fields."""

from __future__ import annotations

from v_dance.parser.vod_parser.battle_models import FieldConditions, PokemonSlot, SideConditions


def test_to_dict_contains_split_ability_fields():
    mon = PokemonSlot(species="Meganium", nickname="Meg", player="p2", slot="b",
                      base_species="Meganium")
    d = mon.to_dict()
    for field in ("known_ability", "pre_mega_ability", "mega_ability",
                  "base_species", "species", "is_mega"):
        assert field in d
    assert d["known_ability"] is None
    assert d["pre_mega_ability"] is None
    assert d["mega_ability"] is None
    assert d["base_species"] == "Meganium"


def test_to_dict_reflects_mega_ability_state():
    mon = PokemonSlot(species="Meganium-Mega", nickname="Meg", player="p2", slot="b",
                      base_species="Meganium",
                      is_mega=True,
                      pre_mega_ability="Overgrow",
                      mega_ability="Mega Sol",
                      known_ability="Mega Sol")
    d = mon.to_dict()
    assert d["species"] == "Meganium-Mega"
    assert d["base_species"] == "Meganium"        # frozen pre-mega name
    assert d["known_ability"] == "Mega Sol"        # currently active
    assert d["pre_mega_ability"] == "Overgrow"     # preserved, not lost
    assert d["mega_ability"] == "Mega Sol"
    assert d["is_mega"] is True


def test_to_dict_base_species_defaults_to_species():
    mon = PokemonSlot(species="Incineroar", nickname="Inc", player="p1", slot="a")
    assert mon.to_dict()["base_species"] == "Incineroar"


# ── Gap #5: hp_pct is a true percentage regardless of the log's HP scale ───────
def test_to_dict_hp_pct_percent_scale():
    """A %-scale mon (X/100) stores its numerator unchanged as the percentage."""
    mon = PokemonSlot(species="Incineroar", nickname="Inc", player="p1", slot="a",
                      hp_current=74.0, hp_max=100.0)
    assert mon.to_dict()["hp_pct"] == 74.0


def test_to_dict_hp_pct_real_scale_is_normalised():
    """A real-HP mon (175/200, owner-recorded replay) must serialise hp_pct as a
    true PERCENTAGE (87.5), not the bare numerator 175 (gap #5)."""
    mon = PokemonSlot(species="Dondozo", nickname="Don", player="p1", slot="a",
                      hp_current=175.0, hp_max=200.0)
    assert mon.to_dict()["hp_pct"] == 87.5


def test_to_dict_hp_pct_none_when_unknown():
    mon = PokemonSlot(species="Ditto", nickname="D", player="p2", slot="a",
                      hp_current=None, hp_max=200.0)
    assert mon.to_dict()["hp_pct"] is None


def test_to_dict_mega_item_placeholder():
    mon = PokemonSlot(species="X-Mega", nickname="X", player="p1", slot="a",
                      is_mega=True, known_item=None)
    assert mon.to_dict()["known_item"] == "mega stone"


def test_side_and_field_conditions_to_dict():
    sc = SideConditions(tailwind=3, screens={"reflect": 4}, spikes=2, stealth_rock=True, mist=True)
    assert sc.to_dict() == {"tailwind_turns_remaining": 3, "screens": {"reflect": 4},
                            "stealth_rock": True, "spikes": 2, "toxic_spikes": 0, "sticky_web": False,
                            "safeguard": False, "mist": True, "lucky_chant": False}

    fc = FieldConditions(weather="SunnyDay", terrain="grassy", trick_room=2,
                         weather_turns=6, terrain_turns=3, gravity=4, magic_room=5, wonder_room=7)
    assert fc.to_dict() == {
        "weather": "SunnyDay",
        "terrain": "grassy",
        "trick_room_turns_remaining": 2,
        "weather_turns_active": 6,    # elapsed > 5 ⇒ a weather-rock 8-turn instance
        "terrain_turns_active": 3,
        "gravity_turns_remaining": 4,    # v11 C.2e
        "magic_room_turns_remaining": 5,    # v11 P5
        "wonder_room_turns_remaining": 7,
    }
