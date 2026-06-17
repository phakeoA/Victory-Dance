"""Tests for vod_parser/pokedex.py — species normalisation, ability lookup,
and mega-forme resolution (the foundation of the Bug 8 fix)."""

from __future__ import annotations

import pytest

from v_dance.parser.vod_parser.pokedex import Pokedex, get_pokedex, is_mega_species_name, norm_species


# ── norm_species ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("Floette-Mega", "floettemega"),
    ("Floette-Eternal", "floetteeternal"),
    ("Rotom-Wash", "rotomwash"),
    ("Mr. Mime", "mrmime"),
    ("Farfetch'd", "farfetchd"),
    ("Charizard-Mega-X", "charizardmegax"),
    ("", ""),
    (None, ""),
])
def test_norm_species(raw, expected):
    assert norm_species(raw) == expected


# ── is_mega_species_name (name heuristic, no dex) ─────────────────────────

@pytest.mark.parametrize("name,expected", [
    ("Venusaur-Mega", True),
    ("Charizard-Mega-X", True),
    ("Charizard-Mega-Y", True),
    ("Meganium", False),          # contains "mega" as substring — must NOT match
    ("Yanmega", False),
    ("Palafin-Hero", False),
    ("Floette-Eternal", False),
    (None, False),
    ("", False),
])
def test_is_mega_species_name(name, expected):
    assert is_mega_species_name(name) is expected


# ── abilities_for ─────────────────────────────────────────────────────────

def test_abilities_for_base_forme(dex):
    assert dex.abilities_for("Meganium") == ["Overgrow", "Leaf Guard"]
    assert dex.abilities_for("Aerodactyl") == ["Rock Head", "Pressure", "Unnerve"]


def test_abilities_for_mega_forme_is_single(dex):
    assert dex.abilities_for("Floette-Mega") == ["Fairy Aura"]
    assert dex.abilities_for("Meganium-Mega") == ["Mega Sol"]


def test_abilities_for_unknown_species_is_empty(dex):
    assert dex.abilities_for("Missingno") == []
    assert dex.abilities_for(None) == []


# ── is_mega_forme (dex-confirmed) ─────────────────────────────────────────

def test_is_mega_forme(dex):
    assert dex.is_mega_forme("Floette-Mega") is True
    assert dex.is_mega_forme("Meganium-Mega") is True
    assert dex.is_mega_forme("Meganium") is False
    assert dex.is_mega_forme("Palafin-Hero") is False
    assert dex.is_mega_forme("Floette-Eternal") is False


def test_is_mega_forme_falls_back_to_name_for_unknown_species(dex):
    # Not in the dex → fall back to suffix heuristic
    assert dex.is_mega_forme("Fakemon-Mega") is True
    assert dex.is_mega_forme("Fakemon") is False


# ── mega_ability_for ──────────────────────────────────────────────────────

def test_mega_ability_for_known_megas(dex):
    assert dex.mega_ability_for("Floette-Mega") == "Fairy Aura"
    assert dex.mega_ability_for("Meganium-Mega") == "Mega Sol"
    assert dex.mega_ability_for("Aerodactyl-Mega") == "Tough Claws"


def test_mega_ability_for_non_mega_is_none(dex):
    """A base forme never has a 'mega ability'."""
    assert dex.mega_ability_for("Meganium") is None
    assert dex.mega_ability_for("Incineroar") is None


def test_mega_ability_refuses_to_guess_when_multiple_listed():
    """Mega formes have exactly one ability — a malformed dex entry listing
    several must yield None, never an arbitrary pick."""
    dex = Pokedex(data={
        "brokenmega": {
            "name": "broken-mega", "forme": "Mega", "baseSpecies": "Broken",
            "abilities": {"0": "A", "1": "B"},
        },
    })
    assert dex.mega_ability_for("Broken-Mega") is None


# ── mega_formes_for ───────────────────────────────────────────────────────

def test_mega_formes_for_base_species(dex):
    assert dex.mega_formes_for("Meganium") == [
        {"forme": "Meganium-Mega", "ability": "Mega Sol"},
    ]


def test_mega_formes_for_resolves_through_non_base_forme(dex):
    """Floette-Eternal (forme of Floette) must reach Floette-Mega via its
    base species — this is exactly the path the example VOD exercises."""
    formes = dex.mega_formes_for("Floette-Eternal")
    assert {"forme": "Floette-Mega", "ability": "Fairy Aura"} in formes


def test_mega_formes_for_twin_megas(dex):
    formes = dex.mega_formes_for("Charizard")
    names = {f["forme"] for f in formes}
    assert names == {"Charizard-Mega-X", "Charizard-Mega-Y"}
    # Each twin still has exactly one (distinct) ability
    abilities = {f["forme"]: f["ability"] for f in formes}
    assert all(abilities.values())
    assert abilities["Charizard-Mega-X"] != abilities["Charizard-Mega-Y"]


def test_mega_formes_for_species_without_mega(dex):
    assert dex.mega_formes_for("Incineroar") == []
    assert dex.mega_formes_for("Rotom-Wash") == []


def test_mega_formes_for_unknown_species(dex):
    assert dex.mega_formes_for("Missingno") == []


# ── singleton ─────────────────────────────────────────────────────────────

def test_get_pokedex_singleton_loads_and_caches():
    a = get_pokedex()
    b = get_pokedex()
    assert a is not None, "data/pokedex.json should be found from the package path"
    assert a is b
