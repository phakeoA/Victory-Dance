"""B1-mechanics step 4: parser tracks per-mon volatiles + ability suppression into the snapshot dict
(Substitute/Ingrain/Taunt/Encore/partial-trap/Gastro-Acid), cleared on switch-out/in."""
from __future__ import annotations

from conftest import HEADER, make_log
from v_dance.parser.vod_parser.battle_models import PokemonSlot
from v_dance.parser.vod_parser.replay_parser import ShowdownReplayParser


def _parse(*lines):
    p = ShowdownReplayParser(make_log(*HEADER, *lines), our_player="p1")
    p.parse()
    return p


def test_to_dict_volatile_flags():
    m = PokemonSlot(species="Gengar", nickname="Gengar", player="p1", slot="a")
    m.volatiles = {"substitute", "taunt"}
    v = m.to_dict()["volatiles"]
    assert v["has_substitute"] and v["move_restricted"]
    assert not v["rooted"] and not v["trapped"] and not v["locked_action"] and not v["ability_suppressed"]

    m.volatiles = {"ingrain", "partiallytrapped", "mustrecharge"}
    m.ability_suppressed = True
    v = m.to_dict()["volatiles"]
    assert v["rooted"] and v["trapped"] and v["locked_action"] and v["ability_suppressed"]
    assert not v["has_substitute"]


def test_start_end_volatiles_tracked():
    p = _parse(
        "|switch|p1a: Gengar|Gengar, L50, M|100/100",
        "|switch|p2a: Snorlax|Snorlax, L50, M|100/100",
        "|turn|1",
        "|-start|p1a: Gengar|Substitute",
        "|-start|p2a: Snorlax|move: Taunt",
        "|-start|p1a: Gengar|move: Gastro Acid",     # ability suppression
        "|turn|2",
        "|win|alice",
    )
    g = p.active_slots["p1a"]
    assert "substitute" in g.volatiles and "gastroacid" in g.volatiles and g.ability_suppressed is True
    assert g.to_dict()["volatiles"]["has_substitute"] and g.to_dict()["volatiles"]["ability_suppressed"]
    s = p.active_slots["p2a"]
    assert "taunt" in s.volatiles and s.to_dict()["volatiles"]["move_restricted"]


def test_end_removes_volatile():
    p = _parse(
        "|switch|p1a: Gengar|Gengar, L50, M|100/100",
        "|turn|1",
        "|-start|p1a: Gengar|Substitute",
        "|turn|2",
        "|-end|p1a: Gengar|Substitute",
        "|win|alice",
    )
    assert "substitute" not in p.active_slots["p1a"].volatiles


def test_volatiles_cleared_on_switch():
    p = _parse(
        "|switch|p1a: Gengar|Gengar, L50, M|100/100",
        "|turn|1",
        "|-start|p1a: Gengar|Substitute",
        "|-start|p1a: Gengar|move: Gastro Acid",
        "|turn|2",
        "|switch|p1a: Pikachu|Pikachu, L50, M|100/100",   # Gengar out
        "|turn|3",
        "|switch|p1a: Gengar|Gengar, L50, M|100/100",      # Gengar back in -> clean
        "|win|alice",
    )
    g = p.active_slots["p1a"]
    assert g.species == "Gengar"
    assert g.volatiles == set() and g.ability_suppressed is False
