"""
test_team_sheet.py
==================
Tests for the Showdown team-paste parser (team_sheet.py) used by the Type A
"Import team" flow — both the UI button and the headless bulk exporter.

Run:  python -m pytest vod_parser/test_team_sheet.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
from v_dance.parser.vod_parser.team_sheet import (
    parse_showdown_team,
    team_to_known_side,
    detect_our_side,
    base_species,
    _parse_first_line,
)

# The real team paste the user attached.
KRONOMONO1 = """\
Sneasler @ White Herb
Ability: Unburden
Level: 50
EVs: 2 HP / 32 Atk / 32 Spe
Jolly Nature
- Fake Out
- Dire Claw
- Protect
- Close Combat

Charizard-Mega-Y @ Charizardite Y
Ability: Solar Power
Level: 50
EVs: 15 HP / 18 Def / 1 SpA / 32 Spe
Modest Nature
- Heat Wave
- Weather Ball
- Protect
- Solar Beam

Whimsicott @ Occa Berry
Ability: Prankster
Level: 50
EVs: 2 HP / 32 SpA / 32 Spe
Timid Nature
- Encore
- Tailwind
- Moonblast
- Helping Hand

Basculegion (M) @ Mystic Water
Ability: Adaptability
Level: 50
EVs: 2 HP / 32 Atk / 32 Spe
Jolly Nature
- Flip Turn
- Wave Crash
- Protect
- Last Respects

Venusaur @ Focus Sash
Ability: Chlorophyll
Level: 50
EVs: 2 HP / 32 SpA / 32 Spe
Timid Nature
- Sludge Bomb
- Sleep Powder
- Earth Power
- Protect

Talonflame
Ability: Flame Body
Level: 50
EVs: 2 HP / 32 Atk / 32 Spe
Jolly Nature
- Acrobatics
- Tailwind
- Flare Blitz
- Protect
"""


class TestParseTeam:
    @pytest.fixture(scope="class")
    def mons(self):
        return parse_showdown_team(KRONOMONO1)

    def test_six_mons(self, mons):
        assert len(mons) == 6
        assert [m["species"] for m in mons] == [
            "Sneasler", "Charizard-Mega-Y", "Whimsicott",
            "Basculegion", "Venusaur", "Talonflame",
        ]

    def test_items_abilities_natures(self, mons):
        by = {m["species"]: m for m in mons}
        assert by["Sneasler"]["item"] == "White Herb"
        assert by["Sneasler"]["ability"] == "Unburden"
        assert by["Sneasler"]["nature"] == "Jolly"
        assert by["Charizard-Mega-Y"]["item"] == "Charizardite Y"
        assert by["Charizard-Mega-Y"]["ability"] == "Solar Power"
        assert by["Charizard-Mega-Y"]["nature"] == "Modest"
        # Item-less mon parses with item=None.
        assert by["Talonflame"]["item"] is None
        assert by["Talonflame"]["ability"] == "Flame Body"

    def test_evs(self, mons):
        by = {m["species"]: m for m in mons}
        assert by["Sneasler"]["evs"] == {"hp": 2, "atk": 32, "spe": 32}
        # The 4-stat Charizard spread, including the SpA: 1 quirk.
        assert by["Charizard-Mega-Y"]["evs"] == {"hp": 15, "def": 18, "spa": 1, "spe": 32}

    def test_moves(self, mons):
        by = {m["species"]: m for m in mons}
        assert by["Sneasler"]["moves"] == ["Fake Out", "Dire Claw", "Protect", "Close Combat"]
        assert by["Whimsicott"]["moves"] == ["Encore", "Tailwind", "Moonblast", "Helping Hand"]

    def test_gender_parsed_not_treated_as_species(self, mons):
        by = {m["species"]: m for m in mons}
        # "Basculegion (M)" — (M) is gender, species stays Basculegion.
        assert "Basculegion" in by
        assert by["Basculegion"]["gender"] == "M"
        assert by["Basculegion"]["nickname"] is None

    def test_level(self, mons):
        assert all(m["level"] == 50 for m in mons)

    def test_crlf_paste_splits_into_six(self):
        # Windows line endings must not collapse the team into one mon — this
        # was the real-file bug (browser File.text() keeps \r\n).
        crlf = KRONOMONO1.replace("\n", "\r\n")
        mons = parse_showdown_team(crlf)
        assert len(mons) == 6
        assert mons[0]["species"] == "Sneasler"
        assert mons[0]["ability"] == "Unburden"   # not bled-over from a later mon
        assert mons[0]["moves"] == ["Fake Out", "Dire Claw", "Protect", "Close Combat"]

    def test_trailing_whitespace_lines(self):
        # The real export has two trailing spaces on every line.
        padded = "\n".join(ln + "  " for ln in KRONOMONO1.split("\n"))
        mons = parse_showdown_team(padded)
        assert len(mons) == 6
        assert mons[1]["species"] == "Charizard-Mega-Y"
        assert mons[1]["item"] == "Charizardite Y"


class TestFirstLineEdgeCases:
    def test_plain(self):
        assert _parse_first_line("Talonflame") == ("Talonflame", None, None, None)

    def test_species_item(self):
        assert _parse_first_line("Sneasler @ White Herb") == \
            ("Sneasler", None, "White Herb", None)

    def test_gender(self):
        assert _parse_first_line("Basculegion (M) @ Mystic Water") == \
            ("Basculegion", None, "Mystic Water", "M")

    def test_nickname(self):
        sp, nick, item, g = _parse_first_line("Birb (Talonflame) @ Sharp Beak")
        assert (sp, nick, item, g) == ("Talonflame", "Birb", "Sharp Beak", None)

    def test_nickname_and_gender(self):
        sp, nick, item, g = _parse_first_line("Spicy (Charizard) (F) @ Charizardite Y")
        assert (sp, nick, item, g) == ("Charizard", "Spicy", "Charizardite Y", "F")


class TestKnownSideConversion:
    def test_keys_are_base_species(self):
        mons = parse_showdown_team(KRONOMONO1)
        side = team_to_known_side(mons)
        # Charizard-Mega-Y collapses to its roster (base) name.
        assert "Charizard" in side
        assert "Charizard-Mega-Y" not in side
        assert set(side) == {
            "Sneasler", "Charizard", "Whimsicott",
            "Basculegion", "Venusaur", "Talonflame",
        }

    def test_inject_shape_complete(self):
        side = team_to_known_side(parse_showdown_team(KRONOMONO1))
        sn = side["Sneasler"]
        # nature + full 6-stat ev_spread → promotable to EXACT stats downstream.
        assert sn["nature"] == "Jolly"
        assert sn["ev_spread"] == {"hp": 2, "atk": 32, "def": 0, "spa": 0, "spd": 0, "spe": 32}
        assert sn["item"] == "White Herb"
        assert sn["ability"] == "Unburden"
        assert sn["moves"] == ["Fake Out", "Dire Claw", "Protect", "Close Combat"]

    def test_charizard_ability_is_base_forme(self):
        # The injected ability must be the BASE ability (Solar Power); the fixed
        # mega ability (Drought) is resolved downstream from the pokedex.
        side = team_to_known_side(parse_showdown_team(KRONOMONO1))
        assert side["Charizard"]["ability"] == "Solar Power"

    def test_base_species_resolution(self):
        assert base_species("Charizard-Mega-Y") == "Charizard"
        assert base_species("Venusaur-Mega") == "Venusaur"
        # Non-mega formes keep their own roster name.
        assert base_species("Rotom-Wash") == "Rotom-Wash"
        assert base_species("Sneasler") == "Sneasler"


class TestDetectSide:
    def test_full_match_wins(self):
        team = {"Sneasler", "Charizard", "Whimsicott",
                "Basculegion", "Venusaur", "Talonflame"}
        rosters = {
            "p1": ["Miraidon", "Flutter Mane", "Incineroar", "Rillaboom"],
            "p2": ["Sneasler", "Charizard", "Whimsicott",
                   "Basculegion", "Venusaur", "Talonflame"],
        }
        assert detect_our_side(rosters, team) == "p2"

    def test_no_overlap_returns_none(self):
        team = {"Sneasler", "Charizard"}
        rosters = {"p1": ["Miraidon"], "p2": ["Koraidon"]}
        assert detect_our_side(rosters, team) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
