"""The LIVE futility mask reads REAL poke-env enums (2026-10-02, found by the mega-fix review).

The pinned poke-env renders ``str(PokemonType.DARK)`` as ``'DARK (pokemon type) object'``, so the serve-side
extractor's ``str(x).split('.')[-1]`` never matched a type, field, weather, side condition or status: every
type/field rule of the shared futility core silently skipped live (since at least 2026-08-05) while TRAINING
masked the same buckets. tests/test_futility_mask.py used plain-string doubles, so it never saw this. These cases
drive a real ``DoubleBattle`` through the protocol and must agree with the offline rule core.
"""
from __future__ import annotations

import logging

import pytest

pytest.importorskip("poke_env")

from poke_env.battle import DoubleBattle  # noqa: E402
from poke_env.battle.move import Move  # noqa: E402

import v_dance.play.vgc_base as VB  # noqa: E402


def _battle(*msgs):
    b = DoubleBattle("battle-gen9championsvgc2026regmc-1", "me", logging.getLogger("t"), gen=9)
    b._player_role = "p1"
    for m in msgs:
        b.parse_message(m)
    return b


def test_enum_name_reads_poke_env_enums_and_plain_strings():
    from poke_env.battle import PokemonType, Field, Weather, SideCondition, Status
    assert str(PokemonType.DARK) != "DARK"                         # the trap: str() is not the name
    assert [VB._enum_name(x) for x in (PokemonType.DARK, Field.PSYCHIC_TERRAIN, Weather.RAINDANCE,
                                        SideCondition.TAILWIND, Status.PAR)] == [
        "DARK", "PSYCHIC_TERRAIN", "RAINDANCE", "TAILWIND", "PAR"]
    assert VB._enum_name("dark") == "dark"


def test_redundant_field_moves_are_futile_live():
    b = _battle(["", "switch", "p1a: Tornadus", "Tornadus, L50", "100/100"],
                ["", "switch", "p1b: Pelipper", "Pelipper, L50", "100/100"],
                ["", "switch", "p2a: Incineroar", "Incineroar, L50", "100/100"],
                ["", "switch", "p2b: Amoonguss", "Amoonguss, L50", "100/100"],
                ["", "-weather", "RainDance", "[from] ability: Drizzle", "[of] p1b: Pelipper"],
                ["", "-sidestart", "p1: me", "move: Tailwind"],
                ["", "-fieldstart", "move: Gravity"])
    mon = b.get_pokemon("p1a: Tornadus")
    for mv in ("tailwind", "raindance", "gravity"):
        assert VB._futile_buckets_serve(b, mon, Move(mv, 9)), mv


def test_prankster_into_dark_and_powder_into_grass_are_futile_live():
    from v_dance.encoders.action_codec import futile_target_buckets
    b = _battle(["", "switch", "p1a: Whimsicott", "Whimsicott, L50", "100/100"],
                ["", "switch", "p1b: Amoonguss", "Amoonguss, L50", "100/100"],
                ["", "switch", "p2a: Kingambit", "Kingambit, L50", "100/100"],
                ["", "switch", "p2b: Rillaboom", "Rillaboom, L50", "100/100"],
                ["", "-ability", "p1a: Whimsicott", "Prankster"])
    w, a = b.get_pokemon("p1a: Whimsicott"), b.get_pokemon("p1b: Amoonguss")
    live_encore = VB._futile_buckets_serve(b, w, Move("encore", 9))
    live_spore = VB._futile_buckets_serve(b, a, Move("spore", 9))
    foes = {0: {"types": ("dark", "steel"), "status": None, "encored": False, "confused": False, "grounded": True},
            1: {"types": ("grass",), "status": None, "encored": False, "confused": False, "grounded": True}}
    assert 0 in live_encore and live_encore == futile_target_buckets("encore", user_ability="prankster", foes=foes)
    assert 1 in live_spore and 1 in futile_target_buckets("spore", user_ability="effectspore", foes=foes)


def test_protect_stays_legal_live_under_psychic_terrain():
    # 2026-10-02 (the g50 Baltimore bisect): once the enum fix made the live terrain rule fire, every Protect /
    # Follow Me under Psychic Terrain was masked (canonical bucket 0 read as 'foe 0'). Self moves are never futile
    # here; a priority move aimed at a grounded foe still is.
    b = _battle(["", "switch", "p1a: Indeedee", "Indeedee-F, L50, F", "100/100"],
                ["", "switch", "p1b: Excadrill", "Excadrill, L50", "100/100"],
                ["", "switch", "p2a: Rillaboom", "Rillaboom, L50", "100/100"],
                ["", "switch", "p2b: Incineroar", "Incineroar, L50", "100/100"],
                ["", "-fieldstart", "move: Psychic Terrain", "[from] ability: Psychic Surge", "[of] p1a: Indeedee"])
    ind = b.get_pokemon("p1a: Indeedee")
    for mv in ("protect", "followme", "detect", "wideguard"):
        assert VB._futile_buckets_serve(b, ind, Move(mv, 9)) == set(), mv
    assert 0 in VB._futile_buckets_serve(b, ind, Move("fakeout", 9))
