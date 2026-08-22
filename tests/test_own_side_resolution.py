"""Tests for live own-side resolution in state_encoder.

A side is the bot's OWN side iff BOTH the player's username matches the
configured one AND the configured team's species are all in that player's
teampreview roster (base formes).  No team configured ⇒ no own side (Type B).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


from v_dance.encoders.live_state_encoder import (  # noqa: E402
    resolve_own_role, own_role_from_log, team_species_from_paste,
    parse_log_players_and_rosters, LiveStateEncoder as StateEncoder,
)
from v_dance.parser.vod_parser.replay_parser import extract_log_from_html  # noqa: E402

_VODS = Path(__file__).resolve().parents[1] / "data" / "vods"
_DUAL_NAME = "Gen9ChampionsVGC2026RegMA-2026-06-13-victoriousdancing-kronomono.html"

_P1_TEAM = ["Sneasler", "Charizard", "Basculegion", "Zoroark-Hisui",
            "Corviknight", "Ditto"]
_P2_TEAM = ["Sneasler", "Charizard", "Whimsicott", "Basculegion",
            "Zoroark-Hisui", "Corviknight"]

_USERS = {"p1": "VictoriousDancing", "p2": "Kronomono"}
_ROSTERS = {"p1": _P1_TEAM, "p2": _P2_TEAM}


# ── Pure resolver ────────────────────────────────────────────────────────────
def test_resolve_match_returns_role():
    assert resolve_own_role(_USERS, _ROSTERS, "VictoriousDancing", _P1_TEAM) == "p1"
    assert resolve_own_role(_USERS, _ROSTERS, "Kronomono", _P2_TEAM) == "p2"


def test_resolve_username_is_normalised():
    # spaces / case / punctuation are ignored (Showdown username id rules)
    assert resolve_own_role(_USERS, _ROSTERS, "victorious dancing", _P1_TEAM) == "p1"
    assert resolve_own_role(_USERS, _ROSTERS, "  Victorious-Dancing ", _P1_TEAM) == "p1"


def test_resolve_mega_matches_base_forme():
    team = ["Charizard-Mega-Y" if s == "Charizard" else s for s in _P1_TEAM]
    assert resolve_own_role(_USERS, _ROSTERS, "VictoriousDancing", team) == "p1"


def test_resolve_requires_username_match():
    # correct team but the username belongs to nobody → no own side
    assert resolve_own_role(_USERS, _ROSTERS, "Nobody", _P1_TEAM) is None


def test_resolve_requires_team_match():
    # right username but a team that isn't that player's roster → no own side
    assert resolve_own_role(_USERS, _ROSTERS, "VictoriousDancing",
                            ["Pikachu", "Charizard"]) is None


def test_resolve_no_team_is_type_b():
    # nuance #1: no team configured ⇒ neither side is "ours" (Type B)
    assert resolve_own_role(_USERS, _ROSTERS, "VictoriousDancing", None) is None
    assert resolve_own_role(_USERS, _ROSTERS, "VictoriousDancing", []) is None


def test_resolve_no_username_is_none():
    assert resolve_own_role(_USERS, _ROSTERS, None, _P1_TEAM) is None


def test_resolve_mirror_selfplay_is_ambiguous():
    # same username AND team on both sides → undecidable here → None
    users = {"p1": "TrainerRed", "p2": "TrainerRed"}
    rosters = {"p1": _P1_TEAM, "p2": _P1_TEAM}
    assert resolve_own_role(users, rosters, "TrainerRed", _P1_TEAM) is None


# ── Team paste parsing ───────────────────────────────────────────────────────
def test_team_species_from_paste():
    paste = (
        "Charizard-Mega-Y @ Charizardite Y\n"
        "Ability: Drought\nLevel: 50\n- Heat Wave\n\n"
        "Zoroark-Hisui @ Choice Specs\nAbility: Illusion\nLevel: 50\n- Shadow Ball\n"
    )
    species = team_species_from_paste(paste)
    assert species == ["Charizard-Mega-Y", "Zoroark-Hisui"]
    # those species still resolve own side against base-forme rosters
    users = {"p1": "Me", "p2": "Foe"}
    rosters = {"p1": ["Charizard", "Zoroark-Hisui"], "p2": ["Pikachu"]}
    assert resolve_own_role(users, rosters, "Me", species) == "p1"


# ── Log-based resolution (the spectating/observing entry point) ──────────────
@pytest.fixture(scope="module")
def dual_log():
    hit = next(_VODS.rglob(_DUAL_NAME), None)
    if hit is None:   # vods are local-only (not in the published repo) — skip on CI
        pytest.skip(f"replay fixture not present (local-only): {_DUAL_NAME}")
    return extract_log_from_html(hit.read_text(encoding="utf-8"))


def test_parse_log_players_and_rosters(dual_log):
    users, rosters = parse_log_players_and_rosters(dual_log)
    assert users == _USERS
    assert rosters["p1"] == _P1_TEAM
    assert rosters["p2"] == _P2_TEAM


def test_own_role_from_log(dual_log):
    assert own_role_from_log(dual_log, "VictoriousDancing", _P1_TEAM) == "p1"
    assert own_role_from_log(dual_log, "Kronomono", _P2_TEAM) == "p2"
    assert own_role_from_log(dual_log, "VictoriousDancing", _P2_TEAM) is None
    assert own_role_from_log(dual_log, "VictoriousDancing", None) is None


# ── Battle adapter (playing case: poke-env has teampreview rosters) ──────────
def test_resolve_from_battle_with_teampreview(dual_log):
    pytest.importorskip("poke_env")
    from poke_env.battle import DoubleBattle, Pokemon
    import logging
    import _parity_harness as H

    battle = H.drive_live_battle(dual_log, "VictoriousDancing")  # player_role=p1
    assert battle.player_role == "p1"
    # In real play the teampreview request populates the OWN roster; a replay has
    # no requests, so inject it to exercise the adapter the way production runs.
    battle._teampreview_team = [Pokemon(gen=9, species=s) for s in _P1_TEAM]

    enc = StateEncoder(own_username="VictoriousDancing", own_team=_P1_TEAM)
    assert enc.resolve_own_role_from_battle(battle) == "p1"

    wrong = StateEncoder(own_username="SomeoneElse", own_team=_P1_TEAM)
    assert wrong.resolve_own_role_from_battle(battle) is None

    no_team = StateEncoder(own_username="VictoriousDancing")
    assert no_team.resolve_own_role_from_battle(battle) is None
