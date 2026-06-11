"""Tests for vod_parser/transitions.py — the mega-aware ability injection
(_inject_known_stats) and the two public entry points, run against the real
example VOD."""

from __future__ import annotations

import json

import pytest

from vod_parser.transitions import (
    _inject_known_stats,
    parse_replay_for_preview,
    replay_to_transitions,
)


# ── _inject_known_stats unit tests ────────────────────────────────────────

def _base_mon(**over):
    d = {
        "species": "Aerodactyl", "base_species": "Aerodactyl",
        "is_mega": False,
        "known_ability": None, "pre_mega_ability": None, "mega_ability": None,
        "known_item": None,
    }
    d.update(over)
    return d


def test_inject_non_mega_fills_ability():
    mon = _base_mon()
    _inject_known_stats(mon, {"ability": "Rock Head", "nature": "Jolly",
                              "item": "Focus Sash", "ev_spread": {"spe": 252},
                              "moves": ["Rock Slide"]})
    assert mon["known_ability"] == "Rock Head"
    assert mon["pre_mega_ability"] == "Rock Head"
    assert mon["nature"] == "Jolly"
    assert mon["known_item"] == "Focus Sash"
    assert mon["known_moves"] == ["Rock Slide"]


def test_inject_non_mega_does_not_override_revealed_ability():
    mon = _base_mon(known_ability="Unnerve", pre_mega_ability="Unnerve")
    _inject_known_stats(mon, {"ability": "Rock Head"})
    # Replay-revealed truth beats the user's guess
    assert mon["known_ability"] == "Unnerve"
    assert mon["pre_mega_ability"] == "Unnerve"


def test_inject_mega_mon_never_gets_base_ability_as_active():
    """THE core Bug 8 guarantee: injecting a base ability into a mega'd mon
    must land in pre_mega_ability only; the active ability is the pokedex
    mega ability."""
    mon = _base_mon(species="Meganium-Mega", base_species="Meganium",
                    is_mega=True)
    _inject_known_stats(mon, {"ability": "Overgrow"})
    assert mon["pre_mega_ability"] == "Overgrow"
    assert mon["known_ability"] == "Mega Sol"      # derived from pokedex
    assert mon["mega_ability"] == "Mega Sol"
    assert mon["known_ability"] != "Overgrow"


def test_inject_mega_mon_keeps_parser_resolved_mega_ability():
    mon = _base_mon(species="Floette-Mega", base_species="Floette-Eternal",
                    is_mega=True, known_ability="Fairy Aura",
                    mega_ability="Fairy Aura")
    _inject_known_stats(mon, {"ability": "Flower Veil"})
    assert mon["known_ability"] == "Fairy Aura"
    assert mon["pre_mega_ability"] == "Flower Veil"


def test_inject_empty_entry_is_noop():
    mon = _base_mon(known_ability="Unnerve")
    before = dict(mon)
    _inject_known_stats(mon, {})
    assert mon == before


# ── replay_to_transitions on the real VOD ─────────────────────────────────

@pytest.fixture(scope="module")
def known_teams(vod_path):
    """A known_teams entry like the team builder UI exports — note the
    ability for Floette-Eternal is the BASE forme ability (Flower Veil)."""
    return {
        # replay_id is the file stem fallback — match it from the path
        "gen9championsvgc2026regma-2026-04-20-stevenhevgc-speedyturtle87": None,
    }


def _entry():
    return {
        "_meta": {"yourSide": "p1", "winner": None,
                  "p1name": "stevenhe vgc", "p2name": "speedyturtle87"},
        "p1": {
            "Floette-Eternal": {"nature": "Modest", "item": "Floettite",
                                "ability": "Flower Veil",
                                "ev_spread": {"hp": 252, "spa": 252},
                                "moves": None},
            "Incineroar": {"nature": "Careful", "item": "Safety Goggles",
                           "ability": "Intimidate", "ev_spread": None,
                           "moves": None},
        },
        "p2": {},
    }


@pytest.fixture(scope="module")
def transitions(vod_path, vod_html):
    from vod_parser.replay_parser import extract_replay_id_from_html
    rid = extract_replay_id_from_html(vod_html) or vod_path.stem
    return replay_to_transitions(
        vod_path, None, None, players=["p1", "p2"],
        known_teams={rid: _entry()},
    )


def test_transitions_count_and_shape(transitions):
    assert transitions, "no transitions produced"
    t0 = transitions[0]
    for field in ("perspective", "turn", "state_before_actions",
                  "state_after_actions", "our_actions", "reward"):
        assert field in t0
    # Two perspectives per turn
    perspectives = {t["perspective"] for t in transitions}
    assert perspectives == {"p1", "p2"}


def test_transitions_mega_floette_ability_injection(transitions):
    """In p1-perspective transitions after the mega: Floette-Mega's active
    ability must be Fairy Aura (fixed), with the injected Flower Veil
    preserved as pre_mega_ability — never the other way round."""
    checked = False
    for t in transitions:
        if t["perspective"] != "p1":
            continue
        for mon in t["state_after_actions"]["our_active"].values():
            if mon["species"] == "Floette-Mega":
                checked = True
                assert mon["known_ability"] == "Fairy Aura"
                assert mon["mega_ability"] == "Fairy Aura"
                assert mon["pre_mega_ability"] == "Flower Veil"
                # Other injected stats matched via BASE species key
                assert mon["nature"] == "Modest"
                assert mon["ev_spread"] == {"hp": 252, "spa": 252}
    assert checked, "never saw a mega'd Floette in p1 transitions"


def test_transitions_non_mega_injection(transitions):
    checked = False
    for t in transitions:
        if t["perspective"] != "p1":
            continue
        for mon in list(t["state_after_actions"]["our_active"].values()) \
                 + list(t["state_after_actions"]["our_bench"]):
            if mon["species"] == "Incineroar" and mon.get("seen", True):
                checked = True
                assert mon["known_ability"] == "Intimidate"
                assert mon["mega_ability"] is None
    assert checked


def test_transitions_are_json_serialisable(transitions):
    for t in transitions:
        json.dumps(t)


def test_transitions_win_signal_on_final_turn(transitions):
    finals = [t for t in transitions if t["turn"] == t["total_turns"] or
              t["reward"]["win"] is not None]
    assert any(t["reward"]["win"] in (1, -1) for t in transitions), \
        "final-turn win signal missing"
    wins = {t["perspective"]: t["reward"]["win"]
            for t in transitions if t["reward"]["win"] is not None}
    assert set(wins.values()) == {1, -1}


# ── parse_replay_for_preview ──────────────────────────────────────────────

def test_preview_shape_and_overrides(vod_html):
    preview = parse_replay_for_preview(vod_html, None, _entry())
    assert preview["replay_id"]
    assert preview["turns"]
    # Flattened single-perspective snapshots for the UI
    assert "our_active" in preview["turns"][0]["state_before_actions"]
    # known_team_overrides surfaced for the inject panel
    assert "p1:Floette-Eternal" in preview["known_team_overrides"]
    assert preview["known_team_overrides"]["p1:Floette-Eternal"]["ability"] == "Flower Veil"
    # revealed_info carries the split ability fields + dropdown options
    ri = preview["revealed_info"]["p1:Floette-Eternal"]
    assert ri["mega_ability"] == "Fairy Aura"
    assert ri["possible_abilities"] == ["Flower Veil", "Symbiosis"]
