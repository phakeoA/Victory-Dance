"""Tests for vod_parser/replay_parser.py — focused on the Bug 8 fix:
a mega-evolving Pokémon has a pre-mega ability AND a (single, fixed)
post-mega ability, and the parser must never conflate the two.

Covers:
  * pre-mega ability demotion on |detailschange|
  * deterministic mega-ability resolution from the pokedex
  * |-ability| routing pre- vs post-mega
  * non-mega |detailschange| (forme change) leaving ability state alone
  * |-mega| safety net when |detailschange| is missing
  * mega switch-out/switch-back continuity
  * the real example VOD end-to-end
"""

from __future__ import annotations

import pytest

from conftest import HEADER, make_log
from vod_parser.replay_parser import (
    ShowdownReplayParser,
    extract_log_from_html,
    extract_replay_id_from_html,
)


def parse(*lines: str) -> dict:
    return ShowdownReplayParser(make_log(*HEADER, *lines), our_player="p1").parse()


# ── Synthetic: ability revealed BEFORE mega ──────────────────────────────

def test_pre_mega_ability_demoted_on_mega():
    """Intimidate revealed pre-mega must survive as pre_mega_ability, and the
    active ability must become the mega forme's fixed ability."""
    r = parse(
        "|switch|p1a: Meganium|Meganium, L50, M|100/100",
        "|switch|p2a: Aerodactyl|Aerodactyl, L50, M|100/100",
        "|turn|1",
        "|-ability|p1a: Meganium|Overgrow",        # pre-mega reveal
        "|turn|2",
        "|detailschange|p1a: Meganium|Meganium-Mega, L50, M",
        "|-mega|p1a: Meganium|Meganium|Meganiumite",
        "|turn|3",
        "|win|alice",
    )
    info = r["revealed_info"]["p1:Meganium"]
    assert info["is_mega"] is True
    assert info["pre_mega_ability"] == "Overgrow"
    assert info["mega_ability"] == "Mega Sol"        # from pokedex, never revealed
    assert info["known_ability"] == "Mega Sol"       # currently active
    assert info["mega_species"] == "Meganium-Mega"


def test_mega_ability_resolved_without_any_ability_line():
    """Even with NO |-ability| line at all, the mega ability is known the
    instant the mega happens — it is fully determined by the forme."""
    r = parse(
        "|switch|p1a: Meganium|Meganium, L50, M|100/100",
        "|turn|1",
        "|detailschange|p1a: Meganium|Meganium-Mega, L50, M",
        "|-mega|p1a: Meganium|Meganium|Meganiumite",
        "|win|alice",
    )
    info = r["revealed_info"]["p1:Meganium"]
    assert info["mega_ability"] == "Mega Sol"
    assert info["known_ability"] == "Mega Sol"
    assert info["pre_mega_ability"] is None          # never revealed → unknown


def test_ability_reveal_after_mega_routes_to_mega_ability():
    r = parse(
        "|switch|p1a: Meganium|Meganium, L50, M|100/100",
        "|turn|1",
        "|detailschange|p1a: Meganium|Meganium-Mega, L50, M",
        "|-mega|p1a: Meganium|Meganium|Meganiumite",
        "|-ability|p1a: Meganium|Mega Sol",
        "|win|alice",
    )
    info = r["revealed_info"]["p1:Meganium"]
    assert info["mega_ability"] == "Mega Sol"
    assert info["pre_mega_ability"] is None
    # The action log flags it as a mega (fixed) ability
    abilities = [a for t in r["turns"] for a in t["actions"]
                 if a["event"] == "ability_revealed"]
    assert abilities and abilities[-1]["is_mega_ability"] is True


def test_ability_reveal_before_mega_routes_to_pre_mega():
    r = parse(
        "|switch|p2a: Aerodactyl|Aerodactyl, L50, M|100/100",
        "|turn|1",
        "|-ability|p2a: Aerodactyl|Unnerve",
        "|win|alice",
    )
    info = r["revealed_info"]["p2:Aerodactyl"]
    assert info["pre_mega_ability"] == "Unnerve"
    assert info["known_ability"] == "Unnerve"
    assert info["mega_ability"] is None
    assert info["is_mega"] is False


def test_mega_evolution_event_carries_both_ability_contexts():
    r = parse(
        "|switch|p1a: Meganium|Meganium, L50, M|100/100",
        "|turn|1",
        "|-ability|p1a: Meganium|Leaf Guard",
        "|detailschange|p1a: Meganium|Meganium-Mega, L50, M",
        "|-mega|p1a: Meganium|Meganium|Meganiumite",
        "|win|alice",
    )
    ev = next(a for t in r["turns"] for a in t["actions"]
              if a["event"] == "mega_evolution")
    assert ev["pre_mega_ability"] == "Leaf Guard"
    assert ev["mega_ability"] == "Mega Sol"
    assert ev["mega_stone"] == "Meganiumite"


# ── Synthetic: non-mega forme changes must NOT trigger the mega path ─────

def test_non_mega_detailschange_is_forme_change():
    """Palafin → Palafin-Hero fires |detailschange| but is NOT a mega:
    no is_mega flag, no ability swap, no mega_evolution event."""
    r = parse(
        "|switch|p2a: Palafin|Palafin, L50, M|100/100",
        "|turn|1",
        "|-ability|p2a: Palafin|Zero to Hero",
        "|detailschange|p2a: Palafin|Palafin-Hero, L50, M",
        "|win|alice",
    )
    info = r["revealed_info"]["p2:Palafin"]
    assert info["is_mega"] is False
    assert info["mega_ability"] is None
    assert info["known_ability"] == "Zero to Hero"   # untouched by forme change
    events = [a["event"] for t in r["turns"] for a in t["actions"]]
    assert "forme_change" in events
    assert "mega_evolution" not in events


def test_meganium_name_does_not_false_positive_as_mega():
    """'Meganium' contains 'mega' — plain switches must never set is_mega."""
    r = parse(
        "|switch|p1a: Meganium|Meganium, L50, M|100/100",
        "|turn|1",
        "|win|alice",
    )
    assert r["revealed_info"]["p1:Meganium"]["is_mega"] is False


# ── Synthetic: |-mega| safety net ─────────────────────────────────────────

def test_mega_line_without_detailschange_still_swaps_ability():
    """Some replays could carry |-mega| without a usable |detailschange|;
    the safety net must still flag the mega and clear the stale ability."""
    r = parse(
        "|switch|p1a: Meganium|Meganium, L50, M|100/100",
        "|turn|1",
        "|-ability|p1a: Meganium|Overgrow",
        "|-mega|p1a: Meganium|Meganium|Meganiumite",
        "|win|alice",
    )
    info = r["revealed_info"]["p1:Meganium"]
    assert info["is_mega"] is True
    assert info["pre_mega_ability"] == "Overgrow"
    # species was never mutated to the mega forme, so the dex can't resolve
    # the mega ability — but the stale pre-mega ability must NOT be kept as
    # the active one.  (known_ability is None until a reveal fills it.)
    assert info["known_ability"] != "Overgrow" or info["known_ability"] is None


# ── Synthetic: mega switch-back continuity ────────────────────────────────

def test_mega_switch_back_in_keeps_identity_and_ability():
    """A mega'd mon switching out and back in shows its MEGA forme name on
    the |switch| line — it must reconcile with its base-keyed seen_mons
    entry instead of forking a duplicate."""
    r = parse(
        "|switch|p1a: Meganium|Meganium, L50, M|100/100",
        "|switch|p1b: Incineroar|Incineroar, L50, F|100/100",
        "|turn|1",
        "|detailschange|p1a: Meganium|Meganium-Mega, L50, M",
        "|-mega|p1a: Meganium|Meganium|Meganiumite",
        "|turn|2",
        "|switch|p1a: Incineroar|Incineroar, L50, F|100/100",
        "|turn|3",
        "|switch|p1a: Meganium|Meganium-Mega, L50, M|80/100",
        "|turn|4",
        "|win|alice",
    )
    # Exactly one Meganium in revealed_info / known_team — no fork
    meg_keys = [k for k in r["revealed_info"] if "Meganium" in k]
    assert meg_keys == ["p1:Meganium"]
    assert r["turns"][-1]["state_before_actions"]["p1"]["known_team"]["p1"].count("Meganium") == 1
    info = r["revealed_info"]["p1:Meganium"]
    assert info["is_mega"] is True
    assert info["mega_ability"] == "Mega Sol"


# ── Real VOD end-to-end ───────────────────────────────────────────────────

@pytest.fixture(scope="module")
def vod_result(vod_html):
    log = extract_log_from_html(vod_html)
    return ShowdownReplayParser(log, our_player="p1").parse()


def test_vod_floette_mega_ability(vod_result):
    info = vod_result["revealed_info"]["p1:Floette-Eternal"]
    assert info["is_mega"] is True
    assert info["mega_species"] == "Floette-Mega"
    assert info["mega_ability"] == "Fairy Aura"
    assert info["known_ability"] == "Fairy Aura"
    # Floette's base ability was never revealed pre-mega in this replay
    assert info["pre_mega_ability"] is None
    # Dropdown data for the inject panel
    assert info["possible_abilities"] == ["Flower Veil", "Symbiosis"]
    assert {"forme": "Floette-Mega", "ability": "Fairy Aura"} in info["mega_formes"]


def test_vod_meganium_mega_ability_resolved_from_dex_alone(vod_result):
    """Meganium-Mega's 'Mega Sol' never appears in the log — it must come
    purely from the pokedex the moment |detailschange| fires."""
    info = vod_result["revealed_info"]["p2:Meganium"]
    assert info["is_mega"] is True
    assert info["mega_ability"] == "Mega Sol"
    assert info["known_ability"] == "Mega Sol"
    assert info["possible_abilities"] == ["Overgrow", "Leaf Guard"]


def test_vod_non_mega_abilities_stay_pre_mega(vod_result):
    aero = vod_result["revealed_info"]["p2:Aerodactyl"]
    assert aero["is_mega"] is False
    assert aero["known_ability"] == "Unnerve"
    assert aero["pre_mega_ability"] == "Unnerve"
    assert aero["mega_ability"] is None

    inc = vod_result["revealed_info"]["p1:Incineroar"]
    assert inc["known_ability"] == "Intimidate"
    assert inc["mega_ability"] is None


def test_vod_active_snapshot_carries_split_ability_fields(vod_result):
    """After Floette megas, every later snapshot of it must show the mega
    ability as active and never the (unknown) base ability."""
    saw_mega_floette = False
    for turn in vod_result["turns"]:
        snap = turn["state_after_actions"]["p1"]
        for mon in snap["our_active"].values():
            if mon["species"] == "Floette-Mega":
                saw_mega_floette = True
                assert mon["is_mega"] is True
                assert mon["known_ability"] == "Fairy Aura"
                assert mon["mega_ability"] == "Fairy Aura"
                assert mon["base_species"] == "Floette-Eternal"
    assert saw_mega_floette


def test_vod_metadata(vod_html, vod_result):
    assert extract_replay_id_from_html(vod_html)
    assert vod_result["players"]["p1"]["username"]
    assert vod_result["players"]["p2"]["username"]
    assert len(vod_result["turns"]) > 0
    assert vod_result["winner"] in (
        vod_result["players"]["p1"]["username"],
        vod_result["players"]["p2"]["username"],
    )
