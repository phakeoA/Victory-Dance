"""Belief resolution must not let an EMPTY -Mega shell shadow the real base data.

Pikalytics rolls a mega's usage into the BASE species (the mega STONE is the base's #1
item), and the roster (enumerated from the Showdown champions mod) still lists every
-Mega forme, so each gets a present-but-empty entry. Without the _is_empty guard the exact
match on that shell wins and the opponent belief for a mega is blank — a real, mega-centric
gap in the Champions format. This locks in the fix (belief_state._resolve), and the reverse
direction (base empty, mega populated — the older synthetic case) still prefers the rich one.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from v_dance.parser.belief_state import BeliefState, is_mega_stone

_REPO = Path(__file__).resolve().parents[1]
_PIKA_MB = _REPO / "data" / "pikalytics_regmb.json"


def _belief(tmp_path, data: dict) -> BeliefState:
    p = tmp_path / "pika.json"
    p.write_text(json.dumps({"pokemon": data}), encoding="utf-8")
    return BeliefState(p)


_RICH = {
    "moves": [{"name": "Flamethrower", "pct": 90.0}],
    "items": [{"name": "Charizardite Y", "pct": 80.0}, {"name": "Life Orb", "pct": 20.0}],
    "abilities": [{"name": "Blaze", "pct": 70.0}],
    "spreads": [{"nature": "Timid", "evs": [0, 0, 0, 32, 0, 32], "pct": 100.0}],
}
_EMPTY = {"usage_pct": None, "moves": [], "items": [], "abilities": [], "spreads": []}


def test_empty_mega_shell_falls_through_to_base(tmp_path):
    b = _belief(tmp_path, {"Charizard": _RICH, "Charizard-Mega-Y": dict(_EMPTY),
                           "Charizard-Mega-X": dict(_EMPTY)})
    # both the hyphenated and the poke-env no-hyphen form resolve to the rich base
    assert b._resolve("Charizard-Mega-Y") == "Charizard"
    assert b._resolve("charizardmegay") == "Charizard"
    assert b.belief_block("Charizard-Mega-Y")["species_key"] == "Charizard"  # base data used
    assert len(b.top_moves("Charizard-Mega-Y")) == 1
    # the base's #1 item is its stone (Charizardite Y); the belief demotes it to the real item
    assert b.top_item("Charizard-Mega-Y") == "Life Orb"


def test_reverse_base_empty_mega_rich_prefers_mega(tmp_path):
    # the older synthetic case (base shell, mega populated): the rich mega entry wins
    b = _belief(tmp_path, {"Meganium": dict(_EMPTY),
                           "Meganium-Mega": {**_RICH, "items": [{"name": "Meganiumite", "pct": 100.0}]}})
    assert b._resolve("Meganium-Mega") == "Meganium-Mega"
    assert b.top_item("Meganium-Mega") == "Meganiumite"


def test_genuinely_empty_mon_stays_empty(tmp_path):
    # a zero-usage mon with no rich base: known() stays True, lookups return empty
    b = _belief(tmp_path, {"Gourgeist": dict(_EMPTY)})
    assert b._resolve("Gourgeist") == "Gourgeist"
    assert b.known("Gourgeist") is True
    assert b.top_moves("Gourgeist") == []


def test_stat_distinct_forme_resolves_exactly(tmp_path):
    # a forme with its OWN rich entry must NOT be stripped to a base
    b = _belief(tmp_path, {"Rotom": dict(_EMPTY), "Rotom-Wash": _RICH})
    assert b._resolve("Rotom-Wash") == "Rotom-Wash"


def test_empty_non_mega_forme_does_not_borrow_base(tmp_path):
    # a stat/ability-distinct non-mega forme (Galar/Midnight/Alola) with an EMPTY shell must NOT
    # borrow the base forme's data — that would hand it the wrong ability/type/spread. Only a
    # MEGA empty shell borrows the base (same mon line). Regression for the bug-review finding.
    b = _belief(tmp_path, {"Stunfisk": _RICH, "Stunfisk-Galar": dict(_EMPTY),
                           "Kangaskhan": dict(_RICH), "Kangaskhan-Mega": dict(_EMPTY)})
    assert b._resolve("Stunfisk-Galar") == "Stunfisk-Galar"   # keeps its own empty entry
    assert b.top_ability("Stunfisk-Galar") is None
    assert b.top_moves("Stunfisk-Galar") == []
    assert b._resolve("Kangaskhan-Mega") == "Kangaskhan"      # mega still borrows the base
    assert b.top_moves("Kangaskhan-Mega")


@pytest.mark.skipif(not _PIKA_MB.exists(), reason="regmb belief absent")
def test_real_galar_forme_not_contaminated():
    b = BeliefState(_PIKA_MB)
    # Stunfisk-Galar is an empty shell in regmb; it must NOT borrow regular Stunfisk's data
    assert b._resolve("Stunfisk-Galar") == "Stunfisk-Galar"
    assert b.top_ability("Stunfisk-Galar") is None


def test_is_empty_predicate(tmp_path):
    b = _belief(tmp_path, {"X": _RICH})
    assert b._is_empty(None) is True
    assert b._is_empty({}) is True
    assert b._is_empty(dict(_EMPTY)) is True
    assert b._is_empty({"abilities": [{"name": "Overgrow", "pct": 1.0}]}) is False  # any usable field
    assert b._is_empty(_RICH) is False


@pytest.mark.skipif(not _PIKA_MB.exists(), reason="regmb belief absent")
def test_real_regmb_megas_inherit_base():
    b = BeliefState(_PIKA_MB)
    for mega in ("Charizard-Mega-Y", "Gardevoir-Mega", "Kangaskhan-Mega", "Abomasnow-Mega"):
        moves = b.top_moves(mega)
        assert moves, f"{mega} still has no belief (shell shadowed the base)"


# ── mega-STONE item demotion: assume the non-stone item unless confirmed mega ─────
def _stoned(name, pct, *more):
    items = [{"name": name, "pct": pct}]
    items += [{"name": n, "pct": p} for n, p in more]
    return {"moves": [{"name": "Tackle", "pct": 1.0}], "items": items,
            "abilities": [{"name": "Torrent", "pct": 1.0}], "spreads": []}


def test_item_belief_demotes_mega_stone_unless_confirmed(tmp_path):
    assert is_mega_stone("Swampertite") and not is_mega_stone("Life Orb")
    b = _belief(tmp_path, {"Swampert": _stoned("Swampertite", 98.5, ("Life Orb", 0.7), ("Sitrus Berry", 0.3))})
    # NOT confirmed mega -> best NON-stone item (the user's Swampert -> Life Orb)
    assert b.top_item("Swampert") == "Life Orb"
    assert b.belief_block("Swampert")["items"][0]["name"] == "Life Orb"   # offline == live parity
    # confirmed mega (revealed stone) still wins outright
    blk = b.belief_block("Swampert", revealed_item="mega stone")
    assert blk["items"] == [{"name": "mega stone", "p": 1.0, "revealed": True}]


def test_double_mega_stone_skips_both(tmp_path):
    # Charizard's top TWO items are stones -> demotion must skip BOTH to the real item
    b = _belief(tmp_path, {"Charizard": _stoned("Charizardite Y", 95.0,
                                                ("Charizardite X", 4.0), ("Focus Sash", 1.0))})
    assert b.top_item("Charizard") == "Focus Sash"
    assert b.belief_block("Charizard")["items"][0]["name"] == "Focus Sash"


def test_all_stone_items_keep_stone_as_last_resort(tmp_path):
    # an always-mega mon with no non-stone usage: keep the stone rather than return nothing
    b = _belief(tmp_path, {"Diancie": _stoned("Swampertite", 99.0)})  # any real stone id
    assert b.top_item("Diancie") == "Swampertite"


def test_demotion_scans_full_item_list_past_top_k_window(tmp_path):
    # >9 leading mega stones before the first non-stone item: the old top_k+4 (=9) window kept the
    # inert stone as belief_block items[0] while top_item (full list) found the real item — a latent
    # live/offline divergence one data refresh away.  belief_block now demotes over the FULL list.
    stones = ["Charizardite Y", "Charizardite X", "Swampertite", "Venusaurite", "Blastoisinite",
              "Gardevoirite", "Kangaskhanite", "Aerodactylite", "Gengarite", "Lucarionite"]
    for s in stones:
        assert is_mega_stone(s), f"{s} is not a recognised mega stone — fix the fixture"
    more = [(s, 90.0 - i) for i, s in enumerate(stones[1:], start=1)] + [("Life Orb", 1.0)]
    b = _belief(tmp_path, {"Charizard": _stoned(stones[0], 95.0, *more)})   # 10 stones, then Life Orb
    assert b.top_item("Charizard") == "Life Orb"
    assert b.belief_block("Charizard")["items"][0]["name"] == "Life Orb"   # full-list demotion parity


@pytest.mark.skipif(not _PIKA_MB.exists(), reason="regmb belief absent")
def test_real_item_belief_parity_and_no_stone():
    b = BeliefState(_PIKA_MB)
    for sp in ("Swampert", "Charizard", "Gardevoir", "Kangaskhan"):
        ti = b.top_item(sp)
        blk = b.belief_block(sp)
        assert ti == blk["items"][0]["name"], f"{sp}: live/offline item belief diverge"
        assert not is_mega_stone(ti), f"{sp}: belief assumed the inert mega stone"


@pytest.mark.skipif(not _PIKA_MB.exists(), reason="regmb belief absent")
def test_real_item_belief_parity_all_species():
    # full-corpus guard: top_item (live) and belief_block[0] (offline) must agree for EVERY species,
    # so a future Pikalytics refresh can't silently reintroduce a stone-vs-real-item divergence.
    b = BeliefState(_PIKA_MB)
    data = json.loads(_PIKA_MB.read_text(encoding="utf-8"))["pokemon"]
    mismatches = []
    for sp in data:
        ti = b.top_item(sp)
        if ti is None:
            continue
        items = (b.belief_block(sp).get("items") or [])
        if items and ti != items[0]["name"]:
            mismatches.append((sp, ti, items[0]["name"]))
    assert not mismatches, f"top_item vs belief_block[0] diverge for: {mismatches[:10]}"
