"""Tests for vod_parser/transitions.py — the mega-aware ability injection
(_inject_known_stats) and the two public entry points, run against the real
example VOD."""

from __future__ import annotations

import json

import pytest

from v_dance.parser.vod_parser.transitions import (
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


# ── known_moves placeholder cleaning + revealed merge ─────────────────────

def test_inject_moves_strips_ui_placeholders():
    """The UI sends fixed 4-slot arrays — '' placeholders must not leak."""
    mon = _base_mon(revealed_moves=["Fake Out"])
    _inject_known_stats(mon, {"moves": ["", "Close Combat", "", ""]})
    assert mon["known_moves"] == ["Close Combat", "Fake Out"]


def test_inject_moves_all_blank_leaves_known_moves_unset():
    mon = _base_mon(revealed_moves=["Fake Out"])
    _inject_known_stats(mon, {"moves": ["", "", "", ""]})
    assert "known_moves" not in mon


def test_inject_moves_merges_and_dedups_revealed():
    mon = _base_mon(revealed_moves=["Close Combat", "Knock Off"])
    _inject_known_stats(mon, {"moves": ["close combat", "Parting Shot"]})
    # injected order first, revealed extras appended, loose-match dedup
    assert mon["known_moves"] == ["close combat", "Parting Shot", "Knock Off"]


def test_inject_moves_caps_at_four():
    mon = _base_mon(revealed_moves=[])
    _inject_known_stats(mon, {"moves": ["A", "B", "C", "D", "E"]})
    assert mon["known_moves"] == ["A", "B", "C", "D"]


def test_inject_moves_revealed_survive_the_cap():
    """A typed move that contradicts 4 replay-revealed moves is the wrong
    datum — reveals are ground truth and must never be dropped."""
    mon = _base_mon(revealed_moves=["Dazzling Gleam", "Protect",
                                    "Calm Mind", "Draining Kiss"])
    _inject_known_stats(mon, {"moves": ["Moonblast", "", "", ""]})
    assert mon["known_moves"] == ["Dazzling Gleam", "Protect",
                                  "Calm Mind", "Draining Kiss"]


def test_inject_moves_typed_kept_when_room_remains():
    mon = _base_mon(revealed_moves=["Knock Off", "Fake Out"])
    _inject_known_stats(mon, {"moves": ["Parting Shot", "Flare Blitz", "", ""]})
    assert mon["known_moves"] == ["Parting Shot", "Flare Blitz",
                                  "Knock Off", "Fake Out"]


def test_inject_no_moves_does_not_promote_revealed():
    mon = _base_mon(revealed_moves=["Fake Out"])
    _inject_known_stats(mon, {"item": "Sitrus Berry"})
    assert "known_moves" not in mon


# ── iv_spread default ──────────────────────────────────────────────────────

def test_inject_stats_fills_default_ivs():
    mon = _base_mon(iv_spread=None)
    _inject_known_stats(mon, {"nature": "Jolly", "ev_spread": {"spe": 252}})
    assert mon["iv_spread"] == [31] * 6


def test_inject_item_only_leaves_ivs_alone():
    mon = _base_mon(iv_spread=None)
    _inject_known_stats(mon, {"item": "Clear Amulet"})
    assert mon["iv_spread"] is None


def test_inject_partial_does_not_clobber_existing_stats():
    mon = _base_mon(ev_spread=[0, 32, 0, 0, 0, 0], nature="Adamant")
    _inject_known_stats(mon, {"item": "Choice Band"})
    assert mon["ev_spread"] == [0, 32, 0, 0, 0, 0]
    assert mon["nature"] == "Adamant"


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
    from v_dance.parser.vod_parser.replay_parser import extract_replay_id_from_html
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


def test_transitions_inject_reaches_state_before(transitions):
    """state_before_actions is the model's INPUT — injected stats must land
    there, not only in state_after_actions."""
    checked = False
    for t in transitions:
        if t["perspective"] != "p1":
            continue
        snap = t["state_before_actions"]
        for mon in list(snap["our_active"].values()) + list(snap["our_bench"]):
            base = mon.get("base_species") or mon["species"]
            if base == "Floette-Eternal" and mon.get("seen", True):
                checked = True
                assert mon["nature"] == "Modest"
                assert mon["ev_spread"] is not None
            if mon["species"] == "Incineroar" and mon.get("seen", True):
                assert mon["known_ability"] == "Intimidate"
                assert mon["known_item"] == "Safety Goggles"
    assert checked, "never saw Floette in p1 state_before snapshots"


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


# ── fill_blanks enrichment inside the export path ─────────────────────────

@pytest.fixture(scope="module")
def transitions_type_a(vod_path, vod_html):
    """Type A (own VOD) export: our side exact from the inject panel,
    opponent side Pikalytics distributions."""
    from v_dance.parser.vod_parser.replay_parser import extract_replay_id_from_html
    rid = extract_replay_id_from_html(vod_html) or vod_path.stem
    return replay_to_transitions(
        vod_path, None, None, players=["p1", "p2"],
        known_teams={rid: _entry()}, source_type="own_vod",
    )


def test_type_a_sheet_complete_mon_exact_in_both_snapshots(transitions_type_a):
    """A nature+EV-complete inject card becomes a full exact block — with
    computed stats, bucket EV list and default IVs — in state_before AND
    state_after (state_before is what the model trains on)."""
    seen_snaps = set()
    for t in transitions_type_a:
        if t["perspective"] != "p1":
            continue
        for snap_key in ("state_before_actions", "state_after_actions"):
            snap = t[snap_key]
            for mon in list(snap["our_active"].values()) + list(snap["our_bench"]):
                base = mon.get("base_species") or mon["species"]
                if base == "Floette-Eternal" and mon.get("seen", True):
                    seen_snaps.add(snap_key)
                    assert mon["exact"]["source"] == "team_sheet"
                    assert mon["exact"]["nature"] == "Modest"
                    assert mon["ev_spread"] == [32, 0, 0, 32, 0, 0]   # bucket scale
                    assert mon["iv_spread"] == [31] * 6
                    assert mon["stats_estimate"]["mode"] == "exact"
                    assert (mon["exact"]["stats"] or {}).get("hp", 0) > 0
    assert seen_snaps == {"state_before_actions", "state_after_actions"}


def test_type_a_opp_side_gets_belief_distributions(transitions_type_a):
    """The spec's biggest export gap: opponent mons must carry Pikalytics
    belief blocks + distribution stats_estimate — including unseen bench."""
    found_active = found_unseen = False
    found_expected_stats = found_predicted_moves = False
    for t in transitions_type_a:
        if t["perspective"] != "p1":
            continue
        snap = t["state_before_actions"]
        for mon in snap["opp_active"].values():
            if mon.get("belief"):
                found_active = True
                assert mon["stats_estimate"]["mode"] == "distribution"
                # sparse Pikalytics entries (e.g. Meganium: usage entry but
                # no EV/move tables) legitimately leave these None/empty —
                # require the mechanism to work somewhere, not on every mon
                if mon["belief"]["expected_stats"]:
                    found_expected_stats = True
                if mon["belief"]["moves_predicted"]:
                    found_predicted_moves = True
        for mon in snap["opp_bench"]:
            if mon.get("seen") is False and mon.get("belief"):
                found_unseen = True
    assert found_active, "no opp active mon carried a belief block"
    assert found_unseen, "unseen opp bench mons should still get belief blocks"
    assert found_expected_stats, "no opp mon ever got expected stats"
    assert found_predicted_moves, "no opp mon ever got predicted moves"


def test_type_a_partial_inject_card_warned_not_exact(transitions_type_a):
    """Incineroar's card has nature but no EVs — it must NOT get fake exact
    stats; it keeps the legacy field injection and raises a warning."""
    warns = transitions_type_a[0]["belief_fill"]["warnings"]
    assert any("Incineroar" in w for w in warns)
    for t in transitions_type_a:
        if t["perspective"] != "p1":
            continue
        for snap_key in ("state_before_actions", "state_after_actions"):
            snap = t[snap_key]
            for mon in list(snap["our_active"].values()) + list(snap["our_bench"]):
                if mon["species"] == "Incineroar" and mon.get("seen", True):
                    assert not mon.get("exact")
                    assert mon["known_ability"] == "Intimidate"


def test_type_b_both_sides_distribution(transitions):
    """Type B (default): OUR side gets belief distributions too — nobody's
    exact stats are known from a ranked player VOD."""
    found = False
    for t in transitions:
        if t["perspective"] != "p1":
            continue
        snap = t["state_before_actions"]
        for mon in snap["our_active"].values():
            if mon.get("belief"):
                found = True
                assert mon["stats_estimate"]["mode"] == "distribution"
    assert found, "no p1 active mon carried a belief block under Type B"


def test_belief_fill_metadata_attached_and_trimmed(transitions):
    for t in transitions:
        bf = t["belief_fill"]
        assert bf["vod_type"] == "B"
        assert bf["fill_modes"] == {"p1": "distribution", "p2": "distribution"}
        assert bf["pikalytics_source"]
        assert "back_calc" not in bf         # bulky stub output stays CLI-only


# ── own-side knowledge retrofit (decision-state reconstruction) ───────────

def test_retrofit_own_bench_stubs_for_unentered_brought(transitions):
    """At turn 1 only the leads have entered, but the acting player knew
    their full brought set — the back two must appear as seen=False bench
    stubs so switch decisions are expressible."""
    t1 = next(t for t in transitions
              if t["turn"] == 1 and t["perspective"] == "p1")
    bench = t1["state_before_actions"]["our_bench"]
    brought = t1["players"]["p1"]["brought"]
    assert len(brought) >= 3          # this VOD brings 4
    bench_species = {m.get("base_species") or m["species"] for m in bench}
    for sp in brought[2:]:            # non-leads
        assert sp in bench_species
    assert all(m.get("seen") is False for m in bench), \
        "unentered brought mons are stubs, not 'seen' mons"


def test_retrofit_own_moves_are_battle_final(transitions):
    """Our own mons carry their battle-end reveal list from turn 1 — the
    acting player knew their own moveset all along."""
    by_persp_turn = {(t["perspective"], t["turn"]): t for t in transitions}
    first = by_persp_turn[("p1", 1)]
    last_turn = max(t["turn"] for t in transitions)
    last = by_persp_turn[("p1", last_turn)]

    def own_moves(t):
        snap = t["state_before_actions"]
        out = {}
        for mon in list(snap["our_active"].values()) + list(snap["our_bench"]):
            base = mon.get("base_species") or mon["species"]
            out[base] = list(mon.get("revealed_moves") or [])
        return out

    first_moves, last_moves = own_moves(first), own_moves(last)
    assert any(first_moves.values()), "retrofit left every own moveset empty"
    for sp, moves in last_moves.items():
        if sp in first_moves:
            assert first_moves[sp] == moves, \
                f"{sp}: own movelist must be battle-stable from turn 1"


def test_retrofit_does_not_leak_into_opp_side(transitions):
    """The opponent half of a snapshot keeps the progressive view — at the
    start of turn 1 nothing of theirs has been revealed yet."""
    t1 = next(t for t in transitions
              if t["turn"] == 1 and t["perspective"] == "p1")
    snap = t1["state_before_actions"]
    for mon in snap["opp_active"].values():
        assert mon.get("revealed_moves") in ([], None), \
            f"opp {mon['species']} leaked future moves at turn 1"


# ── source_type passthrough (UI Type A/B/C/D selector → /export) ──────────

def test_transitions_default_source_type_is_ranked(transitions):
    assert {t["source_type"] for t in transitions} == {"ranked_player_vod"}


@pytest.mark.parametrize("ui_token, canonical", [
    ("own_vod",           "own_vod"),
    ("ranked_player_vod", "ranked_player_vod"),
    ("bot_vod",           "live_bot_battle"),
    ("self_play",         "self_play"),
    ("A",                 "own_vod"),
])
def test_transitions_source_type_canonicalised(vod_path, ui_token, canonical):
    ts = replay_to_transitions(
        vod_path, None, None, players=["p1"], known_teams=None,
        source_type=ui_token,
    )
    assert ts
    assert {t["source_type"] for t in ts} == {canonical}


def test_transitions_unknown_source_type_raises(vod_path):
    with pytest.raises(ValueError):
        replay_to_transitions(vod_path, None, None, players=["p1"],
                              source_type="banana")


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
