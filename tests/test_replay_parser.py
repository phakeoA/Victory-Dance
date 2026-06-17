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
from v_dance.parser.vod_parser.replay_parser import (
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


# ── Synthetic: mega-evolution DECISION label join onto the chosen move ─────
# The mega is a checkbox on the move the player selected; the parser stamps
# action["mega"]=True onto the SAME-turn, SAME-slot move action.  (Task #1 —
# the learned-mega blocker: without this flag the training data has no label.)

def _mega_move_turn():
    """p1a Meganium megas then attacks; p1b Incineroar attacks WITHOUT mega;
    p2a Aerodactyl attacks (no mega).  One turn, three move decisions."""
    return parse(
        "|switch|p1a: Meganium|Meganium, L50, M|100/100",
        "|switch|p1b: Incineroar|Incineroar, L50, F|100/100",
        "|switch|p2a: Aerodactyl|Aerodactyl, L50, M|100/100",
        "|turn|1",
        "|detailschange|p1a: Meganium|Meganium-Mega, L50, M",
        "|-mega|p1a: Meganium|Meganium|Meganiumite",
        "|move|p1a: Meganium|Body Press|p2a: Aerodactyl",
        "|move|p1b: Incineroar|Fake Out|p2a: Aerodactyl",
        "|move|p2a: Aerodactyl|Rock Slide|p1a: Meganium",
        "|turn|2",
        "|win|alice",
    )


def test_mega_label_stamped_on_same_slot_move():
    """The megaing slot's move carries mega=True; a non-mega teammate move does
    not (the flag is only added when a mega actually occurred)."""
    turn = _mega_move_turn()["turns"][0]
    our = {a["slot"]: a for a in turn["our_actions"]}
    assert our["p1a"]["action"] == "move"
    assert our["p1a"].get("mega") is True            # p1a mega'd → flagged
    assert "mega" not in our["p1b"]                   # p1b didn't → no key


def test_mega_label_absent_on_opponent_nonmega_move():
    """A move by a slot that never mega'd (here the opponent) carries no flag."""
    turn = _mega_move_turn()["turns"][0]
    opp = {a["slot"]: a for a in turn["opp_actions_actual"]}
    assert opp["p2a"]["action"] == "move"
    assert "mega" not in opp["p2a"]


def test_forme_change_does_not_stamp_mega():
    """Palafin → Palafin-Hero is an INVOLUNTARY forme change, not a player
    decision — the move that turn must never be flagged as a gimmick."""
    r = parse(
        "|switch|p2a: Palafin|Palafin, L50, M|100/100",
        "|switch|p1a: Meganium|Meganium, L50, M|100/100",
        "|turn|1",
        "|detailschange|p2a: Palafin|Palafin-Hero, L50, M",
        "|move|p2a: Palafin|Jet Punch|p1a: Meganium",
        "|turn|2",
        "|win|alice",
    )
    turn = r["turns"][0]
    # forme_change emitted, mega_evolution NOT — so no flag on the move.
    events = [a["event"] for a in turn["actions"]]
    assert "forme_change" in events and "mega_evolution" not in events
    opp = {a["slot"]: a for a in turn["opp_actions_actual"]}
    assert opp["p2a"]["action"] == "move"
    assert "mega" not in opp["p2a"]


def test_mega_then_flinch_no_label_and_no_teammate_leak():
    """A mon that megas at turn-start then FLINCHES makes no move that turn:
    there is no action to flag (an unavoidable orphan), and the join must not
    leak the flag onto a teammate's move."""
    r = parse(
        "|switch|p1a: Meganium|Meganium, L50, M|100/100",
        "|switch|p1b: Incineroar|Incineroar, L50, F|100/100",
        "|switch|p2a: Aerodactyl|Aerodactyl, L50, M|100/100",
        "|turn|1",
        "|detailschange|p1a: Meganium|Meganium-Mega, L50, M",
        "|-mega|p1a: Meganium|Meganium|Meganiumite",
        "|move|p2a: Aerodactyl|Fake Out|p1a: Meganium",
        "|-damage|p1a: Meganium|86/100",
        "|cant|p1a: Meganium|flinch",
        "|move|p1b: Incineroar|Knock Off|p2a: Aerodactyl",
        "|turn|2",
        "|win|alice",
    )
    turn = r["turns"][0]
    our = {a["slot"]: a for a in turn["our_actions"]}
    assert "p1a" not in our                            # mega'd but never moved
    assert our["p1b"]["action"] == "move"
    assert "mega" not in our["p1b"]                     # teammate not flagged
    assert any(a["event"] == "mega_evolution" for a in turn["actions"])


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


# ── Gap #5: HP parsing (real-HP scale + malformed tokens) ─────────────────────
@pytest.mark.parametrize("hp_str,expected", [
    ("100/100", (100.0, 100.0)),
    ("74/100", (74.0, 100.0)),
    ("175/200", (175.0, 200.0)),      # real-HP scale (owner-recorded replay)
    ("0 fnt", (0.0, 100.0)),          # faint marker stripped
    ("50/100 brn", (50.0, 100.0)),    # status suffix (space) stripped
    ("50/100y", (50.0, 100.0)),       # stray no-separator suffix tolerated
    ("197/197", (197.0, 197.0)),
    ("", (None, None)),
    ("garbage", (None, None)),
])
def test_parse_hp_handles_real_scale_and_malformed(hp_str, expected):
    """_parse_hp returns (numerator, denominator), tolerating real-HP scale and
    stray suffixes (e.g. '50/100y') the same way poke-env's client does — an
    unparsed token previously left an active mon with hp_pct=None (gap #5)."""
    assert ShowdownReplayParser._parse_hp(hp_str) == expected


def test_real_hp_damage_event_reported_as_percentage():
    """A damage line in real-HP units (175/200) must yield a true PERCENTAGE in
    both the snapshot hp_pct AND the damage event's hp_pct_after (gap #5)."""
    r = parse(
        "|switch|p1a: Dondozo|Dondozo, L50, M|200/200",
        "|switch|p2a: Aerodactyl|Aerodactyl, L50, M|100/100",
        "|turn|1",
        "|move|p2a: Aerodactyl|Rock Slide|p1a: Dondozo",
        "|-damage|p1a: Dondozo|150/200",
        "|turn|2",
        "|win|alice",
    )
    snap = r["turns"][0]["state_after_actions"]["p1"]["our_active"]["our_a"]
    assert snap["hp_pct"] == 75.0          # 150/200, NOT 150
    dmg = [e for t in r["turns"] for e in t.get("damage_events", [])
           if e.get("species") == "Dondozo"]
    assert dmg and dmg[0]["hp_pct_after"] == 75.0


# ── Gap #6 sub-cause 1: |swap| (Ally Switch) position tracking ───────────────

def test_handle_swap_exchanges_active_slot_pointers():
    """``|swap|POKEMON|POSITION`` must exchange the two active-slot pointers and
    keep each mon's ``.slot`` letter consistent.  poke-env applies Ally Switch
    to its DoubleBattle, so the offline parser must too (Gap #6 sub-cause 1)."""
    p = ShowdownReplayParser(make_log(*HEADER), our_player="p1")
    p._handle_line("|switch|p2a: Aerodactyl|Aerodactyl, L50, M|100/100")
    p._handle_line("|switch|p2b: Palafin|Palafin, L50, M|100/100")
    aero, pala = p.active_slots["p2a"], p.active_slots["p2b"]

    p._handle_line("|swap|p2a: Aerodactyl|1|[from] move: Ally Switch")

    assert p.active_slots["p2a"] is pala   # b's occupant moved to a
    assert p.active_slots["p2b"] is aero   # a's occupant moved to b (position 1)
    assert p.active_slots["p2a"].slot == "a"
    assert p.active_slots["p2b"].slot == "b"


def _ally_switch_then_faint_log() -> str:
    """The orangeomar pattern: Farigiraf Ally-Switches to p2b, then the mon now
    at p2a (Rampardos) faints and a replacement (Sableye) switches into p2a."""
    return make_log(
        "|player|p1|alice|101|1500",
        "|player|p2|bob|102|1500",
        "|teamsize|p1|4",
        "|teamsize|p2|4",
        "|tier|[Gen 9 Champions] VGC 2026 Reg M-A",
        "|poke|p1|Pikachu, L50, M|",
        "|poke|p1|Eevee, L50, M|",
        "|poke|p2|Farigiraf, L50, F|",
        "|poke|p2|Rampardos, L50, F|",
        "|poke|p2|Sableye, L50, F|",
        "|poke|p2|Drampa, L50, F|",
        "|switch|p1a: Pikachu|Pikachu, L50, M|100/100",
        "|switch|p1b: Eevee|Eevee, L50, M|100/100",
        "|switch|p2a: Farigiraf|Farigiraf, L50, F|100/100",
        "|switch|p2b: Rampardos|Rampardos, L50, F|100/100",
        "|turn|1",
        "|move|p2a: Farigiraf|Ally Switch|p2a: Farigiraf",
        "|swap|p2a: Farigiraf|1|[from] move: Ally Switch",
        "|move|p1a: Pikachu|Wave Crash|p2a: Rampardos",
        "|-damage|p2a: Rampardos|0 fnt",
        "|faint|p2a: Rampardos",
        "|upkeep",
        "|switch|p2a: Sableye|Sableye, L50, F|100/100",
        "|turn|2",
        "|win|alice",
    )


def test_ally_switch_keeps_living_ally_active_through_replacement():
    """End-to-end Gap #6 sub-cause 1: after Ally Switch + a faint + a forced
    replacement, BOTH opponent slots are still active at the next turn — the
    replacement fills the fainted slot, NOT the still-living ally's slot.

    Before the |swap| fix the parser ignored the position shift, so the
    ``|switch|p2a: Sableye`` overwrote the living Farigiraf (which Ally Switch
    had moved to p2b), dropping it out of opp_active and wrongly into opp_bench.
    """
    r = ShowdownReplayParser(_ally_switch_then_faint_log(), our_player="p1").parse()
    # The replacement switches in at the end of turn 1, so turn 1's
    # state_after (== the board at the start of turn 2) reflects it.
    t1 = next(t for t in r["turns"] if t["turn"] == 1)
    snap = t1["state_after_actions"]["p1"]

    active = {d["species"] for d in snap["opp_active"].values()}
    assert active == {"Sableye", "Farigiraf"}, active   # both slots filled
    assert len(snap["opp_active"]) == 2

    # The living Farigiraf must NOT be wrongly benched as a seen-alive mon.
    bench_alive = {m["species"] for m in snap["opp_bench"]
                   if m.get("seen") and not m.get("is_fainted")}
    assert "Farigiraf" not in bench_alive
    # Rampardos correctly fainted (kept on the bench, KO-flagged).
    bench_fainted = {m["species"] for m in snap["opp_bench"]
                     if m.get("is_fainted")}
    assert "Rampardos" in bench_fainted


# ── Gap #6 residual: re-disguise Zoroark phantom double-count ────────────────

def test_perceived_roster_name_maps_mega_disguise_to_base():
    """A perceived MEGA disguise maps to its base roster name; non-mega formes
    keep their own name."""
    from v_dance.parser.vod_parser.replay_parser import _perceived_roster_name
    assert _perceived_roster_name("Charizard-Mega-Y") == "Charizard"
    assert _perceived_roster_name("Charizard-Mega-X") == "Charizard"
    assert _perceived_roster_name("Venusaur-Mega") == "Venusaur"
    assert _perceived_roster_name("Excadrill") == "Excadrill"        # no mega
    assert _perceived_roster_name("Arcanine-Hisui") == "Arcanine-Hisui"  # regional
    assert _perceived_roster_name("Yanmega") == "Yanmega"            # not a mega
    assert _perceived_roster_name("") == ""


def _redisguise_mega_log() -> str:
    """A Zoroark disguised as a MEGA-evolved teammate (Charizard-Mega-Y) while
    the REAL Charizard is alive on the bench — the spottedwoot t8 pattern."""
    return make_log(
        "|player|p1|alice|101|1500",
        "|player|p2|bob|102|1500",
        "|teamsize|p1|4",
        "|teamsize|p2|4",
        "|tier|[Gen 9 Champions] VGC 2026 Reg M-A",
        "|poke|p1|Pikachu, L50, M|",
        "|poke|p1|Eevee, L50, M|",
        "|poke|p2|Charizard, L50, M|",
        "|poke|p2|Zoroark-Hisui, L50, M|",
        "|poke|p2|Sylveon, L50, M|",
        "|poke|p2|Farigiraf, L50, M|",
        "|switch|p1a: Pikachu|Pikachu, L50, M|100/100",
        "|switch|p1b: Eevee|Eevee, L50, M|100/100",
        "|switch|p2a: Farigiraf|Farigiraf, L50, M|100/100",
        "|switch|p2b: Charizard|Charizard, L50, M|100/100",
        "|turn|1",
        "|detailschange|p2b: Charizard|Charizard-Mega-Y, L50, M",
        "|-mega|p2b: Charizard|Charizard|Charizardite Y",
        "|switch|p2b: Sylveon|Sylveon, L50, M|100/100",   # real Char-Mega-Y → bench
        "|turn|2",
        # Zoroark switches into p2a disguised as the mega'd Charizard:
        "|switch|p2a: Charizard|Charizard-Mega-Y, L50, M|100/100",
        "|turn|3",                                         # ← disguise-active snapshot
        "|replace|p2a: Zoroark|Zoroark-Hisui, L50, M",     # unmask (relabels the switch)
        "|turn|4",
        "|win|alice",
    )


def test_redisguise_mega_zoroark_no_phantom_double_count():
    """Gap #6 residual: when a Zoroark is disguised as a MEGA-evolved teammate
    ('Charizard-Mega-Y') while the real Charizard is alive on the bench, the
    opponent snapshot must NOT list that species BOTH active and on the bench.

    The disguise (perceived 'Charizard-Mega-Y') must consume the base 'Charizard'
    roster slot, masking the genuinely-benched same-species teammate from the
    observer's view — matching what a live bot perceives.  Before the
    _perceived_roster_name fix the mega forme name failed to match the base
    roster entry, so the real Charizard was double-listed.
    """
    r = ShowdownReplayParser(_redisguise_mega_log(), our_player="p1").parse()
    # turn 3's start-of-turn snapshot is the disguise-active frame.
    t3 = next(t for t in r["turns"] if t["turn"] == 3)
    snap = t3["state_before_actions"]["p1"]

    active = [d["species"] for d in snap["opp_active"].values()]
    bench = [m["species"] for m in snap["opp_bench"]]
    assert "Charizard-Mega-Y" in active           # perceived active disguise
    # No species appears both active and benched (the phantom signature):
    assert not (set(active) & set(bench)), (
        f"phantom double-count: active={active} bench={bench}")
    assert bench.count("Charizard-Mega-Y") == 0 and bench.count("Charizard") == 0

    # After the unmask (state_after of turn 3, post-|replace|), the real
    # Charizard is no longer masked and reappears on the bench.
    after = t3["state_after_actions"]["p1"]
    after_active = [d["species"] for d in after["opp_active"].values()]
    after_bench = [m["species"] for m in after["opp_bench"]]
    assert "Zoroark-Hisui" in after_active        # disguise resolved
    assert any("Charizard" in s for s in after_bench)   # real Charizard restored


# ── #6: post-faint forced-replacement opponent snapshot ──────────────────────
# opp_snapshot_current reconstructs the CURRENT (mid-turn / post-faint) opponent
# board the way a live bot must encode a forced-switch decision — distinct from
# the start-of-turn snapshot opp_snapshot_from_log_prefix returns.

def _post_faint_doubles_log() -> str:
    """A turn where an opponent (p2) mon faints AND our (p1) mon faints — the
    log ends at the forceSwitch point, BEFORE any replacement switch-in."""
    return make_log(
        "|player|p1|alice|101|1500",
        "|player|p2|bob|102|1500",
        "|teamsize|p1|4",
        "|teamsize|p2|4",
        "|tier|[Gen 9 Champions] VGC 2026 Reg M-A",
        "|poke|p1|Pikachu, L50, M|",
        "|poke|p1|Eevee, L50, M|",
        "|poke|p1|Snorlax, L50, M|",
        "|poke|p2|Farigiraf, L50, F|",
        "|poke|p2|Rampardos, L50, F|",
        "|poke|p2|Sableye, L50, F|",
        "|switch|p1a: Pikachu|Pikachu, L50, M|100/100",
        "|switch|p1b: Eevee|Eevee, L50, M|100/100",
        "|switch|p2a: Farigiraf|Farigiraf, L50, F|100/100",
        "|switch|p2b: Rampardos|Rampardos, L50, F|100/100",
        "|turn|1",
        "|move|p1a: Pikachu|Thunderbolt|p2b: Rampardos",
        "|-damage|p2b: Rampardos|0 fnt",
        "|faint|p2b: Rampardos",          # opponent mon faints mid-turn
        "|move|p2a: Farigiraf|Body Slam|p1a: Pikachu",
        "|-damage|p1a: Pikachu|0 fnt",
        "|faint|p1a: Pikachu",            # our mon faints → forceSwitch point
    )


def test_opp_snapshot_current_reflects_post_faint():
    """opp_snapshot_current must show the post-faint board (the fainted opponent
    mon dropped from opp_active), whereas opp_snapshot_from_log_prefix returns the
    stale start-of-turn board (both opponents still active)."""
    from v_dance.encoders.live_state_encoder import opp_snapshot_current, opp_snapshot_from_log_prefix

    log = _post_faint_doubles_log()
    start = opp_snapshot_from_log_prefix(log, "p1", 1)   # opp = p2, START of turn 1
    cur = opp_snapshot_current(log, "p1")                # CURRENT (post-faint)

    assert start is not None and cur is not None
    assert {d["species"] for d in start["opp_active"].values()} == {"Farigiraf", "Rampardos"}, \
        "start-of-turn snapshot should show both opponents active"
    assert {d["species"] for d in cur["opp_active"].values()} == {"Farigiraf"}, \
        "post-faint snapshot should drop the fainted Rampardos from active"
    fainted_bench = {m["species"] for m in cur["opp_bench"] if m.get("is_fainted")}
    assert "Rampardos" in fainted_bench, "Rampardos should be on the bench, KO-flagged"


def test_opp_snapshot_current_equals_prefix_at_turn_boundary():
    """At a turn boundary (no mid-turn faints yet) the CURRENT snapshot equals the
    START-of-turn snapshot — opp_snapshot_current is the general form of the
    prefix helper."""
    from v_dance.encoders.live_state_encoder import opp_snapshot_current, opp_snapshot_from_log_prefix

    log = _post_faint_doubles_log()
    # Truncate to the |turn|1 boundary (drop the turn-1 action lines).
    lines = log.split("\n")
    cut = lines.index("|turn|1") + 1
    prefix = "\n".join(lines[:cut])

    cur = opp_snapshot_current(prefix, "p1")
    pre = opp_snapshot_from_log_prefix(log, "p1", 1)
    assert cur is not None and pre is not None
    assert ({k: v["species"] for k, v in cur["opp_active"].items()}
            == {k: v["species"] for k, v in pre["opp_active"].items()})
    assert ([m["species"] for m in cur["opp_bench"]]
            == [m["species"] for m in pre["opp_bench"]])


def test_opp_snapshot_current_post_faint_real_replay():
    """Real-replay check: truncating the orangeomar log right after the opponent
    Rampardos faints (post Ally-Switch) yields a CURRENT opp snapshot with only
    Farigiraf active and Rampardos KO-flagged on the bench — the post-faint board
    a forced-replacement decision must encode."""
    from pathlib import Path
    from v_dance.encoders.live_state_encoder import opp_snapshot_current, opp_snapshot_from_log_prefix

    vods = Path(__file__).resolve().parents[1] / "data" / "vods"
    hit = next(vods.rglob("*kronomono-orangeomar*.html"), None)
    if hit is None:
        pytest.skip("orangeomar fixture not found")
    log = extract_log_from_html(hit.read_text(encoding="utf-8"))

    lines = log.split("\n")
    faint_idx = next(i for i, ln in enumerate(lines)
                     if ln.strip() == "|faint|p2a: Rampardos")
    prefix = "\n".join(lines[:faint_idx + 1])   # up to & incl. the opp faint

    cur = opp_snapshot_current(prefix, "p1")     # opp = p2
    assert cur is not None
    assert {d["species"] for d in cur["opp_active"].values()} == {"Farigiraf"}, \
        "only Farigiraf should be active after Rampardos faints"
    assert "Rampardos" in {m["species"] for m in cur["opp_bench"] if m.get("is_fainted")}

    # The stale start-of-turn-1 snapshot still shows both opponents active.
    start = opp_snapshot_from_log_prefix(log, "p1", 1)
    assert {d["species"] for d in start["opp_active"].values()} == {"Farigiraf", "Rampardos"}
