"""
test_bug_fixes.py
=================
Unit + integration tests for the Victory-Dance vod_parser bug fixes:

  Bug 1 — boosts reset on switch-out
  Bug 2 — indirect ([from]-tagged) damage not attributed to last move
  Bug 3 — curestatus recorded in the event log
  Bug 4 — terrain normalised to canonical tokens
  Bug 5 — target_slot is None (never "") for self/no-target moves
  Bug 6 — fainted mons kept in bench, identically on both sides
  Bug 7 — mega forms don't create phantom unseen roster stubs in opp_bench
  Schema — total_turns present on every transition
  Choice — 2+ distinct moves in one stint rule out Choice items
           (can_have_choice_item; belief fill drops Choice Scarf/Band/Specs)
  Formes — Pikalytics lookup falls back to the base forme for suffixed
           species (Vivillon-Garden → Vivillon)

Run with:  python -m pytest tests/ -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Make the package importable when running from the repo root or tests/
_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent, _HERE.parent.parent):
    sys.path.insert(0, str(_p))

from vod_parser.replay_parser import (
    ShowdownReplayParser,
    extract_log_from_html,
    extract_replay_id_from_html,
)
from vod_parser.transitions import parse_replay_for_preview, replay_to_transitions

from belief_state import BeliefState, is_choice_item, ui_fill_suggestions

SAMPLE_REPLAY_NAME = (
    "Gen9ChampionsVGC2026RegMA-2026-04-20-stevenhevgc-speedyturtle87.html"
)


def _find_sample_replay() -> Path | None:
    """
    Locate the sample replay regardless of where this test file lives.

    Search order:
      1. VD_SAMPLE_REPLAY environment variable (full path to the .html)
      2. Common locations relative to this file and the cwd
      3. Recursive search upward from this file (up to 3 levels)
    """
    env = os.environ.get("VD_SAMPLE_REPLAY")
    if env and Path(env).exists():
        return Path(env)

    bases = [_HERE, _HERE.parent, _HERE.parent.parent, Path.cwd()]
    subdirs = ["", "data", "tests/data", "data/vods", "data/vods/ladder_play",
               "data/vods/human_play", "data/vods/self_play"]
    for base in bases:
        for sub in subdirs:
            candidate = base / sub / SAMPLE_REPLAY_NAME
            if candidate.exists():
                return candidate

    # Last resort: walk upward and search recursively for the exact filename.
    for base in (_HERE, _HERE.parent, _HERE.parent.parent):
        try:
            hit = next(base.rglob(SAMPLE_REPLAY_NAME), None)
        except (OSError, PermissionError):
            hit = None
        if hit:
            return hit
    return None


SAMPLE_REPLAY = _find_sample_replay()

requires_sample_replay = pytest.mark.skipif(
    SAMPLE_REPLAY is None,
    reason=(
        f"Sample replay '{SAMPLE_REPLAY_NAME}' not found. Place it near the "
        "tests (e.g. in a data/ folder) or set the VD_SAMPLE_REPLAY env var "
        "to its full path."
    ),
)


# ---------------------------------------------------------------------------
# Helpers — build small synthetic Showdown logs
# ---------------------------------------------------------------------------

HEADER = """\
|player|p1|alice|101|1500
|player|p2|bob|102|1500
|gen|9
|tier|[Gen 9] Test Doubles
|poke|p1|Floette-Eternal, L50, F|
|poke|p1|Sneasler, L50, F|
|poke|p1|Incineroar, L50, F|
|poke|p1|Kingambit, L50, M|
|poke|p2|Aerodactyl, L50, M|
|poke|p2|Meganium, L50, M|
|poke|p2|Kingambit, L50, M|
|poke|p2|Rotom-Wash, L50|
|teamsize|p1|4
|teamsize|p2|4
|start
|switch|p1a: Floette|Floette-Eternal, L50, F|100/100
|switch|p1b: Sneasler|Sneasler, L50, F|100/100
|switch|p2a: Kingambit|Kingambit, L50, M|100/100
|switch|p2b: Aerodactyl|Aerodactyl, L50, M|100/100
"""


def parse_log(body: str, our_player: str = "p1") -> dict:
    """Parse HEADER + body and return the structured battle dict."""
    parser = ShowdownReplayParser(HEADER + body, our_player=our_player)
    return parser.parse()


def make_parser(body: str, our_player: str = "p1") -> ShowdownReplayParser:
    parser = ShowdownReplayParser(HEADER + body, our_player=our_player)
    parser.parse()
    return parser


def get_turn(result: dict, turn_number: int) -> dict:
    for t in result["turns"]:
        if t["turn"] == turn_number:
            return t
    raise AssertionError(f"turn {turn_number} not found in result")


# ---------------------------------------------------------------------------
# Bug 1 — boosts reset on switch-out
# ---------------------------------------------------------------------------

class TestBug1BoostsResetOnSwitch:
    BODY = """\
|turn|1
|move|p1a: Floette|Calm Mind|p1a: Floette
|-boost|p1a: Floette|spa|1
|-boost|p1a: Floette|spd|1
|upkeep
|turn|2
|switch|p1a: Incineroar|Incineroar, L50, F|100/100
|upkeep
|turn|3
|switch|p1a: Floette|Floette-Eternal, L50, F|100/100
|upkeep
|turn|4
|move|p1a: Floette|Protect|p1a: Floette
|upkeep
|win|alice
"""

    def test_boosts_visible_while_active(self):
        result = parse_log(self.BODY)
        # Turn 2's state_before is captured right after turn 1 resolved:
        # Floette is still in with +1/+1.
        state = get_turn(result, 2)["state_before_actions"]["p1"]
        floette = state["our_active"]["our_a"]
        assert floette["species"] == "Floette-Eternal"
        assert floette["boosts"] == {"spa": 1, "spd": 1}

    def test_boosts_cleared_after_switch_out(self):
        parser = make_parser(self.BODY)
        mon = parser.seen_mons["p1:Floette-Eternal"]
        assert mon.boosts == {}, (
            "Boosts must reset when a mon switches out; they currently "
            f"persisted as {mon.boosts}"
        )

    def test_boosts_clean_on_re_switch_in(self):
        result = parse_log(self.BODY)
        # Turn 4: Floette is back in after switching out in turn 2.
        state = get_turn(result, 4)["state_before_actions"]["p1"]
        floette = state["our_active"]["our_a"]
        assert floette["species"] == "Floette-Eternal"
        assert floette["boosts"] == {}, (
            "Re-switched-in mon must start with neutral boosts"
        )

    def test_benched_mon_snapshot_has_no_boosts(self):
        result = parse_log(self.BODY)
        # Turn 3's state_before: Floette is on the bench (Incineroar active).
        state = get_turn(result, 3)["state_before_actions"]["p1"]
        bench_species = {b["species"]: b for b in state["our_bench"]}
        assert "Floette-Eternal" in bench_species
        assert bench_species["Floette-Eternal"]["boosts"] == {}


# ---------------------------------------------------------------------------
# Bug 2 — indirect damage not attributed to last move
# ---------------------------------------------------------------------------

class TestBug2IndirectDamageAttribution:
    BODY = """\
|turn|1
|move|p1a: Floette|Moonblast|p2a: Kingambit
|-damage|p2a: Kingambit|60/100
|-status|p1a: Floette|brn
|-damage|p1a: Floette|88/100|[from] brn
|-damage|p2b: Aerodactyl|90/100|[from] item: Rocky Helmet|[of] p1b: Sneasler
|upkeep
|win|alice
"""

    @pytest.fixture()
    def damage_events(self):
        result = parse_log(self.BODY)
        return get_turn(result, 1)["damage_events"]

    def test_direct_damage_attributed_to_move(self, damage_events):
        direct = next(e for e in damage_events if e["slot"] == "p2a")
        assert direct["source_move"] == "Moonblast"
        assert direct["source_slot"] == "p1a"
        assert direct["source_species"] == "Floette"

    def test_burn_damage_not_attributed(self, damage_events):
        burn = next(e for e in damage_events if e["slot"] == "p1a")
        assert burn["hp_pct_delta"] == -12.0
        assert burn["source_move"] is None, (
            "[from] brn chip damage must not inherit the last move as source"
        )
        assert burn["source_slot"] is None
        assert burn["source_species"] is None

    def test_rocky_helmet_damage_not_attributed(self, damage_events):
        helmet = next(e for e in damage_events if e["slot"] == "p2b")
        assert helmet["source_move"] is None
        assert helmet["source_slot"] is None


# ---------------------------------------------------------------------------
# Bug 3 — curestatus appears in the event log
# ---------------------------------------------------------------------------

class TestBug3CurestatusEvent:
    BODY = """\
|turn|1
|-status|p1a: Floette|brn
|upkeep
|turn|2
|-curestatus|p1a: Floette|brn
|upkeep
|win|alice
"""

    def test_curestatus_event_recorded(self):
        result = parse_log(self.BODY)
        actions = get_turn(result, 2)["actions"]
        cures = [a for a in actions if a["event"] == "curestatus"]
        assert len(cures) == 1, "curestatus must be appended to the turn's actions"
        assert cures[0]["slot"] == "p1a"
        assert cures[0]["species"] == "Floette"
        assert cures[0]["status"] == "brn"

    def test_status_transition_explained_by_event_log(self):
        """status goes brn -> None between snapshots, and the event log says why."""
        result = parse_log(self.BODY)
        turn2 = get_turn(result, 2)
        before = turn2["state_before_actions"]["p1"]["our_active"]["our_a"]
        after = turn2["state_after_actions"]["p1"]["our_active"]["our_a"]
        assert before["status"] == "brn"
        assert after["status"] is None
        assert any(a["event"] == "curestatus" for a in turn2["actions"])


# ---------------------------------------------------------------------------
# Bug 4 — terrain normalised
# ---------------------------------------------------------------------------

class TestBug4TerrainNormalisation:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("move: Electric Terrain", "electric"),
            ("Electric Terrain", "electric"),
            ("move: Grassy Terrain", "grassy"),
            ("move: Misty Terrain", "misty"),
            ("move: Psychic Terrain", "psychic"),
            ("GRASSY TERRAIN", "grassy"),
            ("move: Trick Room", None),
            ("", None),
            (None, None),
        ],
    )
    def test_normalize_terrain(self, raw, expected):
        assert ShowdownReplayParser._normalize_terrain(raw) == expected

    def test_fieldstart_stores_normalised_token(self):
        body = """\
|turn|1
|move|p2b: Aerodactyl|Electric Terrain|p2b: Aerodactyl
|-fieldstart|move: Electric Terrain
|upkeep
|turn|2
|move|p1a: Floette|Protect|p1a: Floette
|upkeep
|win|alice
"""
        result = parse_log(body)
        state = get_turn(result, 2)["state_before_actions"]["p1"]
        assert state["field"]["terrain"] == "electric", (
            "terrain must be a canonical token, not the raw effect string"
        )

    def test_fieldend_clears_terrain(self):
        body = """\
|turn|1
|-fieldstart|move: Grassy Terrain
|upkeep
|turn|2
|-fieldend|move: Grassy Terrain
|upkeep
|turn|3
|move|p1a: Floette|Protect|p1a: Floette
|upkeep
|win|alice
"""
        result = parse_log(body)
        state = get_turn(result, 3)["state_before_actions"]["p1"]
        assert state["field"]["terrain"] is None


# ---------------------------------------------------------------------------
# Bug 5 — target_slot never ""
# ---------------------------------------------------------------------------

class TestBug5EmptyTargetSlot:
    def test_still_move_has_null_target(self):
        body = """\
|turn|1
|move|p1b: Kingambit|Sucker Punch||[still]
|-fail|p1b: Kingambit
|upkeep
|win|alice
"""
        result = parse_log(body)
        moves = [a for a in get_turn(result, 1)["actions"] if a["event"] == "move"]
        assert moves[0]["target_slot"] is None
        assert moves[0]["target_slot"] != ""

    def test_whitespace_target_ident_coerced_to_none(self):
        """A truthy-but-garbage target ident must not yield target_slot == ''."""
        parser = ShowdownReplayParser("", our_player="p1")
        parser._handle_move(["", "move", "p1a: Floette", "Moonblast", " "])
        move = parser._current_turn_actions[-1]
        assert move["target_slot"] is None

    def test_no_empty_string_targets_anywhere(self):
        body = """\
|turn|1
|move|p1a: Floette|Protect|p1a: Floette
|move|p1b: Sneasler|Fake Out|p2a: Kingambit
|-damage|p2a: Kingambit|85/100
|move|p2a: Kingambit|Sucker Punch||[still]
|-fail|p2a: Kingambit
|upkeep
|win|alice
"""
        result = parse_log(body)
        for action in get_turn(result, 1)["actions"]:
            if action["event"] == "move":
                assert action["target_slot"] != "", (
                    f"empty-string target_slot leaked: {action}"
                )


# ---------------------------------------------------------------------------
# Bug 6 — fainted mons kept in bench, identically on both sides
# ---------------------------------------------------------------------------

class TestBug6FaintedBenchConsistency:
    BODY = """\
|turn|1
|move|p2a: Kingambit|Kowtow Cleave|p1b: Sneasler
|-damage|p1b: Sneasler|0 fnt
|faint|p1b: Sneasler
|move|p1a: Floette|Moonblast|p2b: Aerodactyl
|-damage|p2b: Aerodactyl|0 fnt
|faint|p2b: Aerodactyl
|upkeep
|turn|2
|switch|p1b: Incineroar|Incineroar, L50, F|100/100
|switch|p2b: Meganium|Meganium, L50, M|100/100
|move|p1a: Floette|Protect|p1a: Floette
|upkeep
|win|alice
"""

    @pytest.fixture()
    def turn2_state(self):
        result = parse_log(self.BODY)
        return get_turn(result, 2)["state_after_actions"]

    def test_fainted_mon_kept_in_our_bench(self, turn2_state):
        bench = {b["species"]: b for b in turn2_state["p1"]["our_bench"]}
        assert "Sneasler" in bench, "fainted mon must stay visible in our_bench"
        assert bench["Sneasler"]["is_fainted"] is True
        assert bench["Sneasler"]["hp_pct"] == 0.0

    def test_fainted_mon_kept_in_opp_bench(self, turn2_state):
        bench = {b["species"]: b for b in turn2_state["p1"]["opp_bench"]}
        assert "Aerodactyl" in bench, "fainted mon must stay visible in opp_bench"
        assert bench["Aerodactyl"]["is_fainted"] is True
        assert bench["Aerodactyl"]["seen"] is True

    def test_rule_is_symmetric_across_perspectives(self, turn2_state):
        """p1's view of its fainted Sneasler == p2's view of that same mon."""
        p1_own = {b["species"]: b for b in turn2_state["p1"]["our_bench"]}
        p2_opp = {b["species"]: b for b in turn2_state["p2"]["opp_bench"]}
        assert "Sneasler" in p1_own and "Sneasler" in p2_opp
        for field in ("is_fainted", "hp_pct", "status"):
            assert p1_own["Sneasler"][field] == p2_opp["Sneasler"][field]

    def test_fainted_mons_never_in_active(self, turn2_state):
        for side in ("p1", "p2"):
            for group in ("our_active", "opp_active"):
                for mon in turn2_state[side][group].values():
                    assert mon["is_fainted"] is False

    def test_no_duplicate_representation(self, turn2_state):
        """A mon must appear in active XOR bench, never both."""
        for side in ("p1", "p2"):
            state = turn2_state[side]
            active = {m["species"] for m in state["our_active"].values()}
            bench = {m["species"] for m in state["our_bench"]}
            assert not active & bench, f"{side}: {active & bench} in both lists"


# ---------------------------------------------------------------------------
# Bug 7 — mega forms reconcile against the roster (no phantom stubs)
# ---------------------------------------------------------------------------

class TestBug7MegaRosterReconciliation:
    BODY = """\
|turn|1
|detailschange|p2b: Aerodactyl|Aerodactyl-Mega, L50, M
|-mega|p2b: Aerodactyl|Aerodactyl|Aerodactylite
|move|p1a: Floette|Protect|p1a: Floette
|upkeep
|turn|2
|move|p1a: Floette|Protect|p1a: Floette
|upkeep
|win|alice
"""

    def test_no_phantom_unseen_stub_for_mega(self):
        result = parse_log(self.BODY)
        state = get_turn(result, 2)["state_before_actions"]["p1"]
        active_species = {m["species"] for m in state["opp_active"].values()}
        assert "Aerodactyl-Mega" in active_species
        bench_species = [b["species"] for b in state["opp_bench"]]
        assert "Aerodactyl" not in bench_species, (
            "mega-evolved active mon must not also appear as an unseen "
            "roster stub in opp_bench"
        )

    def test_bench_size_matches_remaining_roster(self):
        result = parse_log(self.BODY)
        state = get_turn(result, 2)["state_before_actions"]["p1"]
        # p2 roster has 4 mons, 2 are active -> exactly 2 on the bench.
        assert len(state["opp_bench"]) == 2


# ---------------------------------------------------------------------------
# Bug 9 — abilities revealed via inline [from] ability: tags
# ---------------------------------------------------------------------------

class TestBug9InlineAbilityReveal:
    def test_weather_setter_ability_recorded_from_of_tag(self):
        """Drizzle/Drought never emit a |-ability| line — the reveal rides on
        the |-weather| line as [from] ability: ...|[of] <holder>."""
        body = """\
|-weather|RainDance|[from] ability: Drizzle|[of] p2a: Kingambit
|turn|1
|move|p1a: Floette|Protect|p1a: Floette
|upkeep
|win|alice
"""
        result = parse_log(body)
        info = result["revealed_info"]["p2:Kingambit"]
        assert info["known_ability"] == "Drizzle"
        # Weather itself must still be applied.
        state = get_turn(result, 1)["state_before_actions"]["p1"]
        assert state["field"]["weather"] == "RainDance"

    def test_heal_from_ability_without_of_uses_line_subject(self):
        body = """\
|turn|1
|-heal|p1a: Floette|100/100|[from] ability: Water Absorb
|upkeep
|win|alice
"""
        result = parse_log(body)
        info = result["revealed_info"]["p1:Floette-Eternal"]
        assert info["known_ability"] == "Water Absorb"

    def test_inline_reveal_emits_ability_revealed_event(self):
        body = """\
|turn|1
|-weather|SunnyDay|[from] ability: Drought|[of] p2b: Aerodactyl
|upkeep
|win|alice
"""
        result = parse_log(body)
        events = [a for a in get_turn(result, 1)["actions"]
                  if a["event"] == "ability_revealed"]
        assert events == [{
            "event": "ability_revealed",
            "slot": "p2b",
            "species": "Aerodactyl",
            "ability": "Drought",
            "is_mega_ability": False,
        }]

    def test_non_ability_from_tags_ignored(self):
        body = """\
|turn|1
|-damage|p1a: Floette|88/100|[from] item: Rocky Helmet|[of] p2a: Kingambit
|upkeep
|win|alice
"""
        result = parse_log(body)
        assert result["revealed_info"]["p2:Kingambit"]["known_ability"] is None
        assert result["revealed_info"]["p1:Floette-Eternal"]["known_ability"] is None

    def test_dash_ability_line_still_works(self):
        body = """\
|turn|1
|-ability|p2a: Kingambit|Defiant|boost
|-boost|p2a: Kingambit|atk|2
|upkeep
|win|alice
"""
        result = parse_log(body)
        assert result["revealed_info"]["p2:Kingambit"]["known_ability"] == "Defiant"


# ---------------------------------------------------------------------------
# Schema — total_turns
# ---------------------------------------------------------------------------

@requires_sample_replay
class TestTotalTurnsSchema:
    def test_total_turns_on_every_transition(self):
        transitions = replay_to_transitions(SAMPLE_REPLAY)
        assert transitions, "sample replay should produce transitions"
        n_turns = max(t["turn"] for t in transitions)
        for t in transitions:
            assert t["total_turns"] == n_turns
            assert 1 <= t["turn"] <= t["total_turns"]


# ---------------------------------------------------------------------------
# Integration — full pipeline on the real example VOD
# ---------------------------------------------------------------------------

@requires_sample_replay
class TestIntegrationSampleReplay:
    @pytest.fixture(scope="class")
    def battle(self):
        html = SAMPLE_REPLAY.read_text(encoding="utf-8")
        parser = ShowdownReplayParser(extract_log_from_html(html), our_player="p1")
        return parser.parse()

    def test_metadata(self, battle):
        assert battle["winner"] == "steven he vgc"
        assert battle["players"]["p1"]["username"] == "steven he vgc"
        assert battle["players"]["p2"]["username"] == "speedyturtle87"
        assert battle["players"]["p1"]["rating_delta"] == 13
        assert battle["players"]["p2"]["rating_delta"] == -13
        assert len(battle["turns"]) == 8

    def test_bug1_floette_boosts_reset_in_real_replay(self, battle):
        """Floette gets +1/+1 Calm Mind in turn 3; boosts must survive while it
        stays in, and the parser must not let them leak onto other snapshots."""
        turn4 = get_turn(battle, 4)
        floette = turn4["state_before_actions"]["p1"]["our_active"]["our_a"]
        assert floette["boosts"] == {"spa": 1, "spd": 1}
        # Kingambit switched in fresh in turn 2 — never boosted.
        gambit = turn4["state_before_actions"]["p1"]["our_active"]["our_b"]
        assert gambit["boosts"] == {}

    def test_bug2_drain_heal_and_no_misattribution(self, battle):
        """Turn 7: Draining Kiss damages Kingambit then heals Floette with a
        [from] drain tag — the heal must carry no move attribution."""
        events = get_turn(battle, 7)["damage_events"]
        heal = next(e for e in events if e["event"] == "heal")
        assert heal["slot"] == "p1a"
        assert heal["source_move"] is None
        kiss = next(
            e for e in events
            if e["event"] == "damage" and e["slot"] == "p2a"
        )
        assert kiss["source_move"] == "Draining Kiss"
        assert kiss["source_slot"] == "p1a"

    def test_bug6_bug7_turn8_opp_bench_clean(self, battle):
        """Turn 8: Meganium mega-evolves. The bench must show the two fainted
        mons + unseen mons, with no phantom Meganium stub."""
        state = get_turn(battle, 8)["state_after_actions"]["p1"]
        active = {m["species"] for m in state["opp_active"].values()}
        assert active == {"Kingambit", "Meganium-Mega"}

        bench = {b["species"]: b for b in state["opp_bench"]}
        assert "Meganium" not in bench, "phantom roster stub for active mega"
        assert bench["Aerodactyl"]["is_fainted"] is True
        assert bench["Sneasler"]["is_fainted"] is True
        assert bench["Diggersby"]["seen"] is False
        assert bench["Rotom-Wash"]["seen"] is False

    def test_bug5_no_empty_targets_in_real_replay(self, battle):
        for turn in battle["turns"]:
            for action in turn["actions"]:
                if action["event"] == "move":
                    assert action["target_slot"] != ""

    def test_transitions_pipeline(self):
        transitions = replay_to_transitions(SAMPLE_REPLAY)
        # 8 turns x 2 perspectives
        assert len(transitions) == 16
        final = [t for t in transitions if t["turn"] == 8]
        for t in final:
            expected = 1 if t["perspective"] == "p1" else -1
            assert t["reward"]["win"] == expected
        # Slot notation fully normalised in emitted transitions
        valid = {"our_a", "our_b", "opp_a", "opp_b", None}
        for t in transitions:
            for a in t["our_actions"] + t["opp_actions_actual"]:
                assert a["slot"] in valid
                assert a.get("target_slot") in valid

    def test_replay_id_extracted(self):
        html = SAMPLE_REPLAY.read_text(encoding="utf-8")
        rid = extract_replay_id_from_html(html)
        assert rid  # present in the sample file


# ---------------------------------------------------------------------------
# Choice-item constraint — 2+ distinct moves in one stint rule out Choice items
# ---------------------------------------------------------------------------

class TestChoiceItemConstraint:
    KEY = "p1:Floette-Eternal"

    def _flag(self, body: str) -> bool:
        return parse_log(body)["revealed_info"][self.KEY]["can_have_choice_item"]

    def test_two_moves_one_stint_rules_out_choice(self):
        assert self._flag("""\
|turn|1
|move|p1a: Floette|Moonblast|p2a: Kingambit
|upkeep
|turn|2
|move|p1a: Floette|Protect|p1a: Floette
|upkeep
|win|alice
""") is False

    def test_same_move_repeatedly_stays_possible(self):
        assert self._flag("""\
|turn|1
|move|p1a: Floette|Moonblast|p2a: Kingambit
|upkeep
|turn|2
|move|p1a: Floette|Moonblast|p2a: Kingambit
|upkeep
|win|alice
""") is True

    def test_different_moves_across_stints_stays_possible(self):
        # Switching out releases a Choice lock — variety across separate
        # stays on the field proves nothing.
        assert self._flag("""\
|turn|1
|move|p1a: Floette|Moonblast|p2a: Kingambit
|upkeep
|turn|2
|switch|p1a: Incineroar|Incineroar, L50, F|100/100
|upkeep
|turn|3
|switch|p1a: Floette|Floette-Eternal, L50, F|100/100
|upkeep
|turn|4
|move|p1a: Floette|Protect|p1a: Floette
|upkeep
|win|alice
""") is True

    def test_called_move_does_not_count(self):
        # [from]-tagged moves (Sleep Talk, Dancer, Instruct, …) were never
        # selected by the player.
        assert self._flag("""\
|turn|1
|move|p1a: Floette|Moonblast|p2a: Kingambit
|upkeep
|turn|2
|move|p1a: Floette|Dazzling Gleam|p2a: Kingambit|[from]move: Sleep Talk
|upkeep
|win|alice
""") is True

    def test_struggle_does_not_count(self):
        # A choice-locked mon Struggles once its locked move has no PP.
        assert self._flag("""\
|turn|1
|move|p1a: Floette|Moonblast|p2a: Kingambit
|upkeep
|turn|2
|move|p1a: Floette|Struggle|p2a: Kingambit
|upkeep
|win|alice
""") is True

    def test_item_loss_resets_the_stint(self):
        # Moves used after the item changed hands prove nothing about the
        # item the mon brought.
        assert self._flag("""\
|turn|1
|move|p1a: Floette|Moonblast|p2a: Kingambit
|-enditem|p1a: Floette|Focus Sash
|upkeep
|turn|2
|move|p1a: Floette|Protect|p1a: Floette
|upkeep
|win|alice
""") is True

    def test_violation_before_item_loss_is_permanent(self):
        assert self._flag("""\
|turn|1
|move|p1a: Floette|Moonblast|p2a: Kingambit
|upkeep
|turn|2
|move|p1a: Floette|Protect|p1a: Floette
|-enditem|p1a: Floette|Focus Sash
|upkeep
|win|alice
""") is False


# ---------------------------------------------------------------------------
# Belief side — choice-item filtering + forme-suffix fallback
# ---------------------------------------------------------------------------

import json  # noqa: E402

_MINI_PIKA = {
    "pokemon": {
        "Basculegion": {
            "usage_pct": 42.0,
            "moves": [
                {"name": "Wave Crash", "pct": 90.0},
                {"name": "Flip Turn", "pct": 80.0},
                {"name": "Aqua Jet", "pct": 70.0},
                {"name": "Last Respects", "pct": 60.0},
                {"name": "Protect", "pct": 50.0},
            ],
            "items": [
                {"name": "Choice Scarf", "pct": 55.0},
                {"name": "Choice Band", "pct": 25.0},
                {"name": "Mystic Water", "pct": 10.0},
            ],
            "abilities": [{"name": "Adaptability", "pct": 95.0}],
            "spreads": [
                {"nature": "Jolly", "evs": [0, 32, 0, 0, 0, 32], "pct": 40.0},
            ],
        },
        "Floette-Eternal": {
            "usage_pct": 30.0,
            "moves": [
                {"name": "Moonblast", "pct": 95.0},
                {"name": "Light of Ruin", "pct": 85.0},
                {"name": "Dazzling Gleam", "pct": 70.0},
                {"name": "Protect", "pct": 60.0},
            ],
            "items": [
                {"name": "Choice Scarf", "pct": 60.0},
                {"name": "Fairy Feather", "pct": 20.0},
            ],
            "abilities": [{"name": "Flower Veil", "pct": 99.0}],
            "spreads": [
                {"nature": "Timid", "evs": [0, 0, 0, 32, 0, 32], "pct": 50.0},
            ],
        },
        "Vivillon": {
            "usage_pct": 5.0,
            "moves": [
                {"name": "Hurricane", "pct": 95.0},
                {"name": "Sleep Powder", "pct": 90.0},
                {"name": "Rage Powder", "pct": 80.0},
                {"name": "Protect", "pct": 70.0},
                {"name": "Tailwind", "pct": 30.0},
            ],
            "items": [{"name": "Focus Sash", "pct": 80.0}],
            "abilities": [{"name": "Compound Eyes", "pct": 90.0}],
            "spreads": [
                {"nature": "Timid", "evs": [0, 0, 0, 32, 0, 32], "pct": 60.0},
            ],
        },
    }
}


@pytest.fixture()
def mini_belief(tmp_path):
    p = tmp_path / "pika.json"
    p.write_text(json.dumps(_MINI_PIKA), encoding="utf-8")
    return BeliefState(p)


class TestChoiceItemBeliefFilter:
    def test_is_choice_item_family(self):
        assert is_choice_item("Choice Scarf")
        assert is_choice_item("Choice Band")
        assert is_choice_item("Choice Specs")   # future-proof: not in format yet
        assert not is_choice_item("Leftovers")
        assert not is_choice_item(None)
        assert not is_choice_item("")

    def test_choice_items_kept_by_default(self, mini_belief):
        block = mini_belief.belief_block("Basculegion")
        assert block["items"][0]["name"] == "Choice Scarf"

    def test_choice_items_dropped_when_impossible(self, mini_belief):
        block = mini_belief.belief_block("Basculegion", can_have_choice_item=False)
        names = [i["name"] for i in block["items"]]
        assert names == ["Mystic Water"]

    def test_revealed_choice_item_beats_constraint(self, mini_belief):
        # Replay reveal (e.g. via Trick) is ground truth.
        block = mini_belief.belief_block(
            "Basculegion", revealed_item="Choice Scarf", can_have_choice_item=False
        )
        assert block["items"] == [{"name": "Choice Scarf", "p": 1.0, "revealed": True}]

    def test_mega_stone_reveal_collapses_items(self, mini_belief):
        # A mon that mega evolved on screen is guaranteed to hold its stone —
        # the usage distribution (which never lists stones) must not be used.
        block = mini_belief.belief_block(
            "Floette-Eternal", revealed_item="mega stone", can_have_choice_item=False
        )
        assert block["items"] == [{"name": "mega stone", "p": 1.0, "revealed": True}]

    def test_no_item_suggested_for_megad_mon(self, mini_belief):
        players = {"our_side": "p1", "p1": {"roster": []},
                   "p2": {"roster": ["Floette-Eternal"]}}
        revealed = {"p2:Floette-Eternal": {
            "revealed_moves": ["Moonblast"],
            "known_item": "mega stone",
            "is_mega": True,
            "can_have_choice_item": False,
        }}
        res = ui_fill_suggestions(mini_belief, "B", players, revealed)
        s = res["suggestions"]["p2:Floette-Eternal"]
        # Item is revealed (the stone) → no suggestion, the UI's VOD prefill
        # ("mega stone") must not be overwritten by a belief guess.
        assert s["item"] is None
        assert s["meta"]["item_p"] is None

    def test_parser_to_suggestions_pipeline(self, mini_belief):
        # Floette uses two different moves in one stint → the belief fill
        # must not suggest its (top) Choice Scarf.
        result = parse_log("""\
|turn|1
|move|p1a: Floette|Moonblast|p2a: Kingambit
|upkeep
|turn|2
|move|p1a: Floette|Protect|p1a: Floette
|upkeep
|win|alice
""")
        res = ui_fill_suggestions(
            mini_belief, "B", result["players"], result["revealed_info"]
        )
        s = res["suggestions"]["p1:Floette-Eternal"]
        assert s["item"] == "Fairy Feather"


class TestFormeSuffixFallback:
    def test_resolves_to_base_forme(self, mini_belief):
        assert mini_belief._resolve("Vivillon-Garden") == "Vivillon"
        assert mini_belief._resolve("Vivillon") == "Vivillon"

    def test_exact_entries_still_win(self, mini_belief):
        # Floette-Eternal has its own entry — must NOT fall back to "Floette".
        assert mini_belief._resolve("Floette-Eternal") == "Floette-Eternal"

    def test_unknown_species_still_none(self, mini_belief):
        assert mini_belief._resolve("Missingno-Garden") is None

    def test_belief_block_for_suffixed_forme(self, mini_belief):
        block = mini_belief.belief_block(
            "Vivillon-Garden",
            revealed_moves=["Hurricane", "Protect", "Sleep Powder"],
        )
        assert block is not None
        assert block["species_key"] == "Vivillon"
        # 3 revealed moves → exactly one predicted move for the last slot
        assert [m["name"] for m in block["moves_predicted"]] == ["Rage Powder"]


# ---------------------------------------------------------------------------
# Integration — belief fill against the repo's real data + test VODs
# ---------------------------------------------------------------------------

_DATA_DIR = _HERE.parents[1]            # data/scripts/vod_parser → data/
_PIKA_PATH = _DATA_DIR / "pikalytics_regma.json"
_VODS_DIR = _DATA_DIR / "vods"

requires_real_data = pytest.mark.skipif(
    not _PIKA_PATH.exists(),
    reason="data/pikalytics_regma.json not found",
)


@requires_real_data
class TestBeliefFillIntegration:
    def _fill(self, vod_path: Path):
        belief = BeliefState(_PIKA_PATH)
        preview = parse_replay_for_preview(
            vod_path.read_text(encoding="utf-8"), belief, None
        )
        return preview, ui_fill_suggestions(
            belief, "B", preview["players"], preview["revealed_info"]
        )

    def test_forme_suffixed_species_get_filled(self):
        vod = _VODS_DIR / "Gen9ChampionsVGC2026RegMA-2026-06-02-fuzis-megagenger2023.html"
        if not vod.exists():
            pytest.skip(f"{vod.name} not found")
        _, res = self._fill(vod)
        assert not res["skipped"], f"unexpected skips: {res['skipped']}"
        viv = res["suggestions"]["p1:Vivillon-Garden"]
        # 3 moves confirmed in the VOD → slots 1-3 left for the VOD prefill,
        # slot 4 belief-filled (the original Kleavor-style incident).
        assert viv["moves"][:3] == ["", "", ""]
        assert viv["moves"][3], "4th move slot must be belief-filled"

    @requires_sample_replay
    def test_choice_constraint_end_to_end(self):
        # Floette-Eternal uses several different moves in one stay on the
        # field in the sample replay → the constraint must be recorded.
        # Kingambit does the same and never megas → its item suggestion must
        # not be a Choice item.
        preview, res = self._fill(SAMPLE_REPLAY)
        assert preview["revealed_info"]["p1:Floette-Eternal"]["can_have_choice_item"] is False
        assert preview["revealed_info"]["p1:Kingambit"]["can_have_choice_item"] is False
        item = res["suggestions"]["p1:Kingambit"]["item"]
        assert item and not is_choice_item(item)

    @requires_sample_replay
    def test_megad_mon_keeps_mega_stone(self):
        # Floette-Eternal mega evolves in the sample replay (|-mega| →
        # Floettite) — its item is certain, so belief must not suggest one
        # (the UI keeps its "mega stone" VOD prefill).
        preview, res = self._fill(SAMPLE_REPLAY)
        rev = preview["revealed_info"]["p1:Floette-Eternal"]
        assert rev["is_mega"] is True
        assert rev["known_item"] == "mega stone"
        assert res["suggestions"]["p1:Floette-Eternal"]["item"] is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
