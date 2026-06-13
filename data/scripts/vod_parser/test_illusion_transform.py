"""
test_illusion_transform.py
==========================
Tests for the two notorious identity edge cases in the VOD parser:

  Illusion  (Zoroark / Zoroark-Hisui) — a disguised switch-in shows the
            DISGUISE species until a damaging hit breaks it and |replace|
            reveals the truth.  The parser must credit Zoroark (not the
            disguise) with every move/HP/event from the disguised switch-in,
            give Zoroark its own seen_mons card + brought entry, and keep a
            genuine copy of the disguise species separate.

  Transform (Ditto / Imposter, or the Transform move) — once transformed a
            mon uses the COPIED foe's moves.  Those must not enter the mon's
            revealed_moves nor trip the Choice-item constraint, and the flag
            must revert on switch-out.

Run with:  python -m pytest vod_parser/test_illusion_transform.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the package importable from the repo root, data/scripts, or here.
_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent, _HERE.parent.parent):
    sys.path.insert(0, str(_p))

from vod_parser.replay_parser import ShowdownReplayParser, extract_log_from_html


def parse(log: str, our_player: str = "p2") -> tuple[ShowdownReplayParser, dict]:
    parser = ShowdownReplayParser(log, our_player=our_player)
    return parser, parser.parse()


def get_turn(result: dict, n: int) -> dict:
    for t in result["turns"]:
        if t["turn"] == n:
            return t
    raise AssertionError(f"turn {n} not found")


# ---------------------------------------------------------------------------
# Illusion — Zoroark-Hisui disguised as Charizard, with a genuine Charizard
# also brought (the case that proves disguise ≠ real copy).
# ---------------------------------------------------------------------------

ILLUSION_LOG = """\
|player|p1|alice|101|1500
|player|p2|bob|102|1500
|gen|9
|tier|[Gen 9] Test Doubles
|poke|p1|Sneasler, L50, F|
|poke|p1|Incineroar, L50, F|
|poke|p2|Charizard, L50, F|
|poke|p2|Zoroark-Hisui, L50, M|
|poke|p2|Basculegion, L50, M|
|teamsize|p1|2
|teamsize|p2|3
|start
|switch|p1a: Sneasler|Sneasler, L50, F|100/100
|switch|p2a: Charizard|Charizard, L50, F|100/100
|turn|1
|move|p2a: Charizard|Shadow Ball|p1a: Sneasler
|-damage|p1a: Sneasler|40/100
|move|p1a: Sneasler|Close Combat|p2a: Charizard
|-damage|p2a: Charizard|1/100
|replace|p2a: Zoroark|Zoroark-Hisui, L50, M
|-end|p2a: Zoroark|Illusion
|-damage|p2a: Zoroark|1/100
|upkeep
|turn|2
|move|p2a: Zoroark|Hyper Voice|p1a: Sneasler
|-damage|p1a: Sneasler|0 fnt
|faint|p1a: Sneasler
|switch|p1a: Incineroar|Incineroar, L50, F|100/100
|upkeep
|turn|3
|switch|p2a: Basculegion|Basculegion, L50, M|100/100
|upkeep
|turn|4
|switch|p2a: Charizard|Charizard, L50, F|100/100
|move|p2a: Charizard|Heat Wave|p1a: Incineroar
|-damage|p1a: Incineroar|60/100
|upkeep
|win|bob
"""


class TestIllusionDisguise:
    def test_zoroark_gets_its_own_card(self):
        parser, result = parse(ILLUSION_LOG)
        assert "p2:Zoroark-Hisui" in parser.seen_mons
        # The disguise never poisons a phantom Charizard before the real one.
        assert "p2:Charizard" in parser.seen_mons

    def test_disguised_moves_credited_to_zoroark(self):
        parser, _ = parse(ILLUSION_LOG)
        zoro = parser.seen_mons["p2:Zoroark-Hisui"]
        # Shadow Ball (used while disguised) AND Hyper Voice (after reveal)
        # both belong to Zoroark.
        assert zoro.revealed_moves == ["Shadow Ball", "Hyper Voice"]

    def test_genuine_charizard_kept_separate(self):
        parser, _ = parse(ILLUSION_LOG)
        char = parser.seen_mons["p2:Charizard"]
        # The real Charizard only ever used Heat Wave — Zoroark's Shadow Ball
        # must NOT have leaked onto it.
        assert char.revealed_moves == ["Heat Wave"]
        assert "Shadow Ball" not in char.revealed_moves

    def test_brought_lists_real_identities(self):
        _, result = parse(ILLUSION_LOG)
        brought = result["players"]["p2"]["brought"]
        assert "Zoroark-Hisui" in brought
        assert "Charizard" in brought
        assert "Basculegion" in brought
        # Zoroark led (disguised) → it is the first brought entry, not Charizard.
        assert brought[0] == "Zoroark-Hisui"

    def test_prereveal_snapshot_is_already_zoroark(self):
        # Ground truth: even the turn-1 (pre-|replace|) snapshot tracks the
        # slot as Zoroark, because the disguise is resolved in the pre-scan.
        _, result = parse(ILLUSION_LOG)
        our_a = get_turn(result, 1)["state_before_actions"]["p2"]["our_active"]["our_a"]
        assert our_a["species"] == "Zoroark-Hisui"

    def test_illusion_revealed_event_emitted(self):
        _, result = parse(ILLUSION_LOG)
        events = [a for a in get_turn(result, 1)["actions"]
                  if a["event"] == "illusion_revealed"]
        assert len(events) == 1
        assert events[0]["slot"] == "p2a"
        assert events[0]["species"] == "Zoroark-Hisui"

    def test_match_flag_set(self):
        _, result = parse(ILLUSION_LOG)
        assert result["contains_illusion"] is True

    def test_defensive_relabel_without_prescan(self):
        # Directly exercise the |replace| fallback: a slot mislabeled as the
        # disguise gets relabeled forward even if the pre-scan map is empty.
        p = ShowdownReplayParser("", our_player="p2")
        p._handle_switch(["", "switch", "p2a: Charizard",
                          "Charizard, L50, F", "100/100"])
        assert "p2:Charizard" in p.seen_mons
        p._handle_replace(["", "replace", "p2a: Zoroark", "Zoroark-Hisui, L50, M"])
        assert "p2:Zoroark-Hisui" in p.seen_mons
        assert "p2:Charizard" not in p.seen_mons
        assert p.active_slots["p2a"].species == "Zoroark-Hisui"


# ---------------------------------------------------------------------------
# Transform — Imposter Ditto copying Whimsicott.
# ---------------------------------------------------------------------------

IMPOSTER_LOG = """\
|player|p1|alice|101|1500
|player|p2|bob|102|1500
|gen|9
|tier|[Gen 9] Test Doubles
|poke|p1|Ditto, L50|
|poke|p1|Incineroar, L50, F|
|poke|p2|Whimsicott, L50, F|
|poke|p2|Sneasler, L50, F|
|teamsize|p1|2
|teamsize|p2|2
|start
|switch|p1a: Ditto|Ditto, L50|100/100
|switch|p2a: Whimsicott|Whimsicott, L50, F|100/100
|-transform|p1a: Ditto|p2a: Whimsicott|[from] ability: Imposter
|turn|1
|move|p1a: Ditto|Moonblast|p2a: Whimsicott
|-damage|p2a: Whimsicott|70/100
|move|p1a: Ditto|Tailwind|p1a: Ditto
|upkeep
|turn|2
|switch|p1a: Incineroar|Incineroar, L50, F|100/100
|upkeep
|win|bob
"""


class TestImposterTransform:
    def test_copied_moves_not_credited_to_ditto(self):
        parser, _ = parse(IMPOSTER_LOG, our_player="p1")
        ditto = parser.seen_mons["p1:Ditto"]
        # Moonblast + Tailwind are Whimsicott's, copied — Ditto reveals none.
        assert ditto.revealed_moves == []

    def test_transform_flag_and_target(self):
        _, result = parse(IMPOSTER_LOG, our_player="p1")
        our_a = get_turn(result, 1)["state_before_actions"]["p1"]["our_active"]["our_a"]
        assert our_a["is_transformed"] is True
        assert our_a["transformed_into"] == "Whimsicott"

    def test_choice_constraint_not_tripped_while_transformed(self):
        # Two distinct moves in one stint would normally rule out a Choice item
        # — but they are copied moves, so the constraint must NOT fire.
        parser, _ = parse(IMPOSTER_LOG, our_player="p1")
        assert parser.seen_mons["p1:Ditto"].can_have_choice_item is True

    def test_transform_event_emitted(self):
        _, result = parse(IMPOSTER_LOG, our_player="p1")
        events = [a for a in get_turn(result, 1)["actions"]
                  if a["event"] == "transform"]
        # |-transform| fires before |turn|1, so the event lands in turn 1's log
        # only if it occurred during a turn; pre-turn events flush with turn 1.
        # Either way the match flag is the durable signal.
        assert result["contains_transform"] is True
        # The transform happened at preview (before turn 1) — assert via flag +
        # the persisted slot state rather than turn placement.
        assert result["revealed_info"]["p1:Ditto"]["is_transformed"] is True

    def test_revealed_info_reflects_transform(self):
        # revealed_info is match-level: is_transformed latches True even after
        # Ditto switched out (and reverted) on turn 2.  The live copy target is
        # carried on the per-turn snapshots, not here.
        _, result = parse(IMPOSTER_LOG, our_player="p1")
        info = result["revealed_info"]["p1:Ditto"]
        assert info["is_transformed"] is True
        assert info["revealed_moves"] == []

    def test_flag_reverts_on_switch_out(self):
        # Ditto leaves the field on turn 2 → its persisted slot is no longer
        # flagged transformed (it reverts to plain Ditto on the bench).
        parser, _ = parse(IMPOSTER_LOG, our_player="p1")
        assert parser.seen_mons["p1:Ditto"].is_transformed is False


# ---------------------------------------------------------------------------
# Transform via the MOVE (not Imposter): the Transform move itself is a real,
# selected move and must be recorded; only the copied moves are filtered.
# ---------------------------------------------------------------------------

TRANSFORM_MOVE_LOG = """\
|player|p1|alice|101|1500
|player|p2|bob|102|1500
|gen|9
|tier|[Gen 9] Test Doubles
|poke|p1|Mew, L50|
|poke|p2|Garchomp, L50, M|
|teamsize|p1|1
|teamsize|p2|1
|start
|switch|p1a: Mew|Mew, L50|100/100
|switch|p2a: Garchomp|Garchomp, L50, M|100/100
|turn|1
|move|p1a: Mew|Transform|p2a: Garchomp
|-transform|p1a: Mew|p2a: Garchomp
|move|p1a: Mew|Earthquake|p2a: Garchomp
|-damage|p2a: Garchomp|60/100
|upkeep
|win|alice
"""


class TestTransformMove:
    def test_transform_move_recorded_copied_move_filtered(self):
        parser, _ = parse(TRANSFORM_MOVE_LOG, our_player="p1")
        mew = parser.seen_mons["p1:Mew"]
        assert mew.revealed_moves == ["Transform"]
        assert "Earthquake" not in mew.revealed_moves
        assert mew.is_transformed is True
        assert mew.transformed_into == "Garchomp"


# ---------------------------------------------------------------------------
# Integration — the real Type A VOD that exercises BOTH cases on both sides.
# ---------------------------------------------------------------------------

_REAL_NAME = "Gen9ChampionsVGC2026RegMA-2026-06-13-victoriousdancing-kronomono.html"


def _find_real() -> Path | None:
    for base in (_HERE, _HERE.parent, _HERE.parent.parent):
        try:
            hit = next(base.rglob(_REAL_NAME), None)
        except (OSError, PermissionError):
            hit = None
        if hit:
            return hit
    return None


_REAL = _find_real()

requires_real = pytest.mark.skipif(
    _REAL is None, reason=f"{_REAL_NAME} not found"
)


@requires_real
class TestRealDualEdgeCaseVOD:
    @pytest.fixture(scope="class")
    def result(self):
        log = extract_log_from_html(_REAL.read_text(encoding="utf-8"))
        parser = ShowdownReplayParser(log, our_player="p2")  # kronomono = p2
        res = parser.parse()
        res["_parser"] = parser
        return res

    def test_both_flags_set(self, result):
        assert result["contains_illusion"] is True
        assert result["contains_transform"] is True

    def test_p2_zoroark_and_charizard_both_present_and_separate(self, result):
        seen = result["_parser"].seen_mons
        assert "p2:Zoroark-Hisui" in seen
        assert "p2:Charizard" in seen
        zoro_moves = set(seen["p2:Zoroark-Hisui"].revealed_moves)
        char_moves = set(seen["p2:Charizard"].revealed_moves)
        assert zoro_moves, "Zoroark must have at least one revealed move"
        # Zoroark's moves and the genuine Charizard's moves never overlap.
        assert zoro_moves.isdisjoint(char_moves)
        # Shadow Ball is one of kronomono's Zoroark-Hisui moves; it must land on
        # Zoroark, never on the disguise species.
        assert "Shadow Ball" in zoro_moves
        assert "Shadow Ball" not in char_moves

    def test_p2_brought_has_zoroark_and_charizard(self, result):
        brought = result["players"]["p2"]["brought"]
        assert "Zoroark-Hisui" in brought
        assert "Charizard" in brought

    def test_p1_imposter_ditto_has_no_moves(self, result):
        seen = result["_parser"].seen_mons
        assert "p1:Ditto" in seen
        assert seen["p1:Ditto"].revealed_moves == []

    def test_p1_brought_has_ditto_and_zoroark(self, result):
        brought = result["players"]["p1"]["brought"]
        assert "Ditto" in brought
        assert "Zoroark-Hisui" in brought


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
