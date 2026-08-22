"""15b-feat.robust.1 + .2: the dex-grounded mechanic COVERAGE GUARD.

The guard's job is to FAIL when a reg swap introduces a mechanic-bearing ability/move the
curated tables don't map. Tests: the pure diff logic (a present+unmapped must-have is flagged,
a mapped one isn't); the dex parser detects real bearers (pinned dex); and the live M-A data
has FULL must-have coverage (0 'high' findings) — so this test goes RED on a future data swap
that adds an untagged setter/abuser, which is exactly the point.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
from v_dance.training import mechanic_coverage as MC  # noqa: E402
from v_dance.parser.belief_state import BeliefState  # noqa: E402

_PIKA = _REPO / "data" / "pikalytics_regma.json"
_PIKA_MB = _REPO / "data" / "pikalytics_regmb.json"
_DEX = _REPO / "pokemon-showdown" / "data"
_has_dex = (_DEX / "abilities.ts").exists() and (_DEX / "moves.ts").exists()


# ── pure diff logic (no dex, no data) ─────────────────────────────────────────
def test_diff_flags_unmapped_must_have():
    bearers = {"New Weather Maker": {"weather_setter"}, "Sand Stream": {"weather_setter"}}
    present = {"New Weather Maker", "Sand Stream"}
    curated = {"Sand Stream"}                                  # the novel setter is NOT mapped
    out = MC._diff(bearers, present, curated)
    assert [f["name"] for f in out] == ["New Weather Maker"]
    assert out[0]["severity"] == "high"                        # setter = must-have = FAIL


def test_diff_skips_mapped_absent_and_known_untagged():
    bearers = {"Sand Stream": {"weather_setter"}, "Terrain Pulse": {"terrain_setter"},
               "Future Setter": {"weather_setter"}}
    # Sand Stream mapped; Terrain Pulse in _KNOWN_UNTAGGED; Future Setter present-but... absent from data
    present = {"Sand Stream", "Terrain Pulse"}
    assert MC._diff(bearers, present, {"Sand Stream"}) == []


def test_diff_warn_severity_for_role_class():
    out = MC._diff({"Some Screen Move": {"screens"}}, {"Some Screen Move"}, set())
    assert out and out[0]["severity"] == "warn"               # role heuristic = warn, not fail


# ── dex parser (pinned Showdown dex) ──────────────────────────────────────────
@pytest.mark.skipif(not _has_dex, reason="bundled Showdown dex not present")
def test_dex_parser_detects_real_bearers():
    b = MC.dex_mechanic_bearers()
    assert "weather_setter" in b.get("Drought", set())
    assert "weather_setter" in b.get("Sand Stream", set())
    assert "weather_abuser" in b.get("Sand Rush", set())
    assert "weather_abuser" in b.get("Swift Swim", set())
    assert "terrain_setter" in b.get("Psychic Terrain", set())
    assert "terrain_abuser" in b.get("Surge Surfer", set())
    assert "tailwind" in b.get("Tailwind", set())
    assert "trick_room" in b.get("Trick Room", set())
    assert "redirect" in b.get("Follow Me", set())
    assert "screens" in b.get("Reflect", set())
    assert "spread" in b.get("Earthquake", set())          # ally-hitting spread move (15b-feat.spread)
    assert "spread" in b.get("Discharge", set())
    assert "stat_reverser" in b.get("Contrary", set())     # Contrary reverses stat drops (15b-feat.stat)
    assert "stat_reverser" not in b.get("Simple", set())   # Simple amplifies (*= 2) — not a reverser


@pytest.mark.skipif(not _has_dex, reason="bundled Showdown dex not present")
def test_guard_catches_a_dropped_spread_move(monkeypatch):
    import v_dance.training.tp_features as TF
    patched = dict(TF.SPREAD_MOVE)
    patched.pop("Earthquake", None)
    monkeypatch.setattr(TF, "SPREAD_MOVE", patched)
    findings = MC.audit_mechanic_coverage(BeliefState(_PIKA))
    assert any(f["name"] == "Earthquake" and f["severity"] == "high" for f in findings)


# ── full-coverage on live M-A (goes RED on a swap that adds an untagged must-have) ─
@pytest.mark.skipif(not _has_dex, reason="bundled Showdown dex not present")
def test_live_MA_has_no_unmapped_must_have():
    findings = MC.audit_mechanic_coverage(BeliefState(_PIKA))
    high = [f for f in findings if f["severity"] == "high"]
    assert high == [], f"unmapped must-have mechanics in the data — add to tp_features tables: {high}"


@pytest.mark.skipif(not _has_dex, reason="bundled Showdown dex not present")
def test_guard_catches_a_dropped_tag(monkeypatch):
    # Simulate forgetting a tag: drop Drought from the curated set -> the guard must flag it
    # (Drought is a weather setter present in the M-A data).
    import v_dance.training.tp_features as TF
    patched = dict(TF.SETTER_ABILITY)
    patched.pop("Drought", None)
    monkeypatch.setattr(TF, "SETTER_ABILITY", patched)
    findings = MC.audit_mechanic_coverage(BeliefState(_PIKA))
    assert any(f["name"] == "Drought" and f["severity"] == "high" for f in findings)


# ════════════════════════════════════════════════════════════════════════════════
# v9 BATTLE-ENCODER coverage guard (B1-mechanics step 7a)
# ════════════════════════════════════════════════════════════════════════════════
from v_dance.encoders import mechanic_tags as MT  # noqa: E402


# ── pure logic: the dex-field → required-tag mapping (no dex, no data) ─────────
def test_required_move_tags_pure():
    assert MC.required_move_tags("forceSwitch: true,") == {"force_switch"}
    assert MC.required_move_tags("selfSwitch: true,") == {"pivot"}
    assert MC.required_move_tags("selfSwitch: 'copyvolatile',") == {"pivot"}
    assert MC.required_move_tags("sideCondition: 'stealthrock',") == {"hazard_set"}
    # tox folds into status_psn (matches the mechanic_tags deriver)
    assert MC.required_move_tags("status: 'tox',") == {"status_psn"}
    assert MC.required_move_tags("status: 'brn',") == {"status_brn"}
    # a pure damaging move with no must-have field → no requirement
    assert MC.required_move_tags("basePower: 80, type: 'Fire',") == set()
    # the must-have set names the battle classes
    assert {"force_switch", "pivot", "hazard_set", "trapping"} <= MC.V9_MUST_HAVE


# ── RECALL on the pinned dex: every mechanic-bearing move/ability is tagged ────
@pytest.mark.skipif(not _has_dex, reason="bundled Showdown dex not present")
def test_v9_move_tag_coverage_is_full():
    findings = MC.audit_v9_move_tag_coverage()
    assert findings == [], f"dex moves whose mechanic_tags miss a required tag: {findings}"


@pytest.mark.skipif(not _has_dex, reason="bundled Showdown dex not present")
def test_v9_move_tag_guard_catches_a_dropped_tag(monkeypatch):
    # drop force_switch from every move's tag set -> the guard must flag the phazing moves
    patched = {k: (v - {"force_switch"}) for k, v in MT.MOVE_TAGS.items()}
    monkeypatch.setattr(MT, "MOVE_TAGS", patched)
    findings = MC.audit_v9_move_tag_coverage()
    flagged = {f["id"] for f in findings}
    assert {"roar", "whirlwind", "dragontail", "circlethrow"} <= flagged
    assert all("force_switch" in f["missing"] and f["severity"] == "high" for f in findings)


@pytest.mark.skipif(not _has_dex, reason="bundled Showdown dex not present")
def test_v9_trapping_coverage_is_full():
    assert MC.audit_v9_trapping_coverage() == []


@pytest.mark.skipif(not _has_dex, reason="bundled Showdown dex not present")
def test_v9_trapping_guard_catches_a_dropped_tag(monkeypatch):
    patched = {k: (v - {"trapping"}) for k, v in MT._ABILITY_TAGS.items()}
    monkeypatch.setattr(MT, "_ABILITY_TAGS", patched)
    flagged = {f["id"] for f in MC.audit_v9_trapping_coverage()}
    assert {"shadowtag", "arenatrap", "magnetpull"} <= flagged


# ── VOCAB: every format-present ability/move/item resolves to an embedding index ─
class _StubBelief:
    """Minimal BeliefState surface for the vocab audit (no data file needed)."""
    def __init__(self, abilities, moves, items):
        self._a, self._m, self._i = abilities, moves, items

    def all_pokemon(self):
        return ["Stub"]

    def ability_distribution(self, sp, top_k=8):
        return [{"name": n} for n in self._a]

    def move_distribution(self, sp, top_k=20):
        return [{"name": n} for n in self._m]

    def item_distribution(self, sp, top_k=8):
        return [{"name": n} for n in self._i]


def test_v9_vocab_audit_flags_unknown_name_and_drops_no_item_sentinel():
    b = _StubBelief(abilities=["Intimidate", "Not An Ability"],
                    moves=["Protect"], items=["Nothing", "Leftovers"])
    out = MC.audit_v9_vocab_coverage(b)
    assert out["ability"] == ["Not An Ability"]          # bogus name flagged
    assert out["move"] == [] and out["item"] == []       # "Nothing" sentinel dropped, not flagged


@pytest.mark.skipif(not _has_dex, reason="bundled Showdown dex not present")
def test_v9_vocab_coverage_regma_is_clean():
    out = MC.audit_v9_vocab_coverage(BeliefState(_PIKA))
    assert out == {"ability": [], "move": [], "item": []}, f"M-A names with no vocab index: {out}"


@pytest.mark.skipif(not (_has_dex and _PIKA_MB.exists()), reason="dex or regmb data absent")
def test_v9_vocab_coverage_regmb_is_clean():
    # the actual TRAINING format. This goes RED if a future scrape re-introduces the
    # natures-in-abilities pollution the guard caught in Swampert (see scrape_pikalytics
    # _NATURE_NOISE filter), or any other usage name with no vocab index.
    out = MC.audit_v9_vocab_coverage(BeliefState(_PIKA_MB))
    assert out == {"ability": [], "move": [], "item": []}, f"M-B names with no vocab index: {out}"


@pytest.mark.skipif(not _has_dex, reason="bundled Showdown dex not present")
def test_v9_audit_coverage_combined_shape_and_clean():
    out = MC.audit_v9_coverage(BeliefState(_PIKA))
    assert set(out) == {"move_tags", "trapping", "vocab"}
    assert out["move_tags"] == [] and out["trapping"] == []
    assert out["vocab"] == {"ability": [], "move": [], "item": []}
