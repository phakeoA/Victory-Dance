"""Tests for post-faint replacement transitions (the "#1 gap").

A forced replacement after a faint is now exported as its OWN transition
(decision_type="replacement") with a switch-only legality mask on the fainted
slot, so the in-battle policy can learn "who to send in after a faint".

These build tiny synthetic replays so they don't depend on the corpus.
"""

from __future__ import annotations

import pytest

from v_dance.parser.vod_parser.transitions import replay_to_transitions

# These exercise the action codec → require numpy/state_encoder.
pytest.importorskip("numpy")
from v_dance.encoders.state_encoder import StateEncoder, SWITCH_OFFSET, get_state_dim  # noqa: E402


# ── Synthetic replay helpers ────────────────────────────────────────────────
_HEADER = [
    "|player|p1|Alice|1|1500",
    "|player|p2|Bob|2|1500",
    "|teamsize|p1|6",
    "|teamsize|p2|6",
    "|gen|9",
    "|tier|[Gen 9] VGC 2026 Reg MA",
    "|poke|p1|Incineroar, L50, M|",
    "|poke|p1|Kingambit, L50, M|",
    "|poke|p1|Rillaboom, L50, M|",
    "|poke|p1|Urshifu, L50|",
    "|poke|p1|Flutter Mane, L50|",
    "|poke|p1|Amoonguss, L50|",
    "|poke|p2|Calyrex-Shadow, L50|",
    "|poke|p2|Miraidon, L50|",
    "|poke|p2|Chien-Pao, L50|",
    "|poke|p2|Ogerpon, L50|",
    "|poke|p2|Landorus, L50|",
    "|poke|p2|Grimmsnarl, L50, M|",
    "|teampreview",
    "|start",
    "|switch|p1a: Incineroar|Incineroar, L50, M|100/100",
    "|switch|p1b: Kingambit|Kingambit, L50, M|100/100",
    "|switch|p2a: Calyrex|Calyrex-Shadow, L50|100/100",
    "|switch|p2b: Miraidon|Miraidon, L50|100/100",
]


def _wrap(lines: list[str]) -> str:
    log = "\n".join(lines)
    return (
        '<input type="hidden" name="replayid" value="syn-replay-1" />'
        f'<script type="text/plain" class="battle-log-data">{log}</script>'
    )


def _run(tmp_path, lines, players=("p1", "p2")):
    f = tmp_path / "syn.html"
    f.write_text(_wrap(lines), encoding="utf-8")
    return replay_to_transitions(f, belief=None, players=list(players), source_type="B")


def _replacements(transitions):
    return [t for t in transitions if t.get("decision_type") == "replacement"]


# ── Single forced replacement ───────────────────────────────────────────────
def test_single_replacement_is_exported_with_switch_mask(tmp_path):
    log = _HEADER + [
        "|turn|1",
        "|move|p1a: Incineroar|Fake Out|p2a: Calyrex",
        "|move|p1b: Kingambit|Sucker Punch|p2a: Calyrex",
        "|move|p2b: Miraidon|Electro Drift|p1a: Incineroar",
        "|-damage|p1a: Incineroar|0 fnt",
        "|faint|p1a: Incineroar",
        "|switch|p1a: Rillaboom|Rillaboom, L50, M|100/100",   # forced replacement
        "|turn|2",
        "|move|p1b: Kingambit|Kowtow Cleave|p2b: Miraidon",
        "|move|p1a: Rillaboom|Grassy Glide|p2b: Miraidon",
        "|win|Bob",
    ]
    transitions = _run(tmp_path, log)
    repls = _replacements(transitions)
    assert len(repls) == 1
    r = repls[0]

    assert r["perspective"] == "p1"
    assert r["turn"] == 1
    assert r["reward"]["win"] is None
    assert r["opp_actions_actual"] == []

    # one switch action on the fainted slot, legal under its own mask
    assert len(r["our_actions"]) == 1
    act = r["our_actions"][0]
    assert act["slot"] == "our_a" and act["action"] == "switch"
    assert act["species"] == "Rillaboom"
    idx = act["action_index"]
    assert idx is not None and idx >= SWITCH_OFFSET          # a switch index
    assert r["action_mask"]["our_a"][idx] == 1               # legal under mask
    # the ally slot is NOT deciding
    assert r["action_mask"]["our_b"] == [0] * 16
    # mask offers only switches (no move buckets) for the empty fainted slot
    assert sum(r["action_mask"]["our_a"][:SWITCH_OFFSET]) == 0
    assert sum(r["action_mask"]["our_a"][SWITCH_OFFSET:]) >= 1


def test_replacement_state_excludes_fainted_active_and_encodes(tmp_path):
    log = _HEADER + [
        "|turn|1",
        "|move|p2b: Miraidon|Electro Drift|p1a: Incineroar",
        "|-damage|p1a: Incineroar|0 fnt",
        "|faint|p1a: Incineroar",
        "|switch|p1a: Urshifu|Urshifu, L50|100/100",
        "|turn|2",
        "|win|Alice",
    ]
    r = _replacements(_run(tmp_path, log))[0]
    snap = r["state_before_actions"]
    # the fainted Incineroar is no longer an active mon...
    assert "our_a" not in (snap.get("our_active") or {})
    # ...the living ally still is...
    assert "our_b" in (snap.get("our_active") or {})
    # ...and it sits in the bench, flagged fainted.
    benched = {m.get("species"): m for m in (snap.get("our_bench") or [])}
    assert benched.get("Incineroar", {}).get("is_fainted") is True
    # the encoder accepts the replacement state
    vec = StateEncoder().encode_snapshot(snap, turn=1)
    assert vec.shape == (get_state_dim(),)


# ── Never-before-seen replacement is still selectable ───────────────────────
def test_replacement_into_unseen_mon_is_indexable(tmp_path):
    # Urshifu has never been on the field; the own-side retrofit stubs it onto
    # the bench so the switch still resolves to a real index (not None).
    log = _HEADER + [
        "|turn|1",
        "|move|p2a: Calyrex|Astral Barrage|p1a: Incineroar",
        "|-damage|p1a: Incineroar|0 fnt",
        "|faint|p1a: Incineroar",
        "|switch|p1a: Urshifu|Urshifu, L50|100/100",
        "|turn|2",
        "|win|Alice",
    ]
    r = _replacements(_run(tmp_path, log))[0]
    act = r["our_actions"][0]
    assert act["species"] == "Urshifu"
    assert act["action_index"] is not None
    assert r["action_mask"]["our_a"][act["action_index"]] == 1


# ── Voluntary turn-start switch is NOT a replacement ────────────────────────
def test_voluntary_switch_is_not_a_replacement(tmp_path):
    log = _HEADER + [
        "|turn|1",
        "|switch|p1a: Rillaboom|Rillaboom, L50, M|100/100",   # Incineroar was ALIVE
        "|move|p1b: Kingambit|Sucker Punch|p2a: Calyrex",
        "|turn|2",
        "|win|Alice",
    ]
    transitions = _run(tmp_path, log)
    assert _replacements(transitions) == []
    # the voluntary switch is still the turn-1 our_a decision
    t1 = next(t for t in transitions
              if t["turn"] == 1 and t["perspective"] == "p1"
              and t["decision_type"] == "turn")
    a = next(x for x in t1["our_actions"] if x["slot"] == "our_a")
    assert a["action"] == "switch" and a["species"] == "Rillaboom"


# ── Double faint in one turn → two replacement decisions ────────────────────
def test_double_faint_yields_two_replacements(tmp_path):
    log = _HEADER + [
        "|turn|1",
        "|move|p2a: Calyrex|Astral Barrage|p1a: Incineroar",
        "|-damage|p1a: Incineroar|0 fnt",
        "|-damage|p1b: Kingambit|0 fnt",
        "|faint|p1a: Incineroar",
        "|faint|p1b: Kingambit",
        "|switch|p1a: Rillaboom|Rillaboom, L50, M|100/100",
        "|switch|p1b: Urshifu|Urshifu, L50|100/100",
        "|turn|2",
        "|win|Bob",
    ]
    repls = _replacements(_run(tmp_path, log))
    assert len(repls) == 2
    slots = {r["our_actions"][0]["slot"] for r in repls}
    assert slots == {"our_a", "our_b"}
    for r in repls:
        act = r["our_actions"][0]
        assert act["action_index"] is not None
        assert r["action_mask"][act["slot"]][act["action_index"]] == 1
        # the non-acting slot row is all-zero
        other = "our_b" if act["slot"] == "our_a" else "our_a"
        assert r["action_mask"][other] == [0] * 16


# ── Type A (single perspective) only emits own-side replacements ────────────
def test_type_a_only_emits_own_side_replacement(tmp_path):
    # p2's mon faints and is replaced; from players=["p1"] that is the opponent
    # and must NOT produce a replacement transition.
    log = _HEADER + [
        "|turn|1",
        "|move|p1a: Incineroar|Fake Out|p2a: Calyrex",
        "|-damage|p2a: Calyrex|0 fnt",
        "|faint|p2a: Calyrex",
        "|switch|p2a: Chien-Pao|Chien-Pao, L50|100/100",
        "|turn|2",
        "|win|Alice",
    ]
    transitions = _run(tmp_path, log, players=["p1"])
    assert _replacements(transitions) == []
