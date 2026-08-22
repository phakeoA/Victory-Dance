"""
Level C / B0d — corpus-validation gate helpers (2026-06-30).

Unit-tests the PURE helpers of ``scratch/levelC_b0_validation_probe.py`` — action reconstruction
(move + species→bench_index switch + unresolved opp switch + mega flag), identity pairing over the
active∪bench union (incl. the fainted-relocated-to-bench case + illusion skip), and the alive predicate —
so the gate's correctness is locked independent of the (slow) full-corpus run. Design:
docs/levelC_B0d_validation_design.md.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

# repo root → import the scratch probe as a namespace package (robust to install mode)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# scratch/ is a local-only dev-harness dir (gitignored) — skip cleanly where it's absent (CI)
pytest.importorskip("scratch.levelC_b0_validation_probe",
                    reason="scratch/ probe harness is local-only, not in the published repo")
from scratch.levelC_b0_validation_probe import (  # noqa: E402
    reconstruct_actions, identity_map, _alive, _bench_index_for_species,
)


def _mon(species, hp=100.0, fainted=False, **extra):
    m = {"species": species, "base_species": species, "hp_pct": hp, "is_fainted": fainted}
    m.update(extra)
    return m


def _row(our_actions, opp_actions, our_bench=None, opp_bench=None):
    return {
        "decision_type": "turn",
        "state_before_actions": {
            "our_active": {"our_a": _mon("Sneasler"), "our_b": _mon("Basculegion")},
            "opp_active": {"opp_a": _mon("Corviknight"), "opp_b": _mon("Whimsicott")},
            "our_bench": our_bench if our_bench is not None else [_mon("Charizard", hp=None), _mon("Venusaur", hp=None)],
            "opp_bench": opp_bench if opp_bench is not None else [_mon("Incineroar", hp=None)],
        },
        "our_actions": our_actions,
        "opp_actions_actual": opp_actions,
    }


# ── action reconstruction ─────────────────────────────────────────────────────
def test_reconstruct_move():
    row = _row(
        [{"slot": "our_a", "action": "move", "move": "Close Combat", "target_slot": "opp_a"}],
        [{"slot": "opp_a", "action": "move", "move": "Iron Defense", "target_slot": "opp_a"}],
    )
    actions, unresolved = reconstruct_actions(row)
    assert unresolved is False
    assert actions["our_a"] == {"kind": "move", "move": "Close Combat", "target": "opp_a"}
    assert actions["opp_a"]["kind"] == "move"


def test_reconstruct_switch_resolves_species_to_bench_index():
    # our bench = [Charizard(0), Venusaur(1)]; switching to Venusaur → bench_index 1
    row = _row(
        [{"slot": "our_a", "action": "switch", "species": "Venusaur"}],
        [{"slot": "opp_a", "action": "move", "move": "Tackle", "target_slot": "our_a"}],
    )
    actions, unresolved = reconstruct_actions(row)
    assert unresolved is False
    assert actions["our_a"] == {"kind": "switch", "bench_index": 1}


def test_reconstruct_unresolved_opp_switch():
    # opp switches to a mon NOT on the (revealed) opp bench → unresolved
    row = _row(
        [{"slot": "our_a", "action": "move", "move": "Tackle", "target_slot": "opp_a"}],
        [{"slot": "opp_a", "action": "switch", "species": "Garchomp"}],   # not in opp_bench
    )
    actions, unresolved = reconstruct_actions(row)
    assert unresolved is True


def test_reconstruct_mega_flag():
    row = _row(
        [{"slot": "our_a", "action": "move", "move": "Flare Blitz", "target_slot": "opp_a", "gimmick_index": 1}],
        [{"slot": "opp_a", "action": "move", "move": "Tackle", "target_slot": "our_a"}],
    )
    actions, _ = reconstruct_actions(row)
    assert actions["our_a"].get("mega") is True


def test_bench_index_helper_norm():
    sb = {"our_bench": [_mon("Rotom-Mow"), _mon("Kingambit")]}
    assert _bench_index_for_species(sb, "our", "Rotom-Mow") == 0
    assert _bench_index_for_species(sb, "our", "kingambit") == 1   # norm-insensitive
    assert _bench_index_for_species(sb, "our", "Pikachu") is None


# ── identity pairing ──────────────────────────────────────────────────────────
def test_identity_map_union_active_and_bench():
    snap = {
        "our_active": {"our_a": _mon("Sneasler"), "our_b": _mon("Charizard")},
        "opp_active": {"opp_a": _mon("Corviknight"), "opp_b": None},
        "our_bench": [_mon("Basculegion", hp=100.0)],
        "opp_bench": [],
    }
    m = identity_map(snap, Counter())
    assert ("our", "sneasler") in m
    assert ("our", "charizard") in m
    assert ("our", "basculegion") in m       # bench included
    assert ("opp", "corviknight") in m
    assert ("opp_b") not in m                # None slot ignored


def test_identity_pairs_fainted_active_vs_bench():
    # PREDICTED keeps the fainted mon in its active slot (hp 0); REAL after relocates it to bench (hp 0).
    # Identity pairing by (side, base_species) must match them → same key, both fainted.
    predicted = {
        "our_active": {"our_a": _mon("Sneasler", hp=0.0, fainted=True), "our_b": _mon("Basculegion")},
        "opp_active": {"opp_a": None, "opp_b": None}, "our_bench": [], "opp_bench": [],
    }
    after = {
        "our_active": {"our_a": None, "our_b": _mon("Basculegion")},
        "opp_active": {"opp_a": None, "opp_b": None},
        "our_bench": [_mon("Sneasler", hp=0.0, fainted=True)], "opp_bench": [],
    }
    pm = identity_map(predicted, Counter())
    am = identity_map(after, Counter())
    key = ("our", "sneasler")
    assert key in pm and key in am
    assert (pm[key].get("hp_pct") or 0) <= 0 and (am[key].get("hp_pct") or 0) <= 0


def test_identity_skips_illusion_transform():
    sk = Counter()
    snap = {
        "our_active": {"our_a": _mon("Zoroark", illusion_active=True), "our_b": _mon("Ditto", is_transformed=True)},
        "opp_active": {}, "our_bench": [], "opp_bench": [],
    }
    m = identity_map(snap, sk)
    assert m == {}
    assert sk["illusion_transform"] == 2


def test_alive_predicate():
    assert _alive(_mon("X", hp=50.0)) is True
    assert _alive(_mon("X", hp=0.0)) is False
    assert _alive(_mon("X", hp=None)) is False
    assert _alive(None) is False
