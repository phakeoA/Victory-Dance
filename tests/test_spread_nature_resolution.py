"""#8: recover per-spread natures where Pikalytics' new layout decoupled spread<->nature.

The scraper stamps the single MODAL nature onto every EV spread (the page no longer pairs them).
For a mixed phys/special mon that mislabels spreads (a physical spread tagged with a -Atk special
nature), biasing est-stats. belief_state re-derives each spread's nature from its EV investment +
the preserved `natures` distribution — a NO-OP for M-A (real per-spread natures) and single-nature mons.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from v_dance.parser.belief_state import (
    BeliefState, NATURE_BOOSTS, _best_nature_for_spread, _resolve_spread_natures,
)

_REPO = Path(__file__).resolve().parents[1]
_PIKA_MB = _REPO / "data" / "pikalytics_regmb.json"
_PIKA_MA = _REPO / "data" / "pikalytics_regma.json"

# Arcanine-like natures: Modest (special) modal, then Jolly/Adamant (physical), Careful (defensive)
_NATS = [{"nature": "Modest", "pct": 36.7}, {"nature": "Jolly", "pct": 25.8},
         {"nature": "Careful", "pct": 18.3}, {"nature": "Adamant", "pct": 18.3}]
_PHYS = [0, 7, 0, 0, 0, 7]   # atk-invested bucket spread
_SPEC = [0, 0, 0, 7, 0, 7]   # spa-invested bucket spread
_NEUT = [32, 0, 16, 0, 16, 0]


def test_physical_spread_never_takes_an_atk_lowering_nature():
    n = _best_nature_for_spread(_PHYS, _NATS)
    assert NATURE_BOOSTS[n][1] != "atk", f"{n} lowers Atk on a physical spread"
    assert n == "Adamant"   # offence-boost fix: prefer the +Atk nature over the merely-non-lowering Jolly


def test_special_spread_never_takes_a_spa_lowering_nature():
    n = _best_nature_for_spread(_SPEC, _NATS)
    assert NATURE_BOOSTS[n][1] != "spa", f"{n} lowers SpA on a special spread"
    assert n == "Modest"   # top-pct nature, doesn't drop SpA


def test_neutral_spread_keeps_modal():
    assert _best_nature_for_spread(_NEUT, _NATS) == "Modest"   # highest-pct overall


def test_empty_natures_returns_none():
    assert _best_nature_for_spread(_PHYS, []) is None


def test_broadcast_entry_is_re_natured_per_spread():
    entry = {"spreads": [{"nature": "Modest", "evs": _PHYS, "pct": 50.0},
                         {"nature": "Modest", "evs": _SPEC, "pct": 50.0}],
             "natures": _NATS}
    _resolve_spread_natures(entry)
    assert entry["spreads"][0]["nature"] == "Adamant"  # physical → offence-boosting +Atk nature
    assert entry["spreads"][1]["nature"] == "Modest"   # special → +SpA nature


def test_real_per_spread_natures_left_untouched():
    # M-A shape: spreads already carry DIFFERENT natures → never overwritten
    entry = {"spreads": [{"nature": "Adamant", "evs": _PHYS, "pct": 50.0},
                         {"nature": "Timid", "evs": _SPEC, "pct": 50.0}],
             "natures": _NATS}
    _resolve_spread_natures(entry)
    assert [s["nature"] for s in entry["spreads"]] == ["Adamant", "Timid"]


def test_single_nature_distribution_left_untouched():
    entry = {"spreads": [{"nature": "Jolly", "evs": _PHYS, "pct": 50.0},
                         {"nature": "Jolly", "evs": _SPEC, "pct": 50.0}],
             "natures": [{"nature": "Jolly", "pct": 99.0}]}
    _resolve_spread_natures(entry)
    assert all(s["nature"] == "Jolly" for s in entry["spreads"])


@pytest.mark.skipif(not _PIKA_MB.exists(), reason="regmb belief absent")
def test_real_regmb_mixed_attacker_gets_distinct_spread_natures():
    b = BeliefState(_PIKA_MB)
    e = b._data["Arcanine"]
    natset = {s["nature"] for s in e["spreads"]}
    assert len(natset) > 1, "broadcast not fixed — Arcanine spreads still share one nature"
    # a physical spread (atk>spa) must not be tagged with an Atk-lowering nature
    for s in e["spreads"]:
        ev = s["evs"]
        if ev[1] > ev[3]:
            assert NATURE_BOOSTS[s["nature"]][1] != "atk"
        elif ev[3] > ev[1]:
            assert NATURE_BOOSTS[s["nature"]][1] != "spa"


@pytest.mark.skipif(not _PIKA_MA.exists(), reason="regma belief absent")
def test_regma_spreads_unchanged_by_resolution():
    # M-A has 0 broadcast entries; resolution must not change any regma spread nature
    raw = json.loads(_PIKA_MA.read_text(encoding="utf-8"))["pokemon"]
    b = BeliefState(_PIKA_MA)
    diffs = 0
    for sp, e_raw in raw.items():
        before = [s.get("nature") for s in (e_raw.get("spreads") or [])]
        after = [s.get("nature") for s in (b._data[sp].get("spreads") or [])]
        diffs += int(before != after)
    assert diffs == 0, f"{diffs} regma entries had spread natures changed (M-A must be untouched)"
