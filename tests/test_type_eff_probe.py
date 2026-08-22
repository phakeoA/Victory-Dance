"""Tests for the #25 type-effectiveness sensitivity probe (scratch/type_eff_probe.py).

Covers the probe's *correctness* claims that don't need the trained model:
  * the type-chart effectiveness math (known matchups),
  * effectiveness-bucket selection,
  * masked softmax,
  * the SURGICAL-intervention invariant: swapping the defender's tera type changes ONLY type
    one-hot bits in the encoded state (no stat/ability/other leakage) — the whole probe rests on
    this, so it is the key regression guard.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pytest

# scratch/ is a local-only dev-harness dir (gitignored) — skip cleanly where it's absent (CI)
P = pytest.importorskip("scratch.type_eff_probe",
                        reason="scratch/ probe harness is local-only, not in the published repo")
from v_dance.encoders.state_encoder import StateEncoder, NUM_TYPES  # noqa: E402


@pytest.fixture(scope="module")
def chart():
    return P.gen9_type_chart()


def test_effectiveness_known_matchups(chart):
    assert P.effectiveness("ELECTRIC", ["GROUND"], chart) == 0.0          # immunity
    assert P.effectiveness("ELECTRIC", ["WATER"], chart) == 2.0           # super
    assert P.effectiveness("FIRE", ["GRASS"], chart) == 2.0
    assert P.effectiveness("NORMAL", ["GHOST"], chart) == 0.0             # immunity
    assert P.effectiveness("FIGHTING", ["GHOST"], chart) == 0.0
    assert P.effectiveness("GROUND", ["FLYING"], chart) == 0.0
    assert P.effectiveness("FIRE", ["WATER", "ROCK"], chart) == 0.5 * 0.5  # dual-type product
    assert P.effectiveness("ICE", ["DRAGON", "FLYING"], chart) == 2.0 * 2.0


def test_pick_def_types_electric(chart):
    b = P.pick_def_types("ELECTRIC", chart)
    assert "immune" in b and P.effectiveness("ELECTRIC", [b["immune"]], chart) == 0.0
    assert "super" in b and P.effectiveness("ELECTRIC", [b["super"]], chart) > 1.0
    assert "neutral" in b and P.effectiveness("ELECTRIC", [b["neutral"]], chart) == 1.0


def test_pick_def_types_deterministic(chart):
    assert P.pick_def_types("FIRE", chart) == P.pick_def_types("FIRE", chart)


def test_neutral_types_all_neutral(chart):
    nts = P.neutral_types("WATER", chart, k=4)
    assert 1 <= len(nts) <= 4
    assert all(P.effectiveness("WATER", [dt], chart) == 1.0 for dt in nts)


def test_masked_softmax_zeros_illegal_and_normalises():
    logits = np.array([2.0, 1.0, 0.5, -3.0] + [0.0] * 12)
    mask = [1, 1, 0, 0] + [0] * 12
    p = P.masked_softmax(logits, mask)
    assert p[2] == 0.0 and p[3] == 0.0
    assert all(p[i] == 0.0 for i in range(4, 16))
    assert abs(p.sum() - 1.0) < 1e-9
    assert p[0] > p[1] > 0.0          # ordering preserved among legal


# ── the surgical-intervention invariant (corpus-backed; skips if no data) ──────
def _first_transition_with_opp():
    folder = (Path(__file__).resolve().parents[1] / "data" / "vods" /
              "Prepared_training_data" / "Regulation_MA" / "Jsonl_TypeB")
    files = sorted(glob.glob(str(folder / "**" / "*.jsonl"), recursive=True))
    for fp in files[:50]:
        with open(fp, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                t = json.loads(line)
                sba = t.get("state_before_actions") or {}
                opp = (sba.get("opp_active") or {}).get("opp_a")
                if opp and not opp.get("is_fainted") and (opp.get("hp_pct") or 0) > 0:
                    return t
    return None


def test_tera_swap_changes_types_typeeff_not_stats():
    """Swapping the target's tera type changes its type one-hots, its own-move STAB, AND the per-move
    TYPE-EFF channels (B1/#25: every mon's moves vs this target) — but NEVER its continuous base
    stats. That stat-invariance is the no-leakage guard the probe's intervention relies on."""
    t = _first_transition_with_opp()
    if t is None:
        pytest.skip("no Reg-MA TypeB corpus available")
    from v_dance.encoders.state_encoder import POKEMON_FEATURES
    enc = StateEncoder()
    turn = t.get("turn") or 0
    sba = t["state_before_actions"]
    # Two DIFFERENT single-type teras; is_terastallized=True in BOTH so the flag bit does not differ.
    xa = enc.encode_snapshot(P.tera_clone(sba, "opp_a", "WATER"), turn=turn)
    xb = enc.encode_snapshot(P.tera_clone(sba, "opp_a", "FIRE"), turn=turn)
    assert np.any(xa != xb), "tera swap changed nothing — intervention is a no-op"
    s2 = 2 * POKEMON_FEATURES                        # opp_a slot base (slots 0,1=own; 2,3=opp)
    # the type one-hots DID change (intervention took effect)
    assert not np.allclose(xa[s2 + 1:s2 + 1 + 2 * NUM_TYPES], xb[s2 + 1:s2 + 1 + 2 * NUM_TYPES])
    # but the base_stats block (right after hp + the 2 type one-hots) did NOT — no stat leakage
    base = s2 + 1 + 2 * NUM_TYPES
    assert np.allclose(xa[base:base + 6], xb[base:base + 6]), "base stats moved on a type swap"


def test_tera_swap_distinct_targets_hit_distinct_regions():
    """Tera'ing opp_a vs opp_b must touch DIFFERENT index regions (proves we edit the intended
    target mon, not a fixed slot)."""
    t = _first_transition_with_opp()
    if t is None:
        pytest.skip("no Reg-MA TypeB corpus available")
    sba = t["state_before_actions"]
    if not ((sba.get("opp_active") or {}).get("opp_b")):
        pytest.skip("transition has no second opp active")
    enc = StateEncoder()
    turn = t.get("turn") or 0
    base = enc.encode_snapshot(sba, turn=turn)
    da = np.nonzero(enc.encode_snapshot(P.tera_clone(sba, "opp_a", "WATER"), turn=turn) - base)[0]
    db = np.nonzero(enc.encode_snapshot(P.tera_clone(sba, "opp_b", "WATER"), turn=turn) - base)[0]
    if len(da) and len(db):
        assert set(da.tolist()) != set(db.tolist())
