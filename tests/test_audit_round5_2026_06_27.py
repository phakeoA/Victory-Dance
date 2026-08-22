"""Regression tests for the 2026-06-27 round-5 audit (4 confirmed; 3 are follow-ons to this session's fixes).

  #1  status._numeric / archive._numeric_stats DROP non-finite floats (a NaN opp_ce no longer emits a bare
      `NaN` token that breaks the browser dashboard's JSON.parse).
  #3  corpus_qa treats an all-zero/empty action_mask as a no_mask SKIP (mirroring the BC loader), not illegal.
(#2 play_pairing single-snapshot race and #4 gauntlet None-elo non-persist are covered by the suite + inspection.)
"""
from __future__ import annotations

import json
import math

import pytest


# ── #1: non-finite floats are dropped from the JSON-bound stat dicts ──────────
def test_status_and_archive_numeric_drop_nonfinite():
    from v_dance.selfplay.status import _numeric
    from v_dance.selfplay.archive import _numeric_stats
    src = {"loss": 1.5, "opp_ce": float("nan"), "kl": float("inf"), "halted": True}
    for fn in (_numeric, _numeric_stats):
        out = fn(src)
        assert out["loss"] == pytest.approx(1.5)
        assert out["halted"] == 1.0                  # bool → 0/1 still works
        assert "opp_ce" not in out and "kl" not in out   # NaN/inf dropped
        # and the surviving dict is valid strict JSON (no bare NaN token)
        json.dumps(out, allow_nan=False)             # would raise if any NaN/inf slipped through


# ── #3: corpus_qa treats an all-zero mask as no_mask skip, not illegal ────────
def test_corpus_qa_all_zero_mask_is_no_mask_not_illegal(tmp_path):
    corpus_qa = pytest.importorskip("v_dance.datatools.corpus_qa")
    d = tmp_path / "J"
    d.mkdir()
    # a transition whose our_a slot has a non-null action_index but an ALL-ZERO action_mask row
    row = {
        "replay_id": "rz1", "perspective": "p1", "turn": 1, "decision_type": "turn",
        "our_actions": [{"slot": "our_a", "action_index": 2, "gimmick_index": 0}],
        "action_mask": {"our_a": [0] * 16},
        "state": {},
    }
    (d / "rz1.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    report = corpus_qa.audit_folders([str(d)])
    assert report["illegal_under_mask"] == 0          # must NOT hard-fail on an all-zero mask
    assert report.get("no_mask_skips", 0) >= 1        # counted as a no_mask skip (loader parity)
