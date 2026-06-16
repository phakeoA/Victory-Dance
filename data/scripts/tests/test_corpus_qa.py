"""Tests for corpus_qa.py — the JSONL training-corpus label audit / regression gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import corpus_qa as qa


def _tx(replay_id, perspective="p1", decision_type="turn", our_actions=None,
        action_mask=None, gimmick_mask=None, rating=1800, winner="Me", username="Me"):
    return {
        "replay_id": replay_id,
        "perspective": perspective,
        "decision_type": decision_type,
        "our_actions": our_actions or [],
        "action_mask": action_mask or {},
        "gimmick_mask": gimmick_mask,
        "winner": winner,
        "players": {
            "our_side": perspective,
            perspective: {"username": username, "rating_before": rating},
        },
    }


def _write(folder: Path, name: str, txs):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.jsonl").write_text(
        "\n".join(json.dumps(t) for t in txs), encoding="utf-8")


def _mask(*legal):
    row = [0] * 16
    for i in legal:
        row[i] = 1
    return row


def test_counts_perspective_and_decision_types(tmp_path):
    _write(tmp_path, "r1", [
        _tx("r1", "p1", "turn",
            our_actions=[{"slot": "our_a", "action_index": 3}], action_mask={"our_a": _mask(3)}),
        _tx("r1", "p2", "replacement",
            our_actions=[{"slot": "our_a", "action_index": 12}], action_mask={"our_a": _mask(12)}),
    ])
    rep = qa.audit_folders([str(tmp_path)])
    assert rep["files"] == 1 and rep["transitions"] == 2
    assert rep["perspective"] == {"p1": 1, "p2": 1}
    assert rep["by_decision_type"]["turn"]["transitions"] == 1
    assert rep["by_decision_type"]["replacement"]["transitions"] == 1
    assert rep["illegal_under_mask"] == 0


def test_detects_illegal_under_mask_and_null(tmp_path):
    _write(tmp_path, "r1", [
        _tx("r1", our_actions=[{"slot": "our_a", "action_index": 5}],   # 5 NOT legal
            action_mask={"our_a": _mask(3)}),
        _tx("r1", our_actions=[{"slot": "our_b", "action_index": None}],  # null
            action_mask={"our_b": _mask(3)}),
    ])
    rep = qa.audit_folders([str(tmp_path)])
    assert rep["illegal_under_mask"] == 1
    assert rep["null_index"] == 1
    assert rep["by_decision_type"]["turn"]["illegal_under_mask"] == 1


def test_detects_duplicate_replay_ids_across_files(tmp_path):
    _write(tmp_path, "a", [_tx("dup-1")])
    _write(tmp_path, "b", [_tx("dup-1")])   # same replay_id in a different file
    _write(tmp_path, "c", [_tx("unique-2")])
    rep = qa.audit_folders([str(tmp_path)])
    assert rep["duplicate_replay_ids"] == ["dup-1"]


def test_rating_histogram_and_outcome(tmp_path):
    _write(tmp_path, "r1", [
        _tx("r1", rating=1550, winner="Me", username="Me"),       # 0-1600, won
        _tx("r2", rating=1750, winner="Them", username="Me"),     # 1700-1800, lost
        _tx("r3", rating=2010, winner="Me", username="Me"),       # 2000+, won
    ])
    rep = qa.audit_folders([str(tmp_path)])
    assert rep["rating_hist"]["0-1600"] == 1
    assert rep["rating_hist"]["1700-1800"] == 1
    assert rep["rating_hist"]["2000+"] == 1
    assert rep["rating_min"] == 1550 and rep["rating_max"] == 2010
    assert rep["won"]["True"] == 2 and rep["won"]["False"] == 1


def test_gimmick_coverage_and_positives(tmp_path):
    _write(tmp_path, "r1", [
        _tx("r1", gimmick_mask={"our_a": [1, 1]},
            our_actions=[{"slot": "our_a", "action_index": 3, "gimmick_index": 1}],   # mega
            action_mask={"our_a": _mask(3)}),
        _tx("r1", gimmick_mask=None,                                                   # no mask
            our_actions=[{"slot": "our_a", "action_index": 3, "gimmick_index": 0}],
            action_mask={"our_a": _mask(3)}),
    ])
    rep = qa.audit_folders([str(tmp_path)])
    assert rep["gimmick"]["has_mask"] == 1
    assert rep["gimmick"]["no_mask"] == 1
    assert rep["gimmick"]["mega_positive"] == 1


def test_main_exit_codes(tmp_path, capsys):
    # clean corpus → 0
    _write(tmp_path / "clean", "r1",
           [_tx("r1", our_actions=[{"slot": "our_a", "action_index": 3}],
                action_mask={"our_a": _mask(3)})])
    assert qa.main([str(tmp_path / "clean")]) == 0
    # illegal target → hard fail → 1
    _write(tmp_path / "bad", "r1",
           [_tx("r1", our_actions=[{"slot": "our_a", "action_index": 5}],
                action_mask={"our_a": _mask(3)})])
    assert qa.main([str(tmp_path / "bad")]) == 1
