"""Task G: cap the BC-era diagnostic replay buffer so a long run can't fill the disk.

``prune_replay_buffer`` keeps the N most-recently-modified ``.jsonl`` traces and deletes the rest
(the self-play RL training reads a SEPARATE store, so trimming these is safe)."""
from __future__ import annotations

import os

from v_dance.play.vgc_base import prune_replay_buffer


def test_prune_keeps_newest(tmp_path):
    d = tmp_path / "replay_buffer"
    d.mkdir()
    for i in range(5):                                  # p0 oldest … p4 newest (deterministic mtimes)
        f = d / f"p{i}.jsonl"
        f.write_text("x")
        os.utime(f, (1000 + i, 1000 + i))
    (d / "notes.txt").write_text("not a buffer")        # non-.jsonl must be ignored
    deleted = prune_replay_buffer(d, keep=2)
    assert deleted == 3
    assert sorted(p.name for p in d.glob("*.jsonl")) == ["p3.jsonl", "p4.jsonl"]   # the 2 newest
    assert (d / "notes.txt").exists()                   # untouched


def test_prune_keep_all_when_zero(tmp_path):
    d = tmp_path / "rb"
    d.mkdir()
    (d / "a.jsonl").write_text("x")
    (d / "b.jsonl").write_text("x")
    assert prune_replay_buffer(d, keep=0) == 0          # 0 = keep ALL
    assert len(list(d.glob("*.jsonl"))) == 2


def test_prune_missing_dir_is_noop(tmp_path):
    assert prune_replay_buffer(tmp_path / "does-not-exist", keep=5) == 0
