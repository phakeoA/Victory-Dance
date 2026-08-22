"""Offline tests for per-generation resume snapshots (task #20).

Pure path logic — create dummy ``snap_gen{N}.pt`` files and exercise the resolver / lister /
pruner. No torch state involved (the snapshot CONTENTS are covered by the existing
test_selfplay_resume gold test)."""
from __future__ import annotations

from pathlib import Path

from v_dance.selfplay import resume as RS


def _make_snaps(d: Path, gens):
    for g in gens:
        (d / f"snap_gen{g}.pt").write_text("x")


def _make_sub_snaps(d: Path, gens):
    sub = d / "sub_checkpoints"
    sub.mkdir(parents=True, exist_ok=True)
    for g in gens:
        (sub / f"snap_gen{g}.pt").write_text("x")


def test_snapshot_path_for():
    p = RS.snapshot_path_for("/arch", 5)
    assert p.name == "snap_gen5.pt"
    assert p.parent.name == "sub_checkpoints"              # tidy archive: snapshots in sub_checkpoints/


def test_list_and_latest_snapshot(tmp_path):
    # legacy FLAT layout (snaps at the archive root) still resolves (backward-compat)
    _make_snaps(tmp_path, [0, 2, 10])
    (tmp_path / "gen3.pt").write_text("not a snapshot")     # must be ignored
    (tmp_path / "resume.pt").write_text("legacy")           # ignored too
    gens = [g for g, _ in RS.list_snapshots(tmp_path)]
    assert gens == [0, 2, 10]                               # sorted, only snap_gen*
    assert RS.latest_snapshot(tmp_path).name == "snap_gen10.pt"


def test_list_snapshots_reads_sub_checkpoints(tmp_path):
    _make_sub_snaps(tmp_path, [0, 3, 7])                    # the NEW layout
    assert [g for g, _ in RS.list_snapshots(tmp_path)] == [0, 3, 7]
    assert RS.latest_snapshot(tmp_path).parent.name == "sub_checkpoints"


def test_list_snapshots_backward_compat_root_and_subdir(tmp_path):
    (tmp_path / "snap_gen0.pt").write_text("root")         # old flat snap
    _make_sub_snaps(tmp_path, [0, 1])                       # new ones; gen0 also in sub_checkpoints
    got = dict(RS.list_snapshots(tmp_path))
    assert sorted(got) == [0, 1]
    assert got[0].parent.name == "sub_checkpoints"         # duplicate gen resolved to sub_checkpoints/


def test_latest_snapshot_none_when_empty(tmp_path):
    assert RS.latest_snapshot(tmp_path) is None
    assert RS.list_snapshots(tmp_path) == []


def test_resolve_resume_modes(tmp_path):
    _make_snaps(tmp_path, [0, 1, 2])
    # explicit path wins
    assert RS.resolve_resume(tmp_path, resume_gen="latest", explicit_path="/x/y.pt") == Path("/x/y.pt")
    # None -> fresh start
    assert RS.resolve_resume(tmp_path, resume_gen=None) is None
    # latest / blank / last -> highest
    assert RS.resolve_resume(tmp_path, resume_gen="latest").name == "snap_gen2.pt"
    assert RS.resolve_resume(tmp_path, resume_gen="").name == "snap_gen2.pt"
    assert RS.resolve_resume(tmp_path, resume_gen="last").name == "snap_gen2.pt"
    # explicit generation -> that file (int or numeric string)
    assert RS.resolve_resume(tmp_path, resume_gen=1).name == "snap_gen1.pt"
    assert RS.resolve_resume(tmp_path, resume_gen="1").name == "snap_gen1.pt"


def test_resolve_resume_missing_gen_returns_path_for_caller_to_validate(tmp_path):
    _make_snaps(tmp_path, [0, 1])
    p = RS.resolve_resume(tmp_path, resume_gen=9)           # gen 9 doesn't exist
    assert p.name == "snap_gen9.pt" and not p.exists()      # path returned; caller errors loudly


def test_prune_snapshots_keeps_last_k(tmp_path):
    _make_snaps(tmp_path, [0, 1, 2, 3, 4])
    deleted = RS.prune_snapshots(tmp_path, keep=2)
    assert sorted(p.name for p in deleted) == ["snap_gen0.pt", "snap_gen1.pt", "snap_gen2.pt"]
    assert [g for g, _ in RS.list_snapshots(tmp_path)] == [3, 4]   # last 2 kept


def test_prune_snapshots_keep_all_when_zero(tmp_path):
    _make_snaps(tmp_path, [0, 1, 2])
    assert RS.prune_snapshots(tmp_path, keep=0) == []
    assert len(RS.list_snapshots(tmp_path)) == 3                   # nothing pruned
