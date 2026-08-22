"""P2.3 — HoF eval plumbing (hof_eval) aggregation, offline.

hof_eval plays the candidate vs each suspect via one run_gauntlet 'prev_best' mirror call and
returns the per-suspect (snapshot_id, wins, games) that hall_of_fame_gate consumes. The gauntlet
runner + the checkpoint loader are injected/mocked so the aggregation + the skip-on-unloadable
guard unit-test WITHOUT a Showdown server.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from v_dance.selfplay.generation import hof_eval
from v_dance.selfplay.gate import hall_of_fame_gate, HoFConfig


class _FakeSuspect:
    def __init__(self, sid, path):
        self.snapshot_id = sid
        self.path = path


def _patch_loader(monkeypatch, bad=()):
    import v_dance.play.model_io as model_io

    def fake_load(path):
        if any(b in str(path) for b in bad):
            raise RuntimeError("corrupt checkpoint")
        return object()
    monkeypatch.setattr(model_io, "load_bc_policy", fake_load)


def test_hof_eval_aggregates_per_suspect_and_feeds_the_gate(monkeypatch):
    _patch_loader(monkeypatch)
    canned = {"a/gen0.pt": (40, 60), "a/gen1.pt": (18, 60), "a/gen2.pt": (33, 60)}

    async def fake_gauntlet(**kw):
        key = Path(kw["prev_best_ckpt"]).as_posix()
        assert kw["opponents"] == ["prev_best"] and kw["manage_server"] is False
        return {"prev_best": canned[key]}, {}

    suspects = [_FakeSuspect(f"gen{i}", f"a/gen{i}.pt") for i in range(3)]
    results = asyncio.run(hof_eval("cand.pt", suspects, team_pool=["t1", "t2"],
                                   team_chooser="tc.pt", games_per_snapshot=60,
                                   gauntlet_fn=fake_gauntlet))
    assert results == [("gen0", 40, 60), ("gen1", 18, 60), ("gen2", 33, 60)]
    # flows straight into the gate — the hidden 0.30 counter (gen1) is vetoed
    v, st = hall_of_fame_gate(results, cfg=HoFConfig(min_pool=3, min_games_per_snap=40))
    assert v == "reject" and st["worst_snapshot_id"] == "gen1"


def test_hof_eval_skips_unloadable_suspect_never_vetoes_on_garbage(monkeypatch):
    _patch_loader(monkeypatch, bad=["gen1"])           # gen1 checkpoint won't load

    async def fake_gauntlet(**kw):
        return {"prev_best": (40, 60)}, {}

    suspects = [_FakeSuspect(f"gen{i}", f"a/gen{i}.pt") for i in range(3)]
    results = asyncio.run(hof_eval("cand.pt", suspects, team_pool=["t1"],
                                   team_chooser="tc.pt", games_per_snapshot=60,
                                   gauntlet_fn=fake_gauntlet))
    assert [r[0] for r in results] == ["gen0", "gen2"]  # gen1 skipped, NOT a garbage veto


def test_hof_eval_empty_suspects_returns_empty(monkeypatch):
    _patch_loader(monkeypatch)

    async def fake_gauntlet(**kw):                      # should never be called
        raise AssertionError("no suspects → no gauntlet calls")

    results = asyncio.run(hof_eval("cand.pt", [], team_pool=["t1"], team_chooser="tc.pt",
                                   gauntlet_fn=fake_gauntlet))
    assert results == []
    v, st = hall_of_fame_gate(results)                  # thin pool → fail-open skip
    assert v == "skip" and st["reason"] == "thin_pool_skip"
