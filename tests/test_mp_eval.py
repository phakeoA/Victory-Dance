"""Offline tests for multiprocess eval (task #19, Fix A).

No server/spawn: covers EvalSpec picklability, the gauntlet plan (`build_eval_specs`, incl. the
prev_best mirror override + unique uids), `merge_eval_results`, the per-kind tally DISPATCH
(`_eval_specs` with an injected fake player factory), and the `eval_with_pool` orchestration
(injected submit_fn + mocked load check)."""
from __future__ import annotations

import asyncio
import pickle
import types

import pytest

from v_dance.selfplay import mp_eval as ME
from v_dance.selfplay.mp_eval import EvalSpec


def test_eval_spec_is_picklable():
    s = EvalSpec("prev_best", "A", "B", 24, 3)
    assert pickle.loads(pickle.dumps(s)) == s


def test_build_eval_specs_mirror_override_and_unique_uids():
    specs = ME.build_eval_specs(["random", "prev_best"], ["A", "B"], 4, mirror_battles=10)
    randoms = [s for s in specs if s.kind == "random"]
    prevs = [s for s in specs if s.kind == "prev_best"]
    assert sum(s.n for s in randoms) == 4          # scripted count
    assert sum(s.n for s in prevs) == 10           # mirror override (not 4)
    assert [s.uid for s in specs] == list(range(1, len(specs) + 1))   # globally unique


def test_merge_eval_results_sums_per_kind_and_skips_none():
    r1 = {"acc": {"random": (3, 4)}, "source": {"model": 5}}
    r2 = {"acc": {"random": (2, 4), "max_damage": (1, 2)}, "source": {"model": 3}}
    results, source = ME.merge_eval_results([r1, None, r2])
    assert results == {"random": (5, 8), "max_damage": (1, 2)}
    assert source == {"model": 8}


# ── _eval_specs dispatch (injected fake players; no server) ───────────────────
class _FakeEvalPlayer:
    def __init__(self, wr=0.0):
        self.n_won_battles = 0
        self.n_finished_battles = 0
        self._wr = wr
        self.closed = False
        self._source_counts = {"model": 1}
        self._tp_source = {"model": 1}
        async def _stop():
            return None
        self.ps_client = types.SimpleNamespace(stop_listening=_stop)

    async def battle_against(self, opp, n_battles):
        self.n_finished_battles += n_battles
        self.n_won_battles += int(round(self._wr * n_battles))

    def close(self):
        self.closed = True


def _build(candidate, prev_best, tc, spec):
    return _FakeEvalPlayer(wr=1.0), _FakeEvalPlayer(wr=0.0)   # model wins every battle


def test_eval_specs_tallies_wins_per_opponent_kind():
    res = asyncio.run(ME._eval_specs(
        "cand.pt", None, None,
        [EvalSpec("random", "A", "B", 2, 1), EvalSpec("random", "B", "A", 2, 2),
         EvalSpec("max_damage", "A", "B", 2, 3)],
        battle_timeout=None, async_workers=2, build_players=_build))
    assert res["acc"]["random"] == (4, 4)          # 2 specs × 2 battles, all won
    assert res["acc"]["max_damage"] == (2, 2)
    assert res["source"]["model"] == 3             # per-chunk model source tally
    assert res["source"]["tp_model"] == 3          # team-preview tally surfaced


def test_eval_specs_build_error_skips_only_that_chunk():
    def build(candidate, prev_best, tc, spec):
        if spec.uid == 2:
            raise RuntimeError("team resolve failed")
        return _build(candidate, prev_best, tc, spec)
    res = asyncio.run(ME._eval_specs(
        "c.pt", None, None,
        [EvalSpec("random", "A", "B", 2, 1), EvalSpec("random", "A", "B", 2, 2),
         EvalSpec("random", "A", "B", 2, 3)],
        battle_timeout=None, async_workers=2, build_players=build))
    assert res["acc"]["random"] == (4, 4)          # uids 1 + 3 ran; uid 2 skipped, not all-dropped


# ── eval_with_pool orchestration (injected submit_fn + mocked load) ───────────
def test_eval_with_pool_validates_partitions_and_aggregates(monkeypatch):
    import v_dance.play.model_io as MIO
    monkeypatch.setattr(MIO, "load_bc_policy", lambda p: None)   # skip the real ckpt load
    captured = {}

    def submit(payloads, worker_fn=None):
        captured["worker_fn"] = worker_fn
        captured["payloads"] = payloads
        return [{"acc": {"random": (2, 2)}, "source": {"model": 1}} for _ in payloads]

    results, source = ME.eval_with_pool(
        "cand.pt", opponents=["random", "prev_best"], team_pool=["A", "B"],
        battles_per_opponent=4, team_chooser="tc.pt", submit_fn=submit, prev_best="champ.pt",
        mirror_battles=8, matchup_seed=0, n_procs=2)

    assert captured["worker_fn"] is ME.eval_worker            # the pool runs the EVAL worker
    payloads = captured["payloads"]
    assert all(p[0] == "cand.pt" and p[1] == "champ.pt" and p[2] == "tc.pt" for p in payloads)
    assert results["random"][1] >= 2                          # aggregated finished games
    assert "prev_best" in results                             # fully-absent opponent still keyed (0/0 ok)


def test_eval_with_pool_fails_loud_on_bad_candidate(monkeypatch):
    import v_dance.play.model_io as MIO
    def boom(p):
        raise RuntimeError("checkpoint won't load")
    monkeypatch.setattr(MIO, "load_bc_policy", boom)
    with pytest.raises(RuntimeError):
        ME.eval_with_pool("bad.pt", opponents=["random"], team_pool=["A", "B"],
                          battles_per_opponent=4, team_chooser=None, submit_fn=lambda *a, **k: [])
