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


def test_build_eval_specs_stamps_generation_and_defaults_zero():
    """22d: every spec carries the generation so the worker account names fold it in; the default
    is gen 0 (so legacy callers / the picklability test read a defined value)."""
    specs = ME.build_eval_specs(["prev_best"], ["A", "B"], 4, gen=7)
    assert specs and all(s.gen == 7 for s in specs)
    assert all(s.gen == 0 for s in ME.build_eval_specs(["prev_best"], ["A", "B"], 4))


def test_eval_specs_account_names_disjoint_across_generations():
    """The derived (model, opp) names for the SAME uids must be disjoint between two gens — the
    actual collision the live run hit (`already a challenge between you and OPprev…`)."""
    from v_dance.play.parallel_battles import eval_account_names, gen_salt
    def names(gen):
        return {n for s in ME.build_eval_specs(["prev_best", "random"], ["A", "B"], 4, gen=gen)
                for n in eval_account_names(s.kind, s.uid, salt=gen_salt(s.gen))}
    assert names(7).isdisjoint(names(8))


def test_eval_with_pool_threads_generation_onto_specs(monkeypatch):
    """generation reaches the EvalSpecs shipped to the workers (so the worker builds gen-keyed names)."""
    import v_dance.play.model_io as MIO
    monkeypatch.setattr(MIO, "load_bc_policy", lambda p: None)
    captured = {}

    def submit(payloads, worker_fn=None):
        captured["specs"] = [s for p in payloads for s in p[3]]   # payload[3] = the EvalSpec batch
        return [{"acc": {}, "source": {}} for _ in payloads]

    ME.eval_with_pool("cand.pt", opponents=["random"], team_pool=["A", "B"],
                      battles_per_opponent=4, team_chooser=None, submit_fn=submit,
                      n_procs=2, generation=12)
    assert captured["specs"] and all(s.gen == 12 for s in captured["specs"])


def test_eval_with_pool_assigns_pool_ports_round_robin(monkeypatch):
    """22f.3: eval worker batches spread round-robin across the pool servers (batch i -> ports[i%K])."""
    import v_dance.play.model_io as MIO
    monkeypatch.setattr(MIO, "load_bc_policy", lambda p: None)
    captured = {}

    def submit(payloads, worker_fn=None):
        captured["ports"] = [p[8] for p in payloads]    # index 8 = the assigned server port
        return [{"acc": {}, "source": {}} for _ in payloads]

    ME.eval_with_pool("cand.pt", opponents=["random", "max_damage", "heuristic"],
                      team_pool=["A", "B", "C"], battles_per_opponent=30, team_chooser=None,
                      submit_fn=submit, n_procs=3, ports=[8000, 8001])
    assert captured["ports"] == [8000, 8001, 8000]      # 3 batches over 2 servers, round-robin


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


def _build(candidate, prev_best, tc, spec, live_dir=None, save_replays=False, port=None):
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
    def build(candidate, prev_best, tc, spec, live_dir=None, save_replays=False, port=None):
        if spec.uid == 2:
            raise RuntimeError("team resolve failed")
        return _build(candidate, prev_best, tc, spec)
    res = asyncio.run(ME._eval_specs(
        "c.pt", None, None,
        [EvalSpec("random", "A", "B", 2, 1), EvalSpec("random", "A", "B", 2, 2),
         EvalSpec("random", "A", "B", 2, 3)],
        battle_timeout=None, async_workers=2, build_players=build))
    assert res["acc"]["random"] == (4, 4)          # uids 1 + 3 ran; uid 2 skipped, not all-dropped


# ── backstop-forfeit discount (gate-bias fix) ─────────────────────────────────
def _forfeit_player(wr, forfeits):
    """A fake eval player whose backstop loop-guard forfeits are recorded in _forfeited_tags
    (one distinct battle tag each), exactly like the real LiveVGCBase player."""
    p = _FakeEvalPlayer(wr=wr)
    p._forfeited_tags = {f"ff-{i}" for i in range(forfeits)}
    return p


def test_eval_specs_discounts_opp_forfeits_from_both_numerator_and_denominator():
    # The champion (opp) backstop-forfeits 2 of the candidate's 4 poke-env "wins". Those are
    # engineering escapes, not skill, and otherwise push the promotion gate toward a spurious
    # PROMOTE — so they must drop from BOTH the numerator and the denominator.
    def build(candidate, prev_best, tc, spec, live_dir=None, save_replays=False, port=None):
        return _forfeit_player(1.0, 0), _forfeit_player(0.0, 2)   # model wins all 4; opp forfeited 2
    res = asyncio.run(ME._eval_specs(
        "cand.pt", None, None, [EvalSpec("prev_best", "A", "B", 4, 1)],
        battle_timeout=None, async_workers=1, build_players=build))
    assert res["acc"]["prev_best"] == (2, 2)       # 4 raw − 2 opp forfeits from wins AND finishes


def test_eval_specs_discounts_candidate_forfeits_from_denominator_only():
    # The CANDIDATE backstop-forfeits 1 of 4 battles (a spurious loss, never a win): drop it from
    # the denominator only so it does not deflate the candidate's rate against the gate.
    def build(candidate, prev_best, tc, spec, live_dir=None, save_replays=False, port=None):
        return _forfeit_player(0.75, 1), _forfeit_player(0.0, 0)   # 3/4 won; model forfeited the loss
    res = asyncio.run(ME._eval_specs(
        "cand.pt", None, None, [EvalSpec("prev_best", "A", "B", 4, 1)],
        battle_timeout=None, async_workers=1, build_players=build))
    assert res["acc"]["prev_best"] == (3, 3)       # wins 3 unchanged; finished 4 − 1 own forfeit


def test_eval_specs_forfeit_subtraction_is_clamped_non_negative():
    # Edge: an opp forfeit is tagged at DECISION time but its win is never server-acked (the battle
    # wedged and was stall-abandoned), so w(0) < opp_ff(1). The subtraction must clamp to 0, never
    # feed a NEGATIVE wins/games tally into the promotion gate.
    def build(candidate, prev_best, tc, spec, live_dir=None, save_replays=False, port=None):
        return _forfeit_player(0.0, 0), _forfeit_player(0.0, 1)   # 0 acked wins; opp tagged 1 forfeit
    res = asyncio.run(ME._eval_specs(
        "cand.pt", None, None, [EvalSpec("prev_best", "A", "B", 1, 1)],
        battle_timeout=None, async_workers=1, build_players=build))
    assert res["acc"]["prev_best"] == (0, 0)       # max(0, 0−1)=0, not −1


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
