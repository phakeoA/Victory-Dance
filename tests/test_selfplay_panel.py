"""2026-10-01: the fixed PANEL yardstick — every generation also plays era2 + held-out clones on random teams,
in the SAME parallel eval batch; the run stops when generations in a row fail to beat it."""
import asyncio

import pytest

from v_dance.selfplay import mp_eval as ME
from v_dance.selfplay.gate import (GenConfig, GenerationHistory, GenerationRecord, best_panel_generation,
                                   panel_passed, panel_stop)
from v_dance.selfplay.generation import parse_panel, run_generation
from v_dance.selfplay.league import OpponentLeague

TEAMS = [f"t{i}" for i in range(12)]
PANEL = {"era2": "era2.pt", "a02": "a02.pt", "a09": "a09.pt"}


def test_panel_specs_cover_every_opponent_on_the_same_pairings():
    specs = ME.build_panel_specs(PANEL, TEAMS, 100, matchup_seed=3, gen=7, uid_start=40)
    by = {}
    for s in specs:
        by.setdefault(s.kind, []).append(s)
    assert set(by) == {"panel:era2", "panel:a02", "panel:a09"}
    for kind, ss in by.items():
        assert sum(s.n for s in ss) == 100 and all(s.n % 4 == 0 and s.gen == 7 for s in ss)
        assert {s.opp_ckpt for s in ss} == {PANEL[kind[len("panel:"):]]}
    pairs = {k: [(s.team_a, s.team_b) for s in ss] for k, ss in by.items()}
    assert pairs["panel:era2"] == pairs["panel:a02"] == pairs["panel:a09"]      # comparable opponents
    uids = [s.uid for s in specs]
    assert len(set(uids)) == len(uids) and min(uids) == 41


def test_panel_rides_the_same_parallel_batch_as_the_gauntlet(monkeypatch):
    import v_dance.play.model_io as model_io
    loaded = []
    monkeypatch.setattr(model_io, "load_bc_policy", lambda p, *a, **k: loaded.append(p))
    seen = {}

    def submit(payloads, worker_fn=None):
        seen["payloads"] = payloads
        return [{"acc": {"random": (5, 10), "panel:era2": (30, 50)}, "source": {}}]
    res, _ = ME.eval_with_pool("cand.pt", opponents=["random"], team_pool=TEAMS, battles_per_opponent=20,
                               team_chooser=None, submit_fn=submit, n_procs=4, async_per_proc=2,
                               panel=PANEL, panel_battles=40)
    batches = [p[3] for p in seen["payloads"]]
    assert len(batches) == 4                                              # fanned across all 4 processes
    kinds_per_batch = [{s.kind for s in b} for b in batches]
    assert all(any(k.startswith("panel:") for k in ks) for ks in kinds_per_batch)   # panel games in EVERY process
    assert set(loaded) >= set(PANEL.values()) | {"cand.pt"}              # panel ckpts validated before fan-out
    assert res["panel:era2"] == (30, 50) and res["panel:a02"] == (0, 0)  # a dropped opponent reads 0/0


def test_the_worker_plays_a_panel_chunk_against_its_checkpoint():
    made = []

    def build(candidate, prev_best, team_chooser, spec, live_dir, save_replays, port=None):
        made.append(spec.opp_ckpt)
        return _P(), _P()
    spec = ME.EvalSpec("panel:era2", "t1", "t2", 4, 1, 0, opp_ckpt="era2.pt")

    async def fake_play(a, b, n, battle_timeout=None, label=None):
        return 3, 4
    orig = ME.play_pairing
    ME.play_pairing = fake_play
    try:
        out = asyncio.run(ME._eval_specs("c.pt", None, None, [spec], battle_timeout=None, async_workers=2,
                                         build_players=build))
    finally:
        ME.play_pairing = orig
    assert made == ["era2.pt"] and out["acc"]["panel:era2"] == (3, 4)


class _P:
    _forfeited_tags = set()

    async def stop_listening(self):
        return None


def _rec(g, panel):
    return GenerationRecord(generation=g, n_trajectories=1, scripted_wins=0, scripted_games=0, model_elo=None,
                            verdict="hold", promoted=False, panel=panel)


def test_panel_pass_needs_a_visible_edge():
    assert panel_passed({"era2": (55, 100), "a02": (55, 100), "a09": (55, 100)})       # 55 % of 300
    assert not panel_passed({"era2": (53, 100), "a02": (53, 100), "a09": (54, 100)})   # 53.3 % = noise
    assert not panel_passed(None) and not panel_passed({"era2": (0, 0)})


def test_panel_stop_counts_failures_since_the_last_pass():
    lo, hi = {"era2": (145, 300)}, {"era2": (180, 300)}
    h = GenerationHistory()
    for g in range(4):
        h.add(_rec(g, lo))
    assert panel_stop(h, 5) is None
    h.add(_rec(4, lo))
    assert "5 generations in a row" in panel_stop(h, 5) and "none ever did" in panel_stop(h, 5)
    h.add(_rec(5, hi))
    assert panel_stop(h, 5) is None                                        # a pass resets the clock
    for g in range(6, 11):
        h.add(_rec(g, lo))
    assert "last one that did: gen 5" in panel_stop(h, 5)
    assert panel_stop(h, 0) is None
    assert best_panel_generation(h)[:2] == (5, 0.6)


def test_panel_round_trips_and_old_records_load():
    r = _rec(2, {"era2": (60, 100)})
    back = GenerationRecord.from_obj(r.to_obj())
    assert back.panel == {"era2": (60, 100)}
    old = r.to_obj()
    old.pop("panel")
    assert GenerationRecord.from_obj(old).panel is None


def test_run_generation_records_the_panel():
    class _Tr:
        def warmup_critic(self, *a): return {}
        def rebase_values(self, *a): return 0
        def ppo_update(self, *a): return {"loss": 0.1, "halted": False}
    h = GenerationHistory()
    rep = run_generation(object(), _Tr(), OpponentLeague(latest_path="bc.pt"), h, cfg=GenConfig(warmup_updates=1),
                         collect_fn=lambda ac, lg, gen: ([object()] * 4, {"model": 10}),
                         save_fn=lambda ac, gen: f"gen{gen}.pt",
                         eval_fn=lambda p, pb=None: ({"random": (90, 100), "panel:era2": (60, 100),
                                                     "panel:a02": (52, 100)}, 1500.0),
                         restore_fn=lambda ac, p: None)
    assert rep["panel"] == {"era2": (60, 100), "a02": (52, 100)} and h.records[0].panel == rep["panel"]


def test_parse_panel(tmp_path):
    f = tmp_path / "era2.pt"
    f.write_bytes(b"x")
    assert parse_panel([f"era2={f}"]) == {"era2": str(f)}
    assert parse_panel(None) == {}
    with pytest.raises(SystemExit):
        parse_panel(["era2"])
    with pytest.raises(SystemExit):
        parse_panel([f"era2={tmp_path / 'missing.pt'}"])


def test_no_mirror_revert_turns_a_mirror_collapse_into_a_hold():
    from v_dance.selfplay.gate import GateConfigV2, promotion_gate_v2
    kw = dict(scripted_wins=95, scripted_games=100, high_water=0.9, mirror_wins=100, mirror_games=360,
              h2h_history=[0.28], have_champion=True)
    assert promotion_gate_v2(**kw, cfg=GateConfigV2())[0] == "revert"
    v, st = promotion_gate_v2(**kw, cfg=GateConfigV2(mirror_revert=False))
    assert v == "hold" and not st["mirror"]["collapsed"]


def test_a_panel_pass_is_kept_before_a_revert_deletes_it(tmp_path):
    from v_dance.selfplay.gate import GateConfigV2
    class _Tr:
        def warmup_critic(self, *a): return {}
        def rebase_values(self, *a): return 0
        def ppo_update(self, *a): return {"loss": 0.1, "halted": False}
    h, lg, kept = GenerationHistory(), OpponentLeague(latest_path="bc.pt"), []
    scripted = iter([(95, 100), (40, 100)])                       # gen 1 = a scripted COLLAPSE -> revert

    def save_fn(ac, gen):
        p = tmp_path / f"gen{gen}.pt"
        p.write_bytes(b"w")
        return str(p)
    fns = dict(collect_fn=lambda ac, lg_, gen: ([object()] * 4, {"model": 10}), save_fn=save_fn,
               eval_fn=lambda p, pb=None: ({"random": next(scripted), "panel:era2": (180, 300)}, 1500.0),
               restore_fn=lambda ac, p: None, cleanup_fn=lambda ev: None,
               keep_fn=lambda cand, gen: kept.append((gen, (tmp_path / f"gen{gen}.pt").exists())))
    cfg = GenConfig(warmup_updates=1, gate_v2=GateConfigV2())
    run_generation(object(), _Tr(), lg, h, cfg=cfg, **fns)
    rep = run_generation(object(), _Tr(), lg, h, cfg=cfg, **fns)
    assert rep["verdict"] == "revert" and not (tmp_path / "gen1.pt").exists()     # the revert still cleans up
    assert kept == [(0, True), (1, True)]                         # ...but only AFTER the pass was kept


def test_own_team_panel_puts_the_candidate_on_its_team_every_game():
    specs = ME.build_panel_specs({"era2": "e.pt"}, TEAMS, 40, matchup_seed=1, own_team="t3")
    assert sum(s.n for s in specs) == 40
    assert all(s.team_a == "t3" and s.team_b != "t3" for s in specs)          # the ladder shape, no mirrors
