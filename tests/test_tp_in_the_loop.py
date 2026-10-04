"""The team picker INSIDE self-play (2026-10-03, USER: teach the Rillaboom lesson to both the picker and the battle net).

Stage 1 — the learner picks with a picker + EXPLORES and records the decision (model_io / player / vgc_base /
game_runner / mp_collect); Stage 2 — tp_learning trains the picker on the results; the eval plays each candidate with
ITS picker (mp_eval). Every piece default-OFF = byte-identical (the first-4 heuristic).
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from v_dance.play import model_io as M  # noqa: E402
from v_dance.selfplay import tp_learning as TPL  # noqa: E402

_ROSTER = ["salamence", "sneasler", "indeedee", "tyranitar", "corviknight", "excadrill"]
_OPP = ["rillaboom", "incineroar", "sinistcha", "kingambit", "pelipper", "archaludon"]


# ── a tiny REAL set-head picker on the legacy 46-dim features (no feature_schema → no lockstep guard) ─────────
def _tiny_ckpt(tmp_path: Path, seed: int = 0) -> Path:
    from v_dance.models.teampreview_model import TeamPreviewModel
    torch.manual_seed(seed)
    vocab = {s: i + 1 for i, s in enumerate(_ROSTER + _OPP)}
    cfg = {"vocab_size": len(vocab) + 1, "feat_dim": 46, "emb_dim": 8, "hidden": 16, "dropout": 0.0,
           "use_self_attn": True, "use_cross_attn": True, "attn_heads": 2, "use_teammate_bias": False,
           "use_set_head": True, "use_set_ctx": False, "bring_k": 4, "lead_k": 2}
    m = TeamPreviewModel(vocab_size=cfg["vocab_size"], feat_dim=46, emb_dim=8, hidden=16, dropout=0.0,
                         use_self_attn=True, use_cross_attn=True, attn_heads=2, use_set_head=True)
    with torch.no_grad():                         # non-zero set heads so the set scores are not pure unary sums
        for head in (m.set_pair_mlp, m.set_global_mlp):
            head[-1].weight.normal_(0, 0.1)
    p = tmp_path / "tiny_tp.pt"
    torch.save({"model_state": m.state_dict(), "config": cfg, "vocab": vocab}, p)
    return p


# ── Stage 1: the exploring decode ──────────────────────────────────────────────────────────────────────────
def test_explore_probs_is_softmax_mixed_with_uniform():
    mu = M.explore_probs([1.0, 0.0, -1.0], tau=1.0, eps=0.1)
    sm = np.exp([1.0, 0.0, -1.0]) / np.exp([1.0, 0.0, -1.0]).sum()
    assert np.allclose(mu, 0.9 * sm + 0.1 / 3) and abs(mu.sum() - 1) < 1e-12
    assert np.allclose(M.explore_probs([5.0, 1.0], tau=1.0, eps=0.0), np.exp([5, 1]) / np.exp([5, 1]).sum())


def test_team_order_default_path_records_nothing_and_is_deterministic(tmp_path):
    model, vocab, cfg = M.load_team_chooser(str(_tiny_ckpt(tmp_path)))
    a = M.team_order(model, vocab, cfg, _ROSTER, _OPP, 4)
    b = M.team_order(model, vocab, cfg, _ROSTER, _OPP, 4)
    assert a == b and len(set(a)) == 4 and not M.LAST_TP_EXPLORE      # explore=None: argmax, no record


def test_exploring_decode_samples_from_mu_and_records_it(tmp_path):
    model, vocab, cfg = M.load_team_chooser(str(_tiny_ckpt(tmp_path)))
    rng = np.random.default_rng(1)
    picks = Counter()
    recs = []
    for _ in range(3000):
        order = M.team_order(model, vocab, cfg, _ROSTER, _OPP, 4,
                             explore={"tau": 1.0, "eps": 0.3, "rng": rng})
        r = dict(M.LAST_TP_EXPLORE)
        recs.append(r)
        picks[r["set_idx"]] += 1
        brought = r["subsets"][r["set_idx"]]
        lead = r["pairs"][r["pair_idx"]]
        assert set(order) == set(brought) and set(order[:2]) == set(lead) and set(lead) <= set(brought)
        assert r["pairs"] == [list(q) for q in combinations(sorted(brought), 2)]
    r0 = recs[0]
    assert len(r0["subsets"]) == 15 and r0["of"].shape == (6, 46) and r0["pf"].shape == (6, 46)
    # the recorded behaviour probability of every subset agrees with the empirical frequency
    mu_of = {r["set_idx"]: r["mu_set"] for r in recs}
    for si, c in picks.items():
        assert abs(c / 3000 - mu_of[si]) < 0.03, (si, c / 3000, mu_of[si])
    # p_* = the picker's own softmax (eps 0); mu_* mixes in eps-uniform
    assert all(r["mu_set"] >= 0.3 / 15 - 1e-12 for r in recs)
    assert all(abs(r["mu_set"] - (0.7 * r["p_set"] + 0.3 / 15)) < 1e-9 for r in recs)
    assert all(abs(r["mu_pair"] - (0.7 * r["p_pair"] + 0.3 / 6)) < 1e-9 for r in recs)


# ── the record's path: player → vgc_base capture → EpisodeMeta (in-memory only) ────────────────────────────
def test_teampreview_capture_carries_the_pending_record():
    from v_dance.play.vgc_base import VGCPlayerBase

    class Mon:
        def __init__(self, s):
            self.species = s

    class Battle:
        battle_tag = "battle-x-1"
        teampreview_team = [Mon(s) for s in _ROSTER]
        team = {}
        max_team_size = 4
        teampreview_opponent_team = [Mon(s) for s in _OPP]

    class Me:
        _tp_learn_pending = {"battle-x-1": {"set_idx": 3}}
        _tp_decision: dict = {}

        def _choose_team_order(self, battle, team, n):
            return [0, 1, 2, 3]

    me = Me()
    assert VGCPlayerBase.teampreview(me, Battle()) == "/team 1234"
    assert me._tp_decision["battle-x-1"]["learn"] == {"set_idx": 3}
    assert me._tp_learn_pending == {}                                    # popped: it can never leak
    me2 = Me()
    me2._tp_learn_pending, me2._tp_decision = {}, {}
    VGCPlayerBase.teampreview(me2, Battle())
    assert "learn" not in me2._tp_decision["battle-x-1"]                 # no exploring picker → no key


def test_finalize_puts_the_record_on_the_meta_but_never_serialises_it():
    from v_dance.rl.collector import TrajectoryCollector
    from v_dance.selfplay.game_runner import finalize_trajectory
    c = TrajectoryCollector("battle-x-1", "p1")
    c.add_step(state=np.zeros(4, dtype=np.float32), action_s0=0, action_s1=0, decision_type="turn", turn=1)
    t = finalize_trajectory(c, won=True, terminal_type="win", own_team=_ROSTER, n_turns=1,
                            tp_learn={"set_idx": 2})
    assert t.meta.tp_learn == {"set_idx": 2}
    assert "tp_learn" not in t.meta.to_obj()


# ── worker + eval plumbing ─────────────────────────────────────────────────────────────────────────────────
def test_chunk_specs_carry_the_learner_picker_and_players_get_it(monkeypatch):
    from v_dance.selfplay import mp_collect as MP
    import v_dance.selfplay.game_runner as GR
    import v_dance.play.run_local_battle as R
    made = []

    class FakeSP:
        def __init__(self, ac, **kw):
            self.kw = kw
            made.append(self)

    monkeypatch.setattr(GR, "SelfPlayVGCPlayer", FakeSP)
    monkeypatch.setattr(R, "load_team", lambda p: "TEAM")
    monkeypatch.setattr(R, "resolve_team_path", lambda t: t)
    spec = MP.ChunkSpec("A", "B", 2, "latest", None, None, 1, 3)
    spec.learner_tp, spec.learner_tp_tau, spec.learner_tp_eps = "tp.pt", 1.0, 0.15
    our, opp = MP._build_players_real("AC", spec, 1.0, 0, "default_tp.pt")
    for p in (our, opp):                                                  # BOTH recording players pick + explore
        assert p.kw["team_chooser_path"] == "tp.pt"
        assert p._tp_explore["eps"] == 0.15 and p._tp_explore["rng"] is not None
    plain = MP.ChunkSpec("A", "B", 2, "latest", None, None, 2, 3)
    our2, _ = MP._build_players_real("AC", plain, 1.0, 0, "default_tp.pt")
    assert "team_chooser_path" not in our2.kw and not hasattr(our2, "_tp_explore")   # default: first-4


def test_build_chunk_specs_stamps_every_chunk():
    from v_dance.selfplay import mp_collect as MP

    class League:
        def sample(self, rng):
            return ("latest",)

    specs = MP.build_chunk_specs(League(), ["A", "B", "C"], 20, chunk_size=5, learner_tp="tp.pt",
                                 learner_tp_tau=0.8, learner_tp_eps=0.2)
    assert specs and all(s.learner_tp == "tp.pt" and s.learner_tp_tau == 0.8 and s.learner_tp_eps == 0.2
                         for s in specs)
    assert all(s.learner_tp is None for s in MP.build_chunk_specs(League(), ["A", "B", "C"], 20, chunk_size=5))


def test_eval_candidate_plays_with_its_picker(monkeypatch):
    import v_dance.play.run_local_battle as R
    from v_dance.selfplay import mp_eval as ME
    monkeypatch.delenv("VD_EVAL_CANDIDATE_TP", raising=False)

    class P:
        pass

    def fake_make_player(name, team, *, model_path=None, team_chooser_path=None, **kw):
        p = P()
        p.tc = team_chooser_path
        return p

    monkeypatch.setattr(R, "make_player", fake_make_player)
    monkeypatch.setattr(R, "load_team", lambda p: "TEAM")
    monkeypatch.setattr(R, "resolve_team_path", lambda t: t)
    spec = ME.EvalSpec(ME.PANEL_PREFIX + "era2", "A", "B", 4, 1, 0, opp_ckpt="era2.pt", cand_tp="tp_gen5.pt")
    cand, opp = ME._build_eval_players_real("gen5.pt", None, "shared_tp.pt", spec)
    assert cand.tc == "tp_gen5.pt" and opp.tc == "shared_tp.pt"
    spec.cand_tp = None
    cand, _ = ME._build_eval_players_real("gen5.pt", None, "shared_tp.pt", spec)
    assert cand.tc == "shared_tp.pt"


# ── Stage 2: the picker learns ─────────────────────────────────────────────────────────────────────────────
def test_loo_baseline_excludes_the_record_itself():
    assert TPL.loo_baselines([1, -1, 1, 1], ["a", "a", "a", "b"]) == [0.0, 1.0, 0.0, pytest.approx(1 / 3)]
    assert TPL.loo_baselines([1], ["a"]) == [0.0]


def test_records_from_trajectories_filters_and_keys():
    class Meta:
        def __init__(self, rec, won, trainable=True, opp=("b", "a")):
            self.tp_learn, self.won, self.is_trainable = rec, won, trainable
            self.own_team, self.opp_team = ["x", "y"], list(opp)

    class T:
        def __init__(self, m):
            self.meta = m

    ts = [T(Meta({"set_idx": 1}, True)), T(Meta({"set_idx": 2}, None)), T(Meta(None, True)),
          T(Meta({"set_idx": 3}, False, trainable=False)), T(Meta({"set_idx": 4}, False, opp=("a", "b")))]
    recs = TPL.records_from_trajectories(ts)
    assert [r["set_idx"] for r in recs] == [1, 4] and recs[0]["won"] is True and recs[1]["won"] is False
    assert recs[0]["key"] == recs[1]["key"] == (("x", "y"), ("a", "b"))   # opponent roster key is order-free


def _synthetic_records(model, vocab, cfg, good: int, n: int, rng, eps=0.3):
    """Sample previews from the exploring decode; the bring set `good` always WINS, every other set LOSES."""
    recs = []
    for _ in range(n):
        M.team_order(model, vocab, cfg, _ROSTER, _OPP, 4, explore={"tau": 1.0, "eps": eps, "rng": rng})
        r = dict(M.LAST_TP_EXPLORE)
        recs.append({**r, "won": r["set_idx"] == good, "key": (tuple(_ROSTER), tuple(sorted(_OPP))),
                     "own_team": list(_ROSTER), "opp_team": list(_OPP)})
    return recs


def _p_set(model, rec, idx) -> float:
    with torch.no_grad():
        return _p_set_raw(model, rec, idx)


def _p_set_raw(model, rec, idx) -> float:
    lps, _ = TPL.TPLearner._logps(model, *[x for x in _rec_tensors(rec)][:6], rec["subsets"],
                                  *_rec_tensors(rec)[6:8], _rec_tensors(rec)[8])
    return float(lps.exp()[0, idx])


def _rec_tensors(rec):
    oi = torch.as_tensor([rec["oi"]]); pi = torch.as_tensor([rec["pi"]])
    of = torch.as_tensor(rec["of"][None]); pf = torch.as_tensor(rec["pf"][None])
    pa = torch.as_tensor([[q[0] for q in rec["pairs"]]]); pb = torch.as_tensor([[q[1] for q in rec["pairs"]]])
    return oi, pi, of, pf, None, {}, pa, pb, torch.as_tensor([float(rec["tau"])])


def test_the_picker_learns_the_winning_set_and_stays_anchored(tmp_path):
    ck = _tiny_ckpt(tmp_path)
    cfg = TPL.TPLearnConfig(lr=3e-3, epochs=4, batch=64, kl_coef=0.05, ent_coef=0.0, seed=0)
    L = TPL.TPLearner(ck, cfg=cfg)
    model, vocab, mcfg = M.load_team_chooser(str(ck))
    rng = np.random.default_rng(0)
    first = _synthetic_records(model, vocab, mcfg, good=0, n=1, rng=rng)[0]
    # make subset 9 the winner: a set the start picker does NOT prefer
    p0 = _p_set(L.model, first, 9)
    for gen in range(6):
        recs = _synthetic_records(L.model, vocab, mcfg, good=9, n=300, rng=rng)
        st = L.update(recs)
        assert st["n"] == 300 and not st.get("skipped")
    p1 = _p_set(L.model, first, 9)
    assert p1 > p0 + 0.15, (p0, p1)                                       # it learned the winning set
    assert _p_set(L.anchor, first, 9) == pytest.approx(p0)                 # the anchor never moved
    assert st["kl_anchor"] > 0 and 0.0 <= st["clip_frac"] <= 1.0
    out = L.save(tmp_path / "tp" / "tp_gen5.pt", 5)                       # loads back through the guard
    m2, _v, c2 = M.load_team_chooser(out)
    assert _p_set(m2, first, 9) == pytest.approx(p1, abs=1e-6)
    assert TPL.anchor_of(out) == str(ck)


def test_first_update_ratio_starts_at_one(tmp_path):
    """The update starts from the SAME weights that sampled, in eval mode — pi_new / p_old must be 1 before any step."""
    ck = _tiny_ckpt(tmp_path, seed=3)
    L = TPL.TPLearner(ck, cfg=TPL.TPLearnConfig(lr=0.0, epochs=1))
    model, vocab, mcfg = M.load_team_chooser(str(ck))
    recs = _synthetic_records(model, vocab, mcfg, good=0, n=64, rng=np.random.default_rng(5))
    st = L.update(recs)
    assert st["ratio_mean"] == pytest.approx(1.0, abs=1e-5) and st["clip_frac"] == 0.0
    assert st["argmax_moved"] == 0.0 and st["kl_anchor"] == pytest.approx(0.0, abs=1e-7)


def test_learner_refuses_a_picker_without_a_set_head(tmp_path):
    from v_dance.models.teampreview_model import TeamPreviewModel
    m = TeamPreviewModel(vocab_size=5, feat_dim=46, emb_dim=8, hidden=16, dropout=0.0)
    p = tmp_path / "noset.pt"
    torch.save({"model_state": m.state_dict(), "config": {"vocab_size": 5, "feat_dim": 46, "emb_dim": 8,
                                                          "hidden": 16}, "vocab": {}}, p)
    with pytest.raises(ValueError, match="SET-HEAD"):
        TPL.TPLearner(p)


def test_summary_line_reads_cleanly():
    s = TPL.summarize({"n": 10, "win": 0.6, "explored": 0.2, "kl_anchor": 0.01, "entropy": 1.5, "ratio_mean": 1.0,
                       "clip_frac": 0.0, "argmax_moved": 0.1, "argmax_vs_start": 0.2,
                       "bring_change": {"team_games": 10, "top": [("salamence", 20.0, 35.0)]}})
    assert "salamence 20→35%" in s and "KL-to-start 0.010" in s
    assert "SKIPPED" in TPL.summarize({"n": 2, "skipped": "fewer than 8 records"})


# ── the run: fail fast + the preflight check ───────────────────────────────────────────────────────────────
def test_learner_tp_needs_multiprocess_and_tp_learn_needs_a_picker(tmp_path, capsys):
    from v_dance.selfplay.generation import run_live_generations
    with pytest.raises(SystemExit) as ei:
        run_live_generations("no_such.pt", team_pool=["A", "B"], team_chooser=None,
                             archive_dir=tmp_path / "a1", collect_procs=1, learner_tp="tp.pt")
    assert ei.value.code == 2 and "--learner-tp needs --collect-procs >= 2" in capsys.readouterr().err
    with pytest.raises(SystemExit) as ei:
        run_live_generations("no_such.pt", team_pool=["A", "B"], team_chooser=None,
                             archive_dir=tmp_path / "a2", collect_procs=4, tp_learn=True)
    assert ei.value.code == 2 and "--tp-learn needs --learner-tp" in capsys.readouterr().err


def test_preflight_flags_a_generation_whose_picker_did_not_update():
    from v_dance.selfplay import preflight as PF

    class Rec:
        def __init__(self, g):
            self.generation, self.n_trajectories, self.update_stats = g, 5, {"loss": 0.1}
            self.promoted, self.panel, self.hof = g == 1, {}, None

    class H:
        def __init__(self, n):
            self.records = [Rec(g) for g in range(n)]

    fresh = {"history": H(3), "tp": [{"generation": 0, "n": 9}, {"generation": 2, "n": 9}], "tp_path": "x"}
    resumed = {"history": H(4), "tp": [{"generation": 3, "n": 9}], "tp_path": "y"}
    problems, _w, _p = PF.check(fresh, resumed, tp_learn_on=True)
    assert any("fresh gen 1: the team picker did not update" in p for p in problems)
    resumed_ok = {"history": H(4), "tp": [{"generation": 3, "n": 9, "opt_restored": True}]}
    fresh_ok = {"history": H(3), "tp": [{"generation": g, "n": 9} for g in range(3)], "tp_path": "x"}
    problems, _w, _p = PF.check(fresh_ok, resumed_ok, tp_learn_on=True)
    assert not [p for p in problems if "picker" in p]


# ── review fixes (2026-10-03) ──────────────────────────────────────────────────────────────────────────────
def _bare_player():
    import v_dance.play.player as P
    obj = P.VGCPlayer.__new__(P.VGCPlayer)        # skip the networked __init__ (test_teampreview_serve pattern)
    obj._team_chooser = object()
    obj._tc_vocab = {}
    obj._tc_cfg = {"lead_k": 2}
    obj._device = "cpu"
    obj._tp_source = Counter()
    return obj


def _rec(set_idx, pair_idx):
    subsets = [list(s) for s in combinations(range(6), 4)]
    return {"subsets": subsets, "set_idx": set_idx,
            "pairs": [list(q) for q in combinations(subsets[set_idx], 2)], "pair_idx": pair_idx}


@pytest.mark.parametrize("rec_set, rec_pair, explore, parked", [
    (0, 5, True, True),        # subset {0,1,2,3}, leads {2,3} == the submitted [2, 3, 0, 1]
    (1, 0, True, False),       # a record for a DIFFERENT set never trains the picker
    (0, 0, True, False),       # right set, wrong leads
    (0, 5, False, False),      # no exploring picker → nothing parked even if a record is lying around
])
def test_player_parks_only_a_record_that_matches_the_submission(monkeypatch, rec_set, rec_pair, explore, parked):
    import types
    import v_dance.play.player as P
    p = _bare_player()
    if explore:
        p._tp_explore = {"tau": 1.0, "eps": 0.1, "rng": np.random.default_rng(0)}

    def fake_team_order(*a, **k):
        assert k.get("explore") is getattr(p, "_tp_explore", None)
        P._M.LAST_TP_EXPLORE.clear()
        P._M.LAST_TP_EXPLORE.update(_rec(rec_set, rec_pair))
        return [2, 3, 0, 1]

    monkeypatch.setattr(P._M, "team_order", fake_team_order)
    team = [types.SimpleNamespace(species=s) for s in _ROSTER]
    battle = types.SimpleNamespace(teampreview_opponent_team=[types.SimpleNamespace(species=s) for s in _OPP],
                                   battle_tag="b1")
    assert p._choose_team_order(battle, team, 4) == [2, 3, 0, 1]
    assert ("b1" in (getattr(p, "_tp_learn_pending", None) or {})) is parked


def _same(a, b) -> bool:
    return a is not None and Path(a).resolve() == Path(b).resolve()


def test_paired_tp_for_reads_only_a_verified_sidecar(tmp_path):
    import time
    for d in ("checkpoints", "tp", "panel_pass"):
        (tmp_path / d).mkdir()
    for f in ("checkpoints/gen7.pt", "tp/tp_gen7.pt", "panel_pass/gen7.pt", "panel_pass/tp_gen7.pt",
              "checkpoints/gen8.pt", "tp/tp_gen8.pt"):
        (tmp_path / f).write_bytes(b"x")
    ck7, pp7 = tmp_path / "checkpoints" / "gen7.pt", tmp_path / "panel_pass" / "gen7.pt"
    assert TPL.record_pair(ck7, tmp_path / "tp" / "tp_gen7.pt") and TPL.record_pair(pp7, tmp_path / "panel_pass" / "tp_gen7.pt")
    assert _same(TPL.paired_tp_for(ck7), tmp_path / "tp" / "tp_gen7.pt")
    assert _same(TPL.paired_tp_for(pp7), tmp_path / "panel_pass" / "tp_gen7.pt")
    # review D: a tp file merely LYING next to a checkpoint is never paired (gen8 has no sidecar)
    assert TPL.paired_tp_for(tmp_path / "checkpoints" / "gen8.pt") is None
    # ... and a checkpoint OVERWRITTEN by a later run (same name, new content) loses its stale pairing
    time.sleep(0.02)
    ck7.write_bytes(b"a different run's gen7")
    assert TPL.paired_tp_for(ck7) is None
    assert TPL.paired_tp_for("ai_train_scripts/BC_model/checkpoints_attn_era2/battle_base.pt") is None
    assert TPL.record_pair(tmp_path / "checkpoints" / "missing.pt", tmp_path / "tp" / "tp_gen7.pt") is None


def test_past_selves_play_with_their_paired_picker(monkeypatch, tmp_path):
    import v_dance.play.run_local_battle as R
    import v_dance.eval.gauntlet as G
    from v_dance.selfplay import mp_collect as MP
    from v_dance.selfplay import mp_eval as ME
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "tp").mkdir()
    for f in ("checkpoints/gen3.pt", "tp/tp_gen3.pt"):
        (tmp_path / f).write_bytes(b"x")
    snap = str(tmp_path / "checkpoints" / "gen3.pt")
    TPL.record_pair(snap, tmp_path / "tp" / "tp_gen3.pt")
    seen = {}

    def fake_make_player(name, team, *, model_path=None, team_chooser_path=None, **kw):
        seen[name] = team_chooser_path
        return object()

    monkeypatch.setattr(R, "make_player", fake_make_player)
    monkeypatch.setattr(R, "load_team", lambda p: "TEAM")
    monkeypatch.setattr(R, "resolve_team_path", lambda t: t)
    import v_dance.selfplay.game_runner as GR
    monkeypatch.setattr(GR, "SelfPlayVGCPlayer", lambda ac, **kw: object())
    # collection: a league SNAPSHOT gets its pair; a CLONE keeps the shared picker
    MP._build_players_real("AC", MP.ChunkSpec("A", "B", 2, "snapshot", snap, "g3", 1, 9), 1.0, 0, "shared.pt")
    MP._build_players_real("AC", MP.ChunkSpec("A", "B", 2, "clone", "clone.pt", None, 2, 9), 1.0, 0, "shared.pt")
    assert sorted(Path(v).name for v in seen.values()) == ["shared.pt", "tp_gen3.pt"]
    # eval: prev_best gets its pair
    got = {}

    def fake_make_opponent(kind, name, team, model_path=None, team_chooser_path=None, **kw):
        got["tc"] = team_chooser_path
        return object()

    monkeypatch.setattr(G, "_make_opponent", fake_make_opponent)
    monkeypatch.delenv("VD_EVAL_CANDIDATE_TP", raising=False)
    ME._build_eval_players_real("cand.pt", snap, "shared.pt", ME.EvalSpec("prev_best", "A", "B", 2, 1, 0))
    assert _same(got["tc"], tmp_path / "tp" / "tp_gen3.pt")
    ME._build_eval_players_real("cand.pt", "ai_train_scripts/x/battle_base.pt", "shared.pt",
                                ME.EvalSpec("prev_best", "A", "B", 2, 2, 0))
    assert got["tc"] == "shared.pt"


# ── the three closed gaps (2026-10-03, USER: "fix these gaps") ─────────────────────────────────────────────
def test_hof_plays_the_candidate_and_each_past_champion_with_their_own_picker(monkeypatch, tmp_path):
    import asyncio
    import v_dance.play.model_io as model_io
    from v_dance.selfplay.hof import hof_eval
    monkeypatch.setattr(model_io, "load_bc_policy", lambda p: object())
    monkeypatch.setattr(model_io, "load_team_chooser", lambda p: (object(), {}, {}))
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "tp").mkdir()
    for f in ("checkpoints/gen1.pt", "tp/tp_gen1.pt", "checkpoints/gen2.pt"):   # gen2 has NO picker
        (tmp_path / f).write_bytes(b"x")
    TPL.record_pair(tmp_path / "checkpoints" / "gen1.pt", tmp_path / "tp" / "tp_gen1.pt")
    calls = []

    async def fake_gauntlet(**kw):
        calls.append(kw)
        return {"prev_best": (5, 10)}, {}

    class S:
        def __init__(self, sid, path):
            self.snapshot_id, self.path = sid, path

    suspects = [S("gen1", str(tmp_path / "checkpoints" / "gen1.pt")), S("gen2", str(tmp_path / "checkpoints" / "gen2.pt"))]
    asyncio.run(hof_eval("cand.pt", suspects, team_pool=["t"], team_chooser="shared.pt",
                         gauntlet_fn=fake_gauntlet, candidate_team_chooser="tp_cand.pt"))
    assert [Path(c["candidate_team_chooser"]).name for c in calls] == ["tp_cand.pt", "tp_cand.pt"]
    assert _same(calls[0]["opponent_team_chooser"], tmp_path / "tp" / "tp_gen1.pt")
    assert "opponent_team_chooser" not in calls[1]                       # no pair → the shared picker
    calls.clear()                                                         # no picker in the loop: byte-identical
    asyncio.run(hof_eval("cand.pt", suspects[1:], team_pool=["t"], team_chooser="shared.pt", gauntlet_fn=fake_gauntlet))
    assert "candidate_team_chooser" not in calls[0] and "opponent_team_chooser" not in calls[0]


def test_run_gauntlet_gives_each_side_its_picker(monkeypatch):
    import asyncio
    import v_dance.play.run_local_battle as R
    import v_dance.eval.gauntlet as G
    from v_dance.play import parallel_battles as PB
    seen = {}
    monkeypatch.setattr(R, "load_team", lambda p: "TEAM")
    monkeypatch.setattr(R, "resolve_team_path", lambda t: t)
    monkeypatch.setattr(R, "make_player", lambda name, team, **kw: seen.__setitem__("cand", kw["team_chooser_path"]) or object())
    monkeypatch.setattr(G, "_make_opponent", lambda kind, name, team, **kw: seen.__setitem__("opp", kw["team_chooser_path"]) or object())

    async def fake_play(*a, **k):
        return 1, 2

    async def fake_close(*a):
        return None

    monkeypatch.setattr(PB, "play_pairing", fake_play)
    monkeypatch.setattr(PB, "close_players", fake_close)
    monkeypatch.setattr(PB, "discount_forfeits", lambda w, f, *a: (w, f))
    run = lambda **kw: asyncio.run(G.run_gauntlet(opponents=["prev_best"], team_pool=["A", "B"], battles_per_opponent=2,
                                                  ckpt=Path("c.pt"), team_chooser=Path("shared.pt"),
                                                  prev_best_ckpt=Path("p.pt"), manage_server=False, **kw))
    run(candidate_team_chooser=Path("tp_c.pt"), opponent_team_chooser=Path("tp_p.pt"))
    assert seen == {"cand": Path("tp_c.pt"), "opp": Path("tp_p.pt")}
    run()
    assert seen == {"cand": Path("shared.pt"), "opp": Path("shared.pt")}


def test_registered_arm_carries_the_paired_picker(tmp_path):
    import json
    from v_dance.selfplay.generation import register_promoted_gen
    from v_dance.play import serve_bandit as SB
    arch = tmp_path / "arch"
    (arch / "checkpoints").mkdir(parents=True)
    (arch / "tp").mkdir()
    for f in ("checkpoints/gen4.pt", "tp/tp_gen4.pt", "checkpoints/gen5.pt"):
        (arch / f).write_bytes(b"x")
    TPL.record_pair(arch / "checkpoints" / "gen4.pt", arch / "tp" / "tp_gen4.pt")
    cfg = tmp_path / "bandit.json"
    cfg.write_text(json.dumps({"arms": []}), encoding="utf-8")
    e4 = register_promoted_gen(cfg, battle_ckpt=arch / "checkpoints" / "gen4.pt", generation=4, prefix="tl_g")
    e5 = register_promoted_gen(cfg, battle_ckpt=arch / "checkpoints" / "gen5.pt", generation=5, prefix="tl_g")
    assert Path(e4["tp_ckpt"]).name == "tp_gen4.pt" and "co-trained picker" in e4["note"]
    assert e5["tp_ckpt"] == "default"                                     # no pair → the deployed picker
    arms = {a.name: a for a in SB.load_arms(cfg, exists=lambda p: True)}
    assert Path(arms["tl_g4"].tp_ckpt).name == "tp_gen4.pt" and arms["tl_g5"].uses_default("tp")


def test_the_picker_optimizer_survives_a_resume(tmp_path):
    ck = _tiny_ckpt(tmp_path, seed=7)
    cfg = TPL.TPLearnConfig(lr=1e-3, epochs=1, batch=32, seed=0)
    L = TPL.TPLearner(ck, cfg=cfg)
    model, vocab, mcfg = M.load_team_chooser(str(ck))
    L.update(_synthetic_records(model, vocab, mcfg, good=3, n=64, rng=np.random.default_rng(2)))
    out = L.save(tmp_path / "arch" / "tp" / "tp_gen0.pt", 0)
    op = TPL.opt_path_for(out)
    assert op.name == "tp_gen0_opt.pt" and op.exists()
    rng_after = L._rng.bit_generator.state
    cfg2 = TPL.TPLearnConfig(lr=5e-4, epochs=1, batch=32, seed=99)          # a new --tp-lr on the resume
    L2 = TPL.TPLearner(out, anchor_path=str(ck), cfg=cfg2, opt_state_path=op)
    assert L2.opt_restored and L2._rng.bit_generator.state == rng_after    # Adam moments + shuffle RNG back
    s1, s2 = L.opt.state_dict()["state"], L2.opt.state_dict()["state"]
    assert set(s1) == set(s2) and all(torch.equal(s1[k]["exp_avg"], s2[k]["exp_avg"]) for k in s1)
    assert all(g["lr"] == 5e-4 for g in L2.opt.param_groups)               # the CURRENT lr wins
    L3 = TPL.TPLearner(out, anchor_path=str(ck), cfg=cfg2, opt_state_path=tmp_path / "missing_opt.pt")
    assert not L3.opt_restored                                             # missing → a fresh Adam
    assert not TPL.TPLearner(ck, cfg=cfg2).opt_restored


def test_preflight_requires_the_resumed_run_to_restore_the_optimizer():
    from v_dance.selfplay import preflight as PF

    class Rec:
        def __init__(self, g):
            self.generation, self.n_trajectories, self.update_stats = g, 5, {"loss": 0.1}
            self.promoted, self.panel, self.hof = g == 1, {}, None

    class H:
        def __init__(self, n):
            self.records = [Rec(g) for g in range(n)]

    fresh = {"history": H(3), "tp": [{"generation": g, "n": 9} for g in range(3)], "tp_path": "x"}
    bad = {"history": H(4), "tp": [{"generation": 3, "n": 9, "opt_restored": False}]}
    good = {"history": H(4), "tp": [{"generation": 3, "n": 9, "opt_restored": True}]}
    assert any("optimizer state was NOT restored" in p for p in PF.check(fresh, bad, tp_learn_on=True)[0])
    assert not [p for p in PF.check(fresh, good, tp_learn_on=True)[0] if "picker" in p]


# ── the second review's fixes (2026-10-03) ─────────────────────────────────────────────────────────────────
class _M:
    def __init__(self, bid, won, trainable=True, own=("x", "y"), opp=("a", "b"), rec=True):
        self.battle_id, self.won, self.is_trainable = bid, won, trainable
        self.own_team, self.opp_team = list(own), list(opp)
        self.tp_learn = {"set_idx": 0} if rec else None


class _T:
    def __init__(self, m):
        self.meta = m


def test_training_records_drop_a_forfeited_game_and_its_twin():
    # review A: in a self-copy game the forfeiting seat is FALLBACK and its twin a mislabeled win — both go
    ts = [_T(_M("b1", None, trainable=False)), _T(_M("b1", True)), _T(_M("b2", True)), _T(_M("b2", False))]
    recs = TPL.training_records(ts)
    assert [r["battle_id"] for r in recs] == ["b2", "b2"]
    assert len(TPL.records_from_trajectories(ts)) == 3                    # the raw filter alone keeps b1's twin


def test_loo_baseline_leaves_out_the_whole_game_including_the_twin():
    # review B: two mirror games of one key: each record's baseline is the OTHER game only (never its own twin)
    b = TPL.loo_baselines([1, -1, 1, -1], ["k"] * 4, ["g1", "g1", "g2", "g2"])
    assert b == [0.0, 0.0, 0.0, 0.0]
    b = TPL.loo_baselines([1, -1, 1, 1], ["k"] * 4, ["g1", "g1", "g2", "g3"])
    assert b[0] == pytest.approx(1.0) and b[1] == pytest.approx(1.0)      # g1's records see only g2 + g3
    # a key whose only game is this one falls back to the other GAMES, never to its own twin
    assert TPL.loo_baselines([1, -1, 1], ["m", "m", "z"], ["g1", "g1", "g2"])[:2] == [1.0, 1.0]


def test_tau_and_eps_are_validated_and_the_record_is_exact(tmp_path):
    for bad in ({"tau": 0.0}, {"tau": -1.0}, {"eps": 1.5}, {"eps": -0.1}, {"lr": -1.0}):
        with pytest.raises(ValueError):
            TPL.TPLearnConfig(**bad)
    model, vocab, cfg = M.load_team_chooser(str(_tiny_ckpt(tmp_path)))
    M.team_order(model, vocab, cfg, _ROSTER, _OPP, 4, explore={"tau": 0.0, "eps": 2.0, "rng": np.random.default_rng(0)})
    r = M.LAST_TP_EXPLORE
    assert r["tau"] == 1e-6 and r["eps"] == 1.0                            # review C: the CLAMPED values are recorded
    assert abs(r["mu_set"] - 1.0 / 15) < 1e-9                             # eps 1 = uniform, and the record says so


def test_a_revert_brings_back_the_champions_picker(tmp_path):
    # review F: the champion's recorded pair; the start picker when the champion IS the base checkpoint
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "tp").mkdir()
    for f in ("checkpoints/gen4.pt", "tp/tp_gen4.pt", "base.pt", "start_tp.pt", "checkpoints/gen5.pt"):
        (tmp_path / f).write_bytes(b"x")
    TPL.record_pair(tmp_path / "checkpoints" / "gen4.pt", tmp_path / "tp" / "tp_gen4.pt")
    kw = dict(base_ckpt=tmp_path / "base.pt", learner_tp=str(tmp_path / "start_tp.pt"))
    assert _same(TPL.picker_after_restore(tmp_path / "checkpoints" / "gen4.pt", **kw), tmp_path / "tp" / "tp_gen4.pt")
    assert TPL.picker_after_restore(tmp_path / "base.pt", **kw) == str(tmp_path / "start_tp.pt")
    assert TPL.picker_after_restore(tmp_path / "checkpoints" / "gen5.pt", **kw) is None


def test_preflight_fails_when_every_picker_update_was_skipped():
    from v_dance.selfplay import preflight as PF

    class Rec:
        def __init__(self, g):
            self.generation, self.n_trajectories, self.update_stats = g, 5, {"loss": 0.1}
            self.promoted, self.panel, self.hof = g == 1, {}, None

    class H:
        def __init__(self, n):
            self.records = [Rec(g) for g in range(n)]

    fresh = {"history": H(3), "tp": [{"generation": g, "n": 3, "skipped": "fewer than 8"} for g in range(3)],
             "tp_path": "x"}
    resumed = {"history": H(4), "tp": [{"generation": 3, "n": 9, "opt_restored": True}]}
    assert any("every picker update was SKIPPED" in p for p in PF.check(fresh, resumed, tp_learn_on=True)[0])
