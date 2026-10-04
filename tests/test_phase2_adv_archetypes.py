"""Phase-2 build tests (2026-07-02):

  * compute_advantage_weights — offline advantage weighting (A = G − V(s)):
    win outranks loss on the same state, mean-normalisation, filter mode,
    clip binding, loud refusal on an untrained value head.
  * team_archetypes — per-team mechanic features, seeded numpy k-means, the
    end-to-end build → artifact → load → assign round-trip on a synthetic
    two-team corpus (Trick Room vs switch-offense).
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("numpy")
import numpy as np
import torch

from conftest import write_attn_ckpt
from v_dance.encoders.state_encoder import get_state_dim

SD = get_state_dim()


# ── compute_advantage_weights ─────────────────────────────────────────────────

def _exs():
    x = np.zeros(SD, dtype=np.float32)
    return [{"x": x, "won": True}, {"x": x, "won": False}, {"x": x, "won": None}]


def test_advantage_weights_exp_ranks_win_over_loss(tmp_path):
    from v_dance.training.bc_dataset import compute_advantage_weights
    ckpt = write_attn_ckpt(tmp_path / "v.pt")
    w = compute_advantage_weights(_exs(), str(ckpt), mode="exp", beta=1.6)
    assert w.shape == (3,) and w.dtype == np.float32
    assert abs(float(w.mean()) - 1.0) < 1e-5          # mean-normalised
    assert w[0] > w[1]                                # same state: win > loss
    assert np.isfinite(w).all() and (w > 0).all()


def test_advantage_weights_filter_and_clip(tmp_path):
    from v_dance.training.bc_dataset import compute_advantage_weights
    ckpt = write_attn_ckpt(tmp_path / "v.pt")
    # filter: V ∈ (0,1) strictly ⇒ A>0 iff won; unknown stays 1.0 (pre-norm)
    wf = compute_advantage_weights(_exs(), str(ckpt), mode="filter")
    raw = np.array([1.0, 0.0, 1.0]); raw = raw / raw.mean()
    assert np.allclose(wf, raw, atol=1e-6)
    # huge beta ⇒ the clip binds exactly
    wc = compute_advantage_weights(_exs(), str(ckpt), mode="exp", beta=100.0,
                                   w_min=0.5, w_max=2.0)
    raw = np.array([2.0, 0.5, 1.0]); raw = raw / raw.mean()
    assert np.allclose(wc, raw, atol=1e-6)


def test_advantage_weights_refuses_untrained_value_head(tmp_path):
    from v_dance.training.bc_dataset import compute_advantage_weights
    bad = write_attn_ckpt(tmp_path / "bad.pt", value_trained=False)
    with pytest.raises(ValueError, match="value"):
        compute_advantage_weights(_exs(), str(bad))


def test_advantage_weights_no_labels_all_ones(tmp_path):
    from v_dance.training.bc_dataset import compute_advantage_weights
    ckpt = write_attn_ckpt(tmp_path / "v.pt")
    exs = [{"x": np.zeros(SD, dtype=np.float32), "won": None} for _ in range(3)]
    assert np.allclose(compute_advantage_weights(exs, str(ckpt)), 1.0)


# ── team_archetypes ───────────────────────────────────────────────────────────

def test_team_feature_vector_shape_and_finiteness():
    from v_dance.encoders.state_encoder import StateEncoder
    from v_dance.datatools.team_archetypes import (
        MON_FEATURE_DIM, PLAY_STATS_DIM, TEAM_FEATURE_DIM,
        mon_feature_vector, team_feature_vector)
    enc = StateEncoder()
    vecs = [mon_feature_vector(enc, {"species": s})
            for s in ("Torkoal", "Garchomp", "Hatterene")]
    assert all(np.isfinite(v).all() for v in vecs)
    assert not np.allclose(vecs[0], vecs[1])          # different mechanics differ
    tf = team_feature_vector(vecs, [0.1] * PLAY_STATS_DIM)
    assert tf.shape == (TEAM_FEATURE_DIM,) == (MON_FEATURE_DIM + PLAY_STATS_DIM,)
    assert np.isfinite(tf).all()


def test_kmeans_two_blobs_deterministic():
    from v_dance.datatools.team_archetypes import kmeans
    rng = np.random.RandomState(0)
    X = np.concatenate([rng.randn(30, 4) + 8.0, rng.randn(30, 4) - 8.0])
    c1, l1, i1 = kmeans(X, k=2, seed=7)
    c2, l2, i2 = kmeans(X, k=2, seed=7)
    assert np.array_equal(l1, l2) and np.allclose(c1, c2) and i1 == i2
    assert len(set(l1[:30])) == 1 and len(set(l1[30:])) == 1   # clean split
    assert l1[0] != l1[-1]


def _mon(species):
    return {"species": species}


def _turn(persp, rid, turn, total, actives, bench, actions):
    return {
        "replay_id": rid, "perspective": persp, "turn": turn,
        "total_turns": total, "decision_type": "turn",
        "our_actions": actions,
        "state_before_actions": {
            "our_active": {f"our_{c}": _mon(s) for c, s in zip("ab", actives)},
            "our_bench": [_mon(s) for s in bench],
            "opp_active": {}, "opp_bench": [],
        },
    }


_TR_TEAM = (("Torkoal", "Hatterene"),
            ("Indeedee-F", "Amoonguss", "Ursaluna", "Porygon2"))
_OFF_TEAM = (("Talonflame", "Garchomp"),
             ("Aegislash", "Rotom-Wash", "Sneasler", "Arcanine-Hisui"))
_TR_ACTS = [{"action": "move", "move": "Trick Room", "is_protect": False},
            {"action": "move", "move": "Protect", "is_protect": True}]
_OFF_ACTS = [{"action": "switch", "move": None, "is_protect": False},
             {"action": "move", "move": "Brave Bird", "is_protect": False}]


def _write_battle(folder, rid):
    lines = []
    for turn in (1, 2):
        lines.append(json.dumps(_turn("p1", rid, turn, 2, *_TR_TEAM,
                                      actions=_TR_ACTS)))
        lines.append(json.dumps(_turn("p2", rid, turn, 2, *_OFF_TEAM,
                                      actions=_OFF_ACTS)))
    (folder / f"{rid}.jsonl").write_text("\n".join(lines), encoding="utf-8")


def test_build_archetypes_end_to_end(tmp_path):
    from v_dance.datatools.team_archetypes import (
        assign, build_archetypes, load_archetype_assignments, load_artifact,
        team_feature_vector, mon_feature_vector, PLAY_STATS_DIM)
    from v_dance.encoders.state_encoder import StateEncoder
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write_battle(corpus, "syn-1")
    _write_battle(corpus, "syn-2")

    out = tmp_path / "archetypes_k2.npz"
    art = build_archetypes([str(corpus)], k=2, seed=0, out=str(out))
    assert art["n_teams"] == 2 and art["k"] == 2

    # every (replay, perspective) sighting is assigned; the two TEAMS split
    asg = load_archetype_assignments(str(out))
    assert set(asg) == {("syn-1", "p1"), ("syn-1", "p2"),
                        ("syn-2", "p1"), ("syn-2", "p2")}
    assert asg[("syn-1", "p1")] == asg[("syn-2", "p1")]      # same team, same z
    assert asg[("syn-1", "p1")] != asg[("syn-1", "p2")]      # TR ≠ offense

    # loaded artifact assigns the TR roster back to the TR cluster — including
    # via the mon-only (novel-team / serve) slice
    loaded = load_artifact(str(out))
    enc = StateEncoder()
    mons = [mon_feature_vector(enc, _mon(s))
            for s in _TR_TEAM[0] + _TR_TEAM[1]]
    feats = team_feature_vector(mons, [0.0] * PLAY_STATS_DIM)
    aid_mon_only, dist = assign(feats, loaded, mon_only=True)
    assert aid_mon_only == asg[("syn-1", "p1")]
    assert np.isfinite(dist)
    # per-cluster report data exists for the eyeball check
    assert len(loaded["clusters"]) == 2
    assert all(cl["top_species"] for cl in loaded["clusters"])

    # serve entry point: a SHEET-shaped team (parse_showdown_team keys) through
    # the ckpt-embeddable slim artifact lands on the same cluster
    from v_dance.datatools.team_archetypes import assign_team_sheet, artifact_slim
    sheet = [{"species": s, "item": None, "ability": None, "moves": [],
              "nature": None} for s in _TR_TEAM[0] + _TR_TEAM[1]]
    aid_sheet, dist_sheet = assign_team_sheet(sheet, artifact_slim(loaded))
    assert aid_sheet == asg[("syn-1", "p1")]
    assert np.isfinite(dist_sheet)
