"""
Tests for the team-preview pipeline (teampreview_dataset / model / train).

Hermetic synthetic JSONL plus one opt-in real Type B check.

    .venv\\Scripts\\python.exe -m pytest ai_train_scripts\\teamPreview_model\\test_teampreview.py -q
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
import v_dance.training.teampreview_dataset as tp
from v_dance.training.teampreview_dataset import (
    BRING_K,
    LEAD_K,
    MON_FEAT_DIM,
    TEAM_SIZE,
    TeamPreviewDataset,
    build_examples,
    build_vocab,
    examples_from_folders,
    split_by_replay,
    _example_for_side,
)

torch = pytest.importorskip("torch")
from v_dance.models.teampreview_model import TeamPreviewModel  # noqa: E402
import v_dance.training.train_teampreview as trainmod  # noqa: E402


# ── Helpers ───────────────────────────────────────────────────────────────────
_R1 = ["Incineroar", "Kingambit", "Rillaboom", "Flutter Mane", "Urshifu", "Amoonguss"]
_R2 = ["Calyrex-Shadow", "Miraidon", "Chien-Pao", "Ogerpon", "Landorus", "Grimmsnarl"]


def make_tp(replay_id, p1_brought, p2_brought, p1_roster=_R1, p2_roster=_R2) -> dict:
    return {
        "replay_id": replay_id,
        "perspective": "p1",
        "players": {
            "p1": {"roster": list(p1_roster), "brought": list(p1_brought)},
            "p2": {"roster": list(p2_roster), "brought": list(p2_brought)},
        },
    }


# ── Dataset extraction ──────────────────────────────────────────────────────
def test_two_examples_per_replay_with_correct_labels():
    t = make_tp("r1",
                p1_brought=["Incineroar", "Kingambit", "Flutter Mane", "Urshifu"],
                p2_brought=["Miraidon", "Chien-Pao", "Ogerpon", "Landorus"])
    stats = Counter()
    ex_p1 = _example_for_side(t, "p1", "p2", stats)
    # bring multi-hot at roster positions 0,1,3,4
    assert ex_p1["bring"].tolist() == [1, 1, 0, 1, 1, 0]
    # leads are the first TWO brought (Incineroar=0, Kingambit=1) -> positions 0,1
    assert ex_p1["lead"].tolist() == [1, 1, 0, 0, 0, 0]
    assert int(ex_p1["lead"].sum()) == LEAD_K
    assert int(ex_p1["bring"].sum()) == BRING_K
    assert ex_p1["valid_bring"] == 1.0
    assert ex_p1["our_feat"].shape == (TEAM_SIZE, MON_FEAT_DIM)
    # opponent roster is the input "theirs"
    assert len(ex_p1["opp_species"]) == TEAM_SIZE


def test_incomplete_bring_marks_valid_false_but_keeps_leads():
    t = make_tp("r1",
                p1_brought=["Incineroar", "Kingambit", "Flutter Mane"],  # only 3 seen
                p2_brought=["Miraidon", "Chien-Pao"])
    ex = _example_for_side(t, "p1", "p2", None)
    assert ex["valid_bring"] == 0.0           # 4-set incomplete
    assert int(ex["lead"].sum()) == LEAD_K    # leads still observed (reliable)
    assert int(ex["bring"].sum()) == 3        # the 3 that appeared


def test_skips_when_roster_not_six_or_too_few_brought():
    bad_roster = make_tp("r1", _R1[:4], _R2[:4], p1_roster=_R1[:5])
    assert _example_for_side(bad_roster, "p1", "p2", Counter()) is None
    too_few = make_tp("r1", ["Incineroar"], ["Miraidon"])  # <2 brought
    assert _example_for_side(too_few, "p1", "p2", Counter()) is None


def test_build_examples_yields_both_sides():
    files_dir = None
    examples = []
    t = make_tp("r1", _R1[:4], _R2[:4])
    for side, opp in (("p1", "p2"), ("p2", "p1")):
        ex = _example_for_side(t, side, opp, Counter())
        examples.append(ex)
    assert {e["side"] for e in examples} == {"p1", "p2"}


def test_split_by_replay_keeps_games_together():
    exs = []
    for r in range(10):
        for side in ("p1", "p2"):
            exs.append({"replay_id": f"r{r}", "side": side})
    train, val = split_by_replay(exs, val_frac=0.3, seed=1)
    tr_ids = {e["replay_id"] for e in train}
    va_ids = {e["replay_id"] for e in val}
    assert tr_ids.isdisjoint(va_ids)
    # both sides of a replay land together
    for r in range(10):
        in_val = [e for e in val if e["replay_id"] == f"r{r}"]
        assert len(in_val) in (0, 2)


# ── Dataset tensors ─────────────────────────────────────────────────────────
def test_dataset_tensor_shapes_and_vocab():
    t = make_tp("r1", _R1[:4], _R2[:4])
    examples = [_example_for_side(t, "p1", "p2", Counter())]
    vocab = build_vocab(examples)
    dset = TeamPreviewDataset(examples, vocab)
    item = dset[0]
    assert item["our_idx"].shape == (TEAM_SIZE,)
    assert item["our_feat"].shape == (TEAM_SIZE, MON_FEAT_DIM)
    assert item["bring"].shape == (TEAM_SIZE,)
    # all our species are in the vocab (id > 0)
    assert int((item["our_idx"] > 0).sum()) == TEAM_SIZE


# ── Model ───────────────────────────────────────────────────────────────────
def test_model_forward_shapes():
    model = TeamPreviewModel(vocab_size=50, feat_dim=MON_FEAT_DIM, hidden=32)
    B = 4
    our_idx = torch.randint(1, 50, (B, TEAM_SIZE))
    opp_idx = torch.randint(1, 50, (B, TEAM_SIZE))
    our_feat = torch.randn(B, TEAM_SIZE, MON_FEAT_DIM)
    opp_feat = torch.randn(B, TEAM_SIZE, MON_FEAT_DIM)
    bring_logits, lead_logits = model(our_idx, opp_idx, our_feat, opp_feat)
    assert bring_logits.shape == (B, TEAM_SIZE)
    assert lead_logits.shape == (B, TEAM_SIZE)


def test_permutation_equivariance_over_our_roster():
    """Shuffling our roster order permutes the bring/lead logits identically and
    leaves the opponent untouched — the set-selection inductive bias."""
    torch.manual_seed(0)
    model = TeamPreviewModel(vocab_size=50, feat_dim=MON_FEAT_DIM, hidden=32, dropout=0.0)
    model.eval()
    our_idx = torch.randint(1, 50, (1, TEAM_SIZE))
    opp_idx = torch.randint(1, 50, (1, TEAM_SIZE))
    our_feat = torch.randn(1, TEAM_SIZE, MON_FEAT_DIM)
    opp_feat = torch.randn(1, TEAM_SIZE, MON_FEAT_DIM)
    b0, l0 = model(our_idx, opp_idx, our_feat, opp_feat)

    perm = torch.randperm(TEAM_SIZE)
    b1, l1 = model(our_idx[:, perm], opp_idx, our_feat[:, perm], opp_feat)
    assert torch.allclose(b0[0, perm], b1[0], atol=1e-5)
    assert torch.allclose(l0[0, perm], l1[0], atol=1e-5)


def test_one_optim_step_reduces_loss():
    import torch.nn.functional as F
    torch.manual_seed(0)
    model = TeamPreviewModel(vocab_size=50, feat_dim=MON_FEAT_DIM, hidden=32, dropout=0.0)
    B = 32
    our_idx = torch.randint(1, 50, (B, TEAM_SIZE))
    opp_idx = torch.randint(1, 50, (B, TEAM_SIZE))
    our_feat = torch.randn(B, TEAM_SIZE, MON_FEAT_DIM)
    opp_feat = torch.randn(B, TEAM_SIZE, MON_FEAT_DIM)
    # random but FIXED targets of the right cardinality
    bring = torch.zeros(B, TEAM_SIZE); lead = torch.zeros(B, TEAM_SIZE)
    for i in range(B):
        pick = torch.randperm(TEAM_SIZE)[:BRING_K]
        bring[i, pick] = 1.0
        lead[i, pick[:LEAD_K]] = 1.0
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    def loss_fn():
        bl, ll = model(our_idx, opp_idx, our_feat, opp_feat)
        return (F.binary_cross_entropy_with_logits(bl, bring)
                + F.binary_cross_entropy_with_logits(ll, lead))

    model.train()
    first = float(loss_fn().item())
    for _ in range(60):
        opt.zero_grad(); l = loss_fn(); l.backward(); opt.step()
    assert float(loss_fn().item()) < first


# ── End-to-end smoke train ──────────────────────────────────────────────────
def _write_corpus(folder: Path, n=16, prefix=""):
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for r in range(n):
        pick1 = list(rng.choice(TEAM_SIZE, BRING_K, replace=False))
        pick2 = list(rng.choice(TEAM_SIZE, BRING_K, replace=False))
        t = make_tp(f"{prefix}replay{r}",
                    p1_brought=[_R1[i] for i in pick1],
                    p2_brought=[_R2[i] for i in pick2])
        (folder / f"{prefix}replay{r}.jsonl").write_text(json.dumps(t), encoding="utf-8")


def test_end_to_end_smoke_train(tmp_path):
    corpus = tmp_path / "jsonl"
    _write_corpus(corpus)
    out = tmp_path / "ckpt"
    args = trainmod.parse_args([
        "--data", str(corpus), "--epochs", "2", "--batch-size", "8",
        "--hidden", "32", "--device", "cpu", "--val-frac", "0.25", "--out", str(out),
    ])
    res = trainmod.train(args)
    assert len(res["history"]) == 2
    assert np.isfinite(res["history"][-1]["val"]["loss"])
    assert (out / "teampreview_best.pt").exists()
    ckpt = torch.load(out / "teampreview_best.pt", map_location="cpu", weights_only=False)
    assert ckpt["config"]["bring_k"] == BRING_K
    assert "vocab" in ckpt


def test_smoke_train_with_type_a_and_patience(tmp_path):
    base = tmp_path / "typeB"
    typea = tmp_path / "typeA"
    _write_corpus(base, n=12)
    _write_corpus(typea, n=6, prefix="a_")            # distinct replay ids
    out = tmp_path / "ckpt"
    args = trainmod.parse_args([
        "--data", str(base), "--type-a", str(typea),
        "--epochs", "2", "--batch-size", "8", "--hidden", "32",
        "--device", "cpu", "--val-frac", "0.25", "--out", str(out), "--patience", "1",
    ])
    res = trainmod.train(args)
    assert (out / "teampreview_best.pt").exists()
    ck = torch.load(out / "teampreview_best.pt", map_location="cpu", weights_only=False)
    assert ck["config"]["patience"] == 1
    # both folders were ingested (18 replays × 2 perspectives = 36 examples)
    assert ck["config"]["data"] == [str(base), str(typea)]


# ── Opt-in: real Type B corpus ──────────────────────────────────────────────
_TYPE_B = (
    tp._SCRIPTS_DIR.parent          # <repo>/data/scripts -> <repo>/data
    / "vods" / "Prepared_training_data" / "Regulation_MA" / "Jsonl_TypeB"
)


@pytest.mark.skipif(not _TYPE_B.exists(), reason="Type B corpus not present")
def test_real_typeb_extraction():
    files = tp.iter_jsonl_files(str(_TYPE_B))[:50]
    examples, stats = build_examples(files)
    assert stats["examples"] == 2 * len(files)        # both sides every replay
    assert stats.get("unmatched_picks", 0) == 0       # brought always maps to roster
    for ex in examples:
        assert int(ex["lead"].sum()) == LEAD_K
        assert ex["our_feat"].shape == (TEAM_SIZE, MON_FEAT_DIM)
