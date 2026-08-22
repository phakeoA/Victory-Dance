"""TP contrastive set-scoring head (2026-07-11, docs/tp_contrastive_set_head_design.md).

Contract: flag-gated head (v6/legacy ckpts load unchanged); ONE full-roster trunk forward
(no masking — the null-#8 OOD trap never opens); score(S) = Σ bring_logit + Σ pair MLP +
set-level MLP with BOTH new final linears zero-inited so the set decode == the greedy
decode at initialization; warm start loads every backbone key and leaves ONLY set keys
fresh; serve dispatches on the ckpt stamp with TP_SET_HEAD as the kill-switch.
"""
from itertools import combinations

import numpy as np
import pytest
import torch

from v_dance.models.teampreview_model import TeamPreviewModel
from v_dance.training.teampreview_dataset import (
    BRING_K, LEAD_K, MON_FEAT_DIM, TEAM_SIZE, TeamPreviewDataset,
)
from v_dance.training.train_teampreview import (
    SET_SUBSETS, apply_warm_start, run_epoch,
)

_F = 8   # toy feat dim
_ARCH = dict(vocab_size=16, feat_dim=_F, emb_dim=8, hidden=16, dropout=0.1,
             use_self_attn=True, attn_heads=2)


def _inputs(B=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    our_idx = torch.randint(1, 16, (B, TEAM_SIZE), generator=g)
    opp_idx = torch.randint(1, 16, (B, TEAM_SIZE), generator=g)
    our_feat = torch.rand(B, TEAM_SIZE, _F, generator=g) + 0.1   # non-zero: no pad rows
    opp_feat = torch.rand(B, TEAM_SIZE, _F, generator=g) + 0.1
    return our_idx, opp_idx, our_feat, opp_feat


def _randomize_set_head(model, seed=1):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for head in (model.set_pair_mlp, model.set_global_mlp):
            for p in head.parameters():
                p.copy_(torch.randn(p.shape, generator=g) * 0.3)


def test_flag_off_has_no_set_keys_and_flag_on_forward_identical():
    torch.manual_seed(0)
    base = TeamPreviewModel(**_ARCH, use_set_head=False)
    assert not any(k.startswith("set_") for k in base.state_dict())
    withhead = TeamPreviewModel(**_ARCH, use_set_head=True)
    assert any(k.startswith("set_pair_mlp.") for k in withhead.state_dict())
    assert any(k.startswith("set_global_mlp.") for k in withhead.state_dict())
    # v6-style load: backbone weights in, only set keys fresh — forward byte-identical.
    n_fresh = apply_warm_start(withhead, base.state_dict())
    assert n_fresh == 8   # 2 heads x 2 linears x (weight+bias)
    base.eval(); withhead.eval()
    args = _inputs()
    b0, l0 = base(*args)
    b1, l1 = withhead(*args)
    assert torch.equal(b0, b1) and torch.equal(l0, l1)


def test_zero_init_set_decode_equals_greedy():
    torch.manual_seed(3)
    model = TeamPreviewModel(**_ARCH, use_set_head=True).eval()
    our_idx, opp_idx, our_feat, opp_feat = _inputs(B=4, seed=7)
    scores, bl, ll = model.score_subsets(our_idx, opp_idx, our_feat, opp_feat,
                                         subsets=SET_SUBSETS)
    assert scores.shape == (4, len(SET_SUBSETS))
    # zero-inited pair/global heads -> score(S) is EXACTLY the sum of member bring logits
    sub_idx = torch.tensor(SET_SUBSETS)
    assert torch.allclose(scores, bl[:, sub_idx].sum(-1), atol=1e-6)
    # ... so the argmax subset IS the greedy top-4
    for r in range(4):
        greedy = set(torch.topk(bl[r], BRING_K).indices.tolist())
        assert set(SET_SUBSETS[int(scores[r].argmax())]) == greedy
    # and score_subsets' marginal outputs match plain forward
    b2, l2 = model(our_idx, opp_idx, our_feat, opp_feat)
    assert torch.equal(bl, b2) and torch.equal(ll, l2)


def test_roster_permutation_equivariance():
    torch.manual_seed(5)
    model = TeamPreviewModel(**_ARCH, use_set_head=True).eval()
    _randomize_set_head(model)   # zero-init would make this test vacuous for the new terms
    our_idx, opp_idx, our_feat, opp_feat = _inputs(B=1, seed=9)
    scores, *_ = model.score_subsets(our_idx, opp_idx, our_feat, opp_feat)
    perm = torch.tensor([3, 0, 5, 1, 4, 2])
    scores_p, *_ = model.score_subsets(our_idx[:, perm], opp_idx,
                                       our_feat[:, perm], opp_feat)
    # subset S of the permuted roster = original mons {perm[i] for i in S}
    default = tuple(combinations(range(TEAM_SIZE), 4))
    orig_pos = {s: i for i, s in enumerate(default)}
    for si, s in enumerate(default):
        mapped = tuple(sorted(int(perm[i]) for i in s))
        assert torch.allclose(scores_p[0, si], scores[0, orig_pos[mapped]], atol=1e-5), \
            f"subset {s} -> {mapped} score changed under roster permutation"


def test_warm_start_rejects_arch_mismatch():
    # donor lacks the self-attn block -> its keys are MISSING (and not set-head keys):
    # the guard must fail loud instead of silently leaving backbone modules at random init.
    donor = TeamPreviewModel(**{**_ARCH, "use_self_attn": False})
    recipient = TeamPreviewModel(**_ARCH, use_set_head=True)
    with pytest.raises(SystemExit):
        apply_warm_start(recipient, donor.state_dict())


def _example(bring_first4=True, n_valid=6, rid="r1", seed=1):
    feat = np.zeros((TEAM_SIZE, _F), dtype=np.float32)
    feat[:n_valid] = np.random.default_rng(seed).random((n_valid, _F)).astype(np.float32) + 0.1
    bring = np.zeros(TEAM_SIZE, dtype=np.float32)
    bring[[0, 1, 2, 3] if bring_first4 else [0, 2, 4, 5]] = 1.0
    lead = np.zeros(TEAM_SIZE, dtype=np.float32); lead[:LEAD_K] = 1.0
    return {"our_species": [f"mon{i}" for i in range(TEAM_SIZE)],
            "opp_species": [f"opp{i}" for i in range(TEAM_SIZE)],
            "our_feat": feat, "opp_feat": feat.copy(), "bring": bring, "lead": lead,
            "valid_bring": 1.0, "valid_lead": 1.0, "replay_id": rid}


_VOCAB = {f"mon{i}": i + 1 for i in range(6)} | {f"opp{i}": i + 7 for i in range(6)}


def test_run_epoch_trains_set_head_and_reports_metrics():
    torch.manual_seed(0)
    model = TeamPreviewModel(**_ARCH, use_set_head=True)
    exs = [_example(bring_first4=(i % 2 == 0), rid=f"r{i}", seed=i) for i in range(8)]
    exs.append({**_example(rid="r_inval"), "valid_bring": 0.0})    # excluded from set CE
    exs.append(_example(n_valid=5, rid="r_partial"))               # partial roster excluded
    from torch.utils.data import DataLoader
    loader = DataLoader(TeamPreviewDataset(exs, _VOCAB, feat_dim=_F),
                        batch_size=4, shuffle=False)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    m = run_epoch(model, loader, "cpu", optimizer=opt, set_weight=1.0)
    assert np.isfinite(m["loss"])
    assert m["set_n"] == 8                       # 10 rows - invalid bring - partial roster
    assert 0.0 <= m["set_exact"] <= 1.0
    # set-head grads actually flowed (the final linears left zero-init)
    assert model.set_pair_mlp[-1].weight.abs().sum() > 0 or \
        model.set_global_mlp[-1].weight.abs().sum() > 0
    # a marginal-only model keeps set metrics at zero and runs unchanged
    plain = TeamPreviewModel(**_ARCH, use_set_head=False)
    m2 = run_epoch(plain, loader, "cpu", optimizer=None)
    assert m2["set_n"] == 0 and m2["set_exact"] == 0.0


def test_team_order_dispatches_on_stamp_with_kill_switch(monkeypatch):
    import v_dance.play.model_io as model_io
    torch.manual_seed(2)
    dex_dim = MON_FEAT_DIM                        # legacy recipe: mon_dex_features dim
    model = TeamPreviewModel(vocab_size=16, feat_dim=dex_dim, emb_dim=8, hidden=16,
                             dropout=0.1, use_self_attn=True, attn_heads=2,
                             use_set_head=True).eval()
    cfg = {"feat_dim": dex_dim, "bring_k": BRING_K, "lead_k": LEAD_K, "use_set_head": True}
    our = ["Pikachu", "Charizard", "Garchomp", "Amoonguss", "Incineroar", "Rillaboom"]
    opp = ["Torkoal", "Lilligant", "Flutter Mane", "Iron Hands", "Gholdengo", "Dragonite"]
    vocab = {}                                    # OOV -> PAD idx; dex feats still real
    with_set = model_io.team_order(model, vocab, cfg, our, opp, n=4, device="cpu")
    monkeypatch.setattr(model_io, "TP_SET_HEAD", False)
    greedy = model_io.team_order(model, vocab, cfg, our, opp, n=4, device="cpu")
    # valid serve order both ways: 4 distinct roster indices, leads-first
    for order in (with_set, greedy):
        assert len(order) == 4 and len(set(order)) == 4
        assert all(0 <= i < 6 for i in order)
    # zero-inited head -> the set decode IS the greedy decode (same bring AND leads)
    assert with_set == greedy
    # a learned head may deviate but must stay a valid order
    _randomize_set_head(model)
    monkeypatch.setattr(model_io, "TP_SET_HEAD", True)
    learned = model_io.team_order(model, vocab, cfg, our, opp, n=4, device="cpu")
    assert len(learned) == 4 and len(set(learned)) == 4
    # set-head decode consumed the model: same-bring check via direct helper
    oi_of = model_io._pack_side(our, vocab, dex_dim)
    order, ok = model_io._set_head_order(model, oi_of[0], oi_of[1], *model_io._pack_side(
        opp, vocab, dex_dim), None, 6, BRING_K, LEAD_K, 4, "cpu")
    assert ok and order == learned
