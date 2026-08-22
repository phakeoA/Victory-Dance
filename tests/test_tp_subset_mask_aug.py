"""TP tier-2 (2026-07-10): subset-mask augmentation + slot-masked loss + ladder-rating capture.

Aug contract (docs/teampreview_bring_lead_design.md addendum): with prob p a TRAIN item masks
K∈{1,2} roster slots (idx 0 + zero feats = the pad convention), masking brought and unbrought
alike, EXCLUDED from the BCE via ``slot_mask``; labels unchanged; only full 6-mon rosters; p=0
byte-identical (all-ones mask → sum/6 == the old plain mean).
"""
import json

import numpy as np
import torch

from v_dance.training.teampreview_dataset import BRING_K, LEAD_K, TEAM_SIZE, TeamPreviewDataset

_F = 8   # toy feat dim


def _example(n_valid=6, rid="r1"):
    feat = np.zeros((TEAM_SIZE, _F), dtype=np.float32)
    feat[:n_valid] = np.random.default_rng(1).random((n_valid, _F)).astype(np.float32) + 0.1
    bring = np.zeros(TEAM_SIZE, dtype=np.float32); bring[:BRING_K] = 1.0
    lead = np.zeros(TEAM_SIZE, dtype=np.float32); lead[:LEAD_K] = 1.0
    return {"our_species": [f"mon{i}" for i in range(TEAM_SIZE)],
            "opp_species": [f"opp{i}" for i in range(TEAM_SIZE)],
            "our_feat": feat, "opp_feat": feat.copy(), "bring": bring, "lead": lead,
            "valid_bring": 1.0, "valid_lead": 1.0, "replay_id": rid}


_VOCAB = {f"mon{i}": i + 1 for i in range(6)} | {f"opp{i}": i + 7 for i in range(6)}


def test_p0_is_byte_identical_with_allones_mask():
    ds = TeamPreviewDataset([_example()], _VOCAB, feat_dim=_F)          # default p=0
    item = ds[0]
    assert torch.equal(item["slot_mask"], torch.ones(TEAM_SIZE))
    assert torch.equal(item["our_feat"], ds.t_our_feat[0])              # untouched
    assert torch.equal(item["our_idx"], ds.t_our_idx[0])


def test_p1_masks_one_or_two_slots_with_pad_convention():
    ds = TeamPreviewDataset([_example()], _VOCAB, feat_dim=_F, subset_mask_p=1.0, aug_seed=3)
    for _ in range(20):                                                 # every draw augments
        item = ds[0]
        masked = (item["slot_mask"] == 0).nonzero().ravel().tolist()
        assert 1 <= len(masked) <= 2
        for s in masked:
            assert item["our_idx"][s] == 0                              # pad idx
            assert torch.equal(item["our_feat"][s], torch.zeros(_F))    # all-zero feat row
        assert torch.equal(item["bring"], ds.t_bring[0])                # labels unchanged
        assert torch.equal(item["lead"], ds.t_lead[0])
        assert torch.equal(ds.t_our_feat[0], torch.from_numpy(ds.our_feat[0]))  # backing intact


def test_fixed_k2_masks_exactly_two_slots():
    ds = TeamPreviewDataset([_example()], _VOCAB, feat_dim=_F,
                            subset_mask_p=1.0, subset_mask_k=2, aug_seed=5)
    for _ in range(20):
        item = ds[0]
        assert int((item["slot_mask"] == 0).sum()) == 2     # every augmented item = 4-mon context


def test_partial_rosters_never_augmented():
    ds = TeamPreviewDataset([_example(n_valid=5)], _VOCAB, feat_dim=_F, subset_mask_p=1.0)
    for _ in range(10):
        assert torch.equal(ds[0]["slot_mask"], torch.ones(TEAM_SIZE))


def test_run_epoch_loss_identical_with_and_without_slot_mask():
    """all-ones slot_mask must reproduce the legacy mean(dim=1) loss exactly (back-compat:
    a batch WITHOUT the key takes the legacy path)."""
    from v_dance.models.teampreview_model import build_model
    from v_dance.training.train_teampreview import run_epoch
    torch.manual_seed(0)
    model = build_model(vocab_size=16, feat_dim=_F, emb_dim=8, hidden=16, dropout=0.0,
                        device="cpu")
    ds = TeamPreviewDataset([_example(rid=f"r{i}") for i in range(6)], _VOCAB, feat_dim=_F)
    from torch.utils.data import DataLoader
    batches = list(DataLoader(ds, batch_size=3, shuffle=False))
    stripped = [{k: v for k, v in b.items() if k != "slot_mask"} for b in batches]
    m_with = run_epoch(model, batches, "cpu", optimizer=None)
    m_without = run_epoch(model, stripped, "cpu", optimizer=None)
    assert abs(m_with["loss"] - m_without["loss"]) < 1e-7


# ── ladder-rating capture ──────────────────────────────────────────────────────
def test_parse_rating_lines_poke_env_convention():
    from v_dance.play.play_vs_human_browser import _parse_rating_lines
    payload = (">battle-gen9x-1\n"
               "|raw|VictoriousDancing's rating: 1049 &rarr; <strong>1076</strong><br />(+27)\n"
               "|raw|somefoe's rating: 1102 &rarr; <strong>1075</strong><br />(-27)\n"
               "|raw|Ladder updating...")
    assert _parse_rating_lines(payload) == [("VictoriousDancing", 1049), ("somefoe", 1102)]
    assert _parse_rating_lines(">battle-x\n|move|p1a: A|Tackle|p2a: B") == []


def test_report_joins_rating_update_rows(tmp_path):
    from v_dance.eval.human_benchmark_report import load_rows
    log = tmp_path / "bench.jsonl"
    game = {"session_id": "s", "note": "online_adv", "game_idx": 1, "result": "ai",
            "battle_tag": "battle-gen9x-7", "rating": None, "opponent_rating": None}
    upd1 = {"type": "rating_update", "battle_tag": "battle-gen9x-7", "rating": 1049}
    upd2 = {"type": "rating_update", "battle_tag": "battle-gen9x-7", "opponent_rating": 1102}
    orphan = {"type": "rating_update", "battle_tag": "battle-gen9x-99", "rating": 1000}
    log.write_text("\n".join(json.dumps(r) for r in (game, upd1, upd2, orphan)) + "\n",
                   encoding="utf-8")
    rows = load_rows(log)
    assert len(rows) == 1                                # updates folded in, never counted as games
    assert rows[0]["rating"] == 1049 and rows[0]["opponent_rating"] == 1102
