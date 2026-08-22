"""
Tests for the BC v0 pipeline (bc_dataset / bc_model / train_bc).

Hermetic: builds synthetic transitions in-memory / in tmp dirs so the suite
does not depend on the multi-GB replay corpus.  One opt-in test exercises the
real Type B folder when it is present.

Run (GPU venv), excluding the flask server test as usual:
    .venv\\Scripts\\python.exe -m pytest ai_train_scripts\\test_bc.py -q
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
import v_dance.training.bc_dataset as ds
from v_dance.training.bc_dataset import (
    HEADS,
    BCDataset,
    build_examples,
    split_by_replay,
    transition_to_example,
)
from v_dance.encoders.state_encoder import get_state_dim, ACTIONS_PER_SLOT

torch = pytest.importorskip("torch")
from v_dance.models.bc_model_attn import AttnBCPolicy  # noqa: E402


def _model(hidden_dims=None, **kw):  # noqa: E402
    """#27 attn-only: a tiny AttnBCPolicy as the BC-pipeline test vehicle (hidden_dims ignored)."""
    return AttnBCPolicy(d_model=32, n_layers=1, **kw)
import v_dance.training.train_bc as train_bc  # noqa: E402


# ── Helpers ───────────────────────────────────────────────────────────────────
def _row(*legal: int) -> list:
    """16-wide mask row with the given indices set to 1."""
    row = [0] * ACTIONS_PER_SLOT
    for i in legal:
        row[i] = 1
    return row


def _mon(species: str = "Incineroar", hp: float = 100.0) -> dict:
    return {"species": species, "base_species": species, "hp_pct": hp, "seen": True}


def make_transition(
    replay_id: str,
    our_actions: list,
    mask_a: list,
    mask_b: list,
    turn: int = 1,
) -> dict:
    """A minimal but encode-able transition dict."""
    return {
        "replay_id": replay_id,
        "turn": turn,
        "source_type": "ranked_player_vod",
        "perspective": "p1",
        "state_before_actions": {
            "our_active": {"our_a": _mon("Incineroar"), "our_b": _mon("Kingambit")},
            "opp_active": {"opp_a": _mon("Amoonguss"), "opp_b": _mon("Flutter Mane")},
            "our_bench": [_mon("Rillaboom"), _mon("Urshifu")],
            "opp_bench": [],
            "field": {},
            "side_conditions": {},
        },
        "our_actions": our_actions,
        "action_mask": {"our_a": mask_a, "our_b": mask_b},
    }


# ══════════════════════════════════════════════════════════════════════════════
# Dataset extraction
# ══════════════════════════════════════════════════════════════════════════════
def test_basic_two_head_example():
    t = make_transition(
        "r1",
        our_actions=[
            {"slot": "our_a", "action": "move", "move": "Fake Out", "action_index": 3},
            {"slot": "our_b", "action": "switch", "species": "Rillaboom", "action_index": 12},
        ],
        mask_a=_row(0, 3, 12, 13),
        mask_b=_row(0, 1, 12, 13),
    )
    stats = Counter()
    ex = transition_to_example(t, ds.StateEncoder(), stats)
    assert ex is not None
    assert ex["targets"] == {"our_a": 3, "our_b": 12}
    assert ex["x"].shape == (get_state_dim(),)
    assert ex["x"].dtype == np.float32
    assert np.isfinite(ex["x"]).all()
    assert ex["masks"]["our_a"].shape == (ACTIONS_PER_SLOT,)
    assert stats["usable_examples"] == 2


def test_null_action_index_skips_that_head_only():
    t = make_transition(
        "r1",
        our_actions=[
            {"slot": "our_a", "action": "move", "move": "?", "action_index": None},
            {"slot": "our_b", "action": "move", "move": "Sucker Punch", "action_index": 0},
        ],
        mask_a=_row(0, 3),
        mask_b=_row(0, 1),
    )
    stats = Counter()
    ex = transition_to_example(t, ds.StateEncoder(), stats)
    assert ex is not None
    assert ex["targets"] == {"our_b": 0}
    assert stats["skipped_null_index"] == 1
    assert stats["usable_examples"] == 1


def test_forced_replacement_first_entry_wins():
    # our_b appears twice: turn-start switch (12) then a post-faint replace (13)
    t = make_transition(
        "r1",
        our_actions=[
            {"slot": "our_b", "action": "switch", "species": "Rillaboom", "action_index": 12},
            {"slot": "our_a", "action": "move", "move": "Protect", "action_index": 3},
            {"slot": "our_b", "action": "switch", "species": "Urshifu", "action_index": 13},
        ],
        mask_a=_row(3, 12, 13),
        mask_b=_row(12, 13),
    )
    stats = Counter()
    ex = transition_to_example(t, ds.StateEncoder(), stats)
    assert ex["targets"]["our_b"] == 12  # the turn-start choice, not 13
    assert ex["targets"]["our_a"] == 3
    assert stats["dropped_forced_replacement"] == 1


def test_illegal_target_under_mask_is_dropped():
    t = make_transition(
        "r1",
        our_actions=[
            {"slot": "our_a", "action": "move", "move": "Sucker Punch", "action_index": 0},
            {"slot": "our_b", "action": "move", "move": "Knock Off", "action_index": 1},
        ],
        mask_a=_row(3, 6),       # index 0 NOT legal -> drop our_a
        mask_b=_row(0, 1, 12),   # index 1 legal -> keep our_b
    )
    stats = Counter()
    ex = transition_to_example(t, ds.StateEncoder(), stats)
    assert ex is not None
    assert ex["targets"] == {"our_b": 1}
    assert stats["skipped_illegal_target"] == 1


def test_transition_with_no_valid_head_is_skipped():
    t = make_transition(
        "r1",
        our_actions=[
            {"slot": "our_a", "action": "move", "move": "?", "action_index": None},
        ],
        mask_a=_row(3),
        mask_b=_row(3),
    )
    stats = Counter()
    ex = transition_to_example(t, ds.StateEncoder(), stats)
    assert ex is None


def test_split_by_replay_is_disjoint_and_deterministic():
    examples = []
    for r in range(10):
        examples.append({"x": np.zeros(get_state_dim(), np.float32),
                         "targets": {"our_a": 0}, "masks": {"our_a": np.zeros(16, np.float32)},
                         "replay_id": f"r{r}"})
    train, val = split_by_replay(examples, val_frac=0.3, seed=42)
    train_ids = {e["replay_id"] for e in train}
    val_ids = {e["replay_id"] for e in val}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids | val_ids == {f"r{r}" for r in range(10)}
    assert len(val_ids) == 3
    # deterministic
    train2, val2 = split_by_replay(examples, val_frac=0.3, seed=42)
    assert {e["replay_id"] for e in val2} == val_ids


# ══════════════════════════════════════════════════════════════════════════════
# Demonstrator skill filtering / weighting (TIER-1 #1)
# ══════════════════════════════════════════════════════════════════════════════
def _rated_transition(rid, rating, delta, won, our_side="p1"):
    """A transition carrying our-side rating + a winner consistent with ``won``."""
    t = make_transition(
        rid,
        our_actions=[{"slot": "our_a", "action": "move", "move": "Fake Out",
                      "action_index": 3}],
        mask_a=_row(3, 12), mask_b=_row(0),
    )
    me, opp = ("Me", "Them")
    t["perspective"] = our_side
    t["players"] = {
        "our_side": our_side,
        our_side: {"username": me, "rating_before": rating, "rating_delta": delta},
        ("p2" if our_side == "p1" else "p1"): {"username": opp},
    }
    t["winner"] = me if won else opp
    return t


def test_our_player_meta_extracts_rating_delta_won():
    t = _rated_transition("r1", rating=1850, delta=20, won=True)
    rating, delta, won = ds._our_player_meta(t)
    assert rating == 1850.0 and delta == 20.0 and won is True
    # and it threads onto the example
    ex = transition_to_example(t, ds.StateEncoder(), None)
    assert ex["rating"] == 1850.0 and ex["won"] is True

    lost = _rated_transition("r2", rating=1600, delta=-22, won=False)
    assert ds._our_player_meta(lost) == (1600.0, -22.0, False)


def test_our_player_meta_missing_is_safe():
    t = make_transition("r1",
        our_actions=[{"slot": "our_a", "action": "move", "move": "x", "action_index": 3}],
        mask_a=_row(3), mask_b=_row(0))
    assert ds._our_player_meta(t) == (None, 0.0, None)
    ex = transition_to_example(t, ds.StateEncoder(), None)
    assert ex["rating"] is None and ex["won"] is None


def test_filter_by_rating_keeps_unknown_and_drops_low():
    exs = [
        {"rating": 1900}, {"rating": 1700}, {"rating": None}, {"rating": 1500},
    ]
    kept = ds.filter_by_rating(exs, 1750)
    assert {e.get("rating") for e in kept} == {1900, None}   # 1700/1500 dropped, None kept
    assert ds.filter_by_rating(exs, None) == exs             # None threshold = no-op


def test_compute_sample_weights_off_is_ones():
    exs = [{"rating": 1800, "won": True}, {"rating": 1500, "won": False}]
    w = ds.compute_sample_weights(exs)                       # both flags off
    assert np.allclose(w, [1.0, 1.0])


def test_compute_sample_weights_rating_orders_and_normalises():
    exs = [{"rating": r} for r in (1500, 1700, 1900, 2100)]
    w = ds.compute_sample_weights(exs, rating_weight=True, rating_weight_floor=0.25)
    assert np.isclose(w.mean(), 1.0, atol=1e-5)              # mean-normalised
    assert w[0] < w[1] < w[2] < w[3]                         # monotone in rating
    assert w.min() > 0.0


def test_compute_sample_weights_outcome_downweights_losses():
    exs = [{"won": True}, {"won": False}, {"won": None}]
    w = ds.compute_sample_weights(exs, outcome_weight=True, loss_weight=0.5)
    # pre-normalisation ratio is 1.0 : 0.5 : 1.0 — the loss is relatively lighter
    assert w[1] < w[0]
    assert np.isclose(w[0], w[2])                            # unknown treated like a win (neutral)
    assert np.isclose(w.mean(), 1.0, atol=1e-5)


def test_compute_closed_copy_weights_splits_pairs_and_leaves_unpaired():
    # game "a" has an open + __closed twin; "b" is unpaired; "battle-c"/"c__closed"
    # pair through canonical_rid despite the battle- prefix.
    exs = [{"replay_id": "gen9-a"}, {"replay_id": "gen9-a__closed"},
           {"replay_id": "gen9-b"},
           {"replay_id": "battle-gen9-c"}, {"replay_id": "gen9-c__closed"}]
    w = ds.compute_closed_copy_weights(exs, lam_closed=0.75)
    assert np.isclose(w.mean(), 1.0, atol=1e-5)              # mean-normalised
    # pre-normalisation: a-open 0.25, a-closed 0.75, b 1.0, c-open 0.25, c-closed 0.75
    assert np.isclose(w[0] / w[2], 0.25) and np.isclose(w[1] / w[2], 0.75)
    assert np.isclose(w[3] / w[2], 0.25) and np.isclose(w[4] / w[2], 0.75)
    assert np.isclose(w[0] + w[1], w[2])                     # the pair totals ONE decision


def test_compute_closed_copy_weights_lam1_is_closed_only():
    exs = [{"replay_id": "gen9-a"}, {"replay_id": "gen9-a__closed"},
           {"replay_id": "gen9-b"}]
    w = ds.compute_closed_copy_weights(exs, lam_closed=1.0)
    assert w[0] == 0.0                                       # open copy silenced
    assert np.isclose(w[1], w[2])                            # closed copy = a full decision
    with pytest.raises(ValueError):
        ds.compute_closed_copy_weights(exs, lam_closed=1.5)


def test_compute_closed_copy_weights_no_pairs_is_ones():
    exs = [{"replay_id": "gen9-a"}, {"replay_id": "gen9-b__closed"}]
    w = ds.compute_closed_copy_weights(exs, lam_closed=0.5)  # no canonical overlap
    assert np.allclose(w, [1.0, 1.0])


def test_head_loss_sample_weight_ones_matches_unweighted():
    torch.manual_seed(0)
    logits = torch.randn(8, ACTIONS_PER_SLOT)
    mask = torch.ones(8, ACTIONS_PER_SLOT)
    target = torch.randint(0, ACTIONS_PER_SLOT, (8,))
    valid = torch.ones(8)
    ce0, *_ = train_bc.head_loss_and_acc(logits, mask, target, valid)
    ce1, *_ = train_bc.head_loss_and_acc(logits, mask, target, valid,
                                         sample_weight=torch.ones(8))
    assert torch.allclose(ce0, ce1, atol=1e-5)


def test_bcdataset_weight_propagates():
    ex = transition_to_example(_rated_transition("r1", 1800, 10, True), ds.StateEncoder(), None)
    item = BCDataset([ex], weights=[2.5])[0]
    assert float(item["weight"]) == 2.5
    # default (no weights) → 1.0
    assert float(BCDataset([ex])[0]["weight"]) == 1.0
    with pytest.raises(ValueError):
        BCDataset([ex], weights=[1.0, 2.0])                 # length mismatch


def test_run_epoch_sample_weight_emphasises_high_weight_examples():
    """Two contradictory labels at the SAME input; the weight-1 group's label wins
    over the weight-0 group when sample weighting is on (the mechanism behind
    imitating stronger demonstrators)."""
    from torch.utils.data import DataLoader
    x = np.zeros(get_state_dim(), np.float32)
    mask = np.ones(ACTIONS_PER_SLOT, np.float32)
    good = [{"x": x.copy(), "replay_id": "r", "targets": {"our_a": 0},
             "masks": {"our_a": mask}, "gimmick_targets": {}, "gimmick_masks": {}}
            for _ in range(64)]
    bad = [{"x": x.copy(), "replay_id": "r", "targets": {"our_a": 5},
            "masks": {"our_a": mask}, "gimmick_targets": {}, "gimmick_masks": {}}
           for _ in range(64)]
    weights = [1.0] * 64 + [0.0] * 64                       # ignore the 'bad' group
    loader = DataLoader(BCDataset(good + bad, weights=weights), batch_size=32, shuffle=True)
    torch.manual_seed(0)
    model = _model(hidden_dims=(16,), dropout=0.0, heads=HEADS)
    opt = torch.optim.Adam(model.parameters(), lr=5e-2)
    for _ in range(40):
        train_bc.run_epoch(model, loader, "cpu", opt, sample_weighted=True)
    actions, _, _ = model(torch.as_tensor(x))
    assert int(actions["our_a"].argmax().item()) == 0      # high-weight label won


# ══════════════════════════════════════════════════════════════════════════════
# Torch Dataset
# ══════════════════════════════════════════════════════════════════════════════
def test_bcdataset_tensor_shapes_and_invalid_marking():
    t = make_transition(
        "r1",
        our_actions=[
            {"slot": "our_a", "action": "move", "move": "Fake Out", "action_index": 3},
            {"slot": "our_b", "action": "move", "move": "?", "action_index": None},
        ],
        mask_a=_row(3, 12),
        mask_b=_row(0, 1),
    )
    ex = transition_to_example(t, ds.StateEncoder(), None)
    dset = BCDataset([ex])
    item = dset[0]
    assert item["x"].shape == (get_state_dim(),)
    assert item["target"].tolist() == [3, -1]            # our_b invalid -> -1
    assert item["valid"].tolist() == [1.0, 0.0]
    assert item["mask"].shape == (len(HEADS), ACTIONS_PER_SLOT)


# ══════════════════════════════════════════════════════════════════════════════
# Gimmick (mega) labels — extraction + dataset tensors (Task #5b)
# ══════════════════════════════════════════════════════════════════════════════
def _gim_transition(our_actions, mask_a, mask_b, gmask_a=None, gmask_b=None, rid="r1"):
    t = make_transition(rid, our_actions, mask_a, mask_b)
    gm = {}
    if gmask_a is not None:
        gm["our_a"] = gmask_a
    if gmask_b is not None:
        gm["our_b"] = gmask_b
    if gm:
        t["gimmick_mask"] = gm
    return t


def test_gimmick_targets_extracted():
    t = _gim_transition(
        our_actions=[
            {"slot": "our_a", "action": "move", "move": "Fake Out",
             "action_index": 3, "gimmick_index": 1},   # mega'd
            {"slot": "our_b", "action": "move", "move": "Sucker Punch",
             "action_index": 0, "gimmick_index": 0},   # plain
        ],
        mask_a=_row(0, 3, 12), mask_b=_row(0, 1, 12),
        gmask_a=[1, 1, 0], gmask_b=[1, 0, 0],
    )
    stats = Counter()
    ex = transition_to_example(t, ds.StateEncoder(), stats)
    assert ex["gimmick_targets"] == {"our_a": 1, "our_b": 0}
    assert ex["gimmick_masks"]["our_a"].tolist() == [1.0, 1.0, 0.0]
    assert stats["usable_gimmick_examples"] == 2
    assert stats["gimmick_positives"] == 1


def test_bcdataset_gimmick_tensors():
    t = _gim_transition(
        our_actions=[
            {"slot": "our_a", "action": "move", "move": "Fake Out",
             "action_index": 3, "gimmick_index": 1},
            {"slot": "our_b", "action": "move", "move": "Sucker Punch",
             "action_index": 0, "gimmick_index": 0},
        ],
        mask_a=_row(0, 3, 12), mask_b=_row(0, 1, 12),
        gmask_a=[1, 1, 0], gmask_b=[1, 0, 0],
    )
    ex = transition_to_example(t, ds.StateEncoder(), None)
    item = BCDataset([ex])[0]
    assert item["gimmick_target"].tolist() == [1, 0]
    assert item["gimmick_valid"].tolist() == [1.0, 1.0]
    assert item["gimmick_mask"].shape == (len(HEADS), 3)        # v11 Phase D: 3-wide
    assert item["gimmick_mask"][0].tolist() == [1.0, 1.0, 0.0]


def test_gimmick_absent_is_back_compatible():
    """Pre-gimmick JSONL (no gimmick_mask / gimmick_index) yields no gimmick
    target — the gimmick head simply gets no signal until a re-export."""
    t = make_transition(
        "r1",
        our_actions=[{"slot": "our_a", "action": "move", "move": "Fake Out",
                      "action_index": 3}],
        mask_a=_row(3, 12), mask_b=_row(0),
    )
    ex = transition_to_example(t, ds.StateEncoder(), None)
    assert ex["gimmick_targets"] == {}
    item = BCDataset([ex])[0]
    assert item["gimmick_valid"].tolist() == [0.0, 0.0]
    assert item["gimmick_target"].tolist() == [-1, -1]


def test_gimmick_dropped_when_illegal_under_mask():
    """A mega label whose gimmick mask forbids mega is dropped (never a wrong
    label) while the action label is kept."""
    t = _gim_transition(
        our_actions=[{"slot": "our_a", "action": "move", "move": "Fake Out",
                      "action_index": 3, "gimmick_index": 1}],
        mask_a=_row(3, 12), mask_b=_row(0),
        gmask_a=[1, 0, 0],     # mega NOT legal here (tera col 0 too in this fixture)
    )
    ex = transition_to_example(t, ds.StateEncoder(), None)
    assert "our_a" in ex["targets"]            # action kept
    assert ex["gimmick_targets"] == {}          # gimmick dropped


# ══════════════════════════════════════════════════════════════════════════════
# Auxiliary opponent head — dataset labels (Task #9c)
# ══════════════════════════════════════════════════════════════════════════════
def _opp_transition():
    t = make_transition(
        "r1",
        our_actions=[{"slot": "our_a", "action": "move", "move": "Fake Out", "action_index": 3}],
        mask_a=_row(3, 12), mask_b=_row(0),
    )
    snap = t["state_before_actions"]
    snap["opp_active"]["opp_a"]["known_moves"] = ["Spore", "Pollen Puff", "Rage Powder", "Protect"]
    snap["opp_active"]["opp_b"]["known_moves"] = ["Moonblast", "Dazzling Gleam"]
    snap["opp_bench"] = []
    t["opp_actions_actual"] = [
        {"slot": "opp_a", "action": "move", "move": "Spore", "target_slot": "our_a"},
        {"slot": "opp_b", "action": "move", "move": "Moonblast", "target_slot": "our_a"},
    ]
    return t


def test_with_opp_extracts_opp_targets_and_tensors():
    from v_dance.encoders.state_encoder import OPP_HEADS
    ex = transition_to_example(_opp_transition(), ds.StateEncoder(), None, with_opp=True)
    assert "opp_targets" in ex and ex["opp_targets"]            # at least one opp action encoded
    assert set(ex["opp_targets"]) <= {"opp_a", "opp_b"}

    item = BCDataset([ex], with_opp=True)[0]
    assert item["opp_target"].shape == (len(OPP_HEADS),)
    assert item["opp_mask"].shape == (len(OPP_HEADS), ACTIONS_PER_SLOT)
    # every valid opp target is legal under its opp mask (the codec invariant)
    for o in range(len(OPP_HEADS)):
        if item["opp_valid"][o].item() > 0.5:
            assert item["opp_mask"][o][int(item["opp_target"][o])].item() == 1.0


def test_without_opp_has_no_opp_fields():
    ex = transition_to_example(_opp_transition(), ds.StateEncoder(), None)  # with_opp default False
    assert "opp_targets" not in ex
    item = BCDataset([ex])[0]                                   # with_opp default False
    assert "opp_target" not in item


def test_opp_fields_survive_move_order_augmentation():
    """Augmentation permutes only OUR move blocks → the opp aux labels pass
    through unchanged."""
    ex = transition_to_example(_opp_transition(), ds.StateEncoder(), None, with_opp=True)
    plain = BCDataset([ex], with_opp=True)[0]
    aug = BCDataset([ex], with_opp=True, augment_move_order=True)[0]
    assert aug["opp_target"].tolist() == plain["opp_target"].tolist()
    assert aug["opp_valid"].tolist() == plain["opp_valid"].tolist()


# ══════════════════════════════════════════════════════════════════════════════
# Move-slot permutation augmentation — dataset (Task #7b)
# ══════════════════════════════════════════════════════════════════════════════
def _marked_example(target_move: int = 1):
    """An example whose 4 our_a move blocks are marked (feature 0 = move idx+1),
    target = ``target_move`` at bucket 0, all 4 moves legal."""
    from v_dance.encoders.state_encoder import own_active_move_base, MOVE_FEATURES, NUM_MOVES
    x = np.zeros(get_state_dim(), np.float32)
    base = own_active_move_base(0)
    for m in range(NUM_MOVES):
        x[base + m * MOVE_FEATURES] = m + 1
    mask = np.zeros(ACTIONS_PER_SLOT, np.float32)
    for m in range(NUM_MOVES):
        mask[m * 3] = 1.0
    return {
        "x": x, "replay_id": "r",
        "targets": {"our_a": target_move * 3},
        "masks": {"our_a": mask},
        "gimmick_targets": {"our_a": 1},
        "gimmick_masks": {"our_a": np.array([1, 1, 0], np.float32)},
    }


def test_augment_off_is_identity():
    ex = _marked_example(target_move=1)
    item = BCDataset([ex], augment_move_order=False)[0]
    assert item["target"].tolist() == [3, -1]
    assert np.allclose(item["x"].numpy(), ex["x"])


def test_augment_tracks_target_move_features_and_legality():
    """Over many fetches the target moves around (position-invariant) but always
    points at the SAME move's features and stays legal under the returned mask;
    the gimmick label is never touched."""
    from v_dance.encoders.state_encoder import own_active_move_base, MOVE_FEATURES
    ds = BCDataset([_marked_example(target_move=1)], augment_move_order=True, aug_seed=0)
    base = own_active_move_base(0)
    seen = set()
    for _ in range(40):
        item = ds[0]
        t = int(item["target"][0])
        pos = t // 3
        seen.add(pos)
        assert item["x"][base + pos * MOVE_FEATURES].item() == 2.0   # old move 1 marker
        assert item["mask"][0][t].item() == 1.0                       # still legal
        assert item["gimmick_target"].tolist() == [1, -1]            # gimmick untouched
    assert len(seen) > 1                                              # target really moved


def _pos_example(marked_pos: int, target_pos: int):
    """4 legal moves; the move at ``marked_pos`` carries a strong marker; target =
    ``target_pos``.  In training marker==target==pos 0 (a position bias); at test
    the marker (and target) sit at varied positions."""
    from v_dance.encoders.state_encoder import own_active_move_base, MOVE_FEATURES, NUM_MOVES
    x = np.zeros(get_state_dim(), np.float32)
    base = own_active_move_base(0)
    x[base + marked_pos * MOVE_FEATURES] = 5.0
    mask = np.zeros(ACTIONS_PER_SLOT, np.float32)
    for m in range(NUM_MOVES):
        mask[m * 3] = 1.0
    return {"x": x, "replay_id": "r", "targets": {"our_a": target_pos * 3},
            "masks": {"our_a": mask}, "gimmick_targets": {}, "gimmick_masks": {}}


def _train_tiny(examples, augment, epochs=40, seed=0):
    from torch.utils.data import DataLoader
    torch.manual_seed(seed)
    loader = DataLoader(BCDataset(examples, augment_move_order=augment, aug_seed=seed),
                        batch_size=64, shuffle=True)
    model = _model(hidden_dims=(32,), dropout=0.0, heads=HEADS)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(epochs):
        train_bc.run_epoch(model, loader, "cpu", opt)
    return model


def _move_acc(model, examples):
    correct = 0
    for ex in examples:
        actions, _, _ = model(torch.as_tensor(ex["x"]))
        logits = actions["our_a"].detach().numpy().ravel()
        mask = ex["masks"]["our_a"]
        best = max((i for i in range(ACTIONS_PER_SLOT) if mask[i] > 0.5),
                   key=lambda i: logits[i])
        correct += int(best == ex["targets"]["our_a"])
    return correct / len(examples)


def test_augmentation_makes_policy_move_order_invariant():
    """The fix's PURPOSE: trained WITH augmentation the policy picks the move by
    FEATURE regardless of slot position (generalises to a shifted test order);
    trained WITHOUT it on position-biased data it latches onto position 0 and
    fails the shifted test."""
    train_ex = [_pos_example(0, 0) for _ in range(256)]          # marker+target at pos 0
    test_ex = [_pos_example(p, p) for p in ([0, 1, 2, 3] * 16)]  # marker+target shifted
    acc_aug = _move_acc(_train_tiny(train_ex, augment=True), test_ex)
    acc_noaug = _move_acc(_train_tiny(train_ex, augment=False), test_ex)
    assert acc_aug >= 0.8, f"augmented policy not order-invariant: {acc_aug}"
    assert acc_aug > acc_noaug + 0.2, f"aug {acc_aug} vs no-aug {acc_noaug}"


# ══════════════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════════════
def test_model_forward_shapes():
    model = _model(hidden_dims=(64, 32), heads=HEADS)
    x = torch.randn(5, get_state_dim())
    actions, gimmicks, value = model(x)
    assert set(actions.keys()) == set(HEADS)
    assert set(gimmicks.keys()) == set(HEADS)
    for h in HEADS:
        assert actions[h].shape == (5, ACTIONS_PER_SLOT)
        assert gimmicks[h].shape == (5, model.gimmick_dim)
    assert value.shape == (5,)                       # scalar win-logit per board
    assert model.gimmick_dim == 3
    assert model.count_parameters() > 0


def test_model_with_aux_opp_heads_has_action_but_not_gimmick_heads():
    # 4 action heads (our + opp), gimmick heads ONLY for the own slots (task #9b).
    model = _model(hidden_dims=(32,), heads=("our_a", "our_b", "opp_a", "opp_b"))
    x = torch.randn(3, get_state_dim())
    actions, gimmicks, value = model(x)
    assert set(actions.keys()) == {"our_a", "our_b", "opp_a", "opp_b"}
    assert set(gimmicks.keys()) == {"our_a", "our_b"}         # no opp gimmick head
    assert actions["opp_a"].shape == (3, ACTIONS_PER_SLOT)
    assert model.gimmick_head_names == ("our_a", "our_b")


def test_masked_logits_blocks_illegal():
    logits = torch.zeros(2, ACTIONS_PER_SLOT)
    logits[:, 5] = 10.0          # would-be argmax
    mask = torch.zeros(2, ACTIONS_PER_SLOT)
    mask[:, 3] = 1.0             # only index 3 legal
    ml = train_bc.masked_logits(logits, mask)
    assert ml.argmax(dim=1).tolist() == [3, 3]


def test_one_optim_step_reduces_loss():
    torch.manual_seed(0)
    model = _model(hidden_dims=(64, 32), dropout=0.0, heads=HEADS)
    x = torch.randn(16, get_state_dim())
    mask = torch.ones(16, len(HEADS), ACTIONS_PER_SLOT)
    target = torch.randint(0, ACTIONS_PER_SLOT, (16, len(HEADS)))
    valid = torch.ones(16, len(HEADS))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    def step():
        actions, _gimmicks, _value = model(x)
        loss = x.new_zeros(())
        nv = 0
        for hi, h in enumerate(HEADS):
            ce, n, _, _ = train_bc.head_loss_and_acc(
                actions[h], mask[:, hi], target[:, hi], valid[:, hi]
            )
            loss = loss + ce
            nv += n
        return loss / nv

    model.train()
    first = float(step().item())
    for _ in range(40):
        opt.zero_grad()
        loss = step()
        loss.backward()
        opt.step()
    last = float(step().item())
    assert last < first
    assert np.isfinite(last)


# ══════════════════════════════════════════════════════════════════════════════
# Gimmick head loss / recall / training (Task #5c)
# ══════════════════════════════════════════════════════════════════════════════
def test_gimmick_loss_and_recall_counts():
    logits = torch.tensor([[0.0, 5.0],    # predicts mega
                           [5.0, 0.0],    # predicts none
                           [0.0, 5.0]])   # predicts mega
    mask = torch.ones(3, 2)
    target = torch.tensor([1, 1, 0])      # mega(TP), mega(FN), none(FP-ignored)
    valid = torch.ones(3)
    ce, nv, tp, fn = train_bc.gimmick_loss_and_recall(logits, mask, target, valid)
    assert nv == 3 and tp == 1 and fn == 1     # recall = 1/2


def test_compute_gimmick_class_weights_upweights_rare_mega():
    examples = [{"gimmick_targets": {"our_a": 0}} for _ in range(90)]
    examples += [{"gimmick_targets": {"our_a": 1}} for _ in range(10)]
    w, counts = train_bc.compute_gimmick_class_weights(examples, 2, cap=10.0)
    assert counts.tolist() == [90.0, 10.0]
    assert w[1] > w[0]                          # mega up-weighted


def _gim_example(g: int, rng):
    """A learnable example: the gimmick label is encoded in x[0]."""
    x = np.zeros(get_state_dim(), np.float32)
    x[0] = float(g)
    return {
        "x": x, "replay_id": "r",
        "targets": {"our_a": int(rng.integers(0, ACTIONS_PER_SLOT))},
        "masks": {"our_a": np.ones(ACTIONS_PER_SLOT, np.float32)},
        "gimmick_targets": {"our_a": g},
        "gimmick_masks": {"our_a": np.ones(3, np.float32)},
    }


def test_run_epoch_trains_gimmick_head_and_reports_recall():
    from torch.utils.data import DataLoader
    rng = np.random.default_rng(0)
    examples = [_gim_example(i % 2, rng) for i in range(256)]   # x[0] ⇒ gimmick
    loader = DataLoader(BCDataset(examples), batch_size=64, shuffle=True)

    torch.manual_seed(0)
    model = _model(hidden_dims=(32,), dropout=0.0, heads=HEADS)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    m0 = train_bc.run_epoch(model, loader, "cpu", opt)
    for _ in range(15):
        m = train_bc.run_epoch(model, loader, "cpu", opt)

    assert "gimmick_recall" in m and "gimmick_loss" in m and "gimmick_pos" in m
    assert m["gimmick_pos"] > 0
    assert m["gimmick_loss"] < m0["gimmick_loss"]
    assert m["gimmick_recall"] >= 0.9          # learned mega-from-x[0]


def test_run_epoch_handles_no_gimmick_labels():
    """Pre-gimmick data (gimmick_valid all 0) must not crash; the gimmick term is
    zero and the action head still trains."""
    from torch.utils.data import DataLoader
    rng = np.random.default_rng(1)
    examples = []
    for _ in range(64):
        x = np.zeros(get_state_dim(), np.float32)
        examples.append({
            "x": x, "replay_id": "r",
            "targets": {"our_a": int(rng.integers(0, ACTIONS_PER_SLOT))},
            "masks": {"our_a": np.ones(ACTIONS_PER_SLOT, np.float32)},
            # no gimmick_targets / gimmick_masks
        })
    loader = DataLoader(BCDataset(examples), batch_size=32, shuffle=True)
    model = _model(hidden_dims=(16,), dropout=0.0, heads=HEADS)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    m = train_bc.run_epoch(model, loader, "cpu", opt)
    assert m["gimmick_pos"] == 0
    assert m["gimmick_recall"] == 0.0
    assert m["n"] > 0                          # action head still trained


# ══════════════════════════════════════════════════════════════════════════════
# Auxiliary opponent head — training (Task #9d)
# ══════════════════════════════════════════════════════════════════════════════
def test_run_epoch_aux_opp_head_trains_and_reports_top1():
    from torch.utils.data import DataLoader
    rng = np.random.default_rng(0)
    examples = []
    for i in range(256):
        x = np.zeros(get_state_dim(), np.float32)
        x[1] = float(i % 4)                    # x[1] ⇒ opp action (learnable)
        examples.append({
            "x": x, "replay_id": "r",
            "targets": {"our_a": int(rng.integers(0, ACTIONS_PER_SLOT))},
            "masks": {"our_a": np.ones(ACTIONS_PER_SLOT, np.float32)},
            "gimmick_targets": {}, "gimmick_masks": {},
            "opp_targets": {"opp_a": i % 4},
            "opp_masks": {"opp_a": np.ones(ACTIONS_PER_SLOT, np.float32)},
        })
    loader = DataLoader(BCDataset(examples, with_opp=True), batch_size=64, shuffle=True)
    torch.manual_seed(0)
    model = _model(hidden_dims=(64,), dropout=0.0,
                     heads=("our_a", "our_b", "opp_a", "opp_b"))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    m0 = train_bc.run_epoch(model, loader, "cpu", opt, opp_loss_weight=1.0)
    for _ in range(20):
        m = train_bc.run_epoch(model, loader, "cpu", opt, opp_loss_weight=1.0)
    assert "opp_top1" in m and m["opp_n"] > 0
    assert m["opp_top1"] > m0["opp_top1"]      # opp head learned
    assert m["opp_top1"] >= 0.8
    assert m["n"] > 0                          # our head also trained


def test_run_epoch_baseline_model_reports_zero_opp():
    from torch.utils.data import DataLoader
    examples = [{"x": np.zeros(get_state_dim(), np.float32), "replay_id": "r",
                 "targets": {"our_a": 0}, "masks": {"our_a": np.ones(ACTIONS_PER_SLOT, np.float32)},
                 "gimmick_targets": {}, "gimmick_masks": {}} for _ in range(32)]
    loader = DataLoader(BCDataset(examples), batch_size=16, shuffle=True)
    model = _model(hidden_dims=(16,), heads=HEADS)        # no opp heads
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    m = train_bc.run_epoch(model, loader, "cpu", opt, opp_loss_weight=0.3)
    assert m["opp_top1"] == 0.0 and m["opp_n"] == 0
    assert m["n"] > 0                          # our head trains normally


# ══════════════════════════════════════════════════════════════════════════════
# Value head (win probability) — dataset + training (#2)
# ══════════════════════════════════════════════════════════════════════════════
def _value_example(won, x0=0.0):
    return {"x": np.full(get_state_dim(), 0.0, np.float32) if x0 == 0.0
            else np.concatenate([[x0], np.zeros(get_state_dim() - 1, np.float32)]).astype(np.float32),
            "replay_id": "r", "targets": {"our_a": 0},
            "masks": {"our_a": np.ones(ACTIONS_PER_SLOT, np.float32)},
            "gimmick_targets": {}, "gimmick_masks": {}, "won": won}


def test_value_target_from_won():
    dset = BCDataset([_value_example(True), _value_example(False), _value_example(None)])
    assert dset[0]["value_target"].item() == 1.0 and dset[0]["value_valid"].item() == 1.0
    assert dset[1]["value_target"].item() == 0.0 and dset[1]["value_valid"].item() == 1.0
    assert dset[2]["value_valid"].item() == 0.0          # unknown outcome → no signal


def test_run_epoch_trains_value_head():
    from torch.utils.data import DataLoader
    rng = np.random.default_rng(0)
    examples = []
    for i in range(256):
        won = bool(i % 2)
        ex = _value_example(won, x0=1.0 if won else -1.0)   # outcome encoded in x[0]
        ex["targets"] = {"our_a": int(rng.integers(0, ACTIONS_PER_SLOT))}
        examples.append(ex)
    loader = DataLoader(BCDataset(examples), batch_size=64, shuffle=True)
    torch.manual_seed(0)
    model = _model(hidden_dims=(32,), dropout=0.0, heads=HEADS)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    m0 = train_bc.run_epoch(model, loader, "cpu", opt)
    for _ in range(20):
        m = train_bc.run_epoch(model, loader, "cpu", opt)
    assert "value_acc" in m and m["value_n"] > 0
    assert m["value_loss"] < m0["value_loss"]
    assert m["value_acc"] >= 0.95                          # learned win-from-x[0]
    assert m["value_brier"] < m0["value_brier"]


def test_run_epoch_handles_no_value_labels():
    """Pre-value data (no `won`) → value_valid all 0; the value term is zero and
    the action head still trains."""
    from torch.utils.data import DataLoader
    examples = [{"x": np.zeros(get_state_dim(), np.float32), "replay_id": "r",
                 "targets": {"our_a": 0}, "masks": {"our_a": np.ones(ACTIONS_PER_SLOT, np.float32)},
                 "gimmick_targets": {}, "gimmick_masks": {}} for _ in range(32)]
    loader = DataLoader(BCDataset(examples), batch_size=16, shuffle=True)
    model = _model(hidden_dims=(16,), heads=HEADS)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    m = train_bc.run_epoch(model, loader, "cpu", opt)
    assert m["value_n"] == 0
    assert m["n"] > 0                                      # action head still trained


# ══════════════════════════════════════════════════════════════════════════════
# End-to-end smoke train on synthetic JSONL
# ══════════════════════════════════════════════════════════════════════════════
def _write_synthetic_corpus(folder: Path, n_replays: int = 12, per_replay: int = 6):
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for r in range(n_replays):
        lines = []
        for k in range(per_replay):
            ai_a = int(rng.integers(0, ACTIONS_PER_SLOT))
            ai_b = int(rng.integers(0, ACTIONS_PER_SLOT))
            t = make_transition(
                f"replay{r}",
                our_actions=[
                    {"slot": "our_a", "action": "move", "move": "Fake Out", "action_index": ai_a},
                    {"slot": "our_b", "action": "move", "move": "Protect", "action_index": ai_b},
                ],
                mask_a=_row(*range(ACTIONS_PER_SLOT)),
                mask_b=_row(*range(ACTIONS_PER_SLOT)),
                turn=k + 1,
            )
            lines.append(json.dumps(t))
        (folder / f"replay{r}.jsonl").write_text("\n".join(lines), encoding="utf-8")


def test_end_to_end_smoke_train(tmp_path):
    corpus = tmp_path / "jsonl"
    _write_synthetic_corpus(corpus)
    out = tmp_path / "ckpt"
    args = train_bc.parse_args([
        "--data", str(corpus),
        "--epochs", "2",
        "--batch-size", "16",
        "--d-model", "32", "--n-layers", "1",
        "--device", "cpu",
        "--val-frac", "0.25",
        "--out", str(out),
    ])
    result = train_bc.train(args)
    assert len(result["history"]) == 2
    assert np.isfinite(result["history"][-1]["val"]["loss"])
    assert (out / "battle_base.pt").exists()
    assert (out / "history.json").exists()
    # checkpoint reloads and carries config
    ckpt = torch.load(out / "battle_base.pt", map_location="cpu", weights_only=False)
    assert ckpt["config"]["state_dim"] == get_state_dim()
    assert ckpt["config"]["heads"] == list(HEADS)


# ══════════════════════════════════════════════════════════════════════════════
# Class-imbalance weighting + early stopping
# ══════════════════════════════════════════════════════════════════════════════
def test_compute_class_weights_upweights_rare_and_caps():
    exs = [{"targets": {"our_a": 0}} for _ in range(100)]
    exs.append({"targets": {"our_a": 5}})            # one rare action
    w, counts = train_bc.compute_class_weights(exs, ACTIONS_PER_SLOT, cap=10.0)
    assert w.shape == (ACTIONS_PER_SLOT,)
    assert counts[0] == 100 and counts[5] == 1
    assert w[5] > w[0]                                # rare action up-weighted
    assert w.max() <= 10.0 + 1e-6                     # capped
    assert w[1] == 1.0                               # absent class -> neutral


def test_smoke_train_with_class_weight_and_patience(tmp_path):
    corpus = tmp_path / "jsonl"
    _write_synthetic_corpus(corpus)
    out = tmp_path / "ckpt"
    args = train_bc.parse_args([
        "--data", str(corpus), "--epochs", "3", "--batch-size", "16",
        "--d-model", "32", "--n-layers", "1", "--device", "cpu", "--val-frac", "0.25",
        "--out", str(out), "--class-weight", "balanced", "--patience", "2",
    ])
    res = train_bc.train(args)
    assert (out / "battle_base.pt").exists()
    assert np.isfinite(res["history"][-1]["val"]["loss"])
    ck = torch.load(out / "battle_base.pt", map_location="cpu", weights_only=False)
    assert ck["config"]["class_weight"] == "balanced"
    assert ck["config"]["patience"] == 2


def test_smoke_train_with_rating_and_outcome_weighting(tmp_path):
    """End-to-end train with --rating-min + --rating-weight + --outcome-weight on a
    rated synthetic corpus: runs, filters, persists the config knobs."""
    corpus = tmp_path / "jsonl"
    corpus.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for r in range(12):
        rating = 1500 + 50 * r            # spread 1500..2050; >=1700 keeps 8 replays
        won = (r % 2 == 0)
        lines = []
        for k in range(6):
            t = _rated_transition(f"replay{r}", rating, 20 if won else -20, won)
            t["turn"] = k + 1
            t["our_actions"] = [
                {"slot": "our_a", "action": "move", "move": "Fake Out",
                 "action_index": int(rng.integers(0, ACTIONS_PER_SLOT))},
                {"slot": "our_b", "action": "move", "move": "Protect",
                 "action_index": int(rng.integers(0, ACTIONS_PER_SLOT))},
            ]
            t["action_mask"] = {"our_a": _row(*range(ACTIONS_PER_SLOT)),
                                "our_b": _row(*range(ACTIONS_PER_SLOT))}
            lines.append(json.dumps(t))
        (corpus / f"replay{r}.jsonl").write_text("\n".join(lines), encoding="utf-8")
    out = tmp_path / "ckpt"
    args = train_bc.parse_args([
        "--data", str(corpus), "--epochs", "2", "--batch-size", "16",
        "--d-model", "32", "--n-layers", "1", "--device", "cpu", "--val-frac", "0.25", "--out", str(out),
        "--rating-min", "1700", "--rating-weight", "--outcome-weight",
        "--loss-weight", "0.5", "--seed", "0",
    ])
    res = train_bc.train(args)
    assert (out / "battle_base.pt").exists()
    assert np.isfinite(res["history"][-1]["val"]["loss"])
    ck = torch.load(out / "battle_base.pt", map_location="cpu", weights_only=False)
    cfg = ck["config"]
    assert cfg["rating_min"] == 1700 and cfg["rating_weight"] is True
    assert cfg["outcome_weight"] is True and cfg["loss_weight"] == 0.5


def test_patience_stops_before_max_epochs(tmp_path):
    # tiny random-label set plateaus fast → patience must cut the run short
    corpus = tmp_path / "jsonl"
    _write_synthetic_corpus(corpus, n_replays=6, per_replay=4)
    out = tmp_path / "ckpt"
    args = train_bc.parse_args([
        "--data", str(corpus), "--epochs", "60", "--batch-size", "16",
        "--d-model", "32", "--n-layers", "1", "--device", "cpu", "--val-frac", "0.34",
        "--out", str(out), "--patience", "3", "--seed", "0",
    ])
    res = train_bc.train(args)
    assert len(res["history"]) < 60                  # early-stopped


def test_default_out_is_the_bc_anchor_home_the_configs_read():
    # rename-wiring: a bare BC retrain writes battle_base.pt; its ONLY consumer is the KL-to-BC anchor
    # (the self-play configs' 'ckpt'), so the default --out must be that anchor's home
    # (checkpoints_attn_pre_gen141/), NOT the prod serving dir checkpoints_attn/ (the self-play champion).
    out = Path(train_bc.parse_args([]).out)
    assert out.name == "checkpoints_attn_pre_gen141"
    # trainer output dir and the config anchor dir MUST agree — no silent orphan mismatch.
    # run-configs live under config/ (gitignored, local); skip gracefully if this checkout lacks them.
    _cfg_f = Path(__file__).resolve().parents[1] / "config" / "config_v11_selfplay.json"
    if _cfg_f.exists():
        cfg = json.loads(_cfg_f.read_text())
        assert Path(cfg["generation"]["ckpt"]).resolve().parent == out.resolve()
    # explicit --out always wins
    assert train_bc.parse_args(["--out", "zzz"]).out == "zzz"


# ══════════════════════════════════════════════════════════════════════════════
# Per-situation eval + calibration diagnostic (eval_buckets.py)
# ══════════════════════════════════════════════════════════════════════════════
import v_dance.training.eval_buckets as ev  # noqa: E402


def test_phase_of_buckets():
    assert ev.phase_of(1, "turn") == "lead"
    assert ev.phase_of(2, "turn") == "lead"
    assert ev.phase_of(4, "turn") == "mid"
    assert ev.phase_of(9, "turn") == "endgame"
    assert ev.phase_of(9, "replacement") == "replacement"   # decision_type wins
    assert ev.phase_of(None, "turn") == "lead"


def test_rating_band_buckets():
    assert ev.rating_band(1650) == "<1700"
    assert ev.rating_band(1750) == "1700-1800"
    assert ev.rating_band(1850) == "1800-1900"
    assert ev.rating_band(2010) == "1900+"
    assert ev.rating_band(None) == "unknown"


def test_masked_softmax_zeros_illegal_and_normalises():
    logits = np.array([5.0, 1.0, 9.0, 2.0], np.float32)   # idx2 huge but illegal
    mask = np.array([1, 1, 0, 1], np.float32)
    p = ev.masked_softmax(logits, mask)
    assert p[2] == 0.0
    assert abs(p.sum() - 1.0) < 1e-6
    assert int(np.argmax(p)) == 0                          # best LEGAL


def test_ece_zero_when_perfectly_calibrated_and_max_when_wrong():
    # all confident-0.9 and all correct → acc 0.9 == conf 0.9 → ECE 0
    ece, _ = ev.expected_calibration_error([0.9] * 100, [True] * 90 + [False] * 10)
    assert ece < 0.05
    # all confident-0.9 but all WRONG → |0 - 0.9| = 0.9
    ece2, _ = ev.expected_calibration_error([0.9] * 50, [False] * 50)
    assert abs(ece2 - 0.9) < 1e-6


def test_evaluate_end_to_end_on_synthetic_model():
    model = _model(hidden_dims=(16,), dropout=0.0, heads=HEADS)
    mask = np.ones(ACTIONS_PER_SLOT, np.float32)
    examples = []
    for i in range(20):
        examples.append({
            "x": np.zeros(get_state_dim(), np.float32), "replay_id": f"r{i}",
            "targets": {"our_a": i % ACTIONS_PER_SLOT}, "masks": {"our_a": mask},
            "turn": (i % 8) + 1, "decision_type": "turn" if i % 4 else "replacement",
            "rating": 1750 + i, "won": bool(i % 2),
        })
    rep = ev.evaluate(model, examples)
    assert rep["n_decisions"] == 20
    assert 0.0 <= rep["top1"] <= 1.0 and rep["top3"] >= rep["top1"]
    assert 0.0 <= rep["ece"] <= 1.0
    assert set(rep["by_phase"]) <= {"lead", "mid", "endgame", "replacement"}
    assert sum(b["n"] for b in rep["by_phase"].values()) == 20
    # value head bucketed by phase (#2)
    assert rep["value_n"] == 20
    assert 0.0 <= rep["value_win_acc"] <= 1.0
    assert sum(b["n"] for b in rep["value_by_phase"].values()) == 20


# ══════════════════════════════════════════════════════════════════════════════
# Opt-in: real Type B corpus (skipped if absent)
# ══════════════════════════════════════════════════════════════════════════════
_TYPE_B = (
    ds._SCRIPTS_DIR.parent          # <repo>/data/scripts -> <repo>/data
    / "vods" / "Prepared_training_data" / "Regulation_MA" / "Jsonl_TypeB"
)


@pytest.mark.skipif(not _TYPE_B.exists(), reason="Type B corpus not present")
def test_real_typeb_examples_are_legal():
    files = ds.iter_jsonl_files(str(_TYPE_B))[:20]
    examples, stats = build_examples(files)
    assert stats["usable_examples"] > 0
    assert len(examples) > 0
    # every kept target is legal under its own mask, and X is finite
    for ex in examples:
        assert np.isfinite(ex["x"]).all()
        for head, ai in ex["targets"].items():
            assert ex["masks"][head][ai] == 1.0
