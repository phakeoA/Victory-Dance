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
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import bc_dataset as ds
from bc_dataset import (
    HEADS,
    BCDataset,
    build_examples,
    split_by_replay,
    transition_to_example,
)
from state_encoder import get_state_dim, ACTIONS_PER_SLOT

torch = pytest.importorskip("torch")
from bc_model import BCPolicy, build_model  # noqa: E402
import train_bc  # noqa: E402


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
# Model
# ══════════════════════════════════════════════════════════════════════════════
def test_model_forward_shapes():
    model = BCPolicy(hidden_dims=(64, 32), heads=HEADS)
    x = torch.randn(5, get_state_dim())
    out = model(x)
    assert set(out.keys()) == set(HEADS)
    for h in HEADS:
        assert out[h].shape == (5, ACTIONS_PER_SLOT)
    assert model.count_parameters() > 0


def test_masked_logits_blocks_illegal():
    logits = torch.zeros(2, ACTIONS_PER_SLOT)
    logits[:, 5] = 10.0          # would-be argmax
    mask = torch.zeros(2, ACTIONS_PER_SLOT)
    mask[:, 3] = 1.0             # only index 3 legal
    ml = train_bc.masked_logits(logits, mask)
    assert ml.argmax(dim=1).tolist() == [3, 3]


def test_one_optim_step_reduces_loss():
    torch.manual_seed(0)
    model = BCPolicy(hidden_dims=(64, 32), dropout=0.0, heads=HEADS)
    x = torch.randn(16, get_state_dim())
    mask = torch.ones(16, len(HEADS), ACTIONS_PER_SLOT)
    target = torch.randint(0, ACTIONS_PER_SLOT, (16, len(HEADS)))
    valid = torch.ones(16, len(HEADS))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    def step():
        out = model(x)
        loss = x.new_zeros(())
        nv = 0
        for hi, h in enumerate(HEADS):
            ce, n, _, _ = train_bc.head_loss_and_acc(
                out[h], mask[:, hi], target[:, hi], valid[:, hi]
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
        "--hidden", "64", "32",
        "--device", "cpu",
        "--val-frac", "0.25",
        "--out", str(out),
    ])
    result = train_bc.train(args)
    assert len(result["history"]) == 2
    assert np.isfinite(result["history"][-1]["val"]["loss"])
    assert (out / "bc_best.pt").exists()
    assert (out / "history.json").exists()
    # checkpoint reloads and carries config
    ckpt = torch.load(out / "bc_best.pt", map_location="cpu", weights_only=False)
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
        "--hidden", "64", "32", "--device", "cpu", "--val-frac", "0.25",
        "--out", str(out), "--class-weight", "balanced", "--patience", "2",
    ])
    res = train_bc.train(args)
    assert (out / "bc_best.pt").exists()
    assert np.isfinite(res["history"][-1]["val"]["loss"])
    ck = torch.load(out / "bc_best.pt", map_location="cpu", weights_only=False)
    assert ck["config"]["class_weight"] == "balanced"
    assert ck["config"]["patience"] == 2


def test_patience_stops_before_max_epochs(tmp_path):
    # tiny random-label set plateaus fast → patience must cut the run short
    corpus = tmp_path / "jsonl"
    _write_synthetic_corpus(corpus, n_replays=6, per_replay=4)
    out = tmp_path / "ckpt"
    args = train_bc.parse_args([
        "--data", str(corpus), "--epochs", "60", "--batch-size", "16",
        "--hidden", "32", "--device", "cpu", "--val-frac", "0.34",
        "--out", str(out), "--patience", "3", "--seed", "0",
    ])
    res = train_bc.train(args)
    assert len(res["history"]) < 60                  # early-stopped


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
