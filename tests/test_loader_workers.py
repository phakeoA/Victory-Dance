"""M6 loader-workers (2026-07-11): BCDataset worker-view pickling + train_bc feed workers.

Contract: workers receive tensors via torch shared memory (numpy twins dropped and
rebuilt as tensor views); lazy (mmap) datasets NEVER pickle their memmap row views —
workers re-open the encoded caches via the ``x_src`` (path, row) stamps
``encoded_cache.load_cache`` writes; a lazy dataset without stamps fails LOUD at
pickle time; workers=0 stays the byte-identical legacy path.
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

import v_dance.training.bc_dataset as bcd
from v_dance.parser.vod_parser.transitions import log_to_transitions
from v_dance.training.bc_dataset import (
    ACTIONS_PER_SLOT, BCDataset, HEADS, bc_worker_init,
)
from v_dance.training.encoded_cache import cached_examples_from_folders

from test_hf_ots_ingest import _OTS_LOG


@pytest.fixture()
def corpus_dir(tmp_path):
    """Two real battles' worth of jsonl (the encoded-cache test fixture pattern)."""
    for rid in ("test-lw-1", "test-lw-2"):
        ts = log_to_transitions(_OTS_LOG, rid, players=["p1", "p2"])
        (tmp_path / f"{rid}.jsonl").write_text(
            "\n".join(json.dumps(t, ensure_ascii=False) for t in ts), encoding="utf-8")
    return tmp_path


def _synthetic_examples(n=8, with_src=False, tmp_path=None):
    dim = bcd.get_state_dim()
    rng = np.random.default_rng(0)
    X = rng.random((n, dim)).astype(np.float32)
    if with_src:
        xp = tmp_path / "X.npy"
        np.save(xp, X)
        X = np.load(xp, mmap_mode="r")
    exs = []
    for i in range(n):
        ex = {
            "x": X[i],
            "targets": {HEADS[0]: int(i % ACTIONS_PER_SLOT)},
            "masks": {HEADS[0]: np.ones(ACTIONS_PER_SLOT, dtype=np.float32)},
            "replay_id": f"r{i // 2}",
            "won": bool(i % 2),
        }
        if with_src:
            ex["x_src"] = (str(tmp_path / "X.npy"), i)
        exs.append(ex)
    return exs


def _items_equal(a: dict, b: dict):
    assert a.keys() == b.keys()
    for k in a:
        assert torch.equal(a[k], b[k]), f"field {k} differs"


def test_load_cache_stamps_x_src(corpus_dir):
    exs, _ = cached_examples_from_folders([str(corpus_dir)], mmap=True)   # miss → build → load
    assert exs
    for i, ex in enumerate(exs[:5]):
        path, row = ex["x_src"]
        arr = np.load(path, mmap_mode="r")
        assert np.array_equal(np.asarray(arr[row]), np.asarray(ex["x"]))


def test_worker_view_roundtrip_lazy(corpus_dir):
    exs, _ = cached_examples_from_folders([str(corpus_dir)], mmap=True)
    ds = BCDataset(exs, lazy_x=True)
    assert ds.workers_safe
    ds2 = pickle.loads(pickle.dumps(ds))
    assert ds2._x_rows is None and ds2._x_open == {}      # views never cross; re-open lazily
    assert ds2.replay_ids == []                            # worker view drops bookkeeping
    for idx in (0, len(ds) - 1):
        _items_equal(ds[idx], ds2[idx])
    # numpy twins rebuilt as views of the (shareable) tensors, not copies
    assert np.shares_memory(ds2.target, ds2.target_t.numpy())


def test_worker_view_roundtrip_nonlazy():
    exs = _synthetic_examples()
    ds = BCDataset(exs)
    assert ds.workers_safe
    ds2 = pickle.loads(pickle.dumps(ds))
    assert ds2.X is not None and np.shares_memory(ds2.X, ds2.X_t.numpy())
    for idx in range(len(ds)):
        _items_equal(ds[idx], ds2[idx])


def test_lazy_without_stamps_fails_loud():
    exs = _synthetic_examples()                            # plain in-RAM x, no x_src
    ds = BCDataset(exs, lazy_x=True)
    assert not ds.workers_safe
    with pytest.raises(RuntimeError, match="x_src"):
        pickle.dumps(ds)
    assert ds[0]["x"].shape[0] == bcd.get_state_dim()      # in-process fetch still fine


def test_dataloader_worker_matches_inprocess(corpus_dir):
    """One REAL spawned worker must yield byte-identical batches to the in-process
    path (aug off, shuffle off → deterministic order either way)."""
    exs, _ = cached_examples_from_folders([str(corpus_dir)], mmap=True)
    ds = BCDataset(exs, lazy_x=True)
    plain = list(DataLoader(ds, batch_size=4, shuffle=False))
    worked = list(DataLoader(ds, batch_size=4, shuffle=False, num_workers=1,
                             worker_init_fn=bc_worker_init, persistent_workers=True))
    assert len(plain) == len(worked)
    for b0, b1 in zip(plain, worked):
        _items_equal(b0, b1)


def test_dataloader_worker_augmented_smoke():
    """Augmented fetch through a real worker: runs, shapes hold, targets stay
    legal under their (permuted) masks. Exact bytes differ from in-process by
    design (bc_worker_init reseeds the worker's aug RNG)."""
    exs = _synthetic_examples()
    ds = BCDataset(exs, augment_move_order=True, aug_seed=0)
    (batch,) = list(DataLoader(ds, batch_size=len(exs), shuffle=False, num_workers=1,
                               worker_init_fn=bc_worker_init))
    assert batch["x"].shape == (len(exs), bcd.get_state_dim())
    assert torch.isfinite(batch["x"]).all()
    for r in range(len(exs)):
        for h in range(len(HEADS)):
            t = int(batch["target"][r, h])
            if t >= 0 and float(batch["valid"][r, h]) > 0.5:
                assert float(batch["mask"][r, h, t]) > 0.5   # remapped target stays legal
