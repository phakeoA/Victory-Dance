"""encoded_cache — the 35-minute-load fix. The cache MUST be invisible:
identical example stream (order + every field byte-equal) from cache or fresh,
automatic fingerprint invalidation, smoke-limit bypass."""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pytest

from v_dance.parser.vod_parser.transitions import log_to_transitions
from v_dance.training.bc_dataset import examples_from_folders
from v_dance.training.encoded_cache import (
    cached_examples_from_folders,
    folder_fingerprint,
    load_cache,
)

from test_hf_ots_ingest import _OTS_LOG


@pytest.fixture()
def corpus_dir(tmp_path):
    """Two real battles' worth of jsonl (the OTS test log parsed twice)."""
    for i, rid in enumerate(("test-cache-1", "test-cache-2")):
        ts = log_to_transitions(_OTS_LOG, rid, players=["p1", "p2"])
        (tmp_path / f"{rid}.jsonl").write_text(
            "\n".join(json.dumps(t, ensure_ascii=False) for t in ts), encoding="utf-8")
    return tmp_path


def _assert_examples_equal(a, b, with_opp):
    assert len(a) == len(b)
    for ea, eb in zip(a, b):
        assert np.array_equal(ea["x"], eb["x"])
        assert ea["targets"] == {k: int(v) for k, v in eb["targets"].items()}
        for k in ea["masks"]:
            assert np.array_equal(ea["masks"][k], eb["masks"][k])
        assert ea["gimmick_targets"] == eb["gimmick_targets"]
        for k in ("replay_id", "perspective", "rating", "rating_delta",
                  "won", "turn", "decision_type"):
            assert ea[k] == eb[k], f"{k}: {ea[k]!r} != {eb[k]!r}"
        if with_opp:
            assert ea.get("opp_targets") == eb.get("opp_targets")
        else:
            assert "opp_targets" not in eb


@pytest.mark.parametrize("with_opp", [True, False])
def test_cache_roundtrip_is_invisible(corpus_dir, with_opp):
    fresh, fresh_stats = examples_from_folders([str(corpus_dir)], with_opp=with_opp)
    assert fresh, "fixture produced no examples"
    # MISS → build + save
    built, stats1 = cached_examples_from_folders([str(corpus_dir)], with_opp=with_opp)
    _assert_examples_equal(fresh, built, with_opp)
    # HIT → reconstruct from disk
    hit, stats2 = cached_examples_from_folders([str(corpus_dir)], with_opp=with_opp)
    _assert_examples_equal(fresh, hit, with_opp)
    assert stats2["transitions"] == fresh_stats["transitions"]
    assert stats2["replays"] == fresh_stats["replays"]


def test_cache_invalidates_on_corpus_change(corpus_dir):
    cached_examples_from_folders([str(corpus_dir)])
    fp1 = folder_fingerprint(str(corpus_dir))
    assert load_cache(str(corpus_dir), with_opp=False) is not None
    # touch one file (content growth) → new fingerprint → old cache unmatched
    f = sorted(corpus_dir.glob("*.jsonl"))[0]
    with open(f, "a", encoding="utf-8") as fh:
        fh.write("\n")
    os.utime(f)                        # ensure mtime moves even on coarse clocks
    fp2 = folder_fingerprint(str(corpus_dir))
    assert fp1 != fp2
    assert load_cache(str(corpus_dir), with_opp=False) is None


def test_smoke_limits_bypass_cache(corpus_dir):
    limited, _ = cached_examples_from_folders([str(corpus_dir)], limit_files=1)
    fresh, _ = examples_from_folders([str(corpus_dir)], limit_files=1)
    assert len(limited) == len(fresh)
    # a bypassed call must not have created a cache
    assert load_cache(str(corpus_dir), with_opp=False) is None


def test_duplicate_folder_listed_twice_counted_once(corpus_dir):
    once, _ = cached_examples_from_folders([str(corpus_dir)])
    twice, _ = cached_examples_from_folders([str(corpus_dir), str(corpus_dir)])
    assert len(once) == len(twice)


def test_split_identical_from_cache_or_fresh(corpus_dir):
    """The A/B-critical property: identical splits either way."""
    from v_dance.training.bc_dataset import split_by_replay
    fresh, _ = examples_from_folders([str(corpus_dir)])
    cached_examples_from_folders([str(corpus_dir)])          # warm
    hit, _ = cached_examples_from_folders([str(corpus_dir)])
    for src in (fresh, hit):
        tr, va = split_by_replay(src, val_frac=0.5, seed=0)
        assert {e["replay_id"] for e in va} == \
            {e["replay_id"] for e in split_by_replay(fresh, val_frac=0.5, seed=0)[1]}


# ══════════════════════════════════════════════════════════════════════════════
# Step B (2026-07-02): streaming chunked build + mmap load + lazy BCDataset.
# The whole point is byte-identity with the in-RAM path at a fraction of the RAM.
# ══════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("with_opp", [True, False])
def test_mmap_hit_roundtrip_identical(corpus_dir, with_opp):
    fresh, fresh_stats = examples_from_folders([str(corpus_dir)], with_opp=with_opp)
    cached_examples_from_folders([str(corpus_dir)], with_opp=with_opp)   # warm
    hit, stats = cached_examples_from_folders(
        [str(corpus_dir)], with_opp=with_opp, mmap=True)
    _assert_examples_equal(fresh, hit, with_opp)
    assert stats["transitions"] == fresh_stats["transitions"]
    assert stats["replays"] == fresh_stats["replays"]
    # the mmap path must actually engage: rows are read-only views of the disk X
    assert hit[0]["x"].flags.writeable is False
    assert fresh[0]["x"].flags.writeable is True


def test_streaming_chunked_build_identical(corpus_dir, monkeypatch):
    """Chunk boundaries must be invisible: a 1-file-per-chunk build produces the
    same example stream (and stats) as the single-chunk build / fresh encode."""
    import v_dance.training.encoded_cache as ec
    fresh, fresh_stats = examples_from_folders([str(corpus_dir)], with_opp=True)
    monkeypatch.setattr(ec, "_STREAM_CHUNK_FILES", 1)
    built, stats = cached_examples_from_folders([str(corpus_dir)], with_opp=True)
    _assert_examples_equal(fresh, built, True)
    assert stats["transitions"] == fresh_stats["transitions"]
    assert stats["replays"] == fresh_stats["replays"]
    # and the HIT of that chunk-built cache is identical too (incl. mmap'd)
    hit, _ = cached_examples_from_folders([str(corpus_dir)], with_opp=True, mmap=True)
    _assert_examples_equal(fresh, hit, True)


def _assert_dataset_items_equal(ds_a, ds_b):
    import torch
    assert len(ds_a) == len(ds_b)
    for i in range(len(ds_a)):
        ia, ib = ds_a[i], ds_b[i]
        assert set(ia) == set(ib)
        for k in ia:
            assert torch.equal(ia[k], ib[k]), f"item {i} key {k!r} differs"


def test_bcdataset_lazy_x_items_identical(corpus_dir):
    """lazy_x (--mmap-cache) items must be byte-identical to the eager dataset,
    on both the plain and the augmented (same aug_seed → same RNG) fetch path."""
    from v_dance.training.bc_dataset import BCDataset
    cached_examples_from_folders([str(corpus_dir)], with_opp=True)       # warm
    eager, _ = cached_examples_from_folders([str(corpus_dir)], with_opp=True)
    lazy, _ = cached_examples_from_folders([str(corpus_dir)], with_opp=True, mmap=True)
    _assert_dataset_items_equal(
        BCDataset(eager, with_opp=True),
        BCDataset(lazy, with_opp=True, lazy_x=True))
    _assert_dataset_items_equal(
        BCDataset(eager, with_opp=True, augment_move_order=True, aug_seed=7),
        BCDataset(lazy, with_opp=True, augment_move_order=True, aug_seed=7,
                  lazy_x=True))


def test_lazy_x_rejects_sequence_mode(corpus_dir):
    """No silent defer: sequence BC needs the in-RAM X matrix, so lazy_x must
    refuse it loudly instead of crashing later (or silently degrading)."""
    from v_dance.training.bc_dataset import BCDataset
    cached_examples_from_folders([str(corpus_dir)])                      # warm
    ex, _ = cached_examples_from_folders([str(corpus_dir)], mmap=True)
    with pytest.raises(ValueError, match="sequence_len"):
        BCDataset(ex, sequence_len=4, lazy_x=True)
