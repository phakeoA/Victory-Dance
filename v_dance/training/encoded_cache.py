"""
encoded_cache.py — disk cache of ENCODED BC examples (the 35-minute-load fix)
=============================================================================
``examples_from_folders`` re-encodes every stored snapshot on EVERY trainer /
ruler launch (~35 min for the ~580k-transition corpus, single-threaded, GPU
idle).  The JSONL corpus is the durable source of truth and rarely changes
between runs — so cache the ENCODED examples per corpus folder and reload in
seconds.

Design
------
* One cache per (folder contents, encoder layout): the fingerprint hashes the
  STATE layout version + every ``*.jsonl`` (relpath, size, mtime_ns).  Any
  added/changed/removed replay or an encoder-layout bump ⇒ different
  fingerprint ⇒ automatic MISS + rebuild.  No manual invalidation, ever.
* Layout on disk (``<folder>/.encoded_cache/<fingerprint>/``):
    X.npy    — (N, STATE_DIM) float32, the big array (np.load-able with
               mmap_mode='r' later — the MA-BO3 low-RAM loader foundation)
    meta.npz — every per-example small field, fixed-shape arrays
    stats.json + DONE marker — build stats + atomicity flag
  The build writes into a ``.tmp-`` dir and os.rename()s it into place, so a
  killed build can never be mistaken for a complete cache; older fingerprint
  dirs are swept on save.
* ``cached_examples_from_folders`` is a drop-in for ``examples_from_folders``
  (same return contract) — reconstruction preserves EXAMPLE ORDER and byte
  identity of every field (tested), so split_by_replay / split_with_reference
  produce the SAME splits from cache or fresh.
* Examples are always CACHED with the aux-opp fields (cheap at build time);
  ``with_opp=False`` callers just get the opp keys dropped on reconstruction.
* ``limit_files``/``limit_transitions`` (smoke runs) bypass the cache entirely.

RAM note (Step B, 2026-07-02): the build STREAMS — the folder is encoded in
2000-file chunks and each chunk's X rows are appended straight to disk and
freed, so peak build RAM is ~one chunk + the small meta arrays (never the full
(N, STATE_DIM) matrix — 26 GB for the full 77.8k MA-Bo3 set).  Loading takes
``mmap=True`` to page X from disk (``np.load(..., mmap_mode='r')``; example
dicts hold read-only row views) — pair with ``BCDataset(lazy_x=True)`` /
``train_bc --mmap-cache`` for the low-RAM training path.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from v_dance.encoders.state_encoder import (
    ACTIONS_PER_SLOT,
    OPP_HEADS,
    StateEncoder,
    get_state_dim,
    get_state_layout_version,
)
from v_dance.training.bc_dataset import (
    GIMMICK_DIM,
    HEADS,
    build_examples,
    examples_from_folders,
    iter_jsonl_files,
)

_CACHE_DIRNAME = ".encoded_cache"
_CACHE_SCHEMA = 1          # bump when the serialized example schema changes
_STREAM_CHUNK_FILES = 2000  # ~30-60k examples ≈ 1-2 GB peak per build chunk

# fixed-width string dtypes (replay ids are showdown battle ids ≤ ~64 chars)
_RID_DT = "<U80"
_PERSP_DT = "<U4"
_DTYPE_DT = "<U16"


def folder_fingerprint(folder: str) -> str:
    """Content fingerprint of a corpus folder for cache keying."""
    h = hashlib.sha1()
    h.update(f"schema={_CACHE_SCHEMA};layout={get_state_layout_version()};"
             f"dim={get_state_dim()}".encode())
    base = Path(folder)
    for f in iter_jsonl_files(str(folder)):
        st = os.stat(f)
        rel = os.path.relpath(f, base)
        h.update(f"{rel}|{st.st_size}|{st.st_mtime_ns};".encode())
    return h.hexdigest()[:20]


def _cache_dir(folder: str) -> Path:
    return Path(folder) / _CACHE_DIRNAME


def _new_meta_arrays(n: int) -> dict:
    """Fresh per-example meta arrays for ``n`` examples (the meta.npz schema)."""
    return {
        "target": np.full((n, len(HEADS)), -1, dtype=np.int64),
        "mask": np.zeros((n, len(HEADS), ACTIONS_PER_SLOT), dtype=np.float32),
        "g_target": np.full((n, len(HEADS)), -1, dtype=np.int64),
        "g_mask": np.zeros((n, len(HEADS), GIMMICK_DIM), dtype=np.float32),
        "o_target": np.full((n, len(OPP_HEADS)), -1, dtype=np.int64),
        "o_mask": np.zeros((n, len(OPP_HEADS), ACTIONS_PER_SLOT), dtype=np.float32),
        "rid": np.empty(n, dtype=_RID_DT),
        "persp": np.empty(n, dtype=_PERSP_DT),
        "dtype": np.empty(n, dtype=_DTYPE_DT),
        "rating": np.full(n, np.nan, dtype=np.float64),
        "rating_delta": np.zeros(n, dtype=np.float64),
        "won": np.full(n, -1, dtype=np.int8),        # -1 unknown / 0 lost / 1 won
        "turn": np.zeros(n, dtype=np.int32),
    }


def _fill_meta_row(m: dict, i: int, ex: dict) -> None:
    """Serialize one example's small fields into row ``i`` of the meta arrays."""
    for h_idx, head in enumerate(HEADS):
        if head in ex["targets"]:
            m["target"][i, h_idx] = ex["targets"][head]
            m["mask"][i, h_idx] = ex["masks"][head]
        gt = ex.get("gimmick_targets") or {}
        if head in gt:
            m["g_target"][i, h_idx] = gt[head]
            m["g_mask"][i, h_idx] = (ex.get("gimmick_masks") or {})[head]
    for o_idx, ohead in enumerate(OPP_HEADS):
        ot = ex.get("opp_targets") or {}
        if ohead in ot:
            m["o_target"][i, o_idx] = ot[ohead]
            m["o_mask"][i, o_idx] = (ex.get("opp_masks") or {})[ohead]
    m["rid"][i] = ex.get("replay_id") or ""
    m["persp"][i] = ex.get("perspective") or ""
    m["dtype"][i] = ex.get("decision_type") or ""
    if ex.get("rating") is not None:
        m["rating"][i] = float(ex["rating"])
    m["rating_delta"][i] = float(ex.get("rating_delta") or 0.0)
    if ex.get("won") is not None:
        m["won"][i] = 1 if ex["won"] else 0
    m["turn"][i] = int(ex.get("turn") or 0)


def build_cache_streaming(folder: str, chunk_files: Optional[int] = None) -> Tuple[Path, Counter]:
    """Chunked low-RAM cache build for ``folder`` (Step B, 2026-07-02); atomic.

    Encodes the folder's jsonl files in ``chunk_files``-file chunks; each
    chunk's X rows are appended to a raw ``X.bin`` on disk and the chunk is
    freed, so peak RAM is ~one chunk + the accumulated meta arrays — never the
    full (N, STATE_DIM) matrix.  ``X.npy`` is finalized as npy-header +
    raw-row copy, byte-identical in format to an ``np.save`` (both plain
    ``np.load`` and ``mmap_mode='r'`` work on it).  Examples are always built
    WITH the aux-opp fields (one cache serves both with_opp modes).
    """
    if chunk_files is None:
        chunk_files = _STREAM_CHUNK_FILES
    dim = get_state_dim()
    fp = folder_fingerprint(folder)
    final = _cache_dir(folder) / fp
    tmp = _cache_dir(folder) / f".tmp-{fp}-{os.getpid()}"
    tmp.mkdir(parents=True, exist_ok=True)

    files = iter_jsonl_files(str(folder))
    encoder = StateEncoder()
    total_stats: Counter = Counter()
    meta_chunks: List[dict] = []
    n = 0
    t0 = time.time()
    n_chunks = (len(files) + chunk_files - 1) // chunk_files
    with open(tmp / "X.bin", "wb") as xf:
        for ci in range(n_chunks):
            chunk = files[ci * chunk_files:(ci + 1) * chunk_files]
            examples, stats = build_examples(chunk, encoder=encoder, with_opp=True)
            stats.pop("replays", None)     # per-chunk count is wrong; recomputed below
            total_stats.update(stats)
            nc = len(examples)
            Xc = np.empty((nc, dim), dtype=np.float32)
            m = _new_meta_arrays(nc)
            for j, ex in enumerate(examples):
                Xc[j] = ex["x"]
                _fill_meta_row(m, j, ex)
            Xc.tofile(xf)
            meta_chunks.append(m)
            n += nc
            del examples, Xc
            if n_chunks > 1:
                print(f"[encoded-cache] build {folder}: chunk {ci + 1}/{n_chunks} — "
                      f"{n} examples, X {n * dim * 4 / 2**30:.1f} GB on disk, "
                      f"{time.time() - t0:.0f}s", flush=True)

    meta = {k: (np.concatenate([mc[k] for mc in meta_chunks])
                if meta_chunks else _new_meta_arrays(0)[k])
            for k in _new_meta_arrays(0)}
    del meta_chunks
    total_stats["replays"] = int(np.unique(meta["rid"]).size)

    # Finalize X.npy: standard npy header + the raw rows (format identical to np.save).
    hdr = {"descr": np.lib.format.dtype_to_descr(np.dtype(np.float32)),
           "fortran_order": False, "shape": (n, dim)}
    with open(tmp / "X.npy", "wb") as out:
        np.lib.format.write_array_header_1_0(out, hdr)
        with open(tmp / "X.bin", "rb") as src:
            shutil.copyfileobj(src, out, length=64 * 2**20)
    (tmp / "X.bin").unlink()
    np.savez(tmp / "meta.npz", **meta)
    (tmp / "stats.json").write_text(json.dumps(dict(total_stats)), encoding="utf-8")
    (tmp / "DONE").write_text("ok", encoding="utf-8")

    # Atomic-ish publish + sweep of stale fingerprints and dead tmp dirs.
    if final.exists():
        shutil.rmtree(final, ignore_errors=True)
    os.rename(tmp, final)
    for d in _cache_dir(folder).iterdir():
        if d.is_dir() and d.name != fp:
            shutil.rmtree(d, ignore_errors=True)
    return final, total_stats


def load_cache(folder: str, with_opp: bool,
               mmap: bool = False) -> Optional[Tuple[List[dict], Counter]]:
    """Reconstruct the examples list from a fingerprint-matching cache, else None.

    ``mmap=True`` pages X from disk (``np.load(..., mmap_mode='r')``): each
    example's ``x`` is a READ-ONLY row view of the on-disk array — copy before
    writing.  The small meta arrays are always loaded into RAM.
    """
    fp = folder_fingerprint(folder)
    d = _cache_dir(folder) / fp
    if not (d / "DONE").exists():
        return None
    try:
        if mmap:
            try:
                X = np.load(d / "X.npy", mmap_mode="r")
            except (OSError, ValueError):   # e.g. a zero-row X cannot be mmapped
                X = np.load(d / "X.npy")
        else:
            X = np.load(d / "X.npy")
        m = np.load(d / "meta.npz")
        stats = Counter(json.loads((d / "stats.json").read_text(encoding="utf-8")))
    except (OSError, ValueError, json.JSONDecodeError):
        return None

    n = X.shape[0]
    target, mask = m["target"], m["mask"]
    g_target, g_mask = m["g_target"], m["g_mask"]
    o_target, o_mask = m["o_target"], m["o_mask"]
    rid, persp, dtype_ = m["rid"], m["persp"], m["dtype"]
    rating, rating_delta = m["rating"], m["rating_delta"]
    won, turn = m["won"], m["turn"]

    examples: List[dict] = []
    for i in range(n):
        targets = {}
        masks = {}
        g_targets = {}
        g_masks = {}
        for h_idx, head in enumerate(HEADS):
            if target[i, h_idx] >= 0:
                targets[head] = int(target[i, h_idx])
                masks[head] = mask[i, h_idx]
            if g_target[i, h_idx] >= 0:
                g_targets[head] = int(g_target[i, h_idx])
                g_masks[head] = g_mask[i, h_idx]
        ex = {
            "x": X[i],
            "targets": targets, "masks": masks,
            "gimmick_targets": g_targets, "gimmick_masks": g_masks,
            "replay_id": str(rid[i]),
            "perspective": (str(persp[i]) or None),
            "rating": (None if np.isnan(rating[i]) else float(rating[i])),
            "rating_delta": float(rating_delta[i]),
            "won": (None if won[i] < 0 else bool(won[i])),
            "turn": int(turn[i]),
            "decision_type": (str(dtype_[i]) or None),
        }
        if with_opp:
            opp_targets = {}
            opp_masks = {}
            for o_idx, ohead in enumerate(OPP_HEADS):
                if o_target[i, o_idx] >= 0:
                    opp_targets[ohead] = int(o_target[i, o_idx])
                    opp_masks[ohead] = o_mask[i, o_idx]
            ex["opp_targets"] = opp_targets
            ex["opp_masks"] = opp_masks
        examples.append(ex)
    return examples, stats


def cached_examples_from_folders(
    folders: Sequence[str],
    with_opp: bool = False,
    use_cache: bool = True,
    limit_transitions: Optional[int] = None,
    limit_files: Optional[int] = None,
    mmap: bool = False,
) -> Tuple[List[dict], Counter]:
    """Drop-in for ``examples_from_folders`` with a per-folder encoded cache.

    Smoke limits bypass the cache (a partial build must never be cached, and a
    cached full build must never masquerade as a limited one).  ``mmap=True``
    keeps every folder's X on disk (read-only row views) — the Step-B low-RAM
    path; pair with ``BCDataset(lazy_x=True)``.
    """
    if not use_cache or limit_transitions is not None or limit_files is not None:
        return examples_from_folders(
            folders, with_opp=with_opp,
            limit_transitions=limit_transitions, limit_files=limit_files)

    all_examples: List[dict] = []
    total_stats: Counter = Counter()
    seen_folders: set = set()
    for folder in folders:
        # Mirror examples_from_folders' de-dup (it de-dups by FILE abspath; the
        # realistic overlap is the same folder listed twice — drop repeats).
        key = os.path.abspath(str(folder))
        if key in seen_folders:
            continue
        seen_folders.add(key)
        if not Path(folder).is_dir() or not iter_jsonl_files(str(folder)):
            continue                       # matches examples_from_folders' skip
        t0 = time.time()
        hit = load_cache(folder, with_opp=with_opp, mmap=mmap)
        if hit is not None:
            examples, stats = hit
            print(f"[encoded-cache] HIT  {folder} — {len(examples)} examples "
                  f"in {time.time()-t0:.1f}s")
        else:
            # Streaming chunked build (WITH opp fields so one cache serves both
            # with_opp modes), then reconstruct from the fresh cache — the
            # round-trip is proven byte-identical to a direct in-RAM build.
            rebuilt = None
            try:
                build_cache_streaming(folder)
                rebuilt = load_cache(folder, with_opp=with_opp, mmap=mmap)
            except OSError as exc:      # cache is an optimisation, never a blocker
                print(f"[encoded-cache] WARNING: could not write cache for "
                      f"{folder}: {exc} — falling back to the in-RAM build",
                      file=sys.stderr)
            if rebuilt is not None:
                examples, stats = rebuilt
                print(f"[encoded-cache] MISS {folder} — built + cached "
                      f"{len(examples)} examples in {time.time()-t0:.1f}s")
            else:
                # Fallback (cache unwritable / corpus changed mid-build):
                # plain in-RAM build, nothing cached.
                examples, stats = examples_from_folders([folder], with_opp=with_opp)
                print(f"[encoded-cache] MISS {folder} — built {len(examples)} "
                      f"examples UNCACHED in {time.time()-t0:.1f}s")
        all_examples.extend(examples)
        total_stats.update(stats)
    total_stats["replays"] = len({e["replay_id"] for e in all_examples})
    return all_examples, total_stats
