"""
bc_val_report.py — fine-grained BC checkpoint comparison on the PINNED val set
==============================================================================
The A/B ruler for data-expansion and memory-model runs.  Pooled val top1 alone
is the wrong instrument: most VGC decisions need no history and no extra data,
so real improvements (adaptation, late-game reads) hide inside the pooled mean.
This report scores one or more checkpoints on the SAME fixed val split
(replays chosen exactly like ``train_bc --val-data``: ``split_by_replay`` over
the reference corpus with the given seed/val-frac) and breaks accuracy down:

  * pooled top1 / top3 (masked argmax — mirrors train_bc's metric),
  * per head (our_a / our_b),
  * per TURN bucket (1, 2-3, 4-6, 7-10, 11+) — late buckets are where
    accumulated information (and therefore memory) should show up,
  * per decision type (turn vs post-faint replacement),
  * value head: Brier score + outcome accuracy (where the outcome is known).

Multiple --ckpt args print side by side with deltas vs the FIRST checkpoint.
Memory checkpoints (config memory_dim>0) are automatically evaluated through
``forward_with_memory`` with ``--sequence-len`` frames (default 16); stateless
checkpoints use the plain single-turn forward.

Usage (defaults target the 0.585-anchor val corpus):
    python -m v_dance.eval.bc_val_report \
        --ckpt ai_train_scripts/BC_model/checkpoints_attn_pre_gen141/battle_base.pt \
        --ckpt ai_train_scripts/BC_model/checkpoints_attn_hfdata/battle_base.pt \
        [--val-data <folders>] [--val-frac 0.1] [--seed 0] [--device cpu]

CPU-safe by design (default --device cpu): the report must be runnable while a
training run owns the GPU.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from v_dance.training.bc_dataset import (
    BCDataset,
    HEADS,
    examples_from_folders,
    split_by_replay,
)
from v_dance.training.train_bc import masked_logits

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PREP = _REPO_ROOT / "data" / "vods" / "Prepared_training_data"
# The 0.585 anchor's training folders — the canonical pinned-val reference.
_DEFAULT_VAL_DATA = [
    str(_PREP / "Regulation_MA" / "Jsonl_TypeA"),
    str(_PREP / "Regulation_MA" / "Jsonl_TypeB"),
    str(_PREP / "Regulation_MB" / "Jsonl_TypeB"),
    str(_PREP / "Regulation_MB_Bo3" / "Jsonl_TypeB"),
]

# Turn buckets: label -> inclusive (lo, hi); None = open-ended.
_TURN_BUCKETS = [
    ("t1", 1, 1),
    ("t2-3", 2, 3),
    ("t4-6", 4, 6),
    ("t7-10", 7, 10),
    ("t11+", 11, None),
]


def _bucket_of(turn: Optional[int]) -> str:
    t = int(turn or 0)
    for label, lo, hi in _TURN_BUCKETS:
        if t >= lo and (hi is None or t <= hi):
            return label
    return _TURN_BUCKETS[0][0]


class _Acc:
    """top1/top3 accumulator."""

    __slots__ = ("n", "top1", "top3")

    def __init__(self):
        self.n = self.top1 = self.top3 = 0

    def add(self, ok1: int, ok3: int, n: int) -> None:
        self.n += n
        self.top1 += ok1
        self.top3 += ok3

    def rate(self) -> Optional[float]:
        return (self.top1 / self.n) if self.n else None

    def rate3(self) -> Optional[float]:
        return (self.top3 / self.n) if self.n else None


def evaluate_checkpoint(
    ckpt_path: str,
    val_ex: List[dict],
    device: str = "cpu",
    batch_size: int = 256,
    sequence_len: int = 16,
    archetype_ids: Optional[List[int]] = None,
    slice_labels: Optional[List[str]] = None,
) -> Dict:
    """Score one checkpoint on the fixed val examples.

    Returns a dict of accumulators: pooled/head/turn-bucket/decision-type
    (+ per-ARCHETYPE when ids are supplied — the Phase-2 z success metric)
    action accuracy + value Brier/accuracy + metadata.
    """
    from v_dance.play.model_io import load_bc_policy

    model, _head_names = load_bc_policy(ckpt_path, device=device)
    model.eval()
    memory_dim = int(getattr(model, "memory_dim", 0) or 0)
    if int(getattr(model, "n_archetypes", 0) or 0) and archetype_ids is None:
        print(f"[val-report] WARNING: {ckpt_path} is a z-conditioned checkpoint "
              f"but no --z-archetypes was given — scoring with the UNKNOWN "
              f"embedding (unconditioned; understates the conditioned model)")
    # Eval at the checkpoint's TRAINED window (config sequence_len stamp) — a
    # mismatched T would shift every window off the trained positional regime
    # and depress the memory model's numbers for reasons unrelated to quality
    # (audit 2026-07-02). The CLI --sequence-len is only the fallback for a
    # memory checkpoint that predates the stamp.
    ckpt_T = int(getattr(model, "_sequence_len", 1) or 1)
    seq_len = (ckpt_T if ckpt_T > 1 else sequence_len) if memory_dim > 0 else 1
    if memory_dim > 0 and ckpt_T > 1 and ckpt_T != sequence_len:
        print(f"[val-report] using the checkpoint's trained sequence_len={ckpt_T} "
              f"(CLI --sequence-len {sequence_len} ignored)")

    ds = BCDataset(val_ex, sequence_len=seq_len, archetype_ids=archetype_ids)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    # Per-example metadata arrays, index-aligned with the dataset (shuffle=False).
    turns = [e.get("turn") for e in val_ex]
    dtypes = [e.get("decision_type") or "turn" for e in val_ex]

    pooled = _Acc()
    by_head = {h: _Acc() for h in HEADS}
    by_bucket = {label: _Acc() for label, _, _ in _TURN_BUCKETS}
    by_dtype: Dict[str, _Acc] = {}
    by_arch: Dict[str, _Acc] = {}
    by_slice: Dict[str, _Acc] = {}
    v_n = 0
    v_brier = 0.0
    v_correct = 0

    idx0 = 0
    with torch.no_grad():
        for batch in loader:
            bsz = batch["target"].shape[0]
            aid_b = batch.get("archetype")
            if aid_b is not None:
                aid_b = aid_b.to(device)
            if "x_seq" in batch:
                actions, _gimmicks, value = model.forward_with_memory(
                    batch["x_seq"].to(device),
                    frame_padding_mask=batch["frame_padding_mask"].to(device),
                    archetype_id=aid_b,
                )
            else:
                actions, _gimmicks, value = model(batch["x"].to(device),
                                                  archetype_id=aid_b)
            target = batch["target"].to(device)          # (B, 2)
            mask = batch["mask"].to(device)              # (B, 2, A)
            valid = batch["valid"].to(device)            # (B, 2)

            for h_idx, head in enumerate(HEADS):
                ml = masked_logits(actions[head], mask[:, h_idx])
                vb = valid[:, h_idx] > 0.5
                if not bool(vb.any()):
                    continue
                tgt = target[vb, h_idx]
                top1_pred = ml[vb].argmax(dim=-1)
                ok1_mask = top1_pred == tgt
                k = min(3, ml.shape[-1])
                top3_pred = ml[vb].topk(k, dim=-1).indices
                ok3_mask = (top3_pred == tgt.unsqueeze(-1)).any(dim=-1)

                rows = torch.nonzero(vb, as_tuple=False).squeeze(-1).tolist()
                ok1_l = ok1_mask.tolist()
                ok3_l = ok3_mask.tolist()
                for row, o1, o3 in zip(rows, ok1_l, ok3_l):
                    gi = idx0 + row                      # global example index
                    o1i, o3i = int(o1), int(o3)
                    pooled.add(o1i, o3i, 1)
                    by_head[head].add(o1i, o3i, 1)
                    by_bucket[_bucket_of(turns[gi])].add(o1i, o3i, 1)
                    by_dtype.setdefault(dtypes[gi], _Acc()).add(o1i, o3i, 1)
                    if archetype_ids is not None:
                        by_arch.setdefault(f"z={archetype_ids[gi]}", _Acc()) \
                            .add(o1i, o3i, 1)
                    if slice_labels is not None:
                        by_slice.setdefault(slice_labels[gi], _Acc()) \
                            .add(o1i, o3i, 1)

            # Value head: Brier + accuracy where the outcome label exists.
            v_valid = batch["value_valid"].to(device) > 0.5
            if bool(v_valid.any()) and bool(getattr(model, "_value_trained", True)):
                p = torch.sigmoid(value[v_valid])
                y = batch["value_target"].to(device)[v_valid]
                v_brier += float(((p - y) ** 2).sum())
                v_correct += int(((p > 0.5).float() == y).sum())
                v_n += int(v_valid.sum())

            idx0 += bsz

    return {
        "ckpt": str(ckpt_path),
        "memory_dim": memory_dim,
        "sequence_len": seq_len,
        "pooled": pooled,
        "by_head": by_head,
        "by_bucket": by_bucket,
        "by_dtype": by_dtype,
        "by_arch": by_arch,
        "by_slice": by_slice,
        "value": {"n": v_n, "brier": (v_brier / v_n) if v_n else None,
                  "acc": (v_correct / v_n) if v_n else None},
    }


def _fmt(rate: Optional[float], base: Optional[float] = None) -> str:
    if rate is None:
        return "    —   "
    s = f"{rate:7.4f}"
    if base is not None and base is not rate:
        s += f" ({rate - base:+.4f})"
    return s


def print_report(results: List[Dict]) -> None:
    base = results[0]
    print("\n" + "═" * 72)
    for i, r in enumerate(results):
        tag = "BASE" if i == 0 else f"CAND{i}"
        mem = f" memory_dim={r['memory_dim']} T={r['sequence_len']}" if r["memory_dim"] else ""
        print(f"  [{tag}] {r['ckpt']}{mem}")
    print("═" * 72)

    def _row(label: str, get) -> None:
        cells = []
        base_acc = get(base)
        base_rate = base_acc.rate() if base_acc else None
        for i, r in enumerate(results):
            acc = get(r)
            rate = acc.rate() if acc else None
            cells.append(_fmt(rate, None if i == 0 else base_rate))
        n = base_acc.n if base_acc else 0
        print(f"  {label:<14}" + "  ".join(cells) + f"   (n={n})")

    print(f"  {'':14}" + "  ".join(
        f"{('BASE' if i == 0 else f'CAND{i} (delta)'):>{8 if i == 0 else 18}}"
        for i in range(len(results))))
    _row("top1", lambda r: r["pooled"])
    print(f"  {'top3':<14}" + "  ".join(
        _fmt(r["pooled"].rate3(), None if i == 0 else base["pooled"].rate3())
        for i, r in enumerate(results)) + f"   (n={base['pooled'].n})")
    print("  ── per head ──")
    for h in HEADS:
        _row(h, lambda r, h=h: r["by_head"][h])
    print("  ── per turn bucket ──")
    for label, _, _ in _TURN_BUCKETS:
        _row(label, lambda r, label=label: r["by_bucket"][label])
    print("  ── per decision type ──")
    for dt in sorted(base["by_dtype"]):
        _row(dt, lambda r, dt=dt: r["by_dtype"].get(dt) or _Acc())
    if any(r.get("by_arch") for r in results):
        # Phase-2 z metric: the conditioning must lift MINORITY archetypes
        # (the mush-average harm concentrates there), not just the pooled mean.
        print("  ── per archetype ──")
        arch_keys = sorted({k for r in results for k in (r.get("by_arch") or {})},
                           key=lambda s: (len(s), s))
        for ak in arch_keys:
            _row(ak, lambda r, ak=ak: (r.get("by_arch") or {}).get(ak) or _Acc())
    if any(r.get("by_slice") for r in results):
        # G2 metric: accuracy on val teams the training corpus NEVER saw — "plays a
        # team it never saw" quantified. A big seen-vs-HELD-OUT gap = the model leans
        # on team familiarity, not transferable mechanics.
        print("  ── held-out-team slice (G2) ──")
        for sk in sorted({k for r in results for k in (r.get("by_slice") or {})}):
            _row(sk, lambda r, sk=sk: (r.get("by_slice") or {}).get(sk) or _Acc())
    print("  ── value head ──")
    vals = []
    for i, r in enumerate(results):
        v = r["value"]
        if v["brier"] is None:
            vals.append("    —   ")
        else:
            d = ("" if i == 0 or base["value"]["brier"] is None
                 else f" ({v['brier'] - base['value']['brier']:+.4f})")
            vals.append(f"{v['brier']:7.4f}{d}")
    print(f"  {'brier (↓)':<14}" + "  ".join(vals) + f"   (n={base['value']['n']})")
    print("═" * 72)


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

    ap = argparse.ArgumentParser(
        description="Fine-grained BC checkpoint comparison on the pinned val set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--ckpt", action="append", required=True,
                    help="checkpoint path (repeat to compare; first = baseline)")
    ap.add_argument("--val-data", nargs="+", default=_DEFAULT_VAL_DATA,
                    help="reference corpus folders that define the val split")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sequence-len", type=int, default=16,
                    help="frames for memory checkpoints (ignored for stateless)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cpu",
                    help="cpu by default — must be runnable while training owns the GPU")
    ap.add_argument("--limit-files", type=int, default=None,
                    help="cap JSONL files read (smoke runs)")
    ap.add_argument("--z-archetypes", default=None,
                    help="team-archetype artifact: joins archetype ids onto the val "
                         "examples (required to fairly score a z-conditioned ckpt; "
                         "adds the per-archetype breakdown)")
    ap.add_argument("--held-out-slice", nargs="*", default=None, metavar="TRAIN_FOLDER",
                    help="add the G2 seen-vs-HELD-OUT own-team slice: val examples whose "
                         "roster never occurs in the TRAIN folders (no folders given = the "
                         "4 HF_OTS train folders). First run walks the train corpus once "
                         "(~10-30 min at 80k files; JSON-cached in artifacts/ after).")
    args = ap.parse_args(argv)

    print(f"[val-report] loading + encoding the reference corpus: {args.val_data}")
    t0 = time.time()
    from v_dance.training.encoded_cache import cached_examples_from_folders
    examples, _stats = cached_examples_from_folders(
        args.val_data, limit_files=args.limit_files)
    if not examples:
        raise SystemExit("[val-report] no usable examples in --val-data")
    _train, val_ex = split_by_replay(examples, val_frac=args.val_frac, seed=args.seed)
    del examples, _train
    print(f"[val-report] val set: {len(val_ex)} examples / "
          f"{len({e['replay_id'] for e in val_ex})} replays "
          f"({time.time()-t0:.1f}s)")

    # Phase-2 z: join archetype ids so a conditioned checkpoint is scored with
    # its real z (and the per-archetype breakdown gets printed).
    arch_ids = None
    if args.z_archetypes:
        from v_dance.datatools.team_archetypes import (
            load_archetype_assignments, load_artifact)
        k_unknown = int(load_artifact(args.z_archetypes)["k"])
        asg = load_archetype_assignments(args.z_archetypes)
        arch_ids = [asg.get((e["replay_id"], e.get("perspective")), k_unknown)
                    for e in val_ex]
        n_unk = sum(1 for a in arch_ids if a == k_unknown)
        print(f"[val-report] archetype ids joined: {len(arch_ids) - n_unk}/"
              f"{len(arch_ids)} labelled ({n_unk} → UNKNOWN)")

    # G2 held-out-team slice: label each val example by whether its OWN roster ever
    # occurs in the train folders (team_keys_for_folders is JSON-cached per folder set).
    slice_labels = None
    if args.held_out_slice is not None:
        from v_dance.datatools.team_archetypes import team_keys_for_folders
        _prep = _PREP
        train_folders = args.held_out_slice or [
            str(_prep / "Regulation_MA" / "Jsonl_HF_OTS"),
            str(_prep / "Regulation_MB" / "Jsonl_HF_OTS"),
            str(_prep / "Regulation_MB_Bo3" / "Jsonl_HF_OTS"),
            str(_prep / "Regulation_MA_Bo3" / "Jsonl_HF_OTS"),
        ]
        print(f"[val-report] held-out-team slice — train keys from {len(train_folders)} "
              f"folder(s) (first run walks them once; cached after)")
        train_keys, _ = team_keys_for_folders(train_folders)
        _, val_sight = team_keys_for_folders(args.val_data)
        slice_labels, n_held, n_unk = [], 0, 0
        for e in val_ex:
            k = val_sight.get((e["replay_id"], e.get("perspective")))
            if k is None:
                slice_labels.append("team=?"); n_unk += 1
            elif k in train_keys:
                slice_labels.append("team=seen")
            else:
                slice_labels.append("team=HELD-OUT"); n_held += 1
        print(f"[val-report] held-out-team examples: {n_held}/{len(val_ex)} "
              f"({n_unk} unmatchable)")

    results = []
    for c in args.ckpt:
        t1 = time.time()
        r = evaluate_checkpoint(c, val_ex, device=args.device,
                                batch_size=args.batch_size,
                                sequence_len=args.sequence_len,
                                archetype_ids=arch_ids,
                                slice_labels=slice_labels)
        print(f"[val-report] scored {c} in {time.time()-t1:.1f}s")
        results.append(r)

    print_report(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
