"""
Behaviour-Cloning v0 trainer for Victory-Dance (VGC Reg M-A, doubles).

Trains the two-head MLP policy (bc_model.BCPolicy) to imitate human actions
from parsed replays:

  * input  : 938-dim encoded state (state_before_actions)
  * targets: per-slot action_index (0-15), masked cross-entropy over the
             decision-time legal actions (illegal logits -> -inf before softmax)
  * metrics: per-epoch val top-1 / top-3 action accuracy (pooled + per head)
  * output : checkpoints the best model (by val top-1) to --out

Baseline = Type B only.  Run under the GPU venv:

    .venv\\Scripts\\python.exe ai_train_scripts\\train_bc.py \\
        --data data\\vods\\Prepared_training_data\\Regulation_MA\\Jsonl_TypeB \\
        --epochs 30

Smoke test (few hundred transitions, few epochs, CPU ok):

    .venv\\Scripts\\python.exe ai_train_scripts\\train_bc.py \\
        --data <folder> --limit-transitions 400 --epochs 3 --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ── Local imports (same package) ──────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from bc_dataset import (  # noqa: E402
    ACTIONS_PER_SLOT,
    BCDataset,
    HEADS,
    build_examples,
    examples_from_folders,
    print_stats,
    split_by_replay,
)
from bc_model import build_model  # noqa: E402

# Large negative used to mask illegal actions before softmax.  Finite (not
# -inf) so an all-illegal row can't poison the backward pass; the dataset
# guarantees every target is legal, so the chosen logit is always finite.
_NEG = -1e9


# ══════════════════════════════════════════════════════════════════════════════
def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Set illegal (mask==0) entries to a large negative before softmax."""
    return logits.masked_fill(mask == 0, _NEG)


def head_loss_and_acc(
    logits: torch.Tensor,   # (B, A) raw
    mask: torch.Tensor,     # (B, A) 1=legal
    target: torch.Tensor,   # (B,)  action_index, -1 where invalid
    valid: torch.Tensor,    # (B,)  1.0 where this head has a target
    class_weight: Optional[torch.Tensor] = None,  # (A,) per-action loss weight
) -> Tuple[torch.Tensor, int, int, int]:
    """
    Returns (summed_ce_loss, n_valid, n_correct_top1, n_correct_top3) for one
    head over a batch.  Loss/accuracy are computed only over valid rows.

    ``class_weight`` (optional, length A) scales each target class's loss to
    counter majority-action bias; accuracy is unaffected by it.
    """
    valid_b = valid > 0.5
    n_valid = int(valid_b.sum().item())
    if n_valid == 0:
        return logits.sum() * 0.0, 0, 0, 0

    ml = masked_logits(logits, mask)[valid_b]
    tgt = target[valid_b]

    # Summed cross-entropy (reduction='sum' so multiple heads average per
    # decision, not per head).
    ce = F.cross_entropy(ml, tgt, weight=class_weight, reduction="sum")

    with torch.no_grad():
        top1 = ml.argmax(dim=1)
        n_top1 = int((top1 == tgt).sum().item())
        k = min(3, ml.shape[1])
        top3 = ml.topk(k, dim=1).indices
        n_top3 = int((top3 == tgt.unsqueeze(1)).any(dim=1).sum().item())

    return ce, n_valid, n_top1, n_top3


def compute_class_weights(examples, action_dim: int, cap: float = 10.0):
    """Balanced (sklearn-style) action weights over the train targets (both
    heads): ``w_c = total / (n_present_classes * count_c)``.  This keeps the
    frequency-weighted average weight ≈ 1 (so the overall loss scale is sane —
    common actions land near ~0.25, not ~0), while rare actions are up-weighted
    and clamped to ``cap`` so a handful of samples can't dominate the gradient.
    Absent classes get a neutral weight of 1.

    Returns (weights np.float32 [action_dim], counts np.float64 [action_dim])."""
    counts = np.zeros(action_dim, dtype=np.float64)
    for ex in examples:
        for ai in ex["targets"].values():
            counts[ai] += 1
    w = np.ones(action_dim, dtype=np.float32)
    present = counts > 0
    if present.any():
        total = counts[present].sum()
        n_present = int(present.sum())
        bal = total / (n_present * counts[present])
        w[present] = np.clip(bal, 0.0, cap).astype(np.float32)
    return w, counts


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    class_weight: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """One pass over ``loader``.  Train if ``optimizer`` given, else eval."""
    train = optimizer is not None
    model.train(train)

    totals = {
        "loss": 0.0,
        "n": 0,  # total valid decisions (for loss mean)
    }
    per_head = {h: {"n": 0, "top1": 0, "top3": 0} for h in HEADS}

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            x = batch["x"].to(device)
            target = batch["target"].to(device)   # (B, 2)
            mask = batch["mask"].to(device)        # (B, 2, A)
            valid = batch["valid"].to(device)      # (B, 2)

            out = model(x)  # {head: (B, A)}

            batch_loss = x.new_zeros(())
            batch_valid = 0
            for h_idx, head in enumerate(HEADS):
                ce, n_valid, n1, n3 = head_loss_and_acc(
                    out[head], mask[:, h_idx], target[:, h_idx], valid[:, h_idx],
                    class_weight=class_weight,
                )
                batch_loss = batch_loss + ce
                batch_valid += n_valid
                per_head[head]["n"] += n_valid
                per_head[head]["top1"] += n1
                per_head[head]["top3"] += n3

            if batch_valid == 0:
                continue
            mean_loss = batch_loss / batch_valid

            if train:
                optimizer.zero_grad()
                mean_loss.backward()
                optimizer.step()

            totals["loss"] += float(batch_loss.item())
            totals["n"] += batch_valid

    n = max(totals["n"], 1)
    pooled_top1 = sum(per_head[h]["top1"] for h in HEADS)
    pooled_top3 = sum(per_head[h]["top3"] for h in HEADS)
    pooled_n = sum(per_head[h]["n"] for h in HEADS) or 1

    metrics = {
        "loss": totals["loss"] / n,
        "top1": pooled_top1 / pooled_n,
        "top3": pooled_top3 / pooled_n,
        "n": totals["n"],
    }
    for head in HEADS:
        hn = max(per_head[head]["n"], 1)
        metrics[f"{head}_top1"] = per_head[head]["top1"] / hn
        metrics[f"{head}_top3"] = per_head[head]["top3"] / hn
        metrics[f"{head}_n"] = per_head[head]["n"]
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
def train(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[train_bc] WARNING: cuda requested but unavailable -> cpu")
        device = "cpu"

    # ── Load examples ─────────────────────────────────────────────────────────
    folders: List[str] = list(args.data)
    if args.type_a:
        folders.extend(args.type_a)
    print(f"[train_bc] loading examples from: {folders}")
    t0 = time.time()
    examples, stats = examples_from_folders(
        folders,
        limit_transitions=args.limit_transitions,
        limit_files=args.limit_files,
    )
    print_stats(stats)
    print(f"[train_bc] loaded {len(examples)} examples in {time.time()-t0:.1f}s")
    if not examples:
        raise SystemExit("[train_bc] no usable examples found")

    train_ex, val_ex = split_by_replay(examples, val_frac=args.val_frac, seed=args.seed)
    print(
        f"[train_bc] split -> {len(train_ex)} train / {len(val_ex)} val examples "
        f"({len({e['replay_id'] for e in train_ex})} / "
        f"{len({e['replay_id'] for e in val_ex})} replays)"
    )

    train_ds = BCDataset(train_ex)
    val_ds = BCDataset(val_ex)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # ── Model / optimizer ─────────────────────────────────────────────────────
    model = build_model(
        hidden_dims=tuple(args.hidden),
        dropout=args.dropout,
        heads=HEADS,
        device=device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ── Class-imbalance weighting (optional) ──────────────────────────────────
    class_weight = None
    if args.class_weight == "balanced":
        w_np, counts = compute_class_weights(train_ex, ACTIONS_PER_SLOT,
                                             cap=args.class_weight_cap)
        class_weight = torch.tensor(w_np, device=device)
        print(f"[train_bc] class weights (balanced, cap {args.class_weight_cap}): "
              f"min {w_np.min():.2f} max {w_np.max():.2f} "
              f"(majority action #{int(counts.argmax())}={int(counts.max())} decisions)")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "bc_best.pt"

    config = {
        "state_dim": model.state_dim,
        "action_dim": model.action_dim,
        "hidden_dims": list(args.hidden),
        "dropout": args.dropout,
        "heads": list(HEADS),
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "val_frac": args.val_frac,
        "seed": args.seed,
        "data": folders,
        "class_weight": args.class_weight,
        "patience": args.patience,
    }

    # ── Train loop (val-weighting OFF so val loss/acc stay true) ───────────────
    best_top1 = -1.0
    epochs_no_improve = 0
    history: List[dict] = []
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, device, optimizer, class_weight=class_weight)
        va = run_epoch(model, val_loader, device, optimizer=None)
        history.append({"epoch": epoch, "train": tr, "val": va})
        print(
            f"epoch {epoch:3d} | "
            f"train loss {tr['loss']:.4f} top1 {tr['top1']:.3f} | "
            f"val loss {va['loss']:.4f} top1 {va['top1']:.3f} top3 {va['top3']:.3f} "
            f"(a {va['our_a_top1']:.3f} / b {va['our_b_top1']:.3f})"
        )

        if va["top1"] > best_top1:
            best_top1 = va["top1"]
            epochs_no_improve = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_metrics": va,
                },
                ckpt_path,
            )
            print(f"           ↑ new best (val top1 {best_top1:.3f}) -> {ckpt_path}")
        else:
            epochs_no_improve += 1
            if args.patience and epochs_no_improve >= args.patience:
                print(f"[train_bc] early stop at epoch {epoch} "
                      f"(no val top1 improvement for {args.patience} epochs)")
                break

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"[train_bc] done. best val top1 = {best_top1:.3f}. checkpoint: {ckpt_path}")
    return {"best_top1": best_top1, "history": history, "checkpoint": str(ckpt_path)}


# ══════════════════════════════════════════════════════════════════════════════
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train BC v0 policy")
    _prep = Path("data") / "vods" / "Prepared_training_data" / "Regulation_MA"
    default_data = [str(_prep / f"Jsonl_Type{t}") for t in ("A", "B", "C", "D")]
    ap.add_argument("--data", nargs="+", default=default_data,
                    help="folder(s) of per-replay .jsonl (default: all VOD types "
                         "A-D; missing/empty folders are skipped, files deduped)")
    ap.add_argument("--type-a", nargs="+", default=None,
                    help="extra folder(s) to append (deduped). Type A is already in "
                         "the default --data; use only for ad-hoc extra data.")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, nargs="+", default=[512, 256])
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--class-weight", choices=["none", "balanced"], default="none",
                    help="weight the action loss by inverse class frequency to "
                         "counter majority-action bias (default: none)")
    ap.add_argument("--class-weight-cap", type=float, default=10.0,
                    help="cap on any single class weight (default: 10)")
    ap.add_argument("--patience", type=int, default=0,
                    help="early-stop after N epochs with no val top-1 gain (0=off)")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit-transitions", type=int, default=None,
                    help="cap transitions read (smoke runs)")
    ap.add_argument("--limit-files", type=int, default=None,
                    help="cap JSONL files read (smoke runs)")
    ap.add_argument("--out", default=str(_HERE / "checkpoints"),
                    help="checkpoint output directory")
    return ap.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
