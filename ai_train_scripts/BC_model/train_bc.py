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
) -> Tuple[torch.Tensor, int, int, int]:
    """
    Returns (summed_ce_loss, n_valid, n_correct_top1, n_correct_top3) for one
    head over a batch.  Loss/accuracy are computed only over valid rows.
    """
    valid_b = valid > 0.5
    n_valid = int(valid_b.sum().item())
    if n_valid == 0:
        return logits.sum() * 0.0, 0, 0, 0

    ml = masked_logits(logits, mask)[valid_b]
    tgt = target[valid_b]

    # Summed cross-entropy (reduction='sum' so multiple heads average per
    # decision, not per head).
    ce = F.cross_entropy(ml, tgt, reduction="sum")

    with torch.no_grad():
        top1 = ml.argmax(dim=1)
        n_top1 = int((top1 == tgt).sum().item())
        k = min(3, ml.shape[1])
        top3 = ml.topk(k, dim=1).indices
        n_top3 = int((top3 == tgt.unsqueeze(1)).any(dim=1).sum().item())

    return ce, n_valid, n_top1, n_top3


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
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
                    out[head], mask[:, h_idx], target[:, h_idx], valid[:, h_idx]
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
    }

    # ── Train loop ────────────────────────────────────────────────────────────
    best_top1 = -1.0
    history: List[dict] = []
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, device, optimizer)
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

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"[train_bc] done. best val top1 = {best_top1:.3f}. checkpoint: {ckpt_path}")
    return {"best_top1": best_top1, "history": history, "checkpoint": str(ckpt_path)}


# ══════════════════════════════════════════════════════════════════════════════
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train BC v0 policy")
    default_b = (
        Path("data") / "vods" / "Prepared_training_data" / "Regulation_MA" / "Jsonl_TypeB"
    )
    ap.add_argument("--data", nargs="+", default=[str(default_b)],
                    help="folder(s) of per-replay .jsonl (default: Type B)")
    ap.add_argument("--type-a", nargs="+", default=None,
                    help="optional extra Type A folder(s) to mix in")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, nargs="+", default=[512, 256])
    ap.add_argument("--dropout", type=float, default=0.1)
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
