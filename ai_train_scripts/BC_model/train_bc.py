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
    compute_sample_weights,
    examples_from_folders,
    filter_by_rating,
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
    sample_weight: Optional[torch.Tensor] = None,  # (B,) per-example loss weight
) -> Tuple[torch.Tensor, int, int, int]:
    """
    Returns (summed_ce_loss, n_valid, n_correct_top1, n_correct_top3) for one
    head over a batch.  Loss/accuracy are computed only over valid rows.

    ``class_weight`` (optional, length A) scales each target class's loss to
    counter majority-action bias; accuracy is unaffected by it.

    ``sample_weight`` (optional, length B) scales each EXAMPLE's loss — used to
    imitate stronger / winning demonstrators (TIER-1 #1).  When None the path is
    byte-identical to the original ``reduction='sum'`` cross-entropy; the trainer
    only passes it when a weighting flag is actually enabled.
    """
    valid_b = valid > 0.5
    n_valid = int(valid_b.sum().item())
    if n_valid == 0:
        return logits.sum() * 0.0, 0, 0, 0

    ml = masked_logits(logits, mask)[valid_b]
    tgt = target[valid_b]

    # Summed cross-entropy (reduction='sum' so multiple heads average per
    # decision, not per head).  With a per-example weight, sum the per-row losses
    # scaled by that weight (class_weight still composes via the 'none' path).
    if sample_weight is None:
        ce = F.cross_entropy(ml, tgt, weight=class_weight, reduction="sum")
    else:
        per = F.cross_entropy(ml, tgt, weight=class_weight, reduction="none")
        ce = (per * sample_weight[valid_b]).sum()

    with torch.no_grad():
        top1 = ml.argmax(dim=1)
        n_top1 = int((top1 == tgt).sum().item())
        k = min(3, ml.shape[1])
        top3 = ml.topk(k, dim=1).indices
        n_top3 = int((top3 == tgt.unsqueeze(1)).any(dim=1).sum().item())

    return ce, n_valid, n_top1, n_top3


def gimmick_loss_and_recall(
    logits: torch.Tensor,   # (B, G) raw gimmick logits
    mask: torch.Tensor,     # (B, G) 1=legal
    target: torch.Tensor,   # (B,)  gimmick index, -1 where invalid
    valid: torch.Tensor,    # (B,)  1.0 where this head has a gimmick target
    class_weight: Optional[torch.Tensor] = None,  # (G,) per-class loss weight
) -> Tuple[torch.Tensor, int, int, int]:
    """Returns (summed_ce_loss, n_valid, n_true_pos, n_false_neg) for the gimmick
    head over a batch.  Positives are the RARE mega class (index 1); we track
    recall = TP/(TP+FN) because plain accuracy is dominated by the 'none' class
    and would hide a head that never megas.  ``class_weight`` up-weights the rare
    positive so the gradient does not collapse to always-predict-none."""
    valid_b = valid > 0.5
    n_valid = int(valid_b.sum().item())
    if n_valid == 0:
        return logits.sum() * 0.0, 0, 0, 0

    ml = masked_logits(logits, mask)[valid_b]
    tgt = target[valid_b]
    ce = F.cross_entropy(ml, tgt, weight=class_weight, reduction="sum")

    with torch.no_grad():
        pred = ml.argmax(dim=1)
        pos = tgt == 1
        n_tp = int(((pred == 1) & pos).sum().item())
        n_fn = int(((pred != 1) & pos).sum().item())

    return ce, n_valid, n_tp, n_fn


def compute_gimmick_class_weights(examples, gimmick_dim: int, cap: float = 10.0):
    """Balanced gimmick-class weights over the train gimmick targets (both heads).
    The mega positive is a small minority (≤1 mega per team per game), so without
    this the head learns to always predict 'none' (perfect accuracy, zero recall).
    Same balanced formula as compute_class_weights.

    Returns (weights np.float32 [gimmick_dim], counts np.float64 [gimmick_dim])."""
    counts = np.zeros(gimmick_dim, dtype=np.float64)
    for ex in examples:
        for gi in (ex.get("gimmick_targets") or {}).values():
            counts[gi] += 1
    w = np.ones(gimmick_dim, dtype=np.float32)
    present = counts > 0
    if present.any():
        total = counts[present].sum()
        n_present = int(present.sum())
        bal = total / (n_present * counts[present])
        w[present] = np.clip(bal, 0.0, cap).astype(np.float32)
    return w, counts


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
    gimmick_class_weight: Optional[torch.Tensor] = None,
    gimmick_loss_weight: float = 1.0,
    opp_loss_weight: float = 0.0,
    opp_class_weight: Optional[torch.Tensor] = None,
    sample_weighted: bool = False,
) -> Dict[str, float]:
    """One pass over ``loader``.  Train if ``optimizer`` given, else eval.

    Trains the action heads + the parallel gimmick (mega) heads, and OPTIONALLY
    the auxiliary opponent action heads (task #9) when the model carries opp_*
    heads and the batch carries opp labels.  The backward total is
    ``action_mean + gimmick_loss_weight*gimmick_mean + opp_loss_weight*opp_mean``;
    each term is meaned over its own valid-decision count.  The reported ``loss``/
    ``top1`` stay OUR-action-only so baseline-vs-aux comparison is apples-to-apples;
    the opp head's accuracy is reported separately as ``opp_top1``."""
    train = optimizer is not None
    model.train(train)

    # Opponent aux heads = the model's action heads that are not our own slots.
    opp_heads = [h for h in getattr(model, "head_names", HEADS) if h not in HEADS]

    totals = {"loss": 0.0, "n": 0, "g_loss": 0.0, "g_n": 0, "opp_loss": 0.0, "opp_n": 0}
    per_head = {h: {"n": 0, "top1": 0, "top3": 0} for h in HEADS}
    opp_acc = {h: {"n": 0, "top1": 0} for h in opp_heads}
    gim = {"tp": 0, "fn": 0}

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            x = batch["x"].to(device)
            target = batch["target"].to(device)   # (B, 2)
            mask = batch["mask"].to(device)        # (B, 2, A)
            valid = batch["valid"].to(device)      # (B, 2)
            g_target = batch["gimmick_target"].to(device)  # (B, 2)
            g_mask = batch["gimmick_mask"].to(device)      # (B, 2, G)
            g_valid = batch["gimmick_valid"].to(device)    # (B, 2)
            # Per-example action-loss weight (TIER-1 #1).  Only applied when the
            # trainer enabled a weighting flag; otherwise None → original loss.
            sample_weight = (batch["weight"].to(device)
                             if sample_weighted and "weight" in batch else None)

            actions, gimmicks = model(x)

            batch_loss = x.new_zeros(())
            batch_valid = 0
            g_batch_loss = x.new_zeros(())
            g_batch_valid = 0
            for h_idx, head in enumerate(HEADS):
                ce, n_valid, n1, n3 = head_loss_and_acc(
                    actions[head], mask[:, h_idx], target[:, h_idx], valid[:, h_idx],
                    class_weight=class_weight, sample_weight=sample_weight,
                )
                batch_loss = batch_loss + ce
                batch_valid += n_valid
                per_head[head]["n"] += n_valid
                per_head[head]["top1"] += n1
                per_head[head]["top3"] += n3

                g_ce, g_nv, g_tp, g_fn = gimmick_loss_and_recall(
                    gimmicks[head], g_mask[:, h_idx], g_target[:, h_idx],
                    g_valid[:, h_idx], class_weight=gimmick_class_weight,
                )
                g_batch_loss = g_batch_loss + g_ce
                g_batch_valid += g_nv
                gim["tp"] += g_tp
                gim["fn"] += g_fn

            # ── Auxiliary opponent-action heads ──────────────────────────────
            o_batch_loss = x.new_zeros(())
            o_batch_valid = 0
            if opp_heads and "opp_target" in batch:
                o_target = batch["opp_target"].to(device)   # (B, n_opp)
                o_mask = batch["opp_mask"].to(device)        # (B, n_opp, A)
                o_valid = batch["opp_valid"].to(device)      # (B, n_opp)
                for o_idx, ohead in enumerate(opp_heads):
                    ce, n_valid, n1, _ = head_loss_and_acc(
                        actions[ohead], o_mask[:, o_idx], o_target[:, o_idx],
                        o_valid[:, o_idx], class_weight=opp_class_weight,
                    )
                    o_batch_loss = o_batch_loss + ce
                    o_batch_valid += n_valid
                    opp_acc[ohead]["n"] += n_valid
                    opp_acc[ohead]["top1"] += n1

            if batch_valid == 0:
                continue
            mean_loss = batch_loss / batch_valid
            mean_gimmick = g_batch_loss / max(g_batch_valid, 1)
            mean_opp = o_batch_loss / max(o_batch_valid, 1)
            total = (mean_loss + gimmick_loss_weight * mean_gimmick
                     + opp_loss_weight * mean_opp)

            if train:
                optimizer.zero_grad()
                total.backward()
                optimizer.step()

            totals["loss"] += float(batch_loss.item())
            totals["n"] += batch_valid
            totals["g_loss"] += float(g_batch_loss.item())
            totals["g_n"] += g_batch_valid
            totals["opp_loss"] += float(o_batch_loss.item())
            totals["opp_n"] += o_batch_valid

    n = max(totals["n"], 1)
    pooled_top1 = sum(per_head[h]["top1"] for h in HEADS)
    pooled_top3 = sum(per_head[h]["top3"] for h in HEADS)
    pooled_n = sum(per_head[h]["n"] for h in HEADS) or 1
    g_pos = gim["tp"] + gim["fn"]
    opp_n = sum(opp_acc[h]["n"] for h in opp_heads)
    opp_top1 = sum(opp_acc[h]["top1"] for h in opp_heads)

    metrics = {
        "loss": totals["loss"] / n,
        "top1": pooled_top1 / pooled_n,
        "top3": pooled_top3 / pooled_n,
        "n": totals["n"],
        "gimmick_loss": totals["g_loss"] / max(totals["g_n"], 1),
        "gimmick_recall": gim["tp"] / g_pos if g_pos else 0.0,
        "gimmick_pos": g_pos,
        "gimmick_n": totals["g_n"],
        "opp_loss": totals["opp_loss"] / max(totals["opp_n"], 1),
        "opp_top1": opp_top1 / opp_n if opp_n else 0.0,
        "opp_n": totals["opp_n"],
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
        with_opp=args.aux_opp_head,
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

    # ── Demonstrator skill filtering / weighting (TIER-1 #1, TRAIN ONLY) ───────
    # Imitate STRONGER / winning players.  Val is never filtered or weighted so
    # its top1/top3 stay a true, comparable metric.  All knobs default OFF, in
    # which case the training path is byte-identical to the unweighted baseline.
    if args.rating_min is not None:
        before = len(train_ex)
        train_ex = filter_by_rating(train_ex, args.rating_min)
        print(f"[train_bc] rating filter >= {args.rating_min}: "
              f"{before} -> {len(train_ex)} train examples "
              f"({before - len(train_ex)} dropped; unknown-rating kept)")
        if not train_ex:
            raise SystemExit("[train_bc] rating filter removed all train examples")
    sample_weighted = bool(args.rating_weight or args.outcome_weight)
    train_weights = None
    if sample_weighted:
        train_weights = compute_sample_weights(
            train_ex,
            rating_weight=args.rating_weight,
            outcome_weight=args.outcome_weight,
            rating_weight_floor=args.rating_weight_floor,
            loss_weight=args.loss_weight,
        )
        n_won = sum(1 for e in train_ex if e.get("won") is True)
        n_rated = sum(1 for e in train_ex if e.get("rating") is not None)
        print(f"[train_bc] sample weighting: rating={args.rating_weight} "
              f"(floor {args.rating_weight_floor}, {n_rated}/{len(train_ex)} rated) "
              f"outcome={args.outcome_weight} (loss_weight {args.loss_weight}, "
              f"{n_won}/{len(train_ex)} won) -> weight min {train_weights.min():.3f} "
              f"max {train_weights.max():.3f} mean {train_weights.mean():.3f}")

    # Move-slot permutation augmentation is TRAIN-ONLY (val stays raw so its
    # metrics are true).  It makes the policy order-invariant — see task #22.
    train_ds = BCDataset(train_ex, augment_move_order=args.augment_move_order,
                         aug_seed=args.seed, with_opp=args.aux_opp_head,
                         weights=train_weights)
    val_ds = BCDataset(val_ex, with_opp=args.aux_opp_head)
    print(f"[train_bc] move-slot permutation augmentation: "
          f"{'ON' if args.augment_move_order else 'OFF'} (train only)")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # ── Auxiliary opponent head (task #9): adds opp_a/opp_b ACTION heads (no opp
    # gimmick head).  OUR action/gimmick heads + reported our top1 are unchanged,
    # so the A/B comparison vs the no-aux baseline is apples-to-apples. ──────────
    OPP = ["opp_a", "opp_b"]
    train_heads = list(HEADS) + (OPP if args.aux_opp_head else [])
    if args.aux_opp_head:
        print(f"[train_bc] auxiliary opponent head: ON (heads={tuple(train_heads)}, "
              f"weight {args.aux_opp_weight})")

    # ── Model / optimizer ─────────────────────────────────────────────────────
    model = build_model(
        heads=train_heads,
        gimmick_heads=list(HEADS),
        hidden_dims=tuple(args.hidden),
        dropout=args.dropout,
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

    # Gimmick head is ALWAYS class-balanced — the mega positive is intrinsically
    # rare (≤1 per team per game), so an unweighted head collapses to all-none.
    from bc_dataset import GIMMICK_DIM  # noqa: E402
    gw_np, gcounts = compute_gimmick_class_weights(train_ex, GIMMICK_DIM,
                                                   cap=args.class_weight_cap)
    gimmick_class_weight = torch.tensor(gw_np, device=device)
    print(f"[train_bc] gimmick class weights (balanced, cap {args.class_weight_cap}): "
          f"{[round(float(v), 2) for v in gw_np]} "
          f"(counts none={int(gcounts[0])} mega={int(gcounts[1])})")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "bc_best.pt"

    config = {
        "state_dim": model.state_dim,
        "action_dim": model.action_dim,
        "gimmick_dim": model.gimmick_dim,
        # True only when the train data actually carried gimmick labels — a
        # gimmick head trained on pre-gimmick JSONL is at init and must NOT drive
        # live mega decisions (the serve player honours this flag).
        "gimmick_trained": bool(gcounts.sum() > 0),
        "hidden_dims": list(args.hidden),
        "dropout": args.dropout,
        "heads": list(model.head_names),               # our (+ opp aux when on)
        "gimmick_heads": list(model.gimmick_head_names),
        "aux_opp_head": bool(args.aux_opp_head),
        "aux_opp_weight": args.aux_opp_weight,
        "augment_move_order": bool(args.augment_move_order),
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "val_frac": args.val_frac,
        "seed": args.seed,
        "data": folders,
        "class_weight": args.class_weight,
        "patience": args.patience,
        # Demonstrator skill filtering / weighting (TIER-1 #1).
        "rating_min": args.rating_min,
        "rating_weight": bool(args.rating_weight),
        "rating_weight_floor": args.rating_weight_floor,
        "outcome_weight": bool(args.outcome_weight),
        "loss_weight": args.loss_weight,
    }

    # ── Train loop (val-weighting OFF so val loss/acc stay true) ───────────────
    best_top1 = -1.0
    epochs_no_improve = 0
    history: List[dict] = []
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, device, optimizer, class_weight=class_weight,
                       gimmick_class_weight=gimmick_class_weight,
                       gimmick_loss_weight=args.gimmick_loss_weight,
                       opp_loss_weight=args.aux_opp_weight if args.aux_opp_head else 0.0,
                       sample_weighted=sample_weighted)
        va = run_epoch(model, val_loader, device, optimizer=None)
        history.append({"epoch": epoch, "train": tr, "val": va})
        opp_str = f" | opp top1 {va['opp_top1']:.3f}" if args.aux_opp_head else ""
        print(
            f"epoch {epoch:3d} | "
            f"train loss {tr['loss']:.4f} top1 {tr['top1']:.3f} | "
            f"val loss {va['loss']:.4f} top1 {va['top1']:.3f} top3 {va['top3']:.3f} "
            f"(a {va['our_a_top1']:.3f} / b {va['our_b_top1']:.3f}) "
            f"| gim recall {va['gimmick_recall']:.3f} pos {va['gimmick_pos']}{opp_str}"
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
    ap.add_argument("--gimmick-loss-weight", type=float, default=1.0,
                    help="weight on the gimmick (mega) CE term in the total loss "
                         "(default: 1.0). The gimmick head is always class-balanced.")
    ap.add_argument("--augment-move-order", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="train-only: randomly permute each mon's 4 move slots + "
                         "remap the action label so the policy is move-ORDER "
                         "invariant (task #22). Default ON; --no-augment-move-order "
                         "to disable.")
    ap.add_argument("--aux-opp-head", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="add auxiliary opponent action heads (opp_a/opp_b) that "
                         "predict the opponent's action — a representation-shaping "
                         "aux signal (task #9). Default OFF (the A/B baseline).")
    ap.add_argument("--aux-opp-weight", type=float, default=0.3,
                    help="weight on the auxiliary opponent CE term (default 0.3).")
    # ── Demonstrator skill filtering / weighting (TIER-1 #1) ──────────────────
    ap.add_argument("--rating-min", type=float, default=None,
                    help="drop TRAIN examples whose our-side ladder rating_before "
                         "is below this (val unfiltered; unknown-rating kept). "
                         "Default off — imitate stronger demonstrators.")
    ap.add_argument("--rating-weight", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="weight the action loss by the demonstrator's rating "
                         "PERCENTILE (mapped to [floor,1]); default off.")
    ap.add_argument("--rating-weight-floor", type=float, default=0.25,
                    help="lowest multiplier for --rating-weight (bottom-rated "
                         "demonstrator); top-rated maps to 1.0 (default 0.25).")
    ap.add_argument("--outcome-weight", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="weight the action loss by game outcome: won=1.0, "
                         "lost=--loss-weight, unknown=1.0; default off.")
    ap.add_argument("--loss-weight", type=float, default=0.5,
                    help="weight on decisions from games our side LOST when "
                         "--outcome-weight is on (default 0.5).")
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
