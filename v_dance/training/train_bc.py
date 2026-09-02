"""
Behaviour-Cloning v0 trainer for Victory-Dance (VGC Reg M-A, doubles).

Trains the two-head per-mon set-attention policy (bc_model_attn.AttnBCPolicy) to imitate human actions
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
from v_dance.training.bc_dataset import (  # noqa: E402
    ACTIONS_PER_SLOT,
    BCDataset,
    HEADS,
    bc_worker_init,
    build_examples,
    canonical_rid,
    compute_advantage_weights,
    compute_closed_copy_weights,
    compute_own_team_weights,
    compute_sample_weights,
    own_slice_indices,
    own_team_overlap_map,
    parse_team_species,
    examples_from_folders,
    filter_by_rating,
    print_stats,
    split_by_replay,
    split_with_reference,
)
from v_dance.models.bc_model_attn import build_attn_model  # noqa: E402

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


def compute_class_weights(examples, action_dim: int, cap: float = 10.0,
                          augment_move_order: bool = False):
    """Balanced (sklearn-style) action weights over the train targets (both
    heads): ``w_c = total / (n_present_classes * count_c)``.  This keeps the
    frequency-weighted average weight ≈ 1 (so the overall loss scale is sane —
    common actions land near ~0.25, not ~0), while rare actions are up-weighted
    and clamped to ``cap`` so a handful of samples can't dominate the gradient.
    Absent classes get a neutral weight of 1.

    ``augment_move_order`` (audit fix): when the train loader permutes move-ORDER
    each fetch (``BCDataset(augment_move_order=True)``, the default), it remaps the
    4 MOVE SLOTS to a random permutation while the per-slot BUCKET (index % 3) is
    invariant. So within each bucket the 4 move classes {b, 3+b, 6+b, 9+b} attain
    EQUAL frequency in the data the model actually sees. Counting on the RAW
    (un-permuted) targets would then MIS-calibrate the inverse-frequency weights for
    all 12 move classes. So when augmentation is on we POOL each bucket's 4 move-slot
    counts to their mean before balancing (switch classes ≥12 are permutation-invariant
    → left untouched). The RETURNED counts are the raw counts (for the majority-action
    readout).

    Returns (weights np.float32 [action_dim], counts np.float64 [action_dim])."""
    counts = np.zeros(action_dim, dtype=np.float64)
    for ex in examples:
        for ai in ex["targets"].values():
            counts[ai] += 1
    counts_for_weight = counts
    if augment_move_order:                      # pool the 4 move slots within each of the 3 buckets
        counts_for_weight = counts.copy()
        for b in range(3):
            idx = [b, 3 + b, 6 + b, 9 + b]
            if max(idx) < action_dim:
                counts_for_weight[idx] = counts_for_weight[idx].mean()
    w = np.ones(action_dim, dtype=np.float32)
    present = counts_for_weight > 0
    if present.any():
        total = counts_for_weight[present].sum()
        n_present = int(present.sum())
        bal = total / (n_present * counts_for_weight[present])
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
    value_loss_weight: float = 1.0,
    progress_every: int = 0,
    progress_label: str = "",
    progress_secs: float = 600.0,
    pair_dropout: float = 0.25,
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

    totals = {"loss": 0.0, "n": 0, "g_loss": 0.0, "g_n": 0, "opp_loss": 0.0, "opp_n": 0,
              "v_loss": 0.0, "v_n": 0, "v_correct": 0, "v_brier": 0.0}
    per_head = {h: {"n": 0, "top1": 0, "top3": 0} for h in HEADS}
    opp_acc = {h: {"n": 0, "top1": 0} for h in opp_heads}
    gim = {"tp": 0, "fn": 0}

    ctx = torch.enable_grad() if train else torch.no_grad()
    import time as _time
    _n_batches = len(loader)
    _t0 = _time.time()
    _last_beat = _t0
    with ctx:
        for _bi, batch in enumerate(loader, 1):
            # Intra-epoch heartbeat (2026-07-19, USER): TIME-based and ON BY DEFAULT
            # (progress_secs, default 600s) so a long epoch is never a silent multi-hour
            # wait — plus the original batch-count trigger (--progress-every). Print-only;
            # training math is untouched.
            _now = _time.time()
            _count_due = progress_every and (_bi % progress_every == 0 or _bi == _n_batches)
            _time_due = progress_secs and (_now - _last_beat) >= progress_secs
            if train and (_count_due or _time_due):
                _last_beat = _now
                _dt = max(_now - _t0, 1e-6)
                _eta = (_n_batches - _bi) / max(_bi / _dt, 1e-9) / 60.0
                print(f"    {_time.strftime('%H:%M:%S')} [{progress_label} "
                      f"batch {_bi}/{_n_batches} ({100.0 * _bi / _n_batches:.1f}%)] "
                      f"{_bi/_dt:.1f} batch/s, epoch ETA {_eta:.1f} min",
                      flush=True)
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

            # Phase-2 z: archetype ids ride the batch when the trainer joined
            # them (--z-archetypes); absent → None (models without z ignore it).
            aid = batch.get("archetype")
            if aid is not None:
                aid = aid.to(device)

            # Era-4 2b: teacher-forced partner conditioning (TRAIN only; eval stays
            # zero-cond so the printed val metric remains armB-comparable — the
            # teacher-forced val gap is bc_val_report's go/no-go, not this one).
            # Per sample: a random decode order; the FIRST head in order trains
            # unconditioned, the SECOND sees the partner's HUMAN action one-hot,
            # dropped to zeros with prob ``pair_dropout`` so zero-cond serving
            # stays in-distribution (the kill-switch path keeps training too).
            partner_actions = None
            if train and getattr(model, "pair_cond", False):
                bsz = x.shape[0]
                a_dim = int(getattr(model, "action_dim"))
                b_first = torch.rand(bsz, device=device) < 0.5   # True -> our_b decodes first
                keep = torch.rand(bsz, device=device) >= pair_dropout
                oh = F.one_hot(target.clamp(min=0).long(), a_dim).to(x.dtype)  # (B, 2, A)
                gate_a = (b_first & keep & (valid[:, 1] > 0.5)).to(x.dtype).unsqueeze(1)
                gate_b = (~b_first & keep & (valid[:, 0] > 0.5)).to(x.dtype).unsqueeze(1)
                partner_actions = {"our_a": oh[:, 1] * gate_a,   # our_a conditions on our_b's action
                                   "our_b": oh[:, 0] * gate_b}   # our_b conditions on our_a's action

            # Phase 1b: sequence batches carry x_seq (B, T, D) — feed the memory
            # path, supervising the LAST frame (targets are the item's own).
            if "x_seq" in batch:
                actions, gimmicks, value = model.forward_with_memory(
                    batch["x_seq"].to(device),
                    frame_padding_mask=batch["frame_padding_mask"].to(device),
                    archetype_id=aid,
                    partner_actions=partner_actions,
                )
            else:
                actions, gimmicks, value = model(x, archetype_id=aid,
                                                 partner_actions=partner_actions)
            v_target = batch["value_target"].to(device)    # (B,) win=1/loss=0
            v_valid = batch["value_valid"].to(device)      # (B,) 1 where known

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

            # ── Scalar VALUE head (win probability, BCE over the game outcome) ──
            v_mask = v_valid > 0.5
            v_nv = int(v_mask.sum().item())
            if v_nv > 0:
                v_loss = F.binary_cross_entropy_with_logits(
                    value[v_mask], v_target[v_mask], reduction="sum")
                with torch.no_grad():
                    p = torch.sigmoid(value[v_mask])
                    v_correct = int(((p > 0.5).float() == v_target[v_mask]).sum().item())
                    v_brier = float(((p - v_target[v_mask]) ** 2).sum().item())
            else:
                v_loss = value.sum() * 0.0
                v_correct, v_brier = 0, 0.0

            if batch_valid == 0:
                continue
            mean_loss = batch_loss / batch_valid
            mean_gimmick = g_batch_loss / max(g_batch_valid, 1)
            mean_opp = o_batch_loss / max(o_batch_valid, 1)
            mean_value = v_loss / max(v_nv, 1)
            total = (mean_loss + gimmick_loss_weight * mean_gimmick
                     + opp_loss_weight * mean_opp + value_loss_weight * mean_value)

            if train:
                optimizer.zero_grad()
                total.backward()
                optimizer.step()

            totals["v_loss"] += float(v_loss.item())
            totals["v_n"] += v_nv
            totals["v_correct"] += v_correct
            totals["v_brier"] += v_brier
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
        "value_loss": totals["v_loss"] / max(totals["v_n"], 1),
        "value_acc": totals["v_correct"] / max(totals["v_n"], 1),
        "value_brier": totals["v_brier"] / max(totals["v_n"], 1),
        "value_n": totals["v_n"],
    }
    for head in HEADS:
        hn = max(per_head[head]["n"], 1)
        metrics[f"{head}_top1"] = per_head[head]["top1"] / hn
        metrics[f"{head}_top3"] = per_head[head]["top3"] / hn
        metrics[f"{head}_n"] = per_head[head]["n"]
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# Resumable training (2026-07-19, USER request — long era retrains must pause at the
# USER's convenience). Shared machinery: v_dance/training/resumable.py (also used by
# train_teampreview; the exploiter has its own iteration-style resume).
from v_dance.training.resumable import (  # noqa: E402
    RESUME_NAME,
    config_fingerprint,
    load_resume_state,
    restore_rng,
    save_resume_state,
)


def train(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[train_bc] WARNING: cuda requested but unavailable -> cpu")
        device = "cpu"

    # ── Load examples ─────────────────────────────────────────────────────────
    # Step B low-RAM mode (--mmap-cache): X stays on disk (mmap'd caches +
    # lazy dataset rows) — the only way the full 77.8k MA-Bo3 corpus (~26 GB
    # of X) trains on 32 GB RAM.
    if args.mmap_cache and not args.encoded_cache:
        sys.exit("[train_bc] --mmap-cache requires the encoded cache "
                 "(drop --no-encoded-cache): without a cache there is no "
                 "on-disk X to page from")
    if args.mmap_cache and args.sequence_len > 1:
        sys.exit("[train_bc] --mmap-cache does not support --sequence-len > 1 "
                 "(the sequence path gathers history frames from the in-RAM X "
                 "matrix)")
    if args.mmap_cache:
        print("[train_bc] mmap cache: ON — X pages from the on-disk encoded "
              "caches; dataset rows fetch lazily (one-row copy per item)")
    folders: List[str] = list(args.data)
    if args.type_a:
        folders.extend(args.type_a)
    print(f"[train_bc] loading examples from: {folders}")
    t0 = time.time()
    from v_dance.training.encoded_cache import cached_examples_from_folders
    examples, stats = cached_examples_from_folders(
        folders,
        limit_transitions=args.limit_transitions,
        limit_files=args.limit_files,
        with_opp=args.aux_opp_head,
        use_cache=args.encoded_cache,
        mmap=args.mmap_cache,
    )
    print_stats(stats)
    print(f"[train_bc] loaded {len(examples)} examples in {time.time()-t0:.1f}s")
    if not examples:
        raise SystemExit("[train_bc] no usable examples found")

    if args.val_data:
        # Data-expansion A/B mode: the val set is pinned to a fixed REFERENCE
        # corpus (same seed + val_frac reproduce the anchor run's exact val
        # replays), --data folders only add TRAIN examples, and any overlap
        # with the reference corpus is dropped (no leakage/double-weighting).
        print(f"[train_bc] val reference corpus: {args.val_data}")
        ref_ex, ref_stats = cached_examples_from_folders(
            args.val_data,
            with_opp=args.aux_opp_head,
            use_cache=args.encoded_cache,
            mmap=args.mmap_cache,
        )
        print_stats(ref_stats)
        if not ref_ex:
            raise SystemExit("[train_bc] no usable examples in --val-data")
        train_ex, val_ex = split_with_reference(
            examples, ref_ex, val_frac=args.val_frac, seed=args.seed)
    else:
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

    # ── Era-4 arm 2a: closed-copy regime titration (TRAIN ONLY, default OFF) ───
    # Splits each HF open/__closed twin pair to total ONE decision (closed gets
    # lam, open 1−lam) — removes arm B's verified 2× tournament upweight while
    # keeping the closed VIEW. Composes multiplicatively; val stays unweighted.
    if args.closed_copy_weight is not None:
        cc_w = compute_closed_copy_weights(train_ex, args.closed_copy_weight)
        rids = [str(e["replay_id"]) for e in train_ex]
        _open = {canonical_rid(r) for r in rids if not r.endswith("__closed")}
        _closed = {canonical_rid(r) for r in rids if r.endswith("__closed")}
        _paired = _open & _closed
        n_open = sum(1 for r in rids
                     if not r.endswith("__closed") and canonical_rid(r) in _paired)
        n_closed = sum(1 for r in rids
                       if r.endswith("__closed") and canonical_rid(r) in _paired)
        n_paired = n_open + n_closed
        if n_paired == 0:
            sys.exit("[train_bc] --closed-copy-weight found ZERO open/__closed pairs "
                     "in the train set — the HF_CLOSED folders are missing from "
                     "--data (the flag would silently no-op; refusing to run)")
        train_weights = cc_w if train_weights is None else train_weights * cc_w
        m = float(train_weights.mean())
        if m > 0:
            train_weights = (train_weights / m).astype(np.float32)
        print(f"[train_bc] closed-copy weighting: lam_closed={args.closed_copy_weight} "
              f"({n_paired}/{len(train_ex)} paired examples: {n_closed} closed, "
              f"{n_open} open) -> combined weight min {train_weights.min():.3f} "
              f"max {train_weights.max():.3f} mean {train_weights.mean():.3f}")

    # ── Era-5 W1: own-team SPECIALIST weighting (TRAIN ONLY, default OFF) ─────────
    # docs/era5_specialist_design.md §3.1 — specialize the OWN side without dropping a single
    # game (the opponent side stays fully general): weight = 1 + λ·(fraction of our six in the
    # demonstrator's roster), read from each replay's first line, composed multiplicatively.
    # The val slice with overlap ≥ min/6 is the offline read this arm is FOR (reported per epoch;
    # --select-own-slice makes it the checkpoint-selection metric).
    own_map, own_val_idx, own_species = None, None, None
    if args.own_team:
        own_species = parse_team_species(args.own_team)
        _all_folders = list(dict.fromkeys([*folders, *(args.val_data or []), *(args.type_a or [])]))
        own_map = own_team_overlap_map(_all_folders, own_species, limit_files=args.limit_files)
        ot_w, hist = compute_own_team_weights(train_ex, own_map, args.own_team_weight)
        if sum(hist.get(k, 0) for k in range(1, 7)) == 0:
            sys.exit(f"[train_bc] --own-team {args.own_team!r} ({', '.join(own_species)}) overlaps "
                     f"NO train example — wrong team / paste? Refusing to run a no-op.")
        train_weights = ot_w if train_weights is None else train_weights * ot_w
        m = float(train_weights.mean())
        if m > 0:
            train_weights = (train_weights / m).astype(np.float32)
        print(f"[train_bc] own-team weighting: {', '.join(own_species)} λ={args.own_team_weight:g} "
              f"— train overlap histogram (mons of 6): "
              + ", ".join(f"{k}:{hist.get(k, 0)}" for k in range(7))
              + f", unmapped {hist.get('unmapped', 0)} -> combined weight min {train_weights.min():.3f} "
              f"max {train_weights.max():.3f} mean {train_weights.mean():.3f}")
        own_val_idx = own_slice_indices(val_ex, own_map, args.own_team_min_overlap / 6.0)
        print(f"[train_bc] own-team val slice: {len(own_val_idx)}/{len(val_ex)} val examples with "
              f"overlap ≥ {args.own_team_min_overlap}/6")
        if args.select_own_slice and len(own_val_idx) < 2000:
            print(f"[train_bc] WARNING: --select-own-slice with only {len(own_val_idx)} slice examples "
                  f"— selection falls back to pooled val top1 (slice too small to trust)")

    # ── Phase-2 offline advantage weighting (TRAIN ONLY, default OFF) ──────────
    # w = f(G − V(s)) from a trained value head: imitate what beat expectation.
    # Composes multiplicatively with the rating/outcome weights above; val stays
    # unweighted so its metrics remain comparable across runs.
    if args.adv_weight != "off":
        adv_ckpt = args.adv_value_ckpt or args.warm_start
        if not adv_ckpt:
            sys.exit("[train_bc] --adv-weight needs --adv-value-ckpt (or --warm-start, "
                     "whose value head is borrowed when --adv-value-ckpt is omitted)")
        adv_w = compute_advantage_weights(
            train_ex, adv_ckpt, mode=args.adv_weight, beta=args.adv_beta,
            w_min=args.adv_clip_min, w_max=args.adv_clip_max, device=args.device)
        n_lab = sum(1 for e in train_ex if e.get("won") is not None)
        if n_lab == 0:
            print("[train_bc] WARNING: --adv-weight is ON but no train example has an "
                  "outcome label — advantage weights are all-ones (no effect)")
        train_weights = adv_w if train_weights is None else train_weights * adv_w
        m = float(train_weights.mean())
        if m > 0:
            train_weights = (train_weights / m).astype(np.float32)
        print(f"[train_bc] advantage weighting: mode={args.adv_weight} "
              f"beta={args.adv_beta} clip=[{args.adv_clip_min},{args.adv_clip_max}] "
              f"value_ckpt={adv_ckpt} ({n_lab}/{len(train_ex)} outcome-labelled) -> "
              f"combined weight min {train_weights.min():.3f} "
              f"max {train_weights.max():.3f} mean {train_weights.mean():.3f}")

    # ── Phase-2 z: archetype-id join (TRAIN AND VAL — the conditioned model is
    # evaluated with its real ids; examples missing from the artifact get the
    # UNKNOWN id k, which trains the "no plan info" embedding) ─────────────────
    z_art = None
    train_z = val_z = None
    if args.z_archetypes:
        from v_dance.datatools.team_archetypes import (
            load_archetype_assignments, load_artifact)
        z_art = load_artifact(args.z_archetypes)
        asg = load_archetype_assignments(args.z_archetypes)
        k_unknown = int(z_art["k"])
        train_z = [asg.get((e["replay_id"], e.get("perspective")), k_unknown)
                   for e in train_ex]
        val_z = [asg.get((e["replay_id"], e.get("perspective")), k_unknown)
                 for e in val_ex]
        n_unk = sum(1 for a in train_z if a == k_unknown)
        print(f"[train_bc] archetype-z: k={z_art['k']} z_dim={args.z_dim} "
              f"({len(train_z) - n_unk}/{len(train_z)} train examples labelled, "
              f"{n_unk} → UNKNOWN)")
        if n_unk == len(train_z):
            print("[train_bc] WARNING: NO train example matched the artifact's "
                  "assignments — z will only ever see the UNKNOWN embedding "
                  "(wrong artifact for this corpus?)")

    # Move-slot permutation augmentation is TRAIN-ONLY (val stays raw so its
    # metrics are true).  It makes the policy order-invariant — see task #22.
    train_ds = BCDataset(train_ex, augment_move_order=args.augment_move_order,
                         aug_seed=args.seed, with_opp=args.aux_opp_head,
                         weights=train_weights, sequence_len=args.sequence_len,
                         archetype_ids=train_z, lazy_x=args.mmap_cache)
    val_ds = BCDataset(val_ex, with_opp=args.aux_opp_head,
                       sequence_len=args.sequence_len,
                       archetype_ids=val_z, lazy_x=args.mmap_cache)
    if args.sequence_len > 1:
        print(f"[train_bc] sequence BC: T={args.sequence_len} frames per item "
              f"(memory_dim={args.memory_dim})")
    print(f"[train_bc] move-slot permutation augmentation: "
          f"{'ON' if args.augment_move_order else 'OFF'} (train only)")
    # M6 loader-workers (2026-07-11): the feed is Python-bound (GPU ~20% on the
    # era-2 run) — worker processes parallelize __getitem__ + collate. 0 (default)
    # = the in-process legacy path, byte-identical. Workers receive the tensors
    # via torch shared memory and re-open mmap caches by x_src path (BCDataset
    # worker-view pickling); a lazy dataset without cache stamps fails loud here
    # rather than deep inside a worker.
    loader_kw = {}
    if args.loader_workers > 0:
        for split_name, ds_ in (("train", train_ds), ("val", val_ds)):
            if not ds_.workers_safe:
                sys.exit(f"[train_bc] --loader-workers: the {split_name} dataset has no "
                         f"x_src cache stamps (smoke --limit-*/--no-cache runs bypass the "
                         f"encoded cache) — drop --loader-workers for this run.")
        loader_kw = dict(num_workers=args.loader_workers, persistent_workers=True,
                         worker_init_fn=bc_worker_init, prefetch_factor=4,
                         pin_memory=(args.device == "cuda"))
        print(f"[train_bc] DataLoader workers: {args.loader_workers} (persistent, "
              f"prefetch 4, pin_memory={loader_kw['pin_memory']})")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kw)
    # Era-5 W1: the own-team val slice (same val examples, same stamps → same loader settings).
    own_val_loader = None
    if own_val_idx:
        if args.sequence_len > 1:
            print("[train_bc] own-team val slice skipped: sequence BC needs whole trajectories")
        else:
            own_val_ds = BCDataset([val_ex[i] for i in own_val_idx], with_opp=args.aux_opp_head,
                                   sequence_len=args.sequence_len,
                                   archetype_ids=([val_z[i] for i in own_val_idx] if val_z else None),
                                   lazy_x=args.mmap_cache)
            own_val_loader = DataLoader(own_val_ds, batch_size=args.batch_size, shuffle=False,
                                        **loader_kw)

    # ── Auxiliary opponent head (task #9): adds opp_a/opp_b ACTION heads (no opp
    # gimmick head).  OUR action/gimmick heads + reported our top1 are unchanged,
    # so the A/B comparison vs the no-aux baseline is apples-to-apples. ──────────
    OPP = ["opp_a", "opp_b"]
    train_heads = list(HEADS) + (OPP if args.aux_opp_head else [])
    if args.aux_opp_head:
        print(f"[train_bc] auxiliary opponent head: ON (heads={tuple(train_heads)}, "
              f"weight {args.aux_opp_weight})")

    # ── Model / optimizer (#27 attn-only: the per-mon set-attention AttnBCPolicy) ──
    if args.opp_cond and not args.aux_opp_head:
        sys.exit("[train_bc] --opp-cond (Level B) requires --aux-opp-head (the opp_a/opp_b heads it "
                 "conditions on). Re-run with both, or drop --opp-cond.")
    if args.opp_cond:
        print("[train_bc] Level-B opp-conditioning: ON (the OUR heads read the detached predicted opp action)")
    if args.sequence_len > 1 and args.memory_dim <= 0:
        sys.exit("[train_bc] --sequence-len > 1 requires --memory-dim > 0 (the "
                 "match-memory core that consumes the frame stack).")
    if args.sequence_len > args.max_mem_len:
        sys.exit(f"[train_bc] --sequence-len {args.sequence_len} exceeds --max-mem-len "
                 f"{args.max_mem_len}: forward_with_memory would SILENTLY truncate every "
                 f"window to {args.max_mem_len} frames while the logs and the checkpoint "
                 f"stamp claim {args.sequence_len}. Raise --max-mem-len or shrink the window.")
    if args.memory_dim > 0 and args.sequence_len <= 1:
        print("[train_bc] WARNING: --memory-dim set but --sequence-len is 1 — the "
              "memory core will train on zero history (memory columns stay unused).")
    model = build_attn_model(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ff_mult=args.ff_mult,
        dropout=args.dropout,
        heads=train_heads,
        gimmick_heads=list(HEADS),
        value_readout=args.value_readout,
        opp_cond=args.opp_cond,
        memory_dim=args.memory_dim,
        mem_layers=args.mem_layers,
        mem_heads=args.mem_heads,
        max_mem_len=args.max_mem_len,
        n_archetypes=(int(z_art["k"]) if z_art else 0),
        z_dim=(args.z_dim if z_art else 0),
        pair_cond=args.pair_cond,
        device=device,
    )
    # Warm start (Phase 1b memory / Phase 2 z): initialise from a NARROWER
    # checkpoint with zero-padded trailing head columns for whatever widening
    # features (memory and/or z) this model adds over the checkpoint —
    # forward(x) reproduces the checkpoint's logits exactly at epoch 0, so an
    # A/B isolates the new feature's value.  Same-width → plain strict load.
    if args.warm_start:
        from v_dance.models.bc_model_attn import (
            init_extended_model_from_ckpt, init_pair_model_from_ckpt)
        ck = torch.load(args.warm_start, map_location=device, weights_only=False)
        state = ck.get("model_state", ck)
        ck_cfg = (ck.get("config") or {}) if isinstance(ck, dict) else {}
        extra = ((model.memory_dim + model.z_dim)
                 - (int(ck_cfg.get("memory_dim", 0) or 0)
                    + int(ck_cfg.get("z_dim", 0) or 0)))
        pair_extra = (model.pair_cond and not bool(ck_cfg.get("pair_cond", False)))
        if pair_extra and extra > 0:
            sys.exit("[train_bc] warm-starting pair_cond AND memory/z growth in one step is "
                     "unsupported (the OUR heads would grow by a different width than the other "
                     "heads — the uniform zero-pad surgery can't express that). Stage the "
                     "widenings across two warm-starts.")
        if pair_extra:
            init_pair_model_from_ckpt(model, state)      # 2b: our heads +action_dim, zeroed
        elif extra > 0:
            init_extended_model_from_ckpt(model, state, extra_cols=extra)
        else:
            model.load_state_dict(state, strict=True)
        print(f"[train_bc] warm-started from {args.warm_start}"
              + (f" (zero-padded {extra} new trailing head columns)" if extra > 0 else "")
              + (" (pair_cond: our heads +action_dim zero-padded — zero-cond forwards "
                 "reproduce the donor bit-exactly)" if pair_extra else ""))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ── Class-imbalance weighting (optional) ──────────────────────────────────
    class_weight = None
    if args.class_weight == "balanced":
        w_np, counts = compute_class_weights(train_ex, ACTIONS_PER_SLOT,
                                             cap=args.class_weight_cap,
                                             augment_move_order=args.augment_move_order)
        class_weight = torch.tensor(w_np, device=device)
        print(f"[train_bc] class weights (balanced, cap {args.class_weight_cap}): "
              f"min {w_np.min():.2f} max {w_np.max():.2f} "
              f"(majority action #{int(counts.argmax())}={int(counts.max())} decisions)")

    # Gimmick head is ALWAYS class-balanced — the mega positive is intrinsically
    # rare (≤1 per team per game), so an unweighted head collapses to all-none.
    from v_dance.training.bc_dataset import GIMMICK_DIM  # noqa: E402
    gw_np, gcounts = compute_gimmick_class_weights(train_ex, GIMMICK_DIM,
                                                   cap=args.class_weight_cap)
    gimmick_class_weight = torch.tensor(gw_np, device=device)
    print(f"[train_bc] gimmick class weights (balanced, cap {args.class_weight_cap}): "
          f"{[round(float(v), 2) for v in gw_np]} "
          f"(counts none={int(gcounts[0])} mega={int(gcounts[1])} tera={int(gcounts[2])})")

    # Value head trains only when the data carries game-outcome labels (`won`);
    # the serve player must not trust an untrained value head (the flag gates it).
    n_value = sum(1 for e in train_ex if e.get("won") is not None)
    # audit: gate the stamped flag on the LOSS WEIGHT too — a 0 weight means the head got zero gradient
    # and stays at xavier init, yet the data may still carry labels. Marking it value_trained=True would
    # let ActorCritic.from_bc_checkpoint's require_value_trained guard warm-start the RL critic from noise.
    value_trained = (n_value > 0) and (args.value_loss_weight > 0)
    print(f"[train_bc] value head: {n_value}/{len(train_ex)} train examples have an "
          f"outcome label (value_trained={value_trained}, weight {args.value_loss_weight})")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "battle_base.pt"   # the BC imitation base battle net (<type>_<variant> scheme)

    from v_dance.encoders.state_encoder import get_state_layout_version
    config = {
        # #27 attn-only: model_type drives model_io.load_bc_policy (now attn-only);
        # the attn dims below let the loader rebuild the exact net.
        "model_type": "attn",
        "state_dim": model.state_dim,
        "state_layout_version": get_state_layout_version(),
        "action_dim": model.action_dim,
        "gimmick_dim": model.gimmick_dim,
        "value_loss_weight": args.value_loss_weight,
        # #28: stamp the remaining loss-shaping knobs for provenance/reproducibility (every other
        # weighting knob is recorded; these two were missing). No loader reads them back.
        "gimmick_loss_weight": args.gimmick_loss_weight,
        "class_weight_cap": args.class_weight_cap,
        "value_trained": bool(value_trained),
        # True only when the train data actually carried gimmick labels AND the head was actually
        # trained (weight > 0) — a gimmick head at init (no labels OR zero loss weight) must NOT drive
        # live mega decisions (the serve player honours this flag).
        "gimmick_trained": bool(gcounts.sum() > 0) and (args.gimmick_loss_weight > 0),
        "dropout": args.dropout,
        "heads": list(model.head_names),               # our (+ opp aux when on)
        "gimmick_heads": list(model.gimmick_head_names),
        "aux_opp_head": bool(args.aux_opp_head),
        "aux_opp_weight": args.aux_opp_weight,
        "opp_cond": bool(args.opp_cond),               # Level B: our heads read the predicted opp action
        # Era-4 2b: our heads read the PARTNER slot's action (teacher-forced at train,
        # sequentially decoded at serve). model_io rebuilds + pair-decodes off this stamp.
        "pair_cond": bool(args.pair_cond),
        "pair_dropout": args.pair_dropout,
        "augment_move_order": bool(args.augment_move_order),
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "val_frac": args.val_frac,
        "seed": args.seed,
        "data": folders,
        "val_data": args.val_data,
        "class_weight": args.class_weight,
        "patience": args.patience,
        # Demonstrator skill filtering / weighting (TIER-1 #1).
        "rating_min": args.rating_min,
        "rating_weight": bool(args.rating_weight),
        "rating_weight_floor": args.rating_weight_floor,
        "outcome_weight": bool(args.outcome_weight),
        "loss_weight": args.loss_weight,
        # Phase-2 offline advantage weighting (audit trail — the weights only
        # exist at train time; serve behaviour is unchanged).
        "adv_weight": args.adv_weight,
        "adv_beta": args.adv_beta,
        "adv_clip": [args.adv_clip_min, args.adv_clip_max],
        "adv_value_ckpt": (args.adv_value_ckpt or args.warm_start
                           if args.adv_weight != "off" else None),
    }
    config.update({
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "ff_mult": args.ff_mult,
        "value_readout": args.value_readout,
        # Phase 1a/1b match-memory core — model_io rebuilds from these; absent
        # (pre-memory) checkpoints default to 0/stateless.
        "memory_dim": args.memory_dim,
        "mem_layers": args.mem_layers,
        "mem_heads": args.mem_heads,
        "max_mem_len": args.max_mem_len,
        "sequence_len": args.sequence_len,
        "warm_start": args.warm_start,
        "encoded_cache": bool(args.encoded_cache),
        "mmap_cache": bool(args.mmap_cache),
        # Phase-2 archetype-z — model_io rebuilds from these; absent (pre-z)
        # checkpoints default to 0/off.
        "n_archetypes": (int(z_art["k"]) if z_art else 0),
        "z_dim": (args.z_dim if z_art else 0),
        "z_archetypes": args.z_archetypes,
        # Era-5 W1 own-team specialist weighting (audit trail; serve behaviour unchanged)
        "own_team": args.own_team,
        "own_team_species": own_species,
        "own_team_weight": (args.own_team_weight if args.own_team else None),
        "own_team_min_overlap": (args.own_team_min_overlap if args.own_team else None),
        "select_own_slice": bool(args.select_own_slice and args.own_team),
    })
    # The serve-side nearest-centroid lookup needs the centroid table — embed
    # it in the checkpoint so serve is self-contained (no sidecar artifact).
    z_artifact_slim = None
    if z_art:
        from v_dance.datatools.team_archetypes import artifact_slim
        z_artifact_slim = artifact_slim(z_art)

    # ── Train loop (val-weighting OFF so val loss/acc stay true) ───────────────
    best_top1 = -1.0
    epochs_no_improve = 0
    history: List[dict] = []
    start_epoch = 1
    resume_path = out_dir / RESUME_NAME
    config_fp = config_fingerprint(config)
    if getattr(args, "resume", False):
        rs = load_resume_state(resume_path, config_fp, device=device, label="train_bc")
        # Corpus-drift guard (2026-07-19): the config fingerprint covers the FOLDER LIST,
        # not folder CONTENTS — if the corpus grew while paused (e.g. new Type_C games),
        # the split/epoch semantics silently change. Refuse; the grown corpus is the NEXT
        # era's warm-start run, not this run's resume.
        _ex = rs.get("extra") or {}
        _sizes = (len(train_loader.dataset), len(val_loader.dataset))
        if _ex.get("dataset_sizes") is not None and tuple(_ex["dataset_sizes"]) != _sizes:
            sys.exit(f"[train_bc] --resume REFUSED: dataset sizes changed "
                     f"({tuple(_ex['dataset_sizes'])} -> {_sizes}) — the corpus changed "
                     f"while paused. Start a fresh run (warm-start) on the new corpus.")
        model.load_state_dict(rs["model_state"])
        optimizer.load_state_dict(rs["optimizer_state"])
        best_top1 = float(rs["best"])
        epochs_no_improve = int(rs["epochs_no_improve"])
        history = list(rs.get("history") or [])
        restore_rng(rs["rng"])
        start_epoch = int(rs["epoch"]) + 1
        print(f"[train_bc] RESUMED from {resume_path.name}: completed epoch "
              f"{rs['epoch']}, best val top1 {best_top1:.4f}, continuing at epoch "
              f"{start_epoch}/{args.epochs} (RNG streams restored).")
        if start_epoch > args.epochs:
            print("[train_bc] resume: epoch budget already exhausted — raise --epochs "
                  "to extend the run.")
    print(f"[train_bc] resumable: state -> {resume_path.name} after every epoch "
          f"(interrupt any time; continue with --resume)")
    for epoch in range(start_epoch, args.epochs + 1):
        _ep_t0 = time.time()
        tr = run_epoch(model, train_loader, device, optimizer, class_weight=class_weight,
                       gimmick_class_weight=gimmick_class_weight,
                       gimmick_loss_weight=args.gimmick_loss_weight,
                       opp_loss_weight=args.aux_opp_weight if args.aux_opp_head else 0.0,
                       sample_weighted=sample_weighted,
                       value_loss_weight=args.value_loss_weight,
                       progress_every=args.progress_every,
                       progress_label=f"e{epoch}",
                       progress_secs=args.progress_secs,
                       pair_dropout=args.pair_dropout)
        va = run_epoch(model, val_loader, device, optimizer=None)
        # Era-5 W1: the own-team slice (val examples whose demonstrator played ≥ N of our six)
        own_va = (run_epoch(model, own_val_loader, device, optimizer=None)
                  if own_val_loader is not None else None)
        history.append({"epoch": epoch, "train": tr, "val": va, "val_own": own_va})
        opp_str = f" | opp top1 {va['opp_top1']:.3f}" if args.aux_opp_head else ""
        own_str = (f" | own-slice top1 {own_va['top1']:.3f} (n={len(own_val_loader.dataset)})"
                   if own_va is not None else "")
        print(
            f"{time.strftime('%H:%M:%S')} epoch {epoch:3d} "
            f"[{(time.time() - _ep_t0) / 60.0:.1f} min] | "
            f"train loss {tr['loss']:.4f} top1 {tr['top1']:.3f} | "
            f"val loss {va['loss']:.4f} top1 {va['top1']:.3f} top3 {va['top3']:.3f} "
            f"(a {va['our_a_top1']:.3f} / b {va['our_b_top1']:.3f}) "
            f"| gim recall {va['gimmick_recall']:.3f} pos {va['gimmick_pos']} "
            f"| val win-acc {va['value_acc']:.3f} brier {va['value_brier']:.3f}{opp_str}{own_str}"
        )

        # Checkpoint selection: pooled val top1 (default) or, with --select-own-slice and a
        # large-enough slice, the own-team slice top1 (the specialist's target metric).
        use_own = bool(args.select_own_slice and own_va is not None
                       and len(own_val_loader.dataset) >= 2000)
        select_metric = own_va["top1"] if use_own else va["top1"]
        if select_metric > best_top1:
            best_top1 = select_metric
            epochs_no_improve = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_metrics": va,
                    "val_own_metrics": own_va,
                    "z_artifact": z_artifact_slim,
                },
                ckpt_path,
            )
            print(f"           ↑ new best ({'own-slice' if use_own else 'val'} top1 "
                  f"{best_top1:.3f}) -> {ckpt_path}")
        else:
            epochs_no_improve += 1
            if args.patience and epochs_no_improve >= args.patience:
                print(f"[train_bc] early stop at epoch {epoch} "
                      f"(no val top1 improvement for {args.patience} epochs)")
                break
        # Epoch-boundary resume state (atomic). Written AFTER the best-save/patience
        # bookkeeping so the restored counters are exact. The early-stop ``break`` above
        # skips this save deliberately — an early-stopped run is COMPLETE, not paused.
        save_resume_state(resume_path, epoch=epoch, model=model, optimizer=optimizer,
                          best=best_top1, epochs_no_improve=epochs_no_improve,
                          history=history, config_fp=config_fp,
                          extra={"dataset_sizes": [len(train_loader.dataset),
                                                   len(val_loader.dataset)]})

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
    ap.add_argument("--val-data", nargs="+", default=None,
                    help="REFERENCE corpus folder(s) that pin the val split (same "
                         "seed/val-frac reproduce that corpus's exact val replays, "
                         "e.g. the 0.585 anchor's). --data then contributes TRAIN "
                         "examples only; replay_id overlaps with the reference are "
                         "dropped. Do NOT list the same folders in both.")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--progress-every", type=int, default=0,
                    help="print an intra-epoch batch heartbeat every N TRAIN batches "
                         "(0 = off; e.g. 200 on a long run to confirm it's alive)")
    ap.add_argument("--progress-secs", type=float, default=600.0,
                    help="TIME-based intra-epoch heartbeat: print batch x/y (%%), rate and "
                         "epoch ETA every N seconds of a TRAIN epoch (default 600 = 10 min; "
                         "0 = off). ON by default so long epochs are never a silent wait.")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    # ── #27: the battle net is the per-mon set-attention AttnBCPolicy (attn-only) ──
    ap.add_argument("--d-model", type=int, default=128, help="attn: per-mon token width")
    ap.add_argument("--n-heads", type=int, default=4, help="attn: self-attention heads")
    ap.add_argument("--n-layers", type=int, default=2, help="attn: self-attention layers")
    ap.add_argument("--ff-mult", type=int, default=2, help="attn: feed-forward expansion")
    ap.add_argument("--value-readout", choices=["mean", "concat_active", "cls_query"],
                    default="mean", help="attn: value-head readout arm (#23 measured choice)")
    ap.add_argument("--class-weight", choices=["none", "balanced"], default="none",
                    help="weight the action loss by inverse class frequency to "
                         "counter majority-action bias (default: none)")
    ap.add_argument("--class-weight-cap", type=float, default=10.0,
                    help="cap on any single class weight (default: 10)")
    ap.add_argument("--own-team", default=None,
                    help="era-5 W1 specialist: a team (paste file path, pool team name, or paste "
                         "text). TRAIN examples are weighted 1 + --own-team-weight × (fraction of "
                         "this team's six present in the demonstrator's roster, read from each "
                         "replay's first line); no game is dropped, so the opponent side stays "
                         "general. Also reports an own-team val slice (overlap ≥ "
                         "--own-team-min-overlap of 6) per epoch.")
    ap.add_argument("--own-team-weight", type=float, default=3.0,
                    help="λ for --own-team: a full-overlap game counts (1+λ)× a zero-overlap one.")
    ap.add_argument("--own-team-min-overlap", type=int, default=4,
                    help="own-team val slice threshold in mons (of 6).")
    ap.add_argument("--select-own-slice", action="store_true",
                    help="best-checkpoint selection on the own-team slice top1 instead of pooled "
                         "val (fine-tune instrument; falls back to pooled when the slice < 2000).")
    ap.add_argument("--gimmick-loss-weight", type=float, default=1.0,
                    help="weight on the gimmick (mega) CE term in the total loss "
                         "(default: 1.0). The gimmick head is always class-balanced.")
    ap.add_argument("--value-loss-weight", type=float, default=1.0,
                    help="weight on the scalar value-head (win-prob) BCE term in the "
                         "total loss (default: 1.0; 0 disables the value head). #2")
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
    ap.add_argument("--opp-cond", action=argparse.BooleanOptionalAction, default=False,
                    help="Level B: feed the DETACHED predicted opp-action distribution into the OUR "
                         "action heads (best-response to anticipated opp). Requires --aux-opp-head. "
                         "Default OFF (the Level-A baseline).")
    ap.add_argument("--pair-cond", action=argparse.BooleanOptionalAction, default=False,
                    help="Era-4 2b: OUR heads read the PARTNER slot's action (trailing "
                         "action_dim one-hot; teacher-forced at train with random decode "
                         "order, sequentially decoded at serve). p(a)·p(b) -> p(a)·p(b|a). "
                         "Warm-start from an unconditioned ckpt zero-pads the new columns "
                         "(zero-cond = donor bit-exact, the kill switch). Default OFF.")
    ap.add_argument("--pair-dropout", type=float, default=0.25,
                    help="2b: probability the teacher-forced partner one-hot is dropped to "
                         "zeros (keeps the zero-cond/kill-switch path trained; default 0.25).")
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
    ap.add_argument("--closed-copy-weight", type=float, default=None,
                    metavar="LAM_CLOSED",
                    help="Era-4 arm 2a regime titration: split each HF open/__closed "
                         "twin pair to total ONE decision — closed copy gets LAM, open "
                         "copy 1−LAM (1.0 = closed-only; unpaired replays untouched). "
                         "Removes arm B's 2× tournament upweight while keeping the "
                         "closed VIEW. Errors out if the train set has no pairs.")
    ap.add_argument("--adv-weight", choices=["off", "exp", "filter"], default="off",
                    help="Phase-2 offline advantage weighting (TRAIN only): weight each "
                         "decision by A = outcome − V(s) using --adv-value-ckpt's value "
                         "head. exp = clip(exp(beta·A)) (Metamon Exp); filter = keep "
                         "only A>0 decisions. Composes with the rating/outcome weights.")
    ap.add_argument("--adv-beta", type=float, default=1.6,
                    help="exp-mode advantage temperature (keep conservative — the "
                         "value head's calibration is weak, see the P0.4 ruler note)")
    ap.add_argument("--adv-clip-min", type=float, default=0.2,
                    help="lower clip for exp-mode advantage weights")
    ap.add_argument("--adv-clip-max", type=float, default=5.0,
                    help="upper clip for exp-mode advantage weights")
    ap.add_argument("--adv-value-ckpt", default=None,
                    help="checkpoint whose TRAINED value head scores V(s) for the "
                         "advantage (defaults to --warm-start when omitted)")
    ap.add_argument("--z-archetypes", default=None,
                    help="Phase-2 z: team-archetype artifact (.npz from "
                         "datatools.team_archetypes). Joins each example's own-team "
                         "archetype id by (replay_id, perspective) and conditions "
                         "every head on a learned per-archetype embedding; the "
                         "centroids are embedded into the checkpoint for serve.")
    ap.add_argument("--z-dim", type=int, default=24,
                    help="archetype embedding width (only with --z-archetypes)")
    ap.add_argument("--loss-weight", type=float, default=0.5,
                    help="weight on decisions from games our side LOST when "
                         "--outcome-weight is on (default 0.5).")
    # ── Phase 1a/1b: match-memory core + sequence BC ───────────────────────────
    ap.add_argument("--sequence-len", type=int, default=1,
                    help="frames per training item (the last sequence-len decision "
                         "states of the same replay+perspective; the LAST frame is "
                         "supervised). 1 = stateless BC (unchanged default). "
                         ">1 requires --memory-dim.")
    ap.add_argument("--memory-dim", type=int, default=0,
                    help="width of the match-memory feature every head reads "
                         "(frame-stacked causal time-axis transformer). 0 = build "
                         "the exact stateless architecture (default).")
    ap.add_argument("--mem-layers", type=int, default=2,
                    help="memory core: causal transformer layers")
    ap.add_argument("--mem-heads", type=int, default=4,
                    help="memory core: attention heads")
    ap.add_argument("--max-mem-len", type=int, default=64,
                    help="memory core: maximum frames attended (positional table)")
    ap.add_argument("--warm-start", default=None,
                    help="path to a STATELESS BC checkpoint to initialise from "
                         "(with --memory-dim the new memory columns are zero-padded "
                         "so epoch-0 single-turn behaviour equals the checkpoint's).")
    ap.add_argument("--encoded-cache", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="per-folder disk cache of ENCODED examples (~35 min load → "
                         "seconds on a cache hit; fingerprint auto-invalidates on any "
                         "corpus/layout change). --no-encoded-cache to force fresh "
                         "encoding. Smoke limits always bypass the cache.")
    ap.add_argument("--mmap-cache", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="Step B low-RAM loader: page X from the on-disk encoded "
                         "caches (np.load mmap_mode='r') and fetch dataset rows "
                         "lazily instead of materializing the (N, STATE_DIM) matrix "
                         "in RAM. Needed for the full 77.8k MA-Bo3 corpus (~26 GB X) "
                         "on 32 GB RAM. Requires the encoded cache; incompatible "
                         "with --sequence-len > 1.")
    ap.add_argument("--loader-workers", type=int, default=0,
                    help="M6 (2026-07-11): DataLoader worker processes for the feed. "
                         "0 (default) = in-process, byte-identical legacy path. >0 "
                         "parallelizes item fetch + collate (the era-2 run fed the GPU "
                         "at ~20%% util = Python-bound); tensors ride torch shared "
                         "memory, mmap caches re-open per worker via x_src stamps. "
                         "Needs cache-backed data (not --limit-*/--no-encoded-cache). "
                         "Start with 4 on the 3070 Ti box.")
    ap.add_argument("--patience", type=int, default=0,
                    help="early-stop after N epochs with no val top-1 gain (0=off)")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit-transitions", type=int, default=None,
                    help="cap transitions read (smoke runs)")
    ap.add_argument("--limit-files", type=int, default=None,
                    help="cap JSONL files read (smoke runs)")
    # rename-wiring: default to the BC-ANCHOR home that the self-play configs' 'ckpt' read
    # (checkpoints_attn_pre_gen141/battle_base.pt), NOT the production serving dir checkpoints_attn/ (which
    # holds the self-play champion battle_selfplay_gen141.pt). A bare retrain produces battle_base.pt whose
    # only consumer is the KL-to-BC anchor — so it must land where that anchor is read, else it is an orphan
    # (neither served nor anchored). Back up the current battle_base.pt first if you want to keep it (the
    # established checkpoints_attn_pre_* backup pattern).
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run from <out>/resume_state.pt "
                         "(written after every completed epoch; model+optimizer+"
                         "counters+RNG restored; refuses a config-fingerprint "
                         "mismatch). Also extends a finished run when --epochs is "
                         "raised. Pass the SAME flags as the original run.")
    ap.add_argument("--out",
                    default=str(_HERE.parents[1] / "ai_train_scripts" / "BC_model" / "checkpoints_attn_pre_gen141"),
                    help="checkpoint output dir (default: the BC-anchor home the self-play configs read)")
    return ap.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
