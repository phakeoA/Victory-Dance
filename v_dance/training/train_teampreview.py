"""
Team-preview ("bring") trainer for Victory-Dance (VGC Reg M-A).

Behaviour-cloning of the team-preview decision from parsed replays:

  * input  : both teams' 6-mon rosters (the matchup)
  * targets: which 4 we brought (bring head) + which 2 led (lead head)
  * loss   : per-mon binary cross-entropy; the bring head is trained ONLY on
             examples with a complete observed 4-bring (valid_bring), the lead
             head on every example (leads are always revealed turn 1)
  * metrics: val exact-set match and top-k overlap for both heads
  * output : checkpoints the best model (by mean exact-match) to --out

Run under the GPU venv:

    .venv\\Scripts\\python.exe ai_train_scripts\\teamPreview_model\\train_teampreview.py \\
        --data data\\vods\\Prepared_training_data\\Regulation_MA\\Jsonl_TypeB --epochs 40
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
from v_dance.training.teampreview_dataset import (  # noqa: E402
    BRING_K,
    LEAD_K,
    MON_FEAT_DIM,
    TEAM_SIZE,
    TeamPreviewDataset,
    build_vocab,
    examples_from_folders,
    feature_recipe,
    print_stats,
    split_by_replay,
)
from v_dance.models.teampreview_model import build_model  # noqa: E402


def _topk_set_metrics(logits: torch.Tensor, target: torch.Tensor, k: int,
                      sample_mask: Optional[torch.Tensor] = None):
    """Return (n_exact, sum_overlap, n) for 'predict the k-set' over a batch.

    ``sample_mask`` (bool, (B,)) restricts to valid rows (e.g. valid_bring)."""
    B = logits.shape[0]
    pred = logits.topk(k, dim=1).indices                        # (B, k)
    n = 0
    n_exact = 0
    sum_overlap = 0
    for i in range(B):
        if sample_mask is not None and not bool(sample_mask[i]):
            continue
        n += 1
        ps = set(pred[i].tolist())
        ts = set(torch.nonzero(target[i], as_tuple=False).flatten().tolist())
        ov = len(ps & ts)
        sum_overlap += ov
        if ps == ts:
            n_exact += 1
    return n_exact, sum_overlap, n


def run_epoch(model, loader, device, optimizer=None) -> Dict[str, float]:
    train = optimizer is not None
    model.train(train)

    tot_loss = 0.0
    n_batches = 0
    bring_exact = bring_overlap = bring_n = 0
    lead_exact = lead_overlap = lead_n = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            our_idx = batch["our_idx"].to(device)
            opp_idx = batch["opp_idx"].to(device)
            our_feat = batch["our_feat"].to(device)
            opp_feat = batch["opp_feat"].to(device)
            bring = batch["bring"].to(device)
            lead = batch["lead"].to(device)
            valid = batch["valid_bring"].to(device)
            # 15b-train.1: the teammate-bias prior (None unless the SBDA dataset precomputed it).
            our_aff = batch.get("our_affinity")
            if our_aff is not None:
                our_aff = our_aff.to(device)

            bring_logits, lead_logits = model(our_idx, opp_idx, our_feat, opp_feat, our_aff)

            # lead head: every example. bring head: only valid_bring rows.
            lead_loss = F.binary_cross_entropy_with_logits(lead_logits, lead)
            bring_bce = F.binary_cross_entropy_with_logits(
                bring_logits, bring, reduction="none").mean(dim=1)   # (B,)
            denom = valid.sum().clamp_min(1.0)
            bring_loss = (bring_bce * valid).sum() / denom
            loss = lead_loss + bring_loss

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            tot_loss += float(loss.item())
            n_batches += 1

            be, bo, bn = _topk_set_metrics(bring_logits, bring, BRING_K, valid > 0.5)
            le, lo, ln = _topk_set_metrics(lead_logits, lead, LEAD_K)
            bring_exact += be; bring_overlap += bo; bring_n += bn
            lead_exact += le; lead_overlap += lo; lead_n += ln

    bn = max(bring_n, 1); ln = max(lead_n, 1)
    return {
        "loss": tot_loss / max(n_batches, 1),
        "bring_exact": bring_exact / bn,
        "bring_overlap": bring_overlap / (bn * BRING_K),
        "lead_exact": lead_exact / ln,
        "lead_overlap": lead_overlap / (ln * LEAD_K),
        "bring_n": bring_n,
        "lead_n": lead_n,
    }


def train(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[train_teampreview] WARNING: cuda unavailable -> cpu")
        device = "cpu"

    folders = list(args.data)
    if args.type_a:
        folders.extend(args.type_a)

    # 15b-train.1: choose the per-mon feature recipe + (for the teammate-bias) the affinity provider.
    # 'legacy' = the original 46-dim dex net (no belief, no schema -> byte-identical). 'sbda' = the
    # shared tp_features extractor (FEAT_DIM, stamped feature_schema) — needs the SAME Pikalytics belief
    # the serve path resolves, so train==serve. teammate-bias only acts inside self-attention.
    belief = None
    affinity_fn = None
    if args.features == "sbda":
        from v_dance.parser.belief_state import BeliefState
        from v_dance.formats import pikalytics_path_for, default_format
        fmt = args.format or default_format()
        belief_path = Path(args.belief) if args.belief else pikalytics_path_for(fmt)
        if not (belief_path and Path(belief_path).exists()):
            raise SystemExit(f"[train_teampreview] --features sbda needs a Pikalytics belief; "
                             f"missing for {fmt}: {belief_path}")
        belief = BeliefState(belief_path)
        if args.teammate_bias and not args.self_attn:
            print("[train_teampreview] --teammate-bias requires self-attention (the bias acts inside "
                  "it) -> enabling --self-attn")
            args.self_attn = True
        if args.teammate_bias:
            from v_dance.training.tp_features import teammate_affinity_matrix
            affinity_fn = lambda sp: teammate_affinity_matrix(sp, belief, n=TEAM_SIZE)  # noqa: E731
        print(f"[train_teampreview] features=sbda belief={Path(belief_path).name} "
              f"self_attn={args.self_attn} cross_attn={args.cross_attn} "
              f"teammate_bias={args.teammate_bias}")
    feat_fn, feat_dim, feature_schema = feature_recipe(args.features, belief)

    print(f"[train_teampreview] loading from: {folders}")
    t0 = time.time()
    examples, stats = examples_from_folders(folders, limit_files=args.limit_files, feat_fn=feat_fn)
    print_stats(stats)
    print(f"[train_teampreview] {len(examples)} examples in {time.time()-t0:.1f}s")
    if not examples:
        raise SystemExit("[train_teampreview] no examples found")

    vocab = build_vocab(examples)
    train_ex, val_ex = split_by_replay(examples, val_frac=args.val_frac, seed=args.seed)
    print(f"[train_teampreview] vocab={len(vocab)} species | "
          f"split -> {len(train_ex)} train / {len(val_ex)} val "
          f"({len({e['replay_id'] for e in train_ex})} / "
          f"{len({e['replay_id'] for e in val_ex})} replays)")

    train_loader = DataLoader(
        TeamPreviewDataset(train_ex, vocab, feat_dim=feat_dim, affinity_fn=affinity_fn),
        batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(
        TeamPreviewDataset(val_ex, vocab, feat_dim=feat_dim, affinity_fn=affinity_fn),
        batch_size=args.batch_size, shuffle=False)

    model = build_model(
        vocab_size=len(vocab) + 1,   # +1 for the reserved PAD id 0
        feat_dim=feat_dim,
        emb_dim=args.emb_dim,
        hidden=args.hidden,
        dropout=args.dropout,
        device=device,
        use_self_attn=args.self_attn,
        use_cross_attn=args.cross_attn,
        attn_heads=args.attn_heads,
        use_teammate_bias=args.teammate_bias,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "teampreview_best.pt"
    config = {
        "vocab_size": len(vocab) + 1,
        "feat_dim": feat_dim,
        "emb_dim": args.emb_dim,
        "hidden": args.hidden,
        "dropout": args.dropout,
        "bring_k": BRING_K,
        "lead_k": LEAD_K,
        "lr": args.lr,
        "data": folders,
        "patience": args.patience,
        # 15b-train.1: architecture + feature-recipe stamps so model_io.load_team_chooser rebuilds the
        # exact net and uses_tp_features/the lockstep guard select the matching SERVE recipe. The
        # feature_schema is stamped ONLY for SBDA (legacy carries none -> uses_tp_features False).
        "use_self_attn": bool(args.self_attn),
        "use_cross_attn": bool(args.cross_attn),
        "attn_heads": args.attn_heads,
        "use_teammate_bias": bool(args.teammate_bias),
        "features": args.features,
    }
    if feature_schema:
        config["feature_schema"] = feature_schema

    best = -1.0
    epochs_no_improve = 0
    history: List[dict] = []
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, device, optimizer)
        va = run_epoch(model, val_loader, device, optimizer=None)
        history.append({"epoch": epoch, "train": tr, "val": va})
        print(
            f"epoch {epoch:3d} | train loss {tr['loss']:.4f} | "
            f"val loss {va['loss']:.4f} "
            f"lead exact {va['lead_exact']:.3f} ovlp {va['lead_overlap']:.3f} | "
            f"bring exact {va['bring_exact']:.3f} ovlp {va['bring_overlap']:.3f}"
        )
        score = 0.5 * (va["lead_exact"] + va["bring_exact"])
        if score > best:
            best = score
            epochs_no_improve = 0
            torch.save({"model_state": model.state_dict(), "config": config,
                        "vocab": vocab, "epoch": epoch, "val_metrics": va}, ckpt_path)
            print(f"           ↑ new best (mean exact {best:.3f}) -> {ckpt_path}")
        else:
            epochs_no_improve += 1
            if args.patience and epochs_no_improve >= args.patience:
                print(f"[train_teampreview] early stop at epoch {epoch} "
                      f"(no val mean-exact gain for {args.patience} epochs)")
                break

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"[train_teampreview] done. best mean-exact = {best:.3f}. ckpt: {ckpt_path}")
    return {"best": best, "history": history, "checkpoint": str(ckpt_path), "vocab_size": len(vocab)}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train the team-preview (bring) model")
    _prep = Path("data") / "vods" / "Prepared_training_data" / "Regulation_MA"
    default_data = [str(_prep / f"Jsonl_Type{t}") for t in ("A", "B", "C", "D")]
    ap.add_argument("--data", nargs="+", default=default_data,
                    help="folder(s) of per-replay .jsonl (default: all VOD types "
                         "A-D; missing/empty folders are skipped, files deduped)")
    ap.add_argument("--type-a", nargs="+", default=None,
                    help="extra folder(s) to append (deduped). Type A is already in "
                         "the default --data; use only for ad-hoc extra data.")
    # 15b-train.1: feature recipe + SBDA architecture (defaults reproduce the original legacy net)
    ap.add_argument("--features", choices=["legacy", "sbda"], default="legacy",
                    help="per-mon feature recipe: 'legacy' 46-dim dex (default) or 'sbda' tp_features")
    ap.add_argument("--self-attn", action="store_true",
                    help="SBDA: self-attention over our 6 (makes the synergy tags interact)")
    ap.add_argument("--cross-attn", action="store_true",
                    help="SBDA: our mons cross-attend to the opponent's 6")
    ap.add_argument("--attn-heads", type=int, default=4)
    ap.add_argument("--teammate-bias", action="store_true",
                    help="SBDA: Pikalytics co-occurrence prior as a self-attn bias (implies --self-attn)")
    ap.add_argument("--format", default=None,
                    help="format whose Pikalytics belief feeds SBDA features (default: active format)")
    ap.add_argument("--belief", default=None,
                    help="explicit Pikalytics json for SBDA features (default: --format's resolved file)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--emb-dim", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=0,
                    help="early-stop after N epochs with no val mean-exact gain (0=off)")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit-files", type=int, default=None)
    # T4.1: default to the PRODUCTION TP dir that model_io.DEFAULT_TP_CHECKPOINT serves from (was
    # _HERE/"checkpoints" = v_dance/training/checkpoints, a dir nothing loads → a retrained TP silently
    # landed where it was never served). Mirrors train_bc.py's _HERE.parents[1]/ai_train_scripts pattern.
    ap.add_argument("--out", default=str(
        _HERE.parents[1] / "ai_train_scripts" / "teamPreview_model" / "checkpoints"))
    return ap.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
