"""Fair team-preview ruler (playbook S3): score MULTIPLE TP checkpoints on ONE val set.

Settles v6-vs-v7 style comparisons that training logs cannot (each run's internal val differs).
Examples are built ONCE from the reference folders with the CURRENT sbda recipe — on
closed-sheet corpora the v7 extractor's OTS overlay stays zero, so a v6 checkpoint is scored
on byte-identical inputs (the same guarantee `model_io`'s lockstep guard enforces at serve).
Each checkpoint is evaluated with its OWN vocab + architecture stamps via ``load_team_chooser``.

  python -m v_dance.eval.tp_val_report \
    --ckpt ai_train_scripts/teamPreview_model/checkpoints/teampreview_sbda.pt \
    --ckpt ai_train_scripts/teamPreview_model/checkpoints/teampreview_sbda_v7.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_REPO = Path(__file__).resolve().parents[2]
_PREP = _REPO / "data" / "vods" / "Prepared_training_data"
_DEFAULT_VAL = [str(_PREP / "Regulation_MA" / "Jsonl_TypeA"),
                str(_PREP / "Regulation_MA" / "Jsonl_TypeB"),
                str(_PREP / "Regulation_MB" / "Jsonl_TypeB"),
                str(_PREP / "Regulation_MB_Bo3" / "Jsonl_TypeB")]


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass
    ap = argparse.ArgumentParser(description="Same-val comparison of team-preview checkpoints.")
    ap.add_argument("--ckpt", action="append", required=True,
                    help="TP checkpoint (repeat; first = baseline)")
    ap.add_argument("--val-data", nargs="+", default=_DEFAULT_VAL)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--limit-files", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--joint-ab", action="store_true",
                    help="decode-level A/B on the FIRST ckpt: serve-faithful greedy top-k vs the "
                         "joint conditional subset re-decode (model_io._joint_bring_order). The "
                         "adoption gate for TP_JOINT_BRING.")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ONE shared example build (current sbda recipe; belief = the serve-path Pikalytics prior).
    from v_dance.formats import default_format, pikalytics_path_for
    from v_dance.parser.belief_state import BeliefState
    from v_dance.training.teampreview_dataset import (
        TeamPreviewDataset, examples_from_folders, feature_recipe, split_by_replay,
    )
    from v_dance.training.train_teampreview import run_epoch

    fmt = default_format()
    belief_path = pikalytics_path_for(fmt)
    if not (belief_path and Path(belief_path).exists()):
        raise SystemExit(f"[tp-val] missing Pikalytics belief for {fmt}: {belief_path}")
    belief = BeliefState(belief_path)
    feat_fn, feat_dim, schema = feature_recipe("sbda", belief)
    print(f"[tp-val] building examples ({schema}, feat_dim={feat_dim}) from {args.val_data}")
    t0 = time.time()
    examples, _stats = examples_from_folders(args.val_data, limit_files=args.limit_files,
                                             feat_fn=feat_fn)
    if not examples:
        raise SystemExit("[tp-val] no examples in --val-data")
    _train, val_ex = split_by_replay(examples, val_frac=args.val_frac, seed=args.seed)
    del examples, _train
    print(f"[tp-val] val set: {len(val_ex)} examples "
          f"({len({e['replay_id'] for e in val_ex})} replays, {time.time()-t0:.1f}s)")

    from v_dance.play.model_io import load_team_chooser
    results = []
    for path in args.ckpt:
        model, vocab, cfg = load_team_chooser(path, device=args.device)
        affinity_fn = None
        if cfg.get("use_teammate_bias"):
            from v_dance.training.tp_features import teammate_affinity_matrix
            from v_dance.training.teampreview_dataset import TEAM_SIZE
            affinity_fn = lambda sp: teammate_affinity_matrix(sp, belief, n=TEAM_SIZE)  # noqa: E731
        loader = DataLoader(TeamPreviewDataset(val_ex, vocab, feat_dim=feat_dim,
                                               affinity_fn=affinity_fn),
                            batch_size=args.batch_size, shuffle=False)
        t1 = time.time()
        m = run_epoch(model, loader, args.device, optimizer=None)
        m["mean_exact"] = (m["bring_exact"] + m["lead_exact"]) / 2
        m["ckpt"] = path
        m["schema"] = cfg.get("feature_schema")
        results.append(m)
        print(f"[tp-val] scored {path} in {time.time()-t1:.1f}s")

    base = results[0]
    print("\n" + "═" * 72)
    for i, r in enumerate(results):
        print(f"  [{'BASE' if i == 0 else f'CAND{i}'}] {r['ckpt']}  (schema {r['schema']})")
    print("═" * 72)
    rows = [("bring exact", "bring_exact"), ("bring overlap", "bring_overlap"),
            ("lead exact", "lead_exact"), ("lead overlap", "lead_overlap"),
            ("mean exact", "mean_exact"), ("loss (↓)", "loss")]
    for label, key in rows:
        cells = []
        for i, r in enumerate(results):
            d = "" if i == 0 else f" ({r[key] - base[key]:+.4f})"
            cells.append(f"{r[key]:7.4f}{d}")
        print(f"  {label:<14}" + "  ".join(cells))
    print(f"  (bring n={base['bring_n']}, lead n={base['lead_n']} — shared val split, seed {args.seed})")
    print("═" * 72)

    if args.joint_ab:
        for p in args.ckpt:                       # per-ckpt decode A/B (v6 vs an aug retrain in one run)
            _decode_ab(p, val_ex, belief, feat_dim, args)
    return 0


def _decode_ab(ckpt_path: str, val_ex, belief, feat_dim: int, args) -> None:
    """Serve-faithful decode A/B on ONE checkpoint: greedy per-mon top-k (the current
    ``team_order`` decode) vs the joint conditional subset re-decode. Both sides constrain the
    leads WITHIN the chosen bring (exactly what serve emits), so the numbers here are the real
    serve-behavior delta — run_epoch's table above scores raw logits and is NOT comparable.
    Rows with a non-standard label shape (≠4 brought / ≠2 leads / invalid masks) are skipped."""
    from v_dance.play.model_io import _joint_bring_order, load_team_chooser
    from v_dance.training.teampreview_dataset import TeamPreviewDataset, BRING_K, LEAD_K
    from v_dance.training.tp_features import teammate_affinity_matrix
    from torch.utils.data import DataLoader as _DL

    model, vocab, cfg = load_team_chooser(ckpt_path, device=args.device)
    affinity_fn = None
    if cfg.get("use_teammate_bias"):
        from v_dance.training.teampreview_dataset import TEAM_SIZE
        affinity_fn = lambda sp: teammate_affinity_matrix(sp, belief, n=TEAM_SIZE)  # noqa: E731
    loader = _DL(TeamPreviewDataset(val_ex, vocab, feat_dim=feat_dim, affinity_fn=affinity_fn),
                 batch_size=args.batch_size, shuffle=False)

    stats = {"greedy": [0, 0.0, 0], "joint": [0, 0.0, 0]}   # [bring_exact, bring_overlap, lead_exact]
    n_rows = n_skipped = 0
    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            bl_g, ll_g = model(batch["our_idx"].to(args.device), batch["opp_idx"].to(args.device),
                               batch["our_feat"].to(args.device), batch["opp_feat"].to(args.device),
                               batch.get("our_affinity").to(args.device)
                               if batch.get("our_affinity") is not None else None)
            bl_g, ll_g = np.asarray(bl_g.cpu()), np.asarray(ll_g.cpu())
            for r in range(bl_g.shape[0]):
                bring_lab = set(np.flatnonzero(np.asarray(batch["bring"][r]) > 0.5).tolist())
                lead_lab = set(np.flatnonzero(np.asarray(batch["lead"][r]) > 0.5).tolist())
                of_r = np.asarray(batch["our_feat"][r])
                valid_n = int((np.abs(of_r).sum(axis=-1) > 0).sum())
                if (len(bring_lab) != BRING_K or len(lead_lab) != LEAD_K
                        or float(batch["valid_bring"][r]) < 0.5
                        or float(batch["valid_lead"][r]) < 0.5 or valid_n < 5):
                    n_skipped += 1
                    continue
                # greedy-serve: top-k marginal bring, leads = top lead logits WITHIN the bring
                by_bring = sorted(range(valid_n), key=lambda i: -bl_g[r, i])
                g_bring = by_bring[:BRING_K]
                g_leads = sorted(g_bring, key=lambda i: -ll_g[r, i])[:LEAD_K]
                # joint-serve: the shared serve helper (single source of truth with team_order)
                aff_r = (batch["our_affinity"][r][None].to(args.device)
                         if batch.get("our_affinity") is not None else None)
                order, ok = _joint_bring_order(
                    model, np.asarray(batch["our_idx"][r]), of_r,
                    np.asarray(batch["opp_idx"][r]), np.asarray(batch["opp_feat"][r]),
                    aff_r, valid_n, BRING_K, LEAD_K, BRING_K, args.device)
                if not ok:
                    n_skipped += 1
                    continue
                j_bring, j_leads = order[:BRING_K], order[:LEAD_K]
                for key, bset, lset in (("greedy", set(g_bring), set(g_leads)),
                                        ("joint", set(j_bring), set(j_leads))):
                    stats[key][0] += int(bset == bring_lab)
                    stats[key][1] += len(bset & bring_lab) / BRING_K
                    stats[key][2] += int(lset == lead_lab)
                n_rows += 1
    print(f"\n  ── decode A/B (serve-faithful) on {ckpt_path} — {n_rows} rows "
          f"({n_skipped} skipped, {time.time()-t0:.1f}s) ──")
    g, j = stats["greedy"], stats["joint"]
    nz = max(n_rows, 1)
    for label, gi, ji in (("bring-set exact", g[0] / nz, j[0] / nz),
                          ("bring overlap", g[1] / nz, j[1] / nz),
                          ("lead-pair exact", g[2] / nz, j[2] / nz)):
        print(f"  {label:<16} greedy {gi:7.4f}   joint {ji:7.4f}   ({ji - gi:+.4f})")
    print("  GATE: joint adopted iff bring-set exact improves without a lead-pair regression.")
    print("═" * 72)


if __name__ == "__main__":
    raise SystemExit(main())
