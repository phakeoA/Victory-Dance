"""
eval_buckets.py — per-situation BC eval + action-head calibration (offline)
===========================================================================
A cheap, no-retrain diagnostic that answers two questions the single headline
``val top1`` cannot:

  1. WHERE does the clone fail?  Bucket val top-1 / top-3 by game PHASE —
     ``lead`` (turn ≤2), ``mid`` (3-6), ``endgame`` (≥7), and ``replacement``
     (forced post-faint switches) — plus by demonstrator rating band.  Cloning
     usually degrades in the endgame / on forced switches; this shows it.
  2. Is the policy CALIBRATED?  Expected-Calibration-Error (ECE) + a reliability
     table over the masked-softmax confidence of the chosen action, plus the
     multiclass Brier score.  A serve-side temperature (TIER-4) is only worth
     tuning if the raw policy is miscalibrated — this measures that.

It re-uses the EXACT training val split (same ``--val-frac`` / ``--seed``) so the
numbers line up with a training run.  Loads the saved dict checkpoint, runs the
frozen action heads over the held-out states, and reports — it never trains.

CLI (repo root):
    .venv\\Scripts\\python.exe ai_train_scripts/BC_model/eval_buckets.py \\
        --ckpt ai_train_scripts/BC_model/checkpoints/bc_best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
from v_dance.training.bc_dataset import (  # noqa: E402
    HEADS,
    examples_from_folders,
    split_by_replay,
)
from v_dance.encoders.state_encoder import ACTIONS_PER_SLOT  # noqa: E402

_NEG = -1e9


# ── Pure helpers (unit-testable without torch / a checkpoint) ──────────────────
def phase_of(turn: Optional[int], decision_type: Optional[str]) -> str:
    """Game-phase bucket for one decision."""
    if decision_type == "replacement":
        return "replacement"
    t = turn or 0
    if t <= 2:
        return "lead"
    if t <= 6:
        return "mid"
    return "endgame"


def rating_band(rating: Optional[float]) -> str:
    if rating is None:
        return "unknown"
    for lo, hi, label in [(0, 1700, "<1700"), (1700, 1800, "1700-1800"),
                          (1800, 1900, "1800-1900"), (1900, 1e9, "1900+")]:
        if lo <= rating < hi:
            return label
    return "unknown"


def masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Softmax over legal entries only (illegal → 0 probability)."""
    z = np.where(mask > 0, logits, _NEG).astype(np.float64)
    z -= z.max()
    e = np.exp(z) * (mask > 0)
    s = e.sum()
    return e / s if s > 0 else (mask > 0) / max((mask > 0).sum(), 1)


def expected_calibration_error(
    confidences: Sequence[float], corrects: Sequence[bool], n_bins: int = 10,
) -> Tuple[float, List[dict]]:
    """ECE over equal-width confidence bins.  Returns (ece, reliability_table)."""
    conf = np.asarray(confidences, dtype=np.float64)
    corr = np.asarray(corrects, dtype=np.float64)
    n = len(conf)
    if n == 0:
        return 0.0, []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    table: List[dict] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        cnt = int(in_bin.sum())
        if cnt == 0:
            continue
        acc = float(corr[in_bin].mean())
        avg_conf = float(conf[in_bin].mean())
        ece += (cnt / n) * abs(acc - avg_conf)
        table.append({"range": (round(lo, 2), round(hi, 2)), "n": cnt,
                      "confidence": round(avg_conf, 3), "accuracy": round(acc, 3)})
    return ece, table


# ── Checkpoint load (delegates to the canonical attn-only model_io loader) ─────
def _load_policy(ckpt_path: str, device: str = "cpu"):
    import torch
    from v_dance.play.model_io import load_bc_policy
    cfg = torch.load(ckpt_path, map_location=device, weights_only=False).get("config", {})
    model, _heads = load_bc_policy(ckpt_path, device=device)
    return model, cfg


def evaluate(model, examples: Sequence[dict], device: str = "cpu") -> dict:
    """Run the action heads over ``examples`` and bucket top1 / top3 / calibration."""
    import torch

    buckets: Dict[str, Dict[str, Dict[str, int]]] = {"phase": {}, "rating": {}}

    def _bump(group: str, key: str, correct1: int, correct3: int):
        b = buckets[group].setdefault(key, {"n": 0, "top1": 0, "top3": 0})
        b["n"] += 1
        b["top1"] += correct1
        b["top3"] += correct3

    confidences: List[float] = []
    corrects: List[bool] = []
    brier_sum = 0.0
    n_dec = 0
    overall = {"n": 0, "top1": 0, "top3": 0}
    # Value head (win prob) bucketed by phase: did sigmoid(value)>0.5 match the
    # actual game outcome, and the Brier (p-outcome)^2 — tells us WHERE the value
    # head is useful (a static eval is near-50/50 early, sharper late) (#2).
    value_buckets: Dict[str, Dict[str, float]] = {}
    v_overall = {"n": 0, "correct": 0, "brier": 0.0}

    with torch.no_grad():
        for ex in examples:
            x = torch.as_tensor(np.asarray(ex["x"], dtype=np.float32), device=device)
            actions, _, value = model(x)
            ph = phase_of(ex.get("turn"), ex.get("decision_type"))
            rb = rating_band(ex.get("rating"))

            won = ex.get("won")
            if won is not None:
                vp = 1.0 / (1.0 + np.exp(-float(np.asarray(value.detach().cpu()))))
                outcome = 1.0 if won else 0.0
                vb = value_buckets.setdefault(ph, {"n": 0, "correct": 0, "brier": 0.0})
                vb["n"] += 1
                hit = int((vp > 0.5) == bool(won))
                vb["correct"] += hit
                vb["brier"] += (vp - outcome) ** 2
                v_overall["n"] += 1
                v_overall["correct"] += hit
                v_overall["brier"] += (vp - outcome) ** 2
            for head in HEADS:
                if head not in ex["targets"]:
                    continue
                tgt = int(ex["targets"][head])
                mask = np.asarray(ex["masks"][head], dtype=np.float32)
                logits = np.asarray(actions[head].detach().cpu()).ravel()[:ACTIONS_PER_SLOT]
                probs = masked_softmax(logits, mask)
                pred = int(np.argmax(probs))
                k = min(3, int((mask > 0).sum()))
                top3 = set(np.argsort(-probs)[:k].tolist())
                c1 = int(pred == tgt)
                c3 = int(tgt in top3)

                overall["n"] += 1
                overall["top1"] += c1
                overall["top3"] += c3
                _bump("phase", ph, c1, c3)
                _bump("rating", rb, c1, c3)

                confidences.append(float(probs[pred]))
                corrects.append(bool(c1))
                # multiclass Brier: sum_k (p_k - y_k)^2 over legal actions
                y = np.zeros_like(probs)
                y[tgt] = 1.0
                brier_sum += float(np.sum((probs - y) ** 2))
                n_dec += 1

    ece, table = expected_calibration_error(confidences, corrects)
    value_by_phase = {k: {"n": v["n"],
                          "win_acc": round(v["correct"] / max(v["n"], 1), 3),
                          "brier": round(v["brier"] / max(v["n"], 1), 3)}
                      for k, v in value_buckets.items()}
    return {
        "n_decisions": overall["n"],
        "top1": overall["top1"] / max(overall["n"], 1),
        "top3": overall["top3"] / max(overall["n"], 1),
        "by_phase": _finalise(buckets["phase"]),
        "by_rating": _finalise(buckets["rating"]),
        "ece": ece,
        "brier": brier_sum / max(n_dec, 1),
        "reliability": table,
        "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "value_n": v_overall["n"],
        "value_win_acc": v_overall["correct"] / max(v_overall["n"], 1),
        "value_brier": v_overall["brier"] / max(v_overall["n"], 1),
        "value_by_phase": value_by_phase,
    }


def _finalise(group: Dict[str, Dict[str, int]]) -> Dict[str, dict]:
    out = {}
    for key, b in group.items():
        n = max(b["n"], 1)
        out[key] = {"n": b["n"], "top1": round(b["top1"] / n, 3),
                    "top3": round(b["top3"] / n, 3)}
    return out


def print_report(rep: dict) -> None:
    print("== BC per-situation eval + calibration =====================")
    print(f"  decisions   : {rep['n_decisions']}")
    print(f"  overall top1: {rep['top1']:.3f}   top3: {rep['top3']:.3f}")
    print(f"  mean conf   : {rep['mean_confidence']:.3f}   "
          f"ECE: {rep['ece']:.3f}   Brier: {rep['brier']:.3f}")
    _PHASE_ORDER = ["lead", "mid", "endgame", "replacement"]
    print("  -- by phase --")
    for k in _PHASE_ORDER + [k for k in rep["by_phase"] if k not in _PHASE_ORDER]:
        if k in rep["by_phase"]:
            b = rep["by_phase"][k]
            print(f"    {k:12s}: n {b['n']:6d}  top1 {b['top1']:.3f}  top3 {b['top3']:.3f}")
    print("  -- by rating band --")
    for k in ["<1700", "1700-1800", "1800-1900", "1900+", "unknown"]:
        if k in rep["by_rating"]:
            b = rep["by_rating"][k]
            print(f"    {k:12s}: n {b['n']:6d}  top1 {b['top1']:.3f}  top3 {b['top3']:.3f}")
    print("  -- reliability (confidence -> accuracy) --")
    for row in rep["reliability"]:
        print(f"    ({row['range'][0]:.2f},{row['range'][1]:.2f}]: n {row['n']:6d}"
              f"  conf {row['confidence']:.3f}  acc {row['accuracy']:.3f}")
    if rep.get("value_n"):
        print(f"  -- VALUE head (win-prob) -- overall win-acc {rep['value_win_acc']:.3f} "
              f"brier {rep['value_brier']:.3f}  (0.50 = chance, 0.25 brier = always-0.5)")
        for k in _PHASE_ORDER + [k for k in rep["value_by_phase"] if k not in _PHASE_ORDER]:
            if k in rep["value_by_phase"]:
                b = rep["value_by_phase"][k]
                print(f"    {k:12s}: n {b['n']:6d}  win-acc {b['win_acc']:.3f}  brier {b['brier']:.3f}")
    print("============================================================")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Per-situation BC eval + calibration")
    _prep = Path("data") / "vods" / "Prepared_training_data" / "Regulation_MA"
    default_data = [str(_prep / f"Jsonl_Type{t}") for t in ("A", "B", "C", "D")]
    ap.add_argument("--data", nargs="+", default=default_data)
    ap.add_argument("--ckpt", default=str(_HERE / "checkpoints" / "bc_best.pt"))
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="MUST match the training run to reproduce its val split")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit-files", type=int, default=None)
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not Path(args.ckpt).exists():
        print(f"[eval_buckets] checkpoint not found: {args.ckpt}", file=sys.stderr)
        return 2
    examples, _ = examples_from_folders(args.data, limit_files=args.limit_files)
    _, val = split_by_replay(examples, val_frac=args.val_frac, seed=args.seed)
    print(f"[eval_buckets] {len(val)} val examples "
          f"({len({e['replay_id'] for e in val})} replays)")
    model, cfg = _load_policy(args.ckpt, args.device)
    rep = evaluate(model, val, args.device)
    print_report(rep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
