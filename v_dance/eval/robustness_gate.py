"""Per-era robustness gate (era-4 design, Pillar 0d) — Tier-2 reader over exploit curves.

Reads exploiter curve files (single runs or per-seed probe dirs) for a CANDIDATE and an
INCUMBENT net and prints PASS/FAIL. Metrics per curve (sustained rule everywhere — a lone
spike block never decides anything):

  * games-to-threshold: first games_trained whose WR >= --threshold for --sustain
    consecutive eval blocks (None = never reached within the measured range),
  * max-sustained WR: max over --sustain-block windows of the window's MINIMUM WR,
  * envelope max: max ``best_winrate`` when present (2026-07-20 meter hardening), else max WR.

FAIL if either (a) the candidate reaches threshold in < --speed-ratio x the incumbent's
games (faster-to-exploit beyond the allowed regression), or (b) the candidate's
max-sustained WR exceeds the incumbent's by > --wr-delta. Multi-path sides aggregate by
median across seeds (the 0c ensemble driver's <out>/seed<k>/ dirs drop straight in).

Tier-1 (the bc_val_report sharpness panel) is DOCUMENTATION-driven for now: the 2026-07-20
calibration showed the panel decisively detects armB-scale collapse (det_mass +0.023 ~ 8
sigma) but cannot separate era2-vs-cand-scale differences — so Tier-1 flags obvious
collapse cheaply, and THIS Tier-2 probe is mandatory for every deploy candidate.

Usage (calibration example — era-2 incumbent vs the era-3 arms):
    python -m v_dance.eval.robustness_gate \
        --candidate artifacts/logs/exploit_checkpoints_attn_era3_armB \
        --incumbent artifacts/logs/exploit_checkpoints_attn_era2
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import List, Optional


def _load_curve(path: Path) -> list:
    p = Path(path)
    if p.is_dir():
        p = p / "exploit_curve.json"
    if not p.is_file():
        sys.exit(f"[robustness_gate] no exploit_curve.json at {path}")
    curve = json.loads(p.read_text(encoding="utf-8"))
    return sorted(curve, key=lambda e: e.get("games_trained", 0))


def curve_metrics(curve: list, threshold: float, sustain: int) -> dict:
    wrs = [float(e["exploiter_winrate"]) for e in curve]
    games = [int(e["games_trained"]) for e in curve]
    games_to_thr: Optional[int] = None
    max_sustained = 0.0
    for i in range(len(wrs) - sustain + 1):
        window_min = min(wrs[i:i + sustain])
        max_sustained = max(max_sustained, window_min)
        if games_to_thr is None and window_min >= threshold:
            games_to_thr = games[i]
    envelope = max((float(e.get("best_winrate", e["exploiter_winrate"])) for e in curve),
                   default=0.0)
    return {"games_to_thr": games_to_thr, "max_sustained": round(max_sustained, 3),
            "envelope_max": round(envelope, 3), "blocks": len(curve),
            "max_games": games[-1] if games else 0}


def _aggregate(paths: List[str], threshold: float, sustain: int) -> dict:
    per = [curve_metrics(_load_curve(Path(p)), threshold, sustain) for p in paths]
    reached = [m["games_to_thr"] for m in per if m["games_to_thr"] is not None]
    # Median games-to-threshold counts a never-reached seed as +inf (harder to exploit);
    # only if the MAJORITY of seeds never reach it does the aggregate become None.
    if len(reached) * 2 > len(per):
        gtt = int(statistics.median(reached))
    else:
        gtt = None
    return {"per_seed": per, "games_to_thr": gtt,
            "max_sustained": round(statistics.median(m["max_sustained"] for m in per), 3),
            "envelope_max": round(max(m["envelope_max"] for m in per), 3),
            "min_measured": min(m["max_games"] for m in per)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Tier-2 robustness gate over exploiter curves.")
    ap.add_argument("--candidate", nargs="+", required=True,
                    help="curve file(s)/dir(s) for the deploy candidate (seeds aggregate)")
    ap.add_argument("--incumbent", nargs="+", required=True,
                    help="curve file(s)/dir(s) for the incumbent reference")
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--sustain", type=int, default=2,
                    help="consecutive blocks required (spike rejection)")
    ap.add_argument("--speed-ratio", type=float, default=0.75,
                    help="FAIL if candidate games-to-threshold < ratio x incumbent's")
    ap.add_argument("--wr-delta", type=float, default=0.05,
                    help="FAIL if candidate max-sustained WR > incumbent's + delta")
    args = ap.parse_args(argv)

    cand = _aggregate(args.candidate, args.threshold, args.sustain)
    inc = _aggregate(args.incumbent, args.threshold, args.sustain)

    def row(label, m):
        gtt = m["games_to_thr"] if m["games_to_thr"] is not None else "never"
        print(f"  {label:<10} games-to-{args.threshold:.2f}: {gtt!s:>7}   "
              f"max-sustained: {m['max_sustained']:.3f}   envelope: {m['envelope_max']:.3f}   "
              f"(measured to {m['min_measured']}g, {len(m['per_seed'])} seed(s))")

    print(f"[robustness_gate] threshold={args.threshold} sustain={args.sustain} "
          f"speed-ratio={args.speed_ratio} wr-delta={args.wr_delta}")
    row("candidate", cand)
    row("incumbent", inc)

    fails = []
    if cand["games_to_thr"] is not None:
        if inc["games_to_thr"] is None:
            fails.append("candidate reaches threshold; incumbent never did in its measured range")
        elif cand["games_to_thr"] < args.speed_ratio * inc["games_to_thr"]:
            fails.append(f"faster-to-exploit: {cand['games_to_thr']}g < "
                         f"{args.speed_ratio} x {inc['games_to_thr']}g")
    if cand["max_sustained"] > inc["max_sustained"] + args.wr_delta:
        fails.append(f"max-sustained WR regression: {cand['max_sustained']} > "
                     f"{inc['max_sustained']} + {args.wr_delta}")
    # A candidate measured over far fewer games than the incumbent can pass vacuously —
    # surface that instead of letting it slide silently.
    if cand["min_measured"] < 2500:
        fails.append(f"insufficient measurement: {cand['min_measured']}g < the EXPLOIT@2500 minimum")

    if fails:
        print("VERDICT: FAIL")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("VERDICT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
