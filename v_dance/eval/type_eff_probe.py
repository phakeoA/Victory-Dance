"""#25-probe — type-effectiveness SENSITIVITY probe for the battle net (a reusable GATE, not a test).
(Moved from scratch/type_eff_probe.py in the 2026-09-10 refactor, Phase 1; the chain's gates and Mission
Control's Eval card call it as ``python -m v_dance.eval.type_eff_probe``; the default report path is now
``artifacts/logs/type_eff_probe_report.json``.)

Question (low-cost, no retrain): does the trained battle policy actually RESPOND to type
effectiveness when it picks a move, or is it insensitive (→ a resolved type-eff feature is worth a
layout change)?  Mirrors the 15b synergy-probe methodology: a clean causal intervention measured as
SIGNAL vs NOISE.

INTERVENTION (surgical, encoder-faithful — no byte-poking):
  Take a REAL decision state.  For each of our acting mon's DAMAGING + LEGAL move→target pairs,
  counterfactually TERASTALLIZE the *target* opponent to a chosen type so the move becomes
  immune(x0) / resisted(<1) / neutral(x1) / super-effective(>1).  Tera only rewrites the target's
  type one-hots (stats/ability untouched); the is_terastallized flag is held SET across *every*
  condition, so it cancels.  Re-encode via StateEncoder (guarantees a valid vector) and read the
  champion's masked-softmax probability of selecting that move at that target.

METRICS:
  * P(immune) < P(neutral) < P(super)            ... monotone => respects the type chart
  * signal_delta = mean[P(best) - P(worst)]      ... effectiveness-driven swing (want >> 0)
  * noise_delta  = mean[spread of P across several EQUI-NEUTRAL teras] ... should be ~0
  * immune_argmax_rate = frac. where the model still ARGMAXes a do-nothing move (lower=better)

VERDICT: respects type-eff iff signal_delta is clearly positive AND >> noise_delta.  If
signal_delta ~ noise_delta (or negative) the policy is type-eff INSENSITIVE => the feature is
justified.

  .venv/Scripts/python.exe -m v_dance.eval.type_eff_probe \
      --ckpt ai_train_scripts/BC_model/checkpoints_attn_era2/battle_base.pt --max-samples 800
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parents[2]          # v_dance/eval/ -> repo root
assert (_REPO / "pyproject.toml").is_file(), _REPO

from v_dance.encoders.state_encoder import (  # noqa: E402
    StateEncoder, move_slots_for_mon, norm_species, _get_moves_data,
)

# The 18 battle types usable as a single-type tera defender (Stellar's defensive chart is all
# x1 and ??? isn't a tera option — both are useless as probe defenders, so we exclude them).
BATTLE_TYPES: Tuple[str, ...] = (
    "BUG", "DARK", "DRAGON", "ELECTRIC", "FAIRY", "FIGHTING", "FIRE", "FLYING", "GHOST",
    "GRASS", "GROUND", "ICE", "NORMAL", "POISON", "PSYCHIC", "ROCK", "STEEL", "WATER",
)
HEAD_KEYS = ("our_a", "our_b")          # our acting slots → head index 0,1
OPP_KEYS = ("opp_a", "opp_b")           # target_code 0,1


# ── type-chart helpers (pure) ─────────────────────────────────────────────────
def gen9_type_chart() -> Dict[str, Dict[str, float]]:
    """poke-env's canonical Gen-9 chart: chart[DEFENDING_TYPE][ATTACKING_TYPE] = multiplier."""
    from poke_env.data import GenData
    return GenData.from_gen(9).type_chart


def effectiveness(move_type: str, def_types: Sequence[str], chart: Dict[str, Dict[str, float]]) -> float:
    """Damage multiplier of an attacking ``move_type`` against a defender with ``def_types``."""
    at = move_type.upper()
    mult = 1.0
    for dt in def_types:
        mult *= float(chart.get(dt.upper(), {}).get(at, 1.0))
    return mult


def pick_def_types(move_type: str, chart: Dict[str, Dict[str, float]]) -> Dict[str, str]:
    """Deterministically choose one single defender type per effectiveness bucket for ``move_type``.
    Buckets that don't exist for this move type (e.g. no single-type immunity) are omitted; neutral
    always exists.  Returns ``{level: DEF_TYPE}`` over a subset of immune/resist/neutral/super."""
    buckets: Dict[str, str] = {}
    for dt in BATTLE_TYPES:                       # sorted-stable iteration → deterministic pick
        m = effectiveness(move_type, [dt], chart)
        if m == 0.0:
            buckets.setdefault("immune", dt)
        elif m < 1.0:
            buckets.setdefault("resist", dt)
        elif m == 1.0:
            buckets.setdefault("neutral", dt)
        elif m > 1.0:
            buckets.setdefault("super", dt)
    return buckets


def neutral_types(move_type: str, chart: Dict[str, Dict[str, float]], k: int = 4) -> List[str]:
    """Up to ``k`` distinct single types that are all EXACTLY neutral (x1) to ``move_type`` — the
    noise control: re-tera'ing among these must not move a type-eff-respecting policy."""
    out = [dt for dt in BATTLE_TYPES if effectiveness(move_type, [dt], chart) == 1.0]
    return out[:k]


def masked_softmax(logits: np.ndarray, mask: Sequence[float]) -> np.ndarray:
    """Softmax over legal entries only (mask==1); illegal entries → 0 probability."""
    z = np.asarray(logits, dtype=np.float64).copy()
    m = np.asarray(mask, dtype=np.float64) > 0.5
    z[~m] = -1e30
    z -= z.max()
    e = np.exp(z)
    e[~m] = 0.0
    s = e.sum()
    return e / s if s > 0 else e


def tera_clone(sba: dict, opp_key: str, def_type: str) -> dict:
    """Deep-copy ``state_before_actions`` and terastallize ``opp_active[opp_key]`` to ``def_type``.
    Sets is_terastallized=True (held constant across all conditions → cancels) + the tera type."""
    s2 = copy.deepcopy(sba)
    tmon = (s2.get("opp_active") or {}).get(opp_key)
    tmon["is_terastallized"] = True
    tmon["known_tera_type"] = def_type
    return s2


def _alive(mon: Optional[dict]) -> bool:
    return bool(mon) and not mon.get("is_fainted") and (mon.get("hp_pct") or 0) > 0


def iter_probe_points(t: dict, moves_data: dict) -> Iterator[dict]:
    """Yield each damaging+legal (our move → opp target) decision in transition ``t`` as
    ``{head, head_idx, m_idx, action_idx, tcode, opp_key, move_type}``."""
    sba = t.get("state_before_actions") or {}
    our_active = sba.get("our_active") or {}
    opp_active = sba.get("opp_active") or {}
    mask_all = t.get("action_mask") or {}
    for h_idx, head in enumerate(HEAD_KEYS):
        mon = our_active.get(head)
        if not _alive(mon):
            continue
        row = mask_all.get(head)
        if not row or sum(row) == 0:
            continue
        slots = move_slots_for_mon(mon)
        for m_idx, (mv_name, _conf) in enumerate(slots):
            data = moves_data.get(norm_species(mv_name))
            if not data or data.get("category") == "status" or (data.get("basePower") or 0) <= 0:
                continue
            mtype = (data.get("type") or "").upper()
            if not mtype:
                continue
            for tcode, okey in enumerate(OPP_KEYS):
                a_idx = m_idx * 3 + tcode
                if a_idx >= len(row) or row[a_idx] != 1:
                    continue
                if not _alive(opp_active.get(okey)):
                    continue
                yield {"head": head, "head_idx": h_idx, "m_idx": m_idx, "action_idx": a_idx,
                       "tcode": tcode, "opp_key": okey, "move_type": mtype}


# ── model runner ──────────────────────────────────────────────────────────────
def _move_prob(model, head_names, enc, sba, turn, head_idx, action_idx, mask, device) -> float:
    from v_dance.play.model_io import head_logits
    x = enc.encode_snapshot(sba, turn=turn)
    logits = head_logits(model, head_names, x, device=device)[head_idx]
    return float(masked_softmax(logits, mask)[action_idx])


def run_probe(ckpt: str, data_folders: Sequence[str], max_samples: int, limit_files: Optional[int],
              device: str, verbose: bool = False) -> dict:
    from v_dance.play.model_io import load_bc_policy
    from v_dance.training.bc_dataset import iter_jsonl_files

    model, head_names = load_bc_policy(ckpt, device=device)
    enc = StateEncoder()
    chart = gen9_type_chart()
    moves_data = _get_moves_data()

    files: List[str] = []
    for folder in data_folders:
        files.extend(iter_jsonl_files(folder))
    if limit_files is not None:
        files = files[:limit_files]

    # per-sample records: P at each effectiveness level + noise spread
    levels = ("immune", "resist", "neutral", "super")
    rows: List[dict] = []
    n_points = 0

    for fp in files:
        if len(rows) >= max_samples:
            break
        try:
            with open(fp, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    t = json.loads(line)
                    turn = t.get("turn") or 0
                    sba = t.get("state_before_actions") or {}
                    for pt in iter_probe_points(t, moves_data):
                        if len(rows) >= max_samples:
                            break
                        n_points += 1
                        row = (t.get("action_mask") or {})[pt["head"]]
                        buckets = pick_def_types(pt["move_type"], chart)
                        # need at least the two extremes present to form a signal
                        present = [lv for lv in levels if lv in buckets]
                        if "super" not in buckets or not ({"immune", "neutral"} & set(buckets)):
                            continue
                        p_at: Dict[str, float] = {}
                        for lv in present:
                            s2 = tera_clone(sba, pt["opp_key"], buckets[lv])
                            p_at[lv] = _move_prob(model, head_names, enc, s2, turn,
                                                  pt["head_idx"], pt["action_idx"], row, device)
                        # noise control: re-tera target among equi-neutral types
                        nts = neutral_types(pt["move_type"], chart, k=4)
                        p_noise = []
                        for dt in nts:
                            s2 = tera_clone(sba, pt["opp_key"], dt)
                            p_noise.append(_move_prob(model, head_names, enc, s2, turn,
                                                      pt["head_idx"], pt["action_idx"], row, device))
                        worst = "immune" if "immune" in p_at else "neutral"
                        rec = {"move_type": pt["move_type"], "p": p_at,
                               "signal": p_at["super"] - p_at[worst],
                               "noise_spread": (max(p_noise) - min(p_noise)) if len(p_noise) >= 2 else 0.0,
                               "immune_argmax": None}
                        if "immune" in p_at:
                            # is the do-nothing move the argmax among legal in the immune condition?
                            s2 = tera_clone(sba, pt["opp_key"], buckets["immune"])
                            from v_dance.play.model_io import head_logits
                            x = enc.encode_snapshot(s2, turn=turn)
                            lg = head_logits(model, head_names, x, device=device)[pt["head_idx"]]
                            probs = masked_softmax(lg, row)
                            rec["immune_argmax"] = bool(int(np.argmax(probs)) == pt["action_idx"])
                        rows.append(rec)
                        if verbose and len(rows) % 100 == 0:
                            print(f"  ...{len(rows)} samples")
        except (OSError, json.JSONDecodeError):
            continue

    return _aggregate(rows, n_points, ckpt)


def _paired(rows: List[dict], hi: str, lo: str) -> dict:
    """Within-sample paired contrast P(hi) - P(lo) over samples that have BOTH conditions.
    Returns mean delta, the fraction with the CORRECT sign (hi>lo), and n."""
    deltas = [r["p"][hi] - r["p"][lo] for r in rows if hi in r["p"] and lo in r["p"]]
    if not deltas:
        return {"mean": None, "dir_correct": None, "n": 0}
    d = np.asarray(deltas, dtype=np.float64)
    return {"mean": float(d.mean()), "dir_correct": float(np.mean(d > 0)), "n": int(d.size)}


def _aggregate(rows: List[dict], n_points: int, ckpt: str) -> dict:
    n = len(rows)
    if n == 0:
        return {"ckpt": ckpt, "n_samples": 0, "verdict": "NO SAMPLES (no usable damaging move→target with super + immune/neutral)"}
    noi = np.array([r["noise_spread"] for r in rows], dtype=np.float64)
    noise_delta = float(noi.mean())

    # The honest measures are WITHIN-SAMPLE paired contrasts (the per-level means are over
    # different subsets and aren't directly comparable).
    su_vs_im = _paired(rows, "super", "immune")     # strongest: effective vs no-effect
    su_vs_ne = _paired(rows, "super", "neutral")    # fine grading among hits
    ne_vs_im = _paired(rows, "neutral", "immune")   # binary immunity awareness

    immune_rows = [r for r in rows if r.get("immune_argmax") is not None]
    immune_argmax_rate = (float(np.mean([r["immune_argmax"] for r in immune_rows]))
                          if immune_rows else float("nan"))

    def mean_at(level):
        v = [r["p"][level] for r in rows if level in r["p"]]
        return float(np.mean(v)) if v else None

    # Verdict keys on the STRONGEST contrast (super vs immune) against the noise floor, plus whether
    # fine grading (super vs neutral) exists.
    strong = su_vs_im["mean"] if su_vs_im["mean"] is not None else su_vs_ne["mean"]
    strong = strong if strong is not None else 0.0
    fine = su_vs_ne["mean"] or 0.0
    floor = max(noise_delta, 1e-6)
    if strong > 0.05 and strong > 2.0 * floor and fine > 0.02:
        verdict = (f"RESPECTS type-eff (graded): super-vs-immune={strong:+.3f}, super-vs-neutral={fine:+.3f} "
                   f">> noise={noise_delta:.3f} — a resolved type-eff feature buys little.")
    elif strong > 0.05 and strong > 2.0 * floor:
        verdict = (f"BINARY immunity-aware but NOT graded: super-vs-immune={strong:+.3f} >> noise={noise_delta:.3f}, "
                   f"but super-vs-neutral={fine:+.3f}~0 — feature helps for fine grading/rare matchups.")
    else:
        verdict = (f"INSENSITIVE to type-eff: super-vs-immune={strong:+.3f}, super-vs-neutral={fine:+.3f} "
                   f"<= ~2x noise={noise_delta:.3f} — the resolved-multiplier feature is JUSTIFIED.")

    return {
        "ckpt": ckpt, "n_probe_points": n_points, "n_samples": n,
        "P_immune_subsetmean": mean_at("immune"), "P_neutral_subsetmean": mean_at("neutral"),
        "P_super_subsetmean": mean_at("super"),
        "paired_super_vs_immune": su_vs_im, "paired_super_vs_neutral": su_vs_ne,
        "paired_neutral_vs_immune": ne_vs_im,
        "noise_delta": noise_delta,
        "immune_argmax_rate": immune_argmax_rate, "n_immune_samples": len(immune_rows),
        "verdict": verdict,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="#25 type-effectiveness sensitivity probe")
    from v_dance.play.model_io import DEFAULT_BC_CHECKPOINT    # 2026-10-04: was the deleted v17 gen141 file
    _prep = _REPO / "data" / "vods" / "Prepared_training_data" / "Regulation_MA" / "Jsonl_TypeB"
    ap.add_argument("--ckpt", default=str(DEFAULT_BC_CHECKPOINT))
    ap.add_argument("--data", nargs="+", default=[str(_prep)])
    ap.add_argument("--max-samples", type=int, default=800)
    ap.add_argument("--limit-files", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(_REPO / "artifacts" / "logs" / "type_eff_probe_report.json"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    print(f"[type_eff_probe] ckpt={args.ckpt}\n[type_eff_probe] data={args.data} "
          f"max_samples={args.max_samples} device={args.device}")
    rep = run_probe(args.ckpt, args.data, args.max_samples, args.limit_files, args.device, args.verbose)
    print("\n================= #25 TYPE-EFF PROBE =================")
    for k in ("n_probe_points", "n_samples", "P_immune_subsetmean", "P_neutral_subsetmean",
              "P_super_subsetmean", "noise_delta", "immune_argmax_rate", "n_immune_samples"):
        if k in rep:
            v = rep[k]
            print(f"  {k:26s}: {v:.4f}" if isinstance(v, float) else f"  {k:26s}: {v}")
    for k in ("paired_super_vs_immune", "paired_super_vs_neutral", "paired_neutral_vs_immune"):
        p = rep.get(k)
        if p:
            mean = f"{p['mean']:+.4f}" if p["mean"] is not None else "n/a"
            dc = f"{p['dir_correct']:.0%}" if p["dir_correct"] is not None else "n/a"
            print(f"  {k:26s}: mean={mean}  dir_correct={dc}  n={p['n']}")
    print(f"\n>>> VERDICT: {rep['verdict']}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"[type_eff_probe] report -> {args.out}")
    return rep


if __name__ == "__main__":
    main()
