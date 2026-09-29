"""
Level C / B0d — white-box turn-stepper corpus-validation gate (the GO/NO-GO for B1 search).

Walks the FULL corpus (4 dirs, 6660 jsonl, ~88 969 `turn` rows). For every `turn` decision-row it steps
`state_before_actions` through `white_box_sim` with the ACTUAL recorded joint action, then compares the
predicted next-state to the real `state_after_actions`:

  * per-mon |ΔHP%| (median / mean / p90; split by side and by stat-estimate mode),
  * a signed-Δ residual diagnostic (self-checks whether `state_after` is post-residual),
  * faint precision / recall / F1 (domain = mons alive before the turn),
  * % of turns with ALL mons within ±10%,
  * the ranked Tier-3 `UnmodelledLog` coverage-gap histogram,
  * the full honesty ledger of every turn/mon skipped, by reason (never silent).

Then prints a `B0d VERDICT: GO / NO-GO` against the §0 bar (median |ΔHP%| ≤ 8 AND faint F1 ≥ 0.90).

Design of record: docs/levelC_B0d_validation_design.md.  Identity-pairing / schema facts are verified there.
(Moved from scratch/levelC_b0_validation_probe.py in the 2026-09-10 refactor, Phase 1; its pure helpers are
unit-tested by tests/test_levelC_b0d_validation_2026_06_30.py and tests/test_levelC_b0d_variance_2026_06_30.py.)

Usage: PYTHONUTF8=1 .venv/Scripts/python.exe -X utf8 -m v_dance.eval.white_box_validation \
           [--limit N] [--corpus a,b,c,d] [--report-json PATH] [--max-examples K]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from v_dance.encoders.white_box_sim import white_box_sim, is_fainted, UnmodelledLog
from v_dance.dex.pokedex import norm_species

_REPO = Path(__file__).resolve().parents[2]          # v_dance/eval/ -> repo root
assert (_REPO / "pyproject.toml").is_file(), _REPO
_BASE = _REPO / "data" / "vods" / "Prepared_training_data"
_CORPUS = {
    "a": "Regulation_MA/Jsonl_TypeA",
    "b": "Regulation_MA/Jsonl_TypeB",
    "c": "Regulation_MB/Jsonl_TypeB",
    "d": "Regulation_MB_Bo3/Jsonl_TypeB",
}
_STATUS_RESIDUAL = {"brn", "psn", "tox"}
_SIDES = ("our", "opp")
# UnmodelledLog keys that are CORRECT behaviour, not a coverage gap (annotated in the histogram).
# "applied" = a now-MODELLED effect that fired (Helping Hand, recoil, drain, mega, …) — usage, not a gap.
_INFORMATIONAL = {"blocked_by_protect", "mover_fainted_before_acting", "toxic_no_escalation",
                  "no_live_target", "applied"}


# ── action reconstruction ─────────────────────────────────────────────────────
def _bench_index_for_species(snapshot: dict, side: str, species: str):
    """Resolve a recorded switch's species → its index in ``state_before[side_bench]`` (the stepper wants a
    bench index, the corpus records a species). Returns None if the mon is not in the snapshot bench
    (opp switching to a previously-UNREVEALED mon — honestly unresolvable)."""
    target = norm_species(species or "")
    if not target:
        return None
    for i, m in enumerate(snapshot.get(f"{side}_bench") or []):
        if m and norm_species(m.get("base_species") or m.get("species") or "") == target:
            return i
    return None


def reconstruct_actions(row: dict) -> tuple[dict, bool]:
    """Build the ``white_box_sim`` joint-action dict ``{slot_key: action}`` from the recorded
    ``our_actions`` + ``opp_actions_actual``. Returns ``(actions, switch_unresolved)`` — if an opp switch
    targets an unrevealed mon the turn is unresolvable (caller skips + counts)."""
    sb = row.get("state_before_actions") or {}
    actions: dict = {}
    unresolved = False
    for src in ("our_actions", "opp_actions_actual"):
        for a in (row.get(src) or []):
            slot = a.get("slot")
            if not slot:
                continue
            side = "our" if str(slot).startswith("our") else "opp"
            kind = a.get("action")
            if kind == "move":
                act = {"kind": "move", "move": a.get("move"), "target": a.get("target_slot")}
                if a.get("gimmick_index"):
                    act["mega"] = True               # stepper logs mega_not_applied (Tier-3)
                actions[slot] = act
            elif kind == "switch":
                bi = _bench_index_for_species(sb, side, a.get("species") or "")
                if bi is None:
                    unresolved = True
                else:
                    actions[slot] = {"kind": "switch", "bench_index": bi}
            # any other action kind → leave the slot absent (white_box_sim handles partial joints)
    return actions, unresolved


# ── identity pairing (active ∪ bench, by (side, base_species)) ─────────────────
def identity_map(snapshot: dict, skipped: Counter) -> dict:
    """Map ``(side, norm_species(base_species)) → mon`` over a snapshot's active ∪ bench. Skips (and counts,
    never silent) illusion/transform mons (unreliable identity) and species-less entries. Prefers the
    numeric-HP / in-play copy on the (species-clause-rare) duplicate key."""
    out: dict = {}
    for side in _SIDES:
        mons = list((snapshot.get(f"{side}_active") or {}).values()) + (snapshot.get(f"{side}_bench") or [])
        for m in mons:
            if not m:
                continue
            if m.get("illusion_active") or m.get("is_transformed"):
                skipped["illusion_transform"] += 1
                continue
            bs = m.get("base_species") or m.get("species")
            if not bs:
                skipped["no_base_species"] += 1
                continue
            key = (side, norm_species(bs))
            prev = out.get(key)
            if prev is None or (prev.get("hp_pct") is None and m.get("hp_pct") is not None):
                out[key] = m
    return out


def _alive(mon) -> bool:
    hp = mon.get("hp_pct") if mon else None
    return hp is not None and float(hp) > 0.0


# ── percentile helper ─────────────────────────────────────────────────────────
def _stats(xs: list) -> dict:
    if not xs:
        return {"n": 0, "median": None, "mean": None, "p90": None}
    a = np.asarray(xs, dtype=float)
    return {"n": int(a.size), "median": round(float(np.median(a)), 3),
            "mean": round(float(a.mean()), 3), "p90": round(float(np.percentile(a, 90)), 3)}


def _p_ko(hp_lo, hp_hi) -> float:
    """P(KO) ≈ outcome HP uniform over the NON-crit roll bracket [hp_lo, hp_hi] (hp_lo = lower HP = more
    damage). 1 if even the luckiest roll faints; 0 if even the unluckiest survives; else the linear fraction."""
    if hp_lo is None or hp_hi is None:
        return 0.0
    lo, hi = float(hp_lo), float(hp_hi)
    if hi <= 0.0:
        return 1.0
    if lo >= 0.0:
        return 0.0
    span = hi - lo
    return 1.0 if span <= 0 else max(0.0, min(1.0, (0.0 - lo) / span))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap files PER DIR (0 = all)")
    ap.add_argument("--corpus", type=str, default="a,b,c,d", help="comma list of corpus keys")
    ap.add_argument("--report-json", type=str, default="")
    ap.add_argument("--max-examples", type=int, default=12)
    args = ap.parse_args()

    dirs = [_CORPUS[k.strip()] for k in args.corpus.split(",") if k.strip() in _CORPUS]
    files: list[Path] = []
    for d in dirs:
        fs = sorted((_BASE / d).glob("*.jsonl"))
        files += fs[: args.limit] if args.limit else fs
    print(f"[B0d] {len(files)} files from {dirs} (limit/dir={args.limit or 'all'})", flush=True)

    # accumulators
    turns_stepped = 0
    skipped_decision: Counter = Counter()
    skipped_turn: Counter = Counter()           # switch_unresolved, step_error:*
    skipped_mon: Counter = Counter()            # illusion_transform, none_hp, absent_in_after, no_base_species
    file_errors = 0
    unmodelled = UnmodelledLog()

    dhp_all: list = []
    dhp_by_side = {"our": [], "opp": []}
    dhp_by_mode = {"exact": [], "distribution": [], "other": []}
    signed_all: list = []
    signed_residual_subset: list = []           # statused / sand-exposed mons
    tp = fp = fn = 0
    # faint attribution: does each miss fall on a turn that logged a Tier-3 coverage gap?
    fn_clean = fn_gap = fp_clean = fp_gap = 0
    fn_gap_cats: Counter = Counter()
    fp_gap_cats: Counter = Counter()
    # variance-aware faint (the reframe): KO-probability reliability + reachability brackets
    brier_sum = 0.0
    brier_n = 0
    rel_n = [0] * 10                             # reliability deciles by predicted P(KO)
    rel_faint = [0] * 10
    real_faint_total = 0
    reachable_faint = 0                          # real faint AND can_faint (max roll+crit could KO)
    must_faint_total = 0
    genuine_over = 0                             # real survive AND must_faint (luckiest roll still KOs)
    turns_with_cmp = 0
    turns_within10 = 0
    examples: list = []

    for fi, fp_path in enumerate(files):
        if fi and fi % 500 == 0:
            print(f"  .. {fi}/{len(files)} files, {turns_stepped} turns stepped", flush=True)
        try:
            rows = [json.loads(l) for l in fp_path.open(encoding="utf-8")]
        except Exception:
            file_errors += 1
            continue
        for row in rows:
            dt = row.get("decision_type")
            if dt != "turn":
                skipped_decision[str(dt)] += 1
                continue
            try:
                sb = row.get("state_before_actions") or {}
                sa = row.get("state_after_actions") or {}
                if not sb or not sa:
                    skipped_turn["missing_state"] += 1
                    continue
                actions, unresolved = reconstruct_actions(row)
                if unresolved:
                    skipped_turn["switch_unresolved"] += 1
                    continue
                predicted, log = white_box_sim(sb, actions)
                unmodelled.update(log)
                turns_stepped += 1

                before_map = identity_map(sb, skipped_mon)
                pred_map = identity_map(predicted, Counter())     # already counted via before
                after_map = identity_map(sa, skipped_mon)
                weather = str((sb.get("field") or {}).get("weather") or "").lower()
                is_sand = "sand" in weather

                # ---- HP / signed-Δ metric (predicted ∩ after, numeric hp both) ----
                turn_deltas: list = []
                for key, pmon in pred_map.items():
                    amon = after_map.get(key)
                    if amon is None:
                        continue
                    ph, ah = pmon.get("hp_pct"), amon.get("hp_pct")
                    if ph is None or ah is None:
                        skipped_mon["none_hp"] += 1
                        continue
                    d = abs(float(ph) - float(ah))
                    dhp_all.append(d)
                    dhp_by_side[key[0]].append(d)
                    mode = ((amon.get("stats_estimate") or {}).get("mode")) or "other"
                    dhp_by_mode.setdefault(mode if mode in dhp_by_mode else "other", []).append(d)
                    signed = float(ph) - float(ah)
                    signed_all.append(signed)
                    bmon = before_map.get(key)
                    statused = bool(bmon and str(bmon.get("status") or "").lower() in _STATUS_RESIDUAL)
                    if statused or is_sand:
                        signed_residual_subset.append(signed)
                    turn_deltas.append(d)
                    if d > 10 and len(examples) < args.max_examples:
                        examples.append(f"  {key[0]} {key[1]:<16} pred {float(ph):6.1f} vs real "
                                        f"{float(ah):6.1f}  |Δ|={d:5.1f}")
                if turn_deltas:
                    turns_with_cmp += 1
                    if max(turn_deltas) <= 10.0:
                        turns_within10 += 1

                # ---- faint P/R/F1 (domain = alive-in-before, identifiable in after) ----
                t_fp = t_fn = 0
                for key, bmon in before_map.items():
                    if not _alive(bmon):
                        continue
                    amon = after_map.get(key)
                    if amon is None:
                        skipped_mon["absent_in_after"] += 1
                        continue
                    pmon = pred_map.get(key)
                    if pmon is None:
                        skipped_mon["absent_in_pred"] += 1
                        continue
                    pred_faint = is_fainted(pmon)
                    act_faint = is_fainted(amon)
                    if pred_faint and act_faint:
                        tp += 1
                    elif pred_faint and not act_faint:
                        fp += 1
                        t_fp += 1
                    elif (not pred_faint) and act_faint:
                        fn += 1
                        t_fn += 1
                    # ---- variance-aware: KO-prob reliability + reachability brackets ----
                    php = pmon.get("hp_pct")
                    hp_lo = pmon.get("_hp_lo", php)        # max non-crit damage (lower HP)
                    hp_hi = pmon.get("_hp_hi", php)        # min damage (higher HP)
                    hp_floor = pmon.get("_hp_floor", php)  # max damage + crit (reachability)
                    must_faint = hp_hi is not None and float(hp_hi) <= 0.0
                    can_faint = hp_floor is not None and float(hp_floor) <= 0.0
                    pko = _p_ko(hp_lo, hp_hi)
                    brier_sum += (pko - (1.0 if act_faint else 0.0)) ** 2
                    brier_n += 1
                    b = min(9, int(pko * 10))
                    rel_n[b] += 1
                    rel_faint[b] += 1 if act_faint else 0
                    if act_faint:
                        real_faint_total += 1
                        if can_faint:
                            reachable_faint += 1
                    if must_faint:
                        must_faint_total += 1
                        if not act_faint:
                            genuine_over += 1
                # attribute this turn's misses to clean (roll-variance) vs gap (modelable) turns
                if t_fn or t_fp:
                    turn_gap_cats = {k.split(":")[0] for k in log if k.split(":")[0] not in _INFORMATIONAL}
                    if turn_gap_cats:
                        fn_gap += t_fn
                        fp_gap += t_fp
                        for gk in turn_gap_cats:
                            if t_fn:
                                fn_gap_cats[gk] += 1
                            if t_fp:
                                fp_gap_cats[gk] += 1
                    else:
                        fn_clean += t_fn
                        fp_clean += t_fp
            except Exception as e:
                skipped_turn[f"step_error:{type(e).__name__}"] += 1
                if len([k for k in skipped_turn if k.startswith("step_error")]) <= 5 and len(examples) < args.max_examples:
                    examples.append(f"  [step_error] {type(e).__name__}: {e}")
                continue

    # ── derived ──
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else None
    pct_within10 = (100.0 * turns_within10 / turns_with_cmp) if turns_with_cmp else None
    overall = _stats(dhp_all)
    median_dhp = overall["median"]
    # variance-aware (the reframe): KO-probability reliability + reachability
    brier = (brier_sum / brier_n) if brier_n else None
    reachable_recall = (reachable_faint / real_faint_total) if real_faint_total else None
    genuine_gap_rate = (1.0 - reachable_recall) if reachable_recall is not None else None
    genuine_over_rate = (genuine_over / must_faint_total) if must_faint_total else None
    reliability = [
        {"bin": f"{i / 10:.1f}-{(i + 1) / 10:.1f}", "n": rel_n[i],
         "empirical_faint_rate": round(rel_faint[i] / rel_n[i], 3) if rel_n[i] else None}
        for i in range(10)]
    go_legacy = (median_dhp is not None and median_dhp <= 8.0 and f1 is not None and f1 >= 0.90)
    go = (median_dhp is not None and median_dhp <= 8.0
          and brier is not None and brier <= 0.12
          and genuine_gap_rate is not None and genuine_gap_rate <= 0.10
          and genuine_over_rate is not None and genuine_over_rate <= 0.10)

    gap_hist = [(k, v) for k, v in unmodelled.most_common() if k.split(":")[0] not in _INFORMATIONAL]
    info_hist = [(k, v) for k, v in unmodelled.most_common() if k.split(":")[0] in _INFORMATIONAL]

    report = {
        "files": len(files), "corpus": dirs,
        "turns_stepped": turns_stepped,
        "skipped_decision": dict(skipped_decision),
        "skipped_turn": dict(skipped_turn),
        "skipped_mon": dict(skipped_mon),
        "file_errors": file_errors,
        "dhp_overall": overall,
        "dhp_by_side": {s: _stats(v) for s, v in dhp_by_side.items()},
        "dhp_by_mode": {m: _stats(v) for m, v in dhp_by_mode.items()},
        "signed_mean_overall": round(float(np.mean(signed_all)), 3) if signed_all else None,
        "signed_mean_residual_subset": round(float(np.mean(signed_residual_subset)), 3) if signed_residual_subset else None,
        "signed_residual_n": len(signed_residual_subset),
        "faint": {"tp": tp, "fp": fp, "fn": fn,
                  "precision": round(prec, 4) if prec is not None else None,
                  "recall": round(rec, 4) if rec is not None else None,
                  "f1": round(f1, 4) if f1 is not None else None},
        "faint_attribution": {
            "fn_on_clean_turn": fn_clean, "fn_on_gap_turn": fn_gap,
            "fp_on_clean_turn": fp_clean, "fp_on_gap_turn": fp_gap,
            "fn_gap_categories_top": fn_gap_cats.most_common(12),
            "fp_gap_categories_top": fp_gap_cats.most_common(12)},
        "pct_turns_within_10": round(pct_within10, 2) if pct_within10 is not None else None,
        "variance_aware": {
            "brier": round(brier, 4) if brier is not None else None,
            "reachable_recall": round(reachable_recall, 4) if reachable_recall is not None else None,
            "genuine_gap_rate": round(genuine_gap_rate, 4) if genuine_gap_rate is not None else None,
            "genuine_over_rate": round(genuine_over_rate, 4) if genuine_over_rate is not None else None,
            "real_faint_total": real_faint_total, "reachable_faint": reachable_faint,
            "must_faint_total": must_faint_total, "genuine_over": genuine_over,
            "reliability": reliability},
        "tier3_gap_top": gap_hist[:30],
        "tier3_informational_top": info_hist[:10],
        "verdict": "GO" if go else "NO-GO",
        "verdict_legacy_f1": "GO" if go_legacy else "NO-GO",
    }

    # ── console ──
    P = print
    P("\n==================  B0d white-box validation gate  ==================")
    P(f"turns stepped                         : {turns_stepped}")
    P(f"skipped decision-rows                 : {dict(skipped_decision)}")
    P(f"skipped turns                         : {dict(skipped_turn)}")
    P(f"skipped mons (by reason)              : {dict(skipped_mon)}")
    P(f"file parse errors                     : {file_errors}")
    P(f"\nper-mon |ΔHP%|  overall              : "
      f"median {overall['median']}  mean {overall['mean']}  p90 {overall['p90']}  (n={overall['n']})")
    for s in _SIDES:
        st = _stats(dhp_by_side[s])
        P(f"               {s:>4} side             : median {st['median']}  mean {st['mean']}  p90 {st['p90']}  (n={st['n']})")
    for m in ("exact", "distribution", "other"):
        st = _stats(dhp_by_mode[m])
        if st["n"]:
            P(f"          stat-mode {m:<12}    : median {st['median']}  mean {st['mean']}  p90 {st['p90']}  (n={st['n']})")
    P(f"\nsigned-Δ mean overall                 : {report['signed_mean_overall']}  "
      f"(≈0 ⇒ no global bias)")
    P(f"signed-Δ mean residual subset         : {report['signed_mean_residual_subset']}  "
      f"(n={report['signed_residual_n']}; ≈0 ⇒ state_after is POST-residual ✓; ≈+6 ⇒ pre-residual)")
    P(f"\nfaint (legacy binary, mean-roll)      : TP={tp} FP={fp} FN={fn}  "
      f"precision={report['faint']['precision']}  recall={report['faint']['recall']}  F1={report['faint']['f1']}")
    P(f"  FN attribution: clean-turn (roll/EV variance) {fn_clean}  vs  gap-turn (modelable) {fn_gap}")
    P(f"  FP attribution: clean-turn {fp_clean}  vs  gap-turn {fp_gap}")
    if fn_gap_cats:
        P(f"  top gap categories co-occurring with FN: {fn_gap_cats.most_common(8)}")
    P(f"% turns ALL mons within ±10%          : {report['pct_turns_within_10']}  (n_turns={turns_with_cmp})")
    va = report["variance_aware"]
    P("\n── variance-aware faint (the reframe) ──")
    P(f"KO-probability Brier score            : {va['brier']}   (lower = better calibrated; 0.12 bar)")
    P(f"reachable recall (real KO ≤ max+crit) : {va['reachable_recall']}   "
      f"(real faints {va['real_faint_total']}, reachable {va['reachable_faint']})")
    P(f"genuine GAP rate (real KO unreachable): {va['genuine_gap_rate']}   (≤0.10 bar — true under-model)")
    P(f"genuine OVER rate (phantom sure-KO)   : {va['genuine_over_rate']}   "
      f"(must-faint {va['must_faint_total']}, survived {va['genuine_over']}; ≤0.10 bar)")
    P("reliability (predicted P(KO) bin → empirical faint rate):")
    for r in va["reliability"]:
        if r["n"]:
            P(f"    P(KO) {r['bin']}  →  empirical {r['empirical_faint_rate']}   (n={r['n']})")
    P("\nTier-3 coverage-GAP histogram (top 30, ranked) — what to model next:")
    for k, v in gap_hist[:30]:
        P(f"    {v:>7}  {k}")
    if info_hist:
        P("\nTier-3 informational (correct behaviour, not a gap):")
        for k, v in info_hist[:10]:
            P(f"    {v:>7}  {k}")
    if examples:
        P("\nexamples (|Δ|>10 / errors):")
        for e in examples:
            P(e)
    P("\n--------------------------------------------------------------------")
    P("RE-FRAMED GO bar: median |ΔHP%| ≤ 8  AND  Brier ≤ 0.12  AND  genuine_gap ≤ 0.10  AND  genuine_over ≤ 0.10")
    P(f"        median={median_dhp}  Brier={va['brier']}  gap={va['genuine_gap_rate']}  over={va['genuine_over_rate']}")
    P(f"B0d VERDICT (re-framed)      : {report['verdict']}")
    P(f"B0d VERDICT (legacy F1≥0.90) : {report['verdict_legacy_f1']}  (median {median_dhp} / F1 {report['faint']['f1']})")
    P("====================================================================")

    if args.report_json:
        outp = Path(args.report_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(report, indent=2), encoding="utf-8")
        P(f"\n[report] wrote {outp}")


if __name__ == "__main__":
    main()
