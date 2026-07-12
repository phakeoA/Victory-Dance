"""
corpus_qa.py — JSONL training-corpus quality / label audit (regression gate)
============================================================================
A fast, dependency-light audit over the per-replay JSONL exports produced by
``bulk_parse_replays.py``.  Run it after every scrape + export to catch label
regressions BEFORE they reach a training run:

  * per ``decision_type`` (turn / replacement): transition counts, our-side
    null ``action_index`` rate, and illegal-under-mask targets (the codec
    invariant — ``annotate_transition_actions`` must guarantee 0),
  * gimmick (mega) coverage: how many transitions carry a ``gimmick_mask`` and
    how many mega TRAINING positives (``gimmick_index == 1``),
  * demonstrator rating histogram (our side) — the data #1 weighting reads,
  * duplicate ``replay_id`` across files (a replay exported twice → leakage),
  * perspective balance (p1 vs p2 — Type B dual-perspective should be ~even).

It deliberately does NOT encode states (no torch / no StateEncoder), so it scans
the whole corpus in seconds and stays independent of the training package.

CLI (from repo root):
    .venv\\Scripts\\python.exe data/scripts/corpus_qa.py \\
        data/vods/Prepared_training_data/Regulation_MA/Jsonl_TypeB

Exit code is non-zero when a HARD invariant fails (illegal-under-mask targets or
duplicate replay_ids) so it can gate CI.  ``--json`` dumps the full report.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence

# The two active slots we imitate; gimmick/action masks key on these names.
OUR_HEADS = ("our_a", "our_b")
# Rating histogram bucket edges (Showdown ladder) — matches the #1 weighting scale.
RATING_BUCKETS = [0, 1600, 1700, 1800, 1900, 2000, 10_000]


def _canonical_rid(rid: str) -> str:
    """M8 (2026-07-11): canonical replay-id for the duplicate scan — Type_C
    exports carry a ``battle-`` prefix and a closed-strip twin a ``__closed``
    suffix; raw keys would miss a same-battle cross-source duplicate.
    LOCAL COPY (this module is deliberately stdlib-only): keep in sync with
    ``bc_dataset.canonical_rid`` — tests/test_rid_dedupe.py asserts parity."""
    r = str(rid)
    if r.startswith("battle-"):
        r = r[len("battle-"):]
    if r.endswith("__closed"):
        r = r[: -len("__closed")]
    return r


def iter_jsonl_files(folder: str) -> List[str]:
    return sorted(glob.glob(os.path.join(str(folder), "**", "*.jsonl"), recursive=True))


def _first_action_per_slot(our_actions: Sequence[dict]) -> Dict[str, dict]:
    """First entry per slot (the turn-start decision) — mirrors bc_dataset so the
    audit's null/illegal counts match exactly what the loader would drop."""
    first: Dict[str, dict] = {}
    for act in our_actions or []:
        slot = act.get("slot")
        if slot is not None and slot not in first:
            first[slot] = act
    return first


def _our_meta(t: dict):
    """(rating_before, won) for our side, or (None, None) if absent."""
    players = t.get("players") or {}
    side = players.get("our_side")
    me = players.get(side) if isinstance(side, str) else None
    if not isinstance(me, dict):
        return None, None
    r = me.get("rating_before")
    r = float(r) if isinstance(r, (int, float)) else None
    winner, uname = t.get("winner"), me.get("username")
    won = bool(winner == uname) if (winner is not None and uname is not None) else None
    return r, won


def _rating_bucket(r: float) -> str:
    for lo, hi in zip(RATING_BUCKETS, RATING_BUCKETS[1:]):
        if lo <= r < hi:
            return f"{lo}-{hi}" if hi < 10_000 else f"{lo}+"
    return "other"


def audit_folders(folders: Sequence[str]) -> dict:
    """Scan ``folders`` of per-replay JSONL and return a structured audit report."""
    report: dict = {
        "files": 0,
        "transitions": 0,
        "by_decision_type": defaultdict(lambda: Counter()),
        "perspective": Counter(),
        "rating_hist": Counter(),
        "rating_min": None,
        "rating_max": None,
        "won": Counter(),                  # True / False / None (str keys)
        "gimmick": Counter(),              # has_mask / mega_positive / none
        "slot_decisions": 0,
        "null_index": 0,
        "illegal_under_mask": 0,
        "no_mask_skips": 0,                 # audit: all-zero/empty-mask slots the BC loader drops as no_mask
        "duplicate_replay_ids": [],
        "bad_lines": 0,
        "examples": {"files_scanned": []},
    }
    replay_to_files: Dict[str, set] = defaultdict(set)

    files: List[str] = []
    seen_paths: set = set()
    for folder in folders:
        for f in iter_jsonl_files(folder):
            ap = os.path.abspath(f)
            if ap not in seen_paths:
                seen_paths.add(ap)
                files.append(f)

    for fp in files:
        report["files"] += 1
        try:
            with open(fp, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        t = json.loads(line)
                    except json.JSONDecodeError:
                        report["bad_lines"] += 1
                        continue
                    # audit: key the dup-detector on the ABSPATH, not the basename — bulk_parse names every
                    # output {replay_id}.jsonl, so the same replay in two folders has the SAME basename and a
                    # basename key would collapse them to one (size 1) and miss the cross-folder duplicate the
                    # BC loader (abspath-deduped) would then DOUBLE-WEIGHT.
                    _audit_transition(t, report, os.path.abspath(fp), replay_to_files)
        except OSError:
            report["bad_lines"] += 1

    # Duplicate replay_id = same id across >1 distinct file.
    report["duplicate_replay_ids"] = sorted(
        rid for rid, fs in replay_to_files.items() if len(fs) > 1
    )

    # Normalise defaultdicts → plain dicts for clean JSON / printing.
    report["by_decision_type"] = {k: dict(v) for k, v in report["by_decision_type"].items()}
    report["perspective"] = dict(report["perspective"])
    report["rating_hist"] = dict(report["rating_hist"])
    report["won"] = {str(k): v for k, v in report["won"].items()}
    report["gimmick"] = dict(report["gimmick"])
    report.pop("examples", None)
    return report


def _audit_transition(t: dict, report: dict, fname: str, replay_to_files) -> None:
    report["transitions"] += 1
    dt = t.get("decision_type") or "unknown"
    bucket = report["by_decision_type"][dt]
    bucket["transitions"] += 1
    report["perspective"][t.get("perspective")] += 1

    rid = t.get("replay_id")
    if rid is not None:
        replay_to_files[_canonical_rid(rid)].add(fname)

    rating, won = _our_meta(t)
    if rating is not None:
        report["rating_hist"][_rating_bucket(rating)] += 1
        report["rating_min"] = rating if report["rating_min"] is None else min(report["rating_min"], rating)
        report["rating_max"] = rating if report["rating_max"] is None else max(report["rating_max"], rating)
    report["won"][won] += 1

    gmask_all = t.get("gimmick_mask")
    report["gimmick"]["has_mask" if gmask_all else "no_mask"] += 1

    mask_all = t.get("action_mask") or {}
    first = _first_action_per_slot(t.get("our_actions") or [])
    for head in OUR_HEADS:
        act = first.get(head)
        if act is None:
            continue
        report["slot_decisions"] += 1
        bucket["slot_decisions"] += 1
        ai = act.get("action_index")
        if ai is None:
            report["null_index"] += 1
            bucket["null_index"] += 1
        else:
            row = mask_all.get(head)
            if not row or sum(row) == 0:
                # audit: an all-zero / empty mask is treated by the BC loader (bc_dataset.transition_to_example)
                # as a SKIPPED no_mask slot, NOT illegal — mirror it so the audit's illegal count equals what
                # train_bc actually drops (the docstring's stated parity contract). The genuine codec invariant
                # (an illegal target under a NON-empty legal mask) is still hard-failed below.
                report["no_mask_skips"] += 1
                bucket["no_mask_skips"] += 1
            elif 0 <= ai < len(row) and row[ai] == 1:
                pass  # legal
            else:
                report["illegal_under_mask"] += 1
                bucket["illegal_under_mask"] += 1
        if act.get("gimmick_index") == 1:
            report["gimmick"]["mega_positive"] += 1


def print_report(report: dict) -> None:
    print("== corpus QA ===============================================")
    print(f"  files                : {report['files']}")
    print(f"  transitions          : {report['transitions']}")
    print(f"  slot decisions       : {report['slot_decisions']}")
    print(f"  null action_index    : {report['null_index']}")
    print(f"  no-mask skips (dropd) : {report['no_mask_skips']}  (slot decisions the BC loader drops)")
    print(f"  illegal-under-mask   : {report['illegal_under_mask']}  "
          f"{'OK' if report['illegal_under_mask'] == 0 else '<<< FAIL'}")
    print(f"  bad/unparseable lines: {report['bad_lines']}")
    print("  -- by decision_type --")
    for dt, c in sorted(report["by_decision_type"].items()):
        print(f"    {dt:12s}: tx {c.get('transitions',0):6d}  decisions {c.get('slot_decisions',0):6d}"
              f"  null {c.get('null_index',0):4d}  illegal {c.get('illegal_under_mask',0)}")
    print(f"  perspective          : {report['perspective']}")
    rmin, rmax = report["rating_min"], report["rating_max"]
    print(f"  our rating range     : {rmin} .. {rmax}")
    print(f"  rating histogram     : {dict(sorted(report['rating_hist'].items()))}")
    print(f"  outcome (won)        : {report['won']}")
    print(f"  gimmick coverage     : {report['gimmick']}")
    dups = report["duplicate_replay_ids"]
    print(f"  duplicate replay_ids : {len(dups)}  {'OK' if not dups else '<<< ' + ', '.join(dups[:5])}")
    print("============================================================")


def _default_folders() -> List[str]:
    # audit: this module moved from data/scripts/ to v_dance/datatools/, so "../vods" now resolves to
    # the non-existent <repo>/v_dance/vods — a no-argument run then scanned 0 files and exited 0 (a silent
    # FALSE-CLEAN audit). The corpus actually lives at <repo>/data/vods. Walk up TWO levels to the repo root.
    here = os.path.dirname(os.path.abspath(__file__))                     # v_dance/datatools
    prep = os.path.join(here, "..", "..", "data", "vods", "Prepared_training_data", "Regulation_MA")
    return [os.path.normpath(os.path.join(prep, f"Jsonl_Type{t}")) for t in ("A", "B", "C", "D")]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Audit BC JSONL training corpus")
    ap.add_argument("folders", nargs="*", default=None,
                    help="folder(s) of per-replay .jsonl (default: all Jsonl_Type{A..D})")
    ap.add_argument("--json", action="store_true", help="dump the full report as JSON")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero on ANY warning (default: only hard invariants)")
    args = ap.parse_args(argv)

    folders = args.folders or _default_folders()
    report = audit_folders(folders)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print_report(report)

    # Hard invariants that must hold for a clean export. audit: an EMPTY scan (0 files — a mis-pointed
    # default or an empty corpus) must NOT read as clean/exit 0; it gives false confidence the gate looked.
    if report.get("files", 0) == 0:
        print("<<< FAIL: corpus_qa scanned 0 files — check the folder path(s); an empty scan is NOT clean.",
              file=sys.stderr)
        return 1
    hard_fail = report["illegal_under_mask"] > 0 or bool(report["duplicate_replay_ids"])
    soft_fail = report["bad_lines"] > 0
    if hard_fail or (args.strict and soft_fail):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
