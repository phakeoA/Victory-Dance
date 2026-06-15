"""Gap #6 per-replay breakdown — which replays still diverge on opp COMPOSITION.

Runs the opp-side/global audit over all 27 Zoroark/Ditto replays (both persp)
and reports, per replay, the count of structural (composition) divergences:
base/type1/type2/active/rev/fnt — the blocks that indicate a wrong mon in a slot
(as opposed to inherent boost/status/hp residuals).

Run: PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe data/scripts/tests/_gap6_perreplay.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "data" / "scripts"))
sys.path.insert(0, str(ROOT / "data" / "scripts" / "tests"))

import _parity_harness as H  # noqa: E402
import _gap5_full_audit as A  # noqa: E402
from vod_parser.replay_parser import extract_log_from_html  # noqa: E402

STRUCT = ("base", "type1", "type2", "active", "rev", "fnt", "hp", "status", "mega")


def find_edge_replays():
    out = []
    for v in (ROOT / "data" / "vods").rglob("*.html"):
        try:
            log = extract_log_from_html(v.read_text(encoding="utf-8")).lower()
        except Exception:
            continue
        if any(t in log for t in ("zoroark", "|replace|", "ditto", "imposter",
                                   "|-transform|")):
            out.append(v)
    return out


def main():
    rows = []
    for path in find_edge_replays():
        log = extract_log_from_html(path.read_text(encoding="utf-8"))
        per = Counter()
        detail = {}
        for persp in ("p1", "p2"):
            pmon, _glob = A.audit_log(log, persp)
            for blk in STRUCT:
                lst = pmon.get(blk, [])
                if lst:
                    per[blk] += len(lst)
                    detail.setdefault(persp, {})[blk] = [(t, s) for (t, s, _l, _o) in lst]
        struct_total = sum(per[b] for b in ("base", "type1", "type2", "active", "rev", "fnt"))
        if struct_total:
            rows.append((struct_total, path.name, dict(per), detail))
    rows.sort(reverse=True)
    print(f"replays with STRUCTURAL (composition) divergences: {len(rows)}\n")
    for total, name, per, detail in rows:
        print(f"[{total:3d}] {name}")
        print(f"      blocks: {per}")
        for persp, d in detail.items():
            for blk in ("base", "active", "rev", "fnt"):
                if blk in d:
                    print(f"      {persp} {blk}: {d[blk][:8]}")
        print()


if __name__ == "__main__":
    main()
