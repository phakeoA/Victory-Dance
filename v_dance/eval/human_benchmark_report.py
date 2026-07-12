"""Human-benchmark report — the GOAL METRIC readout (docs/human_benchmark_design.md).

Reads ``artifacts/human_benchmark/human_bench.jsonl`` (rows appended by ``play_vs_human``) and
prints: the overall tally with a Wilson CI, a per-session (per-set) table, and the
**exploitability curve** — AI win% by game index within a session, pooled. A DROPPING curve
means the human is learning the bot's habits faster than the bot copes (G3 fails); a FLAT
curve means an exploit doesn't keep paying (G3 holds).

Usage:
  python -m v_dance.eval.human_benchmark_report [--log PATH] [--note TEXT] [--min-games N]
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_LOG = _REPO / "artifacts" / "human_benchmark" / "human_bench.jsonl"


def wilson(w: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for w wins in n games (0..1)."""
    if n == 0:
        return 0.0, 1.0
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def load_rows(path: Path) -> list[dict]:
    """Game rows, with any ``rating_update`` rows FOLDED IN by battle_tag (2026-07-10: on the
    browser transport the server's ladder-rating lines arrive after the game row was written —
    the online harness appends them as separate crash-safe rows instead of rewriting the log)."""
    rows, updates = [], []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                print(f"  [warn] unparseable row skipped: {line[:80]}")
                continue
            (updates if r.get("type") == "rating_update" else rows).append(r)
    if updates:
        by_tag = {r.get("battle_tag"): r for r in rows if r.get("battle_tag")}
        for u in updates:
            g = by_tag.get(u.get("battle_tag"))
            if g is not None:
                for k in ("rating", "opponent_rating"):
                    if u.get(k) is not None and g.get(k) is None:
                        g[k] = u[k]
    return rows


def _tally(rows) -> tuple[int, int, int, int]:
    ai = sum(1 for r in rows if r.get("result") == "ai")
    hu = sum(1 for r in rows if r.get("result") == "human")
    dr = sum(1 for r in rows if r.get("result") == "draw")
    return ai, hu, dr, ai + hu + dr


def report_text(rows: list[dict], min_games: int = 1) -> str:
    """The full report as a string — shared by the CLI and the online session logger
    (2026-07-10: the harness appends this to artifacts/logs/online_<session>.log on exit)."""
    out: list[str] = []
    out.append("== human benchmark ==========================================")
    ai, hu, dr, n = _tally(rows)
    lo, hi = wilson(ai, max(1, ai + hu))                     # CI over decisive games
    out.append(f"  games                : {n}  (AI {ai} / human {hu} / draw {dr})")
    out.append(f"  AI win% (decisive)   : {100 * ai / max(1, ai + hu):.1f}%   Wilson95 [{100 * lo:.1f}, {100 * hi:.1f}]")
    notes = sorted({r.get("note") or "" for r in rows})
    if len(notes) > 1:
        out.append("  -- by note (era) --")
        for note in notes:
            sub = [r for r in rows if (r.get("note") or "") == note]
            a, h, d, m = _tally(sub)
            out.append(f"    {note or '(none)':<20}: {m} games, AI {a} / human {h} / draw {d}")

    out.append("  -- sessions (sets) --")
    by_sess: dict = defaultdict(list)
    for r in rows:
        by_sess[r.get("session_id", "?")].append(r)
    for sid in sorted(by_sess):
        sub = by_sess[sid]
        a, h, d, m = _tally(sub)
        note = sub[0].get("note") or ""
        team = sub[0].get("ai_team") or "?"
        out.append(f"    {sid}  [{note}] AI on {team:<16} : {m} games  AI {a} / human {h} / draw {d}")

    out.append("  -- exploitability curve (AI win% by game index within a session, pooled) --")
    by_idx: dict = defaultdict(list)
    for r in rows:
        by_idx[int(r.get("game_idx", 0))].append(r)
    for idx in sorted(by_idx):
        sub = by_idx[idx]
        a, h, d, m = _tally(sub)
        if m < min_games:
            continue
        if a + h == 0:                                   # draws only -> no decisive rate to show
            out.append(f"    game {idx:>2}:   n/a       ({a}W/{h}L/{d}D of {m})")
            continue
        bar = "#" * round(20 * a / (a + h))
        out.append(f"    game {idx:>2}: {100 * a / (a + h):5.1f}% AI  ({a}W/{h}L/{d}D of {m})  {bar}")
    out.append("  (flat curve = an exploit stops paying, G3 holds; dropping = the human is learning the bot)")
    out.append("=============================================================")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Human-benchmark report (goal metric + exploitability curve).")
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG)
    ap.add_argument("--note", default=None, help="only rows whose note equals this (an era/ckpt tag).")
    ap.add_argument("--min-games", type=int, default=1,
                    help="curve: only show game indices with at least this many pooled games.")
    args = ap.parse_args()

    if not args.log.is_file():
        raise SystemExit(f"[bench] no log at {args.log} — play some benchmark sets first "
                         f"(python -m v_dance.play.play_vs_human).")
    rows = load_rows(args.log)
    if args.note is not None:
        rows = [r for r in rows if r.get("note") == args.note]
    if not rows:
        raise SystemExit("[bench] no rows match.")
    print(report_text(rows, min_games=args.min_games))


if __name__ == "__main__":
    main()
