"""Ladder rating vs rated games, one chart per format -> a self-contained HTML page (no plotting library).

Reads artifacts/human_benchmark/human_bench.jsonl: every rated game logs a bot ``rating_update`` row (``rating`` before,
``rating_after``, ``arm``) and a game row (``ai_team``, ``result``). Per format it draws the rating after each game,
the running PEAK (dashed), and a rolling EQUILIBRIUM (orange): over the last ``--window`` games, fit
delta = a + b*rating and take the rating where the expected gain is zero, -a/b (memory 20 S1 — the honest skill line;
peaks are mostly streaks). Vertical ticks mark team changes. Re-run any time; the log keeps the whole history.

    .venv/Scripts/python.exe tools/rating_chart.py                  # -> artifacts/charts/rating_chart.html
    .venv/Scripts/python.exe tools/rating_chart.py --window 400
"""
from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
W, H, PAD = 1100, 360, 48


def load(path: Path) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    teams, games, site = {}, {}, {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        tag = r.get("battle_tag") or ""
        if r.get("type") == "site_rating" and r.get("format"):      # the ladder's own numbers (truth)
            site.setdefault(r["format"], []).append(r)
            continue
        if "result" in r and tag:
            teams[tag] = r.get("ai_team")
        if r.get("type") == "rating_update" and r.get("rating") is not None and r.get("rating_after") is not None:
            m = re.search(r"(gen9[a-z0-9]+?)-\d+", tag)
            games.setdefault(m.group(1) if m else "?", []).append(
                {"tag": tag, "r0": r["rating"], "r1": r["rating_after"], "ts": r.get("ts", "")})
    for rows in games.values():
        for g in rows:
            g["team"] = teams.get(g["tag"])
    return games, site


def equilibrium(rows: list[dict], window: int) -> list[float | None]:
    out = []
    for i in range(len(rows)):
        win = rows[max(0, i - window + 1): i + 1]
        if len(win) < max(50, window // 3):
            out.append(None)
            continue
        xs = [g["r0"] for g in win]
        ys = [g["r1"] - g["r0"] for g in win]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
        out.append(mx - my / b if b < 0 else None)          # the rating where the fitted gain is 0
    return out


def svg(fmt: str, rows: list[dict], window: int, site: list[dict]) -> str:
    ys = [g["r1"] for g in rows]
    eq = equilibrium(rows, window)
    peak, best = [], -1e9
    for y in ys:
        best = max(best, y)
        peak.append(best)
    lo = min(ys + [e for e in eq if e]) - 20
    hi = max(ys + [e for e in eq if e]) + 20
    n = len(ys)
    X = lambda i: PAD + (W - 2 * PAD) * i / max(1, n - 1)
    Y = lambda v: H - PAD - (H - 2 * PAD) * (v - lo) / (hi - lo)
    path = lambda vals: " ".join(f"{'M' if k == 0 or vals[k - 1] is None else 'L'}{X(k):.1f},{Y(v):.1f}"
                                 for k, v in enumerate(vals) if v is not None)
    grid = "".join(f'<line x1="{PAD}" x2="{W - PAD}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="grid"/>'
                   f'<text x="{PAD - 6}" y="{Y(v) + 4:.1f}" class="ax" text-anchor="end">{v}</text>'
                   for v in range(int(lo // 100 + 1) * 100, int(hi), 100))
    ticks, prev = "", None
    for k, g in enumerate(rows):
        if g["team"] and g["team"] != prev:
            ticks += (f'<line x1="{X(k):.1f}" x2="{X(k):.1f}" y1="{PAD}" y2="{H - PAD}" class="team"/>'
                      f'<text x="{X(k) + 3:.1f}" y="{PAD - 6}" class="ax">{html.escape(g["team"])}</text>')
            prev = g["team"]
    import bisect
    stamps = [g["ts"] for g in rows]
    dots = "".join(f'<circle cx="{X(max(0, bisect.bisect_right(stamps, s["ts"]) - 1)):.1f}" '
                   f'cy="{Y(min(max(s["elo"], lo), hi)):.1f}" r="3" class="site"/>' for s in site)
    last_eq = next((e for e in reversed(eq) if e), None)
    title = (f"{fmt} — {n} rated games · now {ys[-1]} · peak {max(ys)} · "
             f"equilibrium {'%.0f' % last_eq if last_eq else 'n/a (too few games)'}"
             + (f" · SITE (truth) {site[-1]['elo']:.0f}, {site[-1]['w']}W-{site[-1]['l']}L" if site else ""))
    return (f'<h2>{html.escape(title)}</h2><svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)}">'
            f'{grid}{ticks}<path d="{path(peak)}" class="peak"/><path d="{path(ys)}" class="rating"/>'
            f'<path d="{path(eq)}" class="eq"/>{dots}'
            f'<text x="{W - PAD}" y="{H - 12}" class="ax" text-anchor="end">rated game #</text></svg>')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", default=str(_REPO / "artifacts" / "human_benchmark" / "human_bench.jsonl"))
    ap.add_argument("--out", default=str(_REPO / "artifacts" / "charts" / "rating_chart.html"))
    ap.add_argument("--window", type=int, default=300, help="games in the rolling equilibrium fit")
    a = ap.parse_args()
    games, site = load(Path(a.bench))
    body = "".join(svg(f, rows, a.window, site.get(f, [])) for f, rows in sorted(games.items(), reverse=True) if len(rows) >= 2)
    page = f"""<!doctype html><meta charset="utf-8"><title>Ladder rating</title>
<style>body{{background:#fff;color:#1f2328;font:14px system-ui,sans-serif;margin:16px;max-width:1140px}}
svg{{width:100%;height:auto}} .grid{{stroke:#e5e7eb}} .ax{{fill:#6b7280;font-size:11px}}
.rating{{fill:none;stroke:#2563eb;stroke-width:1.2}} .peak{{fill:none;stroke:#9ca3af;stroke-dasharray:4 3}}
.eq{{fill:none;stroke:#ea580c;stroke-width:2}} .team{{stroke:#16a34a;stroke-dasharray:2 3}} .site{{fill:#111827}}
h2{{font-size:15px;margin:22px 0 4px}}</style>
<p><b style="color:#2563eb">rating</b> · <b style="color:#9ca3af">running peak</b> ·
<b style="color:#ea580c">equilibrium ({a.window}-game fit: where expected gain = 0)</b> ·
<b style="color:#16a34a">team change</b> · <b>● site Elo (the truth)</b></p>{body}"""
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"[chart] {', '.join(f'{f}: {len(r)} games' for f, r in games.items())} -> {out}")


if __name__ == "__main__":
    main()
