"""Ladder rating vs rated games -> artifacts/charts/rating_chart.html (the chart code lives in v_dance/ui/rating_chart.py;
Mission Control's Rating tab serves the same page live).

    .venv/Scripts/python.exe tools/rating_chart.py                  # -> artifacts/charts/rating_chart.html
    .venv/Scripts/python.exe tools/rating_chart.py --window 400
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from v_dance.ui.rating_chart import DEFAULT_BENCH, build_page  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", default=str(DEFAULT_BENCH))
    ap.add_argument("--out", default=str(_REPO / "artifacts" / "charts" / "rating_chart.html"))
    ap.add_argument("--window", type=int, default=300, help="games in the rolling equilibrium fit")
    ap.add_argument("--theme", default="light", choices=["light", "dark"])
    a = ap.parse_args()
    page, counts = build_page(Path(a.bench), a.window, a.theme)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"[chart] {', '.join(f'{f}: {n} games' for f, n in counts.items())} -> {out}")


if __name__ == "__main__":
    main()
