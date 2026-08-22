"""Regression for the scrape_pikalytics nature-bleed bug (B1-mechanics step 7a).

The "Best Abilities" / "Best Items" section slice is bounded by the next heading or the
"Nature / EV Spreads" marker. When that block is captured mid lazy-render the boundary can
be missed and the nature distribution bleeds into the section, so "Adamant 89.9%" parses as
an ability (verified live: Swampert M-B — 'Adamant' even outranked the real 'Torrent'). The
fix filters the finite nature set out of _parse_section. This test feeds a synthetic boundary
miss and asserts only the real abilities survive.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("crawl4ai")  # the scraper imports crawl4ai at module load

_REPO = Path(__file__).resolve().parents[1]
_SCRAPER = _REPO / "data" / "scripts" / "scrapers" / "scrape_pikalytics.py"


def _load_scraper():
    spec = importlib.util.spec_from_file_location("scrape_pikalytics", _SCRAPER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not _SCRAPER.exists(), reason="scraper not present")
def test_parse_section_drops_natures_on_boundary_miss():
    sp = _load_scraper()
    # abilities section bleeds straight into the nature list with NO boundary marker
    # (the lazy-render miss). Two-line name/pct layout, as Pikalytics renders abilities.
    md = (
        "## Best Abilities for Swampert\n"
        "Torrent\n53.6%\n"
        "Damp\n46.4%\n"
        "Adamant\n89.9%\n"
        "Jolly\n8.6%\n"
        "Brave\n0.6%\n"
        "## Best Items for Swampert\n"
        "Leftovers\n30.0%\n"
    )
    out = sp._parse_section(md, "Best Abilities")
    names = [a["name"] for a in out]
    assert names == ["Torrent", "Damp"], f"natures leaked into abilities: {names}"


def test_nature_noise_covers_all_natures():
    sp = _load_scraper()
    # the filter set must match the spread parser's nature alternation (no drift)
    assert len(sp._NATURE_NOISE) == 25
    for n in ("adamant", "jolly", "brave", "modest", "timid", "relaxed"):
        assert n in sp._NATURE_NOISE
