"""Shared fixtures for the Victory-Dance test suite.

Run from the repo root (the `v_dance` package is pip install -e .):
    pytest tests/ -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

# tests/ lives at the repo root; data/ is the sibling data tree.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_DATA = _REPO_ROOT / "data"
# The example VOD lives under data/vods/, which is organised into per-type
# subfolders (Type_A … Type_D) that may be reshuffled — search for it.
_VOD_NAME = "Gen9ChampionsVGC2026RegMA-2026-04-20-stevenhevgc-speedyturtle87.html"
VOD_PATH = next(
    (p for p in [_PROJECT_DATA / "vods" / "Type_B" / _VOD_NAME,
                 _PROJECT_DATA / "vods" / _VOD_NAME]
     if p.exists()),
    None,
) or next((_PROJECT_DATA / "vods").rglob(_VOD_NAME),
          _PROJECT_DATA / "vods" / _VOD_NAME)
POKEDEX_PATH = _PROJECT_DATA / "pokedex.json"


@pytest.fixture(scope="session")
def vod_html() -> str:
    assert VOD_PATH.exists(), f"example VOD missing: {VOD_PATH}"
    return VOD_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def vod_path() -> Path:
    return VOD_PATH


@pytest.fixture(scope="session")
def dex():
    from v_dance.parser.vod_parser.pokedex import Pokedex
    return Pokedex(POKEDEX_PATH)


def make_log(*lines: str) -> str:
    """Build a minimal Showdown protocol log from raw lines."""
    return "\n".join(lines)


# ── Synthetic log building blocks (importable from tests) ────────────────
HEADER = [
    "|player|p1|alice|101|1500",
    "|player|p2|bob|102|1500",
    "|teamsize|p1|4",
    "|teamsize|p2|4",
    "|tier|[Gen 9 Champions] VGC 2026 Reg M-A",
    "|poke|p1|Meganium, L50, M|",
    "|poke|p1|Incineroar, L50, F|",
    "|poke|p2|Aerodactyl, L50, M|",
    "|poke|p2|Palafin, L50, M|",
]
