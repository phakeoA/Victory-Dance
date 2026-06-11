"""Shared fixtures for the Victory-Dance test suite.

Run from data/scripts/:
    pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make data/scripts importable (so `import vod_parser` resolves to the package)
_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

_PROJECT_DATA = _SCRIPTS_DIR.parent          # data/
VOD_PATH = (
    _PROJECT_DATA / "vods" /
    "Gen9ChampionsVGC2026RegMA-2026-04-20-stevenhevgc-speedyturtle87.html"
)
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
    from vod_parser.pokedex import Pokedex
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
