"""
Format registry + resolver — the SINGLE source of truth for which Pokémon
Champions *doubles* format the bot targets, and the per-format resources.

WHY: the state encoder + the BC/PPO models are mechanic/feature-based and so are
FORMAT-AGNOSTIC — the same checkpoint plays any "[Gen 9 Champions] VGC 2026 Reg *"
doubles format (verified live on both M-A and M-B). Only two things are
genuinely format-specific: the *battle-format string* and the *Pikalytics belief
data*. This module keeps both configurable so the bot is backwards-compatible
across regulations (M-A, M-B, and future regs) instead of hardcoded to one.

Selection precedence for the active/default format (highest first):
  1. the ``VDANCE_BATTLE_FORMAT`` env var — SPAWN-SAFE: inherited by mp workers,
     so a ``--format`` chosen in the parent reaches every collect/eval child
     (which re-import this module fresh and read the env var).
  2. the ``active: true`` entry in data/regulations/vod_parser_format_names.json
  3. the hardcoded ``FALLBACK_DEFAULT``.

Pikalytics resolution falls back to the canonical M-A file when a format-specific
``pikalytics_<reg>.json`` does not exist yet (M-A ⊂ M-B, so the M-A usage prior is
a valid stand-in until the new reg's data is scraped — tasks 17.3/17.4). The
belief is therefore never silently zeroed for a newly-added reg.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]          # v_dance/ -> repo root
_DATA_DIR = _REPO_ROOT / "data"
_REGISTRY = _DATA_DIR / "regulations" / "vod_parser_format_names.json"

# Hardcoded last-resort default + the canonical belief prior (M-A is a subset of
# every later Champions reg, so it is the safe fallback usage distribution).
FALLBACK_DEFAULT = "gen9championsvgc2026regmb"
_FALLBACK_PIKALYTICS = _DATA_DIR / "pikalytics_regma.json"

ENV_FORMAT_KEY = "VDANCE_BATTLE_FORMAT"


def _load_registry() -> List[dict]:
    try:
        return json.loads(_REGISTRY.read_text(encoding="utf-8")).get("formats", [])
    except Exception:
        return []


def known_formats() -> List[str]:
    """All format ids declared in the registry JSON (ordered as written)."""
    return [f["id"] for f in _load_registry() if f.get("id")]


def _active_from_registry() -> Optional[str]:
    for f in _load_registry():
        if f.get("active") and f.get("id"):
            return f["id"]
    return None


def default_format() -> str:
    """The active/default format (env > registry-active > hardcoded)."""
    return (os.environ.get(ENV_FORMAT_KEY)
            or _active_from_registry()
            or FALLBACK_DEFAULT)


# Module-level snapshot for cheap import-time consumers (run_local_battle, etc.).
# Computed once per process; mp workers recompute it from the inherited env var.
DEFAULT_FORMAT = default_format()


def is_champions_doubles(fmt: Optional[str]) -> bool:
    """Loose membership test for the Champions doubles family — the bot is meant
    to play 'all Pokémon Champions doubles formats', so accept any gen9 champions
    vgc id, not just the ones in the registry."""
    f = (fmt or "").lower()
    return f.startswith("gen9champions") and "vgc" in f


def reg_token(fmt: Optional[str]) -> Optional[str]:
    """Extract the reg token, tolerating Bo3/Blitz variant id suffixes:
    'gen9championsvgc2026regmb' -> 'regmb'; '...regmb-bo3' / '...regmbbo3' -> 'regmb'."""
    f = (fmt or "").lower()
    f = re.sub(r"-?(bo3|blitz)$", "", f)        # Showdown Bo3/Blitz variants share the reg
    m = re.search(r"(reg[a-z0-9]+)$", f)
    return m.group(1) if m else None


# Backwards-compat alias (older callers / tests used the underscore name).
_reg_token = reg_token


def pikalytics_filename(fmt: Optional[str] = None) -> str:
    """The canonical ``pikalytics_<reg>.json`` name — the SINGLE naming rule shared
    by the resolver AND the scraper, so a scrape can never write to a name the
    resolver won't look for (the corruption the audit flagged)."""
    reg = reg_token(fmt or DEFAULT_FORMAT)
    return f"pikalytics_{reg}.json" if reg else _FALLBACK_PIKALYTICS.name


def pikalytics_path_for(fmt: Optional[str] = None) -> Optional[Path]:
    """Resolve the Pikalytics belief file for ``fmt`` (defaults to the active
    format).

    Prefers ``data/pikalytics_<reg>.json``; falls back to the canonical M-A file
    (M-A ⊂ M-B) so the belief prior is never silently zeroed for a new reg.
    Returns None only if no Pikalytics file exists at all.
    """
    fmt = fmt or DEFAULT_FORMAT
    reg = reg_token(fmt)
    if reg:
        p = _DATA_DIR / pikalytics_filename(fmt)
        if p.exists():
            return p
    if _FALLBACK_PIKALYTICS.exists():
        return _FALLBACK_PIKALYTICS
    return None


def set_active_format(fmt: str) -> str:
    """Select ``fmt`` for THIS process AND its future mp children (sets the env
    var so spawned collect/eval workers inherit it). Updates the module snapshot.
    Returns the format set. Call this from an entry point's --format handling."""
    os.environ[ENV_FORMAT_KEY] = fmt
    global DEFAULT_FORMAT
    DEFAULT_FORMAT = fmt
    return fmt
