"""Data-grounded IDENTITY vocabularies (ability / move / item) for the v9 battle encoder (B1-mechanics).

Every ability/move/item the PINNED Showdown dex defines (incl. the Champions customs that live in the base
``data/`` files + the ``mods/champions`` overlay) is assigned a STABLE integer index, so AttnBCPolicy can
look each up in a learned ``nn.Embedding``. Index 0 = PAD / NONE / UNKNOWN (``padding_idx``).

Indices are DETERMINISTIC (sorted over the pinned dex, PINS.md) so encode-time, train-time and serve-time
agree by construction. ``tests/test_mechanic_vocab.py`` re-derives the vocab and asserts the key mappings +
full coverage, so a dex bump that would shift indices fails loud.

This module deliberately does NOT touch ``STATE_DIM`` / the live encoder layout — it is the substrate the v9
encode path + model will consume at the layout cutover. Ids are the Showdown block keys (lowercase
alphanumeric), the SAME id form poke-env returns for ``mon.ability`` / move / item, so lookups match live.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

_REPO = Path(__file__).resolve().parents[2]
_DEX = _REPO / "pokemon-showdown" / "data"

PAD = 0  # padding_idx / NONE / UNKNOWN — reserved, never a real entry


def _norm(s: Optional[str]) -> str:
    """The canonical Showdown id form (lowercase, alphanumeric only) — matches poke-env ids."""
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _dex_ids(*rel_paths: str) -> List[str]:
    """Sorted union of top-level entry ids across the given dex files. The block key (``\\tid: {``) IS the
    canonical id; missing files (e.g. a mod with no moves.ts) are skipped."""
    ids = set()
    for rel in rel_paths:
        p = _DEX.joinpath(*rel.split("/"))
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        for m in re.finditer(r"(?m)^\t(\w+): \{", text):
            ids.add(_norm(m.group(1)))
    ids.discard("")
    return sorted(ids)


def _build(ids: List[str]) -> Dict[str, int]:
    """id -> index in [1 .. len(ids)]; 0 is reserved for PAD/UNKNOWN."""
    return {i: n + 1 for n, i in enumerate(ids)}


# Base data files already carry the Champions customs (verified: eelevate/firemane/megasol/dragonize are in
# data/abilities.ts + data/moves.ts); the mods/champions overlay is unioned in as belt-and-suspenders.
ABILITY_IDS: List[str] = _dex_ids("abilities.ts", "mods/champions/abilities.ts")
MOVE_IDS: List[str] = _dex_ids("moves.ts", "mods/champions/moves.ts")
ITEM_IDS: List[str] = _dex_ids("items.ts", "mods/champions/items.ts")

ABILITY_VOCAB: Dict[str, int] = _build(ABILITY_IDS)
MOVE_VOCAB: Dict[str, int] = _build(MOVE_IDS)
ITEM_VOCAB: Dict[str, int] = _build(ITEM_IDS)

# nn.Embedding sizes (+1 for the reserved PAD row at index 0)
ABILITY_VOCAB_SIZE = len(ABILITY_VOCAB) + 1
MOVE_VOCAB_SIZE = len(MOVE_VOCAB) + 1
ITEM_VOCAB_SIZE = len(ITEM_VOCAB) + 1


def _index(vocab: Dict[str, int], raw: Optional[str]) -> int:
    return vocab.get(_norm(raw), PAD) if raw else PAD


def ability_index(raw: Optional[str]) -> int:
    """Embedding index for an ability id/name (PAD=0 for empty/unknown/non-dex)."""
    return _index(ABILITY_VOCAB, raw)


def move_index(raw: Optional[str]) -> int:
    return _index(MOVE_VOCAB, raw)


def item_index(raw: Optional[str]) -> int:
    return _index(ITEM_VOCAB, raw)


def missing_from_vocab(kind: str, ids) -> List[str]:
    """Which of ``ids`` (raw names/ids) do NOT resolve to a real vocab index — the build-guard hook.
    ``kind`` in {'ability','move','item'}. Empty list = full coverage."""
    vocab = {"ability": ABILITY_VOCAB, "move": MOVE_VOCAB, "item": ITEM_VOCAB}[kind]
    return sorted({r for r in ids if r and _norm(r) not in vocab})


def vocab_summary() -> dict:
    return {"ability": ABILITY_VOCAB_SIZE, "move": MOVE_VOCAB_SIZE, "item": ITEM_VOCAB_SIZE,
            "dex_dir_exists": _DEX.exists()}
