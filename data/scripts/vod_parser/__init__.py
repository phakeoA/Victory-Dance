"""
vod_parser/__init__.py
======================
Makes data/scripts/vod_parser/ a proper Python package.

Re-exports the public API so callers can do:

    from vod_parser import parse_replay_for_preview, replay_to_transitions

instead of reaching into the sub-modules directly.
"""

from __future__ import annotations

from vod_parser.battle_models import FieldConditions, PokemonSlot, SideConditions
from vod_parser.replay_parser import (
    ShowdownReplayParser,
    extract_log_from_html,
    extract_replay_id_from_html,
)
from vod_parser.transitions import parse_replay_for_preview, replay_to_transitions

__all__ = [
    # models
    "PokemonSlot",
    "SideConditions",
    "FieldConditions",
    # parser
    "ShowdownReplayParser",
    "extract_log_from_html",
    "extract_replay_id_from_html",
    # transitions / server API
    "parse_replay_for_preview",
    "replay_to_transitions",
]
