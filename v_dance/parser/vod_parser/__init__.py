"""
vod_parser package
==================
Public API re-exports. server.py and other callers do:

    from v_dance.parser.vod_parser import parse_replay_for_preview, replay_to_transitions
"""

from v_dance.parser.vod_parser.battle_models import PokemonSlot, SideConditions, FieldConditions
from v_dance.parser.vod_parser.pokedex import (
    Pokedex,
    get_pokedex,
    norm_species,
    is_mega_species_name,
)
from v_dance.parser.vod_parser.replay_parser import (
    ShowdownReplayParser,
    extract_log_from_html,
    extract_replay_id_from_html,
)
from v_dance.parser.vod_parser.transitions import parse_replay_for_preview, replay_to_transitions

__all__ = [
    # models
    "PokemonSlot",
    "SideConditions",
    "FieldConditions",
    # pokedex
    "Pokedex",
    "get_pokedex",
    "norm_species",
    "is_mega_species_name",
    # parser
    "ShowdownReplayParser",
    "extract_log_from_html",
    "extract_replay_id_from_html",
    # transitions / server API
    "parse_replay_for_preview",
    "replay_to_transitions",
]
