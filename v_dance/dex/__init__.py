"""Shared dex utilities (refactor Phase 5, 2026-09-11; formerly ``v_dance.parser.vod_parser.{pokedex, team_sheet}``).

- ``pokedex``     the Pokedex (``data/pokedex.json``): ``get_pokedex``, ``norm_species``, mega / forme helpers — the most-imported
                  module in the repo (encoders, play, training, datatools, eval, the parser itself)
- ``team_sheet``  open-team-sheet parsing + the move / item / ability tables (``data/moves.json`` …)

Layering after this move: ``dex`` ← ``parser`` ← ``encoders`` ← … (the encoders ↔ parser cycle was these two files).
``v_dance.parser.vod_parser`` still re-exports the pokedex names it always did.
"""
