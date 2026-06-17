"""v_dance — Victory-Dance VGC Pokemon Showdown bot.

The single importable package for the project. Subpackages:

- ``encoders``  state/live state encoding (state_encoder, live_state_encoder)
- ``parser``    VOD/replay parsing + belief state (vod_parser, belief_state)
- ``models``    network definitions (BC model, team-preview model, network)
- ``training``  training + dataset + eval scripts (train_bc, bc_dataset, ...)
- ``play``      live battle players (vgc_base, live_vgc_base, player, model_io)
- ``eval``      gauntlet + scripted opponents
- ``selfplay``  PPO self-play stack (the former local_battle/self_play)
- ``datatools`` scrapers, bulk export, corpus QA, team sheets, the flask server

NOTE: subpackages are populated incrementally during restructure Stage 2. The
package is intentionally named ``v_dance`` (not ``victory_dance``) because
"Victory Dance" is an actual in-game Pokemon move and would be confusing.
"""

__version__ = "0.1.0"
