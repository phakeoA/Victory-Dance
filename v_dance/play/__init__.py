"""The serve / decision core + local harnesses: model_io, player, vgc_base, live_vgc_base, adapt_rules, search,
pokeenv_damage, random_player, ots_sheets, bo3_state, team_router, opponent_dossier, live_belief_feed, serve_bandit,
thought_feed, matchup_book, run_local_battle, play_vs_human, parallel_battles. The online bot + its ops moved to
``v_dance.online`` (refactor Phase 3, 2026-09-10) and the ladder recorder to ``v_dance.ladder`` (Phase 2); the old
paths here are ``sys.modules`` shims until Phase 6."""
