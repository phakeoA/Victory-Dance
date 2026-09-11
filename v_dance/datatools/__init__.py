"""Data preparation + team tools: bulk_parse_replays, ingest_hf_logs, corpus_qa, filter_type_c, observed_meta,
belief_blend, policy_analysis, team_archetypes, team_generator, validate_teams, release_check, seed_router_priors,
dossier_mega_backfill; ``scrapers/`` = the manual scrapers (scripts run BY PATH, not a package: ``scrape_pikalytics.py``,
``enum_champions_roster.js``, …; were ``data/scripts/scrapers/`` until refactor Phase 4, 2026-09-11). The three UI
servers moved to ``v_dance.ui`` (Phase 4); the old ``v_dance.datatools.{mission_control, dashboard_server, server}`` paths
are ``sys.modules`` shims until Phase 6."""
