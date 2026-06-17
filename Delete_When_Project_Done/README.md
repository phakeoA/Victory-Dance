# Delete_When_Project_Done

Archived **one-off debug / probe / verification harnesses** from completed work — kept for reference
in case a similar issue recurs, safe to delete once the project ships. None are run by pytest (they
don't match `test_*.py`) and none are imported by the real code or test suite (verified before
archiving), so removing this folder cannot break anything.

Origin (provenance preserved in subfolders):
- `tests/` — probes/audits from the encoder state-rep gaps and codec work, all DONE:
  gap #4 (`_gap4_*`), gap #5 (`_gap5_*`), gap #6 (`_gap6_*`, `_gap_all_verify`), opponent
  reconstruction (`_opp_*`), policy (`_probe_policy`), gimmick/mega (`_verify_gimmick_*`,
  `_verify_mega_*`, `_diag_mega_snapshot`). Superseded by the `test_*.py` suite.
- `local_battle/` — live-play one-offs: `_diag_final_verify`, `_diag_reject_count`,
  `_test_illusion_targeting` (superseded by `tests/test_illusion_targeting.py`), `_verify_splice`,
  `_vs_random`.

- `foul_play_legacy/` — **dead foul-play fork leftovers** archived 2026-06-17 (Victory-Dance was forked
  from foul-play; these were never ported to the VGC pipeline and import the missing `fp.*` / `constants`
  modules, so they don't run here — kept as REFERENCE code, "you never know when they'll be useful").
  Verified unused by the live code + test suite before archiving. Provenance preserved:
  - `teams/{__init__,load_team,team_converter}.py` — foul-play's team-loading package (`load_team`,
    `TeamListIterator`, `export_to_packed`/`export_to_dict`). Superseded by `belief_state.parse_team_sheet`
    + path-based team loading in `run_local_battle.resolve_team_path`. (The `teams/M-A/*` paste FILES stay
    in `teams/`, read by path.)
  - `data/__init__.py` — foul-play's `data` package init: eager-loads `moves.json`/`pokedex.json` into
    `all_move_json`/`pokedex`/`effectiveness` globals (`from data import pokedex`). Replaced by
    `vod_parser.pokedex.get_pokedex()`.
  - `data/pkmn_sets.py` — foul-play set/stat builder (uses `fp.helpers.calculate_stats`, `fp.battle.Pokemon`).
  - `data/mods/{__init__,apply_mods.py}` — foul-play multi-gen mod applier (the `data/mods/gen*_mods.json`
    DATA files stay in `data/mods/`, still present though VGC is Gen 9 only).

## Reusable harnesses kept OUT of here (still live, for recurring issues) — paths post Stage-2
- `tests/_parity_harness.py` — offline↔live encoder byte-parity (imported by `tests/test_encoder_parity.py`).
- `scratch/_smoke_zoroark.py` — Zoroark/illusion smoke (illusion bugs recur).
- `scratch/_ab_headtohead.py` — checkpoint A/B head-to-head (for RL generation comparison).
- `scratch/_diag_rejections.py` — order-rejection diagnosis (if MODEL-DRIVEN% ever regresses).

## Kept-but-relocated foul-play scripts (the user wants these as usable tooling, NOT archived)
- `data/scripts/scrapers/{scrape_replays,scrape_pikalytics,scape_items,update_moves,update_pokedex,
  parse_random_battle_raw_sets}.py` — scrape/data-refresh scripts, fixed to run (fp→local `normalize_name`).
