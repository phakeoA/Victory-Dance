# Tests — Victory-Dance VOD parser & team builder

## Running everything

From `data/scripts/`:

```bash
# Python suite (parser, pokedex, transitions, models, Flask endpoints)
python -m pytest tests/ -v

# JS suite (client-side parser mirror + export pipeline)
cd team_builder
node --test tests/test_tb_parser.mjs tests/test_tb_export.mjs
```

(Node 22's `--test <dir>` directory mode is finicky — pass the files
explicitly as above, or use `node --test "tests/*.mjs"`.)

Requirements: `pip install flask flask-cors pytest --break-system-packages`,
Node ≥ 18 (uses the built-in `node:test` runner, no npm packages).

## What the suites cover (Bug 8: mega ability split)

The structural flaw fixed: a Pokémon that mega evolves has an ability
**before** the mega (one of its base forme's options, the player's choice)
and an ability **after** the mega — which is always **exactly one**, fixed by
the mega forme. The old code stored a single `known_ability` and conflated
the two: a pre-mega Intimidate reveal stayed "current" after the mon mega'd,
and the deterministic mega ability was never resolved.

The fix threads `known_ability` (currently active) / `pre_mega_ability` /
`mega_ability` through every layer, with the mega ability resolved from
`data/pokedex.json` the instant `|detailschange|` fires:

| Layer | File(s) | Tested by |
|---|---|---|
| Dex lookups (`mega_ability_for`, `mega_formes_for`, …) | `vod_parser/pokedex.py` | `tests/test_pokedex.py` |
| Dataclass fields + serialisation | `vod_parser/battle_models.py` | `tests/test_battle_models.py` |
| Protocol handling: ability demotion on mega, `\|-ability\|` routing pre/post mega, non-mega forme changes (Palafin-Hero), `\|-mega\|` safety net, mega switch-back continuity | `vod_parser/replay_parser.py` | `tests/test_replay_parser.py` (synthetic logs + real example VOD) |
| Mega-aware ability injection into training transitions | `vod_parser/transitions.py` | `tests/test_transitions.py` |
| HTTP endpoints incl. end-to-end JSONL export | `server.py` | `tests/test_server.py` |
| Client-side parser mirror (incl. **offline** degradation when the pokedex isn't loaded) + dex helpers | `team_builder/tb_constants.js`, `tb_parser.js` | `team_builder/tests/test_tb_parser.mjs` |
| Export pipeline: mega mon never exports the injected base ability as active | `team_builder/tb_api.js`, `tb_actions.js` | `team_builder/tests/test_tb_export.mjs` |

The invariant asserted everywhere: **after a mega, the active ability is the
mega forme's fixed ability (or unknown), never the base forme's** — and the
base ability is preserved as `pre_mega_ability` instead of being lost.
