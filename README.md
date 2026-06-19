# Victory-Dance ![shiny Hisuian Zoroark](https://play.pokemonshowdown.com/sprites/ani-shiny/zoroark-hisui.gif)

A VGC Pokémon battle bot trained with **behavior cloning + PPO self-play**, running on a self-hosted [Pokémon Showdown](https://pokemonshowdown.com/) server.

Victory-Dance plays **[Gen 9 Champions] VGC 2026 Reg M-A** doubles. Two separate neural networks split the problem — one selects in-battle actions, one chooses which 4 of 6 Pokémon to bring at team preview — and they improve through league self-play.

## How it learns

1. **Behavior cloning (BC).** A policy + value network is pretrained on parsed human VGC replays. The `v_dance/parser` pipeline turns Showdown replay logs into per-turn training transitions, with a belief state that infers hidden opponent sets from usage priors.
2. **PPO self-play.** The BC policy is fine-tuned by playing a league of its own past snapshots. Reward is the terminal result only (win/loss = ±1); a value-head critic and a KL-to-BC anchor keep training stable. A generation is promoted only when it beats the standing champion over a large mirror sample — a frozen-champion Elo ladder with a Hall-of-Fame anti-cycling gate.
3. **Team preview.** A separate, permutation-equivariant network chooses the bring + leads.

Collection and evaluation run multi-process across CPU cores against one or more local Showdown servers (`--servers N`), and a live dashboard streams Elo, win-rates, and replays.

## Requirements

- Python 3.11+
- Node.js (used for the local Showdown server; a venv-local `node.exe` is included)
- PyTorch
- [poke-env](https://github.com/hsahovic/poke-env)

## Getting started

```bash
# 1. clone + install the package
git clone https://github.com/phakeoA/Victory-Dance.git
cd Victory-Dance
pip install -e .

# 2. local Showdown server
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown && npm install && cd ..
```

### Run

```bash
# one self-play battle (starts the server, opens a spectator tab)
python -m v_dance.play.run_local_battle

# a self-play training run, resuming the latest snapshot
python -m v_dance.selfplay.generation --live --servers 2 --collect-procs 6 --resume-gen latest

# the live training dashboard
python -m v_dance.datatools.dashboard_server --port 5175 --archive artifacts/self_play_archive
```

## Project structure

```
Victory-Dance/
├── v_dance/                # installable package (pip install -e .)
│   ├── parser/             # Showdown replay logs -> training transitions + belief state
│   ├── encoders/           # battle state -> float32 vector (STATE_DIM = 1866, layout v4)
│   ├── models/             # battle policy/value net + team-preview net
│   ├── training/           # behavior-cloning training, datasets, feature extraction
│   ├── selfplay/           # PPO self-play loop, league, promotion gate, multiprocess collection/eval
│   ├── play/               # live players, action/order construction, local battle harness
│   ├── eval/               # gauntlet evaluation vs scripted + snapshot opponents
│   └── datatools/          # live dashboard + data utilities
├── teams/Champions/M-A/    # Showdown paste team files (the training / eval pool)
├── ai_train_scripts/       # checkpoints (BC model, team-preview model)
├── artifacts/              # runtime outputs: self-play archive, replays, logs (gitignored)
└── pokemon-showdown/       # local Showdown server (Node.js), cloned separately
```

## The two networks

| Network | Input | Output | Trains on |
|---|---|---|---|
| Battle policy / value | 1866-float battle state | per-slot action (0–15) + gimmick + value | BC on human replays, then PPO self-play |
| Team preview | both teams' 6 rosters | the 4 brought + 2 leads | battle outcomes |

### Action encoding

Each active slot has **16** actions, plus a per-slot 2-way gimmick (mega-evolution) decision:

- `0–11` — move (0–3) × target (opponent slot 0 / opponent slot 1 / ally)
- `12–15` — switch to bench slot (0–3)

### State encoding

`STATE_DIM = 1866` floats (layout v4): the 6 own + 6 opponent Pokémon (species, types, stats, HP, status, stat boosts, and belief-state estimates of hidden moves / item / ability) plus global fields — weather, terrain, side conditions, the turn clock, and Trick Room.

---

*Personal research project. Not affiliated with Nintendo / The Pokémon Company.*
