# Victory-Dance ![kingambit](https://play.pokemonshowdown.com/sprites/xyani/kingambit.gif)

A VGC Pokémon battle-bot built with AlphaZero-style reinforcement learning, running on a self-hosted [Pokémon Showdown](https://pokemonshowdown.com/) server.

Victory-Dance plays **[Gen 9 Champions] VGC 2026 Reg M-A** doubles battles. Two neural networks handle separate responsibilities — one for in-battle action selection, one for teampreview lead selection — and train against each other through self-play.

To use team builder for vod training
python -m http.server 8000 for team_builder.html
## Python Version
Requires Python 3.11+

## Project Structure

```
Victory-Dance/
├── run_local_battle.py   # Battle harness — starts server, runs bots, opens spectator
├── player.py             # VGCPlayer — NN-driven player (falls back to random if no model)
├── random_player.py      # RandomVGCPlayer — fully random baseline
├── vgc_base.py           # Shared base class, replay buffer, action helpers
├── state_encoder.py      # Encodes DoubleBattle → float32 numpy array (STATE_DIM=882)
├── teams/M-A/team1       # Showdown paste team file
├── replay_buffer/        # Per-turn transitions with outcome back-fill (JSONL)
│   ├── TrainerRed.jsonl
│   └── TrainerBlue.jsonl
├── checkpoints/          # Saved model weights (.pt files go here)
├── pokemon-showdown/     # Local Showdown server (Node.js)
└── .venv/                # Python venv, includes venv-local node.exe
```

## Getting Started

### 1. Clone

```bash
git clone https://github.com/phakeoA/Victory-Dance.git
cd Victory-Dance
```

### 2. Install Python Requirements

```bash
pip install -r requirements.txt
```

### 3. Install the Local Showdown Server

```bash
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
npm install
cd ..
```

### 4. Run a Battle

```bash
python run_local_battle.py
```

The harness will automatically start the Showdown server, run one battle between TrainerRed and TrainerBlue, and open a spectator tab in your browser.

Common flags:

```bash
python run_local_battle.py -n 50            # run 50 battles
python run_local_battle.py --no-spectate    # skip opening browser (bulk runs)
python run_local_battle.py -v               # verbose / DEBUG logging
python run_local_battle.py --no-server      # skip server launch if already running
```

## Players

### RandomVGCPlayer
Selects all decisions uniformly at random — team, leads, and in-battle actions. Used as the initial self-play opponent and as a training baseline.

### VGCPlayer
Loads a trained PyTorch model from a checkpoint path and uses it for in-battle decisions. Falls back to random if no path is provided, making it a drop-in replacement for `RandomVGCPlayer` during early training.

To plug in a trained model, edit the two lines in `run_local_battle.py`:

```python
battle_model_path = Path("checkpoints/battle_model.pt")
team_chooser_path = Path("checkpoints/team_chooser.pt")
```

Leave either as `None` to keep using the random / heuristic fallback for that decision.

## Neural Networks

Two separate networks handle separate problems:

| Network | Input | Output | Trains on |
|---|---|---|---|
| Battle model | 882-float battle state | 2 actions (one per active slot, 0–15) | Per-turn transitions from replay buffer |
| Team-chooser | Teampreview snapshot | 4 ranked roster indices | Battle outcomes |

### Action Encoding

Each slot has 16 possible actions:

- `0–11` — move index (0–3) × target (opp slot 0, opp slot 1, ally)
- `12–15` — switch to bench slot (0–3)

### State Encoding

`STATE_DIM = 882` floats: 8 Pokémon slots × 101 features + 74 global features.

- 4 active slots (2 own + 2 opponent) + 4 bench slots
- Global features: weather, terrain, side conditions, turn count, trick room flag

## Replay Buffer

Every turn is recorded to `replay_buffer/<username>.jsonl`. At battle end, outcomes are back-filled into all turns from that battle.

```json
{
  "battle_id": "battle-gen9championsvgc2026regma-78",
  "turn": 3,
  "state": [0.0, ...],
  "action_s0": 4,
  "action_s1": 12,
  "source": "random",
  "outcome": 1
}
```

`outcome`: `1` = win, `0` = loss, `-1` = draw, `null` = battle still in progress.

## Requirements

- Python 3.11+
- Node.js (for the local Showdown server)
- PyTorch (optional — only needed when loading trained models)
- [poke-env](https://github.com/hsahovic/poke-env)
