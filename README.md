# Victory-Dance ![shiny Hisuian Zoroark](https://play.pokemonshowdown.com/sprites/ani-shiny/zoroark-hisui.gif)

**A self-play reinforcement-learning agent that plays competitive VGC doubles Pokémon** — trained by behavioral cloning on human replays, refined through league self-play, and augmented with an opponent-belief model and a belief-weighted search layer. It runs on a self-hosted [Pokémon Showdown](https://pokemonshowdown.com/) server via [poke-env](https://github.com/hsahovic/poke-env).

Format: **Gen 9 "Pokémon Champions" VGC 2026** doubles (Regulations M-A / M-B). Mega Evolution is in the format; Terastallization is disabled by the mod.

---

## Table of contents
- [Why this problem is hard](#why-this-problem-is-hard)
- [System overview](#system-overview)
- [1. Data & the parser](#1-data--the-parser)
- [2. The belief system](#2-the-belief-system)
- [3. The state encoder](#3-the-state-encoder)
- [4. The neural network](#4-the-neural-network)
- [5. Behavioral cloning](#5-behavioral-cloning-bc)
- [6. Self-play reinforcement learning](#6-self-play-reinforcement-learning)
- [7. Belief-weighted search](#7-belief-weighted-search)
- [8. Evaluation](#8-evaluation)
- [9. Serving & tooling](#9-serving--tooling)
- [Engineering practices](#engineering-practices)
- [Results & honest findings](#results--honest-findings)
- [Repository layout](#repository-layout)
- [Getting started](#getting-started)
- [Tech stack](#tech-stack)

---

## Why this problem is hard

VGC doubles is a deep test-bed for sequential decision-making under uncertainty:

- **Imperfect information.** You never see the opponent's items, abilities, EV spreads, natures, or their full moveset — you infer them from usage statistics and in-game evidence.
- **A large, structured action space.** Each turn you choose *two* simultaneous actions (one per active Pokémon), each of which can be a move against one of several targets or a switch — and the two choices interact (spread moves, redirection, protect mind-games).
- **Simultaneous, stochastic resolution.** Both sides commit hidden actions; speed ties, accuracy, damage rolls, and secondary effects are random.
- **Team-level strategy.** Which 4 of 6 Pokémon you bring, and your leads, are decided before the battle from imperfect knowledge of the opponent's six.

The project attacks this with a layered stack: learn a strong prior from humans, sharpen it with self-play, model the hidden opponent explicitly, and (optionally) look ahead with a forward model.

---

## System overview

```
 Human replays          Parser + Belief            Encoder              Behavioral Cloning
 (Showdown logs)  ──▶  per-turn battle states  ──▶  5057-dim state  ──▶  attention policy + value net
                        + usage-prior beliefs                                     │
                                                                                  │  warm-start + KL anchor
                                                                                  ▼
                                                                    Self-play PPO in a league
                                                                    (promotion gate + Hall-of-Fame)
                                                                                  │
   live play  ◀───  Belief-weighted search  ◀───  in-game belief narrowing  ◀────┘
              (depth-1 expectimax over a          (MatchBelief updates the
               white-box forward model)            opponent's hidden set live)
```

Everything downstream of the encoder reads a **frozen, versioned feature layout** (currently `STATE_LAYOUT_VERSION = 19`, `STATE_DIM = 5057`). Freezing the layout means the encoder, the trained checkpoints, and the forward model all agree byte-for-byte, and layout changes are caught by explicit guards rather than silently corrupting a trained net.

---

## 1. Data & the parser

**`v_dance/parser/`** turns raw Pokémon Showdown replay logs into structured, per-turn training transitions.

- **`replay_parser.py` + `vod_parser/transitions.py`** replay the Showdown protocol line-by-line, maintaining a full battle model (`battle_models.py`): both teams, HP, status, boosts, field conditions, revealed moves/items/abilities, and what each side *chose* each turn. Each decision point becomes a training transition `(state_before, action_taken, outcome)`.
- It handles the genuinely hard cases competitive replays throw at a parser: **Illusion/Zoroark** identity spoofing, **Ditto** transform, doubles targeting and redirection, spread-move damage attribution, item/ability reveals *by their effect* (Life Orb recoil, Rocky Helmet chip, Leftovers heal, status orbs), and silent item transfers (Sticky Barb, Symbiosis).
- **Replay "VOD types"** distinguish provenance: **A** = the user's own games (our side is exact from a team sheet, opponent from usage priors), **B** = ranked ladder replays (both sides from priors), **C** = the bot's own live battles (used to score opponent-prediction accuracy), **D** = self-play (both sides exact).

The corpus is exported to per-replay JSONL. Crucially, **each transition stores a structured *snapshot dict*, not a pre-computed vector** — the trainer re-encodes snapshots at train time, so an encoder/layout change requires only a retrain, while a parser or belief change requires a re-export. `datatools/bulk_parse_replays.py` batch-exports and `datatools/corpus_qa.py` audits the result (legality-under-mask, duplicate detection, decision-type coverage).

---

## 2. The belief system

Because most of the opponent is hidden, the agent maintains an explicit **belief** over the opponent's Pokémon.

- **`belief_state.py`** seeds priors from **Pikalytics usage statistics** (scraped per-format): for each species, the distribution over moves, items, abilities, and EV/nature spreads. `fill_blanks` turns these into concrete *estimates* of hidden stats, so a partially-seen opponent still encodes as a plausible full set. Careful data-quality handling: Mega formes inherit their base species' data, inert Mega stones are demoted to the real held item, and offence-invested spreads are matched to offence-boosting natures.
- **`match_belief.py` (`MatchBelief`)** narrows that prior *during the battle* from live evidence: revealed moves constrain the moveset; observed damage (dealt or taken) back-solves plausible stat spreads via a likelihood model; a revealed Choice item forces a consistent spread; a paradox Pokémon's boosted stat reveals its EV investment or Booster Energy. Live observations are fed in by **`play/live_belief_feed.py`**.

This gives the network (and the search) a continuously-sharpening estimate of what it's up against.

---

## 3. The state encoder

**`v_dance/encoders/`** maps a battle snapshot to a fixed **5057-float** vector. Two twin encoders — `state_encoder.py` (offline, from parsed snapshots) and `live_state_encoder.py` (online, from poke-env) — are held to **byte-level parity** by tests, so the model sees identical inputs in training and in live play.

The design choices are what make it robust:

- **No identity one-hots.** Species are encoded as their **types + base stats**, and held items and abilities as a **multi-hot over strategic *effect categories*** (`choice`, `weather_setter`, `focus-sash`-style survival, contact-punish, etc.), not as an opaque ID over the current meta's item list. A brand-new item in a future regulation is still `choice` — so **the layout never changes when the roster does**, only the id→category tables grow. This is the same generalization philosophy applied throughout.
- **Exhaustive mechanic computation.** The encoder *computes* every Champions mechanic rather than merely tagging it — weather/terrain effects, ability and item damage modifiers, spread-move reduction, Trick Room, speed control, Intimidate, paradox boosts, and so on (`battle_mechanics.py`, `damage_mechanics.py`, `mechanic_tags.py`).
- **Learned identity embeddings where identity matters.** For the specific moves/ability/item a Pokémon is *known* to have, the encoder writes vocabulary **indices** into the row and the network looks them up as learned embeddings (see below) — combining the format-stable category features with sharp identity signal when it's actually observed.
- **Frozen orderings.** Type, status, weather, field, and side-condition orderings are pinned in `encoder_layout.py` (mapped onto poke-env's enums by *name*), so a poke-env upgrade can never silently shift an index.

The 5057 dims decompose as **12 Pokémon slots × 413 features + global fields**. The 12 slots are semantically ordered: `own_a, own_b, opp_a, opp_b` (the four actives), then the own bench (4) and opponent bench (4).

**Action space (`action_codec.py`).** Each active slot has **16** actions — `move (0-3) × target (opp-a / opp-b / ally)` for the 12 move-target combinations, plus `switch to bench slot (0-3)` — with a separate **3-way gimmick** decision (`none / mega / tera`; tera is layout-reserved but mod-disabled). Legality masks are computed per slot so the policy only ever chooses a legal action, and the same `move_slots_for_mon` logic drives both the encoded move features and the action indices so they always refer to the same move.

---

## 4. The neural network

**`models/bc_model_attn.py` — `AttnBCPolicy`**, a per-mon **set-attention** architecture (the production battle net, ~**2.3M parameters** at the deployed 256-wide / 4-layer / 8-head config).

The key idea: rather than feed the flat 5057-vector into one big linear layer (which would learn *separate* weights for "the mon in bench slot 3" vs "bench slot 4" and re-learn "these twelve things are all Pokémon" from scratch), the model **reshapes the flat vector back into 12 Pokémon tokens in-model** and processes them with shared, structured weights:

1. **Shared per-mon encoder** — one small MLP applied identically to all 12 Pokémon tokens, so "how to read a Pokémon" is learned once.
2. **Learned identity embeddings** — ability (24-d), item (16-d), and per-move (16-d) embeddings looked up from the indices the encoder wrote, then concatenated onto the token. Identity becomes a learned vector, never a raw ordinal.
3. **Learned per-slot positional embeddings** — tell the model which side/role each of the 12 tokens is.
4. **A Transformer self-attention stack** — the 12 Pokémon **attend to one another**, so the network can reason about synergy and threat assessment (my sweeper vs their walls) directly. Absent Pokémon are masked out (`key_padding_mask`).
5. **A global (field) encoder** for weather/terrain/side-conditions/Trick Room.

Read off this shared representation are several **heads**:

| Head | Reads | Predicts |
|---|---|---|
| `our_a`, `our_b` | own active tokens + global | this slot's action (16 logits) |
| `opp_a`, `opp_b` (auxiliary) | opponent active tokens + global | the **opponent's** action — an auxiliary opponent-modeling task |
| gimmick heads | own active tokens + global | mega/tera decision (3 logits) |
| value head | masked-mean over present tokens + global | win probability (a single win-logit) |

Two research extensions are built in behind flags: **opponent-conditioning** (the "our" heads can read the detached softmax of the opponent-action prediction, to best-respond to their own read) and a **C51 distributional value head** (a per-atom value distribution instead of a scalar). The value readout is itself configurable (masked-mean / concat-active / learned-CLS-query).

---

## 5. Behavioral cloning (BC)

**`training/train_bc.py`** pretrains `AttnBCPolicy` on the human corpus. Each transition is re-encoded at train time (`bc_dataset.py`), and the loss is a sum of:

- **per-slot action cross-entropy** (masked to legal actions) for `our_a` / `our_b`,
- **value BCE** against the game outcome (the win/loss label),
- **gimmick cross-entropy** (when a mega decision was available),
- **auxiliary opponent-action cross-entropy** for the `opp_a` / `opp_b` heads.

Training reports top-1 / top-3 action accuracy, win-accuracy + Brier score for the value head, and opponent-prediction accuracy. The current production anchor reaches **val top-1 ≈ 0.585** on held-out human games. This checkpoint is the warm-start (and KL anchor) for self-play.

---

## 6. Self-play reinforcement learning

**`selfplay/generation.py`** runs a PPO-style league self-play loop starting from the BC anchor.

- **Reward** is deliberately minimal — terminal win/loss = ±1, discounted (γ ≈ 0.997), with a **value-head critic** (GAE advantages) and a **KL-to-BC anchor** that keeps the policy from drifting into degenerate play. Potential-based reward shaping exists but is gated off by default. (`reward.py`, `gae.py`, `actor_critic.py`.)
- **A promotion gate** (`gate.py`) only anoints a new generation when it beats the standing champion over a large **mirror-match** sample, judged by a two-proportion / Wilson-CI significance test — not a noisy single number.
- **A Hall-of-Fame + PFSP league** (`hof.py`, `league.py`) makes the candidate play a curated set of past champions (prioritized by how competitive they are), which prevents strategy-cycling and rock-paper-scissors regressions.
- **Multiprocess collection** (`mp_collect.py`) is the throughput engine: a persistent spawn `ProcessPoolExecutor` (`CollectionPool`) runs battles across CPU cores **GIL-free** (each worker its own interpreter + event loop), with crash recovery, while a pool of local Showdown servers (`ServerPool`, `--servers N`) spreads the connection load. The whole harness is built around an injected player-factory so it is offline-testable without spawning Node.

---

## 7. Belief-weighted search

**`play/search.py`** (Level C) adds an optional look-ahead layer — a **depth-1, belief-weighted expectimax** over a **white-box forward model**:

- **`encoders/white_box_sim.py`** is an analytic, poke-env-free forward model that applies one turn (damage, KOs, switches, entry abilities, item effects, weather/terrain, spread reduction) — its damage and mechanic logic mirrors the encoder's twins so a simulated successor state is consistent with what the value head was trained on.
- For each of the agent's top-K joint actions, the search **averages the value head's win-probability of the resulting state over the opponent's top-M likely actions and top-S stat-belief scenarios**, then picks the best. Because it hedges over *probabilities* (opponent action prior × stat beliefs) rather than committing to a single point-estimate KO, it makes calmer, better-calibrated decisions in sharp positions.
- The core `expectimax(candidates, value_fn)` is a pure, unit-tested function; `search_with_model` wires in the network heads and the belief-enriched snapshot. Search ships **behind a default-off flag** so production behavior is unchanged until it clears its own A/B evaluation.

---

## 8. Evaluation

**`eval/gauntlet.py`** measures strength honestly:

- battles vs a set of **scripted opponents** (`eval_opponents.py`, e.g. a max-damage heuristic) and vs **past-champion snapshots**,
- a reproducible, side-balanced **team-matchup schedule** (`team_matchups`) so a checkpoint plays both sides of every pairing (matchup bias cancels),
- an **Elo** estimate that excludes the self-mirror.

Strength experiments are run as controlled **A/B tests** (belief-on vs belief-off, search-on vs argmax) on single-team mirrors, with the effect reported as a Wilson confidence interval and validated for wiring (did the feature actually fire?) before the number is trusted.

---

## 9. Serving & tooling

- **`play/play_vs_human_browser.py`** — challenge the bot from a browser against the local server.
- **`play/run_local_battle.py`** — start the Showdown server(s), resolve teams, run a battle; also home to `ServerPool` and team discovery.
- **`datatools/dashboard_server.py`** — a live Flask dashboard streaming Elo, win-rates, and spectatable replays during a self-play run.
- **`datatools/policy_analysis.py`** — offline inspection of what the policy does.

---

## Engineering practices

This is a research codebase, but built like production:

- **~1,100+ tests** (`pytest tests`), including encoder byte-parity tests that pin train/serve equivalence, action-legality invariants, and corpus-QA gates.
- **A frozen, versioned tensor layout** with load-time guards (`play/model_io.py` rejects a checkpoint whose layout version or dimensions don't match the current encoder) — so a layout change fails loudly instead of silently degrading a trained net.
- **Adversarial, multi-agent code audits** — the forward model, belief narrowers, and self-play accounting were each hardened by "find → independently verify" audit passes; findings are fixed with regression tests and kept behind flags so production stays byte-identical.
- **A pinned environment** (`PINS.md`: exact poke-env and Showdown commits) for reproducibility, and one installable package (`pip install -e .`, absolute imports, no `sys.path` hacks).
- **Scaling infrastructure** documented in `docs/` (multi-server sharding, a reusable multiprocess battle harness).

---

## Results & honest findings

- **Behavioral cloning** reaches **val top-1 ≈ 0.585** predicting human actions — a strong prior for a 16-way, doubly-simultaneous decision.
- A recurring, scientifically-interesting result: **the served network is corpus-limited, not capacity-limited.** Enlarging the network, adding a distributional (C51) value head, and adding opponent-conditioning all **washed out** in head-to-head strength; only *more human data* moved the needle. Several belief experiments produced **"informative nulls"** — the agent forms measurably sharper opponent estimates, but the reactive policy doesn't yet convert them into extra wins. These negative results are reported as carefully as the positive ones; the search layer above is a direct response to that reactivity gap.

*(The self-play and search layers are active research; the production served net is the behavior-cloned attention policy.)*

---

## Repository layout

```
Victory-Dance/
├── v_dance/                # installable package (pip install -e .)
│   ├── parser/             # Showdown replay logs -> per-turn transitions; belief_state, match_belief
│   ├── encoders/           # battle state -> float32 vector (STATE_DIM = 5057, layout v19); white_box_sim
│   ├── models/             # AttnBCPolicy set-attention battle net + team-preview net
│   ├── training/           # behavior-cloning training loop, datasets, feature extraction
│   ├── selfplay/           # PPO self-play, league, promotion gate, multiprocess collection (mp_collect)
│   ├── play/               # live players, action construction, ServerPool, search, model_io, dashboards
│   ├── eval/               # gauntlet evaluation vs scripted + snapshot opponents, Elo
│   └── datatools/          # corpus QA, bulk export, live dashboard, analysis
├── data/                   # pokedex/moves data, Pikalytics usage, teams, regulations
├── data/scripts/scrapers/  # standalone replay / Pikalytics / dex refresh tooling
├── docs/                   # design docs (search, multi-server, reusable multiprocess harness)
├── tests/                  # pytest suite (byte-parity, legality, corpus QA, audits)
├── ai_train_scripts/       # checkpoints (BC model, team-preview model)
├── artifacts/              # runtime outputs: self-play archive, replays, logs (gitignored)
└── pokemon-showdown/       # local Showdown server (Node.js), cloned separately
```

---

## Getting started

```bash
# 1. clone + install the package (Python 3.11+; a venv is recommended)
git clone https://github.com/<you>/Victory-Dance.git
cd Victory-Dance
pip install -e .          # or: pip install -r requirements.txt

# 2. local Showdown server (Node.js)
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown && npm install && cd ..
```

```bash
# Run a single self-play battle (starts the server, opens a spectator tab)
python -m v_dance.play.run_local_battle

# Train the behavior-cloning anchor on the parsed corpus
python -m v_dance.training.train_bc --data <jsonl folders> \
    --epochs 30 --d-model 256 --n-heads 8 --n-layers 4 --aux-opp-head \
    --out ai_train_scripts/BC_model/checkpoints_attn

# A self-play training run: 6 collection processes across 2 Showdown servers, resume latest
python -m v_dance.selfplay.generation --live --servers 2 --collect-procs 6 --resume-gen latest

# A strength A/B (search-on vs policy-argmax, single-team mirror) with a Wilson-CI verdict
python -m v_dance.selfplay.game_runner --search ab -n 2000 --workers 40 --servers 2 \
    --teams team1 --ckpt ai_train_scripts/BC_model/checkpoints_attn_pre_gen141/battle_base.pt \
    --report-json artifacts/b1_ab.json

# The live training dashboard
python -m v_dance.datatools.dashboard_server --port 5175 --archive artifacts/self_play_archive

# Run the test suite
pytest tests -q
```

---

## Tech stack

**Python** (PyTorch, NumPy) · **[poke-env](https://github.com/hsahovic/poke-env)** for the Showdown protocol · **Node.js** Pokémon Showdown as the battle engine · **Flask** for the dashboard · `concurrent.futures` spawn multiprocessing for GIL-free collection · **pytest** for the test suite.

---

*Personal research project. Not affiliated with Nintendo / Game Freak / The Pokémon Company. Pokémon Showdown is © its respective authors.*
