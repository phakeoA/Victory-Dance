# Victory-Dance ![shiny Hisuian Zoroark](https://play.pokemonshowdown.com/sprites/ani-shiny/zoroark-hisui.gif)

[![Tests](https://github.com/phakeoA/Victory-Dance/actions/workflows/tests.yml/badge.svg)](https://github.com/phakeoA/Victory-Dance/actions/workflows/tests.yml)

**A deep-learning agent that plays competitive VGC doubles Pokémon on [Pokémon Showdown](https://pokemonshowdown.com/)** — trained by behavior cloning on ~86,000 human battles, improved with offline advantage-weighting and a contrastive team-preview set head, and served through a full deployment stack (local, browser-transport, and online) with a recorded human-benchmark protocol. In its first recorded benchmark set it beat its own creator 4–1; on the official ladder its elo-adjusted performance rating climbed from **1212 → 1271 → 1425** across three measured 50-game deployment blocks (peak rating 1499).

Format: **Gen 9 "Pokémon Champions" VGC 2026** doubles (Regulations M-A / M-B). Mega Evolution is legal in this format; Terastallization is disabled by the mod.

> This is an educational research project, but the README describes the **shipped agent** — the model, the pipeline that trains it, and the stack that deploys and measures it. The deliverable is a team-agnostic VGC-doubles player you can point at any legal team, run locally or on the live ladder from a single control UI, and measure against real humans with a recorded exploitability protocol.

---

## Table of contents
- [Why this problem is hard](#why-this-problem-is-hard)
- [System overview](#system-overview)
- [Mission Control — one UI for the whole project](#mission-control--one-ui-for-the-whole-project)
- [1. Data & the parser](#1-data--the-parser)
- [2. The belief system](#2-the-belief-system)
- [3. The state encoder](#3-the-state-encoder)
- [4. The neural network](#4-the-neural-network)
- [5. Training: BC, streaming at 32 GB, advantage weighting](#5-training)
- [6. Evaluation: rulers, A/Bs, and the human benchmark](#6-evaluation)
- [7. Serving & deployment](#7-serving--deployment)
- [8. The adaptation layer](#8-the-adaptation-layer)
- [9. How a single turn actually works](#9-how-a-single-turn-actually-works)
- [Results](#results)
- [Acknowledgements & prior art](#acknowledgements--prior-art)
- [Repository layout](#repository-layout)
- [Building & specializing teams](#building--specializing-teams)
- [Setup: playing locally](#setup-playing-locally)
- [Setup: playing online](#setup-playing-online)
- [Tech stack](#tech-stack)
- [Appendix: research history (for technical readers)](#appendix-research-history-for-technical-readers)

---

## Why this problem is hard

VGC doubles is a deep test-bed for sequential decision-making under uncertainty:

- **Imperfect information.** You never see the opponent's items, abilities, EV spreads, or full movesets — you infer them from usage statistics and in-game evidence.
- **A large, structured action space.** Each turn you pick *two* simultaneous actions (one per active Pokémon) — move × target or switch — and the choices interact (spread moves, redirection, Protect mind-games).
- **Simultaneous, stochastic resolution.** Both sides commit hidden actions; speed ties, accuracy, damage rolls, and secondary effects are random.
- **Team-level strategy.** Which 4 of 6 you bring, and your leads, are chosen before the battle from imperfect knowledge of the opponent's six.
- **A human in the loop.** A human opponent *adapts*: any fixed policy has habits, and habits get exploited. Measuring and countering that is a first-class goal here, not an afterthought.

---

## System overview

```
 Human replays (Showdown logs, ~86k battles)      Pikalytics usage priors
        │                                                │
        ▼                                                ▼
   Parser (protocol replay, Illusion/Ditto-safe)  →  Belief system (priors + in-game narrowing)
        │
        ▼
   Encoder — 5057-dim mechanic-based state (NO species one-hots; frozen layout v19)
        │
        ▼
   AttnBCPolicy — per-mon set-attention net (~2.3M params)
   heads: our-action ×2 · opp-action ×2 (aux) · mega/tera · win-prob value
        │
        ├── Behavior cloning (streaming memmap loader → full corpus on 32 GB RAM)
        └── Offline advantage weighting  w = exp(β·(outcome − V(s)))   ["imitate what won"]
        │
        ▼
   Serving: local harness · browser transport (BattleHost) · online (play.pokemonshowdown.com)
   + SBDA team preview w/ contrastive set-scoring head (bring-4 + leads as a UNIT)
   + adaptation layer (pattern tilt · dossiers · dossier belief warm-start ·
     archetype team routing · Bo3 set context)
   + human-benchmark recording (win rates, ratings, exploitability curve, replays)
   + RL exploiter — worst-case robustness meter (frozen target; never trains the bot)
```

Everything downstream of the encoder reads a **frozen, versioned feature layout** (`STATE_LAYOUT_VERSION = 19`, `STATE_DIM = 5057`) with load-time guards, so a layout change fails loudly instead of silently corrupting a trained net.

---

## Mission Control — one UI for the whole project

Every part of the project — data prep, belief scraping/blending, training the battle and team-preview nets, evaluation, deployment, and both local and online play — is driven from a single local page:

```bash
python -m v_dance.datatools.mission_control      # opens http://127.0.0.1:8990/
```

It's a dependency-light, **torch-free** stdlib server (starts instantly, safe to run alongside anything) that knows every entry point in the project through a typed command registry. Tabs:

- **Overview** — the pipeline map, a live **deploy-parity check** (`.env` ↔ `model_io` defaults, shown ✓/✗), belief freshness, the exploiter curve, and which services are up.
- **Online bot** — the home for online play. Before it's running you set the **launch config** right here — **format**, team pin, and the anti-exploit `--adapt-rules` + opponent-`--dossier` toggles (both **on by default**) — and launch from this tab; picking a format writes it to `.env` (the stack binds it at launch). Once it's live the same tab drives it: start a **ladder run of N rated games**, toggle **auto-accept challenges** and auto-close, send **private challenges**, pin a team, and watch the rating / W–L tally / activity log. (The live controls are proxied from the bot's own control panel, so you never open a second window.)
- **Play** — local AI-vs-AI, vs-human (terminal or browser), and the online launcher. The local harnesses each have team/checkpoint/format pickers and a **"Copy command"** one-liner; the online bot's format and launch config live in its own **Online bot** tab.
- **Train** — the battle-net era retrain, both team-preview trainers, the exploiter, and self-play. Heavy runs launch **one at a time** (GPU/RAM guard) and stream **live progress** — epoch N/total with a progress bar and metric chips (val top-1, set-exact, bring-exact) for trainers; games-trained + exploit win-rate for the exploiter. Prefer a terminal? Every card has a **"Copy command"** button with the exact one-liner.
- **Parser / Teams** — the combined VOD replay parser/annotator (upload a replay → export JSONL training transitions) and team builder (generate/validate/score teams), embedded.
- **Data / Eval** — replay parsing, belief blend, corpus QA, ingest; the pytest suite, rulers, the human-benchmark report, gauntlets.
- **Deploy** — edit the `.env` deploy keys (checkpoint/team/format) from dropdowns of what's actually on disk, with the parity check.
- **Monitor** — the self-play dashboard (live generations, launcher, console), embedded.
- **Jobs** — every launched job with a live log tail and a stop button.

Heavy runs are launchable but **you** click the button; the server never auto-starts training. Credentials never leave the server (a strict env whitelist — `PS_PASSWORD` is never sent to the page).

---

## 1. Data & the parser

**`v_dance/parser/`** turns raw Showdown replay logs into structured per-turn training transitions, handling the genuinely hard cases: **Illusion/Zoroark** identity spoofing, **Ditto** Transform, doubles targeting and redirection, spread-damage attribution, item/ability reveals *by their effect* (Life Orb recoil, Rocky Helmet chip, status orbs), and silent item transfers.

The corpus grew in two stages, and the second is owed to the VGC-Bench team (see credits):

| Stage | Source | Battles |
|---|---|---|
| Own scraping + parsing | ladder replays, own games, live games | ~6,600 |
| **VGC-Bench open dataset** (Champions-format subset, re-parsed through our pipeline with open-team-sheet support) | `cameronangliss/vgc-battle-logs` | **~80,000 rated** |

Total: **≈86k battles → ~1.4M training decisions**, quality-audited by `datatools/corpus_qa.py` (0 illegal-under-mask, 0 duplicates). Transitions store *snapshot dicts*, not vectors — an encoder change needs only a retrain, never a re-export.

## 2. The belief system

The agent maintains an explicit **belief** over the opponent's hidden sets: priors seeded from **Pikalytics** usage statistics (`belief_state.py`), narrowed live by in-game evidence (`match_belief.py`) — revealed moves, damage-based stat back-solving, Choice-item consistency, paradox-boost deduction. The parser and the live player share this machinery, so training and serving see the same kind of opponent estimate.

## 3. The state encoder

**`v_dance/encoders/`** maps a battle snapshot to a fixed **5057-float** vector; twin offline/live encoders are held to **byte-level parity** by tests, so the model sees identical inputs in training and play.

- **No identity one-hots.** Species encode as **types + base stats**; items/abilities as multi-hot **strategic effect categories**. A brand-new Pokémon or item in a future regulation slots into the same layout — the design bet that lets one net generalize across rosters and pilot user-supplied teams.
- **Exhaustive mechanic computation** — weather/terrain/ability/item modifiers, spread reduction, Trick Room, speed control, protect counters — *computed*, not tagged.
- **Learned identity embeddings** where identity is actually known (moves/item/ability vocabulary indices → embeddings inside the net).
- **Frozen orderings** pinned by name against poke-env's enums, so dependency upgrades can't silently shift a feature.

**Action space:** 16 actions per active slot (4 moves × 3 target buckets + 4 switches) + a separate 3-way gimmick head, with per-slot legality masks shared between training and serving.

## 4. The neural network

**`AttnBCPolicy`** (`models/bc_model_attn.py`, ~2.3M params): the flat state is reshaped in-model into **12 Pokémon tokens**; a shared per-mon encoder + learned slot embeddings feed a Transformer self-attention stack (the twelve Pokémon attend to one another — synergy and threat assessment become attention), plus a global field encoder. Heads: two own-action heads (one per active slot), two **auxiliary opponent-action heads** (they sharpen the shared trunk's threat model), gimmick heads for the Mega decision, and a **win-probability value head** that drives both the advantage weighting below and the serve-time exploitability meter.

## 5. Training

- **Behavior cloning** (`training/train_bc.py`): masked per-slot action cross-entropy + value BCE + gimmick and auxiliary-opponent losses; re-encodes snapshots at train time.
- **Streaming memmap loader** (`training/encoded_cache.py`): the full corpus is a ~27 GB encoded matrix — far beyond a 32 GB workstation with PyTorch overhead — so the cache is built in streaming chunks and memory-mapped read-only at train time (one-row copies per item). The full-corpus retrain runs on a single RTX 3070 Ti + 32 GB RAM. DataLoader **worker processes** re-open the caches by path (memmap views can't cross a spawn) and share the label tensors via shared memory — the fix for a feed that left the GPU at 20%.
- **The era-retrain cycle**: every real online game auto-copies into a training folder; a junk gate filters forfeits/timeouts; retrains ingest the winner's perspective only ("imitate whoever won — including whoever beat us"), weighted by opponent rating, against a belief blended from Pikalytics priors and the actually-observed ladder meta.
- **Offline advantage weighting** (Metamon's "exp" scheme): each decision is reweighted by `exp(β·(outcome − V(s)))` using the trained value head — shifting BC from "imitate everyone" to "imitate what beat expectation" without ever leaving the data manifold. This is the sole place a reward signal (the ±1 game outcome) enters the shipped weights, and it's the recipe of the deployed `checkpoints_attn_era2` net.

### How the shipped agent learns

The agent's skill is **behavior cloning (BC)** sharpened by **offline advantage weighting** — supervised learning, no environment interaction:

- **Behavior cloning** is the backbone: ~1.4M human decisions as `(state → the action the human took)`, trained to maximize the log-probability of that action under per-slot legality masks, jointly with the value head (did this player go on to win?) and the auxiliary heads. BC gives a strong, human-like prior for free, and it is the honest ceiling of the data: val top-1 saturated at ~0.61 because the available ladder replays average ~1600 Elo, not because the model ran out of capacity.
- **Advantage weighting** is the only reward-driven step: the ±1 win/loss outcome, compared to the value head's prediction, up-weights the decisions that beat expectation and down-weights the ones that didn't. It's a light, offline form of RL that never leaves the demonstration data — the property that makes it safe (a hand-built forward-model *search* was tried and dropped precisely because evaluating the value head on synthetic states it never trained on pulled the policy off-manifold).

**Online reinforcement learning is not part of the shipped agent.** A full PPO/self-play stack with a principled reward (sparse ±1 terminal, γ=0.997 discounting for speed-without-a-stall-penalty, a KL-to-BC anchor, and gated potential-based shaping off the win-prob value head) exists in `selfplay/`, but at this corpus scale it plateaued at the BC anchor's strength, so the deployed model does not use it. The strength levers that actually shipped are **data, offline advantage weighting, the contrastive set-preview head, and serve-time adaptation**.

### The exploiter — the agent's robustness meter

One piece of RL *is* part of the product, but as **measurement, never training**: an `exploiter` (`selfplay/exploiter.py`) trains a best-response against a **frozen copy** of the deployed net and reports the **exploitability curve** — how fast a dedicated opponent's win-rate climbs vs games trained. The frozen target is loaded from disk and never mutated; the exploiter's own checkpoint is written to a logs folder and its games never touch any training corpus. The number is expected to be high in absolute terms (a best-response can beat any fixed policy); the *product* signal is the trend **across eras at equal games-trained** — a later net whose curve rises more slowly is more robust to a human who's learning its habits. The current deployed net (`checkpoints_attn_era2`) plateaus around **0.65–0.70** exploitability at an 8-hour budget — the baseline the next era has to push down.

## 6. Evaluation

Three layers, in increasing order of what actually matters:

1. **The ruler** (`eval/bc_val_report.py`): fixed reference corpus, per-head / per-turn-bucket / per-decision-type / per-archetype / **held-out-team** slices, value Brier — every checkpoint comparison is apples-to-apples.
2. **Head-to-head A/Bs** (`selfplay/game_runner.py`): thousand-game single-team mirrors with Wilson confidence intervals and wiring verification (did the feature actually fire?) before any number is trusted.
3. **The human benchmark** (`eval/human_benchmark_report.py`): every game against a human (local or online) is recorded — result, teams, opponent, ladder ratings, an HTML replay — and the report computes win rates per session and the **exploitability curve**: does the human's win rate rise as they learn the bot? A flat curve means an exploit stops paying; that curve, not validation accuracy, is the project's real goal metric.

## 7. Serving & deployment

Three transports, one decision core (encoder + net + team-preview model + belief splice):

- **`play/play_vs_human.py`** — challenge the bot on a local Showdown server.
- **`play/play_vs_human_browser.py`** — two-browser-tab local play driven by **`BattleHost`**: a connection-less poke-env player fed raw websocket frames captured from a browser tab, its `/choose` commands shipped back in. The decision pipeline is reused byte-for-byte with no socket of its own.
- **`play/play_online_browser.py`** — the same transport pointed at **play.pokemonshowdown.com**: logs into a real account, the human supervises matchmaking, the AI plays every battle that opens. **`play/play_ladder.py`** is the autonomous alternative (direct websocket, `.ladder()` search) for any server.

All transports record the benchmark data automatically. A separate **team-preview network** (SBDA architecture with self/cross-attention over both rosters) picks the bring-4 and leads — since 2026-07 through a **contrastive set-scoring head**: instead of ranking Pokémon individually (which mode-mixed brings on two-mode teams), it scores every complete 4-subset as a unit — a marginal-sum term plus explicit pairwise-compatibility and set-level terms, trained with a 15-way listwise loss against the human's actual pick. Zero-initialized so it *starts* at exactly the old greedy behavior and has to earn every deviation; it cleared the adoption gate at +3.8pp bring-set accuracy and +154 elo-adjusted performance in its measured online block. In **best-of-3 sets** it additionally receives each side's previous-game bring/leads as a zero-init side input (`bo3_state`), so game-2/3 previews can react to what the opponent actually showed.

## 8. The adaptation layer

Static policies get exploited — our own benchmark proved it (the creator found a Wide Guard exploit in game 3). The counter-exploitation stack keeps the trained net frozen and adapts around it:

- **Serve-time pattern tilt** (`play/adapt_rules.py`): when the opponent shows a high-confidence repeated pattern (e.g. Wide Guard multiple turns running), a small logit bias tilts the policy toward single-target play. A tilt, not an override — the model still chooses, and an overwhelming preference survives.
- **Per-opponent dossiers** (`play/opponent_dossier.py`): every finished game updates a JSON dossier per opponent — revealed sets, items, abilities, W-L history — and can **warm-start the belief** in later games (`apply_dossier`, flag-gated): unknown items/abilities/moves fill from what that opponent showed before, with in-battle evidence always winning.
- **Between-game team routing** (`play/team_router.py`): against a known opponent we just lost to, pick the pool team with the best prior against their revealed archetype — priors seeded from a 172k-sample archetype-vs-archetype win matrix computed over the corpus, never hand-waved.
- **Best-of-3 set state** (`play/bo3_state.py`): games of a Bo3 are linked (protocol-verified on the pinned server); brings, leads, and the opponent's shown Pokémon carry across games into the team-preview net's set-context input.

How well the whole stack resists a *learning* opponent is quantified by the **exploiter** described in the training section — the worst-case robustness meter, run per era against a frozen copy of the deployed net.

## 9. How a single turn actually works

The clearest way to understand the system is to follow one decision end-to-end (online browser mode; local play only differs in transport):

1. **A websocket frame arrives** in the browser tab (`|request|` — Showdown asking for our order). Playwright's `framereceived` hook pushes the raw text onto a queue; the consumer feeds it to `BattleHost.feed_async`, which routes it through **poke-env's own protocol dispatcher** on a background event loop. `BattleHost` holds a real `VGCPlayer` built with `start_listening=False` — the full production player with no socket — and monkey-patches its `send_message` so decisions are *captured* instead of sent.
2. **poke-env updates its battle model** and calls `choose_move(battle)`. The player first replays the public log prefix once to build the **gap-#6 opponent snapshot** — our own reconstruction of the opponent's side (poke-env's view plus belief estimates for everything still hidden), including the just-resolved previous turn for the in-game belief update.
3. **The belief fills the blanks**: unrevealed movesets/items/spreads come from Pikalytics priors, narrowed by everything observed so far (revealed moves, damage-consistent stat ranges, Choice-lock coherence).
4. **The encoder writes the 5057-float state**: 12 Pokémon tokens × 413 features (types, computed stats, status, boosts, item/ability *effect categories*, move features, protect counters, …) + global field state, in the frozen v19 layout — byte-identical to what the trainer produced from parsed replays.
5. **Legality masks** are built per active slot (16 actions each) from Showdown's authoritative usable-move/switch lists — a disabled move or fainted bench slot is masked, so the net can only pick playable actions.
6. **One forward pass** of `AttnBCPolicy` yields per-slot action logits, gimmick logits, and a win-probability. If the **adaptation layer** is on and the opponent has, say, Wide-Guarded two turns running, a small logit bias tilts spread moves down before the masked argmax. Cross-slot switch collisions are re-decoded (both slots can't switch into the same bench mon).
7. **The order is assembled and shipped** — `/choose move 1 2, move 3 1 mega` captured by the host, relayed into the tab's socket by the consumer. If Showdown *rejects* it (a rare mask desync), an escalation ladder retries with fresh legal actions and ultimately falls back to `/choose default` rather than hanging — every fallback is counted and surfaced, never silent.
8. **At battle end**, the recording hooks fire before state is reclaimed: a bench JSONL row (result, teams, opponent, ratings), an HTML replay with the full log, and a dossier update for that opponent. The exploitability report reads it all.

The property that makes the whole thing trustworthy is **train/serve parity**: the offline encoder (reading parsed snapshots) and the live encoder (reading poke-env objects) are held byte-identical by tests, the action codec is shared, and checkpoint loading refuses any layout mismatch. When the model plays badly, it's a *model* problem — not a silent skew between what it saw in training and what it sees live.

---

## Results

- **Data was the lever that worked first.** Val top-1 on held-out human decisions: **0.585** (6.6k battles) → **0.595** (+11k) → **0.608** (+30k) — and then **flat** at the full 69k, isolating a *data-quality ceiling* (the available ladder replays average ~1600 Elo) rather than a volume or capacity limit. Doubling data of the same quality moved nothing; the value head kept improving (Brier 0.26 → 0.19).
- **Offline advantage weighting** added a targeted endgame improvement (+2.9pp on turn-11+ decisions, the game-deciding ones) at zero cost elsewhere.
- **The era-retrain cycle works.** Retraining on the bot's own online games (winner's-perspective only: imitate whoever won, weighted by opponent rating) plus a Pikalytics×observed-ladder belief blend lifted the elo-adjusted performance rating from **1212 to 1271** in a controlled 50-game block.
- **The contrastive team-preview set head** is the architecture that shipped: scoring complete 4-subsets instead of individual Pokémon cleared its offline gate at **+3.8pp bring-set accuracy** over the greedy decode, then delivered **1425 elo-adjusted performance** (peak rating 1499) in its measured block — +154 over the same battle net with the old preview decode, against opponents ~170 elo stronger.
- **First recorded human benchmark: the bot beat its creator 4–1** (best team vs best counter-effort). The one loss came from a discovered exploit — which did not keep paying in the following games, and which the adaptation layer now addresses directly.
- **A production-grade deployment stack**: three serving transports plus a single control UI, checkpoint hot-swap flags with a live deploy-parity check, closed-team-sheet discipline matching the ladder, automatic benchmark/dossier recording, and a 1,200+ test suite with byte-parity guards between training and serving.

## Acknowledgements & prior art

This project stands on excellent prior work and open resources:

- **[Foul Play](https://github.com/pmariglia/foul-play)** by **pmariglia** — the original inspiration. A search-based Showdown battle bot that has reached **#1 on the official Pokémon Showdown ladder**, proving a bot could compete at the top of real human play. This project began by studying its design; early scaffolding was learned from (and eventually rewritten past) its approach, and the ambition — *real ladder play against real humans* — comes straight from it.
- **[VGC-Bench](https://github.com/cameronangliss/vgc-bench)** — Angliss et al., *"VGC-Bench: A Benchmark for Generalizing Across Diverse Team Strategies in Competitive Pokémon"* ([arXiv:2506.10326](https://arxiv.org/abs/2506.10326), MIT license). Three enormous contributions to this project: the **open battle-log dataset** ([`cameronangliss/vgc-battle-logs`](https://huggingface.co/datasets/cameronangliss/vgc-battle-logs)) whose Champions-format subset became ~90% of our training corpus; the **entity-transformer architecture reference** our battle net parallels; and the **scientific grounding** — their measured results on team-count generalization collapse and universal exploitability told us which walls were real before we spent months on them.
- **[Metamon](https://github.com/UT-Austin-RPL/metamon)** — Grigsby et al., *"Human-Level Competitive Pokémon via Scalable Offline Reinforcement Learning"* ([arXiv:2504.04395](https://arxiv.org/abs/2504.04395)). The template for our training philosophy: offline, BC-anchored, advantage-weighted learning on a single GPU rather than online self-play at datacenter scale.
- **PokeChamp** ([arXiv:2503.04094](https://arxiv.org/abs/2503.04094)) and **PokeLLMon** ([arXiv:2402.01118](https://arxiv.org/abs/2402.01118)) — for mapping the LLM-agent corner of the design space, and for the evidence that memoryless per-turn play gets read and exploited by humans.
- **[poke-env](https://github.com/hsahovic/poke-env)** (Haris Sahovic) — the Python interface to Showdown that every player, collector, and transport here is built on.
- **[Pokémon Showdown](https://github.com/smogon/pokemon-showdown)** (Guangcong Luo / Zarel and contributors, Smogon) — the battle simulator itself.
- **[Pikalytics](https://www.pikalytics.com/)** — the usage statistics that seed the belief system's priors.

*Personal educational project. Not affiliated with or endorsed by Nintendo, Game Freak, The Pokémon Company, Smogon, or any of the projects above. Pokémon and all related properties are trademarks of their respective owners.*

---

## Repository layout

```
Victory-Dance/
├── v_dance/                # installable package (pip install -e .)
│   ├── parser/             # Showdown logs -> per-turn transitions; belief_state, match_belief
│   ├── encoders/           # snapshot -> 5057-dim state (layout v19); white-box forward model
│   ├── models/             # AttnBCPolicy set-attention battle net + SBDA team-preview net
│   ├── training/           # BC trainer, streaming memmap cache, advantage weights, z-archetypes
│   ├── selfplay/           # A/B game runner, league/gate machinery, multiprocess collection
│   ├── play/               # serving: local/browser/online transports, adapt_rules, dossiers
│   ├── eval/               # checkpoint ruler, human-benchmark report, gauntlet + Elo
│   └── datatools/          # Mission Control UI, team generator/builder, HF ingest, corpus QA, dashboards
├── data/                   # dex data, Pikalytics usage, teams, prepared training corpora
├── docs/                   # design docs + the execution playbook (audit, decisions, specs)
├── tests/                  # pytest suite: byte-parity, legality, QA gates, unit tests
├── ai_train_scripts/       # model checkpoints (battle + team-preview)
└── artifacts/              # run logs, benchmark records, dossiers, replays (gitignored)
```

## Building & specializing teams

A crucial design property: **the battle net is team-agnostic.** Because the encoder uses mechanics (types, computed stats, item/ability *effect* categories) and never species identities, the *same* checkpoint pilots any Champions-doubles team. `maw_zard` is only the **default team** it brings (`VD_DEFAULT_TEAM`) and the **proving team** used in eval — it is *not* baked into the weights. "Specializing on a different team" therefore needs **no retraining** — you just give the bot the team:

1. **Get a legal paste.** Either write a Showdown export and drop it as a file in `teams/Champions/<regulation>/`, or generate one in the **team-builder** (Mission Control → *Teams*, or `python -m v_dance.datatools.server` → `http://localhost:5174/`). The generator grows a roster by belief-driven beam search over teammate co-occurrence, fills each set from usage stats, and **scores** it three ways (archetype coherence, team-preview-net confidence, and a corpus matchup prior). Every generated or pasted team is checked by **Showdown's own validator**, so illegal teams are dropped, not silently played.
   - ⚠ Champions uses the **0–32 stat-point budget** (66 total), *not* classic 0–252 EVs, and enforces the VGC **Item Clause** (one of each item). The generator and validator handle both; hand-written pastes must too.
2. **Point the bot at it** — any of: set `VD_DEFAULT_TEAM=<name>` (Mission Control → *Deploy*), pass `--ai-team <name>` to any play harness, or pin it live in the Online-bot tab's team dropdown.

Every harness auto-discovers the pool under `teams/Champions/`, so a new file is usable immediately. (If you instead want to RL-*fine-tune* a net's habits to one team via self-play, the machinery exists — `generation.py --train-teams <name>` — but note the honest finding above: self-play-for-strength plateaus; the team-agnostic net + a good team is the intended path.)

## Setup: playing locally

**Prerequisites:** Python 3.11+ (venv recommended), Node.js, and a local Pokémon Showdown checkout that includes the Champions-format mod (exact pinned commits for Showdown and poke-env are in `PINS.md` — the encoder's enum-name mapping and the format both depend on them). The quickest path to *any* of the workflows below — play, train, evaluate — is **Mission Control** (`python -m v_dance.datatools.mission_control`); the explicit commands are given here so you know what each button runs.

```bash
# 1. install the package + the pinned server
pip install -e .
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown && git checkout <commit from PINS.md> && npm install && cd ..
```

Teams live in `teams/Champions/<regulation>/` as Showdown paste files — drop any team you want the bot (or you) to use there; every harness discovers the pool automatically.

**Option A — two-tab browser flow (recommended).** One command starts the server and opens two logged-in browser tabs with every pool team pre-imported into both Teambuilders. You challenge the AI from your tab; it auto-accepts and plays. The AI's team is `--ai-team <name>` if pinned, else whichever team you have *open* in the AI tab's Teambuilder, else random:

```bash
# The battle + team-preview checkpoints default to the deployed pair (model_io + .env), so you
# can omit --ckpt/--tp-ckpt entirely; they're shown here only to make the override explicit.
python -m v_dance.play.play_vs_human_browser --ai-team maw_zard \
    --ckpt ai_train_scripts/BC_model/checkpoints_attn_era2/battle_base.pt \
    --tp-ckpt ai_train_scripts/teamPreview_model/checkpoints_set/teampreview_sbda.pt \
    --adapt-rules --bench-note my_session
```

**Option B — simple flow.** The AI connects as a normal player; you open the printed URL, import the printed team, and challenge `VictoryDanceAI`:

```bash
python -m v_dance.play.play_vs_human --mode choose --ai-team maw_zard --human-team <yours> \
    --ckpt <battle.pt> --tp-ckpt <tp.pt>
```

Every finished game is recorded automatically (result/teams/turns → `artifacts/human_benchmark/human_bench.jsonl`, a playable HTML replay, and a per-opponent dossier). Read the results any time:

```bash
python -m v_dance.eval.human_benchmark_report        # win rates + exploitability curve
python -m v_dance.play.opponent_dossier               # what the bot knows about each opponent
```

## Setup: playing online

Online play runs through a real browser logged into a real account, with **you supervising matchmaking** — the AI plays whatever battle room opens (incoming challenges in the configured format are auto-accepted).

1. **Create `.env` in the repo root** (gitignored — never commit it):

```ini
PS_USERNAME=YourRegisteredAccount
PS_PASSWORD=...
PS_AVATAR=cynthia                       # optional
PS_CLIENT_URL=https://play.pokemonshowdown.com
WEBSOCKET_URI=wss://sim.smogon.com/showdown/websocket   # for the autonomous ladder harness
VDANCE_BATTLE_FORMAT=gen9championsvgc2026regmb          # exported stack-wide at startup
VD_BATTLE_CKPT=ai_train_scripts/BC_model/checkpoints_attn_era2/battle_base.pt
VD_TP_CKPT=ai_train_scripts/teamPreview_model/checkpoints_set/teampreview_sbda.pt
VD_DEFAULT_TEAM=maw_zard
```

These `VD_*` deploy defaults must match the canonical checkpoints hard-coded in `play/model_io.py` (Mission Control's Deploy tab shows a live ✓/✗ parity check). To roll a new checkpoint out, point both at it; to roll back, point both at the previous pair — the anchor `checkpoints_attn_pre_gen141/battle_base.pt` is the fixed historical reference and is never overwritten.

2. **Dry-run first** — connects, logs in (scripted; if the login UI changes it falls back to "log in manually in the window" and waits), imports the team pool, and idles so you can verify everything without playing:

```bash
python -m v_dance.play.play_online_browser --dry-run
```

3. **Go live.** Drop `--dry-run`. Sensible first session: a couple of *unrated* challenge games before touching the rated ladder. All recording (bench rows **with ladder ratings**, replays, dossiers) is automatic; `--adapt-rules` enables the anti-exploit tilt.

```bash
python -m v_dance.play.play_online_browser --adapt-rules --bench-note online_v1
```

Once it's live, a **control panel** comes up (its own local page, and mirrored into Mission Control's *Online bot* tab — which is also where you set the format + launch config and start the bot in the first place, with `--adapt-rules` and `--dossier` on by default) where you drive matchmaking without touching the browser: start a **ladder run of N rated games** (it re-queues after each finished game until the target is hit), toggle **auto-accept** for incoming challenges, send **private challenges** by username, pin the AI's team, and watch the live rating / W–L tally / activity feed. `--dossier` warm-starts the belief against opponents you've faced before; `VD_ROUTE_TEAMS=1` lets the bot pick its best-matchup pool team against a known opponent on challenge-accepts.

There is also a fully-autonomous direct-websocket harness (`play/play_ladder.py`) that searches ladder matches by itself on any server (`--server-url`, `--username/--password`, `--games N`) — the browser flow above is the supervised default.

**Training & evaluation** (the research side):

```bash
python -m v_dance.training.train_bc --data <jsonl folders> --mmap-cache ...   # full corpus on 32 GB RAM
python -m v_dance.eval.bc_val_report --ckpt <base.pt> --ckpt <cand.pt> --held-out-slice
pytest tests -q                                                               # 1,100+ tests
```

## Tech stack

**Python** (PyTorch, NumPy) · **poke-env** for the Showdown protocol · **Node.js** Pokémon Showdown as the battle engine · **Playwright** for the browser transports · spawn-multiprocessing for GIL-free battle collection · **pytest** (1,100+ tests).

---

## Appendix: research history (for technical readers)

> Everything above describes the shipped agent. This appendix is the R&D record behind it — the ideas that were built, measured with controlled A/Bs, and *retired on evidence*, plus the bugs that shaped the current design. It's kept because the negative results are, honestly, some of the project's most useful output: they mirror what the academic literature found at ~100× the compute.

### Negative results

Each of these was built properly, evaluated with controlled A/Bs and confidence intervals, and **retired on evidence** — the same wall the field hit at datacenter scale (VGC-Bench's PPO league needed 8×A40 and still collapsed on generalization; every agent they trained was ~100% exploitable by a best-response):

| Idea | Verdict |
|---|---|
| Belief-weighted expectimax search over a white-box forward model | **Hurts** (38.6% vs argmax, CI [36.5, 40.8]) — an imperfect forward model drags the policy off-manifold |
| PPO-style league self-play from the BC anchor | Plateaued at the anchor's strength |
| C51 distributional value head | No strength change |
| Opponent-conditioned policy heads | No strength change |
| Recurrent / frame-stack memory over the battle | No strength change, value head degraded |
| Latent team-archetype conditioning (z) — run twice, second time with a clean 100%-labelled artifact | No gain either time; closed |
| Team-preview joint decode via subset-mask augmentation | Dose-response trade: the augmentation converts full-roster accuracy into partial-roster validity and cannot win in absolute terms |
| Serve-time stochastic sampling (τ=0.45) | Costs 5.5pp head-to-head — anti-predictability is a dial with a price, not a free win |

The pattern behind all eight: **at this corpus size and quality, the bottleneck is the demonstration data, not bolted-on architecture** — sharper estimates of the opponent don't convert into wins when the policy prior itself caps skill. That conclusion redirected the project toward data levers, offline improvement, and serve-time adaptation — and the one architecture change that *did* win (the set-scoring preview head) is the exception that proves the rule: it fixed a **decode-expressiveness** gap (greedy per-mon ranking mathematically cannot represent "bring A or B, not both"), not a representation-learning one.

### Engineering journey: the bugs that shaped the design

The current architecture is scar tissue from real failures. A selection, because these taught more than the successes:

- **Illusion broke everything, repeatedly.** Zoroark's Illusion means the replay log *lies about which Pokémon is on the field* — damage attribution, faint attribution, targeting, even which side's bench a mon returns to. It took several iterations (species-clause cross-checks, doubles-parity fixes, switch dedup) before the parser survived a corpus scan clean. Lesson: in adversarial log formats, parse defensively and audit with invariants (`corpus_qa` counts illegal-under-mask decisions; the gate is zero).
- **The forward-model audit — and its punchline.** A depth-1 search over a hand-built battle simulator initially looked promising. A targeted audit found **seven** silent bugs in the simulator (Choice Scarf's ×1.5 unapplied, -ate ability retyping omitted, Intimidate-on-entry missed, spread moves hitting the caster's ally, Mega opponents simulated with base-forme stats…). We fixed all seven, re-ran the 2,000-game A/B — and search **still lost by 11 points**. The simulator wasn't the problem: evaluating a value head on *synthetic states it never trained on* was. That off-manifold lesson now guards every design decision (it's why advantage weighting — which never leaves the data — replaced search).
- **The retry storm.** A deterministic policy whose order Showdown rejects will submit the same illegal order forever. Worse, some legal situations are *unrepresentable* in a 16-action codec (Struggle, recharge turns). The fix is an escalation ladder — perturb to fresh legal actions, detect exhaustion, fall back to `/choose default` — with every non-model decision *counted and reported*, because a bot that silently plays random moves invalidates your evaluation without telling you.
- **26 GB of features vs 32 GB of RAM.** The full corpus encodes to a matrix that simply does not fit in memory next to PyTorch. Solution: build the encoded cache in streaming 2,000-file chunks appended to disk, then memory-map it read-only at train time with a lazy dataset (one-row copy per `__getitem__`). Full-corpus training on a desktop. Corollary lesson: Windows' "96% memory used" during mmap training is *reclaimable page cache*, not leak — we verified commit charge (21/52 GB) before panicking.
- **Windows asyncio, three separate times.** Ctrl-C is not delivered to a parked `select()` (fix: accept loops wake every 250 ms); Playwright needs the Proactor loop while poke-env's background loop must stay Selector (fix: two loops, bridge with `run_coroutine_threadsafe`); and a `cp1252` console killed two multi-hour training launches on a *cosmetic box-drawing header* (fix: UTF-8 forced in launch scripts, and the print hardened). Long commands are now shipped as verified `.sh` scripts after a shell paste mangled a flag — losing `--mmap-cache` from a command is the difference between training and OOM.
- **The browser transport's event-loop wedge.** Feeding a captured frame for an *already-finished* battle made poke-env block forever waiting for a battle object that would never come — freezing the whole session. Late frames are real (Showdown re-sends room logs on rejoin). The host now tracks ended battles and drops their strays, bounded so the set can't leak.
- **The teambuilder import that looked right and wasn't.** Injecting teams into the Showdown client's localStorage produced teams that *displayed* but were empty — `Storage.importTeam` returns a parsed array, not the packed wrapper the client saves, and the team picker silently requires `capacity: 6`. Both discovered by live probing the client, both now documented in code comments with the probe scripts kept.
- **Measure twice: the z confound.** The first latent-archetype run looked like a −1.2pp regression — until we noticed 80% of training examples had been assigned the UNKNOWN archetype (the clustering artifact predated the corpus growth). A rebuilt artifact (31k teams, 100% labelled) and a rerun gave z a *fair* trial. Verdicts are only verdicts when the wiring is verified — the same discipline that caught the sampling A/B's "did it actually fire" checks.

---

## What it would take to beat the best — and the hardware wall

This agent is scoped to beat **average** ladder players (its "G1" goal), and it does. Beating the **best** humans is a fundamentally different problem, and it is worth being explicit about why this project doesn't attempt it — the reason is compute, not ambition.

**The ceiling is baked into behavior cloning.** BC trains the policy to reproduce the human action, so its optimum *is* the human policy in the data — and cloning a mixture of players converges to their **average**. This corpus averages ~1600 elo, and the measured validation accuracy went flat at the full dataset, so ~1600-level play is the imitation cap. The project climbs somewhat above that mean — by imitating the **winners** (offline advantage weighting), by playing every turn with **consistency** a tilting human doesn't, and by **aggregating** knowledge no single player has — which is enough to beat the average opponent. It is *not* enough to beat a top human, and no amount of the same-quality data or bolted-on architecture changes that (eight negative results confirm it).

**The levers that *would* reach top-human play, ordered by what's realistic on this hardware:**

1. **Higher-quality data — the one affordable lever.** The bottleneck is data *quality*, not volume. Ingesting top-cut / tournament / high-ladder-only replays raises the imitation ceiling with **zero extra compute**. This is the first thing to do here.
2. **Harder offline improvement.** Push advantage weighting further, or adopt a full offline-RL objective (IQL/CQL), to imitate only the *elite* decisions. Fits on one GPU; the risk is over-filtering a smallish corpus.
3. **Non-collapsing self-play RL — the real superhuman path, and the expensive one.** Genuinely exceeding humans requires *generating* new strategy via self-play, not imitating. The blocker isn't only compute: naive self-play here **collapses into narrow, exploitable equilibria** (the RL exploiter meter exists to measure exactly that). Doing it right needs league/population training (PFSP, AlphaStar-style), careful reward shaping, and far more compute than is available here.
4. **On-manifold search.** Lookahead only helps if evaluation stays on the training distribution — which means a **learned** world model (MuZero-style), not the hand-built simulator that measurably *hurt* here. A large build, GPU-heavy to train, cheaper to run.
5. **Opponent modeling & adaptation — a different axis.** You can beat strong humans partly by exploiting *their* predictability instead of out-playing them move-for-move. The adaptation layer already does a little of this and is the cheapest place to add strength on the current box.

**The hardware wall.** This entire project — training, self-play, evaluation — runs on a **single RTX 3070 Ti (8 GB VRAM) + 32 GB RAM, one heavy run at a time, on a Windows workstation.** That budget shaped every decision in this README: the 2.3M-parameter net, the streaming memmap loader that fits a 27 GB corpus into 32 GB of RAM, the single-GPU offline-improvement strategy, and the decision to *park* online self-play after it plateaued. For contrast, the approaches that reach or approach superhuman play were **datacenter-scale**: VGC-Bench's PPO league used 8×A40 GPUs (and still produced ~100%-exploitable agents), and AlphaStar / OpenAI Five ran on large clusters for weeks.

**Bottom line.** On a 3070 Ti / 32 GB box, levers **1, 2, and 5** are where realistic further gains live, and they can push a bit past "beats average." Levers **3 and 4** — the ones that could plausibly reach *top-human* play — require compute this project does not have; pursuing them would mean renting multi-GPU/cloud time and treating today's agent as the warm-start. That constraint is not a footnote — it is the reason the project's honest goal is *"a strong, robust, average-beating, team-agnostic bot,"* not *"ladder #1."*
