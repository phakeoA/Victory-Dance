# Victory-Dance ![shiny Hisuian Zoroark](https://play.pokemonshowdown.com/sprites/ani-shiny/zoroark-hisui.gif)

[![Tests](https://github.com/phakeoA/Victory-Dance/actions/workflows/tests.yml/badge.svg)](https://github.com/phakeoA/Victory-Dance/actions/workflows/tests.yml)

**A deep-learning agent that plays competitive VGC doubles Pokémon on [Pokémon Showdown](https://pokemonshowdown.com/).** It starts from behavior cloning on ~86,000 human battles, is improved by offline advantage weighting, and is then trained by **self-play reinforcement learning** — PPO in a league that includes behavior-cloned human opponents — with its team-preview network learning in the same loop. It plays the live ladder through a browser transport and records and measures every game.

On the **Regulation M-C** ladder the self-play-trained model reached the **top 500 twice** in October 2026 (seen at **#172**, Elo 1679; session peak Elo 1744), and lifted the bot's ladder performance rating from **~1240** (the behavior-cloned model) to **~1460**.

Format: **Gen 9 "Pokémon Champions" VGC 2026** doubles — **Regulation M-C** on the live ladder (M-A and M-B before it). Mega Evolution is legal in this format; Terastallization is disabled by the mod.

> This is an educational research project. The README describes the **shipped agent** — the models, the pipeline that trains them, and the stack that deploys and measures them — and keeps the negative results in an appendix, because they shaped the design as much as the wins did.

---

## Table of contents
- [Why this problem is hard](#why-this-problem-is-hard)
- [System overview](#system-overview)
- [Mission Control — one UI for the whole project](#mission-control--one-ui-for-the-whole-project)
- [1. Data & the parser](#1-data--the-parser)
- [2. The belief system](#2-the-belief-system)
- [3. The state encoder](#3-the-state-encoder)
- [4. The neural networks](#4-the-neural-networks)
- [5. Training: from imitation to self-play](#5-training-from-imitation-to-self-play)
- [6. Evaluation](#6-evaluation)
- [7. Serving & deployment](#7-serving--deployment)
- [8. The adaptation layer](#8-the-adaptation-layer)
- [9. How a single turn actually works](#9-how-a-single-turn-actually-works)
- [Results](#results)
- [Acknowledgements & prior art](#acknowledgements--prior-art)
- [Repository layout](#repository-layout)
- [Building & specializing teams](#building--specializing-teams)
- [Setup: playing locally](#setup-playing-locally)
- [Setup: playing online](#setup-playing-online)
- [Training & evaluation commands](#training--evaluation-commands)
- [Tech stack](#tech-stack)
- [Appendix: research history (for technical readers)](#appendix-research-history-for-technical-readers)
- [Where the remaining gap is](#where-the-remaining-gap-is)

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
   Encoder — 6377-dim mechanic-based state (NO species one-hots; frozen layout v20)
        │
        ▼
   AttnBCPolicy — per-mon set-attention battle net (~2.4M params)
   heads: our-action ×2 · opp-action ×2 (aux) · Mega · win-prob value
        │
        ├── 1. Behavior cloning on human decisions (streaming memmap loader: full corpus on 32 GB RAM)
        ├── 2. Offline advantage weighting  w = exp(β·(outcome − V(s)))   ["imitate what won"]
        └── 3. Self-play PPO league — KL-anchored to the start policy; opponents = itself, past
               generations, behavior-cloned human players, scripted anchors; targeted drills;
               the team picker trains in the same loop (every checkpoint is a net + picker PAIR)
        │
        ▼
   Serving: local harness · browser transport (BattleHost) · online ladder bot (play.pokemonshowdown.com)
   + SBDA team picker with a contrastive set-scoring head (bring-4 + leads as a UNIT)
   + serve bandit (which net + picker pair plays, per account) · 5 games at once · send pacing · reconnects
   + adaptation layer (pattern tilt · dossiers · dossier belief warm-start · Bo3 set context)
   + recording: benchmark rows with ladder ratings, replays, dossiers, RL trajectories
   + RL exploiter — worst-case robustness meter (frozen target; never trains the bot)
```

Everything downstream of the encoder reads a **frozen, versioned feature layout** (`STATE_LAYOUT_VERSION = 20`, `STATE_DIM = 6377`) with load-time guards, so a layout change fails loudly instead of silently corrupting a trained net. Layout v20 added a Mega-Evolution preview to every Pokémon; v19 checkpoints are upgraded on load with identical outputs, so the training chain did not have to restart.

---

## Mission Control — one UI for the whole project

Every part of the project — data prep, belief scraping/blending, training the battle and team-preview nets, self-play, evaluation, deployment, and both local and online play — is driven from a single local page:

```bash
python -m v_dance.ui.mission_control             # opens http://127.0.0.1:8990/
```

It's a dependency-light, **torch-free** stdlib server (starts instantly, safe to run alongside anything) that knows every entry point in the project through a typed command registry. Tabs:

- **Overview** — the pipeline map, a live **deploy-parity check** (`.env` ↔ `model_io` defaults, shown ✓/✗), belief freshness, the exploiter curve, and which services are up.
- **Online bot** — the home for online play. Before it's running you set the **launch config** right here — **format**, account, team pin, and the anti-exploit `--adapt-rules` + opponent-`--dossier` toggles (both **on by default**) — and launch from this tab; picking a format writes it to `.env` (the stack binds it at launch). Once it's live the same tab drives it: start a **ladder run of N rated games**, choose how many games run at once, toggle **auto-accept challenges** and auto-close, send **private challenges**, pin a team, and watch the rating / W–L tally / activity log. (The live controls are proxied from the bot's own control panel, so you never open a second window.)
- **Play** — local AI-vs-AI and vs-human in the browser. Each harness has team/checkpoint/format pickers and a **"Copy command"** one-liner.
- **Train** — the battle-net era retrain, both team-preview trainers, behavior-cloned league opponents, the exploiter, and self-play. Heavy runs launch **one at a time** (GPU/RAM guard) and stream **live progress**. Prefer a terminal? Every card has a **"Copy command"** button with the exact one-liner.
- **Parser / Teams** — the combined VOD replay parser/annotator (upload a replay → export JSONL training transitions) and team builder (generate/validate/score teams), embedded.
- **Data / Eval** — replay parsing, belief blend, corpus QA, ingest; the pytest suite, rulers, the human-benchmark report, gauntlets.
- **Deploy** — edit the `.env` deploy keys (checkpoint/team/format) from dropdowns of what's actually on disk, with the parity check.
- **Monitor** — the self-play dashboard (live generations, launcher, console), embedded.
- **Jobs** — every launched job with a live log tail and a stop button.

Heavy runs are launchable but **you** click the button; the server never auto-starts training. Credentials never leave the server (a strict env whitelist — passwords are never sent to the page).

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

The agent maintains an explicit **belief** over the opponent's hidden sets: priors seeded from **Pikalytics** usage statistics (`belief_state.py`), narrowed live by in-game evidence (`match_belief.py`) — revealed moves, damage-based stat back-solving, Choice-item consistency, paradox-boost deduction. When both players accept open team sheets, the opponent's sheet replaces the estimate outright. The parser and the live player share this machinery, so training and serving see the same kind of opponent estimate.

## 3. The state encoder

**`v_dance/encoders/`** maps a battle snapshot to a fixed **6377-float** vector; twin offline/live encoders are held to **byte-level parity** by tests, so the model sees identical inputs in training and play.

- **No identity one-hots.** Species encode as **types + base stats**; items/abilities as multi-hot **strategic effect categories**. A brand-new Pokémon or item in a future regulation slots into the same layout — the design bet that lets one net generalize across rosters and pilot user-supplied teams.
- **Exhaustive mechanic computation** — weather/terrain/ability/item modifiers, spread reduction, Trick Room, speed control, protect counters — *computed*, not tagged.
- **A Mega preview per Pokémon** (layout v20): what each mon becomes if it Mega Evolves — so the net can weigh the Mega before committing to it.
- **Learned identity embeddings** where identity is actually known (moves/item/ability vocabulary indices → embeddings inside the net).
- **Frozen orderings** pinned by name against poke-env's enums, so dependency upgrades can't silently shift a feature.

**Action space:** 16 actions per active slot (4 moves × 3 target buckets + 4 switches) + a separate gimmick head for the Mega decision, with per-slot legality masks shared between training and serving.

## 4. The neural networks

**The battle net — `AttnBCPolicy`** (`models/bc_model_attn.py`, ~2.4M params): the flat state is reshaped in-model into **12 Pokémon tokens**; a shared per-mon encoder + learned slot embeddings feed a Transformer self-attention stack (the twelve Pokémon attend to one another — synergy and threat assessment become attention), plus a global field encoder. Heads: two own-action heads (one per active slot), two **auxiliary opponent-action heads** (they sharpen the shared trunk's threat model), gimmick heads for the Mega decision, and a **win-probability value head** — the critic that self-play starts from, and the input to the serve-time exploitability meter.

**The team picker — SBDA** (`models/`, ~0.33M params): self- and cross-attention over both rosters, with a **contrastive set-scoring head**. Instead of ranking Pokémon one by one (which mode-mixed the brings of two-mode teams), it scores every complete 4-subset as a unit — a marginal-sum term plus explicit pairwise-compatibility and set-level terms — and then the lead pair inside it. It was first trained with a 15-way listwise loss against the human's actual pick, zero-initialized so it *started* at exactly the old greedy behavior and had to earn every deviation; since October 2026 it keeps learning from results inside the self-play loop (§5.3). In **best-of-3 sets** it also receives each side's previous-game bring/leads (`bo3_state`), so game-2/3 previews can react to what the opponent showed.

## 5. Training: from imitation to self-play

The shipped agent is the end of a chain: behavior cloning gives a human-like prior, advantage weighting sharpens it offline, and a self-play league turns it into the player on the ladder. Every stage warm-starts from the previous best checkpoint — nothing is trained from scratch.

### 5.1 Behavior cloning

- **Behavior cloning** (`training/train_bc.py`): ~1.4M human decisions as `(state → the action the human took)`, trained with masked per-slot action cross-entropy + value BCE (did this player go on to win?) + gimmick and auxiliary-opponent losses; snapshots are re-encoded at train time. Val top-1 saturated at ~0.61 — the honest ceiling of the data, because the available ladder replays average ~1600 Elo, not because the model ran out of capacity.
- **Streaming memmap loader** (`training/encoded_cache.py`): the full corpus is a ~27 GB encoded matrix — far beyond a 32 GB workstation with PyTorch overhead — so the cache is built in streaming chunks and memory-mapped read-only at train time (one-row copies per item). DataLoader **worker processes** re-open the caches by path (memmap views can't cross a spawn) and share the label tensors via shared memory — the fix for a feed that left the GPU at 20%.
- **The era-retrain cycle**: every real online game auto-copies into a training folder; a junk gate filters forfeits/timeouts; retrains ingest the winner's perspective only ("imitate whoever won — including whoever beat us"), weighted by opponent rating, against a belief blended from Pikalytics priors and the actually-observed ladder meta.

### 5.2 Offline advantage weighting

Metamon's "exp" scheme: each human decision is reweighted by `exp(β·(outcome − V(s)))` using the trained value head — shifting BC from "imitate everyone" to "imitate what beat expectation" without ever leaving the data manifold. It added a targeted endgame gain (+2.9pp on turn-11+ decisions) and is the recipe of `checkpoints_attn_era2`, the behavior-cloned model every self-play run descends from.

### 5.3 Self-play reinforcement learning — how the shipped agent was trained

The shipped battle net and team picker come out of a **self-play PPO league** (`v_dance/selfplay/` + `v_dance/rl/`):

- **PPO** with GAE (γ = 0.997, λ = 0.95 — a discount high enough that slow wins such as Trick Room or stall lines still pay), a clipped surrogate, and a **KL anchor** to the policy the run started from (a light 0.05 penalty; the Mega head's anchor is looser), so results reshape the human prior without erasing it. The critic starts as the value head and is warmed up on the new returns before the policy moves.
- **The league.** One generation = 1,000 games collected by 4 worker processes against a local Showdown server. Opponents: the current policy itself (both sides train), past generations (prioritized fictitious self-play), **behavior-cloned human players** (`selfplay/build_clone.py`: a "nemesis" clone trained on the humans who beat the bot, plus clones of the ladder's team archetypes — ~30% of games), and scripted anchors.
- **Gates.** A frozen-champion promotion gate (a candidate must beat the current champion head-to-head), a Hall-of-Fame veto against strategy cycling, and a fixed **panel** — the behavior-cloned model plus two human clones — played every generation as an external yardstick the league can't overfit.
- **Drills.** The opponent pool can over-sample the matchups the ladder punishes (e.g. Rillaboom, Mega Raichu, Archaludon rain), make non-learning opponents play for the field (switch their weather or terrain setter back in), and report a per-generation scoreboard for those matchups.
- **Mega-hold exploration.** Holding a Mega for a few turns (say, Mega Tyranitar re-setting sand after the opponent's rain) needs several unlikely choices in a row, so self-play rarely discovers it. The learner sometimes decides *at game level* to hold — only for Megas where waiting can pay (a field-setting Mega ability, a guarding base ability the Mega loses, a type change) — and the forced steps are importance-weighted so the gradient stays unbiased.
- **The team picker learns in the loop.** The learner picks its four with the picker and explores; after each generation the picker takes a PPO-clip step on its set and lead choices with a leave-one-out baseline per matchup and a KL anchor to its start. Every saved checkpoint is a **(battle net, picker) pair** that is trained, evaluated and served together.
- **Reward.** Terminal ±1, discounted, read from Showdown's result. Nothing the agent does to its own side is ever rewarded or penalized — sacrifices, Protect, self-damage are priced only by whether the game is won. An opt-in **reward v2** (`--reward-v2`, October 2026, under test) adds a small loss margin (their Pokémon knocked out, on losses only — every loss still sits ≥ 1.75 below every win) and a potential-based weather/terrain-control credit that telescopes, so it can speed up learning field control without changing which policy is best.
- **Preflight.** Before any long run, a shrunk 3-generation run plus a resume exercises every code path (every opponent kind, promotion, the picker update, saving) and aborts on failure.

On one desktop (Ryzen 9 5900X, RTX 3070 Ti, 32 GB) a run of 40 generations × 1,000 games takes ~4.5 hours.

**Why it works now when it didn't before.** The first self-play runs (June–September 2026) either plateaued at the behavior-cloned model's strength or overfit to their own league — in-league Elo above 2000, no gain on the ladder. Four changes made it transfer: opponent diversity from cloned human players, an external panel the league can't game, drills aimed at measured ladder leaks, and training the picker with the battle net (a self-play learner that never used its picker had been practicing brings it would never play on the ladder).

### The exploiter — the agent's robustness meter

One piece of RL is part of the product as **measurement, never training**: an `exploiter` (`selfplay/exploiter.py`) trains a best-response against a **frozen copy** of a deployed net and reports the **exploitability curve** — how fast a dedicated opponent's win-rate climbs vs games trained. The frozen target is loaded from disk and never mutated; the exploiter's games never touch any training corpus. The absolute number is expected to be high (a best-response can beat any fixed policy); the signal is the trend **across models at equal games-trained**. The behavior-cloned model (`checkpoints_attn_era2`) plateaued around **0.65–0.70** exploitability at an 8-hour budget.

## 6. Evaluation

Four layers, in increasing order of what actually matters:

1. **The ruler** (`eval/bc_val_report.py`): fixed reference corpus, per-head / per-turn-bucket / per-decision-type / per-archetype / **held-out-team** slices, value Brier — every imitation checkpoint comparison is apples-to-apples.
2. **Head-to-head A/Bs and re-tests** (`selfplay/game_runner.py`, `eval/panel_eval.py`): thousand-game blocks with Wilson confidence intervals, matchup filters (e.g. only opponents with Archaludon), and wiring verification (did the feature actually fire?) before any number is trusted.
3. **The human benchmark** (`eval/human_benchmark_report.py`): every game against a human (local or online) is recorded — result, teams, opponent, ladder ratings, an HTML replay — and the report computes win rates per session and the **exploitability curve**: does the human's win rate rise as they learn the bot?
4. **The ladder performance rating**: the maximum-likelihood Elo that explains every recorded game's result given the opponent's pre-battle rating. The ladder Elo moves 15–25 points per game and swings roughly ±60 around a model's true level, so models are compared by performance rating over ~1,000 games, never by a peak. Leak reviews (`eval/loss_breakdown.py`, `eval/field_control_report.py`) break it down by matchup — which opponents cost rating, and why.

## 7. Serving & deployment

Two transports, one decision core (encoder + battle net + team picker + belief splice):

- **`online/play_vs_human_browser.py`** — two-browser-tab local play driven by **`BattleHost`**: a connection-less poke-env player fed raw websocket frames captured from a browser tab, its `/choose` commands shipped back in. The decision pipeline is reused byte-for-byte with no socket of its own.
- **`online/bot.py`** — the same transport pointed at **play.pokemonshowdown.com**: it logs into a real account and plays the ladder from its control panel. It runs **up to five games at once**, paces every command under Showdown's chat throttle (a token-bucket send gate), accepts open team sheets, starts the battle timer, and recovers on its own: a link watchdog detects a dead socket, reloads, logs back in and rejoins live battles — ignoring the ad and tracker websockets the page also opens. A second account can run in parallel (`--account 2`).

A **serve bandit** (`play/serve_bandit.py`) decides which (battle net, picker) pair plays each game, per account — fixed shares for A/B tests, Thompson sampling otherwise, with a retire rule for a clearly worse arm. All transports record the benchmark data automatically, and a ladder recorder saves each game's trajectory for later analysis.

## 8. The adaptation layer

Static policies get exploited — our own benchmark proved it (the creator found a Wide Guard exploit in game 3). The counter-exploitation stack keeps the trained net frozen and adapts around it:

- **Serve-time pattern tilt** (`play/adapt_rules.py`): when the opponent shows a high-confidence repeated pattern (e.g. Wide Guard multiple turns running), a small logit bias tilts the policy toward single-target play. A tilt, not an override — the model still chooses, and an overwhelming preference survives.
- **Per-opponent dossiers** (`play/opponent_dossier.py`): every finished game updates a JSON dossier per opponent — revealed sets, items, abilities, W-L history — and can **warm-start the belief** in later games (`apply_dossier`, flag-gated): unknown items/abilities/moves fill from what that opponent showed before, with in-battle evidence always winning.
- **Best-of-3 set state** (`play/bo3_state.py`): games of a Bo3 are linked; brings, leads, and the opponent's shown Pokémon carry across games into the team picker's set-context input.

How well the whole stack resists a *learning* opponent is quantified by the **exploiter** described in the training section.

## 9. How a single turn actually works

The clearest way to understand the system is to follow one decision end-to-end (online browser mode; local play only differs in transport):

1. **A websocket frame arrives** in the browser tab (`|request|` — Showdown asking for our order). Playwright's `framereceived` hook pushes the raw text onto a queue; the consumer feeds it to `BattleHost.feed_async`, which routes it through **poke-env's own protocol dispatcher** on a background event loop. `BattleHost` holds a real `VGCPlayer` built with `start_listening=False` — the full production player with no socket — and monkey-patches its `send_message` so decisions are *captured* instead of sent.
2. **poke-env updates its battle model** and calls `choose_move(battle)`. The player first replays the public log prefix once to build the **opponent snapshot** — our own reconstruction of the opponent's side (poke-env's view plus belief estimates for everything still hidden), including the just-resolved previous turn for the in-game belief update.
3. **The belief fills the blanks**: unrevealed movesets/items/spreads come from Pikalytics priors, narrowed by everything observed so far (revealed moves, damage-consistent stat ranges, Choice-lock coherence) — or from the opponent's open team sheet when both sides accepted.
4. **The encoder writes the 6377-float state**: 12 Pokémon tokens (types, computed stats, status, boosts, item/ability *effect categories*, move features, protect counters, the Mega preview, …) + global field state, in the frozen v20 layout — byte-identical to what the trainer produced from parsed replays and self-play.
5. **Legality masks** are built per active slot (16 actions each) from Showdown's authoritative usable-move/switch lists — a disabled move or fainted bench slot is masked, so the net can only pick playable actions.
6. **One forward pass** of `AttnBCPolicy` yields per-slot action logits, Mega logits, and a win-probability. If the **adaptation layer** is on and the opponent has, say, Wide-Guarded two turns running, a small logit bias tilts spread moves down before the masked argmax. Cross-slot switch collisions are re-decoded (both slots can't switch into the same bench mon).
7. **The order is assembled and shipped** — `/choose move 1 2, move 3 1 mega` captured by the host, relayed into the tab's socket by the send gate. If Showdown *rejects* it (a rare mask desync), an escalation ladder retries with fresh legal actions and ultimately falls back to `/choose default` rather than hanging — every fallback is counted and surfaced, never silent.
8. **At battle end**, the recording hooks fire before state is reclaimed: a bench JSONL row (result, teams, opponent, both ratings), an HTML replay with the full log, a dossier update for that opponent, and the game's trajectory.

The property that makes the whole thing trustworthy is **train/serve parity**: the offline encoder (reading parsed snapshots) and the live encoder (reading poke-env objects) are held byte-identical by tests, the action codec is shared, and checkpoint loading refuses any layout mismatch. When the model plays badly, it's a *model* problem — not a silent skew between what it saw in training and what it sees live.

---

## Results

### On the ladder — Regulation M-C (October 2026)

Every model that served on the M-C ladder, measured the same way — the ladder performance rating over its own games (§6):

| Model | How it was trained | Ladder games | Performance rating |
|---|---|---|---|
| `era2` | behavior cloning + offline advantage weighting | 639 | 1237 ± 14 |
| `lg2_g50` | self-play league with behavior-cloned human opponents | 1,158 | 1243 ± 11 |
| `tploop_g39` | + the team picker training in the loop | 862 | 1401 ± 12 |
| **`megahold_d39`** (shipped) | + drills on the ladder's worst matchups + Mega-hold exploration | **1,008** | **1439 ± 12** (1459 ± 12 once a reconnect bug was fixed) |

- **Top 500 twice** — seen at **#172** (Elo 1679), session peak Elo **1744**. Because the ladder Elo swings ~±60 around the true level, those were streaks above a ~1460 player — staying there is the next goal (see the last section).
- **Like for like**, against opponents rated 1300–1500: the shipped pair won **53.8%** of 621 games, the previous one 50.0% of 644.
- **Head-to-head against the behavior-cloned model** in local self-play (400-game blocks): the shipped pair wins **88%** with its own team and **60%** when both sides draw random teams from the pool — and its worst ladder matchups of the previous generation, Rillaboom and Mega Raichu teams, rose from 66% to 88% and from 76% to 94%.

### Earlier — the behavior-cloning era (Regulations M-A / M-B)

- **Data was the lever that worked first.** Val top-1 on held-out human decisions: **0.585** (6.6k battles) → **0.595** (+11k) → **0.608** (+30k) — and then **flat** at the full 69k, isolating a *data-quality ceiling* rather than a volume or capacity limit. The value head kept improving (Brier 0.26 → 0.19).
- **Offline advantage weighting** added a targeted endgame improvement (+2.9pp on turn-11+ decisions) at zero cost elsewhere.
- **The era-retrain cycle** (winner's-perspective retraining on the bot's own games plus a Pikalytics × observed-ladder belief blend) lifted the elo-adjusted performance rating from **1212 to 1271** in a controlled 50-game block.
- **The contrastive team-preview set head** cleared its offline gate at **+3.8pp bring-set accuracy** over the greedy decode, then delivered **1425 elo-adjusted performance** (peak rating 1499) in its measured block — +154 over the same battle net with the old preview decode, against opponents ~170 elo stronger.
- **First recorded human benchmark: the bot beat its creator 4–1** (best team vs best counter-effort). The one loss came from a discovered exploit, which did not keep paying in the following games and which the adaptation layer now addresses directly.

Underneath all of it: a production-grade deployment stack — three serving transports plus a single control UI, checkpoint hot-swap with a live deploy-parity check, automatic benchmark/dossier recording — and a **3,000+ test** suite with byte-parity guards between training and serving.

## Acknowledgements & prior art

This project stands on excellent prior work and open resources:

- **[Foul Play](https://github.com/pmariglia/foul-play)** by **pmariglia** — the original inspiration. A search-based Showdown battle bot that has reached **#1 on the official Pokémon Showdown ladder**, proving a bot could compete at the top of real human play. This project began by studying its design; early scaffolding was learned from (and eventually rewritten past) its approach, and the ambition — *real ladder play against real humans* — comes straight from it.
- **[VGC-Bench](https://github.com/cameronangliss/vgc-bench)** — Angliss et al., *"VGC-Bench: A Benchmark for Generalizing Across Diverse Team Strategies in Competitive Pokémon"* ([arXiv:2506.10326](https://arxiv.org/abs/2506.10326), MIT license). Three enormous contributions to this project: the **open battle-log dataset** ([`cameronangliss/vgc-battle-logs`](https://huggingface.co/datasets/cameronangliss/vgc-battle-logs)) whose Champions-format subset became ~90% of our training corpus; the **entity-transformer architecture reference** our battle net parallels; and the **scientific grounding** — their measured results on team-count generalization collapse and universal exploitability told us which walls were real before we spent months on them.
- **[Metamon](https://github.com/UT-Austin-RPL/metamon)** — Grigsby et al., *"Human-Level Competitive Pokémon via Scalable Offline Reinforcement Learning"* ([arXiv:2504.04395](https://arxiv.org/abs/2504.04395)). The template for our offline stage: BC-anchored, advantage-weighted learning on a single GPU.
- **[ps-ppo](https://github.com/Nebraskinator/ps-ppo)** by **Nebraskinator** — PPO self-play on Showdown at scale. Its training recipe informed ours, and our optional server-side battle spawner (`selfplay/showdown_plugins/rlspawn.ts`) is a port of its plugin (MIT license).
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
│   ├── dex/                # shared dex utilities: pokedex, team_sheet
│   ├── parser/             # Showdown logs -> per-turn transitions; belief_state, match_belief
│   ├── encoders/           # snapshot -> 6377-dim state (layout v20)
│   ├── models/             # AttnBCPolicy battle net + SBDA team-preview net
│   ├── training/           # BC trainer, streaming memmap cache, advantage weights
│   ├── rl/                 # PPO core: schema, collector, store, reward, GAE, actor-critic, trainer
│   ├── selfplay/           # the league: generation loop, gates, clones, drills, Mega-hold,
│   │                       #   picker in the loop, multiprocess collection, preflight, exploiter
│   ├── ladder/             # learning from ladder games (recorder, update; paused)
│   ├── play/               # the serve core: model_io, player, vgc_base, adapt_rules, dossiers, bandit
│   ├── online/             # the ladder bot, its control panel, the send gate, the browser transport
│   ├── eval/               # rulers, human-benchmark report, gauntlet + Elo, panel eval, leak reports, probes
│   ├── datatools/          # data prep, team generator/builder tools, HF ingest, corpus QA, scrapers/
│   └── ui/                 # Mission Control :8990, dashboard :5175, team builder :5174 + static/ assets
├── teams/Champions/        # team pastes per regulation (every harness discovers them)
├── data/                   # dex data, Pikalytics usage, prepared training corpora
├── tests/                  # pytest suite: byte-parity, legality, QA gates, unit tests
├── ai_train_scripts/       # model checkpoints (local only — not in the repository)
└── artifacts/              # run logs, benchmark records, dossiers, replays (local only)
```

## Building & specializing teams

A crucial design property: **the battle net is team-agnostic.** Because the encoder uses mechanics (types, computed stats, item/ability *effect* categories) and never species identities, the *same* checkpoint pilots any Champions-doubles team. The shipped pair is additionally **specialized by self-play** on its own team — `--own-team Baltimore_Sand_Psy` puts that team on the learner's seat in every training game while the opponent side stays general — so it plays that team best, and it still plays any legal team.

1. **Get a legal paste.** Either write a Showdown export and drop it as a file in `teams/Champions/<regulation>/`, or generate one in the **team-builder** (Mission Control → *Teams*, or `python -m v_dance.ui.team_builder_server` → `http://localhost:5174/`). The generator grows a roster by belief-driven beam search over teammate co-occurrence, fills each set from usage stats, and **scores** it three ways (archetype coherence, team-preview-net confidence, and a corpus matchup prior). Every generated or pasted team is checked by **Showdown's own validator**, so illegal teams are dropped, not silently played.
   - ⚠ Champions uses the **0–32 stat-point budget** (66 total), *not* classic 0–252 EVs, and enforces the VGC **Item Clause** (one of each item). The generator and validator handle both; hand-written pastes must too.
2. **Point the bot at it** — any of: set `VD_DEFAULT_TEAM=<name>` (Mission Control → *Deploy*), pass `--ai-team <name>` to any play harness, or pin it live in the Online-bot tab's team dropdown.
3. **Specialize on it (optional)** — run self-play with `--own-team <name>` (see [Training & evaluation commands](#training--evaluation-commands)).

Every harness auto-discovers the pool under `teams/Champions/`, so a new file is usable immediately.

## Setup: playing locally

**Prerequisites:** Python 3.11+ (venv recommended), Node.js, and a local Pokémon Showdown checkout that includes the Champions-format mod (exact pinned commits for Showdown and poke-env are in `PINS.md` — the encoder's enum-name mapping and the format both depend on them). Model checkpoints are **not** part of the repository: train your own or point the flags below at checkpoints you have. The quickest path to *any* of the workflows below — play, train, evaluate — is **Mission Control** (`python -m v_dance.ui.mission_control`); the explicit commands are given here so you know what each button runs.

```bash
# 1. install the package + the pinned server
pip install -e .
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown && git checkout <commit from PINS.md> && npm install && cd ..
```

Teams live in `teams/Champions/<regulation>/` as Showdown paste files — drop any team you want the bot (or you) to use there; every harness discovers the pool automatically.

**Two-tab browser flow.** One command starts the server and opens two logged-in browser tabs with every pool team pre-imported into both Teambuilders. You challenge the AI from your tab; it auto-accepts and plays. The AI's team is `--ai-team <name>` if pinned, else whichever team you have *open* in the AI tab's Teambuilder, else random:

```bash
# The battle + team-preview checkpoints default to the deployed pair (model_io + .env), so you
# can omit --ckpt/--tp-ckpt entirely; they're shown here only to make the override explicit.
python -m v_dance.online.play_vs_human_browser --ai-team Baltimore_Sand_Psy \
    --ckpt ai_train_scripts/BC_model/<checkpoint>/battle_base.pt \
    --tp-ckpt ai_train_scripts/teamPreview_model/<checkpoint>/teampreview_sbda.pt \
    --adapt-rules --bench-note my_session
```

Every finished game is recorded automatically (result/teams/turns → `artifacts/human_benchmark/human_bench.jsonl`, a playable HTML replay, and a per-opponent dossier). Read the results any time:

```bash
python -m v_dance.eval.human_benchmark_report        # win rates + exploitability curve
python -m v_dance.play.opponent_dossier               # what the bot knows about each opponent
```

## Setup: playing online

Online play runs through a real browser logged into a real account. The bot plays the ladder from its own control panel (or Mission Control's *Online bot* tab), and incoming challenges in the configured format are auto-accepted.

1. **Create `.env` in the repo root** (gitignored — never commit it):

```ini
PS_USERNAME=YourRegisteredAccount
PS_PASSWORD=...
PS_AVATAR=cynthia                       # optional
PS_CLIENT_URL=https://play.pokemonshowdown.com
VDANCE_BATTLE_FORMAT=gen9championsvgc2026regmc          # exported stack-wide at startup
VD_BATTLE_CKPT=ai_train_scripts/BC_model/<checkpoint>/battle_base.pt
VD_TP_CKPT=ai_train_scripts/teamPreview_model/<checkpoint>/teampreview_sbda.pt
VD_DEFAULT_TEAM=Baltimore_Sand_Psy
# a second ladder account (optional): PS2_USERNAME / PS2_PASSWORD / PS2_AVATAR
```

These `VD_*` deploy defaults must match the canonical checkpoints in `play/model_io.py` (Mission Control's Deploy tab shows a live ✓/✗ parity check). To serve several checkpoint pairs side by side — an A/B on the live ladder — list them as arms in a serve-bandit config; each game binds one arm's battle net *and* its paired picker.

2. **Dry-run first** — connects, logs in (scripted; if the login UI changes it falls back to "log in manually in the window" and waits), imports the team pool, and idles so you can verify everything without playing:

```bash
python -m v_dance.online.bot --dry-run
```

3. **Go live.** Drop `--dry-run`. Sensible first session: a couple of *unrated* challenge games before touching the rated ladder. All recording (bench rows **with ladder ratings**, replays, dossiers) is automatic; `--adapt-rules` enables the anti-exploit tilt and `--dossier` warm-starts the belief against opponents you've faced before.

```bash
python -m v_dance.online.bot --adapt-rules --dossier --bench-note online_v1
python -m v_dance.online.bot --account 2 --adapt-rules --dossier   # the second account, in parallel
```

Once it's live, the **control panel** comes up (its own local page, mirrored into Mission Control's *Online bot* tab) where you drive matchmaking without touching the browser: start a **ladder run of N rated games** (it keeps up to five games going and re-queues until the target is hit), toggle **auto-accept** for incoming challenges, send **private challenges** by username, pin the AI's team, and watch the live rating / W–L tally / activity feed.

## Training & evaluation commands

```bash
# imitation
python -m v_dance.training.train_bc --data <jsonl folders> --mmap-cache ...   # full corpus on 32 GB RAM
python -m v_dance.eval.bc_val_report --ckpt <base.pt> --ckpt <cand.pt> --held-out-slice

# behavior-cloned league opponents (the humans who beat the bot)
python -m v_dance.selfplay.build_clone --kind nemesis --reg regmb regmc --dry-run

# a self-play league run (a preflight runs first and aborts on any failure)
python -m v_dance.selfplay.generation --live \
    --ckpt <start battle_base.pt> --learner-tp <start teampreview_sbda.pt> --tp-learn \
    --own-team Baltimore_Sand_Psy --league-clones <clone checkpoints…> --clone-frac 0.3 \
    --kl-coef 0.05 --panel era2=<era2 battle_base.pt> \
    --generations 40 --games 1000 --hours 6 --collect-procs 4 --collect-async 2 \
    --archive artifacts/self_play_archive/<run name>

# where the ladder rating is lost
python -m v_dance.eval.loss_breakdown --team Baltimore_Sand_Psy --arm <arm>
python -m v_dance.eval.field_control_report --team Baltimore_Sand_Psy --arm <arm>

pytest tests -q                                                               # 3,000+ tests
```

## Tech stack

**Python** (PyTorch, NumPy) · **poke-env** for the Showdown protocol · **Node.js** Pokémon Showdown as the battle engine (local servers for self-play) · **Playwright** for the browser transports · spawn-multiprocessing for GIL-free battle collection · **pytest** (3,000+ tests).

---

## Appendix: research history (for technical readers)

> Everything above describes the shipped agent. This appendix is the R&D record behind it — the ideas that were built, measured with controlled A/Bs, and *retired on evidence*, plus the bugs that shaped the current design. It's kept because the negative results are, honestly, some of the project's most useful output.

### Negative results

Each of these was built properly, evaluated with controlled A/Bs and confidence intervals, and **retired on evidence** (most were later removed from the code):

| Idea | Verdict |
|---|---|
| Belief-weighted expectimax search over a white-box forward model | **Hurts** (38.6% vs argmax, CI [36.5, 40.8]) — an imperfect forward model drags the policy off-manifold |
| Early self-play (own snapshots + scripted anchors) | Plateaued at the BC anchor, later overfit to its own league (in-league Elo 2000+, flat on the ladder) — fixed by the changes in §5.3 |
| C51 distributional value head | No strength change; removed |
| Opponent-conditioned policy heads | No strength change; removed |
| Recurrent / frame-stack memory over the battle | No strength change, value head degraded; removed |
| Latent team-archetype conditioning (z) — run twice, second time with a clean 100%-labelled artifact | No gain either time; removed |
| Team-preview joint decode via subset-mask augmentation | Dose-response trade: the augmentation converts full-roster accuracy into partial-roster validity and cannot win in absolute terms |
| Serve-time stochastic sampling (τ=0.45) | Costs 5.5pp head-to-head — anti-predictability is a dial with a price, not a free win |

The pattern behind the imitation-era results: **at this corpus size and quality, the bottleneck is the demonstration data, not bolted-on architecture** — sharper estimates of the opponent don't convert into wins when the policy prior itself caps skill. That redirected the project toward data levers, offline improvement and serve-time adaptation, and finally toward self-play done carefully — the step that got the agent past the imitation ceiling. The one imitation-era architecture change that *did* win (the set-scoring preview head) fixed a **decode-expressiveness** gap (greedy per-mon ranking mathematically cannot represent "bring A or B, not both"), not a representation-learning one.

### Engineering journey: the bugs that shaped the design

The current architecture is scar tissue from real failures. A selection, because these taught more than the successes:

- **Illusion broke everything, repeatedly.** Zoroark's Illusion means the replay log *lies about which Pokémon is on the field* — damage attribution, faint attribution, targeting, even which side's bench a mon returns to. It took several iterations (species-clause cross-checks, doubles-parity fixes, switch dedup) before the parser survived a corpus scan clean. Lesson: in adversarial log formats, parse defensively and audit with invariants (`corpus_qa` counts illegal-under-mask decisions; the gate is zero).
- **The forward-model audit — and its punchline.** A depth-1 search over a hand-built battle simulator initially looked promising. A targeted audit found **seven** silent bugs in the simulator (Choice Scarf's ×1.5 unapplied, -ate ability retyping omitted, Intimidate-on-entry missed, spread moves hitting the caster's ally, Mega opponents simulated with base-forme stats…). We fixed all seven, re-ran the 2,000-game A/B — and search **still lost by 11 points**. The simulator wasn't the problem: evaluating a value head on *synthetic states it never trained on* was.
- **The learner that never used its team picker.** For months every self-play learner brought the first four Pokémon of its roster while the ladder bot used the trained picker — so self-play practiced brings the bot never played, and the picker never got a learning signal. One measurement made it visible: the same battle net won 77.5% with first-four brings but 62.7% with the served picker. Wiring the picker into the loop is what produced the first big ladder gain (`tploop_g39`).
- **Reconnects that weren't.** The ladder bot started reloading its page several times an hour, and the losses that followed looked like a network problem. Every one of 89 "socket closed" events turned out to be an **ad iframe's websocket** closing, which the link watchdog had taken for the Showdown connection; sessions without that ad had zero. The fix ignores any socket that isn't Showdown's.
- **The retry storm.** A deterministic policy whose order Showdown rejects will submit the same illegal order forever. Worse, some legal situations are *unrepresentable* in a 16-action codec (Struggle, recharge turns). The fix is an escalation ladder — perturb to fresh legal actions, detect exhaustion, fall back to `/choose default` — with every non-model decision *counted and reported*, because a bot that silently plays random moves invalidates your evaluation without telling you.
- **27 GB of features vs 32 GB of RAM.** The full corpus encodes to a matrix that simply does not fit in memory next to PyTorch. Solution: build the encoded cache in streaming 2,000-file chunks appended to disk, then memory-map it read-only at train time with a lazy dataset (one-row copy per `__getitem__`). Corollary lesson: Windows' "96% memory used" during mmap training is *reclaimable page cache*, not leak — we verified commit charge before panicking.
- **Windows asyncio, three separate times.** Ctrl-C is not delivered to a parked `select()` (fix: accept loops wake every 250 ms); Playwright needs the Proactor loop while poke-env's background loop must stay Selector (fix: two loops, bridged with `run_coroutine_threadsafe`); and a `cp1252` console killed two multi-hour training launches on a *cosmetic box-drawing header* (fix: UTF-8 forced in launch scripts). Long commands are now shipped as verified `.sh` scripts after a shell paste mangled a flag.
- **The browser transport's event-loop wedge.** Feeding a captured frame for an *already-finished* battle made poke-env block forever waiting for a battle object that would never come — freezing the whole session. Late frames are real (Showdown re-sends room logs on rejoin). The host now tracks ended battles and parks or drops their strays, bounded so nothing can leak.
- **Measure twice: the z confound.** The first latent-archetype run looked like a −1.2pp regression — until we noticed 80% of training examples had been assigned the UNKNOWN archetype. A rebuilt artifact and a rerun gave z a *fair* trial. Verdicts are only verdicts when the wiring is verified.

---

## Where the remaining gap is

**The imitation ceiling was real.** Behavior cloning trains the policy to reproduce the human action, so its optimum *is* the human policy in the data — and cloning a mixture of players converges to their **average**. This corpus averages ~1600 elo, and validation accuracy went flat at the full dataset. Advantage weighting and serve-time adaptation climbed a little above that mean, but no amount of same-quality data or bolted-on architecture moved it further.

**Self-play got past it — on one desktop GPU.** The league in §5.3 is the lever datacenter-scale projects used, scaled down: behavior-cloned human opponents in place of a large population, a fixed panel in place of a sea of evaluators, and drills aimed at measured ladder leaks in place of brute-force coverage. It lifted the ladder performance rating by ~200 points over the behavior-cloned model and put the bot in the top 500.

**The next wall: staying in the top 500.** In October 2026 the top-500 line on the M-C ladder sat at about **1605 Elo**, and the bot plays at about **1460**. Because the ladder Elo swings ~±60 around a player's level, the bot visits the top 500 on streaks; to sit above the line ~90% of the time it needs a level of ~**1670** — about **+215**. A leak review of 1,008 ladder games found that bringing every losing matchup (rain with Archaludon above all) up to the bot's own average would add only ~+30, so most of the remaining gain has to be **general strength**, not patched matchups. The candidate levers, roughly in order of cost:

1. **Reward shaping that teaches what the ladder punishes** — the opt-in reward v2 (weather/terrain control credit + a loss margin) is under test now.
2. **Higher-quality data** — top-cut / tournament / high-ladder replays raise the imitation prior the league starts from, at zero extra compute.
3. **More and more diverse league opponents** — the clones were the change that made self-play transfer; more of them (and stronger ones) is the direct extension.
4. **On-manifold search** — lookahead only helps if evaluation stays on the training distribution, which means a **learned** world model (MuZero-style), not the hand-built simulator that measurably *hurt* here. A large build.
5. **Opponent modeling & adaptation** — beating strong humans partly by exploiting *their* predictability; the adaptation layer already does a little of this.

**The hardware wall.** This entire project — training, self-play, evaluation — runs on a **single RTX 3070 Ti (8 GB VRAM) + 32 GB RAM, one heavy run at a time, on a Windows workstation.** That budget shaped every decision in this README: the 2.4M-parameter net, the streaming memmap loader, multiprocess self-play on CPU with the PPO update on the GPU, and league runs sized to finish overnight. For contrast, the approaches that reach or approach superhuman play were datacenter-scale: VGC-Bench's PPO league used 8×A40 GPUs (and still produced ~100%-exploitable agents), and AlphaStar / OpenAI Five ran on large clusters for weeks.
