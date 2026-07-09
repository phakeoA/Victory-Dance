# Victory-Dance ![shiny Hisuian Zoroark](https://play.pokemonshowdown.com/sprites/ani-shiny/zoroark-hisui.gif)

**A deep-learning agent that plays competitive VGC doubles Pokémon on [Pokémon Showdown](https://pokemonshowdown.com/)** — trained by behavior cloning on ~86,000 human battles, improved with offline advantage-weighting, and served through a full deployment stack (local, browser-transport, and online) with a recorded human-benchmark protocol. In its first recorded benchmark set it beat its own creator 4–1.

Format: **Gen 9 "Pokémon Champions" VGC 2026** doubles (Regulations M-A / M-B). Mega Evolution is legal in this format; Terastallization is disabled by the mod.

> This is an educational research project. Its most valuable output, beyond the agent itself, is a set of **honestly-measured results** — including five carefully-run negative results that mirror what the academic literature found at 100× the compute.

---

## Table of contents
- [Why this problem is hard](#why-this-problem-is-hard)
- [System overview](#system-overview)
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
- [Negative results (reported with the same care)](#negative-results)
- [Engineering journey: the bugs that shaped the design](#engineering-journey-the-bugs-that-shaped-the-design)
- [Acknowledgements & prior art](#acknowledgements--prior-art)
- [Repository layout](#repository-layout)
- [Setup: playing locally](#setup-playing-locally)
- [Setup: playing online](#setup-playing-online)
- [Tech stack](#tech-stack)

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
        ├── Offline advantage weighting  w = exp(β·(outcome − V(s)))   ["imitate what won"]
        └── Latent team-archetype conditioning (k-means z over 31k teams)
        │
        ▼
   Serving: local harness · browser transport (BattleHost) · online (play.pokemonshowdown.com)
   + adaptation layer (pattern tilt, per-opponent dossiers)
   + human-benchmark recording (win rates, ratings, exploitability curve, replays)
```

Everything downstream of the encoder reads a **frozen, versioned feature layout** (`STATE_LAYOUT_VERSION = 19`, `STATE_DIM = 5057`) with load-time guards, so a layout change fails loudly instead of silently corrupting a trained net.

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

**`AttnBCPolicy`** (`models/bc_model_attn.py`, ~2.3M params): the flat state is reshaped in-model into **12 Pokémon tokens**; a shared per-mon encoder + learned slot embeddings feed a Transformer self-attention stack (the twelve Pokémon attend to one another — synergy and threat assessment become attention), plus a global field encoder. Heads: two own-action heads, two **auxiliary opponent-action heads**, gimmick heads, and a **win-probability value head**. Optional, flag-gated research extensions (opponent-conditioning, C51 distributional value, frame-stack memory, latent-z conditioning) are built in — and honestly evaluated (see negative results).

## 5. Training

- **Behavior cloning** (`training/train_bc.py`): masked per-slot action cross-entropy + value BCE + gimmick and auxiliary-opponent losses; re-encodes snapshots at train time.
- **Streaming memmap loader** (`training/encoded_cache.py`): the full corpus is a ~27 GB encoded matrix — far beyond a 32 GB workstation with PyTorch overhead — so the cache is built in streaming chunks and memory-mapped read-only at train time (one-row copies per item). The full-corpus retrain runs on a single RTX 3070 Ti + 32 GB RAM.
- **Offline advantage weighting** (Metamon's "exp" scheme): each decision is reweighted by `exp(β·(outcome − V(s)))` using the trained value head — shifting BC from "imitate everyone" to "imitate what beat expectation" without ever leaving the data manifold (the lesson of our search-hurts result).
- **Team-archetype latent z**: k-means over 31k team feature vectors (no species identity, same mechanic philosophy) → a per-battle archetype embedding, aimed at the "averaged mush" failure mode of one-policy-fits-all-styles.

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

All transports record the benchmark data automatically. A separate **team-preview network** (SBDA architecture with self/cross-attention over both rosters) picks the bring-4 and leads.

## 8. The adaptation layer

Static policies get exploited — our own benchmark proved it (the creator found a Wide Guard exploit in game 3). The counter-exploitation stack keeps the trained net frozen and adapts around it:

- **Serve-time pattern tilt** (`play/adapt_rules.py`): when the opponent shows a high-confidence repeated pattern (e.g. Wide Guard multiple turns running), a small logit bias tilts the policy toward single-target play. A tilt, not an override — the model still chooses, and an overwhelming preference survives.
- **Per-opponent dossiers** (`play/opponent_dossier.py`): every finished game updates a JSON dossier per opponent — revealed sets, items, abilities, W-L history — the substrate for cross-game adaptation (belief warm-starting between games of a set).
- Serve-side sampling, policy switching, and an RL exploiter (as a worst-case *metric*) are specced next, gated on benchmark evidence.

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

- **Data was the lever that worked.** Val top-1 on held-out human decisions: **0.585** (6.6k battles) → **0.595** (+11k) → **0.608** (+30k) — and then **flat** at the full 69k, isolating a *data-quality ceiling* (the available ladder replays average ~1600 Elo) rather than a volume or capacity limit. Doubling data of the same quality moved nothing; the value head kept improving (Brier 0.26 → 0.19).
- **Offline advantage weighting** added a targeted endgame improvement (+2.9pp on turn-11+ decisions, the game-deciding ones) at zero cost elsewhere — the deployed configuration.
- **First recorded human benchmark: the bot beat its creator 4–1** (best team vs best counter-effort). The one loss came from a discovered exploit — which did not keep paying in the following games, and which the adaptation layer now addresses directly.
- **A production-grade deployment stack**: three serving transports, checkpoint hot-swap flags, closed-team-sheet discipline matching the ladder, automatic benchmark/dossier recording, and a 1,100+ test suite with byte-parity guards between training and serving.

## Negative results

Each of these was built properly, evaluated with controlled A/Bs and confidence intervals, and **retired on evidence** — the same wall the field hit at datacenter scale (VGC-Bench's PPO league needed 8×A40 and still collapsed on generalization; every agent they trained was ~100% exploitable by a best-response):

| Idea | Verdict |
|---|---|
| Belief-weighted expectimax search over a white-box forward model | **Hurts** (38.6% vs argmax, CI [36.5, 40.8]) — an imperfect forward model drags the policy off-manifold |
| PPO-style league self-play from the BC anchor | Plateaued at the anchor's strength |
| C51 distributional value head | No strength change |
| Opponent-conditioned policy heads | No strength change |
| Recurrent / frame-stack memory over the battle | No strength change, value head degraded |
| Serve-time stochastic sampling (τ=0.45) | Costs 5.5pp head-to-head — anti-predictability is a dial with a price, not a free win |

The pattern behind all six: **at this corpus size and quality, the bottleneck is the demonstration data, not the architecture** — sharper estimates of the opponent don't convert into wins when the policy prior itself caps skill. That conclusion redirected the project toward data levers, offline improvement, and serve-time adaptation, which is where the wins came from.

---

## Engineering journey: the bugs that shaped the design

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
│   └── datatools/          # HF-dataset ingest, corpus QA, team archetypes, dashboards
├── data/                   # dex data, Pikalytics usage, teams, prepared training corpora
├── docs/                   # design docs + the execution playbook (audit, decisions, specs)
├── tests/                  # pytest suite: byte-parity, legality, QA gates, unit tests
├── ai_train_scripts/       # model checkpoints (battle + team-preview)
└── artifacts/              # run logs, benchmark records, dossiers, replays (gitignored)
```

## Setup: playing locally

**Prerequisites:** Python 3.11+ (venv recommended), Node.js, and a local Pokémon Showdown checkout that includes the Champions-format mod (exact pinned commits for Showdown and poke-env are in `PINS.md` — the encoder's enum-name mapping and the format both depend on them).

```bash
# 1. install the package + the pinned server
pip install -e .
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown && git checkout <commit from PINS.md> && npm install && cd ..
```

Teams live in `teams/Champions/<regulation>/` as Showdown paste files — drop any team you want the bot (or you) to use there; every harness discovers the pool automatically.

**Option A — two-tab browser flow (recommended).** One command starts the server and opens two logged-in browser tabs with every pool team pre-imported into both Teambuilders. You challenge the AI from your tab; it auto-accepts and plays. The AI's team is `--ai-team <name>` if pinned, else whichever team you have *open* in the AI tab's Teambuilder, else random:

```bash
python -m v_dance.play.play_vs_human_browser --ai-team maw_zard \
    --ckpt ai_train_scripts/BC_model/checkpoints_attn_adv/battle_base.pt \
    --tp-ckpt ai_train_scripts/teamPreview_model/checkpoints/teampreview_sbda_v7.pt \
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
VD_BATTLE_CKPT=ai_train_scripts/BC_model/checkpoints_attn_adv/battle_base.pt
VD_TP_CKPT=ai_train_scripts/teamPreview_model/checkpoints/teampreview_sbda_v7.pt
VD_DEFAULT_TEAM=maw_zard
```

2. **Dry-run first** — connects, logs in (scripted; if the login UI changes it falls back to "log in manually in the window" and waits), imports the team pool, and idles so you can verify everything without playing:

```bash
python -m v_dance.play.play_online_browser --dry-run
```

3. **Go live.** Drop `--dry-run`. Sensible first session: a couple of *unrated* challenge games before touching the rated ladder. All recording (bench rows **with ladder ratings**, replays, dossiers) is automatic; `--adapt-rules` enables the anti-exploit tilt.

```bash
python -m v_dance.play.play_online_browser --adapt-rules --bench-note online_v1
```

There is also a fully-autonomous direct-websocket harness (`play/play_ladder.py`) that searches ladder matches by itself on any server (`--server-url`, `--username/--password`, `--games N`) — the browser flow above is the supervised default.

**Training & evaluation** (the research side):

```bash
python -m v_dance.training.train_bc --data <jsonl folders> --mmap-cache ...   # full corpus on 32 GB RAM
python -m v_dance.eval.bc_val_report --ckpt <base.pt> --ckpt <cand.pt> --held-out-slice
pytest tests -q                                                               # 1,100+ tests
```

## Tech stack

**Python** (PyTorch, NumPy) · **poke-env** for the Showdown protocol · **Node.js** Pokémon Showdown as the battle engine · **Playwright** for the browser transports · spawn-multiprocessing for GIL-free battle collection · **pytest** (1,100+ tests).
