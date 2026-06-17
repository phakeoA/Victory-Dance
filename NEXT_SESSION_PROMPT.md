# Next-session handoff — Victory-Dance VGC bot: PPO self-play (3c.7 + 3c.8 DONE → overnight training → v1 TP)

You are Opus 4.x continuing the Victory-Dance VGC Pokémon-Showdown bot, in **ULTRACODE mode**
(author/run Workflows for substantive review/audit/research; adversarially verify findings; token cost
is not a constraint — optimize for the most correct, exhaustive answer). Solo on conversational/trivial turns.

## STEP 0 — read auto-memory FIRST
Read `C:\Users\death\.claude\projects\D--ShowdownProject-Victory-Dance\memory\MEMORY.md`, then the notes it points to.
**THE RESUME POINTER: `ppo-reward-design-2026-06-16` — READ FULLY.** Its PROGRESS section is the running log of
every sub-problem (3a → 3b → 3c.1–3c.8, the team-legality work, and the 5 live-caught bug fixes). The RL design
BIBLE is `docs/ppo_reward_design.md` §1–20 (§11 charter, §12 exploration, §14 TP, §15 teams, §16 generations,
§17 resume, §20 compute caps). Also skim `victory-dance-project-layout`, `showdown-reg-update-pending-2026-06`.

## ⇒ THE USER's REQUIRED CADENCE — follow EXACTLY (this is how the WHOLE build has gone)
- **Sub-problems, ONE at a time. PAUSE after each** — report what you did + the EVIDENCE (test counts, numbers),
  then WAIT for the user. Do NOT chain multiple sub-problems unless the user explicitly says so.
- **After EVERY message give an UPDATED TO-DO LIST** as a table with **Effort (S/M/L) · Priority · Depends-on ·
  Status** columns. Non-negotiable — the user relies on this every single turn.
- **Unit-test every change. Keep ALL tests green — currently 809 pass / 0 skip:**
  `.venv/Scripts/python.exe -m pytest tests -q` (from repo root; `testpaths=tests`, so bare `pytest` works).
  Add a REGRESSION TEST for every bug found.
- **Give a MANUAL test for each feature too:** an OFFLINE demo (`--dry-run` / a tiny script printing interpretable
  output) AND, for live features, the exact command + what-to-look-for (pass criteria). The user likes running
  things and re-running until clean.
- **The USER runs all the live stuff** (local Showdown server: the generation runner, gauntlet, run_local_battle,
  dashboard). Give the exact command + pass criteria. **Don't run live battles yourself.**
- **⚠ LIVE INTEGRATION IS THE BUG-PRONE SURFACE — the user's live smokes caught FIVE real bugs THIS session that
  the offline tests missed** (status.json crash, Ctrl-C ConnectionRefused flood, the no-op parallel collection, the
  mask-desync, the scripted-player loop). ALWAYS hand the user a live smoke after a live-touching change, and when
  one surfaces a bug: fix it + add a regression test + a diagnostic if needed.
- **Do NOT retrain / re-export without explicit permission.** All RL work so far is new code / serve-side.
- **Judge model STRENGTH on WIN-RATE (the gauntlet), not val top1.**
- **Checkpoint progress into the `ppo-reward-design-2026-06-16` memory note** when meaningful work lands; update the
  `MEMORY.md` index hooks too. **The USER commits via GitHub Desktop — suggest logical commit groups, do NOT commit yourself.**

## ⇐ WHERE WE LEFT OFF (end of 2026-06-17 session) ⇒ NEXT = the OVERNIGHT TRAINING RUN
**The entire self-play stack (3c.1–3c.8) + exploration seeding is DONE, fast, machine-capped, and every live bug the
user hit is fixed. 809 pass / 0 skip. ALL UNCOMMITTED on `dev`.** Run modules with `python -m v_dance.<...>`.

**THIS session shipped (all green):**
- **3c.7 exploration seeding** — 3c.7a KL-to-BC turned ON (`build_train_configs`, CLI `--kl-coef`/`--target-kl-bc`;
  fixed `--tau`→`PPOConfig.tau` mismatch); 3c.7b archetype-rich TRAIN pool (`discover_teams()` globs
  `teams/Champions/`; train=all 71 vs eval=curated 6-team `DEFAULT_EVAL_TEAMS`; `--eval-battles` auto-sizes to full
  side-balanced coverage); 3c.7c collection `tau` anneal (`tau_for_generation`, `--tau-start 1.3`→`--tau 1.0`/12 gens).
  `optional` Choice-lock/Encore mask = VERIFIED already done (the `available_moves` gate) + named regression test.
- **3c.8 throughput** — 3c.8a `ResourceBudget` caps (`v_dance/selfplay/resources.py`; CPU thread cap + games/min
  readout); 3c.8b GPU PPO-update hybrid (AC+optims on cuda; collection on CPU via `ActorCritic.inference_copy('cpu')`;
  VRAM enforce) — VERIFIED on real CUDA; 3c.8c **across-pairing parallel collection** (`build_collection_chunks` +
  `Semaphore(workers)`+`gather`) — **5.7× live (1.9→10.9 games/min)**; **eval also parallelized** (`run_gauntlet(n_workers)`);
  `--collect-workers` (collection is latency-bound → can exceed the CPU cap; ~30% CPU at 6); **INTERACTIVE WIZARD**
  (bare `python -m v_dance.selfplay.generation` prompts for params).
- **5 LIVE-CAUGHT BUG FIXES:** (1) `status.py` crash under parallel writes → `_atomic_write` retry+swallow +
  `LiveStatus(min_interval=0.5)` throttle (cosmetic feed never crashes training). (2) Win-Ctrl-C ConnectionRefused
  flood → `collect_with_league(stop_check=...)` + `start_showdown` `CREATE_NEW_PROCESS_GROUP`. (3) parallel-collection
  no-op (within-chunk) → rewrote to ACROSS-pairing. (4)+(5) mask desync: an active slot whose only usable order is a
  forced move the 16-action codec can't express (Struggle/recharge) → empty mask → illegal Pass / loop; fixed with
  `VGCPlayerBase._active_empty_mask(battle)` on the ROOT base → `/choose default` (covers model + scripted players).
  Diagnostics added: `_safe_order` logs `reason=` + `species=… mask_legal=…`.
- **TEAM LEGALITY:** `scratch/validate_teams.py` (poke-env pack → Showdown `validate-team`, authoritative). The
  71-team pool had 14 illegal in `gen9championsvgc2026regma` (format = all IVs must be 31 + ≤66 EV "stat points"/mon,
  ≤32/stat + Item Clause). Stripped `Tera Type:`+`IVs:` lines → **70/71 legal; only `Golurk` left** (Hatterene 67 stat
  pts: change `EVs: 32 HP / 3 Def / 32 SpA` → `2 Def`). The 6 eval teams are all legal.

## ⇒ IMMEDIATE NEXT (the user mostly DRIVES these — confirm + assist)
1. **Golurk** (last illegal team) — trim Hatterene by 1 EV (above), then `python scratch/validate_teams.py` → `ILLEGAL 0`.
2. **Launch the OVERNIGHT RUN** (two terminals): T1 `python -m v_dance.datatools.dashboard_server --port 5175`;
   T2 the wizard `python -m v_dance.selfplay.generation` (answer: Generations 0, Hours 8, Games 300, Eval blank,
   CPU 0.5, Collection workers 12, VRAM 4, **Resume N** = fresh from BC, Verbose Y). Open http://127.0.0.1:5175/.
   You'll see Gen 0 auto-PROMOTE then Gen 1+ with REAL gate verdicts + a rising Elo curve. Continue next night with **Resume Y**.
3. **After training accumulates:** judge on the gauntlet Elo. Then **v1 TP co-development** (§14 — alternating
   best-response, gauntlet-gated; needs 3c stable + archetype-competent battle policy). Build **3c.7d** (scripted demo
   episodes) ONLY if real training shows the rare tactics stay under-clicked.

## Full to-do list (carry this forward + UPDATE every message)
| # | Task | Effort | Priority | Status |
|---|---|---|---|---|
| 3c.7 (a/b/c + optional) | exploration seeding + Choice-lock/Encore | — | — | ✅ DONE |
| 3c.8 (a/b/c/eval/collect-workers/wizard) | ResourceBudget + GPU hybrid + parallel collection & eval + launcher | — | — | ✅ DONE |
| 5 live-bug fixes | status crash · Ctrl-C flood · parallel no-op · mask desync ×2 | — | — | ✅ DONE |
| team legality | validate_teams.py + IV/Tera strip → 70/71 legal | — | — | ✅ DONE |
| **Golurk team** | trim Hatterene 1 EV → 71/71 legal (then re-validate) | S | **P0 (user manual)** | ⬜ |
| **🌙 overnight run** | full 71-team, capped, resumable — the actual training | — | **NEXT (user runs)** | ⬜ ready |
| **gate upgrade (prev_best)** | add `prev_best` (best-self) anchor to the promotion eval — non-saturating + cycle-safe. The scripted ladder (random/max_damage/heuristic) goes BLIND once the policy crushes it (gen10≈gen20≈98% → HOLD forever = false plateau). NOT gen-N-vs-N-1 (non-transitivity → RPS cycle promotes forever going in circles). Fuller fix: gate on population league Elo over the whole snapshot history (infra EXISTS — league/PFSP/`model_elo`/`_make_opponent("prev_best")` — the gate just reads only the scripted win-rate). | M | recommended BEFORE the model saturates scripted | ⬜ |
| v1.* TP co-dev | alternating best-response, gauntlet-gated (§14) | L | P5 (after 3c stable + archetype-competent) | ⬜ |
| 3c.7d | scripted demo episodes (conditional, §12) | M | only if training shows rare tactics under-clicked | ⬜ deferred |
| 3c.8d | multiprocessing collection (true multi-core; async is GIL/Node-bound) | L | optional (only if want >linear-at-N workers) | ⬜ |
| polish | multi-battle Spectate (no flicker under concurrency) | S | optional | ⬜ |
| state-rep #A/#B | TP-net ability feature / Defiant-Competitive split | S | batch w/ next re-export+retrain | ⬜ |
| reg M-B | migration (format string + teams together) | L | **GATED ~2026-06-24** (ecosystem) | ⬜ blocked |

## Environment + commands (Windows; user runs Git Bash + PowerShell; Bash tool available)
- venv python (torch/poke-env): `.venv/Scripts/python.exe` — PATH `python` lacks ML deps. venv node at `.venv/node/node.exe`.
- Tests: `.venv/Scripts/python.exe -m pytest tests -q` (809 pass / 0 skip). `PYTHONIOENCODING=utf-8` when capturing
  stdout (cp1252 errors on unicode). **Log files contain unicode → `grep` needs `-a` (treats binary-detected file as text).**
- Interactive launcher: `python -m v_dance.selfplay.generation` (bare = wizard). Live run flags: `--live --generations 0
  --hours 8 --games 300 --max-cpu-fraction 0.5 --max-vram-gb 4 --collect-workers 12 [--resume artifacts/self_play_archive/resume.pt]`.
- Team validation: `python scratch/validate_teams.py` (re-run after editing any team; exit 1 = some illegal).
- Dashboard (live metrics + spectate): T1 `python -m v_dance.datatools.dashboard_server --port 5175`; T2 the live run.
- Offline demos: `python scratch/_demo_offline.py`, `python -m v_dance.selfplay.{league --demo, generation --dry-run}`.

## Gotchas / standing facts
- **ENV PINNED (`PINS.md`):** poke-env `@a6e4f67`, Showdown `@ecf39eef1`. ⚠ Reg M-B due ~2026-06-24 (gated on the
  ecosystem; stay on M-A; see [[showdown-reg-update-pending-2026-06]]).
- **STATE_DIM 1854, LAYOUT v3, ACTION_DIM 16, GIMMICK_DIM 2.** Production policy = base BC v3 at
  `ai_train_scripts/BC_model/checkpoints/bc_best.pt` (value+gimmick trained, dropout=0.1). Tera = placeholder.
- **Every `--live` run WITHOUT `--resume` starts FRESH from base BC** (`from_bc_checkpoint`); **gen 0 ALWAYS
  auto-promotes** (no_baseline); gen 1+ are really gated. Within ONE run the AC accumulates across gens (KL ref stays
  frozen BC). Trained models live in `artifacts/self_play_archive/gen{N}.pt` — NEVER written back to bc_best.pt.
- **ResourceBudget (sec 20):** GPU for the PPO UPDATE only; collection on CPU (per-game GPU transfer loses). Async
  collection is LATENCY-bound (one Node server) + GIL-bound → throughput up, NOT 80% CPU; `--collect-workers` can
  exceed the CPU cap. User caps: 0.5 CPU (5900X→6) + 4 GB VRAM (3070 Ti).
- **γ=0.997 FLOOR; PBRS OFF; reward = terminal ±1 only** (charter §11). **Collection runs stochastic (tau>0)** so the
  behaviour log-prob is real; the gimmick is argmax (v0 approximation). De-dup keeps only EXECUTED model decisions.
- **Teams:** `teams/Champions/M-A/` (M-B subfolder empty till the migration). `discover_teams()` globs `teams/Champions/`.
  Mask-desync diagnostics live in `vgc_base._safe_order` (logs `reason=`) + `_active_empty_mask` (root base).
- Reusable diagnostics: `tests/_parity_harness.py`, `scratch/_smoke_zoroark.py`, `_ab_headtohead.py`,
  `_diag_rejections.py`, `_demo_offline.py`, `scratch/validate_teams.py`.
