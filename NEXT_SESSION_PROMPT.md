# Next-session handoff — Victory-Dance VGC bot: state-rep #B + gate upgrade DONE → OVERNIGHT TRAINING RUN

You are Opus 4.x continuing the Victory-Dance VGC Pokémon-Showdown bot, in **ULTRACODE mode**
(author/run Workflows for substantive review/audit/root-cause; adversarially verify findings; token cost
is not a constraint — optimize for the most correct, exhaustive answer). Solo on conversational/trivial turns.

## STEP 0 — read auto-memory FIRST
Read `C:\Users\death\.claude\projects\D--ShowdownProject-Victory-Dance\memory\MEMORY.md`, then the notes it points to.
**THE RESUME POINTER: `ppo-reward-design-2026-06-16` — READ FULLY.** Its PROGRESS section is the running log of
every sub-problem (3a → 3b → 3c.1–3c.8, team legality, the live-caught bug fixes, the switch-collision fix, and the
state-rep #B + gate upgrade). The RL design BIBLE is `docs/ppo_reward_design.md` §1–20. Also skim
`victory-dance-project-layout`, `item-ability-staterep-2026-06-16`, `showdown-reg-update-pending-2026-06`.

## ⇒ THE USER's REQUIRED CADENCE — follow EXACTLY (this is how the WHOLE build has gone)
- **After EVERY message, reprint the FULL UPDATED TO-DO LIST** as a table with **Effort (S/M/L) · Priority ·
  Depends-on · Status** columns. NON-NEGOTIABLE — the user relies on this every single turn, even on a one-line reply.
- **Sub-problems, ONE at a time. PAUSE after each** — report what you did + the EVIDENCE (test counts, gauntlet
  numbers, the actual output), then WAIT. Do NOT chain multiple sub-problems unless the user explicitly says so
  (they sometimes batch several tasks in one instruction — then do them in sequence and pause at the end).
- **Unit-test every change. Keep ALL tests green — currently 830 pass / 0 skip:**
  `python -m pytest tests -q` (venv active; from repo root; `testpaths=tests`). Add a REGRESSION TEST for every bug.
- **Give a MANUAL test for each feature too:** an OFFLINE demo (a focused pytest file / `--dry-run` / tiny script)
  AND, for live features, the EXACT command + what-to-look-for (pass criteria). The user likes running things and
  re-running until clean.
- **The USER runs the long live stuff** (overnight generation runner, full gauntlets, run_local_battle). **You MAY
  run bounded live smokes yourself when the user grants it** (they did this session) — but ALWAYS wrap them in a hard
  `timeout` and redirect to a log file you can read even if the timer fires. ⚠ A past incident: a stuck run wasted
  3 HOURS with the board not advancing — so NEVER run an unbounded live command; cap it and log it.
- **⚠ LIVE INTEGRATION IS THE BUG-PRONE SURFACE — the user's live smokes have caught ~7 real bugs the offline tests
  missed** (status.json crash, Ctrl-C flood, no-op parallel collection, mask-desync ×2, the cross-slot switch
  collision). ALWAYS hand the user a live smoke after a live-touching change; when one surfaces a bug, fix it + add a
  regression test + a diagnostic.
- **Do NOT retrain / re-export without explicit permission.** When permitted, prefer the CHEAPEST correct path —
  e.g. this session a re-export turned out UNNECESSARY because `bc_dataset` lazy-encodes the snapshot at load time, so
  a pure encoding change needs only a RETRAIN (don't do destructive rewrites that add no value).
- **Judge model STRENGTH on WIN-RATE (the gauntlet), not val top1.** And run an A/B before shipping a state-rep
  feature — don't ship a non-improvement (see #A this session).
- **Checkpoint progress into the `ppo-reward-design-2026-06-16` memory note** when meaningful work lands; update the
  `MEMORY.md` index hooks too. **The USER commits via GitHub Desktop — suggest logical commit groups, do NOT commit yourself.**

## ⇐ WHERE WE LEFT OFF (end of 2026-06-17 session) ⇒ NEXT = the OVERNIGHT TRAINING RUN
Everything shipped + verified; **830 pass / 0 skip; committed on `dev`** (user committed via GitHub Desktop).
Production BC is **v4 (1866 / layout v4)**, gauntlet-confirmed. Run modules with `python -m v_dance.<...>` from the repo root.

**THIS session shipped (all green):**
- **Cross-slot SWITCH collision fix** (the live "slot N can only switch in once" rejection): an ORDER-LEVEL dedup —
  `vgc_base._switch_order_target` + a `taken_switch_targets` set threaded into the 2nd `_safe_order` in BOTH
  `choose_move` bodies (root `vgc_base` + spliced `live_vgc_base`). An adversarial Workflow proved the action-level
  `_select_actions` dedup (a0==a1) missed TWO emitters: the retry perturbation (`_fresh_legal`) and the
  under-illusion case where two MOVE actions both fall back to the same switch inside `_safe_order`. LIVE-VERIFIED
  (0 collisions; 100% model-driven at scale). `tests/test_switch_collision_dedup.py` (6).
- **State-rep #B — Defiant/Competitive split (SHIPPED):** new `statdrop_boost` ability-effect category
  (`NUM_ABILITY_EFFECTS 16→17`), `defiant`+`competitive` moved out of `reactive_boost`; STATE_DIM 1854→1866,
  STATE_LAYOUT_VERSION 3→4. **No re-export needed** (lazy-encode) → BC RETRAINED → `checkpoints_v4/bc_best.pt`
  (val top1 0.392 ≈ v3's 0.394; value win-acc 0.647; gimmick recall 0.964) → PROMOTED to production.
  **Gauntlet-confirmed: scripted 55–63%, Elo 1315–1425 (v3 was 1240), 100% model-driven, 100% TP-driven.**
  `tests/test_statdrop_ability_split.py` (6).
- **State-rep #A — TP-net ability feature (MEASURED → NOT shipped, reverted):** our replays have NO open team sheets
  (`|showteam|`=0), so a belief-most-likely-ability multi-hot was trialled in `teampreview_dataset.mon_dex_features`.
  A clean same-recipe/seed A/B gave **0.221 WITH vs 0.228 control** = no improvement (the species embedding already
  captures a species' typical ability). Reverted per the project's evidence-based standard; production TP unchanged
  (46-dim). Re-apply only if open-sheet data ever exists.
- **Gate upgrade — `prev_best` non-saturating anchor (SHIPPED, code-only):** `promotion_gate` gained
  `prevbest_wins/games`. Revert on scripted COLLAPSE first (safety); else PROMOTE if scripted improves OR the
  candidate beats `prev_best` head-to-head significantly; else HOLD. `prev_best` = the accepted BEST (NOT gen N-1 →
  no RPS cycle). Wiring: `gauntlet_eval(prev_best_path=...)` appends the `"prev_best"` mirror (gauntlet infra already
  existed; `model_elo` excludes it → Elo stays scripted-calibrated); `aggregate_prev_best`; `run_generation` passes
  `history.best_path` to `eval_fn(candidate, prev_best_path)`. LIVE-VERIFIED (2-gen smoke: gen1 ran `prev_best 2/4`
  → correctly HELD). `tests/test_prev_best_gate.py` (9).
- **Gen checkpoints relocated:** the old v3-based `artifacts/self_play_archive/*` (gen0–7 + resume) → moved to
  `artifacts/self_play_archive_PRE_V4/` (reversible); the active archive is cleared for a fresh v4 self-play run.

## ⇒ IMMEDIATE NEXT (the user mostly DRIVES these — confirm + assist)
1. **Launch the OVERNIGHT RUN** (two terminals): T1 `python -m v_dance.datatools.dashboard_server --port 5175`;
   T2 `python -m v_dance.selfplay.generation --ckpt ai_train_scripts/BC_model/checkpoints_v4/bc_best.pt
   --generations 0 --hours 8 --games 300 --max-cpu-fraction 0.5 --collect-workers 12 --max-vram-gb 4 -v`.
   Open http://127.0.0.1:5175/. **Watch:** gen 0 PROMOTE → rising Elo; each eval line shows `prev_best X/Y` and once
   scripted saturates you should see `verdict_reason=beats_prev_best` (the gate upgrade keeping it from a false
   plateau); **0** "can only switch in once". Continue next night by re-running (resume.pt in the archive).
2. **After training accumulates:** judge on the gauntlet Elo. Then **v1 TP co-development** (§14 — alternating
   best-response, gauntlet-gated; needs 3c stable + an archetype-competent battle policy). Build **3c.7d** (scripted
   demo episodes) ONLY if real training shows the rare tactics stay under-clicked.

## Full to-do list (carry this forward + REPRINT/UPDATE every message)
| # | Task | Effort | Priority | Status |
|---|------|--------|----------|--------|
| switch-collision fix | order-level cross-slot switch dedup + tests | — | — | ✅ done · live + gauntlet (100% model-driven) |
| state-rep #B | Defiant/Competitive split → statdrop_boost (v4) + BC retrain + promote | — | — | ✅ done · gauntlet-confirmed (55–63%, Elo 1315–1425 > v3 1240) |
| state-rep #A | TP-net belief-ability feature | — | — | ✅ measured (A/B) → REVERTED (no improvement) |
| gate upgrade | `prev_best` non-saturating, cycle-safe promotion anchor | — | — | ✅ done + live-verified + tests |
| session work commit | switch fix + #B/v4 + gate upgrade | — | — | ✅ committed on `dev` (GitHub Desktop) |
| **🌙 overnight run** | full 71-team, capped, resumable — the actual training (use `--ckpt …checkpoints_v4/bc_best.pt`) | — | **NEXT (user runs)** | ⬜ ready |
| v1.* TP co-dev | alternating best-response, gauntlet-gated (§14) | L | P5 (after 3c stable + archetype-competent) | ⬜ |
| 3c.7d | scripted demo episodes (conditional, §12) | M | only if rare tactics under-clicked | ⬜ deferred |
| 3c.8d | multiprocessing collection (true multi-core; async is GIL/Node-bound) | L | optional | ⬜ |
| polish | multi-battle Spectate (no flicker under concurrency) | S | optional | ⬜ |
| state-rep (future) | next encoder idea — batch w/ a future re-export+retrain; A/B before shipping | S | when one arises | ⬜ |
| reg M-B | migration (format string + teams together) | L | **GATED ~2026-06-24** (ecosystem) | ⬜ blocked |

## Environment + commands (Windows; user runs Git Bash + PowerShell; Bash tool available)
- venv is active in the user's shell (`((.venv) )`), so `python` = the venv python. Headless tool calls: use
  `.venv/Scripts/python.exe` (PATH `python` may lack ML deps). venv node at `.venv/node/node.exe`.
- **Run all commands from the REPO ROOT** `D:\ShowdownProject\Victory-Dance` (the Bash tool's default CWD is the
  `data/scripts/vod_parser` subdir — `cd /d/ShowdownProject/Victory-Dance` first; the PowerShell tool defaults to the
  subdir too, so use absolute paths there).
- Tests: `python -m pytest tests -q` (830 pass / 0 skip). `PYTHONIOENCODING=utf-8` when capturing stdout (cp1252
  errors on unicode). **Log files contain unicode → `grep` needs `-a` (treat binary-detected file as text).**
- Gauntlet (win-rate yardstick): `python -m v_dance.eval.gauntlet --ckpt ai_train_scripts/BC_model/checkpoints_v4/bc_best.pt --workers 8`.
  Pass = scripted >v3 (~47–50%), Elo >1240, MODEL-DRIVEN 100%, 0 "NO model loaded" / "can only switch in once".
- Live run flags: `--generations 0 --hours 8 --games 300 --max-cpu-fraction 0.5 --collect-workers 12 --max-vram-gb 4
  [--ckpt …checkpoints_v4/bc_best.pt] [--resume artifacts/self_play_archive/resume.pt]`. Bare `python -m
  v_dance.selfplay.generation` = interactive WIZARD.
- Dashboard: `python -m v_dance.datatools.dashboard_server --port 5175`.
- Offline demos (no server): `python scratch/_demo_offline.py`, `python -m v_dance.selfplay.{league --demo,
  generation --dry-run}`, `python -m v_dance.selfplay.archive`.
- Team validation: `python scratch/validate_teams.py` (re-run after editing any team; exit 1 = some illegal).

## Gotchas / standing facts
- **⚠ bc_best.pt CLOBBER (resolved):** production `ai_train_scripts/BC_model/checkpoints/bc_best.pt` is **gitignored**;
  it got overwritten with the v3 backup twice this session — the cause was the USER manually running the rollback
  `cp bc_best.PRE_V4_BACKUP.pt bc_best.pt` (NOT git/sync/scheduled-task/any process). v4 PASSED, so it never needed
  rolling back. **DURABLE v4 SOURCE = `ai_train_scripts/BC_model/checkpoints_v4/bc_best.pt`** (untouched). If
  `bc_best.pt` ever reads `state_dim=1854`: `cp ai_train_scripts/BC_model/checkpoints_v4/bc_best.pt
  ai_train_scripts/BC_model/checkpoints/bc_best.pt`. The overnight run's `--ckpt …checkpoints_v4/…` bypasses it.
- **STATE_DIM 1866, LAYOUT v4, ACTION_DIM 16, GIMMICK_DIM 2.** Production policy = base BC **v4** at
  `ai_train_scripts/BC_model/checkpoints/bc_best.pt` (value+gimmick trained, dropout 0.1; backup
  `bc_best.PRE_V4_BACKUP.pt` = v3). Production TP = `teamPreview_model/checkpoints/teampreview_best.pt` (46-dim,
  unchanged — #A not shipped). Tera = placeholder.
- **NO re-export was needed for #B** because `bc_dataset` encodes the input from the stored SNAPSHOT
  (`encode_snapshot(state_before_actions)`) at LOAD time (`state_vector` is `None` in the corpus). Any FUTURE
  encoding-only change is likewise retrain-only; a re-export is needed only if the SNAPSHOT or stored masks change
  (a parser change). Corpus: `data/vods/Prepared_training_data/Regulation_MA/Jsonl_TypeB` (3150 files); HTML source
  `data/vods/Type_B/gen9championsvgc2026regma`.
- **Retrain recipe (BC):** `python -m v_dance.training.train_bc --value-loss-weight 0.3 --epochs 30 --out
  <versioned dir>` (augment-move-order on by default; device cuda). TP: `python -m v_dance.training.train_teampreview
  --epochs 40 --out <versioned dir>`. Always train to a VERSIONED dir, validate, then promote with a backup.
- **Gate (sec 16):** `promotion_gate` now non-saturating — revert on scripted collapse first, else promote if scripted
  improves OR beats `prev_best` head-to-head, else hold. `prev_best` = accepted BEST (cycle-safe). It only kicks in
  once `history.best_path` exists (gen 1+); gen 0 auto-promotes.
- **Every `--live` run WITHOUT `--resume` starts FRESH from the `--ckpt` BC** (`from_bc_checkpoint`); **gen 0 ALWAYS
  auto-promotes** (no_baseline); gen 1+ are really gated. Within ONE run the AC accumulates across gens (KL ref stays
  frozen BC). Trained models live in `artifacts/self_play_archive/gen{N}.pt` — NEVER written back to bc_best.pt.
- **ResourceBudget (sec 20):** GPU for the PPO UPDATE only; collection on CPU. Async collection is LATENCY-bound
  (one Node server) + GIL-bound → throughput up, NOT 80% CPU; `--collect-workers` can exceed the CPU cap. User caps:
  0.5 CPU (5900X→6) + 4 GB VRAM (3070 Ti).
- **γ=0.997 FLOOR; PBRS OFF; reward = terminal ±1 only** (charter §11). Collection runs stochastic (tau>0, annealed
  1.3→1.0 over 12 gens) so the behaviour log-prob is real. De-dup keeps only EXECUTED model decisions
  (`_record_rl_decision` skips non-`_MODEL_SOURCES`).
- **ENV PINNED (`PINS.md`):** poke-env `@a6e4f67`, Showdown `@ecf39eef1`. ⚠ Reg M-B due ~2026-06-24 (gated on the
  ecosystem; stay on M-A; see [[showdown-reg-update-pending-2026-06]]).
- Reusable diagnostics: `tests/_parity_harness.py`, `scratch/_smoke_zoroark.py`, `scratch/validate_teams.py`,
  `scratch/_demo_offline.py`. Mask-desync diagnostics live in `vgc_base._safe_order` (logs `reason=`) +
  `_active_empty_mask` (root base). The adversarial root-cause Workflow pattern (path-tracers → refuters →
  completeness-critic) caught the switch-collision second emitter — reuse it for subtle live bugs.
