# Next-session handoff — Victory-Dance VGC bot: GATE v2 (frozen-champion ladder) — Phase 1 DONE → Phase 2 (HoF)

You are Opus 4.x continuing the Victory-Dance VGC Pokémon-Showdown self-play bot, in **ULTRACODE mode**
(author/run Workflows for substantive review/design/root-cause; adversarially verify findings; token cost is
NOT a constraint — optimize for the most correct, exhaustive answer). Solo only on conversational/trivial turns.

## STEP 0 — read auto-memory FIRST
Read `C:\Users\death\.claude\projects\D--ShowdownProject-Victory-Dance\memory\MEMORY.md`, then the notes it points to.
**THE RESUME POINTER for the CURRENT work: `gate-redesign-2026-06-17` — READ FULLY** (the gate v2 design, decisions,
and the P0/P1.1–P1.4 progress log). Also read `ppo-reward-design-2026-06-16` (the RL bible, design in
`docs/ppo_reward_design.md` §1–20). Skim `victory-dance-project-layout`, `showdown-reg-update-pending-2026-06`.

## ⇒ THE USER's REQUIRED CADENCE — follow EXACTLY (this is how the WHOLE build has gone)
- **After EVERY message, reprint the FULL UPDATED TO-DO LIST** as a table with **Effort (S/M/L) · Priority ·
  Depends-on · Status** columns. NON-NEGOTIABLE — every single turn, even a one-line reply. (The to-do list is in
  the §"Full to-do list" below — carry it forward and keep it current.)
- **Sub-problems, ONE at a time. PAUSE after each** — report what you did + the EVIDENCE (test counts, the actual
  dry-run/sim output, the gauntlet numbers), then WAIT. Don't chain sub-problems unless the user batches several in
  one instruction — then do them in sequence and test after each, pausing at the end.
- **Unit-test every change. Keep ALL tests green — currently 872 pass / 0 skip:** `python -m pytest tests -q`
  (venv active; from repo root). Add a REGRESSION TEST for every bug.
- **Give a MANUAL test for each feature:** an OFFLINE demo (a focused pytest / `--dry-run` / `--demo` / tiny script)
  AND, for live features, the EXACT command + what-to-look-for (pass criteria). The user likes running things.
- **The USER runs the long live stuff** (overnight generation, full gauntlets, run_local_battle). **You MAY run
  bounded live smokes yourself when granted** — ALWAYS wrap in a hard `timeout`/`--hours` cap and redirect to a log
  you can read. ⚠ A past incident wasted 3 HOURS on a stuck run — NEVER run an unbounded live command.
- **⚠ LIVE INTEGRATION IS THE BUG-PRONE SURFACE — the user's live smokes have caught ~8 real bugs offline tests
  missed.** ALWAYS hand the user a live smoke after a live-touching change; on a surfaced bug, fix + regression test +
  diagnostic. (This session's gate v2 passed its live smoke clean: 0 mask-desync/"Can't pass"/switch-collision/errors.)
- **Do NOT retrain / re-export without explicit permission.** When permitted, prefer the CHEAPEST correct path.
- **Checkpoint progress into the `gate-redesign-2026-06-17` memory note** when meaningful work lands; update the
  `MEMORY.md` index hooks. **The USER commits via GitHub Desktop — suggest logical commit groups, do NOT commit yourself.**
- **Don't be afraid to create new files / split long modules** (user's explicit ask). Example done this session: the
  gate logic moved to `gate.py`. The NEXT good extraction = the shared **parallel-battle runner** (collection + eval),
  to do WITH the multiprocessing work (#13/#14).

## ⇐ WHERE WE LEFT OFF (end of this session) ⇒ NEXT = Phase 2 (HoF anti-cycle)
The promotion gate was REDESIGNED this session into the **v2 FROZEN-CHAMPION LADDER** (the user's "static until
proven" design, calibrated on a Monte-Carlo sim). **Phase 0 + Phase 1 (P1.1–P1.4) are DONE and VERIFIED — offline
(dry-run/sim/872 tests) AND a clean live smoke. 872 pass / 0 skip. ALL UNCOMMITTED on `dev`** (suggest commit groups).

**THE v2 GATE (in `v_dance/selfplay/gate.py`, `promotion_gate_v2` + `GateConfigV2`):** the champion stays FROZEN
until the candidate either **(a) clears a HIGH bar — observed mirror win-rate ≥ 0.70 over ≥ `min_h2h_games`=200
games (→ `beat_champion`)**, OR **(b) the head-to-head climb PLATEAUS** at a not-losing level (windowed `is_plateau`
detector → `plateau_reanchor` backstop). A scripted **COLLAPSE → REVERT** is the safety net (highest priority), using
a Wilson-lower-bound `scripted_high_water` floor (never regresses). Calibration (`gate_sim`) showed the **mirror needs
~240 games** to be reliable (at 60 the bar leaks + the detector false-fires) → `--mirror-battles 240`.

**WHY frozen-champion (user's insight, corrects an earlier framing):** freezing the champion breaks the Nash-50%
trap — an improving policy pulls ahead of a STATIC benchmark, so h2h vs the frozen champion can climb past 70%.

**This session shipped (all green + live-verified):**
- **Phase 0** (current-code bugs the red-team found): h2h SE was double-counting variance (`_two_prop_se(p,n,0.5,n)` →
  `sqrt(0.25/n)`); `OpponentLeague.reset_pfsp()` on champion change; per-gen eval `matchup_seed=seed+gen`;
  `PPOTrainer.reset_optimizers()` on revert; deleted dead `refresh_phi` (static BC anchor) + a relaxing
  `target_kl_for_generation` schedule (`--target-kl-relax`, default off).
- **gate_sim** (`v_dance/selfplay/gate_sim.py`, `--demo`): Monte-Carlo promote-rate curves + sawtooth freeze/valve +
  `simulate_frozen_ladder` + the `is_plateau` detector. The calibration tool — extend it before trusting new thresholds.
- **P1.1** pure `promotion_gate_v2` + `GateConfigV2` (12 tests).
- **P1.2** EXTRACTED gate→`gate.py` (generation.py re-exports, −260 lines); `GenerationHistory` gained champion state
  (`scripted_high_water`, `h2h_history`, `champion_elo`, `record_h2h`/`advance_champion`); `run_generation` uses v2
  (record_h2h BEFORE the gate; advance_champion resets); live mirror bump (`run_gauntlet`/`gauntlet_eval`
  `mirror_battles` + `--mirror-battles 240`); resume back-compat; `_dry_run` demonstrates the full ladder.
- **P1.3** DECOUPLED competence-gated admission — admit EVERY competent gen (verdict≠revert) as a league snapshot
  (`is_champion=promoted`), not just promotes (PFSP diversity). `OpponentLeague.prune(cap,keep_recent)` = diversity-aware
  eviction (keep champions + recent + a generation-strided spread; soft cap) + `cleanup_fn` deletes evicted files.
  `GenConfig.league_cap=20/keep_recent=6`.
- **P1.4** OBSERVABILITY — champion-LINEAGE Elo (non-saturating; steps `+400·log10(p/(1-p))` per promote, vs the
  saturating scripted `model_elo`); manifest first-class `champion_path/champion_generation/champion_elo` +
  `best_generation`=champion (dashboard stars the REAL champion, not argmax-scripted — live-confirmed: G2 had higher
  Elo but G0 kept the ★ because G2 lost the h2h); dashboard.js features the lineage Elo; `operator_alert(history)`
  watchdog (3 reverts=collapse loop / 25-gen stall) printed each live gen.

**LIVE SMOKE RESULT (this session, clean):** gen0 PROMOTE(no_baseline)+champElo seeded; gen1–3 HOLD (h2h 44–52%, under
70% — correct); `prev_best X/48` (mirror bump); league 1→4 all competent gens admitted; **0** mask-desync/"Can't pass"/
"can only switch in once"/no-model/errors/retry; dashboard ★ on the champion. The G2-vs-G0 case is the best live proof
the v2 gate works (scripted Elo ≠ real strength).

## ⇒ IMMEDIATE NEXT
1. **Phase 2 — HoF anti-cycle** (the only Phase-2 item; gate-design §). Add a Hall-of-Fame check so a champion can't
   advance by LATERALLY beating one anchor while losing to orthogonal archetypes (RPS). Design (from the red-team):
   **cluster-stratified WORST-CASE** (beat ≥50% within EACH archetype cluster, not the average/majority), **conditional
   on a mirror-promote** (don't pay HoF cost on a HOLD), **exclude the champion's own lineage** from the sample. REUSE
   the existing bounded PFSP league as the HoF pool (don't rebuild). Calibrate via `gate_sim` first; test after each step.
2. **Commit** the gate work (user does it; suggest groups — see memory note: A) gate v2+sim+mirror bump,
   B) P1.3 admission + P1.4 observability).
3. **Overnight run** (deferred — "no longer night"): fresh from v4 BC, `--mirror-battles 240`, watch champElo climb /
   the gate HOLD then PROMOTE(beat_champion)/BACKSTOP(plateau). Clear `artifacts/self_play_archive/` first (smoke left
   gen0–3 + resume.pt) or `--resume` it.

## Full to-do list (carry forward + REPRINT/UPDATE every message)
| # | Task | Effort | Priority | Depends-on | Status |
|---|------|--------|----------|------------|--------|
| 1 | Sneasler mask-desync escape | M | P0 | — | ✅ done · live-verified |
| 2 | Wizard `prev_best` toggle | S | P1 | — | ✅ done · verified |
| 3 | Gate red-team (5-lens Workflow) | M | P1 | — | ✅ done |
| 4 | Phase 0 (6 current-code bug fixes) | M | P1 | #3 | ✅ done |
| 5 | Gate calibration sim + plateau detector | M | P1 | #3 | ✅ done |
| 6 | P1.1 pure v2 gate fn | M | P1 | #5 | ✅ done |
| 7 | P1.2 extract gate.py + wire v2 + mirror bump | L | P1 | #6 | ✅ done |
| 8 | P1.3 competence-gated bounded admission + eviction | M | P1 | #7 | ✅ done |
| 9 | P1.4 observability (lineage Elo, dashboard champion, abort) | M | P1 | #7 | ✅ done · live-verified |
| 10 | **Phase 2: HoF anti-cycle** (cluster-stratified worst-case, conditional on promote) | L | **P2 (NEXT)** | #8,#9 | ⬜ |
| 11 | Commit session gate work (suggest groups; USER commits) | S | P1 | — | ⬜ uncommitted |
| 12 | 🌙 Overnight training run (`--mirror-battles 240`) | — (user) | P2 | #10 (or now) | ⏸ deferred |
| 13 | Extract shared parallel-battle runner (multi-core prep) | M | P3 | — | ⬜ |
| 14 | 3c.8d true multiprocessing collection | L | P3 | #13 | ⬜ |
| 15 | v1.* TP co-development (§14) | L | P5 | archetype-competent policy (#12) | ⬜ |
| 16 | 3c.7d scripted demo episodes | M | conditional | live training | ⬜ deferred |
| 17 | Reg M-B migration | L | GATED ~2026-06-24 | ecosystem | ⬜ blocked |

## Environment + commands (Windows; user runs Git Bash + PowerShell; Bash tool available)
- venv active in the user's shell (`((.venv) )`) → `python` = venv python. Headless tool calls: use
  `.venv/Scripts/python.exe`. venv node at `.venv/node/node.exe` (`--check` a .js for syntax).
- **Run all commands from the REPO ROOT** `D:\ShowdownProject\Victory-Dance` (Bash default CWD is the
  `data/scripts/vod_parser` subdir → `cd /d/ShowdownProject/Victory-Dance` first; PowerShell tool defaults there too).
- Tests: `python -m pytest tests -q` (**872 pass / 0 skip**). `PYTHONIOENCODING=utf-8` when capturing stdout.
  **Log files contain unicode → `grep` needs `-a`.** Full-suite count via PowerShell `Select-Object -Last 3` (the
  Bash-tool tail can drop the summary line).
- **Gate calibration sim:** `python -m v_dance.selfplay.gate_sim --demo` (promote-rate curves + freeze/valve +
  frozen-ladder). EXTEND this to validate any new threshold/HoF logic BEFORE wiring live.
- **Dry-run (no server, demonstrates the v2 gate):** `python -m v_dance.selfplay.generation --dry-run --generations 8`
  → PROMOTE(no_baseline)→HOLD→PROMOTE(beat_champion, champElo steps)→REVERT(collapse); league grows on competent gens.
- **Live run flags:** `--live --generations 0 --hours 8 --games 300 --eval-battles <auto> --mirror-battles 240
  --max-cpu-fraction 0.5 --collect-workers 12 --max-vram-gb 4 --ckpt …checkpoints_v4/bc_best.pt`. Bare
  `python -m v_dance.selfplay.generation` = WIZARD. Dashboard: `python -m v_dance.datatools.dashboard_server --port 5175`.
- **Live-smoke recipe (bounded):** add `--generations 4 --games 40 --eval-battles 12 --mirror-battles 48 --hours 0.5 -v
  2> artifacts/logs/smoke.log`; then `grep -aE "would PASS an ACTIVE|Can't pass|can only switch in once|NO model loaded|
  REJECTED" artifacts/logs/smoke.log` MUST be 0.
- Gauntlet: `python -m v_dance.eval.gauntlet --ckpt …checkpoints_v4/bc_best.pt --workers 8`. Team validation:
  `python scratch/validate_teams.py`.

## Gotchas / standing facts
- **GATE = v2 frozen-champion ladder (sec 16), in `gate.py`.** `run_generation` uses `promotion_gate_v2` (NOT the
  legacy `promotion_gate`, which still exists/tested + is used by `gate_sim` fidelity). Verdicts: revert(collapse) →
  promote(`beat_champion` ≥70% over ≥200 mirror games) → promote(`plateau_reanchor`) → hold. `GenConfig` carries
  `gate_v2`, `league_cap`, `keep_recent`. `--no-prev-best` skips the mirror → gate can only hold/collapse (freeze).
- **gen 0 ALWAYS auto-promotes** (`no_baseline`, seeds the champion + lineage Elo). `--live` WITHOUT `--resume` starts
  FRESH from `--ckpt` BC. Trained models → `artifacts/self_play_archive/gen{N}.pt`; champion = `history.best_path`;
  NEVER written back to bc_best.pt. The smoke left gen0–3 + resume.pt in the archive — clear or `--resume`.
- **⚠ bc_best.pt CLOBBER:** production `…checkpoints/bc_best.pt` is gitignored + got overwritten with the v3 backup
  twice (user's manual `cp PRE_V4_BACKUP.pt`). **DURABLE v4 SOURCE = `…checkpoints_v4/bc_best.pt`.** If `bc_best.pt`
  ever reads `state_dim=1854`, re-promote from `checkpoints_v4`; or just pass `--ckpt …checkpoints_v4/bc_best.pt`.
- **STATE_DIM 1866, LAYOUT v4, ACTION_DIM 16, GIMMICK_DIM 2.** Production policy = base BC **v4**; TP = 46-dim
  (`teampreview_best.pt`); Tera = placeholder.
- **γ=0.997 FLOOR; PBRS OFF; reward = terminal ±1 only.** KL anchor = STATIC gen-0 BC (preserves rare tactics;
  `refresh_phi` deleted). Collection stochastic (tau annealed 1.3→1.0 over 12 gens). De-dup keeps only EXECUTED model
  decisions. ResourceBudget: GPU for PPO update only, collection on CPU (latency/GIL-bound; `--collect-workers` can
  exceed the CPU cap). User caps: 0.5 CPU (5900X→6) + 4 GB VRAM (3070 Ti).
- **ENV PINNED (`PINS.md`):** poke-env `@a6e4f67`, Showdown `@ecf39eef1`. ⚠ Reg M-B due ~2026-06-24 (gated on the
  ecosystem; stay on M-A; see [[showdown-reg-update-pending-2026-06]]).
- **The adversarial root-cause/red-team Workflow pattern** (path-tracers → refuters → completeness-critic; lens-diverse
  reviewers) is the project's tool for subtle bugs + design review — it found the gate's blocker + 6 real code bugs this
  session. Reuse it. Diagnostics: `tests/_parity_harness.py`, `scratch/_smoke_zoroark.py`, `scratch/validate_teams.py`.
