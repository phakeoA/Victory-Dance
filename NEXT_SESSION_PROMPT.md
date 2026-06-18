# Next-session handoff — Victory-Dance VGC bot: PHASE 2 (HoF anti-cycle) BUILD COMPLETE → P2.5 (user smoke) + commit

You are Opus 4.x continuing the Victory-Dance VGC Pokémon-Showdown self-play bot, in **ULTRACODE mode**
(author/run Workflows for substantive review/design/root-cause; adversarially verify findings; token cost is
NOT a constraint — optimize for the most correct, exhaustive answer). Solo only on conversational/trivial turns.

## STEP 0 — read auto-memory FIRST
Read `C:\Users\death\.claude\projects\D--ShowdownProject-Victory-Dance\memory\MEMORY.md`, then the notes it points to.
**THE RESUME POINTER for the CURRENT work: `gate-redesign-2026-06-17` — READ FULLY** (the gate v2 design, the Phase-2
HoF design + its champion pivot, and the P0/P1/P2 progress log). Also read `ppo-reward-design-2026-06-16` (the RL bible,
design in `docs/ppo_reward_design.md` §1–20) and **`docs/hof_anticycle_design.md`** (the full Phase-2 design + the
P2.1 calibration + the champion pivot). Skim `victory-dance-project-layout`, `showdown-reg-update-pending-2026-06`.

## ⇒ THE USER's REQUIRED CADENCE — follow EXACTLY (this is how the WHOLE build has gone)
- **After EVERY message, reprint the FULL UPDATED TO-DO LIST** as a table with **Effort (S/M/L) · Priority ·
  Depends-on · Status** columns. NON-NEGOTIABLE — every single turn, even a one-line reply. (The list is below — carry
  it forward and keep it current.) Keep the Phase-2 sub-breakdown table too while Phase 2 is live.
- **Sub-problems, ONE at a time. PAUSE after each** — report what you did + the EVIDENCE (test counts, the actual
  dry-run/sim output, the gauntlet/HoF numbers), then WAIT. Don't chain sub-problems unless the user batches several in
  one instruction — then do them in sequence and test after each, pausing at the end.
- **Unit-test every change. Keep ALL tests green — currently 901 pass / 0 skip:** `python -m pytest tests -q`
  (venv active; from repo root). Add a REGRESSION TEST for every bug.
  - ⚠ **Test-output gotcha (Windows):** pytest's trailing `N passed` summary line does NOT flush into a redirected
    stream here. Judge pass/fail by the **exit code** (`$LASTEXITCODE`/`EXIT=$?` == 0) + the all-dots progress (no
    `F`/`E`/`s`). To count: `([regex]::Matches($content,'\.')).Count` in PowerShell on the captured log.
- **Give a MANUAL test for each feature:** an OFFLINE demo (a focused pytest / `--dry-run` / `--demo` / tiny script)
  AND, for live features, the EXACT command + what-to-look-for (pass criteria). The user likes running things.
- **The USER runs the long live stuff** (overnight generation, full gauntlets, run_local_battle). **You MAY run
  bounded live smokes yourself when granted** — ALWAYS wrap in a hard `timeout`/`--hours` cap + a `--generations` cap,
  redirect to a log you can read, and KILL/verify-no-orphan after (check port 8000). ⚠ A past incident wasted 3 HOURS
  on a stuck run — NEVER run an unbounded live command. Put smoke logs in a named folder (this session used
  `artifacts/logs/p2_task/`) so the user knows what to delete.
- **⚠ LIVE INTEGRATION IS THE BUG-PRONE SURFACE — the user's live smokes have caught ~8 real bugs offline tests
  missed.** ALWAYS hand the user a live smoke after a live-touching change; on a surfaced bug, fix + regression test +
  diagnostic. (This session's Phase-2 work all passed clean live smokes.)
- **Do NOT retrain / re-export without explicit permission.** When permitted, prefer the CHEAPEST correct path.
- **Use the adversarial/design Workflow** (the project's standard) for substantive design + bug-hunt + root-cause —
  it produced the gate v2 design, the Phase-2 design, and caught real bugs. The pattern: N design lenses → adversarial
  refuters → synthesis; or path-tracers → refuters → completeness-critic.
- **Checkpoint progress into the `gate-redesign-2026-06-17` memory note** when meaningful work lands; update the
  `MEMORY.md` index hook (⚠ MEMORY.md is over its size cap — UPDATE the existing line in place, don't grow it). **The
  USER commits via GitHub Desktop — suggest logical commit groups, do NOT commit yourself.**
- **Don't be afraid to create new files / split long modules** (user's explicit ask). Done this session: the gate
  logic is in `gate.py`, the HoF LIVE orchestration is in the new `v_dance/selfplay/hof.py`.

## ⇐ WHERE WE LEFT OFF — PHASE 2 (HoF anti-cycle) IS BUILT, GREEN, LIVE-CLEAN ⇒ NEXT = P2.5 + commit
The promotion gate is the **v2 FROZEN-CHAMPION LADDER** (Phase 0+1, prior sessions) PLUS **Phase 2 (HoF anti-cycle +
the 0.55 bar + the mirror-collapse degradation guard)**, all built this session (P2.0–P2.4). **901 pass / 0 skip. ALL
UNCOMMITTED on `dev`.** The gate now checks three things every generation:
- **DEPTH** — beat your FROZEN champion: observed mirror win-rate ≥ **0.55** over ≥ **360** mirror games (`beat_champion`),
  OR the h2h climb PLATEAUS at a not-losing level (`plateau_reanchor` backstop).
- **BREADTH (Phase 2 HoF)** — on a promote, ALSO not-LOSE to your last **5 PAST CHAMPIONS** (excluding the current one
  the mirror already tests). A proven loss to an older champion = a lineage cycle (the "G5 promoted backward over G4"
  failure that motivated the redesign) → the promote is DOWNGRADED to a HOLD (`hof_reject`).
- **GROUNDING + degradation** — scripted-ladder Elo; a scripted COLLAPSE reverts (Wilson high-water floor); and the NEW
  **mirror-collapse** revert fires when the learner erodes significantly BELOW its own champion (`wilson_upper(mirror) <
  0.45` over ≥360 games) — a real-strength erosion the scripted floor misses.

**Phase-2 sub-problems all DONE this session:**
- **P2.0** design lock (design-panel + red-team Workflow) → the per-SNAPSHOT significance-veto (band-mean washes out a
  single counter). **User pivot (better): HoF tests PAST CHAMPIONS, not non-champion gens** — older champions aren't
  redundant with the mirror and catch lineage cycling directly. **User also lowered the bar 0.70→0.55** (promote on
  CONVINCING not crushing improvement; couples to the HoF, which becomes load-bearing). **User added the mirror-collapse
  revert** (the correct version of "reset on degradation").
- **P2.1** `gate_sim.py` Monte-Carlo calibration (`--demo`) → LOCKED: HoF z=1.96 / bar wilson_upper<0.50 / n=60/suspect
  (catch@0.30 90%, FWER@0.55 1.9%); mirror thr=0.55 / n=360 (FP 3.1%, pow@0.58 89%; correlation-sensitive → laterals
  caught by the HoF+collapse); mirror-collapse margin 0.05 (bar 0.45) / n=360 (false-revert@0.50 ~0%, catch@0.35 99%).
  **Force-valve SIMPLIFIED** (sim finding: the auto-advance "marginal" window is empty at n=60) → freeze + loud
  operator alert + manual `--hof-override`, NOT an auto-promote. +`wilson_upper_bound` in gate.py.
- **P2.2** gate code: `GateConfigV2` (0.55 / min_h2h_games 360 / mirror_collapse_*); `promotion_gate_v2` + mirror-collapse
  revert; `HoFConfig` + `cluster_hof_suspects` (last 5 past champions excl current) + `hall_of_fame_gate` (worst-of-
  snapshots, thin-pool→skip fail-open). **Audit fixed a real live bug: `--mirror-battles` default 240→360 (the 0.55 bar
  was UNREACHABLE live).**
- **P2.3** `hof_eval` (one `run_gauntlet` 'prev_best' mirror per suspect — zero new battle code; pre-validates each
  suspect so a corrupt checkpoint skips, never false-vetoes on garbage).
- **P2.4** WIRING (new file `v_dance/selfplay/hof.py`): `apply_hof_gate` (cluster → injected runner → gate → downgrade
  → streak → override) called in `run_generation` on a promote; `GenerationHistory.hof_reject_streak` +
  `GenerationRecord.hof` persisted; `operator_alert` 3-way (collapse-loop / **hof-standoff @ streak≥2** / generic-stall);
  manifest `hof` block; CLI `--hof/--no-hof/--hof-champions/--hof-games/--hof-override`; banner + wizard HoF prompt +
  per-gen console line; **`--dry-run` demonstrates the reject offline** (synthetic: gen7 beats the champion but loses to
  gen0 → `HOLD (hof_reject)`).

**LIVE SMOKES (this session, all clean):** P2.2 2-gen smoke (banner shows 0.55/360/mirror-collapse-0.45; 0 bug-patterns;
server clean); P2.3 `scratch/_hof_smoke.py` (candidate vs archived gen0–3 → confirm; at 30 games the CIs tighten
0.85→0.67); P2.4 2-gen smoke (banner HoF line; gen0 promote + HoF skip = 0 past champions; manifest hof field present;
0 bug-patterns; port 8000 clear). The HoF only TRIGGERS with ≥2 real past champions, so the natural live trigger is the
overnight run (P2.5).

## ⇒ IMMEDIATE NEXT
1. **Commit** the Phase-2 work (user does it; suggest groups): **(1) Calibration** — `gate_sim.py`, `gate.py`
   `wilson_upper_bound`, `tests/test_gate_sim.py`, `docs/hof_anticycle_design.md`. **(2) Gate brain** — `gate.py`
   (0.55, mirror-collapse, HoF config/fns, history/record/alert), `tests/test_gate_v2.py`,
   `tests/test_hall_of_fame_gate.py`, `tests/test_prev_best_gate.py`. **(3) HoF live + wiring** — `hof.py`,
   `generation.py`, `archive.py`, `tests/test_hof_eval.py`, `tests/test_hof_wiring.py`, `scratch/_hof_smoke.py`.
   (Plus the wizard `--mirror-battles` + champion-pivot edits already folded into 2/3.)
2. **P2.5 — the OVERNIGHT RUN** (user runs; this is where the HoF triggers on real divergent champions):
   `python -m v_dance.selfplay.generation --live --generations 0 --hours 8 --games 300 --mirror-battles 360
   --max-cpu-fraction 0.5 --collect-workers 12 --max-vram-gb 4 --ckpt ai_train_scripts/BC_model/checkpoints_v4/bc_best.pt
   -v 2> artifacts/logs/p2_task/overnight.log` (or the bare-command WIZARD, which now prompts for mirror-battles + HoF).
   **Clear `artifacts/self_play_archive/` first** (or `--resume` it). WATCH: banner `HoF (Phase 2): ON`; early gens
   `HoF: skipped (… < min_pool 2)`; once ≥2 champions accrue → `HoF[HOF_CONFIRM] vs past champions: …` on promotes, a
   `HOLD (hof_reject)` if a candidate cycles, the `OPERATOR ALERT: … HoF rejects` standoff line if it rejects ≥2 in a
   row (then re-run with `--hof-override` to force it through). Bug-scan the log (grep below) = 0.
3. **THEN** (after Phase 2 ships + the run shows archetype competence): v1.* TP co-development (§14); P3 throughput
   (#13 extract shared parallel runner → #14 multiprocessing); #18 multi-battle spectate (optional polish).

## Full to-do list (carry forward + REPRINT/UPDATE every message)
| # | Task | Effort | Priority | Depends-on | Status |
|---|------|--------|----------|------------|--------|
| 1–9 | Phase 0 + Phase 1 (gate v2, sim, extract, admission, lineage-Elo observability) | — | P1 | — | ✅ done |
| 10 | **Phase 2: HoF anti-cycle + 0.55 bar + mirror-collapse** (P2.0–P2.4) | L | P2 | #8,#9 | ✅ build complete (P2.5 = user smoke) |
| 11 | Commit session gate work (groups 1–3; USER commits) | S | P1 | — | ⬜ uncommitted |
| 12 | 🌙 Overnight run = **P2.5** (`--mirror-battles 360`; HoF triggers live) | — (user) | P2 | #10 | ⏸ ready |
| 13 | Extract shared parallel-battle runner (collection + eval + HoF) | M | P3 | — | ⬜ |
| 14 | 3c.8d true multiprocessing collection | L | P3 | #13 | ⬜ |
| 15 | v1.* TP co-development (§14) | L | P5 | archetype-competent policy (#12) | ⬜ |
| 16 | 3c.7d scripted demo episodes | M | conditional | live training | ⬜ deferred |
| 17 | Reg M-B migration | L | GATED ~2026-06-24 | ecosystem | ⬜ blocked |
| 18 | Multi-battle Spectate (dashboard shows several concurrent battles, not just one) | S | optional | 3c.6 dashboard | ⬜ |

### Phase 2 sub-breakdown (keep until P2.5 ships)
| Sub | Sub-problem | Status |
|---|---|---|
| P2.0 | Design lock (per-snapshot veto; champion pivot; 0.55 bar; mirror-collapse) | ✅ done |
| P2.1 | gate_sim calibration (HoF + 0.55 mirror + mirror-collapse + force-valve) | ✅ done |
| P2.2 | Gate code (HoF fns + 0.55 + mirror-collapse + `--mirror-battles` 360 audit fix) | ✅ done |
| P2.3 | Live HoF eval plumbing (`hof_eval`) | ✅ done |
| P2.4 | Wire into run_generation + force-valve alert + CLI + wizard + observability | ✅ done |
| P2.5 | Full overnight run where the HoF triggers on real champions (USER) | ⬜ ready |

## Environment + commands (Windows; user runs Git Bash + PowerShell; Bash tool available)
- venv active in the user's shell → `python` = venv python. Headless tool calls: `.venv/Scripts/python.exe`. venv node
  at `.venv/node/node.exe`. **Run all commands from the REPO ROOT** `D:\ShowdownProject\Victory-Dance` (Bash default
  CWD is `data/scripts/vod_parser` → `cd /d/ShowdownProject/Victory-Dance` first; PowerShell tool defaults there too).
- Tests: `python -m pytest tests -q` (**901 pass / 0 skip**). `PYTHONIOENCODING=utf-8` when capturing stdout; logs have
  unicode → `grep -a`. See the test-output gotcha in the cadence (use exit code, not the summary line).
- **Gate calibration sim:** `python -m v_dance.selfplay.gate_sim --demo` (HoF veto + 0.55 mirror + mirror-collapse +
  force-valve tables). EXTEND this to validate any new threshold BEFORE wiring live.
- **Dry-run (no server; demonstrates the FULL v2+HoF ladder):** `python -m v_dance.selfplay.generation --dry-run
  --generations 8` → gen0 PROMOTE(no_baseline) → HOLD → PROMOTE(beat_champion, champElo steps) → HOLD →
  REVERT(scripted_collapse) → REVERT(mirror_collapse) → PROMOTE → **gen7 HOLD(hof_reject)** with `HoF vs past champions:
  gen2:40/60 gen0:18/60* -> hof_reject`.
- **HoF eval smoke (server; forces the games vs archived champions):** `python scratch/_hof_smoke.py --games 30
  --workers 6` (needs gen*.pt in `artifacts/self_play_archive/`).
- **Live run flags:** `--live --generations 0 --hours 8 --games 300 --mirror-battles 360 --eval-battles <auto>
  --max-cpu-fraction 0.5 --collect-workers 12 --max-vram-gb 4 --ckpt …checkpoints_v4/bc_best.pt`. HoF flags:
  `--hof/--no-hof` (default ON), `--hof-champions 5`, `--hof-games 60`, `--hof-override`. Bare
  `python -m v_dance.selfplay.generation` = WIZARD. Dashboard: `python -m v_dance.datatools.dashboard_server --port 5175`.
- **Live-smoke recipe (bounded):** `--live --generations 2 --games 10 --eval-battles 4 --mirror-battles 24 --hours 0.12
  -v > artifacts/logs/p2_task/smoke.log 2>&1`; then `grep -acE "would PASS an ACTIVE|Can't pass|can only switch in
  once|NO model loaded|order REJECTED|Traceback|won't load" artifacts/logs/p2_task/smoke.log` MUST be 0. (HoF will SKIP
  in a 2-gen smoke — too few champions; that's correct.) After: confirm port 8000 is clear (no orphan server).
- Gauntlet: `python -m v_dance.eval.gauntlet --ckpt …checkpoints_v4/bc_best.pt --workers 8`. Team validation:
  `python scratch/validate_teams.py`. **Clean up `artifacts/logs/p2_task/` when Phase 2 ships.**

## Gotchas / standing facts
- **GATE = v2 frozen-champion ladder + Phase-2 HoF, in `gate.py` (pure) + `hof.py` (live).** `run_generation` calls
  `promotion_gate_v2` then `apply_hof_gate` (hof.py) on a promote. **Verdict priority:** revert(scripted_collapse →
  mirror_collapse) → promote(beat_champion ≥0.55 over ≥360) → promote(plateau_reanchor) → hold; THEN on a promote the
  HoF can downgrade to hold(hof_reject). `GenConfig` carries `gate_v2`, `hof`, `league_cap`, `keep_recent`.
- **Key config defaults:** `GateConfigV2` promote_threshold=0.55, promote_z=1.645, min_h2h_games=360,
  mirror_collapse_margin=0.05 (bar wilson_upper<0.45), mirror_collapse_z=1.645, mirror_collapse_min_games=360.
  `HoFConfig` enabled=True, n_champions=5, games_per_snapshot=60, min_games_per_snap=40, z=1.96, min_pool=2,
  force_limit=4, override=False. `--mirror-battles` default 360 (was the live bar-unreachable bug at 240).
- **HoF suspects = the last 5 PAST CHAMPIONS** (accepted promotions), newest-first, EXCLUDING the current champion
  (`cluster_hof_suspects(snapshots, n, current_champion_path=history.best_path)`). The gate fn is selection-agnostic
  (only sees `(id,wins,games)`), so a future diverse-non-champion supplement is a one-fn change. min_pool=2 → fail-open
  skip until ≥2 past champions (nothing to cycle early). Rule = NOT-LOSING (significance-veto, wilson_upper<0.5);
  "must-beat-all" was rejected (freezes). Cap ~8 (FWER grows with count).
- **gen 0 ALWAYS auto-promotes** (`no_baseline`). `--live` WITHOUT `--resume` starts FRESH from `--ckpt` BC. Trained
  models → `artifacts/self_play_archive/gen{N}.pt`; champion = `history.best_path`; NEVER written back to bc_best.pt.
- **⚠ bc_best.pt CLOBBER:** production `…checkpoints/bc_best.pt` is gitignored + got overwritten with the v3 backup
  twice. **DURABLE v4 SOURCE = `…checkpoints_v4/bc_best.pt`** — pass `--ckpt …checkpoints_v4/bc_best.pt` to bypass.
- **STATE_DIM 1866, LAYOUT v4, ACTION_DIM 16, GIMMICK_DIM 2.** Production policy = base BC **v4**; TP = 46-dim
  (`teampreview_best.pt`); Tera = placeholder. γ=0.997 FLOOR; PBRS OFF; reward = terminal ±1. KL anchor = STATIC gen-0
  BC. Collection stochastic (tau annealed 1.3→1.0 over 12 gens). ResourceBudget: GPU for PPO update only, collection on
  CPU. User caps: 0.5 CPU (5900X→6) + 4 GB VRAM (3070 Ti).
- **ENV PINNED (`PINS.md`):** poke-env `@a6e4f67`, Showdown `@ecf39eef1`. ⚠ Reg M-B due ~2026-06-24 (gated on the
  ecosystem; stay on M-A; see [[showdown-reg-update-pending-2026-06]]).
- **The adversarial Workflow** (design panel / red-team / path-tracers→refuters→completeness-critic) is the project's
  tool for subtle bugs + design review — it produced the gate v2 + Phase-2 designs this session. Reuse it. Diagnostics:
  `tests/_parity_harness.py`, `scratch/_hof_smoke.py`, `scratch/validate_teams.py`.
