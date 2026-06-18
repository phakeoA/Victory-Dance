# Next-session handoff — Victory-Dance VGC bot: PARALLELISM/MULTICORE/RESUME/SPECTATE shipped → COMMIT + OVERNIGHT RUN

You are Opus 4.x continuing the Victory-Dance self-play Pokémon-Showdown VGC bot, in **ULTRACODE mode**
(author/run adversarial Workflows for substantive design/review/root-cause; adversarially verify findings;
token cost is NOT the constraint — optimise for the most correct, exhaustive answer). Solo only on
conversational/trivial turns.

## STEP 0 — read auto-memory FIRST
Read `C:\Users\death\.claude\projects\D--ShowdownProject-Victory-Dance\memory\MEMORY.md` (the lean index;
it was de-bloated — the FULL old index is archived at the **repo root `memory_before.md`**, gitignored).
**THE RESUME POINTER for the CURRENT work: [[multicore-collection-2026-06-17]] — READ FULLY** (the #13
parallel runner + stop_showdown, #14 multiprocess collection, #19 multiprocess eval, #20 per-gen resume,
#18 + #18b multi-battle spectate — each with its design + the adversarial-review findings + the live
numbers). Then [[ppo-reward-design-2026-06-16]] (the RL bible, `docs/ppo_reward_design.md` §1-20) and
[[gate-redesign-2026-06-17]] (the gate v2 + Phase-2 HoF — that work is COMMITTED). Skim
[[victory-dance-project-layout]], [[showdown-reg-update-pending-2026-06]].

## ⇒ THE USER's REQUIRED CADENCE — follow EXACTLY (this is how the whole project has gone)
- **After EVERY message, REPRINT THE FULL UPDATED TO-DO LIST** as a table with **# · Task · Effort (S/M/L) ·
  Priority · Depends-on · Status** columns. NON-NEGOTIABLE — every single turn, even a one-line reply. (The
  list is below — carry it forward and keep it current. Keep a per-feature sub-breakdown table while a
  multi-part feature is live.)
- **Sub-problems, ONE at a time. PAUSE after each** — report what you did + the EVIDENCE (test counts, the
  actual smoke output, the numbers), then WAIT. Don't chain sub-problems unless the user batches several
  ("green light", "proceed", "do X through Y") — then do them in sequence, test after each, pause at the end.
- **Unit-test every change. Keep ALL tests green — currently 971 pass / 0 skip:** `python -m pytest tests -q`
  (venv active; from repo root). Add a REGRESSION TEST for every bug.
  - ⚠ **Windows test-output gotcha:** pytest's trailing `N passed` summary does NOT flush into a redirected
    stream. Judge pass/fail by the **exit code** + the all-dots progress (no `F`/`E`/`s`). Count via
    `python -m pytest tests --co -q | grep -aoE ": [0-9]+$" | grep -aoE "[0-9]+" | awk '{s+=$1} END{print s}'`.
- **Give a MANUAL test for each feature** (an offline pytest / `--dry-run` AND, for live features, the EXACT
  command + pass criteria). The user likes running things.
- **The USER runs the long live stuff** (overnight). **You MAY run BOUNDED live smokes when granted** —
  ALWAYS wrap in a hard `timeout` + `--hours`/`--generations` cap, redirect to a log under a named folder
  (this session used `artifacts/logs/<task>/`), and AFTER: bug-scan the log (grep
  `Traceback|NO model loaded|can only switch in once|order REJECTED|won't load`) AND **kill any orphan +
  verify port 8000 is clear** (`powershell "Get-NetTCPConnection -LocalPort 8000 -State Listen"`). NEVER run
  an unbounded live command. Also delete smoke `data/vods/Type_D/gen_*_g*.html` (live runs pollute the corpus).
- **⚠ LIVE INTEGRATION IS THE BUG-PRONE SURFACE.** Hand the user a live smoke after a live-touching change;
  run a bounded one yourself when granted. Clean up after every smoke (and note where the logs are).
- **Use the ADVERSARIAL WORKFLOW for substantive design + bug-hunt + the RISKY refactors** (the project's
  standard; it caught the #14 wrong-critic bug and the #18b dashboard-scan/staleness bugs THIS session).
  Pattern: 2-4 lenses → adversarial verify (default-refute) → synthesis; relay confirmed findings, fix +
  regression-test them. Launch it in the BACKGROUND and run the live smoke meanwhile.
- **Checkpoint progress into [[multicore-collection-2026-06-17]]** when meaningful work lands; update the
  `MEMORY.md` index hook IN PLACE (⚠ MEMORY.md is near its cap — keep hooks short; full detail in the topic
  note + `memory_before.md`). **The USER commits via GitHub Desktop — SUGGEST logical commit groups, do NOT
  commit yourself.**
- **Don't retrain / re-export without explicit permission** ([[dont-retrain-until-told-2026-06-14]]). When
  permitted, prefer the CHEAPEST correct path. Production model = **BC v4** (`--ckpt …checkpoints_v4/bc_best.pt`).
- **Don't be afraid to create new files / split modules** (the user's explicit ask).

## ⇐ WHERE WE LEFT OFF — this session SHIPPED #13/#14/#19/#20/#18/#18b, ALL UNCOMMITTED on `dev`
Throughput + UX overhaul of the self-play loop. **971 pass / 0 skip. All live-verified; the risky refactors
adversarially reviewed.** (Phase-2 gate work from the PRIOR session is already COMMITTED.)
- **#13 shared parallel-battle runner** — `v_dance/play/parallel_battles.py` (`play_pairing`/`close_players`/
  `run_jobs`); `run_gauntlet` + `collect_with_league` + `run_self_play_games` refactored onto it. PLUS
  **`run_local_battle.stop_showdown` TREE-KILL** fix (it orphaned the forked Node server on port 8000).
- **#14 TRUE multiprocessing collection** — `v_dance/selfplay/mp_collect.py`: 1 shared Showdown server + N
  CPU worker PROCESSES (`ProcessPoolExecutor`, spawn). Opt-in `--collect-procs N` (default 1 = asyncio path
  VERBATIM, zero regression) + `--collect-async`. Probe (`scratch/mp_collection_probe.py`) showed it scales;
  live A/B: collection ~158→363 games/min @ asyncio-12 → 8 procs. ⚠ review-fixed: worker loaded the WRONG
  (re-cloned) critic → `_worker_ac` now `restore_from` loads the trained critic_state.
- **#19 multiprocess EVAL** — `v_dance/selfplay/mp_eval.py`, reuses the SAME pool via `submit(worker_fn=)`.
- **#20 per-generation resume** — `--resume-gen N|latest` (loads `snap_gen{N}.pt`, continues at N+1; the old
  single `resume.pt` is OBSOLETE), `--keep-snapshots K` (default 25, 0=all). `resume.py` helpers.
- **#18 + #18b multi-battle spectate** — file-per-battle feed; STRUCTURED saved replays
  `artifacts/<archive>/live/<run-stamp>/gen_<N>/{replays,eval}/<tag>.json`; **EVAL matches spectate too**
  (live-reporting moved to the BASE player `live_vgc_base`, guarded → non-spectate play byte-identical);
  `--save-replays` (default OFF). Dashboard shows several concurrent battles (chips, self/eval labels);
  review-fixed: scope the dashboard scan to the current run's latest gen + `stale_after` 15→45s.
- **Memory de-bloated**: `MEMORY.md` 27KB→~4KB lean index; full archive at repo-root `memory_before.md`
  (gitignored). Old `--resume resume.pt` references are obsolete (use `--resume-gen latest`).

## ⇒ IMMEDIATE NEXT
1. **COMMIT this session's work** (USER does it via GitHub Desktop). Several features interleave in shared
   files (`generation.py`, `game_runner.py`), so clean per-feature commits aren't possible with file-level
   staging — simplest is **1-2 commits**: (A) parallelism + multicore + resume backend & wiring
   (`play/parallel_battles.py`, `play/run_local_battle.py`, `eval/gauntlet.py`, `selfplay/mp_collect.py`,
   `selfplay/mp_eval.py`, `selfplay/resume.py`, `selfplay/game_runner.py`, `selfplay/generation.py` + tests);
   (B) spectate (`selfplay/status.py`, `play/live_vgc_base.py`, `datatools/dashboard_server.py`,
   `data/scripts/dashboard/dashboard.{js,css}` + spectate tests) — or commit it all as one. `.gitignore`
   gained `memory_before.md`.
2. **🌙 THE OVERNIGHT RUN (P2.5)** — the user runs it; this is where the HoF triggers on real divergent
   champions. **Clear `artifacts/self_play_archive/` first** (or `--resume-gen latest`):
   `python -m v_dance.selfplay.generation --live --generations 0 --hours 8 --games 300 --mirror-battles 360
   --max-cpu-fraction 0.8 --collect-procs 8 --collect-async 3 --max-vram-gb 5 [--save-replays]
   --ckpt ai_train_scripts/BC_model/checkpoints_v4/bc_best.pt -v 2> artifacts/logs/overnight.log`
   (or the bare-command WIZARD — it now prompts for collect-procs, save-replays, resume-gen). Dashboard:
   `python -m v_dance.datatools.dashboard_server --port 5175` → Spectate tab shows several concurrent
   collection + eval battles. WATCH: banner `MULTIPROCESS 8 procs`; per-gen `throughput`; `HoF[...]` on
   promotes once ≥2 champions accrue; bug-scan the log = 0.
3. **THEN the backlog**: #19b staged short→full mirror (cut eval cost — eval is now the per-gen bottleneck
   since collection got fast); #15 v1.* TP co-development (§14, after the policy is archetype-competent);
   #16 scripted demo episodes (only if rare tactics stay under-clicked); #17 Reg M-B migration (~2026-06-24).

## Full to-do list (carry forward + REPRINT/UPDATE every message)
| # | Task | Effort | Priority | Depends-on | Status |
|---|------|--------|----------|------------|--------|
| 1–11 | Phase 0/1/2 gate (v2 ladder + HoF anti-cycle) + its commit | — | P1/P2 | — | ✅ committed (prior session) |
| 11b | **Commit THIS session's work** (#13/14/19/20/18/18b; USER commits, groups above) | S | P1 | — | ⬜ uncommitted |
| 12 | 🌙 Overnight run = P2.5 (`--collect-procs 8`; HoF triggers live) | — (user) | P2 | #10 | ⏸ ready |
| 13 | Shared parallel-battle runner (+ stop_showdown tree-kill) | M | P3 | — | ✅ done + live-verified |
| 14 | True multiprocessing collection (`--collect-procs`) | L | P3 | #13 | ✅ done + live-verified |
| 19 | Multiprocess eval (reuse the pool) | M | P3 | #14 | ✅ done + live-verified |
| 20 | Per-generation resume (`--resume-gen N|latest`) | M | P3 | — | ✅ done + live-verified |
| 18 | Multi-battle spectate (several concurrent battles) | M | P4 | 3c.6 | ✅ done + live-verified |
| 18b | Saved/structured spectator replays + EVAL spectate | L | P4 | #18 | ✅ done + reviewed + live-verified |
| 19b | Staged short→full mirror (cut eval cost; gate.py + gate_sim) | S–M | P4 (next code) | — | ⬜ |
| 15 | v1.* TP co-development (§14) | L | P5 | archetype-competent policy (#12) | ⬜ |
| 16 | 3c.7d scripted demo episodes | M | conditional | live training | ⬜ deferred |
| 17 | Reg M-B migration | L | GATED ~2026-06-24 | ecosystem | ⬜ blocked |

## Environment + commands (Windows; user runs Git Bash + PowerShell; Bash tool available)
- venv active in the user's shell → `python` = venv python. Headless tool calls: `.venv/Scripts/python.exe`;
  venv node `.venv/node/node.exe`. **Run all commands from the REPO ROOT** `D:\ShowdownProject\Victory-Dance`
  (Bash default CWD is `data/scripts/vod_parser` → `cd /d/ShowdownProject/Victory-Dance` first).
- Tests: `python -m pytest tests -q` (**971 pass / 0 skip**). `PYTHONIOENCODING=utf-8` when capturing stdout.
- **Dry-run (no server; the full v2+HoF ladder):** `python -m v_dance.selfplay.generation --dry-run --generations 8`.
- **Multicore probe (throwaway):** `python scratch/mp_collection_probe.py --procs 1 2 4 --conc 3 --per 4`.
- **Live-smoke recipe (bounded, throwaway archive):** `--live --generations 1 --games 24 --eval-battles 4
  --mirror-battles 24 --hours 0.12 --collect-procs 4 --collect-async 3 --max-cpu-fraction 0.8 --max-vram-gb 5
  --archive artifacts/logs/<task>/arch --ckpt …checkpoints_v4/bc_best.pt -v`; then bug-scan + kill orphan +
  port-8000 check + delete smoke `data/vods/Type_D/gen_*_g*.html`.
- **Dashboard:** `python -m v_dance.datatools.dashboard_server --port 5175` (point `--archive` at the run's
  archive). Gauntlet: `python -m v_dance.eval.gauntlet --ckpt …checkpoints_v4/bc_best.pt --workers 8`.

## Gotchas / standing facts
- **GATE = v2 frozen-champion ladder + Phase-2 HoF** (`gate.py` pure + `hof.py` live); see [[gate-redesign-2026-06-17]].
- **⚠ bc_best.pt CLOBBER:** production `…checkpoints/bc_best.pt` got overwritten with the v3 backup twice →
  **DURABLE v4 SOURCE = `…checkpoints_v4/bc_best.pt`** (pass `--ckpt …checkpoints_v4/bc_best.pt`). STATE_DIM
  1866 / LAYOUT v4 / ACTION_DIM 16 / GIMMICK_DIM 2. TP net = 46-dim. ENV PINNED (`PINS.md`): poke-env
  @a6e4f67, Showdown @ecf39eef1. ⚠ Reg M-B due ~2026-06-24 ([[showdown-reg-update-pending-2026-06]]).
- **Resource caps** (the user's machine, 5900X/12-phys + 3070 Ti 8GB): `--max-cpu-fraction 0.8`,
  `--collect-procs 8`, `--max-vram-gb 5`. ~32 GB RAM but ~13 GB used by Chrome/VSCode/Claude → keep workers
  modest (~0.6 GB each). GPU = PPO UPDATE only; collection/eval on CPU.
- **mp collection STOP semantics:** a stop (Ctrl-C/--hours) is honoured at GEN BOUNDARIES (the current gen's
  batch finishes; the finer per-chunk drain is asyncio-only). Per-gen mp ckpt + spectate files are swept.
- **The adversarial Workflow** is the project's tool for subtle bugs + design. Diagnostics:
  `scratch/mp_collection_probe.py`, `scratch/_hof_smoke.py`, `scratch/validate_teams.py`,
  `tests/_parity_harness.py`.
