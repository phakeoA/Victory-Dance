# Next-session handoff — Victory-Dance VGC bot: LIVE-RUN HARDENED → relaunch overnight + commit

You are Opus 4.x continuing the Victory-Dance self-play Pokémon-Showdown VGC bot, in **ULTRACODE mode**
(author/run adversarial Workflows for substantive design/review/root-cause when it genuinely helps;
adversarially verify findings; token cost is NOT the constraint — optimise for the most correct,
exhaustive answer). Solo only on conversational/trivial turns.

## STEP 0 — read auto-memory FIRST
Read `C:\Users\death\.claude\projects\D--ShowdownProject-Victory-Dance\memory\MEMORY.md` (the lean index;
the FULL old index is archived at the **repo-root `memory_before.md`**, gitignored).
**THE RESUME POINTER: [[multicore-collection-2026-06-17]] — READ FULLY.** Its BOTTOM half is the
per-fix log for THIS session (Fixes A–E, G, H, H2, #22 stall-watchdog, #22c server-recycle) plus the
layered eval/server-bloat root-cause diagnosis (#22 / 22c / 22d / 22e / 22f). Then
[[ppo-reward-design-2026-06-16]] (the RL bible, `docs/ppo_reward_design.md`) and
[[gate-redesign-2026-06-17]] (gate v2 + Phase-2 HoF). Skim [[victory-dance-project-layout]],
[[showdown-reg-update-pending-2026-06]].

## ⇒ THE USER's REQUIRED CADENCE — follow EXACTLY (this is how the whole project runs)
- **After EVERY message, REPRINT THE FULL UPDATED TO-DO LIST** as a table with **# · Task · Effort
  (S/M/L) · Priority · Depends-on · Status** columns. NON-NEGOTIABLE — every single turn, even a
  one-line reply. ⚠ The user EXPLICITLY wants the **FULL** list every time, NOT an abbreviated/"lazy"
  one. Carry the list below forward and keep it current.
- **Sub-problems, ONE at a time. PAUSE after each** — report what you did + the EVIDENCE (test counts,
  the actual smoke output, the numbers), then WAIT. Don't chain unless the user batches ("do both",
  "proceed", "do X through Y") — then sequence them, test after each, pause at the end.
- **Unit-test every change. Keep ALL tests green — currently 992 pass / 0 skip:** `python -m pytest
  tests -q` (venv active, from repo root). Add a REGRESSION test for every bug. ⚠ Windows: pytest's
  trailing `N passed` doesn't flush to a redirect — judge by the **exit code** + all-dots (no F/E/s).
  Count: `python -m pytest tests --co -q | grep -aoE ": [0-9]+$" | grep -aoE "[0-9]+" | awk '{s+=$1}
  END{print s}'`.
- **Give a MANUAL test for each feature** (offline pytest/`--dry-run` AND, for live features, the EXACT
  command + pass criteria). The user likes running things himself.
- **Live runs:** the USER normally runs the overnight, BUT this session he asked me to kill / fix /
  resume / smoke DIRECTLY — do that when asked. **Bounded live smokes only when granted:** hard
  `--hours`/`--generations` cap, THROWAWAY `--archive`, redirect to a log; AFTER: bug-scan the log
  (`Traceback|ConnectionRefused|keepalive ping|Expected .* logged in|cannot reject open team|already a
  challenge|did not open port`) + kill any orphan + verify port 8000 clear + delete smoke
  `data/vods/Type_D/gen_*_g*.html`. ⚠ **smoke Type_D shares filenames with the real corpus** (both
  write `gen_<N>_g<N>.html`) — a smoke OVERWRITES the real run's gen-0..N showcase replays; route smoke
  Type_D elsewhere or accept the cosmetic overwrite.
- **⚠ WINDOWS PROCESS-KILL GOTCHAS (learned hard this session):** spawn (mp) workers ORPHAN when the
  parent dies — `taskkill /PID <main> /T /F` to kill the tree. A run launched as `python
  generation.py` has cmdline `generation.py`, NOT `v_dance.selfplay.generation`, so a
  `CommandLine -match 'v_dance.selfplay.generation'` kill MISSES it → **always launch via `-m
  v_dance.selfplay.generation`** so the cmdline-match kill works. After killing, sweep
  `spawn_main|multiprocessing-fork` python + clear port 8000 (`Get-NetTCPConnection -LocalPort 8000`).
- **Use the ADVERSARIAL WORKFLOW for substantive design + bug-hunt + risky refactors** (project
  standard). Launch in the BACKGROUND and run the live smoke meanwhile.
- **Checkpoint progress into [[multicore-collection-2026-06-17]]** when meaningful work lands; update
  the `MEMORY.md` hook IN PLACE (it's near its cap — keep hooks short; full detail in the topic note).
  **The USER commits via GitHub Desktop — SUGGEST commit groups, do NOT commit yourself.**
- **Don't retrain / re-export without explicit permission** ([[dont-retrain-until-told-2026-06-14]]).
  Production = **BC v4**; ✅ VERIFIED the DEFAULT `ai_train_scripts/BC_model/checkpoints/bc_best.pt` is
  BYTE-IDENTICAL to `checkpoints_v4/bc_best.pt` (md5 `e297b5de…`), so the live run IS on v4. On RESUME
  use the SAME `--ckpt` the run started with (the default) — do NOT switch to `…checkpoints_v4/…`
  mid-run (it changes the frozen-BC / KL-to-BC anchor → inconsistent gate dynamics).
- **⚠ The `preview_start` / Claude-Preview browser tool is BROKEN on this Windows box** (runs a wrong /
  cached command, ignores `.claude/launch.json`). Verify the dashboard via `curl` + `node --check`
  (`.venv/node/node.exe --check data/scripts/dashboard/dashboard.js`) + the USER eyeballing in-browser.

## ⇐ WHERE WE LEFT OFF — this session HARDENED the live run. **992 pass / 0 skip. ALL UNCOMMITTED on `dev`.**
Started from the parallelism/multicore/resume/spectate work (shipped last session) and fixed a cascade
of issues the user hit running the loop live:
- **A** `--save-replays` now writes real playable Showdown **.html** (was litter JSON the dashboard
  ignored). **B** Ctrl-C no longer wedges the terminal (force `WindowsSelectorEventLoopPolicy` in
  `v_dance/__init__.py` — the Windows Proactor `_poll` race under Ctrl-C was killing poke-env's
  POKE_LOOP thread + hanging the process). **C** stray `./replays/` + `v_dance/selfplay/replays/` dump
  KILLED — an `_save_replays` **attribute COLLISION** with poke-env's native flag (we set it →
  poke-env's `_create_battle` re-enabled its native dump); renamed ours `_save_html_replays`. **D**
  archive tidied: `genN.pt`→`checkpoints/`, `snap_genN.pt`→`sub_checkpoints/` (`list_snapshots`
  backward-compat reads the root too). **E** eval `--save-replays` organised by opponent:
  `eval/<kind>/gen<N>_vs_<kind>_<tag>.html` (scripted) + `eval/league/gen<N>_vs_gen<M>_<tag>.html`
  (prev_best mirror + HoF; live spectate JSON stays flat so the dashboard is untouched).
- **G** capped `artifacts/replay_buffer/` (`--keep-replay-buffers`, default 200 — it's a write-only
  BC-era per-turn trace, NOT read by RL training). **H** dashboard Spectate tab now browses a selected
  gen's SAVED eval replays grouped by section (league/championship first); new `/eval_replays.json` +
  `/eval_replay/<gen>/<kind>/<name>` endpoints (+ `create_app` now `.resolve()`s the archive — a
  relative `--archive` 404'd `send_from_directory`). **H2** the league section shows the per-gen HoF
  status from the manifest (`thin_pool_skip` / passed / vetoed) so an absent `gen<N>_vs_gen0` reads as
  intentional, not missing.
- **🌙 THE OVERNIGHT RUN went live and hit TWO failures, both FIXED + verified:**
  - **#22 — eval DEADLOCKED @ gen 14.** Under 8-proc mp eval (~24 concurrent model-vs-model mirror
    battles) the single Node server stalled → eval websockets dropped (`1011 keepalive ping timeout`) +
    logins timed out, and the OLD watchdog `wait_for(battle_against(n), battle_timeout*n)` was one
    COARSE per-chunk timeout → a worker blocked for MINUTES. **FIX: a STALL watchdog** in
    `play/parallel_battles.play_pairing` — abandons a chunk only when NO battle FINISHES for
    `battle_timeout`s (~90s), polled, regardless of n; retrieves `task.exception()` (kills the "Task
    exception was never retrieved" spam). Plus `run_local_battle.make_player` bumps poke-env
    `ping_timeout=60`/`open_timeout=30` (`WS_*` consts). +2 tests.
  - **#22c — server BLOAT degraded it @ gen 54.** Even at lower concurrency, the single long-lived Node
    server leaks over ~40k battles → eval crawled 700→51 games/min, handshake timeouts, stale
    challenges (`already a challenge between you and OPprev98`), `cannot reject open team sheets`. **FIX:
    `--restart-server-every N`** (default 20) recycles the server at gen boundaries (`stop_showdown` +
    `start_showdown`; next gen's players connect fresh). LIVE-VERIFIED via a 3-gen smoke (2 clean
    recycles, zero errors). The stall watchdog kept the run ALIVE to gen 54 (no deadlock) — proof it
    works.
- **Root-cause honesty (user asked "root fix or band-aid?"):** #22 stall-watchdog = a REAL root fix for
  OUR coarse-timeout flaw; the timeout bump + #22c recycle = MITIGATIONS for an external Showdown leak
  we can't patch (pinned dep). Deeper roots LEFT ON THE TABLE: **#22d unique account names**
  (recommended, XS — see below), 22e (clean server-side state on abandon), 22f (multi-server, fundamental).

## ⇒ IMMEDIATE NEXT
1. **The overnight run is STOPPED @ gen 54** (user stopped it; machine clean, port 8000 free, latest =
   `snap_gen54.pt`). **RELAUNCH** (user runs it in HIS terminal so Ctrl-C works; or run it for him if
   asked — launch via `-m`, detached `Start-Process` + `-u`, redirect to a log):
   ```
   python -m v_dance.selfplay.generation --live --generations 0 --hours 7 --games 300 \
     --max-cpu-fraction 0.8 --collect-procs 6 --collect-async 2 --max-vram-gb 5 \
     --mirror-battles 360 --hof-champions 8 --save-replays --restart-server-every 20 --resume-gen latest
   ```
   (12-in-flight is STABLE and actually FASTER end-to-end than 24 — the eval saturation that halved
   throughput is gone; DEFAULT ckpt = v4; recycle every 20 gens stays ahead of the bloat.)
2. **#22d (RECOMMENDED, XS) — globally-unique battle account names.** `build_eval_specs`
   (mp_eval.py:48) and `build_chunk_specs` (mp_collect.py:89) both reset `uid=0` each gen, so usernames
   `BC{uid}`/`OPprev…{uid}`/`LG…{uid}` REPEAT across gens → a stale server-side challenge from one gen
   collides with the next gen's reuse → the `already a challenge` errors. Fix = include the gen in the
   name (e.g. `BC{gen}_{uid}`). A genuine ROOT fix for that symptom; I offered it, user hadn't yet said
   go — CONFIRM before doing.
3. **COMMIT this session's work (11b)** — USER via GitHub Desktop. Touched: `v_dance/__init__.py`;
   `play/{run_local_battle,live_vgc_base,vgc_base,parallel_battles,player}.py`;
   `selfplay/{status,generation,resume,mp_eval,hof,archive}.py`; `eval/gauntlet.py`;
   `datatools/dashboard_server.py`; `data/scripts/dashboard/dashboard.{js,css}`; `.gitignore`; tests
   `test_{win_event_loop,live_battles,resume_snapshots,replay_buffer_prune,parallel_battles,dashboard_server}.py`.
   Suggest ~2-3 commits (live-run fixes / dashboard / archive-layout) or one — file-level staging can't
   cleanly split the interleaved generation.py changes.
4. **THEN the backlog**: 22e, 22f (above); 19b staged short→full mirror; 15a TP bring-diversity
   DIAGNOSTIC FIRST (the "never brings 2 mons" concern was HYPOTHETICAL, not observed); 15b full TP
   co-dev (gated on a competent policy); 16 demo eps; 17 Reg M-B (~2026-06-24); 21 team-builder AI.

## Full to-do list (carry forward + REPRINT/UPDATE every message)
| # | Task | Effort | Priority | Depends-on | Status |
|---|------|--------|----------|------------|--------|
| 1–11 | Phase 0/1/2 gate (v2 ladder + HoF anti-cycle) | — | P1/P2 | — | ✅ committed (prior session) |
| 13–18b | parallelism / multicore / resume / spectate | — | P3/P4 | — | ✅ done (uncommitted) |
| A–E | HTML replays / Ctrl-C / stray dump / archive subfolders / eval routing | S–L | P1/P2 | — | ✅ done + tested |
| G | Cap `artifacts/replay_buffer/` | S | P2 | — | ✅ done + tested |
| H / H2 | Dashboard saved-replay browser + HoF-skip note | L / XS | P2 / P4 | E | ✅ done + tested |
| 22 | Stall watchdog (eval deadlock fix) | M | P2 | — | ✅ done + live-verified |
| 22c | Server recycle (`--restart-server-every`, bloat fix) | S | P2 | — | ✅ done + smoke-verified |
| 11b | **Commit this session's work** (USER via GitHub Desktop) | S | P1 | — | ⬜ uncommitted |
| 12 | 🌙 Overnight run — relaunch from gen 54 w/ `--restart-server-every 20` | — (user) | P2 | 11b | ⏸ stopped @ gen54, ready |
| 22d | **Globally-unique account names** (root fix for "already a challenge") | XS | P2 | — | 💬 offered, pending user OK |
| 22e | Clean up server-side state when abandoning a chunk | S | P3 | — | ⬜ |
| 22f | Multi-server architecture (1 server per worker / pool — fundamental) | L | P4 | — | ⬜ |
| 19b | Staged short→full mirror (cut eval cost; `gate.py`+`gate_sim`) | S–M | P4 | — | ⬜ |
| 15a | TP bring-diversity DIAGNOSTIC (concern was hypothetical → verify first) | S–M | P5 | — | ⬜ |
| 15b | Full outcome-driven TP co-development (§14) | L | P5 | competent policy (#12) | ⬜ |
| 16 | 3c.7d scripted demo episodes | M | conditional | live training | ⬜ deferred |
| 17 | Reg M-B migration | L | GATED ~2026-06-24 | ecosystem | ⬜ blocked |
| 21 | Team-builder AI — generate a full format-legal VGC team (6 mons + sets) | L | P6 (later) | legal-team validation + strong policy | ⬜ future |

## Environment + commands (Windows; user runs Git Bash + PowerShell; Bash + PowerShell tools available)
- venv active → `python` = venv python. Headless: `.venv/Scripts/python.exe`, venv node
  `.venv/node/node.exe`. **Run all from the REPO ROOT** `D:\ShowdownProject\Victory-Dance` (Bash CWD is
  `data/scripts/vod_parser` → `cd /d/ShowdownProject/Victory-Dance` first).
- Tests: `python -m pytest tests -q` (**992 pass / 0 skip**). `PYTHONIOENCODING=utf-8` when capturing.
- **Dry-run (no server):** `python -m v_dance.selfplay.generation --dry-run --generations 8`.
- **Dashboard:** `python -m v_dance.datatools.dashboard_server --port 5175 --archive
  artifacts/self_play_archive` → Spectate tab (live battles + a selected gen's saved eval replays,
  league/championship first, with the HoF-status note). Hard-refresh to pick up JS/CSS changes.
- **Relaunch overnight:** the command in IMMEDIATE NEXT #1. Detached + unbuffered if launching for the
  user: `$env:PYTHONUNBUFFERED='1'; Start-Process .venv\Scripts\python.exe -ArgumentList '-u','-m',…
  -RedirectStandardOutput/Error <log> -WindowStyle Hidden`.

## Gotchas / standing facts (this session's additions ★)
- **GATE = v2 frozen-champion ladder + Phase-2 HoF** ([[gate-redesign-2026-06-17]]). The HoF runs ONLY
  on a PROMOTE and only when ≥`min_pool` (=2) PAST champions exist (excluding the current champ); the
  manifest's per-gen `hof` field records `thin_pool_skip` / `hof_confirm` / `hof_reject` / null
  (=didn't promote). Champion lineage of the live run: gen0→gen11→gen21→… (the HoF first FIRED @ gen23).
- ★ **STATE_DIM 1866 / LAYOUT v4 / ACTION_DIM 16 / GIMMICK_DIM 2** (a retrain is forced ONLY by a
  STATE_LAYOUT_VERSION bump or an ACTION/GIMMICK change — nothing on the to-do does that; Reg M-B is a
  WARM-STARTED fine-tune, never from-scratch — Mega→Tera one day = a layout bump + warm-start too). TP
  net = 46-dim. ENV PINNED (`PINS.md`): poke-env @a6e4f67, Showdown @ecf39eef1. ⚠ Reg M-B due ~2026-06-24.
- ★ **Production = BC v4; the DEFAULT `checkpoints/bc_best.pt` == `checkpoints_v4/bc_best.pt`** (md5
  `e297b5de…`, verified 2026-06-18). The live run is on v4. Resume with the SAME (default) ckpt.
- ★ **Resource caps** (5900X/12-phys + 3070 Ti 8GB; ~13GB used by Chrome/VSCode/Claude): `--collect-procs
  6 --collect-async 2` (=12 in-flight) is the SWEET SPOT — stable AND faster than 24 (the 24-concurrent
  saturation halved eval throughput). `--max-cpu-fraction 0.8 --max-vram-gb 5`.
- ★ **The eval/server failure ladder** (read the #22/22c memory log): the SINGLE shared Node server is
  the bottleneck — it saturates under high concurrency AND bloats over a long run. Mitigations shipped
  (stall watchdog + recycle + timeouts + lower conc); deeper roots = unique account names (22d),
  server-state cleanup (22e), multi-server (22f).
- **Diagnostics:** `scratch/mp_collection_probe.py`, `scratch/_hof_smoke.py`,
  `scratch/validate_teams.py`, `tests/_parity_harness.py`. The adversarial Workflow is the standard
  tool for subtle bugs + design.
