# Next-session handoff — Victory-Dance VGC bot: multi-server + 22d/22e + leak fix + Showdown-native replays DONE → overnight resuming

You are Opus 4.x continuing the Victory-Dance self-play Pokémon-Showdown VGC bot, in **ULTRACODE mode**
(author/run adversarial Workflows for substantive design/review/root-cause when it genuinely helps;
adversarially verify findings; token cost is NOT the constraint — optimise for the most correct answer).
Solo only on conversational/trivial turns.

## STEP 0 — read auto-memory FIRST
Read `C:\Users\death\.claude\projects\D--ShowdownProject-Victory-Dance\memory\MEMORY.md` (the lean index;
the FULL old index is archived at the **repo-root `memory_before.md`**, gitignored).
**THE RESUME POINTER: [[multicore-collection-2026-06-17]] — READ FULLY.** Its BOTTOM half is the per-fix
log for the recent sessions (22/22c stall+recycle; 22d/22e; 22f multi-server; the memory-leak hunt;
Type_D removal; 19b CLOSED; the Showdown-native replay emitter). Then [[ppo-reward-design-2026-06-16]]
(the RL bible), [[gate-redesign-2026-06-17]] (gate v2 + HoF). Skim [[victory-dance-project-layout]],
[[showdown-reg-update-pending-2026-06]]. **NEW standing feedback this session — obey both:**
[[dont-recommend-commits-2026-06-18]] and [[keep-calibration-artifacts-2026-06-18]].

## ⇒ THE USER's REQUIRED CADENCE — follow EXACTLY (this is how the whole project runs)
- **After EVERY message, REPRINT THE FULL UPDATED TO-DO LIST** as a table with **# · Task · Status**
  (add Effort/Priority/Depends-on columns if useful). NON-NEGOTIABLE — every single turn, even a
  one-line reply. ⚠ The user EXPLICITLY wants the **FULL** list every time, NOT an abbreviated one.
  Carry the table below forward and keep it current.
- **Sub-problems, ONE at a time. PAUSE after each** — report what you did + the EVIDENCE (test counts,
  the actual smoke output, the numbers), then WAIT. Only chain if the user batches ("do X through Y").
- **Unit-test every change. Keep ALL tests green — currently 1024 pass / 0 skip:** `python -m pytest
  tests -q` (venv active, from repo root). Add a REGRESSION test for every bug. ⚠ Windows: judge by the
  **exit code** + all-dots (no F/E/s). Count: `python -m pytest tests --co -q | grep -aoE ": [0-9]+$" |
  grep -aoE "[0-9]+" | awk '{s+=$1}END{print s}'`. (Use `.venv/Scripts/python.exe` — bare `python` in
  Bash lacks the venv.)
- **Give a MANUAL test for each feature** (offline pytest/`--dry-run` AND, for live features, the EXACT
  command + pass criteria). The user likes running things himself.
- **⚠ DO NOT recommend/suggest/offer commits or commit groupings** ([[dont-recommend-commits-2026-06-18]]).
  The USER commits on his own via **GitHub Desktop**. Stating "X is uncommitted on `dev`" as a neutral
  fact is fine; "want me to draft commit groups?" is NOT. NEVER run `git commit` yourself.
- **⚠ KEEP gate_sim calibration artifacts** ([[keep-calibration-artifacts-2026-06-18]]) — don't revert the
  shelved-19b staged-mirror code; gate_sim is a DOC/calibration module, the code IS the decision record.
  (Contrast: DELETE superseded PRODUCTION code, e.g. Type_D, which was right to remove.)
- **Live runs: the USER runs the overnight in HIS terminal** (so Ctrl-C exits clean — that fix holds).
  Run bounded live smokes yourself ONLY when granted: hard `--hours`/`--generations` cap, THROWAWAY
  `--archive`, redirect to a log; AFTER: bug-scan + kill orphans + clear ports + delete smoke debris.
- **⚠ WINDOWS PROCESS GOTCHAS:** spawn (mp) workers ORPHAN when the parent dies → `taskkill /PID <main>
  /T /F` to kill the tree; **always launch via `-m v_dance.selfplay.generation`** so a cmdline-match kill
  works; sweep spawn_main python + clear ports 8000.. after. **⚠ LAUNCH ONLY ONE RUN** — the user
  accidentally ran TWO against the same archive/ports this session → account-name + checkpoint collisions.
- **Use the ADVERSARIAL WORKFLOW for substantive design + bug-hunt + risky refactors** (project standard;
  the memory-leak hunt this session was one). Launch in the BACKGROUND and work meanwhile.
- **Checkpoint into [[multicore-collection-2026-06-17]]** when meaningful work lands; keep the `MEMORY.md`
  hook short. **Don't retrain/re-export without explicit permission** ([[dont-retrain-until-told-2026-06-14]]);
  production = **BC v4** (the DEFAULT `--ckpt` IS v4; resume with the SAME default — don't switch mid-run).
- **⚠ `preview_start` browser tool BROKEN on this box.** Verify dashboard via `curl` + `node --check` +
  the USER eyeballing in-browser.

## ⇐ WHERE WE LEFT OFF — 1024 pass / 0 skip. ALL UNCOMMITTED on `dev`.
The overnight is **STOPPED** (user Ctrl-C'd cleanly — verified 0 orphans). **Latest snapshot = `snap_gen57.pt`**
(champion gen 56, champElo **1499** and climbing from ~1383@gen54). The user is **relaunching now** because he's
**not yet satisfied with the battle AI** — wants to keep training. The resume command (he runs it in HIS terminal):
```
python -m v_dance.selfplay.generation --live --generations 0 --hours 8 --games 300 --collect-procs 6 --collect-async 2 --max-cpu-fraction 0.8 --max-vram-gb 5 --mirror-battles 360 --hof-champions 8 --save-replays --servers 2 --restart-server-every 20 --resume-gen latest
```
(`--resume-gen latest` → loads snap_gen57, continues at gen 58. NO `--ckpt` = default v4. ONE run only.)

**This session's shipped work (all DONE + tested, UNCOMMITTED on `dev`):**
- **22d** — globally-unique PER-GEN battle account names (root fix for `already a challenge between you and
  OPprev…`). `play/parallel_battles.py`: `gen_salt`/`eval_account_names`/`collect_account_names` (toID-safe
  'x' delimiter, <18 chars, salt='' == legacy). Threaded through mp_collect/mp_eval (`.gen` on ChunkSpec/
  EvalSpec) + asyncio `collect_with_league(name_salt=)` + `run_gauntlet(name_salt=)` + HoF (salt `{cand}h{susp}`).
  Live-verified.
- **22e** — `parallel_battles.abandon_server_state(model, opp)`: on the stall-watchdog abandon, FORFEIT hung
  battles (`/forfeit` to the room) + `/cancelchallenge` (verified vs poke-env + bundled Showdown source).
  Called from `play_pairing`'s stall branch.
- **22f (.1–.5) — MULTI-SERVER architecture.** `run_local_battle`: `start_showdown_on(port)` + `ServerPool`
  (K servers on 8000..8000+K-1, `port_for_worker` round-robin, `recycle`, `stop_all`) + `localhost_server_config(port)`;
  `make_player(port=)`/`_make_opponent(port=)`/`_sp` bind per-server. `mp_collect`/`mp_eval` `(ports=)` spread
  worker batches round-robin (port in the payload). `generation`: **CLI `--servers N`** (default 1 = byte-identical
  single-server), pure `server_recycle_index(done, every, K)` STAGGERED recycle. **Live A/B validated** (`--servers 2`
  ran both servers w/ battle traffic on BOTH + staggered recycle, clean).
- **Memory-leak fix (adversarial Workflow, 5 auditors → 10 confirmed → 1 real RAM leak):** poke-env interns a
  unique-named logger+StreamHandler per player FOREVER in `logging.Logger.manager.loggerDict`; 22d's single-use
  usernames leak ~150-300/gen/worker. **`VGCPlayerBase.close()` now evicts it** (removeHandler+close+loggerDict.pop,
  guarded). Empirically verified.
- **Type_D removal (user ask):** deleted ALL Type_D replay-saving (`write_type_d`/`render_replay_html`/`prune_type_d`/
  `TYPE_D_DIR` in archive.py) + the whole `showcase`/`_proto_log` plumbing it fed → **`collect_with_pool`/
  `collect_with_league` now return 2-tuples `(trajs, source_counts)`**; `write_generation_artifacts` = manifest-only.
- **19b CLOSED** (gate_sim calibration said don't build it). `gate_sim.py` staged-mirror model
  (`mirror_gate_verdict`/`mirror_escalate_mask`/`staged_mirror_stats`, `--staged` demo) STAYS as the record — DO NOT
  REVERT. Finding: staged short→full mirror saves ~0% under decision-equivalence (the 0.55 bar sits too close to the
  ~0.50 mirror rate); ~10-13% only by relaxing equivalence — not worth it now eval is mp+multi-server.
- **Showdown-native replay emitter (user ask: less poke-env):** NEW `v_dance/selfplay/replay_html.py`
  (`render_replay_html` → canonical Type_B HTML from the raw `|`-log + CDN replay-embed.js; `battle_replay_lines` =
  `["|".join(sm) for sm in battle._replay_data]`). `status.LiveBattles.save_html_replay` now renders THAT instead of
  poke-env's `battle.save_replay`. Covers gauntlet eval + collection; live spectate untouched; **no mem leak** (log is
  transient, only the .html persists; user said LEAVE ALL saved replays, no disk cap). Animates via CDN (user always
  has internet). **Offline-animated vendoring was PoC'd (worked!) then ABANDONED; `scratch/replay_poc` DELETED.**

## ⇒ IMMEDIATE NEXT
1. **The overnight is resuming from snap_gen57** (user runs it). Let it train (he's not satisfied with the AI yet).
   If asked, assess the training trajectory (manifest Elo curve / scripted WR / gate verdicts / KL+EV health) to
   judge improving-vs-plateauing.
2. **THE BACKLOG** (P-ordered): 15a TP bring-diversity DIAGNOSTIC FIRST (the "never brings 2 mons" concern is
   HYPOTHETICAL — verify before building); 15b full outcome-driven TP co-dev (gated on a competent policy); 16 demo
   eps; 17 Reg M-B migration (~2026-06-24); 21 team-builder AI.

## Full to-do list (carry forward + REPRINT/UPDATE every message)
| # | Task | Effort | Priority | Status |
|---|------|--------|----------|--------|
| 1–11 | Phase 0/1/2 gate (v2 ladder + HoF anti-cycle) | — | P1/P2 | ✅ committed (prior) |
| 13–18b | parallelism / multicore / resume / spectate | — | P3/P4 | ✅ done (uncommitted) |
| A–E, G, H/H2 | HTML replays / Ctrl-C / archive / dashboard | S–L | P1/P2 | ✅ done + tested |
| 22 / 22c | Stall watchdog / server recycle | M / S | P2 | ✅ done + live-verified |
| 22d / 22e | Per-gen account names / abandon cleanup | S / S | P2/P3 | ✅ done + live-verified |
| 22f (.1–.5) | Multi-server architecture (`--servers N`) | L | P4 | ✅ done + tested + live-validated |
| Leak-A | poke-env per-player logger eviction (RAM leak) | S | P2 | ✅ done + tested |
| Type_D-rm | Remove Type_D replay-saving + showcase plumbing | M | P3 | ✅ done + tested |
| 19b | Staged short→full mirror | — | P4 | ✅ CLOSED (calibration → not worth it) |
| Replay | Showdown-native replay emitter (drop poke-env save_replay) | S | P3 | ✅ done + tested |
| 12 | 🌙 Overnight run — resume from snap_gen57 | — (user) | P2 | ▶ resuming (user not satisfied w/ AI) |
| Leak-C | Saved `--save-replays` HTML disk growth | XS | — | 💤 user: LEAVE ALL (no cap) |
| 15a | TP bring-diversity DIAGNOSTIC (verify the hypothetical first) | S–M | P5 | ⬜ |
| 15b | Full outcome-driven TP co-development (§14) | L | P5 | ⬜ (gated on competent policy) |
| 16 | 3c.7d scripted demo episodes | M | conditional | ⬜ deferred |
| 17 | Reg M-B migration | L | GATED ~2026-06-24 | ⬜ blocked |
| 21 | Team-builder AI — generate a full format-legal VGC team | L | P6 | ⬜ future |

## Environment + commands (Windows; user runs Git Bash + PowerShell; Bash + PowerShell tools available)
- venv: `.venv/Scripts/python.exe` (bare `python` in Bash lacks deps), venv node `.venv/node/node.exe`.
  **Run all from the REPO ROOT** `D:\ShowdownProject\Victory-Dance` (Bash CWD is `data/scripts/vod_parser` →
  `cd /d/ShowdownProject/Victory-Dance` first).
- Tests: `.venv/Scripts/python.exe -m pytest tests -q` (**1024 pass / 0 skip**). `PYTHONIOENCODING=utf-8` when capturing.
- Dry-run (no server): `python -m v_dance.selfplay.generation --dry-run --generations 8`.
- gate_sim calibrations: `python -m v_dance.selfplay.gate_sim --demo` / `--staged` (the 19b record).
- Dashboard: `python -m v_dance.datatools.dashboard_server --port 5175 --archive artifacts/self_play_archive`.
- Resume overnight: the command in WHERE-WE-LEFT-OFF. Latest snapshot = snap_gen57.

## Gotchas / standing facts (★ = this session)
- ★ **Multi-server (22f)** is the fundamental fix for the single-server saturation/bloat — use `--servers 2` w/
  `--collect-procs 6 --collect-async 2 --restart-server-every 20`. K=1 (default) = byte-identical single-server.
- ★ **Replays are now Showdown-native** (`replay_html.render_replay_html`), NOT poke-env. poke-env's `save_replay`
  is GONE from our code. The live in-dashboard spectate is the dashboard's OWN text `renderTurnViewer`; the animated
  `▶ Watch` link → `localhost:8000/<tag>` redirects to the CDN client (no local client bundle). Offline-animated
  needs vendoring the Showdown client+sprites (PoC recipe in the memory note) — NOT done (user always has internet).
- **GATE = v2 frozen-champion ladder + Phase-2 HoF** ([[gate-redesign-2026-06-17]]); champion lineage gen0→11→21→…
- **STATE_DIM 1866 / LAYOUT v4 / ACTION_DIM 16 / GIMMICK_DIM 2** (retrain forced ONLY by a layout/action/gimmick
  bump — nothing on the to-do does that; Reg M-B = a warm-started fine-tune). ENV PINNED (`PINS.md`): poke-env
  @a6e4f67, Showdown @ecf39eef1. ⚠ Reg M-B due ~2026-06-24.
- **Production = BC v4; the DEFAULT `checkpoints/bc_best.pt` == `checkpoints_v4/bc_best.pt`.** Resume with the SAME
  (default) ckpt; don't switch mid-run.
- **Diagnostics:** `scratch/mp_collection_probe.py`, `scratch/_hof_smoke.py`, `scratch/validate_teams.py`,
  `tests/_parity_harness.py`. The adversarial Workflow is the standard tool for subtle bugs + design.
