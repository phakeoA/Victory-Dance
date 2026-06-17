# Next-session handoff — Victory-Dance VGC bot: PPO self-play build (3c.6 dashboard DONE → 3c.7 exploration → …)

You are Opus 4.x continuing the Victory-Dance VGC Pokémon-Showdown bot, in **ULTRACODE mode**
(author/run Workflows for substantive tasks — review/audit/research; adversarially verify findings;
token cost is not a constraint — optimize for the most correct, exhaustive answer). Solo on
conversational/trivial turns.

## STEP 0 — read auto-memory FIRST
Read `C:\Users\death\.claude\projects\D--ShowdownProject-Victory-Dance\memory\MEMORY.md`, then the notes
it points to. **THE RESUME POINTER:**
- **`ppo-reward-design-2026-06-16` — READ FULLY.** Its PROGRESS section is the running log of every
  sub-problem done (3a → 3b → 3c.1–3c.5 → restructure Stage 0), the exact NEXT, file lists, gotchas, and
  the two LIVE bugs the user's smokes caught + fixed.
- **`docs/ppo_reward_design.md` (§1–20) — the RL design BIBLE.** §19 = the 3a/3b/3c/v1 work breakdown;
  §20 = compute throughput + resource caps (3c.8).
- `victory-dance-project-layout`, `item-ability-staterep-2026-06-16`, `available-switches-desync-fix-2026-06-16`.

## ⇒ THE USER's REQUIRED CADENCE — follow EXACTLY (this is how the whole build has gone)
- **Sub-problems, ONE at a time. PAUSE after each** — report what you did + the EVIDENCE (test counts,
  numbers), then WAIT for the user. Do not chain multiple sub-problems unless the user explicitly says so.
- **After EVERY message give an UPDATED TO-DO LIST with EFFORT (S/M/L) + PRIORITY + DEPENDENCE columns +
  a Status column.** Non-negotiable — the user relies on this every single turn.
- **Unit-test every change.** Keep ALL tests green — currently **769 pass / 0 skip**:
  `.venv/Scripts/python.exe -m pytest tests -q` (run from repo root; `testpaths=tests` in pyproject so bare
  `pytest` works too). Add a regression test for every bug found.
- **Give a MANUAL test for each feature too** (the user runs things + reviews): an OFFLINE demo (`--dry-run`
  / `--demo` / a tiny script that prints interpretable output) AND, for live features, the exact command +
  what-to-look-for. The user LIKES running things and re-running until clean.
- **The USER runs the live stuff** (the local Showdown server: gauntlet, run_local_battle, the self-play
  game runner, generation `--live`). Give the exact command + pass criteria. **Don't run live battles
  yourself.** When a live smoke surfaces a bug, fix it + add a regression test (this has happened twice and
  both were real bugs the offline tests missed — the value-clip-throttle and the dropout/save + silent-eval).
- **Do NOT retrain / re-export without explicit permission.** All RL work so far is new code / serve-side.
- **Judge model STRENGTH on WIN-RATE (the gauntlet), not val top1.**
- **Checkpoint progress into the `ppo-reward-design-2026-06-16` memory note** when meaningful work lands, so
  a fresh session can always resume. Update `MEMORY.md` index hooks too.

## Where things stand (the catch-up — ALL of 3a/3b + 3c.1–3c.5 DONE, live-verified)
The full offline RL stack AND the live self-play loop are built and verified end-to-end:
- **3a.* / 3b.1–3b.7** (data layer, GAE, actor-critic, log-prob, PPO loss, warm-up+collapse-guards,
  value-space, gated PBRS) — all in `local_battle/self_play/`, unit-tested.
- **3c.1a/b/c + 3a.6** — recording player + live runner + de-dup + Phase-0 harness. LIVE-VERIFIED:
  clean/legal/symmetric, **100% model-driven** on a clean file, 0 duplicate steps.
- **3c.2** league (PFSP + anchor decay), **3c.3 + 3c.3b** generation loop (gate + history + live wiring).
  LIVE-VERIFIED: one full generation runs collect→PPO update→gauntlet eval→promotion gate→admit/save
  (gen0 scripted 46.7%, update EV 0.843 / kl 2.1e-3 / clip 0.02 / halted=False).
- **3c.4** resumability — snapshot (weights+critic+opt+RNG+league+history) + Ctrl-C/heartbeat;
  gold test `chunked==continuous` (bit-identical weights after resume) PASSES.
- **3c.5** archive + Type_D replays (CORRECTED per user): `manifest.json` (data for the future dashboard,
  NO static graph) + **Type_D = real Showdown replay HTML** (battle-log-data + replay-embed.js, the
  `pokemon-showdown/test/common.js saveReplay` format) → `data/vods/Type_D/`, VERIFIED parser-ingestible.
- **Restructure Stage 0 DONE:** all runtime outputs consolidated under gitignored **`artifacts/`**
  (`artifacts/{self_play_archive,replay_buffer,logs,eval_results}`). Root is clean. (User redirects should
  now use `2> artifacts/logs/...`.)

## ⇒ IMMEDIATE NEXT TASK: **3c.7 exploration seeding** (3c.6 dashboard DONE & live-verified)
**3c.6 DASHBOARD DONE 2026-06-17 (769 pass / 0 skip; user live-verified — spectated a real battle + gen0
promoted, charts/bar driven by the real run).** Browser dashboard `data/scripts/dashboard/{dashboard.html,
.css,.js}` (team_builder-styled): Overview (cards + auto notables + Elo/win-rate SVG charts) · PPO/Critic
Health (small-multiples) · 🎬 Spectate (live turn-viewer) · Generation Detail. Backend:
`v_dance/selfplay/status.py` (`LiveStatus` → `status.json` + `live_log.json`), `v_dance/datatools/
dashboard_server.py` (Flask serves dashboard + manifest/status/live_log, no-cache), enriched `build_manifest`,
and `run_live_generations`/`collect_with_league`/`SelfPlayVGCPlayer._report_active` write the live feed.
**Spectator: the local-Showdown iframe was DROPPED** (the server only JS-redirects to the psim.us client →
needs internet + frame-blocked) → **in-dashboard TURN VIEWER** parses the live |-log (split board / HP / boosts
/ field / action-log / turn scrubber) + a **▶ Watch-in-tab** button (open-↗ to the real animated client).
**RUN (TWO separate terminals — the server blocks):** T1 `python -m v_dance.datatools.dashboard_server
--port 5175`; T2 `python -m v_dance.selfplay.generation --live --generations 1 --games 30 --eval-battles 10
-v 2> artifacts/logs/gen.log`; open http://127.0.0.1:5175/. Dev preview w/o a run:
`python scratch/_demo_live_status.py`.

**3c.7 = exploration seeding** — KL-to-BC prior + scripted demo episodes + **archetype injection** into the
opponent league (Perish-trap / Trick-Room / setup / mega cores, incl. Intimidate-vs-Defiant) + a collection
`tau` knob. Goal: DISCOVER the hard-to-find tactics sparse self-play can't (the reward already rewards them —
this is exploration, not reward; see [[ppo-reward-design-2026-06-16]] charter §11). Then 3c.8 throughput, v1 TP.

**The codebase (post Stage 2):** real installable package **`v_dance/`** (named `v_dance` not `victory_dance` — that's a real in-game move):
```
v_dance/  encoders/ parser/(vod_parser/) models/ training/ play/ eval/ selfplay/ datatools/   ← pip install -e .
tests/                    ← ALL 51 test files (was data/scripts/tests + vod_parser + ai_train); bare `pytest` finds them
data/                     ← PURE DATA (vods incl. Type_D, pokedex.json, moves.json, pikalytics, teams, team_builder UI)
data/scripts/scrapers/    ← standalone scrape/update scripts (scrape_replays, scrape_pikalytics, scape_items, update_moves, update_pokedex) — tooling, not library; import v_dance.* via editable install
artifacts/                ← runtime outputs (Stage 0)
ai_train_scripts/{BC_model,teamPreview_model}/checkpoints/   ← checkpoints STAYED here (production bc_best.pt)
scratch/(_smoke_*,_ab_*,_diag_*,_demo_offline)   pokemon-showdown/  docs/
```
~120 files git-mv'd, 371 imports → absolute `v_dance.`, 291 dead sys.path lines scrubbed. Import-smoke 58/58
modules; 4 offline demos green. **NOW DO 3c.6** (the dashboard, see the to-do table).

3c.6 = **Metrics/logging dashboard** — html/js/css UI reading `artifacts/self_play_archive/manifest.json` + a
per-gen metrics file (§7 diagnostics: PPO/critic health, MODEL-DRIVEN%, action-mix, sacrifice rate, p1/p2,
ep-length), updates over time. Then 3c.7 exploration, 3c.8 throughput, v1 TP.

## Full to-do list (effort: S ≈ 1 step · M ≈ 2–3 · L ≈ multi-step/session)
| # | Task | Effort | Priority | Depends on | Status |
|---|---|---|---|---|---|
| restruct Stage 2 | Full `v_dance/` package + pyproject + killed 291 sys.path hacks + tests/ | L | DONE 2026-06-17 | Stage 0 | ✅ |
| 3c.6 | **Metrics/logging dashboard + live + spectate** — `data/scripts/dashboard/` UI + `status.py` + `dashboard_server.py`; live polling, in-gen progress, turn-viewer spectator | L | DONE 2026-06-17 | 3c.3 | ✅ live-verified |
| 3c.7 | Exploration seeding (KL-to-BC + scripted demos + **archetype injection** incl. Intimidate-vs-Defiant) + collection `tau` | M | **NEXT (P3)** | 3c.2 | ⬜ |
| 3c.8 | Throughput / hybrid GPU+CPU + **resource caps** (doc §20: max CPU cores/frac, max VRAM GB) | M–L | P4 (after a throughput MEASUREMENT) | 3c.3 | ⬜ |
| v1.* | TP co-development (alternating best-response, gauntlet-gated) | L | P5 | 3c stable + archetype-competent | ⬜ |
| prereq | Pin poke-env / Showdown SHA before long runs | S | before long runs | — | ⬜ |
| state-rep #A | TP-net ability feature (open-sheet ability → `mon_dex_features`) — from the ability audit | S | batch w/ next re-export+retrain | — | ⬜ |
| state-rep #B | Split Defiant/Competitive into a dedicated `stat-drop-punish` ability category (layout bump) | S | batch w/ next re-export+retrain | — | ⬜ |
| optional | Model Choice-lock/Encore in `build_legal_action_mask` for literal 100% model-driven | S | only if it hurts training | — | ⬜ |

## Environment + commands (Windows; user runs Git Bash + PowerShell; Bash tool available)
- venv python (torch/poke-env): `.venv/Scripts/python.exe` — PATH `python` lacks ML deps. Git Bash = FORWARD slashes.
- Tests: `.venv/Scripts/python.exe -m pytest tests -q` (769 pass / 0 skip; run from repo root, `testpaths=tests`).
  Set `PYTHONIOENCODING=utf-8` when capturing stdout (unicode → cp1252 error otherwise).
- Self-play code: `v_dance/selfplay/`; tests `tests/test_selfplay_*.py`. Imports as `v_dance.selfplay.<mod>`
  (editable install — `pip install -e .` already done; no path hacks). Run a module: `python -m v_dance.selfplay.<mod>`.
- **USER live smokes (give the command + pass criteria; user runs):**
  - Phase-0: `.venv/Scripts/python.exe -m v_dance.selfplay.game_runner --games 200 --teams team1 WolfeGlick Kronomono1 Kronomono3 -v 2> artifacts/logs/phase0.log` → want 0 dup steps, MODEL-DRIVEN≥99%, symmetry 0, p1≈50%, terminal clean.
  - Live generation: `.venv/Scripts/python.exe -m v_dance.selfplay.generation --live --generations 1 --games 20 --eval-battles 10 -v 2> artifacts/logs/gen.log` (gen0 auto-promotes; resume with `--resume artifacts/self_play_archive/resume.pt`; `--generations 0`=until-Ctrl-C; `--hours N`).
  - **Dashboard (3c.6, live metrics + spectate):** TWO terminals (the server BLOCKS — don't paste both in one). T1: `.venv/Scripts/python.exe -m v_dance.datatools.dashboard_server --port 5175`; T2: the live generation above; open `http://127.0.0.1:5175/`. Watch the LIVE bar + Spectate turn-viewer follow a real game. `▶ Watch in tab` opens the real animated client (needs internet). Dev preview w/o a run: `.venv/Scripts/python.exe scratch/_demo_live_status.py`.
- Offline demos (no server): `python scratch/_demo_offline.py` (full 3a/3b chain on real data),
  `python -m v_dance.selfplay.league --demo`, `… generation --dry-run`, `… archive` (writes a sample Type_D + manifest).

## Gotchas / standing facts
- **STATE_DIM 1854, LAYOUT v3, ACTION_DIM 16, GIMMICK_DIM 2.** Production policy = base BC (v3) at
  `ai_train_scripts/BC_model/checkpoints/bc_best.pt` (value_trained + gimmick_trained, **dropout=0.1** —
  the save-checkpoint must preserve it). Tera = placeholder only.
- **Value space = `value_pm` ∈ [−1,1]** everywhere (`2σ−1`); `Transition.value` stores value_pm, NOT win-prob.
- **γ=0.997 FLOOR; PBRS gated OFF; reward = terminal ±1 only** (charter §11: nothing the agent does to its
  OWN side is ever a reward term). Intimidate-into-Defiant etc. handled by the terminal reward, BUT the
  STATE-REP gap (Defiant/Competitive not distinct; TP net sees no abilities) = state-rep #A/#B.
- **Collection runs stochastic (`tau`>0)** so the behaviour log-prob is real; the gimmick is argmax (tiny
  v0 approximation). De-dup keeps only the EXECUTED model decision per turn.
- Live integration is the bug-prone surface — the user's live smokes have twice caught bugs offline tests
  missed. ALWAYS hand the user a live smoke after a live-touching change.
- Reusable diagnostics: `tests/_parity_harness.py` (encoder byte-parity), `scratch/_smoke_zoroark.py`,
  `scratch/_ab_headtohead.py`, `scratch/_diag_rejections.py`, `scratch/_demo_offline.py`.
