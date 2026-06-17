# Next-session handoff — Victory-Dance VGC bot: continue the PPO self-play build (3b → 3c)

You are Opus 4.x continuing the Victory-Dance VGC Pokémon-Showdown bot, in **ULTRACODE mode**
(author/run Workflows for substantive tasks; adversarially verify findings; token cost is not a
constraint — optimize for the most correct, exhaustive answer).

## STEP 0 — read auto-memory FIRST
Read `C:\Users\death\.claude\projects\D--ShowdownProject-Victory-Dance\memory\MEMORY.md`, then the notes
it points to. **Most relevant right now:**
- **`ppo-reward-design-2026-06-16` — THE RESUME POINTER. READ FULLY.** Has the implementation progress
  (3a data layer + 3b.2 GAE DONE), the exact NEXT, the file list, and the gotchas.
- **`docs/ppo_reward_design.md` (§1–19) — the RL design BIBLE.** Reward, critic, GAE, gated PBRS,
  self-play structure, edge-case "must-not-suppress" charter, team sampling, generations, resumability,
  Type_D archive — and **§19 = the 3a/3b/3c/v1 work breakdown** this build follows.
- `available-switches-desync-fix-2026-06-16` — the 100%-MODEL-DRIVEN work (the reason self-play
  trajectories are clean — every live order is legal + accepted).
- `item-ability-staterep-2026-06-16` — v3 model promoted (STATE_DIM **1854**, layout v3).
- `victory-dance-project-layout` — paths, env quirks, the 2026-06-16 cleanup.

## Working style — the USER's required cadence (follow EXACTLY)
- **Sub-problems, ONE at a time. PAUSE after each** — report what you did + the evidence, then wait.
- **After EVERY message give an UPDATED TO-DO LIST with an EFFORT column** (S/M/L + rough time). Non-negotiable.
- **Unit-test every change.** Keep all tests green — currently **621 pass / 0 skip**:
  `.venv/Scripts/python.exe -m pytest data/scripts ai_train_scripts -q`.
- **Do NOT retrain / re-export without explicit permission.** The RL work is new code / serve-side.
- **Judge model STRENGTH on WIN-RATE (the gauntlet), not val top1.**
- **The USER runs the live stuff** (gauntlet, run_local_battle, any self-play smoke needing the local
  Showdown server) — give them the exact command + what to look for. Don't run live battles yourself.
- **Update memory** when meaningful work lands; checkpoint progress into the `ppo-reward-design` note so a
  fresh session can always resume.

## Where things stand (the catch-up)
- **100% MODEL-DRIVEN achieved + confirmed at scale** (slot-based switch order + Pattern-A replacement +
  cross-slot mega dedup). The live bot emits only legal, accepted orders → clean self-play trajectories.
- **v3 model PROMOTED** to production `ai_train_scripts/BC_model/checkpoints/bc_best.pt` (STATE_DIM 1854 /
  layout v3 / value_trained + gimmick_trained). Rollback: `bc_best.PRE_V3_BACKUP.pt`.
- **Repo cleaned** (2026-06-16): `replay_buffer/` emptied (gitignored transient; RL repopulates), ~26
  one-off debug harnesses archived to `Delete_When_Project_Done/`, 4 reusable kept.
- **RL DESIGN COMPLETE** — `docs/ppo_reward_design.md` §1–19.
- **RL IMPLEMENTATION: 3a.1–3a.5 (data layer) + 3b.2 (GAE) DONE + tested** (3a.6 Phase-0 harness still
  pending — it needs the live server, so it's folded into 3c.1), all **torch-free**, in
  `local_battle/self_play/`:
  - `schema.py` (Transition / EpisodeMeta / Trajectory / TerminalType; logprob+value recorded at
    collection time; TP decision recorded), `collector.py` (TrajectoryCollector + `assert_zero_sum` §6
    symmetry guard), `reward.py` (`place_terminal_reward` + MODEL-DRIVEN% hard-fail + `prepare_batch`),
    `store.py` (Type-C jsonl store/converter + `assert_terminal_rewards_clean`), `diagnostics.py`
    (per-bring/per-matchup win-rates), `gae.py` (`compute_gae`/`standardize`/`compute_batch_gae`,
    γ=0.997 floor, horizon-cut bootstrap).
  - Tests (47): `data/scripts/tests/test_selfplay_{schema,collector,reward,store,diagnostics,gae,integration}.py`.
    `test_selfplay_integration.py` exercises the FULL chain. Full suite = **621 pass / 0 skip**.

## ⇒ TASK: continue the PPO build — finish 3b (torch, offline) then 3c (live)

The torch-free data + advantage layer is done. Remaining:

**3b — PPO trainer** (torch; couples to `ai_train_scripts/BC_model/bc_model.BCPolicy` +
`local_battle/model_io.py`; **OFFLINE / unit-testable — no live server needed**):
- **3b.1** Actor-critic init FROM the BC checkpoint — **cloned SEPARATE critic** (copy trunk+value_head
  from BC weights so policy-gradient drift can't wreck the calibrated value surface; doc §2). Load via
  `model_io`.
- **3b.5** Per-head forward → joint log-prob of (a0,g0,a1,g1) under the policy (2 slot heads + gimmick
  head), masked with `build_legal_action_mask` (serve parity) — covering BOTH normal turns AND
  `decision_type=="replacement"` steps (switch-only `build_replacement_mask`). Needed to recompute the
  new-log-prob for the PPO ratio.
- **3b.3** PPO clip loss + value loss (BCE win-prob in Phase 1; Huber on shaped return in Phase 3) +
  entropy bonus + **KL-to-BC prior** penalty; value-clip on.
- **3b.4** Critic-only warm-up (freeze actor K updates → small-LR actor) + warm-start-collapse guards
  (KL-from-BC / explained-variance auto-halt).
- **3b.6** Single value-space assertion (terminal / bootstrap / target same numeric space).
- **3b.7** (GATED, default OFF) PBRS: separate frozen-Φ snapshot module (zero-grad, `id != critic`),
  `F = γ·Φ(s′) − Φ(s)`, **`Φ(terminal)=0` unit-test (`shaped_r == terminal_r`)**, shaping-fraction cap
  <0.3 + λ-anneal, frozen-Φ edge-state validation (doc §13).

**3c — Self-play loop** (LIVE; needs the local Showdown server; **USER runs smokes**):
- **3c.1** Game runner — wire 2 model players on local Showdown, feed the collector (`add_step` with
  logprob+value), write the Type-C store. **Also unblocks 3a.6** (Phase-0 harness: ≥200 games,
  MODEL-DRIVEN ≥99%, symmetry holds, p1/p2 ~50%, terminal-space clean). + the moved §13 env asserts
  (no VecNormalize/reward-wrapper; live masking-confirmation keeps edge moves legal).
- **3c.2** Opponent league (frozen snapshots ~50% latest / ~30% past-accepted PFSP / ~20% scripted
  anchors; reuse `gauntlet._make_opponent`).
- **3c.3** Generation loop → gauntlet eval (≥4 teams, side-balanced) → statistical promotion gate →
  league admission + Φ-snapshot refresh.
- **3c.4** Resumability (resume snapshot = weights+critic+optimizer+gen-counter+RNG+league+Elo+team-cursor;
  heartbeat + graceful shutdown; chunked == continuous) — **the "can't run 24/7 on a personal PC"
  requirement; essential.**
- **3c.5** Generation archive + **Type_D Showdown-HTML replays** (offline-vendored player JS/CSS + raw
  `|`-log → `data/.../Type_D/gen_<N>_<tag>`; fixed showcase opponent; Elo-vs-generation curve).
- **3c.6** Metrics/logging (doc §7). **3c.7** Exploration seeding (KL-to-BC + scripted demos + archetype
  injection — doc §12).

**v1 — TP co-development** (after 3c stable AND the battle policy is competent at the archetypes — doc §14
ordering rule): alternating best-response (freeze battle → train TP → freeze TP → battle), gauntlet-gated.

**Recommended order:** finish **3b** first, THEN **3c.1 + 3a.6** (live; user runs the smoke), then the
rest of 3c, then v1. **Why 3b before 3c:** 3b is offline / fully unit-testable — the agent can complete +
verify it ALONE, with no pause for the user — whereas 3c needs the live Showdown server the USER must
drive. Front-loading the self-testable work maximizes autonomous progress.

## Environment + commands (Windows; user runs Git Bash + PowerShell; Bash tool available)
- venv python (torch/poke-env): `.venv/Scripts/python.exe` — the PATH `python` lacks the ML deps. In Git
  Bash use FORWARD slashes.
- Tests: `.venv/Scripts/python.exe -m pytest data/scripts ai_train_scripts -q` (621 pass / 0 skip).
- Self-play code lives in `local_battle/self_play/`; tests in `data/scripts/tests/test_selfplay_*.py`.
  Modules import as `self_play.<mod>` — the import root is `local_battle/` (pytest/conftest already adds
  it to sys.path; a standalone torch script must add `local_battle/` to PYTHONPATH).
- Gauntlet (USER runs): `.venv/Scripts/python.exe local_battle/gauntlet.py --battles 30 --teams Trickery
  Kronomono3 WolfeGlick team1 --ckpt ai_train_scripts/BC_model/checkpoints/bc_best.pt -v 2> logs/run.log`
  — judge MODEL-DRIVEN% (want 100%) + scripted win-rate. **Set `PYTHONIOENCODING=utf-8` if capturing
  stdout** (the report prints unicode → cp1252 UnicodeEncodeError otherwise).

## Full to-do list (effort: S ≈ 1 step · M ≈ 2–3 steps · L ≈ multi-step / spans a session)

| # | Task | Effort | Status |
|---|---|---|---|
| 3a.1–3a.5 | Data layer: schema → collector → reward → store → diagnostics | — | ✅ DONE |
| 3b.2 | GAE advantage/return core | S | ✅ DONE |
| 3b.1 | Actor-critic init from BC (cloned separate critic) | M | next (torch, offline) |
| 3b.5 | Per-head forward → joint masked log-prob (2 slot + gimmick) | M | pending (torch) |
| 3b.3 | PPO clip + value + entropy + KL-to-BC losses | M | pending (torch) |
| 3b.4 | Critic-only warm-up + warm-start-collapse guards | M | pending (torch) |
| 3b.6 | Single value-space assertion | S | pending |
| 3b.7 | Gated PBRS (frozen-Φ, Φ(terminal)=0 test, cap+anneal) | M | pending |
| 3c.1 + 3a.6 | Live game runner + Phase-0 harness (**needs local Showdown; USER runs smoke**) | L | pending |
| 3c.2 | Opponent league (snapshots + PFSP + scripted anchors) | M | pending |
| 3c.3 | Generation loop + gauntlet eval + statistical promotion gate | M | pending |
| 3c.4 | Resumability (resume snapshot + heartbeat + graceful shutdown) | M | pending |
| 3c.5 | Generation archive + Type_D Showdown-HTML replays | M | pending |
| 3c.6 | Metrics/logging dashboard | S | pending |
| 3c.7 | Exploration seeding (KL-to-BC + demos + archetype injection) | M | pending |
| v1.* | TP co-development (alternating best-response) | L | pending |
| prereq | Pin poke-env/Showdown SHA before long multi-session runs | S | pending |

## Gotchas / standing facts
- **STATE_DIM 1854, STATE_LAYOUT_VERSION 3, ACTION_DIM 16, GIMMICK_DIM 2.** Tera = placeholder only.
- Production policy = base BC (v3); the **trained value head is the PPO critic init** (the big
  sample-efficiency lever). 100% model-driven holds → clean trajectories.
- **γ = 0.997 is a FLOOR (never lower).** Terminal reward ±1 zero-sum. **PBRS is gated OFF until a
  measured stall** (doc §4 phased rollout).
- **Edge-case charter (doc §11):** the terminal reward handles ally-boost / Perish-stall / sacrifice /
  Trick-Room strategies — diagnostics (ally-damage, stall, Protect rate…) are **NEVER** reward terms;
  rare strategies are an **EXPLORATION** problem (seed via KL-to-BC + demos + archetype injection), not a
  reward problem.
- **Self-play:** both perspectives per game (`assert_zero_sum`), random M-A teams both sides + both-side
  paired sampling, **frozen TP net for v0**, **the gauntlet is the only trustworthy progress metric**
  (in-league Elo misleads under non-transitivity).
- USER runs self-training on a **personal PC that can't be 24/7** → resumability (3c.4) is essential;
  chunked == continuous with seeded RNG.
- poke-env on git master; **pin its SHA before long runs** (prereq to-do).
- Reusable diagnostics kept live: `data/scripts/tests/_parity_harness.py` (encoder parity),
  `local_battle/_smoke_zoroark.py`, `_ab_headtohead.py`, `_diag_rejections.py`. Archived one-offs in
  `Delete_When_Project_Done/`.
