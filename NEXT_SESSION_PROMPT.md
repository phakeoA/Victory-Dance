# Next-session handoff — Victory-Dance VGC bot (finish #11 precisely, then the RL path)

You are Opus 4.8 continuing the Victory-Dance VGC Pokémon-Showdown bot, in **ULTRACODE mode**.

## STEP 0 — read auto-memory FIRST
Read `C:\Users\death\.claude\projects\D--ShowdownProject-Victory-Dance\memory\MEMORY.md`, then the notes
it points to. **Most relevant right now:**
- `available-switches-desync-fix-2026-06-16` — **#11 (the task this session finishes). READ IT FULLY** — it
  has the precise diagnosis + the instrumented DIAG + the likely fix.
- `item-ability-staterep-2026-06-16` — the #5/#6/#7 encoder batch + the #9 re-export+retrain (DONE).
- `gauntlet-eval-2026-06-16` — the win-rate gauntlet (judge on win-rate, not val top1).
- `dont-retrain-until-told-2026-06-14` — the user's retrain workflow (IMPORTANT).

## Working style — the USER's required cadence (follow EXACTLY)
- **Give an UPDATED TO-DO LIST after EVERY message** (with effort / priority / dependence). Non-negotiable.
- **One task at a time. PAUSE after each** — report what you did + the evidence, then wait.
- **Do NOT retrain/re-export without explicit permission.** (The #11 work below is SERVE-SIDE — no retrain.)
- **Unit-test every change.** Keep all tests green (currently **556 pass / 7 skip**:
  `.venv/Scripts/python.exe -m pytest data/scripts ai_train_scripts -q`).
- **Judge model changes on WIN-RATE (the gauntlet), not val top1.**
- **The USER runs the live stuff** (`gauntlet.py`, `run_local_battle.py`) — give them the command + what to
  look for; don't run gauntlet.py yourself unless asked.
- **Update auto-memory** when you finish meaningful work.

## Environment (Windows; user runs Git Bash MINGW64 + PowerShell; Bash tool available)
- Repo root `D:\ShowdownProject\Victory-Dance`. venv python (torch/poke-env): `.venv/Scripts/python.exe`.
  **In Git Bash use FORWARD slashes** and the explicit `.venv/Scripts/python.exe` (the user's `python`
  resolves to system Python without the ML deps; their `(.venv)` prompt is misleading).
- Tests: `.venv/Scripts/python.exe -m pytest data/scripts ai_train_scripts -q`
- Gauntlet (USER runs): `.venv/Scripts/python.exe local_battle/gauntlet.py --battles N --teams <names> --ckpt <ckpt> [-v --spectate]`

## Where things stand (the catch-up)
- **Encoder batch #5 (item/ability effect-flags) + #6 (is_spread move flag) + #7 (real PP) — DONE +
  re-exported + retrained.** STATE_DIM **1854**, STATE_LAYOUT_VERSION **3**. New model at
  `ai_train_scripts/BC_model/checkpoints_v3/bc_best.pt` (val top1 0.394 ≈ prior; value win-acc 0.62; config
  state_dim 1854 / layout v3 / value_trained + gimmick_trained). **NOT promoted** — production `bc_best.pt`
  is still the OLD 1398 model (backed up `bc_best.PRE_V3_BACKUP.pt`; layout-INCOMPATIBLE with the 1854
  encoder, so the 7 model_io tests SKIP and there's no `--prev-best` A/B vs it).
- **#11 (poke-env `available_switches` Illusion desync) — #11a/b/c shipped but the FORCED-SWITCH-UNDER-
  ILLUSION case is NOT fully fixed.** This is THE task this session. See the memory note for the full
  precise diagnosis. The simulator is FUNCTIONALLY fine (legal play, no hangs, normal win-rates) — this is a
  rare-endgame quality fix, not a correctness fix.

## ⇒ TASK THIS SESSION: finish #11 precisely (serve-side, NO retrain)

**The bug:** in a Trickery (Zoroark+Ditto) double faint, poke-env's `available_switches` both DROPS the real
brought benchers (zoroark+ditto) AND wrongly LISTS an un-brought mon (basculegion). `build_replacement_mask`
now offers all `own_bench_mons`, but `own_bench_mons` itself returns the WRONG set under illusion, so the
model still falls to `/choose default`.

**STEP 1 — get the ground truth (the DIAG is already instrumented in `vgc_base._log_force_switch_state`):**
Ask the USER to run, then send you the grep output:
```
.venv/Scripts/python.exe local_battle/gauntlet.py --battles 8 --teams Trickery --ckpt ai_train_scripts/BC_model/checkpoints_v3/bc_best.pt -v 2> trickery_v3.log
grep "forceSwitch DIAG" trickery_v3.log
```
The DIAG now prints `own_bench_mons=[...]`, `request_state=...` (None ⇒ the request path fell back to flags),
and `team_keys=[...]`. From it determine:
- Does `request_state` engage (≠ None)? If None, `_request_own_state` / `battle.last_request` isn't usable
  at a forceSwitch → investigate why (the forceSwitch request structure).
- Does `own_bench_mons` include the un-brought mon (basculegion) and/or DROP the brought ones (zoroark/ditto)?

**STEP 2 — the likely fix (`data/scripts/live_state_encoder.py` + maybe `vgc_base.py`):**
Derive the replacement bench PURELY from `request_state` (Showdown's authoritative own-side truth): brought =
present in the request; switchable = `not active and not fainted`. Build the `/switch` order from the request
**ident/position**, NOT from a `battle.team` object (battle.team can be keyed by the illusion DISGUISE name,
which is why `own_bench_mons` mismatches). STOP trusting `available_switches` for brought-ness anywhere in the
forced-switch path. The flag-path fallback (`_is_brought` via `switch_set`=available_switches) is the prime
suspect for surfacing the un-brought basculegion — fix or bypass it when a request is present.

**STEP 3 — distinguish Pattern A from Pattern B** so you don't "fix" correct behavior: a genuine
1-brought-replacement-for-2-fainted-slots double faint SHOULD `/choose default` for the second slot. Only
Pattern B (≥2 brought benchers exist but get dropped) is the bug.

**STEP 4 — test + re-verify.** Add/adjust a faithful regression test (the current
`test_replacement_mask_offers_full_bench_on_partial_double_faint_desync` uses 3 *revealed* mons, which isn't
faithful to the un-brought-basculegion case — make a test where some dropped mons are brought and some are
un-brought, asserting ONLY the brought ones are offered). Keep all tests green. Then have the USER re-run the
Trickery gauntlet and confirm `forced_switch`/`forced_switch_escape` drop (excluding the legitimate Pattern-A
defaults) and MODEL-DRIVEN rises.

**Files:** `vgc_base.py` (`build_replacement_mask`, `_log_force_switch_state` already instrumented),
`data/scripts/live_state_encoder.py` (`_request_own_state`, `own_bench_mons`, `brought_team_mons`,
`_is_brought`, `switchable_union`), `local_battle/player.py` (`_select_replacement_actions` dedup),
`local_battle/live_vgc_base.py` (`_replacement_order`, `_handle_force_switch`),
`data/scripts/tests/test_replacement.py`.

**SEPARATE rare gap (defer unless it recurs):** own-ACTIVE mon under illusion with no decodable action →
Pass → "must make a move" retry (1× in v3 gauntlet, battle 1500 t6). Fix = reconstruct own active mon's
MOVES from the request.

## Full to-do list (effort / priority / dependence)
1. **#11 precise forced-switch-under-illusion fix** — serve-side, no retrain. *(M, HIGH)* **← THIS SESSION.**
   Then the own-active-illusion move-codec residual *(M, LOW)*.
2. **Promote v3** once #11 holds + win-rate ≥ baseline: `cp ai_train_scripts/BC_model/checkpoints_v3/bc_best.pt
   ai_train_scripts/BC_model/checkpoints/bc_best.pt` *(S, HIGH, depends on #11 + user go)*. Then the 7
   model_io tests run against v3.
3. **PPO self-play — the strength path** *(L, multi-session; the BC heuristic-wall is structural, RL is the
   route)*: (a) Type-C ReplayBuffer→training-schema converter *(M — the clean first brick)*; (b) PPO trainer
   w/ GAE + value baseline *(L)*; (c) self-play loop, gauntlet-gated checkpoint acceptance *(L)*.
4. **Deferred:** draft is at `docs/poke_env_available_switches_illusion_bug.md` (upstream poke-env bug
   report); pin `requirements.txt` to the poke-env master SHA *(S)*; TP-model refinements *(later)*.

## Gotchas / standing facts
- Production policy = base BC; rating/outcome weighting A/B'd and DROPPED. Value weight kept at 0.3.
- The gauntlet's MODEL-DRIVEN % UNDERCOUNTS when legitimate Pattern-A double-faults occur — don't treat
  every fallback as a bug.
- STATE_DIM **1854**, STATE_LAYOUT_VERSION **3**, ACTION_DIM 16, GIMMICK_DIM 2. Tera = placeholder only.
- poke-env on git master (a6e4f67, reads 0.15.0). poke-engine UNINSTALLED (no doubles sim).
