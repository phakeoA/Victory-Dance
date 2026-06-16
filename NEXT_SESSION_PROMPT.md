# Next-session handoff — Victory-Dance VGC bot (state-rep #5, then the RL path)

You are Opus 4.8 continuing the Victory-Dance VGC Pokémon-Showdown bot, in **ULTRACODE mode**.

## STEP 0 — read auto-memory FIRST (this is how you catch up)
Read `C:\Users\death\.claude\projects\D--ShowdownProject-Victory-Dance\memory\MEMORY.md`, then the
notes it points to. **Most relevant for right now:**
- `value-head-2026-06-16` (the value head — shipped, promoted, live-verified)
- `gauntlet-eval-2026-06-16` (the win-rate gauntlet + all the live-play / poke-env desync fixes)
- `cheap-changes-session-2026-06-16` (rating-weight, corpus_qa, serve temperature, eval_buckets)
- `mega-learned-decision-2026-06-16` (the gimmick/mega head — the model of how a head is added end-to-end)
- `dont-retrain-until-told-2026-06-14` (the user's retrain workflow — IMPORTANT)
- `state-layout-v2-2026-06-14`, `encoder-train-serve-split-2026-06-14` (how STATE_DIM/the encoder work — you change these in #5)

## Where things stand (the catch-up)
- **`bc_best.pt` now carries THREE trained heads on one MLP trunk over the 1398-dim state:**
  two action heads (our_a/our_b, ACTION_DIM 16), per-slot gimmick/mega heads (GIMMICK_DIM 2,
  `gimmick_trained=True`), and a scalar **value head** (win-prob, `value_trained=True`).
  `forward(x) → (actions, gimmicks, value)`. Policy val top1 **0.396**, value win-acc **0.64**
  (sharpest in the endgame, where the policy is weakest). Backup: `bc_best.PREVALUE_BACKUP.pt`.
- **The bot is average-human:** beats random ~75% but LOSES to a simple type/speed heuristic
  (gauntlet). Behavior cloning caps here. **The strength path is RL (PPO self-play), not more
  cloning tweaks.** The value head is the prerequisite for that and is now in place + live-verified
  (the per-turn `value head win-prob` readout in `local_battle/player.py` tracks the game).
- **Infra that exists + is trusted:** the win-rate **gauntlet** (`local_battle/gauntlet.py`:
  scripted ladder random<max_damage<heuristic + a prev_best mirror, Elo, versioned history,
  regression gate, seeded team rotation, `--battle-timeout` watchdog, `--spectate`, `-v`, and a
  decision-source `MODEL-DRIVEN %` in the report); the **TP net** is served + live-confirmed; the
  live player drives ~99% of decisions (force-switch/retry desyncs are contained, not fixed — they
  are a poke-env `available_switches` bug under Ditto/Zoroark, rare, handled via `/choose default`
  + forfeit backstop).
- **poke-env is on git master** (`a6e4f67`, version still reads 0.15.0) — it did NOT fix the desync,
  did NOT break anything. **poke-engine is UNINSTALLED** (singles-only, can't simulate doubles → no
  forward sim → 1-ply lookahead is dead; PPO self-play is the route).
- **540 tests green:** `data/scripts` + `ai_train_scripts`. Everything is **uncommitted on `dev`**.

## Working style — the USER's required cadence (follow EXACTLY)
- **Give an UPDATED TO-DO LIST after EVERY message.** Non-negotiable — the user wants the roadmap
  re-stated (with priorities) at the end of every reply.
- **Do NOT retrain or re-export WITHOUT the user's explicit permission.** Batch all data/layout
  changes, then ONE retrain on their say-so. When you do retrain, write to a SEPARATE `checkpoints_*`
  dir and back up `bc_best.pt` first; the user decides whether to promote.
- **One task at a time. PAUSE after each** — report what you did + the evidence, then wait.
- **Unit-test every change** (`data/scripts/tests/`, `ai_train_scripts/BC_model/test_bc.py`,
  `data/scripts/tests/test_model_io.py`). Keep ALL tests green (currently **540**).
- **ULTRACODE:** author + run Workflow-tool workflows for multi-angle audits/verification; do serial,
  tightly-coupled edits inline. **Adversarially verify every finding against real evidence** (run the
  code, the corpus, a live battle — never trust one probe).
- **Judge model changes on WIN-RATE (the gauntlet), not val top1** — they repeatedly disagree.
- **The user runs the live stuff** (`gauntlet.py`, `run_local_battle.py`) themselves — do NOT run
  gauntlet.py yourself unless asked; give them the command + what to look for.
- **Update auto-memory** (MEMORY.md + the relevant note) when you finish meaningful work.

## Environment (Windows / PowerShell; Bash tool available)
- Repo root: `D:\ShowdownProject\Victory-Dance`. venv python (torch/poke-env): `.venv/Scripts/python.exe`
  (PATH `python` lacks them). Prefix non-ASCII stdout with `PYTHONIOENCODING=utf-8`.
- Tests (repo root): `.venv/Scripts/python.exe -m pytest data/scripts ai_train_scripts -q`
- Train: `.venv/Scripts/python.exe ai_train_scripts/BC_model/train_bc.py --epochs 40 --seed 0 --out <dir>`
  (defaults: all Jsonl_Type{A,B,C,D}, move-order aug ON, gimmick + value heads; flags
  `--value-loss-weight 0.3` (the kept value weight), `--rating-weight`/`--rating-min`/`--outcome-weight`).
- Re-export: `.venv/Scripts/python.exe data/scripts/bulk_parse_replays.py --input <raw_html_dir> --output <jsonl_dir> --type B --overwrite` (raw HTML in `data/vods/Type_{A,B}/…`; JSONL out in
  `data/vods/Prepared_training_data/Regulation_MA/Jsonl_Type*`). Type A also takes `--team teams/M-A/Kronomono{n}`.
- Corpus QA (run after every export): `.venv/Scripts/python.exe data/scripts/corpus_qa.py`
- Offline policy+value diagnostic: `.venv/Scripts/python.exe ai_train_scripts/BC_model/eval_buckets.py --ckpt <ckpt>`
- Gauntlet (USER runs): `local_battle/gauntlet.py --battles 30 --teams <≥4 names>` (`--prev-best <ckpt>` to A/B).
- Live battle (USER runs): `local_battle/run_local_battle.py -n 5` (TP + value-head logs show at INFO by default).
- Checkpoints (DICTs via `local_battle/model_io.py`): `BC_model/checkpoints/bc_best.pt` (policy 0.396 +
  gimmick + value, STATE_DIM 1398, ACTION_DIM 16). `teamPreview_model/checkpoints/teampreview_best.pt`.

## ⇒ FIRST TASK THIS SESSION: #5 — item + ability features in-battle (re-export+retrain)
**Why:** the state encoder reads **ZERO item/ability features**, yet the data is 100% plumbed —
`battle_models.py` serializes `known_item`/`known_ability`, and `belief_state.py` has
`top_item`/`top_ability`/`item_distribution`/`ability_distribution`. Choice-lock/Focus-Sash/Booster
Energy/Intimidate/weather are **first-order VGC drivers the policy AND the value head currently
CANNOT see** — the single highest-leverage state-rep gap. (The value head being blind to items is a
big reason its win-acc tops out ~0.64.)

**What to build (batch #5 + #6 + #7 into ONE re-export+retrain so the layout re-freezes exactly once):**
- **#5** ~2–3 floats per mon: `item_id`, `ability_id` (small learned-id encodings or normalized
  indices via `pokedex`), plus an `item_known` flag. Use `known_item`/`known_ability` when revealed,
  else the belief `top_item`/`top_ability`. Do it for own + opp active + bench slots.
- **#6** a move spread/target-shape flag in `MOVE_FEATURES` (currently 9: priority/accuracy/protect/
  STAB but NOT spread-vs-single — the core doubles tradeoff), from `data/moves.json`. (+ secondary-
  effect prob / high-crit as a P2 stretch.)
- **#7** real PP: un-pin `pp_fraction` from the hardcoded 1.0 in BOTH encoders (parser counts per-mon
  move uses); preserve train/serve parity. Lower value — fold in.
- **Add a `STATE_DIM` VERSION CONSTANT + a load-time dim assert** in `model_io.load_bc_policy` so a
  stale-layout checkpoint can't silently mismatch (the 938→1386→1398 churn did exactly that).

**Critical constraints:**
- This **changes STATE_DIM** (currently frozen 1398) ⇒ the current `bc_best.pt` becomes layout-
  incompatible. So this is a **re-export + retrain**, both gated on the user's explicit go.
- **Encoder PARITY:** every feature must be added to BOTH `data/scripts/state_encoder.py` (offline/
  training, the single codec source) AND `data/scripts/live_state_encoder.py` (serve), byte-identical
  — verify with the parity harnesses in `data/scripts/tests/` (e.g. `test_encoder_parity.py`).
- **Workflow:** implement + unit-test the encoder changes, regenerate a SMALL sample + run `corpus_qa.py`,
  verify parity, THEN STOP and ask the user before the full re-export + retrain. Tera stays a
  `# TODO(tera)` seam (NOT in Reg M-A).

**Key files for #5:** `data/scripts/state_encoder.py` (encoder + STATE_DIM + MOVE_FEATURES),
`data/scripts/live_state_encoder.py` (serve parity), `data/scripts/battle_models.py`
(known_item/known_ability), `data/scripts/belief_state.py` (top_item/top_ability/distributions),
`data/scripts/vod_parser/` (PP counting for #7), `data/scripts/bulk_parse_replays.py` (re-export),
`local_battle/model_io.py` (the dim-assert), `ai_train_scripts/BC_model/{bc_model,train_bc}.py`
(STATE_DIM flows through `get_state_dim()` automatically — confirm).

## Full to-do list (the roadmap after #5)
1. **#5 item/ability + #6 spread flag + #7 PP** — ONE re-export+retrain (STATE_DIM change + version
   const + dim assert). **← THIS SESSION (gated on user for the actual re-export/retrain).** *(M)*
2. **PPO self-play — the strength path** *(L, multi-session)*:
   a. **Type-C ReplayBuffer → training-schema converter** (TODO #9) — the live buffer logs
      `(state, action, outcome)` per turn but `bc_dataset` can't read the flat schema; this IS the
      self-play data pipeline. *(M, the clean first brick.)*
   b. **PPO trainer** — clipped objective + GAE advantage (value head as baseline) + value regression
      (also fixes the value head's slight early-game pessimism). *(L)*
   c. **Self-play loop** (model vs frozen snapshot, collect trajectories, update) + **gauntlet-gated**
      checkpoint acceptance vs previous-best (the gauntlet already exists — reuse it; this is the
      "winner moves on" idea done right). *(L)*  NOT MCTS (no doubles sim), NOT a 100-battle bracket.
3. **Type-C live data** in parallel — RL trajectories + a ladder reality-check, once the bot is decent.
4. **Deferred:** (b) draft an upstream poke-env bug report (`available_switches` drops a healthy bench
   mon after an Imposter/Illusion turn) *(S)*; the ~2% endgame forced-switch that goes to
   `/choose default` instead of a model pick — order replacement by POSITION from our own
   reconstruction, only for the last slice of model control *(M)*; TP-model refinements (richer belief
   features / set-transformer) *(later)*; pin `requirements.txt` to the poke-env master SHA *(S)*.

## Gotchas / standing facts
- Production policy = **base BC** (the rating-weighted variant `checkpoints_rw` was A/B'd and DROPPED —
  within noise; the weakness is structural). `checkpoints_value/` = the weight-1.0 value run (regressed
  policy 0.368); `checkpoints_value03/` = the kept one (now == bc_best).
- The gauntlet's `MODEL-DRIVEN %` in its report is the trust gauge — a win-rate is only meaningful when
  it's ~95%+ model-driven. The desync drops it on Ditto/Zoroark teams; ~99% on normal teams.
- Don't trust a value win-prob in isolation — it's modest (0.64) and slightly pessimistic at turn 1.
- ACTION_DIM frozen 16. STATE_DIM 1398 **until #5 re-freezes it.**
