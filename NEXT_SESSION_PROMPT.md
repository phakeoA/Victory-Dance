# Next-session handoff — Victory-Dance VGC bot (raising the BC ceiling)

You are Opus 4.8 continuing the Victory-Dance VGC Pokémon-Showdown bot, in **ULTRACODE mode**.

## The one thing to internalize first
The bot is a **behavior clone** → it caps at **average-human** play (val top1 ~0.40; it beats a
random player ~75% but a *strong* player beats random ~95%). Cloning tweaks (mega head,
move-order aug, aux-opp head) are all **marginal moves within that ceiling** — they cannot make it
*strong*. **The whole point of this TODO is to RAISE the ceiling.** Two levers do that cheaply and
need NO re-export, plus the infra to *judge* them honestly. Everything below is sorted by that.

## Working style (the user's required cadence — follow exactly)
- **Read auto-memory FIRST:** `C:\Users\death\.claude\projects\D--ShowdownProject-Victory-Dance\memory\MEMORY.md`,
  then the notes it points to. Most relevant: `mega-learned-decision-2026-06-16`,
  `aux-opp-head-ab-2026-06-16`, `illusion-targeting-switch-dedup-2026-06-16`,
  `live-splice-wiring-2026-06-15`, `bc-v0-baseline-2026-06-14`, `dont-retrain-until-told-2026-06-14`.
- **Work in ULTRACODE:** author + run Workflow-tool workflows for multi-angle audits/verification;
  do serial, tightly-coupled edits inline. **Adversarially verify every finding against real
  evidence** (run the code, the battle, the corpus — never trust one probe).
- **One task at a time. PAUSE after each** — report what you did + the evidence, and **update the
  user's Task list every time** (TaskCreate/TaskUpdate). Don't barrel through tasks silently.
- **Unit-test every fix** (`data/scripts/tests/`, `ai_train_scripts/BC_model/test_bc.py`). Keep ALL
  tests green (**currently 473**).
- **Do NOT retrain or re-export until the user explicitly says so.** Batch all data/layout changes,
  then ONE retrain on their say-so. Judge model changes on **play strength** (win-rate), not just
  val top1 — this session repeatedly found they DISAGREE.
- Update memory (MEMORY.md + the relevant note) when you finish meaningful work.

## Environment (Windows / PowerShell; Bash tool available)
- Repo root: `D:\ShowdownProject\Victory-Dance`. venv python (torch/poke-env): `.venv/Scripts/python.exe`
  (PATH `python` lacks them). Prefix non-ASCII output with `PYTHONIOENCODING=utf-8`.
- Tests (repo root): `.venv/Scripts/python.exe -m pytest data/scripts ai_train_scripts -q`
- Train BC: `.venv/Scripts/python.exe ai_train_scripts/BC_model/train_bc.py --epochs 40 --seed 0`
  (defaults: all Jsonl_Type{A,B,C,D}, move-order aug ON, gimmick head; `--aux-opp-head` OFF/opt-in).
- Re-export: `.venv/Scripts/python.exe data/scripts/bulk_parse_replays.py --input <raw> --output <jsonl> --type B --overwrite`
  (raw HTML in `data/vods/Type_{A,B}/…`; JSONL out in `data/vods/Prepared_training_data/Regulation_MA/Jsonl_Type*`).
- Live battles / eval (need Showdown server on :8000; harness auto-starts it):
  `local_battle/run_local_battle.py`, and the kept harnesses `_vs_random.py` (abs skill),
  `_ab_headtohead.py` (ckpt-vs-ckpt, swap-capable), `_test_illusion_targeting.py` (Kronomono3 mirror),
  `_diag_rejections.py` (Showdown reject reasons), `_smoke_zoroark.py`.
- Checkpoints (DICTs via `local_battle/model_io.py`):
  `BC_model/checkpoints/bc_best.pt` (val top1 **0.397**, gimmick recall 0.964, **state_dim 1398,
  action_dim 16, heads our_a/our_b**), `teamPreview_model/checkpoints/teampreview_best.pt` (mean-exact
  0.245), `bc_aux_opp_EXPERIMENT.pt` (aux-opp A/B — REJECTED, kept for reference).
- **STATE_DIM frozen 1398, ACTION_DIM frozen 16.** Changing STATE_DIM ⇒ re-export+retrain.

## Legend
**Priority** P1 (do first) → P3 (later). **Effort** S/M/L. **Retrain**: `none` (serve/code only) ·
`retrain` (train-only, NO re-export — data already in JSONL) · `re-export+retrain` (changes JSONL/state layout).

---

## TIER 1 — RAISE THE CEILING (cheap, high-leverage, **no re-export**) — start here
1. **Rating-filter / rating-weight the demonstrators.** `players[*].rating_before` / `rating_delta`
   are ALREADY in every JSONL transition (our-side spread 1670-2019, median 1763) but `bc_dataset.py`
   never reads them. Add a dataset-loader rating threshold and/or per-example loss weight by rating
   percentile → the clone imitates STRONGER players. **THE cheapest path past val 0.40.**
   *(P1 · S · retrain.)* Companion: a per-example **`outcome_weight`** = f(game `winner`, `rating_delta`)
   to up-weight decisions from won / rating-gaining games (also already in JSONL). *(P1 · S · retrain.)*
2. **Add a VALUE HEAD (win probability).** `bc_model.py` is trunk→action+gimmick heads, no scalar
   head; `ai_train_scripts/network.py` already defines `VGCNet`+`ValueHead`+`AlphaZeroLoss` (orphaned —
   port it). Label is derivable at load with NO re-export: `reward.win` is set only on the final turn,
   but every transition carries the `winner` username → back-fill +1/0 (optionally discounted by
   turns-to-end) to ALL turns. Add `Linear(trunk_out,1)` + BCE/MSE term; report val win-acc / Brier.
   **Unblocks search/RL + position eval — the real strength path.** *(P1 · M · retrain.)*
3. **Win-rate eval gauntlet (the metric that matters).** Today only `_vs_random.py` exists (no opponent
   above random, no persisted history, no Elo). Add a **max-damage** and a **simple type/speed
   heuristic** opponent, run N side-balanced battles vs {random, max-damage, heuristic, previous-best
   ckpt} on a rotating team pool, emit a versioned JSON/CSV row + Elo + delta-vs-last gate. Reuses
   `run_local_battle.make_player`/`battle_against`. **Prereq to trust every TIER-1/TIER-3 change.**
   *(P1 · M · none.)*
4. **Serve the team-preview net + a TP smoke.** `teampreview_best.pt` exists but the live player only
   drives per-turn BC — TP is never exercised in real games, so its quality is unmeasured. Wire
   `team_order` into the live teampreview step + a mirror-battle 0-rejection smoke. *(P1 · M · none.)*

## TIER 2 — STATE-REP RE-FREEZE (batch into ONE re-export+retrain)
*All change STATE_DIM → do them together so the layout re-freezes exactly once; add a STATE_DIM
version constant + a load-time dim assert so a stale-layout checkpoint can't silently mismatch
(the 938→1398 churn did).* 
5. **Item + ability features in-battle.** The encoder reads ZERO item/ability features, yet the data is
   100% plumbed: `battle_models.to_dict` serializes `known_item`/`known_ability`, and `belief_state`
   has `top_item`/`top_ability`/`item_distribution`/`ability_distribution`. Add ~2-3 floats/mon
   (item_id, ability_id, item_known). Choice-lock/Sash/Booster/Intimidate/weather are first-order VGC
   drivers the clone currently CANNOT see. **Highest-leverage state-rep gap.** *(P1 · M · re-export+retrain.)*
6. **Move spread/target-shape flag** (+ secondary-effect prob, high-crit). `MOVE_FEATURES=9` has
   priority/accuracy/protect/stab but NOT spread-vs-single (the core doubles tradeoff) — from
   `data/moves.json`, no new data. *(P1 spread-flag · S; P2 secondary/crit · M · re-export+retrain.)*
7. **Track real PP** (parser counts per-mon move uses; un-pin `pp_fraction` from the hardcoded 1.0 in
   both encoders, preserving parity). Lower value (PP-stall is rare); fold into the batch. *(P2 · M · re-export+retrain.)*
   (Tera-available/tera-type: leave as a `# TODO(tera)` seam — Tera is NOT in Reg M-A.)

## TIER 3 — THE STRENGTH PATH (large; sequenced behind the value head)
8. **Self-play / RL on the BC prior (AlphaZero direction).** `network.py` is a head start but MCTS,
   the self-play generator, and the replay-buffer→training loop are all unbuilt. Sequence: value head
   (TIER 1 #2) → greedy 1-ply value lookahead at serve → MCTS(policy+value) → self-play. **Where real
   strength comes from.** *(P1-conceptually but gated · L · retrain.)*
9. **Persist the live Type-C data loop + a converter.** The live `ReplayBuffer` already logs
   `{state(1398), action_s0/1, source, outcome}` per battle but in a FLAT schema `bc_dataset` can't
   read. Write a converter to the training schema (or a Type-C loader path) so live/self-play games
   become ingestible → enables DAgger-style on-policy correction (attacks BC distribution shift, a
   known reason BC underperforms its val top1 in real play). *(P2 · M-L · re-export+retrain.)*
10. **Live belief updating during games.** `_default_belief()` is a STATIC pikalytics prior, never
    conditioned on what the opponent reveals mid-game. The protocol log needed to update it is ALREADY
    captured (`self._proto_log` for the gap-#6 splice) — run the offline `belief_state.fill_blanks`/
    `compute_prediction_error` update over that prefix each turn → sharper opp bytes. Prereq for an
    honest value head / RL (reason over an updating opponent model). *(P2 · S-M · none.)*

## TIER 4 — POLISH / LOWER-LEVERAGE
- **Class-imbalance: MEASURE it.** `--class-weight balanced` is built but production used `none` for
  the action head. One A/B vs none — but judge on **win-rate** (balanced trades top1 for tail-action
  coverage; for a clone top1 may DROP while real play improves). *(P2 · S · retrain.)*
- **Serve temperature / top-p + calibration (ECE/Brier).** Pure-argmax is brittle/exploitable on OOD
  boards; a serve-side sampling knob + calibration tracking, no retrain. *(P2 · S · none.)*
- **Per-situation eval (leads/mid/endgame) + calibration.** Bucket val top1 by game phase (turn +
  team-count globals already present; replacement turns already split) + ECE/Brier — tells you WHERE
  cloning fails (likely endgame/forced-switch). Cheap offline diagnostic. *(P2 · S · none.)*
- **Corpus QA / label-audit script** (null-index rate per decision_type, illegal-under-mask, rating
  histogram, dup replay_ids) — run after every scrape+export as a regression gate. *(P2 · S · none.)*
- **Team-preview model:** richer per-mon belief features *(P1-within-TP · M · retrain)*; order-2
  pairwise matchup features (type/speed vs each opp) before a set-transformer *(P2 · M · retrain)*;
  attention/set-transformer over the 12 mons *(P2 · M · retrain — data-limited, ~6300 examples)*;
  bring-4 positive-unlabeled learning to recover the ~20% dropped 2-3-brought rows *(P2 · S · retrain)*;
  enforce leads ⊂ brought hierarchy *(P3 · S · retrain)*.
- **More data:** scrape more high-rated Type B (pair with #1 so it doesn't dilute) *(P2 · S ·
  re-export+retrain)*; more Type A exact-stat VODs *(P3 · M · re-export+retrain — 28 games is a drop
  vs 3150)*; damage-calc back-solver for Type B opp EVs (documented stub, second-order belief refine)
  *(P3 · L · re-export+retrain — defer until cheap levers exhausted)*.

---

## DONE this session / earlier — removed from the TODO (don't re-do)
- **Learned mega-evolution** (gimmick head end-to-end: parser label → state_encoder codec/mask →
  transitions → vgc_base serve byte-parity → bc_model/dataset/train/model_io/player). Re-exported +
  retrained; val 0.397, gimmick **recall 0.964**. SHIPPED.
- **Move-slot permutation augmentation** (train-only) → move-ORDER invariant + it ALSO **fixed
  overfitting** (train≈val now; the "overfits after ep 12-15 / needs early-stopping" item is largely
  obsolete — `--patience` is wired, default 0; turning it on only saves wall-clock).
- **Aux opponent head** — built + A/B'd (cloning AND 40-battle win-rate); **no gain, NOT promoted**
  (bc_best.pt has no opp heads). Opp codec + `with_opp` dataset remain if revisited.
- **Post-faint replacement** (data export `decision_type='replacement'` + model-driven serve).
- **State layout v2**: opp bench + per-mon is_fainted + team-count globals (STATE_DIM 1398).
- **Ditto/Zoroark**: is_transformed flag + Zoroark fooled-view + gap-#6 live opp reconstruction.
  (No literal opp `is_illusion` bit BY DESIGN — encoder shows the disguise species, train==serve.)
- **Live poke-env player (Type C live path)**: `local_battle/` drives both nets vs a localhost server;
  model drives ~100% of turns, model-driven forced replacement, 0 rejections.
- **Cross-slot switch-collision serve fix** (`player._select_actions` re-decodes slot 1 with the
  colliding switch masked) → 0 rejections over 80 Kronomono3-mirror battles (was ~3.5%). Plus the
  **#15 deliberate illusion-targeting** safety-net (unit-tested; rarely triggers — poke-env keeps a
  targetable phantom). Mixing Type A+B is ON by default (only the marginal-effect A/B is unmeasured).

## Suggested first move
Spin up the **TIER-1 eval gauntlet (#3)** and the **rating-weighted dataset (#1)** together: the
gauntlet gives you the win-rate yardstick, and rating-weighting is the cheapest ceiling-raiser to
measure against it — both train-only, no re-export, and they de-risk everything after. Then the
**value head (#2)** to unlock the strength path. Pause + report after each, as always.
