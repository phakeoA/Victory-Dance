# Next-session handoff prompt — Victory-Dance VGC bot (learned mega/tera)

You are Opus 4.8 continuing work on the Victory-Dance VGC Pokémon-Showdown bot.

Work in **ULTRACODE mode**: author and run Workflow-tool workflows for the substantive
investigation/verification, fan out to cover angles, and **adversarially verify every
finding against real evidence** (run the code, read the protocol, don't trust a single
probe). Token cost is not a constraint; correctness and thoroughness are. Do the serial,
tightly-coupled edits inline; use workflows for the multi-angle diagnosis and final
cross-checking.

## Working style (the user's required cadence — follow exactly)
- **Read the auto-memory index FIRST:**
  `C:\Users\death\.claude\projects\D--ShowdownProject-Victory-Dance\memory\MEMORY.md`,
  then the notes it points to (most relevant first — start with
  `live-splice-wiring-2026-06-15.md`, `bc-v0-baseline-2026-06-14.md`,
  `encoder-train-serve-split-2026-06-14.md`, `dont-retrain-until-told-2026-06-14.md`).
- **One task at a time. PAUSE after each task** — report what you did + the evidence, and
  **update the user's TODO list every time** (use the Task tools; mark done, add items you
  discover). Do not barrel through multiple tasks silently.
- **Unit-test every fix** (`data/scripts/tests/`, `ai_train_scripts/`). Keep ALL tests
  green (currently **393**).
- **Do NOT retrain or re-export until the user explicitly says so.** Batch all data/layout
  changes, then ONE retrain on their say-so.
- Update memory (MEMORY.md + the relevant note) when you finish meaningful work.

## Environment (Windows / PowerShell; Bash tool available)
- Repo root: `D:\ShowdownProject\Victory-Dance`
- venv python (has poke-env/torch/numpy): `.venv/Scripts/python.exe` (PATH `python` lacks
  them). Prefix `PYTHONIOENCODING=utf-8` for non-ASCII output.
- Tests: from repo root → `.venv/Scripts/python.exe -m pytest data/scripts ai_train_scripts -q`
- Live bot is in `local_battle/` (run_local_battle.py, player.py, random_player.py,
  live_vgc_base.py, model_io.py). **Root now has ONLY `vgc_base.py`** — the shared
  codec/mask/order base that local_battle imports (the old root player/random_player/
  run_local_battle copies were deleted this session).
- Run battles: `.venv/Scripts/python.exe local_battle/run_local_battle.py -n 5 --no-server`
  (Showdown server on :8000). `--team1/--team2` take a team NAME under `teams/M-A/` or a
  path; `--team X` = mirror. Zoroark smoke harness:
  `.venv/Scripts/python.exe local_battle/_smoke_zoroark.py 12`.
- Checkpoints are DICTS (`{model_state, config, ...}`, loaded via `local_battle/model_io.py`):
  `ai_train_scripts/BC_model/checkpoints/bc_best.pt` (val top1 0.464, **state_dim 1398,
  action_dim 16, heads our_a/our_b**), `ai_train_scripts/teamPreview_model/checkpoints/teampreview_best.pt`.
- **STATE_DIM frozen 1398. ACTION_DIM frozen 16.** Changing either invalidates the net → retrain.

---

## PRIMARY TASK — make the AI **LEARN** to mega-evolve (NOT a heuristic)

> **Scope:** the learnable decision RIGHT NOW is **mega-evolution only**. **Terastallization is
> NOT in the Reg M-A format yet and won't be for a while** — treat tera as a clearly-marked
> *placeholder/future extension*, do NOT build the tera label/serve branch against zero data
> (the corpus has ~0 tera events and Showdown won't accept a tera order in this format).

The bot currently **never** mega-evolves: the 16-action space (12 move×target + 4 switch) has no
gimmick dimension, and `vgc_base.action_to_order` never passes `mega=True` to poke-env
`create_order`. A serve heuristic ("always mega when able") is **wrong** because the decision is
**board-dependent** — e.g. with **Charizard-Y** out, if the opponent has **Pelipper** in the back
you may want to **delay** mega so their Drizzle doesn't overwrite your Drought. Only a **learned**
policy captures this. So: add mega as a learned DECISION.

### Grounded scoping (verified this session via a 4-agent workflow — start here, but re-verify adversarially)

**Recommended design: a SEPARATE per-slot GIMMICK HEAD** — implement it **2-way now: `{0=none,
1=mega}`** (`GIMMICK_DIM=2`), **orthogonal** to the existing 16-way move/switch head. Keeps
ACTIONS_PER_SLOT/ACTION_DIM **frozen at 16** (move/switch policy untouched), models Showdown's real
semantics (the gimmick is a checkbox alongside the move, not a competing action), and concentrates
the rare positive signal. *(Reject "double the move buckets to 28" and "extra logit in the same
head" — both spread the sparse signal / force a wrong softmax competition.)*
**Future tera:** when the format eventually adds terastallization (and replays with it exist),
bump the head to 3-way `{none, mega, tera}`, add the tera label-join + serve branch, re-export +
retrain. Leave `# TODO(tera): future 3rd class` markers so the seams are obvious — but build
**nothing** tera-specific now.

**PRIMARY BLOCKER (do FIRST): the training data has NO mega LABEL.**
`vod_parser/replay_parser.py::_extract_actions` (~line 1254) hard-filters to `move`/`switch`
events; mega is a separate `mega_evolution` event (~732), never stamped onto the chosen move. So
`our_actions` entries carry no mega flag (verify: grep the Type-B JSONL — **0 `"mega"` keys today**).
You must join each same-turn `mega_evolution` event to the **same-slot** move action and stamp
`action["mega"]=True`. **Join by SLOT, not execution_index** (mega events lack execution_index; a
slot megas at most once/game). **Do NOT** join `forme_change` events (Palafin/Terapagos auto-formes
are involuntary, not choices). *(The `-terastallize` event handler exists in the parser but emits
nothing in this format — leave it; the tera join is a future TODO.)*

### Ordered implementation (one batch; re-export + retrain LAST, on the user's say-so)
1. **`state_encoder.py`** — keep ACTIONS_PER_SLOT / ACTION_DIM / MOVE_TARGET_PAIRS / SWITCH_OFFSET /
   `index_to_action` / `build_action_mask` UNCHANGED. Add `GIMMICK_DIM=2` + `get_gimmick_dim()`;
   `action_to_gimmick(action)->0/1` (1 iff `action.get("mega")`); `build_gimmick_mask(snap)->
   {our_a:[2],our_b:[2]}` (bucket 0 always legal; mega legal iff the acting mon holds a mega stone
   AND no teammate has used mega this game). In `annotate_transition_actions` also stamp
   `gimmick_index` (preserve the "non-null index must be mask-legal" invariant). *(Leave a
   `# TODO(tera)` note for the future 3rd class.)*
2. **`vod_parser/replay_parser.py::_extract_actions`** — the slot-join above (PRIMARY BLOCKER):
   stamp `action["mega"]=True` from same-turn same-slot `mega_evolution` events (tera join = future).
3. **`vod_parser/transitions.py`** — pass the mega flag through `our_actions`; emit `gimmick_mask`
   next to `action_mask`.
4. **`vgc_base.py::action_to_order`** — pass `mega=True` on the move branches when the decoded gimmick
   says so, **gated on `battle.can_mega_evolve[slot]` + `used_mega_evolve`**; fall back to the plain
   order if the live battle says it's illegal (never emit a rejected `/choose`). Add
   `build_gimmick_legal_mask(battle, slot)` mirroring `state_encoder.build_gimmick_mask`
   **byte-for-byte** (train/serve parity is a hard invariant — this project fought 6 such gaps).
   Switch + replacement paths never gimmick. *(Never set `terastallize`/`z_move`/`dynamax`.)*
5. **Model** — `bc_model.py`: add parallel `gimmick_heads` (`nn.Linear(prev, GIMMICK_DIM)`) per slot;
   `forward` returns both action + gimmick logits; persist `gimmick_dim` in config. `bc_dataset.py`:
   add `gimmick_targets`/`gimmick_masks` (2-wide). `train_bc.py`: add a masked gimmick CE term + a
   gimmick-**recall** metric (positives are RARE — ≤1 mega per team per game — so plain accuracy
   hides failure; class-weight the gimmick head). `model_io.py`: load/rebuild the gimmick head, return
   its masked-argmax index from `bc_action_indices`. `local_battle/player.py`: thread the gimmick
   index into the order (replacement never gimmicks).
6. **Re-export ALL Type A–D** (`bulk_parse_replays.py`) so JSONL carries the new keys (STATE vectors
   unchanged → STATE_DIM frozen), then **retrain once — ONLY on the user's explicit say-so.**

### Key risks / how to verify
- **Parser is the blocker** — after re-export, grep `our_actions` for `"mega": true` and confirm the
  count ≈ the `mega_evolution` event count (non-zero positives MUST exist or the head can't learn).
- **Train/serve gimmick-mask parity byte-for-byte** (or Showdown rejects the order).
- **Class imbalance** — mega positives are a tiny minority (≤1/team/game); track recall, weight the head.
- **Checkpoint incompatibility** — adding the head breaks old checkpoints; codec+data+retrain ship
  together; never point the live bot at a pre-gimmick checkpoint.
- **Behavioral verify (the motivating case):** after retrain, a local smoke (reuse the
  `_smoke_zoroark.py` harness pattern) should show the bot megas Charizard-Y in neutral boards but
  **withholds** mega when the opponent's back line shows **Pelipper** (Drizzle would overwrite
  Drought), and Showdown logs **0 rejected mega orders**.
- **Tera is out of scope** — do NOT build a tera label/mask/serve branch (no data, format won't
  accept it); only leave `# TODO(tera)` seams.

Touched files: `state_encoder.py`, `vod_parser/replay_parser.py` (~1254), `vod_parser/transitions.py`,
`vgc_base.py` (~330 / ~199), `ai_train_scripts/BC_model/{bc_model.py ~80, bc_dataset.py ~100,
train_bc.py}`, `local_battle/{model_io.py, player.py, live_vgc_base.py}`, plus tests in
`data/scripts/tests/{test_action_space.py, test_encoder_parity.py, test_model_io.py}` + new
parser/gimmick tests.

---

## Rest of the TODO (after mega; keep it updated each task)
- **#22 Move-slot permutation augmentation.** The BC net is ~96% move-order sensitive (it leans on
  slot *position* — a "slot 0 = main move" prior from reveal-order training — misapplied to serve's
  request order). Fix train-only: randomly permute the 4 move-feature blocks + matching action targets
  during training so the net is order-invariant; re-probe with `data/scripts/tests/_probe_policy.py`;
  retrain. **Batch with the mega retrain.**
- **#15 Deliberate opp_a/opp_b targeting during a same-species Zoroark illusion** (poke-env loses a
  foe; the codec currently redirects via `OPPONENT_1_POSITION`). Thread the gap-#6 reconstructed opp
  slot into `action_to_order`.
- **Long-term:** aux-opponent head A/B; Type C live logger → self-play/RL (to beat the ~0.46 cloning
  ceiling).

## Working rules (recap)
Read MEMORY.md first. One task at a time; **PAUSE + update the TODO after each**. Unit-test
everything. Adversarially verify via workflows. Keep 393+ tests green. **Do NOT retrain/re-export
until told** — batch the mega head + the move-order augmentation into a single retrain.
