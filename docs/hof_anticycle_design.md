# Phase 2 — Hall-of-Fame anti-cycle gate (DESIGN LOCK, 2026-06-17)

> ## ★ PIVOT (2026-06-17): HoF suspects = the candidate's PAST CHAMPIONS (not non-champion gens)
> User insight (correct): the P2.0 "exclude ALL champion-lineage" decision was too aggressive — that
> rationale ("correlated/redundant with the mirror") only holds for the CURRENT champion (which IS the
> mirror). OLDER champions are several gens removed → NOT redundant, and testing vs them directly catches
> **lineage cycling** (a candidate that beats the current champion but loses to an older one = the accepted
> lineage A→B→C isn't monotonic improvement, it's an RPS cycle — the exact "G5 promoted backward over G4"
> failure that MOTIVATED the v2 redesign). Past champions are the STRONG milestones AND spread across training
> eras (promoted at different times) → naturally diverse, sharper suspects than the weaker held non-champion gens.
> **NEW design:** `cluster_hof_suspects(snapshots, n=5, current_champion_path=…)` = the last **5 PAST CHAMPIONS**
> (accepted promotions), newest-first, EXCLUDING the current champion (mirror covers it). `HoFConfig.n_champions=5`,
> `min_pool=2` (champions accumulate slowly → activate once ≥2 prior champions exist; early run = fail-open skip,
> nothing to cycle yet). Everything below (the per-snapshot significance-veto RULE, z=1.96, n=60, the gate fn,
> P2.1 calibration) is UNCHANGED — P2.1 is suspect-SELECTION-agnostic (depends only on the COUNT + the rule). The
> "era-strided non-champion" text below is SUPERSEDED as the selection; the gate fn stays selection-agnostic so a
> future diverse-non-champion supplement is a one-fn change. Caveats kept: exclude the CURRENT champion (no
> double-count); cap ~8 (FWER grows with suspect count); rule stays not-LOSING (must-beat-all freezes, P2.1).


Status: **P2.0 design locked + user-signed-off.** No production code yet. Implementation = P2.1 (gate_sim
calibration) → P2.2 (pure gate fns + tests) → P2.3 (live eval plumbing) → P2.4 (wire into `run_generation` +
force valve + observability) → P2.5 (user live smoke). Produced by the project's design-panel + red-team
Workflow (4 design lenses → 3 adversaries → synthesis).

## Why this exists
The shipped v2 frozen-champion gate (`promotion_gate_v2`) tests only **depth** vs ONE anchor — the frozen
champion mirror (≥70% over ≥200 games, or a plateau backstop). VGC has rock-paper-scissors non-transitivity
across the ~71-team pool (A>B>C>A is expected), so a candidate can go ≥70% vs the champion mirror while
**quietly losing to an orthogonal slice of the opponent league it "forgot" how to beat** → it promotes
laterally and the run cycles with zero real progress. The HoF adds **breadth**: the candidate must not be
*proven losing* to the worst suspect of the league pool, not merely good on average.

## THE key design correction (vs the original decomposition)
The veto operates at **per-SNAPSHOT resolution, NOT per-band-mean.** Band-mean aggregation provably washes
out the single-snapshot/single-team RPS counter the whole phase exists to catch: one true-20% matchup among
60%-fodder gives a band mean of 0.40–0.53 (S=2–6), so the band's Wilson-upper sits ≥0.50 and the veto **never
fires — P(catch)=0**. At snapshot grain, n=60 games catches a true-20% counter **99.9%** and a true-30%
counter **~90%**, while spuriously vetoing an honest 0.55 snapshot only **~0.3%**.

Cluster definition **A (generation-strided bands)** — the user's choice — is **retained as the SAMPLING
STRATUM** only: it guarantees the suspect set spans early/mid/late training eras. The gate fn never sees a
generation number, so the deferred behavioral-k-means builder is a one-file swap.

## Suspect builder (pure; reuses league fields)
`cluster_hof_suspects(snapshots, *, K=3, per_band=2, exclude_champions=True) -> List[LeagueSnapshot]`
1. Filter to `is_champion == False` (**champion-lineage exclusion**, Q6 confirmed): the candidate is the
   frozen champion's own descendant, so beating its accepted ancestors is correlated with / dominated by the
   mirror test it already passed — including them dilutes the breadth signal with redundant depth.
2. Sort by `generation`; partition into `K=3` **contiguous equal-count** era-bands (equal-count, not
   equal-width, so per-band n is comparable however generations bunch up).
3. From EACH band take the `per_band=2` **lowest `latest_winrate()`** members (PFSP already tracks exactly the
   snapshots the latest *loses to* → the suspects are handed to us for free; tie-break lowest generation then
   lowest elo for determinism).
→ a bounded suspect set of ≤ `K*per_band = 6` distinct snapshots.

## Gate rule (pure; `--dry-run`-importable, no numpy)
`hall_of_fame_gate(snap_results: List[(snapshot_id, wins, games)], *, cfg) -> (verdict, stats)`
- For each suspect with `games >= cfg.hof_min_games_per_snap`, compute `U = wilson_upper_bound(wins, games,
  z=cfg.hof_z)`.
- **VETO (`verdict="reject"`) iff ANY eligible suspect has `U < 0.50`** (proven losing, one-sided ~97.5%);
  else `"confirm"`. Worst-of-sampled-snapshots — no averaging.
- A suspect with `games < hof_min_games_per_snap` is reported but **cannot veto** (no noise-induced freeze).
- Significance-veto (act only on *evidence* of a weak suspect) mirrors the existing scripted collapse-revert
  (`scr_upper < floor`). STRICT (prove every suspect ≥0.50 lower-bound) is **rejected** — it false-rejects
  honest strong candidates constantly (a true-0.60 suspect fails Wilson-lower≥0.50 33–68% at n=40–120, which
  compounds over K) and is itself a freeze generator.
- Family-wise false-reject is controlled by **bounding the suspect count to ≤6**, not by over-widening z.

`wilson_upper_bound(wins, games, z)` = symmetric sibling of the existing `gate.wilson_lower_bound`:
`(centre + margin) / denom`, clamped to [0,1]. Pure-Python.

## Thresholds & config
- Bar = **0.50** (not-losing boundary — NOT the 0.70 crowning bar). `z = 1.96` (one-sided ~97.5%; halves
  per-snapshot false-reject 4.6%→2.8% at the boundary vs the gate's usual 1.645).
- **Starting** `hof_games_per_snapshot = 60`; hard floor `hof_min_games_per_snap = 40`. **P2.1 must lock n
  before wiring.**
- `GenConfig` gains: `hof_enabled=True, hof_k=3, hof_per_band=2, hof_games_per_snapshot=60,
  hof_min_games_per_snap=40, hof_z=1.96, hof_min_pool=6, hof_exclude_champions=True, hof_force_limit=4`.

## Integration semantics (in `run_generation`)
- **Runs only when `promotion_gate_v2` returns `"promote"`** (covers `beat_champion` AND `plateau_reanchor`;
  HOLD/REVERT skip it → no eval cost on the common path). Wired AFTER the v2 verdict but BEFORE
  `league.latest_path` / `advance_champion` commit.
- **Eval reuses `run_gauntlet`**: one `prev_best`-style mirror call per sampled suspect
  (`opponents=["prev_best"]`, `prev_best_ckpt=suspect.path`, `battles=hof_games_per_snapshot`, reuse
  `n_workers`) → `aggregate_prev_best` per suspect → `hall_of_fame_gate`. **Zero new battle code.**
- **Thin pool (`eligible < hof_min_pool=6`) → `confirm` with `reason="thin_pool_skip"`** (fail-open, never
  block on absence of evidence; the promote stands). Logged + surfaced (a long skip streak means the
  anti-cycle guard is dark).
- **On REJECT**: downgrade `promote → "hold"`, `reason="hof_reject"`, attach the offending
  `snapshot_id` + stats. Champion does NOT advance; `h2h_history` **preserved** (the climb vs the still-frozen
  champion is live; resetting would erase the plateau backstop's evidence); optimizers NOT reset (not a
  collapse); candidate **still admitted** to the league as `is_champion=False` (it's a useful new orthogonal
  PFSP opponent that trains the next candidate against the very hole).

## Freeze valve (the plateau-backstop interaction)
A `plateau_reanchor` that HoF rejects forever would defeat the backstop AND strand `scripted_high_water` /
the rollback floor (`advance_champion` is its only writer → train-into-a-corner-then-revert trap).
- After `hof_force_limit=4` **consecutive** hof-rejects of **`plateau_reanchor` promotes** (NEVER
  `beat_champion` — a real breadth hole on an earned 70% mirror should hold) **AND** only when the worst
  suspect is **marginal** (`Wilson-lower ≥ 0.45`): **force the re-anchor THROUGH `advance_champion`**
  (`reason="plateau_reanchor_hof_forced"`) so the floor stays current + h2h resets + champion_elo steps, plus
  a WARNING-level `operator_alert`.
- A **catastrophic** suspect (`Wilson-lower < 0.45`) is **NEVER auto-force-promoted** → **[USER-CONFIRMED
  POLICY] freeze the champion + fire a loud, distinct hof-standoff alert + offer a one-flag `--hof-override`
  manual release.** Never bake a proven hard-counter into the rollback floor. (User chose this over an
  auto-release-after-larger-limit option.)

## Observability
1. Per-gen `hof` manifest/history block: `{ran, skipped+reason, K, per_band, n_per_snap, suspects:[{snapshot_id,
   generation, band_id, wins, games, wilson_lower, wilson_upper, vetoed}], worst_snapshot_id, worst_upper,
   verdict, reason}`.
2. `hof_reject_streak` persisted on `GenerationHistory` (survives resume; drives the force valve + alert).
3. `operator_alert` gains a **3-way stall taxonomy**: collapse-loop (≥revert_limit reverts), generic-stall
   (≥25 no-promote), and the **new hof-standoff** (consecutive hof_reject holds while plateau keeps firing) —
   the standoff fires LOUD at **streak ≥ 2** (well before the 25-gen generic stall, useless for a ~10-gen
   overnight budget), naming the failing suspect's `snapshot_id`+generation. A softer `hof_inactive_streak`
   alert fires if HoF has SKIPPED (thin pool) for >15 promotes ("guard is dark / promotes are mirror-only").
4. Dashboard: worst-suspect Wilson-upper as a per-gen **breadth-health line** beside the mirror **depth** h2h
   line; flag the gen where verdict flipped promote→hof_reject ("mirror said yes, breadth said no"); show SKIP
   distinctly from a pass.
5. Per-HoF console line: `HoF n=60 suspects [g3:0.62 g7:0.41* g12:0.58] worst=gen7 upper=0.49 -> REJECT`.
6. Carry `run_gauntlet`'s decision-source % per suspect so a "weak suspect" that is actually a poke-env
   Illusion/Ditto `/choose default` artifact is marked **inconclusive** (cannot veto), not a real RPS hole.

## P2.1 — calibration targets (gate_sim must lock the numbers BEFORE wiring)
1. **Planted single-snapshot-counter** (the decisive new scenario): one suspect at true 0.20/0.30/0.40, rest
   0.55–0.60 → confirm snapshot-resolution catches 0.20 ≥99% / 0.30 ≥88% at the chosen n, AND that the OLD
   band-mean rule catches it ~0% (regression-guard the design choice itself).
2. **Family-wise false-reject sweep**: M ∈ {3,4,6} honest suspects all at true 0.50 (worst case) and at 0.55
   (realistic post-70%-mirror), n ∈ {40,60,80,120}, z ∈ {1.645,1.96,2.24}. Pick the smallest (n, M-cap, z)
   where FWER@0.55 ≤ ~5% AND FWER@0.50-boundary ≤ ~12% AND catch@0.30 ≥ ~88%. (Start n=60, M≤6, z=1.96.)
3. **Correlated-draw variant** (NOT i.i.d.): each suspect's per-game outcomes from a shared latent matchup
   strength + small idiosyncratic term (real model-vs-model games are correlated → effective n < nominal — the
   same reason the mirror needed 240 not 200). Re-run; bump n if FWER/catch degrade.
4. **`hof_min_games_per_snap` floor**: find the floor where a true-0.50 suspect's veto rate at that game count
   is < ~3% (underpowered suspect must never noise-veto).
5. **Force-valve**: persistent-weak trajectory (plateau every gen + one suspect held at true 0.40–0.45 marginal
   vs 0.30 catastrophic); sweep `hof_force_limit` ∈ {3,4,6,8}. Pick the smallest where a transient noise-veto
   doesn't trigger a spurious force-promote, a real marginal hole force-re-anchors within ≤4 gens, and a
   catastrophic hole correctly NEVER force-promotes. (Start 4; marginal/catastrophic split at Wilson-lower 0.45.)
6. **Thin-pool / champion-flood trace**: a `beat_champion` burst (recent snapshots become champions) + a young
   run → confirm `hof_min_pool=6` → skip==confirm and `hof_inactive_streak` surfaces a dark guard.
7. **Real HoF smoke** (post-sim, pre-trust): one live gen that clears the mirror but is planted against a
   known-weak archived snapshot → confirm empirical per-snapshot variance matches the (correlated) assumption
   and the wiring (skip, reject→hold→admit, force valve) behaves; bump n if variance exceeds the sim.

## Residual risks (carried forward)
1. **Single-lineage false comfort**: gen-strided bands of one BC-seeded lineage can be near-identical weights →
   even the lowest-winrate suspect may be behaviorally close to the candidate. Partially mitigated (lowest-PFSP
   suspect per era-band + snapshot resolution); the true fix is the deferred behavioral-k-means builder
   (user chose to **defer until live evidence**). A band-separation diagnostic should warn when all suspects
   read within mirror-noise of each other ("guard is theater").
2. **Marginal-counter under-detection**: a true-0.40 hole is caught only ~35–46% per HoF run at n=60–80 → a run
   cycling on a mild 0.40 hole may take 2–3 promotes to catch. Accepted (0.40 barely cycles; the league
   retains the suspect so PFSP re-exposes it). Bump n→120 only if real 0.40 cyclers appear.
3. **Correlated-outcome effective-n gap**: all sim numbers assume binomial; real games share matchups →
   effective n < nominal. P2.1's correlated sweep + the real smoke are the required mitigation.
4. **Champion-flood / thin-pool dark window**: champion-exclusion can starve the suspect pool < min_pool during
   a `beat_champion` burst → skip==confirm exactly when rapid lateral promotes are cheapest. Chosen safe
   default (never block on absence of evidence); `hof_inactive_streak` makes the dark window visible. We do NOT
   relax champion-exclusion (would reintroduce mirror correlation).
5. **Force-valve mis-fire / catastrophic freeze**: depends on `hof_force_limit` + the 0.45 split (P2.1) and a
   loud, actionable alert; a missed alert wastes overnight compute on a real ceiling.
6. **Poke-env artifact as fake RPS hole**: mitigated by carrying decision-source % per suspect (low
   model-driven → inconclusive, cannot veto); the "enough model-driven to count" threshold is confirmed on the
   real smoke.

## User-confirmed decisions (2026-06-17)
- Catastrophic-hole policy: **freeze + alert + manual `--hof-override`** (not auto-release).
- HoF eval budget: **~360 games/promote (6 suspects × 60)**; P2.1 may trim.
- Behavioral-k-means builder: **deferred** until a live run catches/misses a real cycler.

## Mirror promotion bar lowered: 0.70 → 0.55 (user decision 2026-06-17)
The v2 gate's `beat_champion` bar (`GateConfigV2.promote_threshold`) drops from **0.70 → ~0.55** — promote when
the candidate is *convincingly* (not *crushingly*) better than the frozen champion. This is **coupled to the
HoF**, not independent. Monte-Carlo over the exact v2 rule (`observed >= threshold AND lower-CI(z) > 0.5`):

| | true 0.50 (false-promote) | true 0.55 | true 0.58 | true 0.60 | true 0.65 |
|---|---|---|---|---|---|
| **thr 0.70, n=240** | 0.0% | 0.0% | 0.0% | 0.1% | **5.8%** |
| **thr 0.55, n=240** | 5.3% | ~47% | 81% | 93% | 99.9% |
| **thr 0.55, n=360** | 3.2% | ~52% | 88% | 98% | 100% |
| **thr 0.55, n=500** | 1.4% | ~52% | 92% | 99% | 100% |

- WHY change: at 0.70 even a genuinely strong true-65% policy promotes only ~6% of evals (n=240) → the champion
  effectively only advances via the plateau backstop → it "freezes" (the user's complaint).
- **Power for a true-55% policy is ~50%/gen and CANNOT be raised by more games** — `observed >= 0.55` lands above
  a true-55%'s own mean only ~half the time, so a real 55% policy promotes over **~2 gens**, not 1. (To promote a
  true-55% in one shot you'd have to drop the bar below 55%.) Fine: it advances within a gen or two.
- **CONSEQUENCE 1 (cost):** the mirror needs MORE games, not fewer — n rises from 240 toward **~360** to keep
  false-promote ~3%. `--mirror-battles` is re-locked in P2.1 (the correlated-draw variant may push it higher).
- **CONSEQUENCE 2 (anti-cycle):** a 55% edge vs ONE frozen champion is far more likely to be a LATERAL RPS step
  than a 70% edge → the **HoF breadth veto becomes LOAD-BEARING**, not a nice-to-have. Lowering the depth bar and
  adding the breadth requirement are two halves of the SAME design.
- `promote_z` stays 1.645 (it controls false-promote at smaller n; once n≥360 the 0.55 threshold dominates and z
  is nearly irrelevant — confirmed by the z∈{1.645,1.0,0.0} sweep converging).
- **P2.1 jointly locks** (`promote_threshold=0.55`, `promote_z`, `min_h2h_games`, `--mirror-battles`) from the sim
  + the correlated-draw variant; the `GateConfigV2` edit + the test updates (`test_gate_v2` currently assumes
  0.70) land in **P2.2**. NOT flipped blind this turn — at the current n=240 it would leak ~5% false-promotes and
  break the gate-v2 tests.

## Mirror-collapse revert — degradation guard (user decision 2026-06-17)
The scripted collapse-revert only watches the SCRIPTED axis (a weak, stylistically narrow floor). A learner can
erode in REAL strength — drift significantly BELOW 50% vs its own frozen champion — while scripted stays flat;
today that case just HOLDs and keeps training the eroded model indefinitely (the plateau backstop won't fire
below 50%). This is the legitimate kernel of the (otherwise-harmful) "reset after N failed evals" idea, done on
the CORRECT signal: measured degradation vs your own best self, not failure to clear the high bar.
- **Rule (in `promotion_gate_v2`):** if the mirror ran with enough games AND `wilson_upper(mirror_wins,
  mirror_games, z) < 0.5 - mirror_collapse_margin` (the learner is significantly worse than the frozen champion)
  → verdict `"revert"`, reason `"mirror_collapse"`. Priority: alongside `scripted_collapse` (both are reverts),
  BEFORE `beat_champion`/`plateau`. Reuses the existing revert wiring (`restore_fn` → champion, reset optimizers,
  reset PFSP + h2h_history) — minimal new code.
- **False-revert guard:** right after a champion advance the learner restarts near 50% vs the NEW (stronger)
  champion, so the bar must be SIGNIFICANTLY below 50% (start `mirror_collapse` at Wilson-upper < ~0.45) AND
  require enough games — both calibrated in P2.1's sim so a healthy just-re-anchored learner is never falsely
  reverted. Reuses the same `wilson_upper_bound` the HoF adds.
- **P2.1** calibrates (`mirror_collapse_margin`, z, min games); **P2.2** adds the rule + `GateConfigV2` fields +
  a test. Complementary to the plateau backstop (which handles *stalled at ≥50%*); together they cover the full
  below-bar space: climbing→hold, stalled-not-losing→backstop-promote, eroding-below-self→revert.

## P2.1 — calibration LOCKED (2026-06-17; evidence in `artifacts/logs/p2_task/`, `gate_sim --demo`)
Monte-Carlo over the exact Wilson math (vectorised bounds pinned == `gate.py` scalar ones by a fidelity test).
- **HoF veto:** z=1.96, bar wilson_upper < 0.50, n=60/suspect, min_games_floor=40, K=3, per_band=2 (<=6 suspects).
  Catch@0.20 99.9%, catch@0.30 89.9%, per-snapshot false-reject@0.50 2.6%. FWER@0.55(M6) 1.9% (realistic),
  FWER@0.50(M6) 14.4% (pathological coin-flip-vs-all — accepted, only a HOLD). Robust to rho <= ~0.005. REGRESSION
  GUARD: per-snapshot catches a [0.20]+[0.60]x5 counter 99.9% vs band-mean 0.0% (band mean 0.533 >= 0.5, blind).
- **Mirror 0.55:** promote_threshold=0.55, promote_z=1.645, min_h2h_games=360 (--mirror-battles 360). FP@0.50 3.1%,
  pow@0.58 89%, pow@0.60 97.5%. true-0.55 promotes ~50%/gen (~2 gens) — fundamental, can't be raised by n.
  CORRELATION-SENSITIVE: FP rises to ~8-14% at rho 0.002-0.005 — accepted; laterals caught by HoF + collapse-revert.
- **Mirror-collapse revert:** bar wilson_upper < 0.45 (margin 0.05), z=1.645, min games 360. false-revert@0.50 0.0%,
  @0.48 0.3% (healthy re-anchored learner never reverted); catch@0.35 98.7%, @0.30 100%, @0.40 ~60%.
- **FORCE-VALVE SIMPLIFIED (sim finding):** the auto-force-MARGINAL branch is DEGENERATE — a persistently-vetoed
  suspect at n=60 ALWAYS has point estimate < 0.40 (catastrophic), so the marginal-advance window is EMPTY
  (force_valve_rate cut=0.40 -> 0%; cut=0.0 -> 100%). RESOLUTION: drop auto-advance; a plateau_reanchor HoF-rejected
  hof_force_limit=4x consecutive -> FREEZE + loud operator alert + manual --hof-override (force_limit = the ALERT
  threshold). A real marginal auto-advance would need ESCALATING games on the blocking suspect — deferred.
