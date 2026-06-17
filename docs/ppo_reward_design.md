# PPO Self-Play Reward Design — Victory-Dance VGC bot

Design plan for the reinforcement-learning (PPO self-play) phase. Produced 2026-06-16 from a
multi-perspective design pass (4 independent proposals — theory-purist, sample-efficiency, VGC
domain, self-play systems — each adversarially red-teamed for reward-hacking, then synthesized).
This is the reference for tasks #3a/#3b/#3c. **Decisions marked _(default)_ are recommendations
the user can override before the relevant phase.**

## 1. Objective — what the agent optimizes

**Terminal win/loss only, zero-sum.** `r_T = +1` for the side that wins (all 4 of the opponent's
brought mons faint, or the opponent forfeits/times-out in our favor), `r_T = −1` for the loser, `0`
on all intermediate steps (pure variant). Symmetric ±1 (not 0/1) so the two self-play perspectives
are exact negations and advantages stay zero-mean across the pair. Map the value head onto this axis
at use-time: `v_pm = 2·sigmoid(value_logit) − 1`.

Terminal edge cases (baked-in red-team fixes):
- **Showdown-adjudicated timeout** (server declares winner by most-mons then HP%): take the server's
  ±1 verbatim. Do **not** invent a separate timeout reward — that incentivizes stalling.
- **True draw** (server declares no winner): `r_T = 0`, **no bootstrap** — stalling-to-a-real-draw
  while ahead must be strictly worse than winning.
- **Artificial training-horizon cut** (our own step budget, set ABOVE the 99th-pct game length,
  ~80–100 turns): bootstrap with `γ·v_pm(s_cut)`, **never** hand ±1 (would teach "reach the cap").
- **Engineering failure / fallback-induced forfeit** (protocol desync, illusion-class bug): score as
  **draw(0) for both sides AND discard the trajectory** from the PPO buffer. Never +1 to the
  survivor — in shared-weight self-play that turns a robustness bug into a learned adversarial target.
  Reserve −1 only for a genuine Showdown rules-rejection (should be impossible at 100%-legal serve;
  if it fires, it's a logged bug alarm).

## 2. Critic — reuse the trained value head

**Initialize the PPO critic FROM the existing BC value head** (do not reinitialize). It's the biggest
sample-efficiency lever: a `Linear(trunk_out, 1)` on the shared BC trunk, BCE-calibrated (ECE
~0.01–0.02; phase acc lead 0.58 / mid 0.64 / endgame 0.68) — a far better cold-start than zero-init,
and calibrated exactly where wins are decided.

- **_(default)_ Clone a SEPARATE critic** copy of `trunk+value_head` from the BC weights so policy and
  critic optimize independently. The trunk is small `((512,256))`; decoupling protects the calibrated
  value surface from policy-gradient drift / warm-start collapse. (Alt: shared trunk — least code, but
  needs small value-loss weight + value clipping to stay stable.)
- **Hard critic-only warm-up**: freeze the actor for K updates so V migrates from its flat-{0,1}
  (≈γ=1) BC scale to the γ=0.997 discounted-return scale; THEN unfreeze the actor at a **small** LR.
- Keep the win-prob parameterization so the live `win-prob: 0.XX` readout keeps working as a
  diagnostic.

## 3. Discount & advantage

- **γ = 0.997** (not 0.99 — discounts a 30-step-out win to 0.74, killing sacrifice credit; not 1.0 —
  ill-conditioned with bootstrapped truncation/draws). Use the **same γ** inside the PBRS `F` term.
- **GAE λ = 0.95.**
- **Reward:** do NOT z-score / running-normalize / clip the terminal reward (would break the PBRS
  scale guarantee and zero-sum antisymmetry). Verify no `VecNormalize`/reward-clip wrapper is silently
  applied.
- **Advantage:** standardize GAE advantages to mean-0/unit-std per minibatch (affine, symmetry-safe);
  use a running std with a floor when minibatches are small. Sample BOTH perspectives of a game into
  the same minibatches.
- **Value space:** pick ONE space and assert it everywhere (the #1 silent bug). Phase 1: keep the
  critic in win-prob [0,1] (BCE on the bootstrapped {0,1} outcome), derive `V=2p−1` for advantages.
  Phase 3: switch the GAE critic to Huber/MSE on the bootstrapped shaped return, in the same space as
  the reward; keep value-loss clipping on.

## 4. Shaping — gated PBRS only (Phase 3, _default_ = start without it)

Start **pure sparse ±1, no shaping**. Add shaping ONLY if Phase 1/2 show measurably slow learning
(flat gauntlet Elo for N updates + high advantage variance + value loss not dropping). When on, it is
**PBRS only**: `F_t = γ·Φ(s_{t+1}) − Φ(s_t)`, Φ a function of **state only** (no action, no history,
no seen-gating). Telescopes to a policy-independent constant `−Φ(s_0)` (Ng/Harada/Russell 1999, Thm 1)
→ provably cannot move the optimum.

**No non-potential dense bonus, ever** (no per-KO, per-HP-damage, per-turn survival bonus, raw time
penalty). **Dropped for v0** (high hack-surface, flagged by red-team): HP-differential potential
(punishes correct sacrifices, enables chip/protect milking), boost/hazard/screen/Trick-Room terms
(cycling/re-cast farms).

**Recommended Φ (default):** a FROZEN-snapshot value-head win-prob —
`Φ(s) = c · (2·sigmoid(value_logit_FROZEN(s)) − 1)`, `c = 0.1–0.3`, from a **separate** `nn.Module`
snapshot of the pre-RL BC value head (`.eval()`, `requires_grad_(False)`), refreshed only on
gauntlet-accepted checkpoints. It already prices hazards/Tailwind/boosts/positioning implicitly and
gives the *correct positive* shaping on a sac-into-sweeper.

**Fallback Φ** (if the frozen snapshot proves OOD-weak on self-play states): privileged ground-truth
mon-count differential `Φ_ko(s) = c · (own_brought_alive − opp_brought_alive)/4`, `c=0.25`, using the
**true brought-4 faint counts of BOTH sides** (self-play owns both perspectives via
`_request_own_state`). **Critical:** do NOT source the opponent term from the encoder's seen-gated
globals (`opp_seen_alive`/`opp_seen_fainted`) — they move on a pure reveal, making Φ path-dependent.
Own-side globals are exact and fine. Φ shapes reward only (never feeds the policy net), so privileged
opp info leaks no hidden state and PBRS invariance holds.

**λ anneal:** ramp λ → 0 over the final ~30–40% of training and hold at 0, so the shipped policy
converges to and is judged on the pure ±1 objective.

## 5. Invariance unit-tests / anti-hacking safeguards (MUST hold if shaping is on)

- **`Φ(s_terminal) := 0` hard-coded** for all terminals. Unit-test: `shaped_r == terminal_r` exactly
  at every terminal transition. (Otherwise the last F-term is a ~±1.25 outcome-correlated dense reward.)
- **Frozen-Φ separateness:** `assert id(phi_net) != id(critic)`; assert `phi_net` params have zero grad
  every update. (The telescoping assert alone does NOT catch a non-frozen Φ.)
- **Per-episode telescoping residual** `sum_t F_t + Φ(s_0) ≈ 0` (alarm on drift), PLUS the
  cross-episode check: shaped-return vs gauntlet win-rate — shaped return rising while win-rate flat =
  Φ is being gamed.
- **Shaping fraction** `mean(|Σ λ·F|)/mean(|terminal R|)` hard-capped < 0.3 and → 0 as λ anneals.
- **Same γ** in F as the RL discount.
- **Standing shaping-OFF ablation:** shaped-run asymptotic gauntlet Elo must be ≥ sparse-run Elo (PBRS
  changes speed, not the optimum). If the optimum shifts, the shaping is mis-implemented — debug, don't
  ship.

## 6. Self-play structure

- **Zero-sum, single shared policy plays both sides.** Each game → two trajectories (own side via
  `_request_own_state`, opponent side via the gap-#6 `opp_snapshot` reconstruction). Enforce symmetry
  in code: `r_us == −r_opp`; discounted returns sum to ~0 at equal T. **Same turn clock for both sides**
  so γ^T cancels and discount-asymmetry self-collusion is impossible.
- **Opponent league (anti-collapse):** frozen-snapshot pool — ~50% latest checkpoint, ~30–35% past
  **gauntlet-accepted** snapshots (lightly PFSP-weighted toward ones beating the latest), **~20–25%
  scripted gauntlet anchors** (random/max_damage/heuristic) early, decaying toward ~10–15% as the pool
  grows (tie decay to pool size). Reuse `gauntlet.py` `_make_opponent('prev_best')` plumbing.
- **Team diversity:** randomize both sides' 6 + bring-4 from the ≥16-team pool via the seeded
  `team_matchups` rotation + side-balancing already in `gauntlet.py`. Vary leads.
- **Team preview: FREEZE the TP net for v0** (sample brings from it; don't RL bring-selection in this
  loop — separate optimization, revisit later). Episode = one full battle from brought-4 to terminal.
- **League admission = the gauntlet regression gate:** a checkpoint enters the pool (and refreshes the
  frozen-Φ snapshot) ONLY if it doesn't regress Elo vs the scripted ladder (≥4 teams, enough battles —
  small-sample noise is documented). This couples self-play progress to the same yardstick that accepts
  checkpoints, so the loop can't reinforce a strong-vs-self / weak-vs-field local optimum.

## 7. Metrics to log

Ground truth: **gauntlet Elo + win-rate vs each scripted opp every N updates (≥4 teams)** — the
acceptance gate; everything else is diagnostic.

Diagnostics: self-play win-rate of latest vs pool (~50% healthy); PBRS telescoping residual + shaped-
return-vs-winrate divergence; shaping fraction (<0.3, →0); episode-length distribution + timeout/draw
fraction (spike toward cap = stall hack; collapse = reckless); value-head calibration on fresh data
(ECE/Brier/phase-acc vs 0.58/0.64/0.68); critic explained-variance, value-loss, advantage mean(~0)/std,
advantage-vs-Φ correlation; PPO KL(old‖new), KL-from-BC, clip-fraction, per-head entropy; action-mix
per slot (switch/Protect/mega/forced-replacement rates, no-progress-turn rate); damage-to-own-ally
rate + target sanity; sacrifice-rate / KOs-conceded-then-won (confirm long-horizon sacrifice is
LEARNED, not suppressed); **MODEL-DRIVEN% (must stay ≥99%)** + illegal/fallback rate (~0); p1-vs-p2
win-rate (~50% under symmetry — skew flags a perspective-flip bug).

## 8. Phased rollout

| Phase | Reward | Goal | Ship-gate |
|---|---|---|---|
| **0 — Plumbing** | ±1 computed, not optimized | clean, legal, symmetric, fallback-free two-perspective trajectories | ≥200 games: MODEL-DRIVEN ≥99%, symmetry assert never trips, p1/p2 ~50±5%, terminal-space asserts pass |
| **1 — Sparse PPO** | pure ±1, warm-started critic, critic-only warm-up → small-LR actor, KL-to-BC + entropy | validate end-to-end learning; does the warm critic + GAE solve credit assignment? | gauntlet Elo rises & doesn't regress vs BC; no warm-start collapse; ≥1 checkpoint passes the gate. **May STOP here.** |
| **2 — League maturation** | same ±1; grow pool + PFSP + anchors | mature league, confirm no cycling, MEASURE if speed is a real problem | if Elo still climbing → ship; only flat-Elo-for-N + high-variance triggers Phase 3 |
| **3 — Gated PBRS** | `±1 + λ(k)·F`, frozen value-head Φ, Φ(s_T)=0, λ→0; critic→Huber on shaped return | per-turn gradient WITHOUT moving the optimum; prove unbiased via ablation | shaped run hits the SAME asymptotic Elo as sparse but faster; telescoping ~0; shaping fraction <0.3→0; no stall/sacrifice-suppression. Else revert to sparse. |

## 9. Open decisions (defaults chosen; override before the relevant phase)

1. **Shaping at all** — _(default)_ gated PBRS (sparse-first, add only on measured stall). Alts:
   never-shape; always-on PBRS.
2. **Critic architecture** — _(default)_ cloned separate critic. Alts: shared trunk; shared +
   stop-gradient.
3. **Phase-3 Φ source** — _(default)_ frozen value-head Φ, with privileged mon-count Φ_ko as fallback.
   Decide at Phase 3.
4. **Scripted-anchor mixture** — _(default)_ ~20–25% early, decay to ~10–15% with pool size. Decide at
   Phase 2.
5. **Phase-3 critic loss** — _(default)_ dual-head: Huber on shaped return for GAE + frozen BCE
   win-prob head kept as the live diagnostic. Decide at Phase 3.

## 10. Risks

- Sparse ±1 may learn slowly over the 10–40 step horizon until the critic localizes credit, esp. in the
  lead phase where the value head is weakest (0.58) — exactly where Protect/switch mindgames live.
- **Warm-start collapse:** BC value head trained on flat {0,1}; as a γ=0.997 baseline its scale is
  initially off — large early advantages + aggressive actor LR can blow away the BC prior. Mitigated by
  critic-only warm-up, small actor LR, value clip, KL-to-BC, more-frequent early gauntlet runs.
- **Silent PBRS invariance breaks** (Phase 3): nonzero terminal Φ, non-frozen Φ, value-space mismatch,
  γ mismatch — all covered by §5 asserts but the most likely week-wasting bugs.
- **Non-transitive self-play cycling** (VGC rock-paper-scissors team intransitivity) — Elo within the
  league can mislead; the scripted-anchor gauntlet stays the only trustworthy yardstick (≥4 teams).
- **Partial observability:** privileged-Φ works in training but the deployed policy faces hidden info —
  some self-play strength may not transfer; the value-head Φ may be OOD-weak on self-play states (no
  bias under PBRS, just no speed benefit).
- **Stall-to-tiebreak when ahead** is endorsed by the TRUE objective (server adjudicates by mon-count/HP),
  so PBRS can't remove it; mitigated (cut above 99th-pct, draw=0-no-bootstrap, tiebreak logging) not
  eliminated.
- **Frozen TP** decouples bring-selection — a co-adapted TP+battle optimum is out of reach until a later
  phase.
- **Operational cost:** league store + PFSP + per-checkpoint gauntlet gating + the assertion/logging
  suite is substantially more infra than a single dense-reward loop.

## 11. Edge-case strategies that must NOT be suppressed (charter)

VGC is full of legitimate, sometimes win-essential lines where the agent **pays a cost on its own
board** (loses HP, self-statuses, faints a mon, hits/heals an ally, sits idle, Protects, stalls) for a
delayed or non-local payoff. A multi-perspective sweep (2026-06-16) confirmed the **terminal-only
reward handles essentially all of them** — it is strategy-agnostic: it rewards *winning*, however you
win, and credit flows back to the cost turn through γ=0.997 (+ the warm critic / GAE).

**CHARTER:** Reward is terminal, zero-sum ±1, read from Showdown's adjudicated winner. **Nothing the
agent does to its OWN side is ever a reward term, a penalty, or a gate.** Every cost a strategy pays is
priced ONLY by whether the episode is won. Diagnostics describe behavior; they never shape it. The only
acceptance gate is gauntlet win-rate.

Must-not-suppress (representative, non-exhaustive):
- **Friendly-fire ability/item procs** — Anger Point, Justified, Berserk, Weakness Policy, Steam Engine,
  Stamina, Cell Battery, Rattled (hit your own ally to trigger); Beat Up as a multi-hit boost/breakpoint
  engine; Pollen Puff ally-heal; Helping Hand/Coaching/Decorate ally-support.
- **Self-status sets** — Flame/Toxic Orb for Guts/Quick Feet/Facade/Poison Heal/Marvel Scale/Magic
  Guard; Rest (+Sleep Talk/Chesto).
- **Self-HP-cost moves** — Belly Drum, Substitute (incl. deliberately breaking your own Sub), Curse
  (Ghost), Pain Split, Steel Beam, Mind Blown, Life Orb chip; pinch-berry/threshold self-chip
  (Salac/Liechi/Sitrus/Berserk-below-50/Wimp Out timing).
- **Deliberate-faint utility** — Memento, Parting Shot, Healing Wish, Lunar Dance, Explosion/
  Self-Destruct, Final Gambit, Destiny Bond, Endure-into-a-payoff.
- **Sacrifice / pivot-sac** — hard-switch or slow-pivot (U-turn/Volt Switch/Flip Turn) to bring a
  sweeper in safely; sack a walled mon to reset the matchup; sack a non-mega body to protect the mega;
  late-game sack the spent mega to enable a partner / Healing-Wish reset.
- **Perish + trap / stall / residual** — Perish Song + trapping (Shadow Tag/Arena Trap/Mean Look/Block/
  Magnet Pull/partial-trap) and the Perish-pivot loop; PP-stall to Struggle; Protect/Detect/Wide Guard
  positioning; redirection (Follow Me/Rage Powder/Spotlight/Ally Switch) eating hits for a setter;
  Toxic/Leech Seed/Salt Cure/sand/snow/Leftovers/Poison-Heal asymmetry; Wish-tect; Rest-Talk; soft-lock
  (Encore/Taunt/Disable); **stalling to server-adjudication when ahead.**
- **Field-control setup that deals no damage** — Trick Room (and the correct choice to **stay slow** /
  decline Tailwind on TR), Tailwind, weather/terrain (incl. mega weather-setters), Reflect/Light Screen/
  Aurora Veil, situational hazards; Commander (Tatsugiri-in-mouth, non-acting); charging/recharge/
  Sky-Drop/semi-invulnerable non-acting states; **mega-evolution as a non-damaging bulk/ability/weather
  setup decision.**

**Diagnostic discipline** (these are LOGGED counters, NEVER reward terms or gates — see §7). The
discriminator between a real hack and a legit tactic is always **win-correlation**, plus:
- **damage-to-own-ally** — split by whether an ally ability/item/threshold actually *triggered* (legit);
  many ally-hits with *zero* triggers + no downstream state change is the hack signature. Ally-SUPPORT
  moves (Helping Hand/Coaching/Pollen Puff) are `_ALLY_KINDS`, never counted as "ally damage."
- **no-progress turn** — redefine: a turn is NOT no-progress if any board-state delta occurred (boost/
  drop, hazard/screen/weather/terrain/TR/Tailwind/Perish/trap/status/Sub/redirection change, residual
  tick, sacrifice/pivot). Legit setup/stall runs **terminate in a win**; a true stall-hack never
  converts. Alarm only if no-progress turns rise *while gauntlet win-rate falls*.
- **Protect/switch rate** — no "healthy band"; both extremes are situationally correct. Legit when it
  co-occurs with an opposing Perish/TR/Tailwind/weather timer ticking or a partner setting up.
- **episode length / timeout** — split "reached cap but SERVER-ADJUDICATED a win" vs "TRUE draw" vs
  "engineering-fallback abort" (different terminal handling, §1). Alarm only on cap-pileup with
  undecided bootstrap (~0.5) + flat Elo; never on long-but-converting games.
- **sacrifice rate / KOs-conceded-then-won** — log as CONFIRMATION sacrifice is being *learned*, not
  suppressed (healthy = nonzero, correlates with wins). A perspective whose sacrifices correlate with
  *losses*, or a p1/p2 skew, flags a **self-collusion / perspective bug**, not a strategy to suppress.

## 12. Exploration: correctly rewarded ≠ easily discovered

The sharp finding: many of the above are **rewarded correctly but unlikely to be DISCOVERED by sparse
self-play** — the setup action looks locally bad (friendly-fire, −50% HP, sacrifice) and the payoff is
delayed/non-local (a partner's next-turn sweep, a 3-turn Perish, a Commander alignment). This is an
**exploration problem, not a reward problem — never "fix" it with a reward bonus** (that re-opens the
hack surface). Seed discovery instead:
1. **Preserve the BC prior via the KL-to-BC term** (§2/§6) — the warm-started policy already clicks
   these where the human data did; KL keeps PPO from crushing them early.
2. **Scripted demonstration episodes in the early replay buffer** for the precise, rare sequences if
   they're still under-clicked (sack-then-sweep, Healing-Wish reset, Weakness-Policy friendly-fire).
3. **Opponent-league + gauntlet team-pool archetype injection** — make sure the ≥16-team pool and the
   frozen-snapshot/scripted opponents actually CONTAIN Perish-trap (Mega Gengar Shadow Tag / Gothitelle),
   Commander (Tatsugiri+Dondozo), Trick Room, setup-sweep, and mega trap/stall/weather cores, so those
   states recur, the value head learns the boosted/setup board is winning, and the policy must mirror
   them. Reuse `gauntlet.py` `_make_opponent('prev_best')` plumbing to inject them.
4. The **warm-started critic** localizes long-chain credit far faster than a zero-init critic — the main
   reason it must be reused (§2).

## 13. Additional invariants (from the edge-case sweep)

- **γ is a FLOOR (≥0.997) — never lower it to speed convergence.** Lowering silently suppresses Perish
  (3-turn), screen/TR payoff windows (5–8 turn), and 25–40-turn residual-stall wins. If truncation is
  ill-conditioned, fix it with the bootstrap (γ·v at the cut), not a smaller γ.
- **Validate the step-cut against the ACTUAL episode-length distribution** (not assumed): set it above
  the 99th-pct of legitimate 25–40-turn stall/Perish wins, or those wins get truncated-and-bootstrapped
  and the slow archetypes are silently undervalued even under terminal-only reward.
- **Masking-confirmation startup assert / regression test:** the self-play action mask must keep the
  whole edge-case class LEGAL — ally-target bucket present; self-faint/self-HP moves (Explosion, Final
  Gambit, Memento, Belly Drum, Curse, Substitute, Steel Beam) selectable; Protect/Perish/setup/
  redirection selectable; the gimmick (mega) head usable on a non-damaging turn; the ally-damage block
  stays reverted. Guards against a future masking "cleanup" silently re-suppressing the class.
- **No silent reward wrapper:** assert at env-build that NO VecNormalize / reward-clip / z-score wrapper
  touches the terminal reward (would break PBRS scale + zero-sum antisymmetry); terminal is the raw ±1
  from the server's adjudicated winner.
- **Before enabling PBRS, validate the frozen Φ on held-out edge-case states** (post-Belly-Drum,
  post-Memento, mid-Perish, TR-up, statused-Guts): if Φ *drops* on those vs the prior state, the
  snapshot is undertrained on that archetype → keep PBRS off or re-snapshot from a checkpoint that has
  seen them. The `<0.3` shaping-fraction cap + λ-anneal is the bound that keeps even a mis-signed Φ from
  dominating the terminal return on a sac/Drum/TR turn. Prefer the frozen value-head Φ over the
  mon-count fallback for this whole class (the fallback injects a transient negative F on every
  sacrifice turn — optimum-invariant asymptotically but biases the partially-trained policy).
- **Self-collusion guard is STRUCTURAL, never a per-faint penalty:** the one place "sacrifice" can be a
  genuine hack is shared-weight self-play learning to throw a perspective. Address it with `r_us==−r_opp`
  + same turn clock + both-perspective minibatching + p1/p2≈50% and sacrifice-vs-loss correlation
  alarms + discarding fallback episodes — NOT by penalizing faints (which would suppress real sacrifice).
- **Diagnostics-to-penalty drift is a standing risk:** a future well-meaning change could promote a
  logged counter (ally-damage, no-progress, Protect-rate, episode-length, sacrifice-rate) into a reward
  term or gate and re-suppress the whole class. The charter (§11) must stay loud, and the acceptance
  gate reads **gauntlet win-rate ONLY**.

## 14. Team-Preview (TP) co-development roadmap

TP (pick 4 of 6 + leads) is the **episode-root action**: it gets the same terminal ±1 as the battle.
That shared objective couples TP and battle correctly — **no TP-specific reward** (a coverage /
speed-tier / "balanced bring" bonus is a means, not the goal, and re-opens the hack surface). TP is
trained on win/loss only, like everything else.

- **Staging (don't fully co-train from scratch — TP has one decision/game vs the battle's ~20, so its
  gradient is far noisier):**
  - **v0 — freeze TP** (sample brings from the trained TP net). Get the battle loop working against a
    fixed bring distribution first.
  - **v1 — alternating best-response (coordinate ascent):** freeze battle -> train TP against the
    improved battle policy (baseline = a TP value head, or bootstrap from the **turn-1 in-battle value
    head**, which already estimates "how good was this bring?"); then freeze TP -> continue battle.
    Repeat, each phase gauntlet-gated. Alternating also absorbs the **distribution shift** an improved
    TP imposes on the battle policy.
  - **v2 (optional) — joint fine-tune** only if v1 plateaus and the TP critic exists for variance
    reduction.
- **Ordering pitfall (critical):** a bring's value is **entangled with the battle AI's competence at
  piloting it.** If TP judges brings against an incompetent pilot it learns "Perish-trap / TR /
  Belly-Drum lose" — because the *pilot* is bad, not the bring — and TP+battle **co-collapse onto
  easy-to-pilot brings.** So **seed battle competence at the hard archetypes (sec 12) BEFORE opening
  TP-RL.**
- **TP is a one-shot, hidden, simultaneous decision** (you see opp's 6, not their bring; both pick
  blind) -> a deterministic TP net is exploitable. Keep TP **stochastic** (sample, tuned temperature),
  lean hard on **league/opponent-roster diversity**, optionally track **TP exploitability**
  (best-response win-rate of an opponent who knows your bring).
- **Keep TP a separate net** (perm-equivariant over rosters; different inputs/timescale from the
  board-state battle net). Couple via the shared reward + alternation, **not** a shared trunk.
- **Evaluate TP only jointly, on gauntlet win-rate** — never on `bring-exact` (many brings are
  near-equivalent; current 0.20 bring-exact understates it).
- **Cheap now (do in 3a):** record the TP decision (rosters seen -> bring + leads) + the game outcome in
  the trajectory schema even while TP is frozen — unblocks v1 with zero re-plumbing and gives per-bring
  / per-matchup win-rate diagnostics immediately.

## 15. Training team sampling & matchup-variance handling

Draw **both sides' teams randomly from the M-A pool (~70 edge/meta teams), independent draws**, for
archetype + matchup exposure (the team-level version of sec 12's archetype injection — the agent learns
to pilot *and* counter each team's win-cons). Reuse `gauntlet.py`'s seeded `team_matchups` rotation +
side-balancing.

**The lopsided-matchup worry is a VARIANCE problem, not a fairness problem — never fix it in the
reward.** Some teams hard-counter others; that win/loss is confounded with the team draw. But:
- It does **not bias** the gradient (team draw is independent of the policy + both sides played) — it
  only adds variance, so the whole fix is variance reduction.
- **Both-sides (antithetic) paired sampling** — play each team-pair once per orientation (ideally same
  engine seed). Team advantage cancels in the pair; the difference isolates skill. (Free given
  side-balancing.)
- **The matchup-aware turn-1 value baseline is THE handler.** The critic conditions on both rosters, so
  it learns "these two teams -> win-prob 0.85 at turn 1." Advantage = return - V(s) then measures *"did
  I do better than expected for THIS matchup"* -> a lopsided matchup won as expected contributes ~0
  advantage -> ~0 gradient. The learning signal concentrates on games decided by *play*. (Another
  reason the value-head critic is non-negotiable here.)
- **Random teams in TRAINING, controlled/side-balanced fixed teams in EVALUATION** — so a generation's
  measured strength isn't "got lucky with draws." **Do NOT** reward-adjust by matchup difficulty
  (biased + hacky).
- **Non-transitivity (A>B>C>A) is expected** with 70 teams; the frozen-snapshot league + scripted
  anchors + diverse sampling prevent archetype-forgetting; the **gauntlet stays the only trustworthy
  progress metric** (in-league Elo can mislead under cycles).

## 16. Generation sizing & the promotion gate

No magic number — **gate generations on EVIDENCE, not a fixed game count.** A checkpoint advances /
enters the league only when it beats the current best on the gauntlet by **more than noise** (win-rate
confidence interval); the game count is whatever clears that bar.

Ballparks (VGC ~15-30 decisions/side, ~2 trajectories/game):
- **>=250-500 games per PPO update** (floor for a non-noise advantage estimate; more is smoother).
- **~1,000-5,000 training games per generation** (lean higher given matchup variance).
- **>=100-200 side-balanced evaluation games** for a trustworthy generation-over-generation delta
  (8-game reads are pure noise — already learned; 30x4-team was trustworthy).

**Measure throughput first** (games/hour on the actual machine — poke-env + local Showdown is
wall-clock-bound) and size generations to wall-clock. Expect **fast early jumps** (BC warm-start
low-hanging fruit) then a slow grind — convenient for a timelapse.

## 17. Resumability & multi-session training (personal-PC reality)

Training runs on a personal machine that **can't stay on 24/7** — so the design constraint is **full
resumability**, NOT fixed games-per-generation (which does nothing for interruptibility). With
resumable checkpoints you can stop *any time* and lose almost nothing; chunked training is
mathematically identical to a continuous run (seeded RNG), so **pausing costs only wall-clock.**

- **Resume snapshot** = policy + critic weights, **optimizer state**, generation counter, **RNG state**,
  league/snapshot pool, Elo history, and the team-sampling seed/cursor. Load all -> continue exactly.
- **PPO is on-policy -> cheap to resume:** no large replay buffer to persist; an interrupted batch is
  dropped and re-collected. Snapshots are small/fast -> write them **often**.
- **Heartbeat checkpoint every K minutes / N games + per-generation**, plus **graceful shutdown**
  (catch SIGINT/close -> flush -> exit). Offer a "run N hours then checkpoint-and-stop" mode and a
  "run until Ctrl-C" mode, both landing on a clean resumable state. A power blip costs <= K minutes.
- **Decouple "generation" (logical unit) from "session" (wall-clock)** — generations span sittings.
- **Pin the poke-env / local-Showdown SHA (to-do #4) BEFORE long runs** — multi-session training over
  weeks means a silent dep/server update between sittings could change behavior mid-run; pinning makes
  every session identical. (Elevated priority now that training is multi-session.)

## 18. Per-generation archive & documentation (timelapse / YouTube)

Two **separate** stores:
- **Resume snapshot** — full training state, rotated/overwritten — for *continuing*.
- **Generation archive** — the gauntlet-gated best model per generation, tagged by generation + Elo,
  kept forever. This archive **doubles as the self-play league pool, the "battle generation N"
  checkpoints, and the timelapse source** (~3-4 MB each -> keep them all).

**Type_D replays (standard Showdown viewer HTML):**
- Source = the **omniscient spectator protocol stream** (what `--spectate` shows; full info since
  self-play owns both sides). Capture it for the showcase game.
- **Self-template** it into Showdown's replay-HTML and save as `data/.../Type_D/gen_<N>_<tag>.html`
  (full control over naming/location). **Vendor the replay-player JS/CSS offline** (don't CDN-link) so
  years-old generation replays still render.
- **Also save the raw `|`-log** alongside the HTML: it's what the VOD parser ingests, lets you
  regenerate the HTML, and lets **self-play games become future training data**.
- **Use a FIXED showcase opponent + team per generation** so the timelapse shows improvement against a
  constant baseline (not confounded by a changing opponent). Save the **Elo-vs-generation curve** as the
  honest headline metric (the cherry-picked best game is a highlight; the Elo curve is the truth).
- **Expectation:** BC warm-start -> gen-0 already plays decently; clear, documentable improvement over
  the BC baseline is achievable on chunked personal-PC compute; superhuman from self-play on one machine
  is a long shot (compute-bound). The improvement *arc* is the story.

## 19. Implementation work breakdown (sub-problems)

Derived from this doc. Dependencies in brackets. **3a/3b/3c = the PPO v0 (frozen-TP) path; v1 = TP
co-development; pin-deps is a prerequisite for long runs.**

**3a — Self-play data pipeline & schema (Phase 0 plumbing) [needs: reward design]**
- 3a.1 Trajectory schema: per-step (state, action_s0/s1, gimmick, value, log-prob, reward, done) +
  episode meta (both teams seen, **TP decision = bring+leads (sec 14)**, outcome, terminal-type
  {win/loss/draw/server-adjudicated/horizon-cut/engineering-fallback}, turn clock).
- 3a.2 Two-perspective collector (own via `_request_own_state`, opp via gap-#6 `opp_snapshot`) -> 2
  trajectories/game + **symmetry assert** (`r_us==-r_opp`, returns sum 0 at equal T, shared turn clock).
- 3a.3 Terminal reward (sec 1): server +-1; draw=0-no-bootstrap; horizon-cut=bootstrap gamma*v;
  **engineering-fallback=draw(0)+DISCARD**; **MODEL-DRIVEN% hard-fail (<99%)** on the corpus.
- 3a.4 **Type-C ReplayBuffer -> training-schema converter** (the named first brick) + **no-reward-
  wrapper assert** + **masking-confirmation startup test** (edge-case moves/targets stay legal, sec 13).
  *(IMPLEMENTED as a standalone `local_battle/self_play/store.py` Trajectory-jsonl store — NOT a
  `vgc_base.ReplayBuffer` extension, which is left untouched for the BC path. The env-build
  no-VecNormalize assert + the live masking-confirmation test were MOVED to 3c (runtime checks needing
  the env/player); store.py carries a data-level `assert_terminal_rewards_clean` instead.)*
- 3a.5 Record TP decision + outcome + per-bring/per-matchup win-rate diagnostic (sec 14).
- 3a.6 Phase-0 validation harness: >=200 games clean/legal/symmetric, p1/p2 ~50+-5%, terminal-space
  asserts.

**3b — PPO trainer [needs: 3a]**
- 3b.1 Actor-critic init from BC; **cloned separate critic** (sec 2); load `BCPolicy` weights.
- 3b.2 GAE (**gamma=0.997 floor (sec 13)**, lambda=0.95) + advantage standardization (per-minibatch, std
  floor, both perspectives in the same minibatch).
- 3b.3 PPO clip loss + value loss (BCE win-prob in Phase 1; value-clip on) + entropy bonus + **KL-to-BC
  prior** penalty.
- 3b.4 **Critic-only warm-up** (freeze actor K updates -> small-LR actor unfreeze) + warm-start-collapse
  guards (KL-from-BC / explained-variance auto-halt).
- 3b.5 Per-head action handling (2 slot heads + gimmick + forced-replacement) with serve-parity dedup.
- 3b.6 Single value-space assertion (terminal/bootstrap/target).
- 3b.7 **(Phase 3, GATED)** PBRS: separate frozen-Phi snapshot module (zero-grad, id!=critic),
  `F=gamma*Phi'-Phi`, **`Phi(terminal)=0` unit-test (`shaped_r==terminal_r`)**, shaping-fraction cap
  <0.3 + lambda-anneal, frozen-Phi edge-state validation (sec 13), telescoping + cross-episode checks.

**3c — Self-play loop, league, resumability, archive [needs: 3b]**
- 3c.1 Game runner: random team-pair from M-A pool, **both-sides paired sampling (sec 15)**, frozen TP
  for brings.
- 3c.2 Opponent league: frozen-snapshot pool + sampling (~50% latest / ~30% past-accepted PFSP / ~20%
  scripted anchors, decay by pool size); reuse `_make_opponent`.
- 3c.3 Generation loop: collect -> update -> **gauntlet eval (>=4 teams, side-balanced)** ->
  **statistical promotion gate (sec 16)** -> league admission + Phi-snapshot refresh.
- 3c.4 **Resumability (sec 17):** resume snapshot + heartbeat + graceful shutdown + run-N-hours / run-
  until-Ctrl-C modes.
- 3c.5 **Generation archive + Type_D replays (sec 18):** best-model-per-gen tagged by Elo; showcase-game
  omniscient-log capture -> offline-vendored Showdown HTML + raw log -> `Type_D/gen_<N>_<tag>`; fixed
  showcase opponent; Elo-curve log.
- 3c.6 Metrics/logging (sec 7): gauntlet Elo, self-play win-rate, telescoping, shaping fraction,
  episode-length dist, calibration, critic/PPO health, action-mix, sacrifice rate, MODEL-DRIVEN%, p1/p2.
- 3c.7 **Exploration seeding (sec 12):** KL-to-BC (in 3b) + scripted demo episodes in the early buffer +
  archetype injection into the team pool & league.

**v1 — TP co-development [needs: 3c stable + battle competent at archetypes (sec 14 ordering)]**
- v1.1 TP value baseline (turn-1 value head or a dedicated TP critic).
- v1.2 Alternating best-response loop (freeze battle -> train TP -> freeze TP -> battle), gauntlet-gated.
- v1.3 TP stochastic sampling + exploitability tracking.

**Prereq — pin deps:** pin poke-env / Showdown SHA before long multi-session runs (sec 17).
