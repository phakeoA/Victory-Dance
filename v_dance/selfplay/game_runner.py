"""Self-play game runner — recording player + collection helpers (task 3c.1a).

``SelfPlayVGCPlayer`` is the 100%-model-driven live player (``VGCPlayer`` + gap-#6
splice) instrumented to RECORD each model decision into a PPO ``Trajectory``: it
overrides the ``_record_rl_decision`` hook (added no-op to the base) to capture
(state, actions, gimmicks, decision-time masks, behaviour log-prob, critic value)
per turn AND per forced replacement, and ``_battle_finished_callback`` to finalise
the trajectory with the real outcome (+ terminal reward).

Both self-play sides share ONE ``ActorCritic`` (the shared policy plays both seats —
doc sec 6); each player keeps its OWN per-battle collector keyed by ``battle_tag``,
so the runner (3c.1b) can pair the two perspectives of a game by tag, assert zero-sum,
and write the Type-C store.

The pure helpers (``resolve_action`` / ``record_decision`` / ``finalize_trajectory`` /
``terminal_type_for``) carry no poke-env dependency beyond the shared ActorCritic, so
the recording logic is unit-tested offline; the thin battle glue (rebuilding the masks
from the live battle) is exercised by the live Phase-0 smoke (3c.1b / 3a.6).

Collection runs the policy STOCHASTICALLY (``tau`` > 0) so the behaviour log-prob is a
real distribution and the PPO ratio is meaningful (argmax has no usable log-prob).
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

# ── Path bootstrap (local_battle FIRST — mirrors player.py) ───────────────────
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
import torch

from v_dance.selfplay import policy_eval
from v_dance.selfplay.collector import (
    TrajectoryCollector, align_opponent_actions, assert_zero_sum,
)
from v_dance.selfplay.reward import _MODEL_DRIVEN_SOURCES, place_terminal_reward
from v_dance.selfplay.schema import PASS_ACTION, Transition, Trajectory
from v_dance.selfplay.store import assert_terminal_rewards_clean, write_trajectories

log = logging.getLogger(__name__)


# ── pure recording helpers (no poke-env) ──────────────────────────────────────
def _aslist(mask) -> Optional[List[int]]:
    return None if mask is None else [int(bool(x)) for x in mask]


def resolve_action(a, mask) -> int:
    """The collector's action index for one slot: ``PASS_ACTION`` when the slot made
    no decision (``a is None``) or has no legal action (mask absent / all-zero — an
    empty / fainted active slot the live player passes); otherwise ``int(a)``.

    This recovers the TRUE pass that the live player encodes as 0 on a normal turn
    (``None -> 0`` happens iff the slot's legal mask is all-zero), so a real action 0
    is never confused with a pass."""
    if a is None:
        return PASS_ACTION
    if mask is None or not any(mask):
        return PASS_ACTION
    return int(a)


def record_decision(collector: TrajectoryCollector, actor_critic, *, state,
                    a0, a1, g0=0, g1=0, mask0=None, mask1=None, gmask0=None,
                    gmask1=None, decision_type="turn", turn=0, tau=1.0):
    """Build one RL step from a model decision and append it to ``collector`` with the
    behaviour log-prob + critic value (``value_pm``) computed via ``actor_critic`` (one
    no-grad forward). Returns ``(resolved_a0, resolved_a1)``."""
    ra0 = resolve_action(a0, mask0)
    ra1 = resolve_action(a1, mask1)
    m0, m1 = _aslist(mask0), _aslist(mask1)
    gm0, gm1 = _aslist(gmask0), _aslist(gmask1)
    tmp = Transition(state=np.asarray(state, np.float32), action_s0=ra0, action_s1=ra1,
                     gimmick_s0=int(g0), gimmick_s1=int(g1),
                     mask_s0=m0, mask_s1=m1, gmask_s0=gm0, gmask_s1=gm1)
    # τ<=0 = a deterministic argmax seat (--sample ab p2): logits/τ would NaN the
    # log-softmax and the NaN would be silently stored. The argmax action has
    # probability 1 in the τ→0 limit → behaviour log-prob 0.0; the critic value
    # is τ-independent, so evaluate it at τ=1.
    with torch.no_grad():
        lp, _ent, vpm = policy_eval.evaluate_actions(
            actor_critic, [tmp], tau=(tau if tau > 0 else 1.0))
    collector.add_step(state=tmp.state, action_s0=ra0, action_s1=ra1,
                       gimmick_s0=int(g0), gimmick_s1=int(g1),
                       logprob=(float(lp[0]) if tau > 0 else 0.0), value=float(vpm[0]),
                       mask_s0=m0, mask_s1=m1, gmask_s0=gm0, gmask_s1=gm1,
                       decision_type=decision_type, turn=int(turn))
    return ra0, ra1


def terminal_type_for(won: Optional[bool], lost: Optional[bool]) -> str:
    """Map a finished battle's win/lost flags to a TerminalType value. Neither set =>
    a true draw/tie (reward 0, no bootstrap). (A FALLBACK forfeit is detected by the
    runner from the source counts, 3c.1b — not here.)

    ⚠ #22 (KNOWN GAP): this never returns HORIZON_CUT or ADJUDICATED — the live collector has no
    training step-cap, so a long game runs to a real ±1 terminal (correct today, nothing mislabeled).
    The HORIZON_CUT scaffolding (schema/gae bootstrap/reward) is therefore DEAD until a truncation cap is
    added; if one is, the producer must also record V(s_cut) and thread bootstrap_value through
    compute_batch_gae (it currently hard-drops it). Deferred to the RL launch."""
    if won:
        return "win"
    if lost:
        return "loss"
    return "draw"


def finalize_trajectory(collector: TrajectoryCollector, *, won: Optional[bool],
                        terminal_type: str, own_team: Sequence[str], n_turns: int,
                        tp_bring: Optional[Sequence[int]] = None,
                        tp_leads: Optional[Sequence[int]] = None,
                        opp_team: Sequence[str] = (),
                        sampling: Optional[dict] = None) -> Trajectory:
    """Finalise a collector into a Trajectory and place the terminal reward. ``n_turns``
    is the battle's real final Showdown turn (NOT the step count) so both perspectives
    share the clock even if one skipped a non-model turn (zero-sum stays valid).
    ``sampling`` labels the behaviour params (tau/top_p) the recorded logprobs assume."""
    bring = list(tp_bring) if tp_bring is not None else list(range(min(4, len(own_team))))
    leads = list(tp_leads) if tp_leads is not None else bring[:2]
    traj = collector.finish(own_team=list(own_team), opp_team=list(opp_team),
                            tp_bring=bring, tp_leads=leads, won=won,
                            terminal_type=terminal_type, n_turns=int(n_turns),
                            sampling=sampling)
    # T3.1: a FALLBACK trajectory (backstop-forfeit) carries no terminal reward — it must be
    # discarded from the batch, not rewarded. place_terminal_reward asserts is_trainable, so skip it.
    if traj.meta.is_trainable:
        place_terminal_reward(traj)
    return traj


# ── recording live player ──────────────────────────────────────────────────────
from v_dance.play.player import VGCPlayer  # noqa: E402  (local spliced player)
from v_dance.play.live_vgc_base import _norm_tag  # noqa: E402  (T3.1: match the forfeit-tag key)
from v_dance.play.vgc_base import (  # noqa: E402
    build_legal_action_mask, build_replacement_mask, build_gimmick_legal_mask,
)

# The step-RECORDING gate must mirror the MODEL-DRIVEN accounting set EXACTLY (else a model-driven
# source — notably B1 'search' — is counted as model-driven in phase0/reward but its turns are never
# recorded as Transitions, so n_steps→0 and MODEL-DRIVEN reads 0% on a legit search run, and the search
# training corpus is empty). Derive it from the canonical reward set so the two can never drift (audit 2026-06-30).
_MODEL_SOURCES = _MODEL_DRIVEN_SOURCES


class SelfPlayVGCPlayer(VGCPlayer):
    """VGCPlayer that records every MODEL decision into per-battle trajectories.

    Drives action selection with ``actor_critic.policy`` (the live, updating actor) so
    each generation collects under the current weights; records the critic's value
    (``value_pm``) + behaviour log-prob via the shared ActorCritic. Non-model decisions
    (retry / default / forced-switch-escape) are NOT recorded as steps — they are still
    counted in ``_source_counts`` so the Phase-0 MODEL-DRIVEN% reflects them."""

    def __init__(self, actor_critic, *, tau: float = 1.0,
                 sample_seed: Optional[int] = None, live_dir=None, save_replays=False, **kwargs):
        # #18: live_dir / save_replays flow to the BASE, which owns the spectate feed (self._live)
        # — so collection AND eval players share one implementation. No-op when live_dir is unset.
        super().__init__(model_path=None, temperature=tau, sample_seed=sample_seed,
                         live_dir=live_dir, save_replays=save_replays, **kwargs)
        self._ac = actor_critic
        self._model = actor_critic.policy            # drive with the live actor
        # Memory ckpt under RECORDING self-play (scan 2026-07-02): decisions run
        # on the frame-stack, but evaluate_actions has no memory path, so the
        # recorded behaviour logprob/value are SINGLE-FRAME approximations.
        # Win-rate A/Bs stay valid (gameplay is correct); the recorded store
        # must not feed importance-sampled offline RL without recomputation —
        # warn once and stamp the trajectory meta so it can never be mistaken.
        self._memory_single_frame = bool(int(getattr(self._model, "memory_dim", 0) or 0))
        if self._memory_single_frame:
            log.warning(
                "recording self-play with a MEMORY checkpoint: recorded logprob/value "
                "are single-frame approximations of stack-driven decisions — valid for "
                "win-rate A/Bs, NOT for importance-sampled offline RL (meta-stamped).")
        self._collect_sample = True                  # #9/#9b: sample gimmick + forced-replacement at tau (match recorded log-prob)
        self._record_masks = True                    # #10/#11: log the deduped behaviour mask the model sampled under
        self._model_heads = actor_critic.head_names
        self._tau = float(tau)
        self._collectors: dict = {}                  # battle_tag -> TrajectoryCollector
        self._finished: dict = {}                    # battle_tag -> Trajectory

    def _collector_for(self, battle) -> TrajectoryCollector:
        tag = battle.battle_tag
        c = self._collectors.get(tag)
        if c is None:
            c = TrajectoryCollector(tag, getattr(battle, "player_role", None) or "p?")
            self._collectors[tag] = c
        return c

    def _record_rl_decision(self, battle, state_vec, a0, a1, g0, g1, source, decision_type):
        # (the spectate feed is published by the base choose_move via _report_active, #18)
        if source not in _MODEL_SOURCES:
            return  # retry / fallback / escape are not the model's decision
        try:
            # #10/#11: prefer the per-slot mask the model was ACTUALLY sampled under (carries the cross-slot
            # switch/replacement dedup), stashed by _select_actions/_select_replacement_actions; fall back to
            # a fresh build per slot when absent (e.g. a non-recorded edge).
            _sm = getattr(self, "_sampling_masks", None)
            _stash = _sm.pop((battle.battle_tag, decision_type), None) if _sm is not None else None
            if decision_type == "replacement":
                m0 = _stash[0] if (_stash and _stash[0] is not None) else build_replacement_mask(battle, 0)
                m1 = _stash[1] if (_stash and _stash[1] is not None) else build_replacement_mask(battle, 1)
                gm0 = gm1 = None
            else:
                m0 = _stash[0] if (_stash and _stash[0] is not None) else build_legal_action_mask(battle, 0)
                m1 = _stash[1] if (_stash and _stash[1] is not None) else build_legal_action_mask(battle, 1)
                gm0, gm1 = build_gimmick_legal_mask(battle, 0), build_gimmick_legal_mask(battle, 1)
            c = self._collector_for(battle)
            # De-dup (3c.1c): poke-env re-calls for the SAME (turn, decision_type) only
            # when Showdown REJECTED the prior order. Replace the rejected step (keep just
            # the finally-EXECUTED pick) and count the rejection — the temperature-sampled
            # policy hits the approximate legal mask (Choice-lock / Encore / Taunt …) and
            # resamples a fresh legal action, which the source counts would otherwise miss.
            last = c.last_step()
            if (last is not None and last.turn == battle.turn
                    and last.decision_type == decision_type):
                c.pop_step()
                self._source_counts["rejected_resample"] += 1
            record_decision(c, self._ac, state=state_vec, a0=a0, a1=a1, g0=g0, g1=g1,
                            mask0=m0, mask1=m1, gmask0=gm0, gmask1=gm1,
                            decision_type=decision_type, turn=battle.turn, tau=self._tau)
        except Exception:
            log.debug("self-play record failed (non-fatal)", exc_info=True)

    def _discard_rl_decision(self, battle, decision_type):
        """Drop the current (turn, decision_type) step when a non-model order executed
        (retry perturbation / default / forced-switch escape) — so the trajectory holds
        only EXECUTED model decisions."""
        try:
            c = self._collectors.get(battle.battle_tag)
            if c is None:
                return
            last = c.last_step()
            if (last is not None and last.turn == battle.turn
                    and last.decision_type == decision_type):
                c.pop_step()
        except Exception:
            log.debug("self-play discard failed (non-fatal)", exc_info=True)

    def _battle_finished_callback(self, battle):
        try:
            tag = battle.battle_tag
            # #10/#11: drop any per-battle sampling-mask stashes for a last turn that was selected but not
            # recorded (discarded forced-partial/retry), so the dict can't accrete over a long run.
            _sm = getattr(self, "_sampling_masks", None)
            if _sm is not None:
                for _dt in ("turn", "replacement"):
                    _sm.pop((tag, _dt), None)
            c = self._collectors.get(tag)
            if c is not None and len(c) > 0:
                won = True if battle.won else False if getattr(battle, "lost", False) else None
                # 15b.0: prefer the TRUE TP decision captured at teampreview (real
                # submitted bring/leads + opponent roster). own_team comes from the
                # SAME capture so the bring indices stay consistent with the roster
                # they index. Fall back to the battle roster + finalize's first-4
                # default only when no decision was recorded (defensive).
                tp = (getattr(self, "_tp_decision", None) or {}).get(tag) or {}
                own_team = tp.get("own_team") or [
                    getattr(m, "species", "?")
                    for m in (getattr(battle, "team", None) or {}).values()]
                # T3.1: if WE forfeited this battle as a loop-guard backstop, it is NOT a real loss —
                # tag it FALLBACK (won=None) so finalize skips the −1 and run_generation drops BOTH
                # perspectives by battle_id (the opponent's mirror +1 is equally mislabeled). The base
                # _handle_force_switch records the forfeit in self._forfeited_tags keyed by _norm_tag.
                _forfeited = _norm_tag(tag) in getattr(self, "_forfeited_tags", ())
                _tt = "fallback" if _forfeited else terminal_type_for(
                    battle.won, getattr(battle, "lost", False))
                self._finished[tag] = finalize_trajectory(
                    c, won=None if _forfeited else won, terminal_type=_tt,
                    own_team=own_team, n_turns=getattr(battle, "turn", len(c)),
                    tp_bring=tp.get("bring"), tp_leads=tp.get("leads"),
                    opp_team=tp.get("opp_team") or (),
                    sampling={"tau": float(getattr(self, "_tau", 1.0)),
                              "top_p": float(getattr(self, "_top_p", 1.0)),
                              **({"memory_single_frame": True}
                                 if getattr(self, "_memory_single_frame", False)
                                 else {})})
        except Exception:
            # A finalize failure silently drops this game from self._finished → it can't be paired →
            # it vanishes from the A/B win-rate DENOMINATOR with no trace. Surface it (WARNING + a counted,
            # NON-model bookkeeping source in _NON_DECISION_COUNTERS) so the run report shows the shrunken
            # sample instead of hiding it (audit 2026-06-30).
            try:
                self._source_counts["finalize_failed"] += 1
            except Exception:
                pass
            log.warning("self-play finalize FAILED for %s — game DROPPED from results (non-fatal)",
                        getattr(battle, "battle_tag", "?"), exc_info=True)
        return super()._battle_finished_callback(battle)   # base (#18) saves/removes the spectate file

    def finished_trajectories(self) -> dict:
        """battle_tag -> finalised Trajectory for every completed battle."""
        return dict(self._finished)


# ── trajectory pairing + Phase-0 report (pure, offline-tested) ────────────────
def pair_trajectories(finished_p1: dict, finished_p2: dict) -> List[tuple]:
    """Pair the two perspectives of each game by battle_tag. Returns
    ``[(traj_a, traj_b), ...]`` only for tags BOTH players finalised (an unfinished /
    hung battle that only one side closed is dropped)."""
    return [(finished_p1[t], finished_p2[t]) for t in finished_p1 if t in finished_p2]


def duplicate_turn_steps(traj) -> int:
    """Number of EXTRA steps a trajectory carries for an already-seen (turn,
    decision_type) — i.e. rejected orders that were recorded but did NOT execute. After
    the 3c.1c de-dup this must be 0."""
    seen, extra = set(), 0
    for t in traj.transitions:
        key = (t.turn, t.decision_type)
        if key in seen:
            extra += 1
        seen.add(key)
    return extra


def phase0_report(pairs: List[tuple], source_counts: dict, *, min_games: int = 200,
                  target_model_driven: float = 0.99, balance_tol: float = 0.10) -> dict:
    """Compute the Phase-0 plumbing check (doc sec 8 / 3a.6) over paired self-play
    games. MODEL-DRIVEN is measured from the DE-DUPED trajectories (turns where a model
    pick actually EXECUTED) vs the non-model executions (retry / default / escape) — NOT
    from the raw per-attempt source counts, which over-count rejected resamples. The
    silent-resample rate (the approximate-mask quality signal) is reported separately,
    and ``data_clean`` asserts the de-dup left zero duplicate-turn steps."""
    n = len(pairs)
    sym_fail = 0
    p1_wins = p1_games = 0
    n_steps = dup_steps = 0
    for a, b in pairs:
        try:
            assert_zero_sum(a, b)
        except AssertionError:
            sym_fail += 1
        for t in (a, b):
            n_steps += len(t.transitions)
            dup_steps += duplicate_turn_steps(t)
        by_role = {a.meta.own_role: a, b.meta.own_role: b}
        p1t = by_role.get("p1")
        if p1t is not None and p1t.meta.won is not None:
            p1_games += 1
            p1_wins += 1 if p1t.meta.won else 0

    # A3/B1: pull the belief-feed AND search diagnostics OUT before the non-model math (else the
    # belief_*/search_* counts would be mis-counted as non-model executions and deflate MODEL-DRIVEN%).
    # Surfaced separately in rep["belief"] / rep["search_diag"]. The bare "search" DECISION source is
    # NOT stripped here — it's a real model-grounded pick and is excluded as model-driven below
    # (reward._MODEL_DRIVEN_SOURCES now lists it), exactly like "model".
    belief = {k[len("belief_"):]: v for k, v in source_counts.items() if str(k).startswith("belief_")}
    search_diag = {k[len("search_"):]: v for k, v in source_counts.items() if str(k).startswith("search_")}
    # W2 spawn path (2026-09-03): the spawner's THROUGHPUT counters (pairs / decisions / elapsed_s /
    # stalls / ghosts / abandoned) ride source_counts too — bookkeeping, not decisions. Stripped here
    # exactly like belief_* / search_* (the USER's first block read MODEL-DRIVEN 46% because ~290
    # spawn_* units were being summed as non-model steps) and surfaced as rep["spawn"].
    spawn = {k[len("spawn_"):]: v for k, v in source_counts.items() if str(k).startswith("spawn_")}
    sc = {k: v for k, v in source_counts.items()
          if not str(k).startswith("tp_") and not str(k).startswith("belief_")
          and not str(k).startswith("search_") and not str(k).startswith("spawn_")}
    resamples = sc.get("rejected_resample", 0)
    # audit: count EVERY executed NON-model source (forced_default / forced_partial / forced_switch /
    # forfeit / retry / retry_default / forced_switch_escape …) — a BLOCKLIST, not a 3-key whitelist that
    # silently drops a forced source it forgot to list (which biased MODEL-DRIVEN% HIGH and could green-
    # light a desync-corrupted run). Reuse reward.py's canonical sets so the two measures can't drift.
    from v_dance.selfplay.reward import _MODEL_DRIVEN_SOURCES, _NON_DECISION_COUNTERS
    _excluded = set(_MODEL_DRIVEN_SOURCES) | set(_NON_DECISION_COUNTERS)   # model + bookkeeping (tp_* already stripped)
    non_model = sum(int(v or 0) for k, v in sc.items() if k not in _excluded)
    finalize_failed = int(sc.get("finalize_failed", 0) or 0)   # games that failed to finalize → dropped from the sample
    md = (n_steps / (n_steps + non_model)) if (n_steps + non_model) else 0.0
    resample_rate = (resamples / (n_steps + resamples)) if (n_steps + resamples) else 0.0

    trajs = [t for pair in pairs for t in pair]
    try:
        assert_terminal_rewards_clean(trajs)
        clean, clean_err = True, None
    except AssertionError as e:
        clean, clean_err = False, str(e)

    # audit: the documented #1 silent PPO bug is win-prob [0,1] stored as value_pm [-1,1] — a range assert
    # cannot catch it (a [0,1] batch is a subset of [-1,1]). looks_like_winprob over the POOLED both-
    # perspective values (which MUST contain negative/losing states) is the real catch; wire it into the gate.
    from v_dance.selfplay.value_space import looks_like_winprob
    all_vals = [t.value for pair in pairs for traj in pair for t in traj.transitions]
    winprob_suspect = looks_like_winprob(all_vals)
    value_space_ok = not winprob_suspect

    p1_wr = (p1_wins / p1_games) if p1_games else None
    enough = n >= min_games
    md_ok = md >= target_model_driven
    sym_ok = sym_fail == 0
    bal_ok = (p1_wr is not None and abs(p1_wr - 0.5) <= balance_tol)
    data_clean = dup_steps == 0
    rep = {
        "n_games": n, "enough_games": enough, "min_games": min_games,
        "spawn": spawn,                       # W2 spawn-path throughput counters ({} on the challenge path)
        "model_steps": n_steps, "duplicate_steps": dup_steps, "data_clean": data_clean,
        "model_driven": md, "model_driven_ok": md_ok, "non_model_executed": non_model,
        "resample_count": resamples, "resample_rate": resample_rate,
        "finalize_failed": finalize_failed,
        "symmetry_failures": sym_fail, "symmetry_ok": sym_ok,
        "p1_win_rate": p1_wr, "p1_games": p1_games, "balance_ok": bal_ok,
        "terminal_clean": clean, "terminal_error": clean_err,
        "looks_like_winprob": winprob_suspect, "value_space_ok": value_space_ok,
        "belief": belief, "search_diag": search_diag,
    }
    rep["PASS"] = bool(enough and md_ok and sym_ok and bal_ok and clean and data_clean and value_space_ok)
    return rep


def print_phase0_report(rep: dict) -> None:
    def mark(ok):
        return "PASS" if ok else "FAIL"
    print("== Phase-0 self-play plumbing report =======================")
    print(f"  games (paired)    : {rep['n_games']}  (need >= {rep['min_games']})  "
          f"[{mark(rep['enough_games'])}]")
    print(f"  MODEL-DRIVEN      : {rep['model_driven']*100:.2f}%  "
          f"({rep['model_steps']} executed model steps, {rep['non_model_executed']} non-model)  "
          f"(need >= 99%)  [{mark(rep['model_driven_ok'])}]")
    print(f"  recorded data     : {rep['duplicate_steps']} duplicate-turn steps  "
          f"(must be 0 after de-dup)  [{mark(rep['data_clean'])}]")
    print(f"  silent resamples  : {rep['resample_count']}  "
          f"({rep['resample_rate']*100:.1f}% of model turns hit the approximate legal "
          f"mask + resampled)  [info]")
    if rep.get("finalize_failed"):
        print(f"  finalize failures : {rep['finalize_failed']} game(s) DROPPED from the sample "
              f"(trajectory failed to finalize — see WARNING log)  [warn]")
    print(f"  zero-sum symmetry : {rep['symmetry_failures']} failures  "
          f"[{mark(rep['symmetry_ok'])}]")
    wr = rep["p1_win_rate"]
    wr_s = f"{wr*100:.1f}%" if wr is not None else "n/a"
    print(f"  p1 win-rate       : {wr_s} over {rep['p1_games']} decisive  "
          f"(want ~50%)  [{mark(rep['balance_ok'])}]")
    print(f"  terminal space    : {'clean' if rep['terminal_clean'] else rep['terminal_error']}  "
          f"[{mark(rep['terminal_clean'])}]")
    print(f"  value space       : {'value_pm OK' if rep.get('value_space_ok', True) else 'looks like WIN-PROB [0,1] — value_pm mismatch!'}  "
          f"[{mark(rep.get('value_space_ok', True))}]")
    bel = rep.get("belief") or {}
    if bel:
        fired = sum(int(v or 0) for k, v in bel.items() if "used" in k)
        print(f"  belief feed       : {fired} constraints FIRED  "
              f"({', '.join(f'{k}={v}' for k, v in sorted(bel.items()))})  [info]")
    sd = rep.get("search_diag") or {}
    if sd:
        fired = int(sd.get("search", 0) or 0)
        fell = int(sd.get("fallback", 0) or 0)
        tot = fired + fell
        rate = (fell / tot) if tot else 0.0
        print(f"  search feed       : {fired} search decisions, {fell} fallbacks  "
              f"({rate*100:.1f}% fallback{' — search NEVER fired, A/B is NULL' if not fired else ''})  [info]")
    print("  ----------------------------------------------------------")
    print(f"  PHASE 0           : {'PASS - plumbing clean, ready for Phase 1' if rep['PASS'] else 'FAIL - investigate above'}")
    print("============================================================")


# ── live orchestration (reuses run_local_battle; exercised by the smoke) ──────
def _subdivide_matchups(matchups, target_chunks):
    """Split matchup chunks into ~``target_chunks`` total so a small team pool can still saturate the
    worker pool (task #13). Each ``(team_a, team_b, n)`` may become several ``(team_a, team_b, n_sub)``
    sub-chunks (``n_sub >= 1``, summing to ``n``); the extra splits go to the chunks with the most
    games-per-sub-chunk (largest-load-first), so the resulting sub-chunks are as even as possible.

    Motivating case: a SINGLE-team MIRROR A/B is ONE ``(team, team, n_games)`` chunk, which would pin the
    whole run to a single sequential pairing regardless of ``--workers``. Splitting it into up-to-n_workers
    sub-chunks — each later given a DISTINCT uid → distinct accounts + seeds — fills the worker slots with
    INDEPENDENT seed-streams that still sum to ``n_games`` and aggregate into the IDENTICAL A/B verdict.

    NO-OP when the pool already yields ``>= target_chunks`` pairings (multi-team / gauntlet) or when
    ``target_chunks <= 1`` (the default sequential run). Total games are preserved exactly; the (team_a,
    team_b) pair of every sub-chunk equals its parent, so per-matchup game totals are unchanged."""
    # 2026-09-03 (W2): ONE implementation, now in gauntlet (stdlib-pure) so the EVAL planners
    # share it - the own-vs-own champion mirror has exactly the same single-chunk problem.
    from v_dance.eval.gauntlet import subdivide_pairings
    return subdivide_pairings(matchups, target_chunks)


def _seat_sampling(sample: str, tau: float, serve_tau: float, serve_top_p: float):
    """Per-seat (p1_tau, p2_tau, p1_top_p, p2_top_p) for the serve-sampling modes.

    Serve-sampling A/B (the anti-predictability lever): VGC is a SIMULTANEOUS-move
    game, so a deterministic argmax policy is exploitable by construction — the
    principled serve policy is a MIXED strategy.  ``"ab"`` = p1 samples the masked
    policy softmax at ``serve_tau`` (optionally top-p nucleus-restricted) while p2
    plays pure argmax (τ=0) with the SAME net → p1's win-rate isolates whether
    serve-time mixing costs head-to-head strength.  ``"both"`` = smoke (both seats
    sample at serve_tau).  ``"off"`` (default) keeps the collection ``tau`` on both
    seats — byte-identical to the existing self-play path.
    """
    if sample == "ab":
        return float(serve_tau), 0.0, float(serve_top_p), 1.0
    if sample == "both":
        return float(serve_tau), float(serve_tau), float(serve_top_p), float(serve_top_p)
    return float(tau), float(tau), 1.0, 1.0


async def run_self_play_games(
    actor_critic, team_pool: Sequence[str], n_games: int, *,
    store_path=None, tau: float = 1.0, seed: int = 0, manage_server: bool = True,
    battle_timeout: Optional[float] = 90.0, matchup_seed: int = 0, n_workers: int = 1,
    match_belief: str = "off", search: str = "off", live_dir=None, save_replays: bool = False,
    n_servers: int = 1, sample: str = "off", serve_tau: float = 0.45, serve_top_p: float = 1.0,
    spawn_rooms: int = 0,
):
    """Play ``n_games`` self-play battles (the shared policy on BOTH sides) over a
    side-balanced team rotation, collect both perspectives of each game, and write the
    Type-C store. Returns ``(pairs, source_counts)``. ``n_workers`` (task #13) runs that
    many pairings concurrently via the shared runner (default 1 = sequential, as before).

    ``spawn_rooms`` > 0 (W2 throughput, 2026-09-03): each pairing runs through the SERVER-SIDE
    spawner instead of the challenge protocol — the plugin keeps that many battles alive between
    the pair's two accounts (``spawn_client.play_pairing_spawned``; the plugin is installed into the
    clone before the servers start). 0 = the challenge path, byte-identical. ``source_counts`` then
    also carries ``spawn_*`` throughput keys (pairs, decisions, elapsed_s, stalls, ghosts).

    ``match_belief`` (A3 within-game belief A/B): ``"off"`` (default — neither side),
    ``"both"`` (the live-wiring Phase-0 smoke — belief ON on both seats), or ``"ab"`` (the
    strength A/B — belief ON on p1/SP1 only, OFF on p2/SP2, so p1's win-rate isolates the
    belief's effect; run with a SINGLE team for a clean MIRROR). The per-seat belief-feed
    counts are aggregated into ``source_counts`` under ``belief_*`` keys."""
    p1_belief = match_belief in ("both", "ab")        # SP1 = belief-on side of the A/B
    p2_belief = match_belief == "both"
    p1_search = search in ("both", "ab")              # B1 search A/B: SP1 = search-on side
    p2_search = search == "both"
    p1_tau, p2_tau, p1_top_p, p2_top_p = _seat_sampling(sample, tau, serve_tau, serve_top_p)
    import logging as _logging
    from collections import Counter

    import v_dance.play.run_local_battle as R
    from poke_env import AccountConfiguration
    from v_dance.play import parallel_battles as PB
    from v_dance.selfplay import spawn_client as SC
    from v_dance.eval.gauntlet import team_matchups

    # #22f: shard workers across a POOL of K local Showdown servers (ports 8000..8000+K-1). ONE server's
    # connection-accept STORMS at startup under high --workers (single-server ceiling ~40 workers = 80
    # sockets; 200 workers = 400 sockets = a handshake-timeout flood), so K servers each take ~1/K the
    # sockets and the safe worker ceiling scales ~K×. n_servers<=1 (or an externally-managed server,
    # manage_server=False) reduces to the single localhost:8000 path, byte-identical to before.
    _n_servers = max(1, int(n_servers)) if manage_server else 1
    _spawn = max(0, int(spawn_rooms or 0))
    if _spawn and manage_server:
        # W2 step 2: the server-side spawner must be in the clone BEFORE the servers start (the
        # launcher's `node build` at start compiles it). Idempotent copy; a failure falls back loudly.
        try:
            from v_dance.selfplay.spawn_plugin import install_rlspawn
            _ins = install_rlspawn()
            log.info("rlspawn plugin %s", "installed" if _ins["changed"] else "current")
        except Exception:
            log.error("rlspawn plugin install FAILED — spawn path may not work", exc_info=True)
    pool = R.ServerPool(_n_servers, manage=manage_server).start_all()
    _multi_server = _n_servers > 1
    pairs: List[tuple] = []
    source_counts: Counter = Counter()
    # uid is assigned up front (sequentially, race-free) so concurrent pairings get
    # collision-free account names / replay paths.
    descriptors = []
    uid = 0
    # #13 scaling: a single-team MIRROR (and any pool with fewer pairings than workers) is subdivided into
    # up-to-n_workers sub-chunks so --workers actually saturates. Each sub-chunk still gets its own uid → its
    # own accounts/seeds, so a split is just an independent seed-stream — transparent to the aggregated A/B.
    for team_a, team_b, n in _subdivide_matchups(
            team_matchups(team_pool, n_games, seed=matchup_seed),
            min(max(1, n_workers), n_games)):
        uid += 1
        descriptors.append((team_a, team_b, n, uid))

    async def _run_chunk(team_a, team_b, n, uid):
        ta = R.load_team(R.resolve_team_path(team_a))
        tb = R.load_team(R.resolve_team_path(team_b))
        # #22f: bind BOTH seats of this pairing to the SAME pool server (round-robin by uid) — they battle
        # each other, so they must share a server. Empty dict on the single-server path → poke-env's
        # localhost:8000 default (byte-identical to before).
        _srv = ({"server_configuration": R.localhost_server_config(pool.port_for_worker(uid))}
                if _multi_server else {})
        # W2 step 2: under the spawner a pair holds `spawn_rooms` battles at once — the poke-env
        # count queue must sit above that (spawn_client.max_concurrent_for); 1 on the challenge path.
        _mcb = SC.max_concurrent_for(_spawn) if _spawn else 1
        p1 = SelfPlayVGCPlayer(
            actor_critic, tau=p1_tau, sample_seed=seed + uid, top_p=p1_top_p,
            replay_path=_REPO_ROOT / "artifacts" / "replay_buffer" / f"SP1_{uid}.jsonl",
            account_configuration=AccountConfiguration(f"SP1x{uid}", None),
            battle_format=R.BATTLE_FORMAT, team=ta, max_concurrent_battles=_mcb,
            log_level=_logging.WARNING, use_match_belief=p1_belief, use_search=p1_search,
            live_dir=live_dir, save_replays=save_replays, **_srv)
        p2 = SelfPlayVGCPlayer(
            actor_critic, tau=p2_tau, sample_seed=seed + 100000 + uid, top_p=p2_top_p,
            replay_path=_REPO_ROOT / "artifacts" / "replay_buffer" / f"SP2_{uid}.jsonl",
            account_configuration=AccountConfiguration(f"SP2x{uid}", None),
            battle_format=R.BATTLE_FORMAT, team=tb, max_concurrent_battles=_mcb,
            log_level=_logging.WARNING, use_match_belief=p2_belief, use_search=p2_search,
            live_dir=live_dir, save_replays=save_replays, **_srv)
        try:
            if _spawn:
                _st = await SC.play_pairing_spawned(
                    p1, p2, n, fmt=R.BATTLE_FORMAT,
                    cfg=SC.SpawnConfig(rooms=_spawn, battle_timeout=battle_timeout),
                    label=f"self-play spawn uid{uid}")
                source_counts["spawn_pairs"] += int(_st.pairs)
                source_counts["spawn_decisions"] += int(_st.decisions)
                source_counts["spawn_elapsed_s"] += int(round(_st.elapsed_s))
                source_counts["spawn_stalls"] += int(_st.stalls_forfeited)
                source_counts["spawn_ghosts"] += int(_st.ghosts_rescued)
                source_counts["spawn_abandoned"] += int(_st.abandoned)
            else:
                await PB.play_pairing(p1, p2, n, battle_timeout=battle_timeout, label="self-play")
        except Exception:
            log.warning("self-play chunk failed — continuing.", exc_info=True)
        finally:
            source_counts.update(getattr(p1, "_source_counts", {}) or {})
            source_counts.update(getattr(p2, "_source_counts", {}) or {})
            for _p in (p1, p2):                          # A3 belief-feed diagnostics → belief_* keys
                for _k, _v in (getattr(_p, "_belief_stats", None) or {}).items():
                    source_counts["belief_" + _k] += int(_v or 0)
            for _p in (p1, p2):                          # B1 search diagnostics → search_* keys
                for _k, _v in (getattr(_p, "_search_counts", None) or {}).items():
                    source_counts["search_" + _k] += int(_v or 0)
            pairs.extend(pair_trajectories(p1.finished_trajectories(),
                                           p2.finished_trajectories()))
            await PB.close_players(p1, p2)

    try:
        await PB.run_jobs([lambda d=d: _run_chunk(*d) for d in descriptors],
                          workers=n_workers)
    finally:
        pool.stop_all()                                   # #22f: tree-kill every managed pool server

    # Piece 3 (Level-A opp-prediction): fill each perspective's opp_a/opp_b_action from
    # its paired opposite perspective (seat-symmetric codec -> direct copy). In-place, so
    # both the returned `pairs` and the store written below carry the targets.
    for a, b in pairs:
        align_opponent_actions(a, b)

    if store_path is not None:
        flat = [t for pair in pairs for t in pair]
        write_trajectories(store_path, flat)
        log.info("wrote %d trajectories (%d games) -> %s", len(flat), len(pairs), store_path)
    return pairs, dict(source_counts)


def _wilson_ci(wins: int, n: int, z: float = 1.96):
    """(lo, hi) 95% Wilson score interval for a win-rate, or (None, None) for n==0."""
    if not n:
        return None, None
    import math as _m
    p = wins / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * _m.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def main(argv=None) -> int:
    import argparse
    import asyncio
    import json

    ap = argparse.ArgumentParser(description="Self-play Phase-0 plumbing smoke + A3 belief A/B (3c.1b / 3a.6)")
    ap.add_argument("--games", "-n", type=int, default=200, help="self-play games to collect")
    ap.add_argument("--match-belief", choices=["off", "both", "ab"], default="off",
                    help="A3 within-game belief: off | both (Phase-0 live-wiring smoke, belief on both "
                         "seats) | ab (strength A/B — belief ON on p1 only; run with ONE team for a clean "
                         "MIRROR so p1's win-rate isolates the belief effect)")
    ap.add_argument("--search", choices=["off", "both", "ab"], default="off",
                    help="B1 belief-weighted expectimax search: off | both (smoke — search on both seats) | "
                         "ab (strength A/B — search ON on p1 only vs the policy-argmax net on p2; ONE team = "
                         "clean MIRROR so p1's win-rate isolates the search effect)")
    ap.add_argument("--sample", choices=["off", "both", "ab"], default="off",
                    help="TIER-4 serve-sampling: off | both (smoke — both seats sample at --serve-tau) | "
                         "ab (strength A/B — p1 samples the masked policy softmax at --serve-tau/--serve-top-p, "
                         "p2 plays pure argmax, SAME net; ONE team = clean MIRROR so p1's win-rate isolates "
                         "the mixing cost/benefit). Simultaneous-move games make deterministic play "
                         "exploitable by construction — this measures what the mixing costs in head-to-head.")
    ap.add_argument("--serve-tau", type=float, default=0.45,
                    help="serve-sampling temperature for --sample modes (softmax over legal actions)")
    ap.add_argument("--serve-top-p", type=float, default=1.0,
                    help="top-p nucleus restriction for --sample modes (1.0 = no restriction)")
    ap.add_argument("--report-json", default=None, metavar="PATH",
                    help="also write the full report dict (incl. the A/B verdict + belief diagnostics) "
                         "as JSON to PATH, so a run is self-describing without scraping the terminal")
    ap.add_argument("--live-dir", default=None,
                    help="write per-battle spectate/replay files here (enables save_replays)")
    # 2026-09-03: defaults follow the SERVED stack — the old defaults were four Reg M-A teams and the
    # gen141 checkpoint (layout v17), which the v19 encoder refuses (USER hit it launching the W2 block).
    ap.add_argument("--teams", nargs="+", default=None,
                    help="team names from the served format's Champions pool (default: The_Big_6 + the "
                         "first three other teams discovered for the format)")
    ap.add_argument("--ckpt", default=None,
                    help="battle net to warm-start the actor-critic from (default: the served incumbent "
                         "checkpoints_attn_era4_2b/battle_base.pt, else model_io's DEFAULT_BC_CHECKPOINT)")
    ap.add_argument("--tau", type=float, default=1.0, help="collection temperature (>0)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--matchup-seed", type=int, default=0)
    ap.add_argument("--store", default=str(_REPO_ROOT / "artifacts" / "replay_buffer" / "selfplay_phase0.jsonl"))
    ap.add_argument("--battle-timeout", type=float, default=90.0)
    ap.add_argument("--workers", type=int, default=1,
                    help="run this many pairings concurrently (task #13; default 1 = sequential)")
    ap.add_argument("--servers", type=int, default=1,
                    help="#22f: shard workers across N local Showdown servers (ports 8000..8000+N-1) so a "
                         "high --workers count doesn't storm one server's startup handshakes (single-server "
                         "ceiling ~40 workers). Ignored with --no-server. Default 1 = single server.")
    ap.add_argument("--no-server", action="store_true")
    ap.add_argument("--spawn-rooms", type=int, default=0,
                    help="W2 throughput (2026-09-03): run each pairing through the SERVER-SIDE spawner "
                         "(rlspawn plugin) keeping this many battles alive between the pair's two accounts "
                         "instead of the challenge protocol. 0 = challenge path (default). The report then "
                         "carries games/min + decisions/min.")
    ap.add_argument("--min-games", type=int, default=200)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)
    if not args.verbose:
        logging.getLogger("poke_env").setLevel(logging.WARNING)
        logging.getLogger("websockets").setLevel(logging.WARNING)

    if args.ckpt is None:
        from v_dance.play.model_io import DEFAULT_BC_CHECKPOINT
        _inc = _REPO_ROOT / "ai_train_scripts" / "BC_model" / "checkpoints_attn_era4_2b" / "battle_base.pt"
        args.ckpt = str(_inc if _inc.is_file() else DEFAULT_BC_CHECKPOINT)
    if not args.teams:
        import v_dance.play.run_local_battle as _R
        _pool = [Path(p).name for p in _R.discover_teams(reg=_R.BATTLE_FORMAT)]
        # the OWN team = .env VD_DEFAULT_TEAM (the served team; The_Big_6_v2 since 2026-09-03), else The_Big_6
        _env_team = None
        try:
            for _line in (_REPO_ROOT / ".env").read_text(encoding="utf-8").splitlines():
                if _line.strip().startswith("VD_DEFAULT_TEAM="):
                    _env_team = _line.split("=", 1)[1].strip()
        except Exception:
            _env_team = None
        _own_name = _env_team if _env_team in _pool else ("The_Big_6" if "The_Big_6" in _pool else None)
        _own = [_own_name] if _own_name else []
        args.teams = (_own + [t for t in _pool if t != _own_name][:3]) if _pool else []
        if not args.teams:
            print(f"[phase0] no teams discovered for {_R.BATTLE_FORMAT} — pass --teams", file=sys.stderr)
            return 2
    print(f"[phase0] ckpt {args.ckpt}\n[phase0] teams {args.teams}")
    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        print(f"[phase0] checkpoint not found: {ckpt}", file=sys.stderr)
        return 2
    # Single-variable protection: a sampling A/B must not overlap another A/B arm.
    if args.sample == "ab" and (args.search != "off" or args.match_belief != "off"):
        print("[phase0] --sample ab must run with --search off and --match-belief off "
              "(one variable per A/B).", file=sys.stderr)
        return 2
    from v_dance.selfplay.actor_critic import ActorCritic
    ac = ActorCritic.from_bc_checkpoint(ckpt)

    # The Type-C store is APPEND-only (training accumulates across generations), so a
    # repeated SMOKE would stack stale data and mislead a whole-file inspection. Start
    # the smoke from a clean file so the store == exactly this run's trajectories.
    store_path = Path(args.store)
    if store_path.exists():
        store_path.unlink()

    try:
        _t_run = time.monotonic()
        pairs, sources = asyncio.run(run_self_play_games(
            ac, team_pool=args.teams, n_games=args.games, store_path=store_path,
            tau=args.tau, seed=args.seed, manage_server=not args.no_server,
            battle_timeout=args.battle_timeout, matchup_seed=args.matchup_seed,
            n_workers=args.workers, match_belief=args.match_belief, search=args.search,
            n_servers=args.servers, live_dir=args.live_dir, save_replays=bool(args.live_dir),
            sample=args.sample, serve_tau=args.serve_tau, serve_top_p=args.serve_top_p,
            spawn_rooms=args.spawn_rooms))
        _elapsed = max(1e-9, time.monotonic() - _t_run)
        _decisions = sum(len(t.transitions) for pair in pairs for t in pair)
        print(f"[phase0] THROUGHPUT ({'spawn x%d' % args.spawn_rooms if args.spawn_rooms else 'challenge path'}, "
              f"workers {args.workers}, servers {args.servers}): {len(pairs)} games in {_elapsed:.0f}s = "
              f"{60.0 * len(pairs) / _elapsed:.1f} games/min, {60.0 * _decisions / _elapsed:.0f} decisions/min"
              + (f"; stalls {sources.get('spawn_stalls', 0)} ghosts {sources.get('spawn_ghosts', 0)} "
                 f"abandoned {sources.get('spawn_abandoned', 0)}" if args.spawn_rooms else ""))
    except KeyboardInterrupt:
        # Ctrl-C now fires promptly (the parallel_battles heartbeat keeps the Windows select() loop awake);
        # run_self_play_games' `finally` has already stopped the Showdown server during unwind. Exit cleanly
        # with the conventional 130, no traceback (audit 2026-07-01 — Ctrl-C previously hung, forcing a
        # terminal kill).
        print("\n[game_runner] interrupted (Ctrl-C) — Showdown server stopped, partial run discarded.",
              file=sys.stderr)
        return 130

    rep = phase0_report(pairs, sources, min_games=args.min_games)
    rep["match_belief"] = args.match_belief
    rep["search"] = args.search
    rep["sample"] = args.sample
    rep["serve_tau"] = args.serve_tau
    rep["serve_top_p"] = args.serve_top_p
    rep["n_servers"] = args.servers
    rep["ckpt"] = str(ckpt)
    rep["teams"] = list(args.teams)
    rep["tau"] = args.tau

    # ── A3 belief A/B verdict ────────────────────────────────────────────────
    # In 'ab' mode p1 = belief-ON, p2 = belief-OFF (same net), so p1's win-rate IS the belief effect.
    # (The standard phase-0 'balance_ok ~50%' check is EXPECTED to read FAIL here — that divergence is
    # the signal, not a problem.) Report it with a 95% Wilson CI so significance is unambiguous.
    if args.match_belief == "ab":
        wr, ng = rep.get("p1_win_rate"), int(rep.get("p1_games") or 0)
        wins = int(round((wr or 0.0) * ng))
        lo, hi = _wilson_ci(wins, ng)
        verdict = ("INCONCLUSIVE (CI spans 50%)" if lo is None or (lo <= 0.5 <= hi)
                   else "belief HELPS (CI > 50%)" if lo > 0.5 else "belief HURTS (CI < 50%)")
        bel = rep.get("belief") or {}
        fired = sum(int(v or 0) for k, v in bel.items() if "used" in k)
        rep["ab"] = {"belief_on_win_rate": wr, "wins": wins, "games": ng,
                     "ci95_low": lo, "ci95_high": hi, "verdict": verdict,
                     "belief_constraints_fired": fired}
        print("== A3 belief A/B (belief-ON p1 vs belief-OFF p2, same net) ==")
        print(f"  belief-ON win-rate : {wr*100:.1f}%  ({wins}/{ng})" if wr is not None
              else "  belief-ON win-rate : n/a (no decisive games)")
        if lo is not None:
            print(f"  95% Wilson CI      : [{lo*100:.1f}%, {hi*100:.1f}%]")
        print(f"  belief fired       : {fired} constraints "
              f"({'OK — A/B is meaningful' if fired else 'ZERO — belief never fired, A/B is NULL; check wiring'})")
        print(f"  VERDICT            : {verdict}")
        print("============================================================")

    # ── B1 search A/B verdict ────────────────────────────────────────────────
    # In 'ab' mode p1 = search-ON, p2 = policy-argmax (same net), so p1's win-rate IS the search effect.
    if args.search == "ab":
        wr, ng = rep.get("p1_win_rate"), int(rep.get("p1_games") or 0)
        wins = int(round((wr or 0.0) * ng))
        lo, hi = _wilson_ci(wins, ng)
        verdict = ("INCONCLUSIVE (CI spans 50%)" if lo is None or (lo <= 0.5 <= hi)
                   else "search HELPS (CI > 50%)" if lo > 0.5 else "search HURTS (CI < 50%)")
        sd = rep.get("search_diag") or {}
        fired = int(sd.get("search", 0) or 0)
        fell = int(sd.get("fallback", 0) or 0)
        rep["search_ab"] = {"search_on_win_rate": wr, "wins": wins, "games": ng,
                            "ci95_low": lo, "ci95_high": hi, "verdict": verdict,
                            "search_decisions": fired, "fallbacks": fell}
        print("== B1 search A/B (search-ON p1 vs policy-argmax p2, same net) ==")
        print(f"  search-ON win-rate : {wr*100:.1f}%  ({wins}/{ng})" if wr is not None
              else "  search-ON win-rate : n/a (no decisive games)")
        if lo is not None:
            print(f"  95% Wilson CI      : [{lo*100:.1f}%, {hi*100:.1f}%]")
        print(f"  search decisions   : {fired} (fallbacks {fell}) "
              f"({'OK — search drove turns' if fired else 'ZERO — search never ran, A/B is NULL; check wiring'})")
        print(f"  VERDICT            : {verdict}")
        print("============================================================")

    # ── TIER-4 serve-sampling A/B verdict ────────────────────────────────────
    # In 'ab' mode p1 = τ-sampled serve policy, p2 = pure argmax (same net), so
    # p1's win-rate IS the head-to-head cost/benefit of serve-time mixing.  Note
    # the asymmetric interpretation: mixing is expected to trade a LITTLE
    # head-to-head strength for anti-exploitability vs a REPEATED human opponent
    # (which this mirror cannot measure) — so 'no significant loss' is already a
    # green light, and a CI entirely below ~45% is the real red flag.
    if args.sample == "ab":
        wr, ng = rep.get("p1_win_rate"), int(rep.get("p1_games") or 0)
        wins = int(round((wr or 0.0) * ng))
        lo, hi = _wilson_ci(wins, ng)
        verdict = ("INCONCLUSIVE (CI spans 50%)" if lo is None or (lo <= 0.5 <= hi)
                   else "sampling WINS head-to-head (CI > 50%)" if lo > 0.5
                   else "sampling COSTS strength (CI < 50%)")
        rep["sample_ab"] = {"sampling_win_rate": wr, "wins": wins, "games": ng,
                            "ci95_low": lo, "ci95_high": hi, "verdict": verdict,
                            "serve_tau": args.serve_tau, "serve_top_p": args.serve_top_p}
        print(f"== serve-sampling A/B (τ={args.serve_tau} top-p={args.serve_top_p} p1 "
              f"vs argmax p2, same net) ==")
        print(f"  sampling win-rate  : {wr*100:.1f}%  ({wins}/{ng})" if wr is not None
              else "  sampling win-rate  : n/a (no decisive games)")
        if lo is not None:
            print(f"  95% Wilson CI      : [{lo*100:.1f}%, {hi*100:.1f}%]")
        print(f"  VERDICT            : {verdict}")
        print("  (mixing that holds ~50% here is a GREEN light — its value is "
              "anti-exploitability vs repeated opponents, which a mirror can't show)")
        print("============================================================")

    print_phase0_report(rep)

    if args.report_json:
        try:
            Path(args.report_json).write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
            print(f"[phase0] wrote report JSON -> {args.report_json}")
        except Exception as exc:
            print(f"[phase0] could not write report JSON: {exc}", file=sys.stderr)

    return 0 if rep["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())

