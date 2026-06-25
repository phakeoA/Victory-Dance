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
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

# ── Path bootstrap (local_battle FIRST — mirrors player.py) ───────────────────
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
import torch

from v_dance.selfplay import policy_eval
from v_dance.selfplay.collector import TrajectoryCollector, assert_zero_sum
from v_dance.selfplay.reward import place_terminal_reward
from v_dance.selfplay.schema import PASS_ACTION, Transition, Trajectory
from v_dance.selfplay.store import assert_terminal_rewards_clean, write_trajectories

log = logging.getLogger(__name__)

# Sources that mean the MODEL drove the decision (everything else is a fallback /
# mask-desync escape, counted but not model-driven). Mirrors gauntlet.print_sources.
MODEL_DRIVEN_SOURCES = ("model", "forced_switch_model")


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
    with torch.no_grad():
        lp, _ent, vpm = policy_eval.evaluate_actions(actor_critic, [tmp], tau=tau)
    collector.add_step(state=tmp.state, action_s0=ra0, action_s1=ra1,
                       gimmick_s0=int(g0), gimmick_s1=int(g1),
                       logprob=float(lp[0]), value=float(vpm[0]),
                       mask_s0=m0, mask_s1=m1, gmask_s0=gm0, gmask_s1=gm1,
                       decision_type=decision_type, turn=int(turn))
    return ra0, ra1


def terminal_type_for(won: Optional[bool], lost: Optional[bool]) -> str:
    """Map a finished battle's win/lost flags to a TerminalType value. Neither set =>
    a true draw/tie (reward 0, no bootstrap). (A FALLBACK forfeit is detected by the
    runner from the source counts, 3c.1b — not here.)"""
    if won:
        return "win"
    if lost:
        return "loss"
    return "draw"


def finalize_trajectory(collector: TrajectoryCollector, *, won: Optional[bool],
                        terminal_type: str, own_team: Sequence[str], n_turns: int,
                        tp_bring: Optional[Sequence[int]] = None,
                        tp_leads: Optional[Sequence[int]] = None,
                        opp_team: Sequence[str] = ()) -> Trajectory:
    """Finalise a collector into a Trajectory and place the terminal reward. ``n_turns``
    is the battle's real final Showdown turn (NOT the step count) so both perspectives
    share the clock even if one skipped a non-model turn (zero-sum stays valid)."""
    bring = list(tp_bring) if tp_bring is not None else list(range(min(4, len(own_team))))
    leads = list(tp_leads) if tp_leads is not None else bring[:2]
    traj = collector.finish(own_team=list(own_team), opp_team=list(opp_team),
                            tp_bring=bring, tp_leads=leads, won=won,
                            terminal_type=terminal_type, n_turns=int(n_turns))
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

_MODEL_SOURCES = ("model", "forced_switch_model")


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
                    opp_team=tp.get("opp_team") or ())
        except Exception:
            log.debug("self-play finalize failed (non-fatal)", exc_info=True)
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

    sc = {k: v for k, v in source_counts.items() if not str(k).startswith("tp_")}
    resamples = sc.get("rejected_resample", 0)
    # turns where a NON-model order executed (default / forced-switch escape / a random
    # retry perturbation). model-driven = executed model steps / (those + non-model).
    non_model = sc.get("retry_default", 0) + sc.get("forced_switch_escape", 0) + sc.get("retry", 0)
    md = (n_steps / (n_steps + non_model)) if (n_steps + non_model) else 0.0
    resample_rate = (resamples / (n_steps + resamples)) if (n_steps + resamples) else 0.0

    trajs = [t for pair in pairs for t in pair]
    try:
        assert_terminal_rewards_clean(trajs)
        clean, clean_err = True, None
    except AssertionError as e:
        clean, clean_err = False, str(e)

    p1_wr = (p1_wins / p1_games) if p1_games else None
    enough = n >= min_games
    md_ok = md >= target_model_driven
    sym_ok = sym_fail == 0
    bal_ok = (p1_wr is not None and abs(p1_wr - 0.5) <= balance_tol)
    data_clean = dup_steps == 0
    rep = {
        "n_games": n, "enough_games": enough, "min_games": min_games,
        "model_steps": n_steps, "duplicate_steps": dup_steps, "data_clean": data_clean,
        "model_driven": md, "model_driven_ok": md_ok, "non_model_executed": non_model,
        "resample_count": resamples, "resample_rate": resample_rate,
        "symmetry_failures": sym_fail, "symmetry_ok": sym_ok,
        "p1_win_rate": p1_wr, "p1_games": p1_games, "balance_ok": bal_ok,
        "terminal_clean": clean, "terminal_error": clean_err,
    }
    rep["PASS"] = bool(enough and md_ok and sym_ok and bal_ok and clean and data_clean)
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
    print(f"  zero-sum symmetry : {rep['symmetry_failures']} failures  "
          f"[{mark(rep['symmetry_ok'])}]")
    wr = rep["p1_win_rate"]
    wr_s = f"{wr*100:.1f}%" if wr is not None else "n/a"
    print(f"  p1 win-rate       : {wr_s} over {rep['p1_games']} decisive  "
          f"(want ~50%)  [{mark(rep['balance_ok'])}]")
    print(f"  terminal space    : {'clean' if rep['terminal_clean'] else rep['terminal_error']}  "
          f"[{mark(rep['terminal_clean'])}]")
    print("  ----------------------------------------------------------")
    print(f"  PHASE 0           : {'PASS - plumbing clean, ready for Phase 1' if rep['PASS'] else 'FAIL - investigate above'}")
    print("============================================================")


# ── live orchestration (reuses run_local_battle; exercised by the smoke) ──────
async def run_self_play_games(
    actor_critic, team_pool: Sequence[str], n_games: int, *,
    store_path=None, tau: float = 1.0, seed: int = 0, manage_server: bool = True,
    battle_timeout: Optional[float] = 90.0, matchup_seed: int = 0, n_workers: int = 1,
):
    """Play ``n_games`` self-play battles (the shared policy on BOTH sides) over a
    side-balanced team rotation, collect both perspectives of each game, and write the
    Type-C store. Returns ``(pairs, source_counts)``. ``n_workers`` (task #13) runs that
    many pairings concurrently via the shared runner (default 1 = sequential, as before)."""
    import logging as _logging
    from collections import Counter

    import v_dance.play.run_local_battle as R
    from poke_env import AccountConfiguration
    from v_dance.play import parallel_battles as PB
    from v_dance.eval.gauntlet import team_matchups

    server = R.start_showdown() if manage_server else None
    pairs: List[tuple] = []
    source_counts: Counter = Counter()
    # uid is assigned up front (sequentially, race-free) so concurrent pairings get
    # collision-free account names / replay paths.
    descriptors = []
    uid = 0
    for team_a, team_b, n in team_matchups(team_pool, n_games, seed=matchup_seed):
        uid += 1
        descriptors.append((team_a, team_b, n, uid))

    async def _run_chunk(team_a, team_b, n, uid):
        ta = R.load_team(R.resolve_team_path(team_a))
        tb = R.load_team(R.resolve_team_path(team_b))
        p1 = SelfPlayVGCPlayer(
            actor_critic, tau=tau, sample_seed=seed + uid,
            replay_path=_REPO_ROOT / "artifacts" / "replay_buffer" / f"SP1_{uid}.jsonl",
            account_configuration=AccountConfiguration(f"SP1x{uid}", None),
            battle_format=R.BATTLE_FORMAT, team=ta, max_concurrent_battles=1,
            log_level=_logging.WARNING)
        p2 = SelfPlayVGCPlayer(
            actor_critic, tau=tau, sample_seed=seed + 100000 + uid,
            replay_path=_REPO_ROOT / "artifacts" / "replay_buffer" / f"SP2_{uid}.jsonl",
            account_configuration=AccountConfiguration(f"SP2x{uid}", None),
            battle_format=R.BATTLE_FORMAT, team=tb, max_concurrent_battles=1,
            log_level=_logging.WARNING)
        try:
            await PB.play_pairing(p1, p2, n, battle_timeout=battle_timeout, label="self-play")
        except Exception:
            log.warning("self-play chunk failed — continuing.", exc_info=True)
        finally:
            source_counts.update(getattr(p1, "_source_counts", {}) or {})
            source_counts.update(getattr(p2, "_source_counts", {}) or {})
            pairs.extend(pair_trajectories(p1.finished_trajectories(),
                                           p2.finished_trajectories()))
            await PB.close_players(p1, p2)

    try:
        await PB.run_jobs([lambda d=d: _run_chunk(*d) for d in descriptors],
                          workers=n_workers)
    finally:
        if server is not None:
            R.stop_showdown(server)

    if store_path is not None:
        flat = [t for pair in pairs for t in pair]
        write_trajectories(store_path, flat)
        log.info("wrote %d trajectories (%d games) -> %s", len(flat), len(pairs), store_path)
    return pairs, dict(source_counts)


def main(argv=None) -> int:
    import argparse
    import asyncio

    ap = argparse.ArgumentParser(description="Self-play Phase-0 plumbing smoke (3c.1b / 3a.6)")
    ap.add_argument("--games", "-n", type=int, default=200, help="self-play games to collect")
    ap.add_argument("--teams", nargs="+",
                    default=["team1", "WolfeGlick", "Kronomono1", "Kronomono3"])
    ap.add_argument("--ckpt", default=str(_REPO_ROOT / "ai_train_scripts" / "BC_model"
                                          / "checkpoints_attn" / "battle_selfplay_gen141.pt"))
    ap.add_argument("--tau", type=float, default=1.0, help="collection temperature (>0)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--matchup-seed", type=int, default=0)
    ap.add_argument("--store", default=str(_REPO_ROOT / "artifacts" / "replay_buffer" / "selfplay_phase0.jsonl"))
    ap.add_argument("--battle-timeout", type=float, default=90.0)
    ap.add_argument("--workers", type=int, default=1,
                    help="run this many pairings concurrently (task #13; default 1 = sequential)")
    ap.add_argument("--no-server", action="store_true")
    ap.add_argument("--min-games", type=int, default=200)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)
    if not args.verbose:
        logging.getLogger("poke_env").setLevel(logging.WARNING)
        logging.getLogger("websockets").setLevel(logging.WARNING)

    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        print(f"[phase0] checkpoint not found: {ckpt}", file=sys.stderr)
        return 2
    from v_dance.selfplay.actor_critic import ActorCritic
    ac = ActorCritic.from_bc_checkpoint(ckpt)

    # The Type-C store is APPEND-only (training accumulates across generations), so a
    # repeated SMOKE would stack stale data and mislead a whole-file inspection. Start
    # the smoke from a clean file so the store == exactly this run's trajectories.
    store_path = Path(args.store)
    if store_path.exists():
        store_path.unlink()

    pairs, sources = asyncio.run(run_self_play_games(
        ac, team_pool=args.teams, n_games=args.games, store_path=store_path,
        tau=args.tau, seed=args.seed, manage_server=not args.no_server,
        battle_timeout=args.battle_timeout, matchup_seed=args.matchup_seed,
        n_workers=args.workers))

    rep = phase0_report(pairs, sources, min_games=args.min_games)
    print_phase0_report(rep)
    return 0 if rep["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())

