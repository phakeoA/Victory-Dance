"""Generation loop — collect -> update -> gauntlet eval -> promotion gate -> admit
(task 3c.3).

One GENERATION = collect a self-play batch (league opponents) -> a PPO update (critic
warm-up on the first) -> evaluate the candidate on the gauntlet (>=4 teams, side-balanced)
-> a STATISTICAL promotion gate (docs/ppo_reward_design.md sec 16: admit only when the
candidate beats the current best by MORE than noise) -> on pass: league.admit + best
pointer; on a significant regression: revert (collapse recovery, optimisers reset too).

The orchestration (``run_generation``) takes the live steps as INJECTED callables
(``collect_fn`` / ``eval_fn`` / ``save_fn`` / ``restore_fn``), so the
WHOLE loop — gate, promotion, league admission, history, Elo curve — is unit-tested
offline with fakes, while the real wiring (``collect_with_league`` over the validated
runner, ``gauntlet_eval`` over ``gauntlet.run_gauntlet``) is exercised by the live smoke.

The promotion gate is a two-proportion test on the scripted-ladder win-rate: promote when
the lower bound of the win-rate-delta CI clears ``min_delta`` (so noise can't promote),
revert when the upper bound is below ``-min_delta`` (a real regression), else hold.
"""
from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
from v_dance.selfplay.league import OpponentLeague
# The GATE + history bookkeeping live in gate.py (pure, offline-tested). Re-exported here so
# existing `from ...generation import promotion_gate / GenerationHistory / GateConfig / ...`
# imports keep working unchanged after the split (sec 16).
from v_dance.selfplay.gate import (  # noqa: F401,E402
    SCRIPTED_OPPONENTS, _two_prop_se, wilson_lower_bound, wilson_upper_bound,
    GateConfig, promotion_gate, is_plateau, GateConfigV2, promotion_gate_v2,
    HoFConfig, cluster_hof_suspects, hall_of_fame_gate,
    aggregate_scripted, aggregate_prev_best, operator_alert,
    GenerationRecord, GenerationHistory, GenConfig,
)
# Phase-2 HoF LIVE orchestration (pure gate logic stays in gate.py); re-exported so existing
# `from ...generation import hof_eval` imports (test_hof_eval, scratch/_hof_smoke) keep working.
from v_dance.selfplay.hof import hof_eval, apply_hof_gate  # noqa: F401,E402

log = logging.getLogger(__name__)

# Controlled, side-balanced EVAL pool (sec 15): the gauntlet gate judges on THESE so a
# generation's measured strength isn't confounded by the random training team draw. This is
# a USER-CURATED archetype-spanning set (one per axis: rain, a mega core, Perish/balance,
# etc.) so the gate rewards archetype robustness, not just generic-team win-rate — kept to 6
# so FULL both-orientation coverage is cheap (6 teams => 30 pairs; --eval-battles auto-sizes
# to 2x that = 60/opp, ~180 eval games/gen). Training still sees all 71 teams.
DEFAULT_EVAL_TEAMS = (
    "Kronomono3", "Jonah_The_Juggernut", "Rain_Paradise", "Hojun_Y",
    "Justified_Mega_Gallade", "WolfeGlick",
)


# ── generation orchestration ──────────────────────────────────────────────────
def run_generation(
    actor_critic, trainer, league: OpponentLeague, history: GenerationHistory, *,
    collect_fn: Callable, eval_fn: Callable, save_fn: Callable,
    restore_fn: Optional[Callable] = None, cleanup_fn: Optional[Callable] = None,
    hof_eval_fn: Optional[Callable] = None,
    status=None, cfg: GenConfig = GenConfig(),
) -> dict:
    """Run ONE generation. Injected live steps:
      * ``collect_fn(actor_critic, league, gen) -> (trajectories, source_counts)``
      * ``save_fn(actor_critic, gen) -> candidate_path``
      * ``eval_fn(candidate_path, prev_best_path) -> (results{opp:(wins,n)}, model_elo)``
        (``prev_best_path`` is None at gen 0; when given, the eval adds a ``"prev_best"``
        head-to-head so the gate has the non-saturating best-self anchor)
      * ``restore_fn(actor_critic, best_path)`` (on a collapse REVERT)
    The KL-to-BC anchor stays STATIC at gen-0 BC (preserves rare tactics, docs sec 12); the
    old ``refresh_phi_fn`` promote-hook is removed — it was never wired into the live run and
    moving the anchor would weaken that preservation. Returns a report; appends to ``history``."""
    gen = history.generation

    trajectories, sources = collect_fn(actor_critic, league, gen)
    if status is not None:
        status.phase("updating", generation=gen)
    if gen == 0 and cfg.warmup_updates > 0 and trajectories:
        trainer.warmup_critic(trajectories, cfg.warmup_updates)
    update_stats = trainer.ppo_update(trajectories) if trajectories else {"halted": False}
    if status is not None:
        status.set_update(update_stats)

    candidate = save_fn(actor_critic, gen)
    if status is not None:
        status.phase("evaluating", generation=gen)
    # The v2 frozen-champion gate's PRIMARY signal is the head-to-head vs the FROZEN champion
    # (sec 16), so the mirror runs whenever there's a champion to beat (None at gen 0).
    # --no-prev-best skips it → the gate can then only hold/collapse (a 'freeze past gen 0' mode).
    pb_path = history.best_path if cfg.gate.use_prev_best else None
    results, elo = eval_fn(candidate, pb_path)
    sw, sg = aggregate_scripted(results)
    pbw, pbg = aggregate_prev_best(results)
    mirror_rate = (pbw / pbg) if pbg else None
    have_champion = history.best_path is not None

    # Record this gen's observed mirror win-rate BEFORE the gate so the plateau detector's
    # window INCLUDES the current generation (advance_champion / revert reset it below).
    history.record_h2h(mirror_rate)

    verdict, gate_stats = promotion_gate_v2(
        scripted_wins=sw, scripted_games=sg, high_water=history.scripted_high_water,
        mirror_wins=pbw, mirror_games=pbg, h2h_history=history.h2h_history,
        have_champion=have_champion, cfg=cfg.gate_v2)
    # Phase 2 (P2.4): the HoF breadth veto runs ONLY on a PROMOTE — require the candidate to also
    # not-lose to its past champions (the lineage-cycle guard). A REJECT downgrades promote->hold
    # (reason 'hof_reject'); the candidate is STILL admitted below as a non-champion league opponent.
    # No-op when the HoF is disabled / no runner is injected (offline tests + --dry-run).
    verdict, hof_reason, hof_stats = apply_hof_gate(
        verdict, gate_stats.get("reason"), candidate_path=candidate, league=league,
        history=history, cfg=cfg.hof, hof_eval_fn=hof_eval_fn)
    promoted = verdict == "promote"
    if status is not None:
        status.set_update(update_stats, last_verdict=verdict)
    # Admission is DECOUPLED from champion promotion (sec 16): admit EVERY COMPETENT gen as a
    # frozen league opponent (not just promotes) so PFSP diversity keeps growing while the
    # champion holds — but NOT a collapsed/reverted policy (gate on competence). Champions are
    # tagged so the diversity-aware eviction never drops them.
    if verdict != "revert":
        league.admit(f"gen{gen}", candidate, gen, elo if elo is not None else 1000.0,
                     is_champion=promoted)
    if promoted:
        league.latest_path = candidate
        league.reset_pfsp()                     # latest changed → reset stale PFSP weighting
        # raise Wilson high-water; reset h2h; step the champion-lineage Elo by the mirror gain
        history.advance_champion(candidate, (sw, sg), mirror_rate=mirror_rate,
                                 base_elo=elo if elo is not None else 1000.0)
    elif verdict == "revert" and restore_fn is not None and history.best_path is not None:
        restore_fn(actor_critic, history.best_path)   # collapse recovery (optimisers reset too)
        league.reset_pfsp()
        history.h2h_history = []                 # abandon the climb (restored policy == champion)

    # Bound the league (between gens — safe vs in-flight chunk id lookups); cleanup_fn (live
    # only) deletes the evicted checkpoint files (champions are kept, so the rollback target
    # and the manifest/resume references survive).
    evicted = league.prune(cfg.league_cap, cfg.keep_recent)
    if cleanup_fn is not None and evicted:
        cleanup_fn(evicted)

    rec = GenerationRecord(generation=gen, n_trajectories=len(trajectories),
                           scripted_wins=sw, scripted_games=sg, model_elo=elo,
                           verdict=verdict, promoted=promoted, update_stats=update_stats,
                           champion_elo=history.champion_elo, hof=hof_stats)
    history.add(rec)
    return {"generation": gen, "verdict": verdict, "promoted": promoted,
            "reason": hof_reason,
            "scripted_win_rate": (sw / sg) if sg else None, "model_elo": elo,
            "champion_elo": history.champion_elo, "mirror_win_rate": mirror_rate,
            "league_size": len(league.snapshots), "gate": gate_stats, "hof": hof_stats,
            "n_trajectories": len(trajectories), "update_stats": update_stats}


def print_generation_report(rep: dict) -> None:
    wr = rep["scripted_win_rate"]
    wr_s = f"{wr*100:.1f}%" if wr is not None else "n/a"
    elo_s = f"{rep['model_elo']:.0f}" if rep["model_elo"] is not None else "n/a"
    mwr = rep.get("mirror_win_rate")
    mwr_s = f" | h2h {mwr*100:.0f}%" if mwr is not None else ""
    celo = rep.get("champion_elo")
    celo_s = f" champElo {celo:.0f}" if celo is not None else ""
    reason = rep.get("reason")
    reason_s = f" ({reason})" if reason and reason != "hold" else ""
    print(f"  gen {rep['generation']:>2} | {rep['n_trajectories']:>4} trajs | "
          f"scripted {wr_s:>6}{mwr_s} | Elo {elo_s:>5}{celo_s} | {rep['verdict'].upper():7s}{reason_s} | "
          f"league={rep['league_size']}")


# ── live wiring (reuses the validated runner + gauntlet; USER runs the smoke) ──
def build_collection_chunks(league, team_pool, n_games, *, chunk_size, matchup_seed, seed):
    """Plan collection as a FLAT list of independent chunk descriptors (3c.8c) — one per
    (team-matchup, league-opponent) battle group, each with a unique ``uid`` for
    collision-free account names. Built sequentially (league sampling + uid assignment are
    race-free here) so the chunks can then run with BOUNDED CONCURRENCY. With a large team
    pool ``team_matchups`` yields one battle per matchup, so each chunk is ~1 battle and the
    parallelism is ACROSS pairings (the real lever) — NOT within a single pairing. Pure (no
    poke-env)."""
    import numpy as np
    from v_dance.eval.gauntlet import team_matchups
    rng = np.random.default_rng(seed)
    chunks = []
    uid = 0
    for team_a, team_b, n in team_matchups(team_pool, n_games, seed=matchup_seed):
        remaining = n
        while remaining > 0:
            cn = min(chunk_size, remaining)
            remaining -= cn
            uid += 1
            chunks.append({"team_a": team_a, "team_b": team_b, "cn": cn,
                           "spec": league.sample(rng), "uid": uid})
    return chunks


async def collect_with_league(actor_critic, league: OpponentLeague, n_games: int, *,
                              team_pool, tau: float = 1.0, seed: int = 0,
                              matchup_seed: int = 0, chunk_size: int = 10,
                              n_workers: int = 1, stop_check=None,
                              battle_timeout: Optional[float] = 90.0, team_chooser=None,
                              status=None, live_dir=None, save_replays: bool = False,
                              name_salt: str = ""):
    """Collect ``n_games`` self-play games against LEAGUE-sampled opponents (assumes the
    Showdown server is already up — the caller manages it). Each chunk draws one opponent:
      * latest    -> SelfPlayVGCPlayer(ac) on both seats; collect BOTH trajectories (both
                     on-policy, both trained);
      * snapshot  -> our recorder vs a FROZEN checkpoint player; collect OUR trajectory only
                     and record the latest-vs-snapshot outcome for PFSP;
      * scripted  -> our recorder vs a gauntlet anchor; collect OUR trajectory only.
    ``n_workers`` (3c.8c / task #13) bounds how many distinct (matchup, opponent) chunks run
    CONCURRENTLY via the shared runner (``play.parallel_battles.run_jobs``) — the dominant
    throughput lever (sec 20): collection is latency/Node-bound, so overlapping pairings hide
    each other's websocket round-trips. ``stop_check`` drains the queued chunks on a soft stop.
    Returns ``(trajectories, source_counts)`` — a flat list ready for the trainer."""
    import logging as _logging
    from collections import Counter

    import v_dance.play.run_local_battle as R
    from poke_env import AccountConfiguration
    from v_dance.play import parallel_battles as PB
    from v_dance.eval.gauntlet import _make_opponent
    from v_dance.selfplay.game_runner import SelfPlayVGCPlayer

    chunks = build_collection_chunks(league, team_pool, n_games, chunk_size=chunk_size,
                                     matchup_seed=matchup_seed, seed=seed)
    trajectories: list = []
    source_counts: Counter = Counter()
    prog = {"games": 0, "won": 0, "decided": 0}    # live progress (3c.6e-3)

    def _sp(team, who, uid, cn, name, live=None):
        return SelfPlayVGCPlayer(
            actor_critic, tau=tau, sample_seed=seed + (who * 10_000) + uid, live_dir=live,
            save_replays=save_replays,
            replay_path=_REPO_ROOT / "artifacts" / "replay_buffer" / f"{name}.jsonl",
            account_configuration=AccountConfiguration(name, None),
            battle_format=R.BATTLE_FORMAT, team=team, max_concurrent_battles=max(1, cn),
            log_level=_logging.WARNING)

    async def _run_chunk(d):
        # Bounded concurrency + the soft-stop drain are owned by the shared runner (run_jobs)
        # below; distinct (matchup, opponent) battles overlap so each one's websocket round-trips
        # hide the others' — collection is latency/Node-bound, not compute-bound (the dominant
        # lever, sec 20). The shared-state mutations in `finally` run synchronously (no await
        # between them), so they're atomic under asyncio interleaving.
        ta = R.load_team(R.resolve_team_path(d["team_a"]))
        tb = R.load_team(R.resolve_team_path(d["team_b"]))
        cn, spec, uid = d["cn"], d["spec"], d["uid"]
        kind = spec[0]
        # 22d: gen-keyed account names (name_salt = str(gen)) so a stale server-side challenge
        # from one gen can't collide with the next gen's reuse of the same uid.
        opp_ref = spec[1] if kind == "scripted" else None
        our_name, opp_name = PB.collect_account_names(kind, uid, salt=name_salt, opp_ref=opp_ref)
        our = _sp(ta, 0, uid, cn, our_name, live=live_dir)   # #18: our recorder publishes the room+log
        if kind == "latest":
            opp = _sp(tb, 1, uid, cn, opp_name)              # opp shares the same battle_tag; one writer is enough
        elif kind == "snapshot":
            opp = R.make_player(opp_name, tb, model_path=spec[1].path,
                                team_chooser_path=team_chooser, max_concurrent_battles=max(1, cn))
        else:   # scripted
            opp = _make_opponent(spec[1], opp_name, tb,
                                 max_concurrent_battles=max(1, cn))
        try:
            # play_pairing owns the per-chunk hung-battle watchdog; the broad except keeps a
            # non-timeout chunk failure (a build / desync error) from aborting the whole batch.
            await PB.play_pairing(our, opp, cn, battle_timeout=battle_timeout, label=f"vs {kind}")
        except Exception:
            log.warning("league collect chunk failed (vs %s) — continuing.", kind, exc_info=True)
        finally:
            source_counts.update(getattr(our, "_source_counts", {}) or {})
            our_trajs = our.finished_trajectories()
            trajectories.extend(our_trajs.values())
            prog["games"] += cn
            for _t in our_trajs.values():
                if _t.meta.won is not None:
                    prog["decided"] += 1
                    prog["won"] += 1 if _t.meta.won else 0
            if status is not None:
                status.games(prog["games"], (prog["won"] / prog["decided"]) if prog["decided"] else None)
            if kind == "latest":
                source_counts.update(getattr(opp, "_source_counts", {}) or {})
                trajectories.extend(opp.finished_trajectories().values())
            elif kind == "snapshot":
                for traj in our_trajs.values():
                    if traj.meta.won is not None:
                        league.record_result(spec[1].snapshot_id, bool(traj.meta.won))
            await PB.close_players(our, opp)

    # task #13: bounded-concurrency across pairings + soft-stop drain via the shared runner.
    await PB.run_jobs([lambda d=d: _run_chunk(d) for d in chunks],
                      workers=n_workers, stop_check=stop_check)
    return trajectories, dict(source_counts)


def gauntlet_eval(candidate_path, *, teams, team_chooser, battles: int = 30,
                  matchup_seed: int = 0, battle_timeout: Optional[float] = 90.0,
                  manage_server: bool = False, n_workers: int = 1,
                  prev_best_path: Optional[str] = None,
                  mirror_battles: Optional[int] = None, stop_check=None,
                  live_dir=None, save_replays: bool = False, name_salt: str = ""):
    """Evaluate a saved checkpoint on the scripted gauntlet (>=4 teams, side-balanced).
    Returns ``(results{opp:(wins,n)}, model_elo)``. ``n_workers`` (3c.8c) parallelises the
    eval the same way as collection so the promotion gate isn't the throughput bottleneck.

    When ``prev_best_path`` is given (the current accepted best), a ``"prev_best"`` mirror
    is added to the ladder so the gate gets a NON-SATURATING best-self anchor (the gauntlet
    already supports it via ``prev_best_ckpt``).  ``model_elo`` excludes the mirror, so the
    Elo stays scripted-calibrated; ``aggregate_prev_best`` reads the head-to-head."""
    import asyncio

    import v_dance.eval.gauntlet as GA
    import v_dance.play.model_io as model_io
    # Fail LOUD if the candidate won't load — otherwise each gauntlet player silently
    # falls back to a no-model picker and the promotion gate evaluates garbage (3c.3b bug).
    model_io.load_bc_policy(candidate_path)
    opponents = list(SCRIPTED_OPPONENTS)
    if prev_best_path:
        opponents.append("prev_best")
    results, _sources = asyncio.run(GA.run_gauntlet(
        opponents=opponents, team_pool=list(teams),
        battles_per_opponent=battles, ckpt=Path(candidate_path),
        prev_best_ckpt=Path(prev_best_path) if prev_best_path else None,
        team_chooser=Path(team_chooser), manage_server=manage_server,
        matchup_seed=matchup_seed, battle_timeout=battle_timeout, n_workers=n_workers,
        mirror_battles=mirror_battles, stop_check=stop_check,
        live_dir=live_dir, save_replays=save_replays, name_salt=name_salt))
    return results, GA.model_elo(results)


def build_train_configs(*, kl_coef: float = 0.5, target_kl_bc: Optional[float] = 0.15,
                        tau: float = 1.0, min_ev: Optional[float] = None,
                        target_kl_relax_per_gen: float = 0.0,
                        target_kl_max: Optional[float] = None):
    """Build the (PPOConfig, TrainConfig) for a live run with BC-prior PRESERVATION ON
    (task 3c.7a / docs sec 12 exploration-seeding item 1):

      * ``kl_coef > 0``  -> the ``+kl_coef·KL(BC||new)`` penalty keeps PPO from crushing
        the warm-started BC prior early (the rare-tactic clicks the human data already has).
      * ``target_kl_bc`` -> EARLY-HALT a generation whose policy drifts too far from BC
        (collapse guard, sec 2/10); a non-positive value disables the guard.
      * ``min_ev``       -> early-halt if the critic's explained variance collapses
        (value-surface guard); ``None`` disables it.
      * ``tau``          -> forced IDENTICAL into ``PPOConfig.tau`` so the PPO log-prob
        recompute matches the collection temperature (was a latent mismatch: ``--tau``
        reached collection but not the loss).

    Imported lazily so this module stays importable (``--dry-run``) without torch."""
    from v_dance.selfplay.ppo import PPOConfig
    from v_dance.selfplay.trainer import TrainConfig
    tkl = target_kl_bc if (target_kl_bc is not None and target_kl_bc > 0) else None
    ppo = PPOConfig(kl_coef=float(kl_coef), tau=float(tau))
    train = TrainConfig(target_kl_from_bc=tkl, min_explained_variance=min_ev,
                        target_kl_relax_per_gen=float(target_kl_relax_per_gen),
                        target_kl_max=target_kl_max)
    return ppo, train


def resolve_train_pool(spec):
    """Expand the TRAINING-team spec into a concrete pool (task 3c.7b). ``["all"]`` (the
    default) or any spec containing ``"all"`` => EVERY team under ``teams/Champions/``
    (the archetype-rich draw, sec 15 — auto-includes M-B teams as they're added);
    otherwise the given names/paths are used verbatim. Explicit specs need no torch /
    poke-env import (so the expansion is unit-testable offline)."""
    spec = list(spec) if spec else ["all"]
    if any(str(s).lower() == "all" for s in spec):
        import v_dance.play.run_local_battle as R   # lazy: pulls poke_env
        return R.discover_teams()
    return spec


def tau_for_generation(gen: int, tau_start: float, tau_end: float,
                       anneal_gens: int) -> float:
    """Collection temperature for generation ``gen`` (task 3c.7c, exploration seeding sec 12):
    anneal LINEARLY from ``tau_start`` (gen 0, more stochastic => broader exploration) to
    ``tau_end`` (held from gen ``anneal_gens`` on, the calibrated baseline => exploitation).
    ``anneal_gens <= 0`` (or equal endpoints) => constant ``tau_end`` (no anneal). The KL-to-BC
    anchor (3c.7a) makes the early high-tau exploration safe. **Whatever this returns is used
    for BOTH collection AND that generation's PPO log-prob recompute** (the 3c.7a tau-match
    invariant) — the caller drives both from this single value."""
    if anneal_gens <= 0 or tau_start == tau_end:
        return float(tau_end)
    frac = min(1.0, max(0.0, gen / float(anneal_gens)))     # 0 at gen0 -> 1 at anneal_gens
    return float(tau_start + (tau_end - tau_start) * frac)


def target_kl_for_generation(gen: int, base: Optional[float], relax_per_gen: float = 0.0,
                             cap: Optional[float] = None) -> Optional[float]:
    """KL-to-BC early-halt threshold for generation ``gen`` (sec 12). The KL anchor stays
    STATIC at gen-0 BC, so as a genuinely-stronger champion drifts further from the human
    prior its KL-to-BC grows; a FIXED threshold would eventually halt good generations. This
    relaxes the bar LINEARLY: ``base + relax_per_gen*gen`` (optionally capped). The default
    ``relax_per_gen=0`` returns ``base`` unchanged — no behaviour change unless opted into.
    ``base is None`` (guard off) stays None. The relaxed bar still rises SLOWLY, so a sudden
    collapse spike in KL-to-BC stays above it and still halts."""
    if base is None:
        return None
    t = float(base) + max(0.0, float(relax_per_gen)) * max(0, int(gen))
    return min(t, float(cap)) if cap is not None else t


def resolve_eval_battles(eval_battles, n_eval_teams: int) -> int:
    """Auto-size the gauntlet eval budget when unset (task 3c.7b). ``None`` => 2x the ordered
    team-pairs, i.e. FULL both-orientation side-balanced coverage with each matchup sampled
    ~twice (so the promotion-gate read isn't matchup-luck; sec 15/16). An explicit int passes
    through unchanged (use a small one, e.g. 12, for fast UI smokes)."""
    if eval_battles is not None:
        return int(eval_battles)
    pairs = max(1, n_eval_teams * (n_eval_teams - 1))
    return 2 * pairs


def server_recycle_index(done: int, restart_server_every: int, n_servers: int):
    """Which pool-server INDEX to recycle at gen boundary ``done`` — or ``None`` to recycle nothing
    this gen (22f staggered anti-bloat). Refresh ONE server every ``restart_server_every // K`` gens,
    round-robin, so each is recycled every ``restart_server_every`` gens but they never all go down
    together (which would stall the whole run). ``K=1`` reduces to "recycle the single server every
    ``restart_server_every`` gens"; ``restart_server_every=0`` disables recycling. Pure → unit-tested."""
    if not restart_server_every or done <= 0:
        return None
    k = max(1, int(n_servers))
    stride = max(1, int(restart_server_every) // k)
    if done % stride != 0:
        return None
    return (done // stride - 1) % k


def run_live_generations(ckpt, *, n_generations=None, team_pool, team_chooser,
                         archive_dir, eval_team_pool=None,
                         gen_cfg: GenConfig = GenConfig(), ppo_cfg=None,
                         train_cfg=None, tau_start: float = 1.3, tau_end: float = 1.0,
                         tau_anneal_gens: int = 12, seed: int = 0,
                         eval_battles=None, mirror_battles: int = 360, manage_server: bool = True,
                         device: str = "cpu", n_workers: int = 1, collect_workers=None,
                         collect_procs: int = 1, collect_async_per_proc: int = 3,
                         resume_from=None, resume_gen=None, keep_snapshots: int = 25,
                         save_replays: bool = False, keep_replay_buffers: int = 200,
                         restart_server_every: int = 20, n_servers: int = 1,
                         snapshot_path=None, max_hours=None) -> dict:
    """Run real generations end-to-end (collect via the league -> PPO update -> gauntlet
    eval -> promotion gate -> admit/refresh/revert), RESUMABLY (3c.4 / #20): a PER-GENERATION
    snapshot (``snap_gen{N}.pt`` in ``archive/sub_checkpoints/``) is written after every generation, so a later run
    can ``resume_gen`` from ANY kept generation (an int N continues at N+1; ``"latest"`` = the
    newest; ``None`` = fresh from ``--ckpt``). ``resume_from`` is the legacy explicit-file path.
    ``keep_snapshots`` bounds how many per-gen snapshots are retained (0 = all). The run stops
    cleanly on Ctrl-C or after ``max_hours``; ``n_generations=None`` => run until stopped. The
    Showdown server is started ONCE and reused across collect + eval.

    ``team_pool`` is the TRAINING pool (sec 15: draw both sides from the full archetype-rich
    set for pilot/counter exposure); ``eval_team_pool`` (default = ``team_pool``) is the
    CONTROLLED, side-balanced set the gauntlet gate judges on — so a generation's measured
    strength isn't 'got lucky with the team draw' (sec 15)."""
    import asyncio
    import time as _time

    import v_dance.play.run_local_battle as R
    from v_dance.selfplay.actor_critic import ActorCritic
    from v_dance.selfplay.trainer import PPOTrainer
    from v_dance.selfplay import mp_collect as MP
    from v_dance.selfplay import resume as RS
    from v_dance.selfplay.status import LiveStatus

    archive = Path(archive_dir)
    archive.mkdir(parents=True, exist_ok=True)
    # #18b spectate: a per-RUN folder live/<start-stamp>/gen_<N>/{replays,eval}. In-flight battles
    # write there; on finish they're SAVED (save_replays) or deleted. Each run = a fresh folder, so
    # no clearing needed; the dashboard recursively globs live/ for the currently-live battles.
    from v_dance.selfplay.status import run_stamp as _run_stamp, gen_kind_dir as _gen_kind_dir
    live_run_dir = archive / "live" / _run_stamp()
    print(f"   spectate: live/{live_run_dir.name}/gen_*/<replays|eval>  "
          f"({'SAVING .html replays' if save_replays else 'live-only (deleted on finish)'})")
    _eval_pool = list(eval_team_pool) if eval_team_pool else list(team_pool)
    _eval_pairs = len(_eval_pool) * (len(_eval_pool) - 1)
    eval_battles = resolve_eval_battles(eval_battles, len(_eval_pool))
    # 3c.8c: collection is latency-bound (light CPU) so it can run MORE concurrent battles
    # than the cap-derived `n_workers`; eval is heavier (fast scripted battles) so it stays at
    # n_workers. collect_workers defaults to n_workers (no change unless the user opts in).
    cw = max(1, int(collect_workers)) if collect_workers else n_workers
    # 14b.3: collect_procs > 1 → TRUE multiprocessing collection (bypasses the GIL on the per-turn
    # model inference; probe 14a ~5x). The pool object is cheap — workers spawn lazily on the first
    # submit, after the server is up — and is torn down in the finally. procs=1 keeps the asyncio
    # path verbatim (zero behaviour change).
    _mp = bool(collect_procs and int(collect_procs) > 1)
    pool = MP.CollectionPool(int(collect_procs)) if _mp else None
    if _mp:
        MP.sweep_mp_ckpts(archive)   # reclaim any per-gen worker ckpts orphaned by a prior crash
        print(f"   collection: MULTIPROCESS {collect_procs} procs x {collect_async_per_proc} "
              f"battles each (~{int(collect_procs) * int(collect_async_per_proc)} concurrent, "
              f"~{round(int(collect_procs) * 0.6, 1)}GB RAM)")
    else:
        print(f"   workers: collect={cw}  eval={n_workers}")
    print(f"   teams: TRAIN pool={len(team_pool)} (archetype draw, sec 15)  "
          f"EVAL pool={len(_eval_pool)} ({_eval_pairs} pairs)  "
          f"eval-battles={eval_battles}/opp (~{3 * eval_battles} games/gen)")
    if 0 < eval_battles < _eval_pairs:
        print(f"   [eval] {eval_battles} battles/opp < {_eval_pairs} ordered team-pairs -> "
              f"PARTIAL (but seed-fixed, gen-over-gen reproducible) coverage. For full "
              f"both-orientation side-balancing use --eval-battles >= {_eval_pairs}.")

    # Build from the base ckpt (architecture + frozen-BC reference), THEN overlay the
    # resume snapshot's trained state if present (sec 17 — ref/arch re-derived, not stored).
    # 3c.8b: the actor-critic + optimisers live on `device` (cuda => GPU PPO update); collection
    # uses a CPU inference-copy (see collect_fn). Resume maps the snapshot onto `device` too.
    ac = ActorCritic.from_bc_checkpoint(ckpt, device=device)
    trainer = PPOTrainer(ac, ppo_cfg, train_cfg, seed=seed, device=device)
    # Base KL-to-BC early-halt threshold (the relax schedule rises from this each gen; sec 12).
    _base_target_kl = trainer.tcfg.target_kl_from_bc
    league = OpponentLeague(latest_path=str(ckpt))
    history = GenerationHistory()
    try:
        _resume_path = RS.resolve_resume(archive, resume_gen=resume_gen, explicit_path=resume_from)
    except (ValueError, TypeError):
        print(f"[resume] invalid --resume-gen {resume_gen!r} (use an int or 'latest')",
              file=sys.stderr)
        sys.exit(2)
    if _resume_path is not None:
        if not Path(_resume_path).exists():
            avail = [g for g, _ in RS.list_snapshots(archive)]
            print(f"[resume] snapshot not found: {_resume_path}  (available generations: "
                  f"{avail if avail else 'none'})", file=sys.stderr)
            sys.exit(2)
        league, history, _snap = RS.load_into(_resume_path, actor_critic=ac, trainer=trainer,
                                              device=device)
        print(f"[resume] loaded {Path(_resume_path).name} — continuing at generation "
              f"{history.generation} (league={len(league.snapshots)})")
    stop = RS.StopController(max_hours=max_hours)
    status = LiveStatus(archive / "status.json", min_interval=0.5)   # live feed; throttled (3c.8c)
    status.start_run(n_generations, hours=max_hours)

    thru = {"games": 0, "secs": 0.0}     # 3c.8a throughput measurement (sec 20: measure first)

    def collect_fn(ac_, lg, gen):
        # 3c.7c: one tau drives BOTH collection AND this gen's PPO recompute (3c.7a invariant).
        tau_gen = tau_for_generation(gen, tau_start, tau_end, tau_anneal_gens)
        trainer.cfg.tau = tau_gen
        # sec 12: relax this gen's KL-to-BC early-halt bar from its base (static BC anchor's
        # KL grows as a stronger champion drifts; default relax=0 → unchanged).
        trainer.tcfg.target_kl_from_bc = target_kl_for_generation(
            gen, _base_target_kl, trainer.tcfg.target_kl_relax_per_gen, trainer.tcfg.target_kl_max)
        # 3c.8b: collection runs on CPU (sec 20) — use a CPU inference-copy when the trainer
        # lives on the GPU; the copy reflects the latest trained weights (remade each gen).
        status.phase("collecting", generation=gen, games_total=gen_cfg.n_games)
        if tau_anneal_gens > 0 and tau_start != tau_end:
            print(f"   collect: tau={tau_gen:.3f} (gen {gen}; anneal {tau_start}->{tau_end} "
                  f"over {tau_anneal_gens} gens)")
        t0 = _time.perf_counter()
        if pool is not None:
            # 14b.3: multiprocess collection — freeze the current weights to a per-gen ckpt the
            # workers reload, fan the chunks across processes, fold results + apply PFSP in main.
            # STOP SEMANTICS: a stop (Ctrl-C / --hours) is honoured at GEN BOUNDARIES — the current
            # gen's batch runs to completion (bounded by mp throughput, ~tens of seconds), then the
            # loop exits. The finer per-chunk drain is the asyncio path's (stop_check → run_jobs).
            _ckpt = MP.mp_ckpt_path(archive, gen)
            _coll_live = _gen_kind_dir(live_run_dir, gen, "replays")
            try:
                trajs, src = MP.collect_with_pool(
                    ac_, lg, gen_cfg.n_games, team_pool=team_pool, ckpt_path=_ckpt,
                    submit_fn=pool.submit,
                    save_ckpt_fn=lambda a, p, _g=gen: MP.save_inference_ckpt(a, p, generation=_g),
                    n_procs=int(collect_procs), async_per_proc=int(collect_async_per_proc),
                    tau=tau_gen, seed=seed + gen * 1000, matchup_seed=gen,
                    team_chooser=team_chooser, status=status, live_dir=_coll_live,
                    save_replays=save_replays, generation=gen,   # 22d: gen-keyed account names
                    ports=_pool_ports)                           # 22f: spread workers across the pool
            finally:
                # ALWAYS drop the per-gen ckpt — even if collection raised (Ctrl-C lands inside the
                # blocking submit) — else a full-weight file orphans each crashed gen (review fix).
                try:
                    Path(_ckpt).unlink(missing_ok=True)
                except Exception:
                    pass
        else:
            coll_ac = ac_.inference_copy("cpu") if device != "cpu" else ac_
            trajs, src = asyncio.run(collect_with_league(
                coll_ac, lg, gen_cfg.n_games, team_pool=team_pool, tau=tau_gen,
                seed=seed + gen * 1000, matchup_seed=gen, team_chooser=team_chooser,
                n_workers=cw, stop_check=stop.should_stop, status=status,
                live_dir=_gen_kind_dir(live_run_dir, gen, "replays"), save_replays=save_replays,
                name_salt=str(gen)))   # 22d: gen-keyed account names
        dt = _time.perf_counter() - t0
        thru["games"] += gen_cfg.n_games
        thru["secs"] += dt
        gpm = gen_cfg.n_games / dt * 60.0 if dt > 0 else float("nan")
        avg = thru["games"] / thru["secs"] * 60.0 if thru["secs"] > 0 else float("nan")
        # sec 20 throughput readout: games/min this gen + running avg (sizes generations to
        # wall-clock; the measurement that gates the 3c.8b/c optimisations).
        print(f"   throughput: {gen_cfg.n_games} games in {dt:.1f}s = {gpm:.1f} games/min "
              f"(avg {avg:.1f}/min, {len(trajs)} trajs)")
        return trajs, src

    def save_fn(ac_, gen):
        # Per-gen policy checkpoints live in <archive>/checkpoints/ (tidy archive; the full-state
        # resume snapshots go in <archive>/sub_checkpoints/). The league/gate store + reload THIS
        # returned path, so moving the dir needs no other change.
        ckpt_dir = archive / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        p = ckpt_dir / f"gen{gen}.pt"
        ac_.save(p, generation=gen)
        return str(p)

    def eval_fn(path, prev_best_path=None):
        t0 = _time.perf_counter()
        # The eval matchup seed varies per generation (keyed to the gen) so the fixed 6-team pool
        # isn't a static target a champion-exploiter overfits — deterministic/resumable, averaging
        # the fixed-pool matchup luck out across gens (red-team finding).
        _eval_live = _gen_kind_dir(live_run_dir, history.generation, "eval")   # #18b: eval spectate dir
        if pool is not None:
            # #19: multiprocess EVAL on the SAME pool — fan the gauntlet descriptors across the
            # collection worker processes (the GIL win applied to eval). model_elo stays in main.
            from v_dance.selfplay import mp_eval as ME
            import v_dance.eval.gauntlet as GA
            opps = list(SCRIPTED_OPPONENTS) + (["prev_best"] if prev_best_path else [])
            results, _src = ME.eval_with_pool(
                path, opponents=opps, team_pool=_eval_pool, battles_per_opponent=eval_battles,
                team_chooser=team_chooser, submit_fn=pool.submit, prev_best=prev_best_path,
                mirror_battles=mirror_battles, matchup_seed=seed + history.generation,
                n_procs=int(collect_procs), async_per_proc=int(collect_async_per_proc),
                live_dir=_eval_live, save_replays=save_replays,
                generation=history.generation,   # 22d: gen-keyed account names
                ports=_pool_ports)               # 22f: spread eval workers across the pool
            out = (results, GA.model_elo(results))
            _eval_label = f"{int(collect_procs)} procs"
        else:
            out = gauntlet_eval(path, teams=_eval_pool, team_chooser=team_chooser,
                                battles=eval_battles, manage_server=False, n_workers=n_workers,
                                prev_best_path=prev_best_path, mirror_battles=mirror_battles,
                                matchup_seed=seed + history.generation, stop_check=stop.should_stop,
                                live_dir=_eval_live, save_replays=save_replays,
                                name_salt=str(history.generation))   # 22d: gen-keyed account names
            _eval_label = f"{n_workers} workers"
        dt = _time.perf_counter() - t0
        n_eval = sum(g for _w, g in out[0].values())     # total finished eval battles
        gpm = n_eval / dt * 60.0 if dt > 0 else float("nan")
        pb = out[0].get("prev_best")
        pb_s = f" | prev_best {pb[0]}/{pb[1]}" if pb else ""
        print(f"   eval throughput: {n_eval} games in {dt:.1f}s = {gpm:.1f} games/min "
              f"({_eval_label}){pb_s}")
        return out

    def hof_run(candidate_path, suspects):
        # Phase-2 HoF: play the candidate vs its past-champion suspects (server already up). Sync
        # wrapper (like collect_fn/eval_fn) so run_generation stays sync + offline-injectable.
        if not suspects:
            return []
        t0 = _time.perf_counter()
        out = asyncio.run(hof_eval(
            candidate_path, suspects, team_pool=_eval_pool, team_chooser=team_chooser,
            games_per_snapshot=gen_cfg.hof.games_per_snapshot, n_workers=n_workers,
            manage_server=False, matchup_seed=seed + history.generation,
            # task E: HoF past-champion battles spectate + save to eval/league/ too
            live_dir=_gen_kind_dir(live_run_dir, history.generation, "eval"),
            save_replays=save_replays))
        dt = _time.perf_counter() - t0
        n = sum(g for _i, _w, g in out)
        print(f"   HoF eval: {len(out)} past champions, {n} games in {dt:.1f}s")
        return out

    def restore_fn(ac_, path):
        ac_.restore_from(path)        # reload champion policy + critic (collapse recovery)
        trainer.reset_optimizers()    # clear the stale Adam moments that drove the collapse,
                                      # so the next update doesn't re-shove the restored policy

    def cleanup_fn(evicted):
        # Delete the checkpoint files of league snapshots that eviction dropped (sec 16) so the
        # archive doesn't grow without bound. Only OUR archived gen files, never a champion (those
        # aren't in `evicted`) or the base ckpt — and only after prune + before the resume save,
        # so nothing still references them.
        # only delete OUR archived gen files — the archive root (legacy flat layout) or the new
        # checkpoints/ sub-dir — never a champion, the base ckpt, or anything outside the archive.
        allowed = {archive.resolve(), (archive / "checkpoints").resolve()}
        for s in evicted:
            try:
                fp = Path(s.path)
                if fp.exists() and fp.parent.resolve() in allowed:
                    fp.unlink()
            except Exception:
                log.debug("evicted-snapshot file cleanup failed (non-fatal)", exc_info=True)

    def _save():
        # Per-gen snapshot named by the LAST COMPLETED generation (history.generation-1), so a
        # later run can --resume-gen any kept generation (task #20). Nothing to save until a gen
        # has completed (an interrupted gen 0 just restarts fresh). Prune to bound disk (~21MB ea).
        completed = history.generation - 1
        if completed < 0:
            return
        RS.save_snapshot(RS.snapshot_path_for(archive, completed), actor_critic=ac, trainer=trainer,
                         league=league, history=history, ppo_cfg=trainer.cfg, train_cfg=trainer.tcfg,
                         gen_cfg=gen_cfg, seed=seed)
        RS.prune_snapshots(archive, keep_snapshots)

    # 22f: a POOL of K Showdown servers (ports 8000..8000+K-1). Spreading the worker processes
    # across them keeps any single server from saturating under concurrency (#22) or bloating over a
    # long run (#22c) — each sees ~1/K the battles. K=1 (default) is the single shared server,
    # byte-identical to before. An externally-managed server (manage_server=False) is always single.
    _n_servers = max(1, int(n_servers)) if manage_server else 1
    server_pool = R.ServerPool(_n_servers, manage=manage_server).start_all()
    _pool_ports = server_pool.ports if _n_servers > 1 else None   # None => poke-env's 8000 default
    reports = []
    done = 0
    # Cap the BC-era diagnostic replay buffer (one uid-keyed .jsonl per battle player, never read by
    # the RL training) so a long run can't fill the disk. Pruned at run start (clears prior bloat) +
    # each gen boundary (no battle is mid-write then). 0 = keep all.
    from v_dance.play.vgc_base import prune_replay_buffer as _prune_rb
    _rb_dir = _REPO_ROOT / "artifacts" / "replay_buffer"
    _prune_rb(_rb_dir, keep_replay_buffers)
    try:
        while (n_generations is None or done < n_generations) and not stop.should_stop():
            _prune_rb(_rb_dir, keep_replay_buffers)        # trim the previous gen's traces
            # #22c/22f anti-bloat: a Node Showdown server leaks over a long run (accumulated battle
            # rooms → slow handshakes, stale challenges, eval STALLS by ~gen 50 even at low
            # concurrency). Recycle servers at GEN BOUNDARIES (no battle in flight → the next gen's
            # players just reconnect to a fresh one). With a POOL we STAGGER (one server per recycle
            # event, round-robin) so they never all go down at once; K=1 reduces to the old behaviour.
            _idx = server_recycle_index(done, restart_server_every, _n_servers) if manage_server else None
            if _idx is not None:
                _port = server_pool.ports[_idx]
                print(f"   [server] recycling pool server :{_port} after {done} gens "
                      f"({_n_servers} server(s), staggered anti-bloat)")
                server_pool.recycle(_port)
            rep = run_generation(ac, trainer, league, history, collect_fn=collect_fn,
                                 eval_fn=eval_fn, save_fn=save_fn, restore_fn=restore_fn,
                                 cleanup_fn=cleanup_fn,
                                 hof_eval_fn=(hof_run if gen_cfg.hof.enabled else None),
                                 status=status, cfg=gen_cfg)
            print_generation_report(rep)
            _hof = rep.get("hof")                   # Phase-2 HoF breadth-veto readout (P2.4)
            if _hof and _hof.get("reason") != "thin_pool_skip":
                susp = "  ".join(f"{r['snapshot_id']}:{r['wins']}/{r['games']}"
                                 f"{'*' if r['vetoed'] else ''}" for r in _hof.get("suspects", []))
                tag = "OVERRIDDEN" if _hof.get("overridden") else str(_hof.get("reason", "")).upper()
                wu = _hof.get("worst_upper")
                print(f"        HoF[{tag}] vs past champions: {susp}  worst={_hof.get('worst_snapshot_id')}"
                      + (f" upper={wu:.2f}" if wu is not None else ""))
            elif _hof:
                print(f"        HoF: skipped ({_hof.get('eligible', 0)} past champions < min_pool "
                      f"{gen_cfg.hof.min_pool}) — too few promotions to cycle yet")
            _alert = operator_alert(history)        # unattended-run watchdog (sec 16)
            if _alert:
                print(f"        ⚠ {_alert}")
            us = rep.get("update_stats", {}) or {}
            print(f"        update: loss={us.get('loss', float('nan')):+.4f} "
                  f"kl_to_bc={us.get('kl_to_bc', float('nan')):.3e} "
                  f"EV={us.get('explained_variance', float('nan')):+.3f} "
                  f"clip_frac={us.get('clip_fraction', float('nan')):.2f} "
                  f"halted={us.get('halted')}")
            _save()                          # per-generation heartbeat snapshot
            from v_dance.selfplay import archive as AR   # manifest (self-play) — dashboard data source
            AR.write_generation_artifacts(archive, history, league)
            print("        archive: manifest.json")
            status.phase("idle", generation=rep["generation"])
            reports.append(rep)
            done += 1
    finally:
        status.finish_run()
        _save()                              # final flush (also on Ctrl-C / exception)
        if pool is not None:
            pool.close()                     # tear down the collection worker processes
            MP.sweep_mp_ckpts(archive)       # reclaim any per-gen worker ckpt left by a crashed gen
        server_pool.stop_all()               # 22f: tree-kill every pool server
    _latest = RS.latest_snapshot(archive)
    print(f"\n  ran {done} generation(s); latest snapshot -> "
          f"{_latest.name if _latest else '(none)'}  (resume with --resume-gen latest)")
    print(f"  Elo curve: {[(g, round(e) if e else None) for g, e in history.elo_curve()]}")
    print(f"  league   : {[s.snapshot_id for s in league.snapshots]}")
    return {"history": history, "league": league, "reports": reports,
            "snapshot": str(_latest) if _latest else None}


# ── offline dry-run demo (no server): the loop logic over synthetic generations ─
def _dry_run(n_generations: int = 6, seed: int = 0) -> None:
    """Simulate the generation loop with synthetic eval (rising then noisy then a dip),
    so the gate / promote / hold / revert / league-growth logic is visible without a
    server or the model. No torch / poke-env needed."""
    league = OpponentLeague(latest_path="bc_best.pt")
    history = GenerationHistory()

    # a fake trainer/ac/refresh/restore that just record calls
    class _AC:
        pass
    ac = _AC()

    class _Trainer:
        def warmup_critic(self, trajs, n): return {"value_loss": 0.2}
        def ppo_update(self, trajs): return {"loss": 0.1, "halted": False, "kl_to_bc": 0.01}
    trainer = _Trainer()

    # Synthetic per-gen (scripted, head-to-head-vs-champion) win-rates that drive the v2
    # frozen-champion gate through every verdict: gen0 auto-promote; hold below the 0.55 bar;
    # clear it (beat_champion); a scripted dip → scripted-collapse REVERT; a mirror erosion below
    # the champion → mirror-collapse REVERT (P2.2); then more beat_champion promotes. Mirror runs
    # at 360 games (>= min_h2h_games) so the 0.55 bar and the mirror-collapse revert can fire.
    scripted_wr = [0.55, 0.58, 0.60, 0.60, 0.30, 0.60, 0.60, 0.60][:n_generations]
    mirror_wr = [None, 0.52, 0.60, 0.50, 0.55, 0.38, 0.58, 0.58][:n_generations]
    eval_games, mirror_games = 400, 360
    calls = {"restore": 0}

    def collect_fn(ac, league, gen):
        return [object()] * 200, {"model": 4000}     # 200 fake trajectories

    def save_fn(ac, gen):
        return f"archive/checkpoints/gen{gen}.pt"

    def eval_fn(path, prev_best_path=None):
        gen = len(history.records)
        scr = scripted_wr[min(gen, len(scripted_wr) - 1)]
        wins = round(scr * eval_games)
        results = {"random": (wins // 3, eval_games // 3),
                   "max_damage": (wins // 3, eval_games // 3),
                   "heuristic": (wins - 2 * (wins // 3), eval_games - 2 * (eval_games // 3))}
        mir = mirror_wr[min(gen, len(mirror_wr) - 1)]
        if prev_best_path is not None and mir is not None:        # the mirror vs the champion
            results["prev_best"] = (round(mir * mirror_games), mirror_games)
        return results, 1000 + (scr - 0.5) * 800

    def restore_fn(ac, path): calls["restore"] += 1

    def hof_fn(candidate_path, suspects):
        # Synthetic HoF (P2.4): the candidate LOSES to its OLDEST champion (gen0) but beats the
        # rest — so once >= min_pool past champions exist (late in the run) the breadth veto
        # downgrades that promote to a HOLD (hof_reject), visible offline without a server.
        return [(s.snapshot_id, (18 if s.snapshot_id == "gen0" else 40), 60) for s in suspects]

    print("== Generation-loop dry run (no server) =====================")
    print(f"  synthetic scripted win-rate per gen: {scripted_wr}")
    print(f"  synthetic mirror   win-rate per gen: {mirror_wr}")
    print("  v2 gate: keep the champion frozen until the candidate clears the 0.55 bar vs it OR "
          "the climb plateaus; revert on a scripted collapse OR a mirror collapse (eroded below "
          "its own champion)\n")
    for _ in range(n_generations):
        rep = run_generation(ac, trainer, league, history, collect_fn=collect_fn,
                             eval_fn=eval_fn, save_fn=save_fn, restore_fn=restore_fn,
                             hof_eval_fn=hof_fn, cfg=GenConfig(warmup_updates=3))
        print_generation_report(rep)
        _h = rep.get("hof")
        if _h and _h.get("reason") != "thin_pool_skip":
            susp = "  ".join(f"{r['snapshot_id']}:{r['wins']}/{r['games']}"
                             f"{'*' if r['vetoed'] else ''}" for r in _h.get("suspects", []))
            print(f"        HoF vs past champions: {susp} -> {_h.get('reason')}")
    print(f"\n  reverts (on regression): {calls['restore']}")
    print(f"  league snapshots admitted : {len(league.snapshots)}  "
          f"({[s.snapshot_id for s in league.snapshots]})")
    print(f"  Elo curve                 : "
          f"{[(g, round(e)) for g, e in history.elo_curve()]}")
    print("============================================================")


def _launch_live(args):
    """Apply the resource budget + build the configs and run the live generation loop for a
    parsed ``args`` namespace. Shared by ``--live`` and the ``--wizard`` interactive launcher."""
    import logging as _logging
    _logging.basicConfig(level=_logging.DEBUG if args.verbose else _logging.WARNING)
    if not args.verbose:
        _logging.getLogger("poke_env").setLevel(_logging.WARNING)
        _logging.getLogger("websockets").setLevel(_logging.WARNING)
    if not Path(args.ckpt).exists():
        print(f"[gen] checkpoint not found: {args.ckpt}", file=sys.stderr)
        sys.exit(2)
    n_gen = None if args.generations <= 0 else args.generations   # 0 => run until stopped
    # Resource caps (sec 20 / 3c.8): CPU thread cap + parallel-collection worker count
    # (3c.8c) + the GPU PPO-update device & VRAM cap (3c.8b) — all enforced here.
    from v_dance.selfplay.resources import ResourceBudget, apply_resource_budget, summarize
    try:
        _rb = apply_resource_budget(
            ResourceBudget(max_cpu_fraction=args.max_cpu_fraction, max_cores=args.max_cores,
                           max_vram_gb=args.max_vram_gb, device=args.device),
            set_threads=True, enforce_vram=True)
    except ValueError as e:
        print(f"[gen] resource budget error: {e}", file=sys.stderr)
        sys.exit(2)
    ppo_cfg, train_cfg = build_train_configs(
        kl_coef=args.kl_coef, target_kl_bc=args.target_kl_bc,
        tau=args.tau_start, min_ev=args.min_ev,
        target_kl_relax_per_gen=args.target_kl_relax,
        target_kl_max=args.target_kl_max)         # gen-0 tau; collect_fn re-sets per gen
    # train = full archetype-rich pool; eval = controlled set (sec 15). --teams (deprecated)
    # overrides BOTH with an explicit list.
    train_pool = resolve_train_pool(args.teams if args.teams else args.train_teams)
    eval_pool = list(args.teams) if args.teams else list(args.eval_teams)
    if not train_pool:
        print("[gen] no training teams found under teams/Champions/ — add team files "
              "or pass --train-teams <names>", file=sys.stderr)
        sys.exit(2)
    print(f"== Live generation run: {n_gen if n_gen else 'until-stop'} gen x "
          f"{args.games} games (eval {args.eval_battles if args.eval_battles is not None else 'auto'}/opp"
          f"{f', max {args.hours}h' if args.hours else ''}) ==")
    _tau_desc = (f"tau {args.tau_start}->{args.tau} over {args.tau_anneal_gens} gens"
                 if args.tau_anneal_gens > 0 and args.tau_start != args.tau
                 else f"tau={args.tau} (flat)")
    print(f"   exploration (sec 12): KL-to-BC coef={args.kl_coef} "
          f"target_kl_bc={'off' if args.target_kl_bc <= 0 else args.target_kl_bc} "
          f"min_ev={'off' if args.min_ev is None else args.min_ev} {_tau_desc}")
    _gc = GateConfigV2()
    print(f"   gate (sec 16): frozen-champion ladder — beat_champion >= {_gc.promote_threshold:.2f} over "
          f"{args.mirror_battles} mirror games; mirror-collapse revert < {0.5 - _gc.mirror_collapse_margin:.2f}; "
          f"prev_best mirror = {'ON' if args.prev_best else 'OFF (pure scripted ladder)'}")
    print(f"   HoF (Phase 2): {'ON' if args.hof else 'OFF'} — not-lose to last {args.hof_champions} "
          f"past champions @ {args.hof_games} games each"
          + ("  [--hof-override: rejects logged, NOT blocking]" if args.hof_override else ""))
    _cw = args.collect_workers if args.collect_workers else _rb["workers"]
    print(f"   resources (sec 20): {summarize(_rb)}")
    _upd = "PPO update on GPU (VRAM capped)" if _rb["device"] == "cuda" else "CPU PPO update"
    print(f"   hybrid (3c.8): {_upd}; collection on CPU, {_cw} parallel battles "
          f"(eval {_rb['workers']}).")
    if args.collect_procs and args.collect_procs > 1:
        print(f"   multicore (14b): collection across {args.collect_procs} PROCESSES x "
              f"{args.collect_async} battles each (~{args.collect_procs * args.collect_async} "
              f"concurrent, ~{round(args.collect_procs * 0.6, 1)}GB RAM) — GIL-free per process "
              f"(probe 14a ~5x). [supersedes --collect-workers]")
    run_live_generations(
        Path(args.ckpt), n_generations=n_gen, team_pool=train_pool,
        eval_team_pool=eval_pool,
        team_chooser=args.team_chooser, archive_dir=args.archive,
        gen_cfg=GenConfig(n_games=args.games, warmup_updates=args.warmup,
                          gate=GateConfig(use_prev_best=args.prev_best),
                          hof=HoFConfig(enabled=args.hof, n_champions=args.hof_champions,
                                        games_per_snapshot=args.hof_games, override=args.hof_override)),
        ppo_cfg=ppo_cfg, train_cfg=train_cfg,
        tau_start=args.tau_start, tau_end=args.tau, tau_anneal_gens=args.tau_anneal_gens,
        seed=args.seed, eval_battles=args.eval_battles, mirror_battles=args.mirror_battles,
        device=_rb["device"], n_workers=_rb["workers"], collect_workers=args.collect_workers,
        collect_procs=args.collect_procs, collect_async_per_proc=args.collect_async,
        manage_server=not args.no_server, resume_from=args.resume,
        resume_gen=args.resume_gen, keep_snapshots=args.keep_snapshots,
        save_replays=args.save_replays, keep_replay_buffers=args.keep_replay_buffers,
        restart_server_every=args.restart_server_every, n_servers=args.servers,
        snapshot_path=args.snapshot, max_hours=args.hours)


def _wizard(ap):
    """Interactive launcher so you don't have to memorise the flags: prompts for the key run
    parameters (with examples), echoes the equivalent command for next time, and returns the
    args namespace (everything not prompted keeps its default). Triggered by ``--wizard`` or by
    running the module with NO arguments in a terminal."""
    args = ap.parse_args(["--live"])     # start from all defaults, with live=True

    def _read(prompt):
        try:
            return input(prompt)
        except EOFError:                     # no interactive input -> don't hang / don't auto-launch
            print("\n  (no input available; run with explicit flags, e.g. "
                  "--live --generations 0 --hours 8 --max-vram-gb 4)")
            sys.exit(0)

    def ask(label, default, example=None, cast=str):
        shown = "auto/none" if default is None else default
        ex = f"   (e.g. {example})" if example is not None else ""
        raw = _read(f"  {label} [{shown}]{ex}: ").strip()
        if raw == "":
            return default
        try:
            return cast(raw)
        except Exception:
            print(f"    ! couldn't parse {raw!r}; keeping {shown}")
            return default

    def ask_yn(label, default=False):
        raw = _read(f"  {label} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
        return default if raw == "" else raw in ("y", "yes")

    print("\n=== Victory-Dance self-play launcher ===")
    print("Press Enter to accept the [default]. Overnight = Generations 0 + an Hours cap.\n")
    args.generations = ask("Generations (0 = run until you press Ctrl-C)", 0,
                           "5 for a fixed count, 0 for overnight", int)
    args.hours = ask("Hour cap (stops cleanly between gens)", None,
                     "8 for an 8-hour overnight; blank = no cap", float)
    args.games = ask("Self-play games per generation", 300,
                     "300 normal, 50 for a quick test (sec 16)", int)
    args.eval_battles = ask("Eval battles per scripted opponent", None,
                            "blank = auto full coverage (60/opp); 12 for a fast test", int)
    args.max_cpu_fraction = ask("Max CPU fraction (of physical cores)", 0.5,
                                "0.5 = half your cores", float)
    args.collect_workers = ask("Collection workers (concurrent battles)", None,
                               "12 to use collection's CPU headroom; blank = cap", int)
    args.collect_procs = ask("Collection PROCESSES (multicore; 1 = single-process asyncio)", 1,
                             "4-6 to bypass the GIL (~5x, probe 14a); ~0.6GB RAM each", int)
    args.save_replays = ask_yn("Save spectator replays as playable .html? (under "
                               "live/<stamp>/gen_N/{replays,eval}/<tag>.html; N = deleted on finish)", False)
    args.max_vram_gb = ask("Max VRAM GB for the GPU update", None,
                           "4 (of 8 GB); blank = uncapped", float)
    args.mirror_battles = ask("Mirror battles vs the champion (the 0.55 beat_champion bar)", 360,
                              "360 = the calibrated floor (P2.1); 48 for a fast test", int)
    args.prev_best = ask_yn("Use the prev_best head-to-head promotion bar? "
                            "(promotes past a scripted plateau; N = pure scripted ladder)", True)
    args.hof = ask_yn("Run the Hall-of-Fame breadth check on a promote? (also require beating "
                      "your last few PAST champions, not just the current one — catches cycling)", True)
    if args.hof:
        args.hof_champions = ask("  How many past champions must the candidate beat", 5,
                                 "5 (cap ~8); excludes the current champion the mirror tests", int)
    if ask_yn("Resume the previous run (continue training)?", False):
        from v_dance.selfplay import resume as RS
        avail = [g for g, _ in RS.list_snapshots(args.archive)]
        if avail:
            print(f"    available generations: {avail}")
            args.resume_gen = ask("  From which generation (blank = latest)", "latest",
                                  f"{avail[-1]} = latest; an earlier one rolls back", str)
        else:
            print(f"    ! no per-gen snapshots in {args.archive} — starting FRESH from base BC.")
            args.resume_gen = None
    else:
        args.resume_gen = None
    args.verbose = ask_yn("Verbose (-v) logging?", True)

    parts = ["python -m v_dance.selfplay.generation --live",
             f"--generations {args.generations}", f"--games {args.games}",
             f"--max-cpu-fraction {args.max_cpu_fraction}"]
    if args.hours:
        parts.append(f"--hours {args.hours}")
    if args.eval_battles is not None:
        parts.append(f"--eval-battles {args.eval_battles}")
    if args.mirror_battles != 360:
        parts.append(f"--mirror-battles {args.mirror_battles}")
    if args.collect_workers:
        parts.append(f"--collect-workers {args.collect_workers}")
    if args.collect_procs and args.collect_procs > 1:
        parts.append(f"--collect-procs {args.collect_procs}")
    if args.save_replays:
        parts.append("--save-replays")
    if args.max_vram_gb is not None:
        parts.append(f"--max-vram-gb {args.max_vram_gb}")
    if not args.prev_best:
        parts.append("--no-prev-best")
    if not args.hof:
        parts.append("--no-hof")
    elif args.hof_champions != 5:
        parts.append(f"--hof-champions {args.hof_champions}")
    if args.resume_gen:
        parts.append(f"--resume-gen {args.resume_gen}")
    elif args.resume:
        parts.append(f"--resume {args.resume}")
    if args.verbose:
        parts.append("-v")
    print("\n  Equivalent command (copy this to skip the wizard next time):")
    print("    " + " ".join(parts) + " 2> artifacts/logs/gen.log\n")
    if not ask_yn("Start now?", True):
        print("Aborted - no run started.")
        sys.exit(0)
    print()
    return args


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Generation loop (3c.3)")
    ap.add_argument("--dry-run", action="store_true",
                    help="simulate the loop with synthetic eval (no server / model)")
    ap.add_argument("--live", action="store_true",
                    help="run REAL generations on the local Showdown server")
    ap.add_argument("--wizard", action="store_true",
                    help="interactive launcher — prompts for the run parameters with examples "
                         "(also the default when you run the module with NO arguments in a terminal)")
    ap.add_argument("--generations", type=int, default=6)
    ap.add_argument("--games", type=int, default=100, help="self-play games per generation")
    ap.add_argument("--eval-battles", type=int, default=None,
                    help="gauntlet battles per scripted opponent (split across the eval "
                         "team-pairs). DEFAULT auto-sizes to 2*N*(N-1) = full both-orientation "
                         "side-balanced coverage of the eval pool (6 teams => 60/opp, ~180 "
                         "games/gen). Pass a small int (e.g. 12) for a fast UI smoke.")
    ap.add_argument("--mirror-battles", type=int, default=360,
                    help="head-to-head battles vs the CHAMPION for the v2 gate's 0.55 bar (sec 16). "
                         "Default 360 (>= the gate's min_h2h_games=360 floor; P2.1 gate_sim: the 0.55 "
                         "bar needs ~360 to hold false-promote ~3%% AND the mirror-collapse revert "
                         "needs it). The scripted ladder stays at --eval-battles; only the mirror is bumped.")
    ap.add_argument("--warmup", type=int, default=5, help="critic-only warm-up updates (gen 0)")
    ap.add_argument("--ckpt", default=str(_REPO_ROOT / "ai_train_scripts" / "BC_model"
                                          / "checkpoints" / "bc_best.pt"))
    ap.add_argument("--train-teams", nargs="+", default=["all"],
                    help="TRAINING team pool: 'all' (default) = every team under "
                         "teams/Champions/ (archetype-rich draw, sec 15; auto-picks up "
                         "M-B teams as added); or explicit names/paths")
    ap.add_argument("--eval-teams", nargs="+", default=list(DEFAULT_EVAL_TEAMS),
                    help="controlled, side-balanced EVAL pool the gauntlet gate judges on")
    ap.add_argument("--teams", nargs="+", default=None,
                    help="(deprecated) set BOTH the train and eval pools to this explicit list")
    ap.add_argument("--team-chooser", default=str(_REPO_ROOT / "ai_train_scripts"
                    / "teamPreview_model" / "checkpoints" / "teampreview_best.pt"))
    ap.add_argument("--archive", default=str(_REPO_ROOT / "artifacts" / "self_play_archive"))
    ap.add_argument("--tau", type=float, default=1.0,
                    help="FINAL/baseline collection temperature the anneal settles to (also the "
                         "PPO recompute temperature each gen — kept == collection, 3c.7a)")
    ap.add_argument("--tau-start", type=float, default=1.3,
                    help="INITIAL collection temperature for early exploration (sec 12); anneals "
                         "linearly to --tau over --tau-anneal-gens. Set == --tau to disable.")
    ap.add_argument("--tau-anneal-gens", type=int, default=12,
                    help="generations to anneal tau from --tau-start down to --tau (0 = no anneal)")
    ap.add_argument("--kl-coef", type=float, default=0.5,
                    help="KL-to-BC penalty weight (sec 12: >0 preserves the BC prior so PPO "
                         "can't crush rare tactics; 0 = off)")
    ap.add_argument("--target-kl-bc", type=float, default=0.15,
                    help="early-halt a generation if mean KL(BC||new) exceeds this "
                         "(warm-start-collapse guard); <=0 disables")
    ap.add_argument("--target-kl-relax", type=float, default=0.0,
                    help="relax --target-kl-bc by this much PER GENERATION (sec 12: the static "
                         "BC anchor's KL grows as a stronger champion drifts, so a fixed bar "
                         "would eventually halt good gens; 0 = fixed, the default)")
    ap.add_argument("--target-kl-max", type=float, default=None,
                    help="cap for the relaxed --target-kl-bc threshold (default uncapped)")
    ap.add_argument("--min-ev", type=float, default=None,
                    help="early-halt if critic explained-variance falls below this "
                         "(value-collapse guard; default off)")
    # ── resource budget (sec 20 / 3c.8) ───────────────────────────────────────
    ap.add_argument("--max-cpu-fraction", type=float, default=0.5,
                    help="cap CPU to this fraction of PHYSICAL cores (sec 20): sets torch "
                         "threads now + the parallel-collection worker count (3c.8c). Default 0.5")
    ap.add_argument("--max-cores", type=int, default=None,
                    help="explicit worker/thread count (overrides --max-cpu-fraction)")
    ap.add_argument("--collect-workers", type=int, default=None,
                    help="concurrent COLLECTION battles (3c.8c); defaults to the CPU-cap "
                         "worker count. Collection is latency-bound (~30%% CPU at 6) so you "
                         "can set this HIGHER (e.g. 12) to use the headroom; eval stays capped.")
    ap.add_argument("--collect-procs", type=int, default=1,
                    help="TRUE multiprocessing collection across this many worker PROCESSES "
                         "(14b; default 1 = single-process asyncio). >1 bypasses the GIL on model "
                         "inference (~5x in probe 14a); ~0.6GB RAM each. Try 4-6. Supersedes "
                         "--collect-workers when >1.")
    ap.add_argument("--collect-async", type=int, default=3,
                    help="concurrent battles WITHIN each collection process (14b; default 3); "
                         "total concurrency ~= --collect-procs x --collect-async.")
    ap.add_argument("--max-vram-gb", type=float, default=None,
                    help="GPU memory ceiling for the PPO update (enforced via "
                         "set_per_process_memory_fraction; allocations beyond it fail loudly)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                    help="PPO-update device (auto=GPU if present, else CPU; collection always CPU)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None,
                    help="resume from an EXPLICIT snapshot file (legacy/power-user); prefer "
                         "--resume-gen. Use the SAME --ckpt as the original run.")
    ap.add_argument("--resume-gen", default=None,
                    help="resume from a GENERATION's snapshot in <archive>/sub_checkpoints/ (#20): an "
                         "int N loads snap_gen{N}.pt and continues at N+1; 'latest' = the newest; "
                         "omitted = FRESH from --ckpt BC. (Snapshots are kept per generation.)")
    ap.add_argument("--keep-snapshots", type=int, default=25,
                    help="keep only the last N per-gen snapshots (~21 MB each; default 25 ≈ 525 MB "
                         "of rollback). 0 = keep ALL.")
    ap.add_argument("--keep-replay-buffers", type=int, default=200,
                    help="cap artifacts/replay_buffer/ to its N most-recent per-battle .jsonl traces "
                         "(the BC-era diagnostic log; NOT read by RL training). Pruned at run start + "
                         "each gen boundary so a long run can't fill the disk. Default 200; 0 = keep ALL.")
    ap.add_argument("--restart-server-every", type=int, default=20,
                    help="recycle the local Showdown server every N generations (#22c anti-bloat): it "
                         "leaks memory/rooms over a long run and by ~gen 50 the eval STALLS (slow "
                         "handshakes, stale challenges). Restarting between gens keeps it fast. "
                         "Default 20; 0 = never (only for short runs). With --servers>1 the recycle "
                         "is STAGGERED across the pool (one server per N/K gens, round-robin).")
    ap.add_argument("--servers", type=int, default=1,
                    help="run a POOL of N local Showdown servers on ports 8000..8000+N-1 (22f) and "
                         "spread the collection/eval worker processes across them, so no single "
                         "server saturates under concurrency (#22) or bloats over a long run (#22c) "
                         "— each handles ~1/N the battles. Default 1 (single shared server). A good "
                         "rule: N ~= --collect-procs / 3.")
    ap.add_argument("--save-replays", action="store_true",
                    help="SAVE finished battles as real, playable Showdown .html replays under "
                         "artifacts/.../live/<start-stamp>/gen_<N>/{replays,eval}/<tag>.html "
                         "(open in a browser) instead of deleting the live feed on finish "
                         "(default off = live-only).")
    ap.add_argument("--snapshot", default=None,
                    help="(deprecated; ignored) snapshots are now per-gen snap_gen{N}.pt in <archive>/sub_checkpoints/")
    ap.add_argument("--hours", type=float, default=None,
                    help="stop cleanly after ~this many wall-clock hours (between gens)")
    ap.add_argument("--prev-best", action=argparse.BooleanOptionalAction, default=True,
                    help="use the prev_best head-to-head promotion bar (sec 16: lets a gen "
                         "promote past a scripted plateau by beating the accepted-best mirror). "
                         "--no-prev-best = pure scripted gate (also skips the mirror's eval games)")
    # ── Phase-2 Hall-of-Fame breadth veto (sec 16) ────────────────────────────
    ap.add_argument("--hof", action=argparse.BooleanOptionalAction, default=True,
                    help="Phase-2 HoF breadth veto: on a PROMOTE, ALSO require the candidate to "
                         "not-LOSE to its last --hof-champions PAST champions (catches lineage "
                         "cycling — beats the current champion but loses to an older one). "
                         "--no-hof = promote on the mirror bar alone.")
    ap.add_argument("--hof-champions", type=int, default=HoFConfig().n_champions,
                    help="how many PAST champions the candidate must not-lose to (default 5; "
                         "excludes the current champion the mirror already tests; cap ~8 for FWER)")
    ap.add_argument("--hof-games", type=int, default=HoFConfig().games_per_snapshot,
                    help="battles vs EACH past-champion suspect (default 60; the P2.1 floor)")
    ap.add_argument("--hof-override", action="store_true",
                    help="operator escape: let HoF-rejected promotes THROUGH (still evaluated + "
                         "logged loud) to release a frozen lineage-cycle standoff")
    ap.add_argument("--no-server", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        _dry_run(args.generations, args.seed)
    elif args.wizard:
        _launch_live(_wizard(ap))
    elif args.live:
        _launch_live(args)
    elif len(sys.argv) == 1 and sys.stdin.isatty():
        _launch_live(_wizard(ap))          # bare `python -m …generation` in a terminal -> wizard
    else:
        ap.print_help()
