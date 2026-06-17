"""Generation loop — collect -> update -> gauntlet eval -> promotion gate -> admit
(task 3c.3).

One GENERATION = collect a self-play batch (league opponents) -> a PPO update (critic
warm-up on the first) -> evaluate the candidate on the gauntlet (>=4 teams, side-balanced)
-> a STATISTICAL promotion gate (docs/ppo_reward_design.md sec 16: admit only when the
candidate beats the current best by MORE than noise) -> on pass: league.admit + best
pointer + frozen-Phi refresh; on a significant regression: revert (collapse recovery).

The orchestration (``run_generation``) takes the live steps as INJECTED callables
(``collect_fn`` / ``eval_fn`` / ``save_fn`` / ``refresh_phi_fn`` / ``restore_fn``), so the
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

log = logging.getLogger(__name__)

SCRIPTED_OPPONENTS = ("random", "max_damage", "heuristic")

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


# ── promotion gate (pure stats) ───────────────────────────────────────────────
@dataclass
class GateConfig:
    z: float = 1.0               # one-sided significance band (1.0 ~84%, 1.645 ~95%)
    min_delta: float = 0.0       # require at least this absolute win-rate improvement
    revert_on_regression: bool = True


def _two_prop_se(p1: float, n1: int, p2: float, n2: int) -> float:
    return math.sqrt(p1 * (1 - p1) / max(n1, 1) + p2 * (1 - p2) / max(n2, 1))


def promotion_gate(new_wins: int, new_games: int, base_wins: int, base_games: int,
                   cfg: GateConfig = GateConfig()) -> Tuple[str, dict]:
    """Decide promote / hold / revert from the scripted-ladder win-rates. With no
    baseline (first generation) the candidate is auto-promoted to establish it."""
    if base_games <= 0:
        return "promote", {"reason": "no_baseline", "p_new": (new_wins / new_games)
                           if new_games else None}
    p_new = new_wins / new_games if new_games else 0.0
    p_base = base_wins / base_games
    delta = p_new - p_base
    se = _two_prop_se(p_new, new_games, p_base, base_games)
    lo, hi = delta - cfg.z * se, delta + cfg.z * se
    if lo > cfg.min_delta:
        verdict = "promote"
    elif cfg.revert_on_regression and hi < -cfg.min_delta:
        verdict = "revert"
    else:
        verdict = "hold"
    return verdict, {"p_new": p_new, "p_base": p_base, "delta": delta, "se": se,
                     "ci": (lo, hi), "z": cfg.z}


def aggregate_scripted(results: Dict[str, Tuple[int, int]]) -> Tuple[int, int]:
    """Sum (wins, games) over the scripted-anchor opponents in a gauntlet result dict."""
    w = g = 0
    for name, (wins, n) in results.items():
        if name in SCRIPTED_OPPONENTS:
            w += wins
            g += n
    return w, g


# ── generation history (pure, persistable for 3c.4) ───────────────────────────
@dataclass
class GenerationRecord:
    generation: int
    n_trajectories: int
    scripted_wins: int
    scripted_games: int
    model_elo: Optional[float]
    verdict: str
    promoted: bool
    update_stats: dict = field(default_factory=dict)

    def to_obj(self) -> dict:
        return {"generation": self.generation, "n_trajectories": self.n_trajectories,
                "scripted_wins": self.scripted_wins, "scripted_games": self.scripted_games,
                "model_elo": self.model_elo, "verdict": self.verdict,
                "promoted": self.promoted,
                "update_stats": {k: v for k, v in self.update_stats.items()
                                 if isinstance(v, (int, float))}}

    @classmethod
    def from_obj(cls, d: dict) -> "GenerationRecord":
        return cls(generation=int(d["generation"]), n_trajectories=int(d.get("n_trajectories", 0)),
                   scripted_wins=int(d.get("scripted_wins", 0)),
                   scripted_games=int(d.get("scripted_games", 0)),
                   model_elo=d.get("model_elo"), verdict=d.get("verdict", "hold"),
                   promoted=bool(d.get("promoted", False)),
                   update_stats=d.get("update_stats", {}))


@dataclass
class GenerationHistory:
    records: List[GenerationRecord] = field(default_factory=list)
    best_path: Optional[str] = None
    best_scripted: Tuple[int, int] = (0, 0)   # (wins, games) of the accepted best

    @property
    def generation(self) -> int:
        return len(self.records)

    def add(self, rec: GenerationRecord) -> None:
        self.records.append(rec)

    def elo_curve(self) -> List[Tuple[int, Optional[float]]]:
        return [(r.generation, r.model_elo) for r in self.records]

    def to_obj(self) -> dict:
        return {"records": [r.to_obj() for r in self.records], "best_path": self.best_path,
                "best_scripted": list(self.best_scripted)}

    @classmethod
    def from_obj(cls, d: dict) -> "GenerationHistory":
        return cls(records=[GenerationRecord.from_obj(r) for r in d.get("records", [])],
                   best_path=d.get("best_path"),
                   best_scripted=tuple(d.get("best_scripted", (0, 0))))


# ── generation config + orchestration ─────────────────────────────────────────
@dataclass
class GenConfig:
    n_games: int = 300            # self-play games collected per generation
    warmup_updates: int = 5       # critic-only warm-up updates on the FIRST generation
    gate: GateConfig = field(default_factory=GateConfig)


def run_generation(
    actor_critic, trainer, league: OpponentLeague, history: GenerationHistory, *,
    collect_fn: Callable, eval_fn: Callable, save_fn: Callable,
    refresh_phi_fn: Optional[Callable] = None, restore_fn: Optional[Callable] = None,
    status=None, cfg: GenConfig = GenConfig(),
) -> dict:
    """Run ONE generation. Injected live steps:
      * ``collect_fn(actor_critic, league, gen) -> (trajectories, source_counts)``
      * ``save_fn(actor_critic, gen) -> candidate_path``
      * ``eval_fn(candidate_path) -> (results{opp:(wins,n)}, model_elo)``
      * ``refresh_phi_fn(actor_critic)`` (on promote) / ``restore_fn(actor_critic, best_path)`` (on revert)
    Returns a generation report dict; appends to ``history``; mutates ``league`` on promote."""
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
    results, elo = eval_fn(candidate)
    sw, sg = aggregate_scripted(results)

    verdict, gate_stats = promotion_gate(sw, sg, *history.best_scripted, cfg.gate)
    promoted = verdict == "promote"
    if status is not None:
        status.set_update(update_stats, last_verdict=verdict)
    if promoted:
        league.admit(f"gen{gen}", candidate, gen, elo if elo is not None else 1000.0)
        league.latest_path = candidate
        history.best_path = candidate
        history.best_scripted = (sw, sg)
        if refresh_phi_fn is not None:
            refresh_phi_fn(actor_critic)        # frozen-Phi snapshot refresh (sec 4/6)
    elif verdict == "revert" and restore_fn is not None and history.best_path is not None:
        restore_fn(actor_critic, history.best_path)   # collapse recovery

    rec = GenerationRecord(generation=gen, n_trajectories=len(trajectories),
                           scripted_wins=sw, scripted_games=sg, model_elo=elo,
                           verdict=verdict, promoted=promoted, update_stats=update_stats)
    history.add(rec)
    return {"generation": gen, "verdict": verdict, "promoted": promoted,
            "scripted_win_rate": (sw / sg) if sg else None, "model_elo": elo,
            "league_size": len(league.snapshots), "gate": gate_stats,
            "n_trajectories": len(trajectories), "update_stats": update_stats}


def print_generation_report(rep: dict) -> None:
    wr = rep["scripted_win_rate"]
    wr_s = f"{wr*100:.1f}%" if wr is not None else "n/a"
    elo_s = f"{rep['model_elo']:.0f}" if rep["model_elo"] is not None else "n/a"
    print(f"  gen {rep['generation']:>2} | {rep['n_trajectories']:>4} trajs | "
          f"scripted {wr_s:>6} | Elo {elo_s:>5} | {rep['verdict'].upper():7s} | "
          f"league={rep['league_size']}")


# ── live wiring (reuses the validated runner + gauntlet; USER runs the smoke) ──
async def collect_with_league(actor_critic, league: OpponentLeague, n_games: int, *,
                              team_pool, tau: float = 1.0, seed: int = 0,
                              matchup_seed: int = 0, chunk_size: int = 10,
                              battle_timeout: Optional[float] = 90.0, team_chooser=None,
                              status=None):
    """Collect ``n_games`` self-play games against LEAGUE-sampled opponents (assumes the
    Showdown server is already up — the caller manages it). Each chunk draws one opponent:
      * latest    -> SelfPlayVGCPlayer(ac) on both seats; collect BOTH trajectories (both
                     on-policy, both trained);
      * snapshot  -> our recorder vs a FROZEN checkpoint player; collect OUR trajectory only
                     and record the latest-vs-snapshot outcome for PFSP;
      * scripted  -> our recorder vs a gauntlet anchor; collect OUR trajectory only.
    Returns ``(trajectories, source_counts)`` — a flat list ready for the trainer."""
    import asyncio
    import logging as _logging
    from collections import Counter

    import numpy as np
    import v_dance.play.run_local_battle as R
    from poke_env import AccountConfiguration
    from v_dance.eval.gauntlet import team_matchups, _make_opponent
    from v_dance.selfplay.game_runner import SelfPlayVGCPlayer

    rng = np.random.default_rng(seed)
    trajectories: list = []
    source_counts: Counter = Counter()
    showcase_log = None              # raw |-log of one game, for a Type_D replay (3c.5)
    uid = 0
    games_done = won = decided = 0    # live progress (3c.6e-3)

    def _sp(team, who, status=None):
        nonlocal uid
        uid += 1
        return SelfPlayVGCPlayer(
            actor_critic, tau=tau, sample_seed=seed + (who * 10_000) + uid, status=status,
            replay_path=_REPO_ROOT / "artifacts" / "replay_buffer" / f"LG{who}_{uid}.jsonl",
            account_configuration=AccountConfiguration(f"LG{who}x{uid}", None),
            battle_format=R.BATTLE_FORMAT, team=team, max_concurrent_battles=1,
            log_level=_logging.WARNING)

    for team_a, team_b, n in team_matchups(team_pool, n_games, seed=matchup_seed):
        ta, tb = R.load_team(R.resolve_team_path(team_a)), R.load_team(R.resolve_team_path(team_b))
        remaining = n
        while remaining > 0:
            cn = min(chunk_size, remaining)
            remaining -= cn
            spec = league.sample(rng)
            kind = spec[0]
            our = _sp(ta, 0, status=status)
            if kind == "latest":
                opp = _sp(tb, 1)
            elif kind == "snapshot":
                uid += 1
                opp = R.make_player(f"LGsx{uid}", tb, model_path=spec[1].path,
                                    team_chooser_path=team_chooser)
            else:   # scripted
                uid += 1
                opp = _make_opponent(spec[1], f"LGc{spec[1][:3]}{uid}", tb)
            try:
                coro = our.battle_against(opp, n_battles=cn)
                if battle_timeout and battle_timeout > 0:
                    await asyncio.wait_for(coro, timeout=battle_timeout * cn)
                else:
                    await coro
            except asyncio.TimeoutError:
                log.warning("league collect WATCHDOG fired (%d games vs %s) — continuing.", cn, kind)
            finally:
                source_counts.update(getattr(our, "_source_counts", {}) or {})
                our_trajs = our.finished_trajectories()
                trajectories.extend(our_trajs.values())
                games_done += cn                                 # live progress (3c.6e-3)
                for _t in our_trajs.values():
                    if _t.meta.won is not None:
                        decided += 1
                        won += 1 if _t.meta.won else 0
                if status is not None:
                    status.games(games_done, (won / decided) if decided else None)
                if kind == "latest":
                    source_counts.update(getattr(opp, "_source_counts", {}) or {})
                    trajectories.extend(opp.finished_trajectories().values())
                elif kind == "snapshot":
                    for traj in our_trajs.values():
                        if traj.meta.won is not None:
                            league.record_result(spec[1].snapshot_id, bool(traj.meta.won))
                if showcase_log is None:                  # grab one game's raw |-log
                    for _lines in (getattr(our, "_proto_log", {}) or {}).values():
                        if _lines:
                            showcase_log = list(_lines)
                            break
                await our.ps_client.stop_listening()
                await opp.ps_client.stop_listening()
                our.close()
                opp.close()
    return trajectories, dict(source_counts), showcase_log


def gauntlet_eval(candidate_path, *, teams, team_chooser, battles: int = 30,
                  matchup_seed: int = 0, battle_timeout: Optional[float] = 90.0,
                  manage_server: bool = False):
    """Evaluate a saved checkpoint on the scripted gauntlet (>=4 teams, side-balanced).
    Returns ``(results{opp:(wins,n)}, model_elo)``."""
    import asyncio

    import v_dance.eval.gauntlet as GA
    import v_dance.play.model_io as model_io
    # Fail LOUD if the candidate won't load — otherwise each gauntlet player silently
    # falls back to a no-model picker and the promotion gate evaluates garbage (3c.3b bug).
    model_io.load_bc_policy(candidate_path)
    results, _sources = asyncio.run(GA.run_gauntlet(
        opponents=list(SCRIPTED_OPPONENTS), team_pool=list(teams),
        battles_per_opponent=battles, ckpt=Path(candidate_path),
        team_chooser=Path(team_chooser), manage_server=manage_server,
        matchup_seed=matchup_seed, battle_timeout=battle_timeout))
    return results, GA.model_elo(results)


def build_train_configs(*, kl_coef: float = 0.5, target_kl_bc: Optional[float] = 0.15,
                        tau: float = 1.0, min_ev: Optional[float] = None):
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
    train = TrainConfig(target_kl_from_bc=tkl, min_explained_variance=min_ev)
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


def resolve_eval_battles(eval_battles, n_eval_teams: int) -> int:
    """Auto-size the gauntlet eval budget when unset (task 3c.7b). ``None`` => 2x the ordered
    team-pairs, i.e. FULL both-orientation side-balanced coverage with each matchup sampled
    ~twice (so the promotion-gate read isn't matchup-luck; sec 15/16). An explicit int passes
    through unchanged (use a small one, e.g. 12, for fast UI smokes)."""
    if eval_battles is not None:
        return int(eval_battles)
    pairs = max(1, n_eval_teams * (n_eval_teams - 1))
    return 2 * pairs


def run_live_generations(ckpt, *, n_generations=None, team_pool, team_chooser,
                         archive_dir, eval_team_pool=None,
                         gen_cfg: GenConfig = GenConfig(), ppo_cfg=None,
                         train_cfg=None, tau_start: float = 1.3, tau_end: float = 1.0,
                         tau_anneal_gens: int = 12, seed: int = 0,
                         eval_battles=None, manage_server: bool = True,
                         resume_from=None, snapshot_path=None, max_hours=None) -> dict:
    """Run real generations end-to-end (collect via the league -> PPO update -> gauntlet
    eval -> promotion gate -> admit/refresh/revert), RESUMABLY (3c.4): a snapshot is
    written after every generation, ``resume_from`` continues exactly, and the run stops
    cleanly on Ctrl-C or after ``max_hours``. ``n_generations=None`` => run until stopped.
    The Showdown server is started ONCE and reused across collect + eval.

    ``team_pool`` is the TRAINING pool (sec 15: draw both sides from the full archetype-rich
    set for pilot/counter exposure); ``eval_team_pool`` (default = ``team_pool``) is the
    CONTROLLED, side-balanced set the gauntlet gate judges on — so a generation's measured
    strength isn't 'got lucky with the team draw' (sec 15)."""
    import asyncio

    import v_dance.play.run_local_battle as R
    from v_dance.selfplay.actor_critic import ActorCritic
    from v_dance.selfplay.trainer import PPOTrainer
    from v_dance.selfplay import resume as RS
    from v_dance.selfplay.status import LiveStatus

    archive = Path(archive_dir)
    archive.mkdir(parents=True, exist_ok=True)
    snap_path = Path(snapshot_path) if snapshot_path else (archive / "resume.pt")
    _eval_pool = list(eval_team_pool) if eval_team_pool else list(team_pool)
    _eval_pairs = len(_eval_pool) * (len(_eval_pool) - 1)
    eval_battles = resolve_eval_battles(eval_battles, len(_eval_pool))
    print(f"   teams: TRAIN pool={len(team_pool)} (archetype draw, sec 15)  "
          f"EVAL pool={len(_eval_pool)} ({_eval_pairs} pairs)  "
          f"eval-battles={eval_battles}/opp (~{3 * eval_battles} games/gen)")
    if 0 < eval_battles < _eval_pairs:
        print(f"   [eval] {eval_battles} battles/opp < {_eval_pairs} ordered team-pairs -> "
              f"PARTIAL (but seed-fixed, gen-over-gen reproducible) coverage. For full "
              f"both-orientation side-balancing use --eval-battles >= {_eval_pairs}.")

    # Build from the base ckpt (architecture + frozen-BC reference), THEN overlay the
    # resume snapshot's trained state if present (sec 17 — ref/arch re-derived, not stored).
    ac = ActorCritic.from_bc_checkpoint(ckpt)
    trainer = PPOTrainer(ac, ppo_cfg, train_cfg, seed=seed)
    league = OpponentLeague(latest_path=str(ckpt))
    history = GenerationHistory()
    if resume_from and Path(resume_from).exists():
        league, history, _snap = RS.load_into(resume_from, actor_critic=ac, trainer=trainer)
        print(f"[3c.4] resumed from {resume_from} at generation {history.generation} "
              f"(league={len(league.snapshots)})")
    stop = RS.StopController(max_hours=max_hours)
    status = LiveStatus(archive / "status.json")     # live dashboard feed (3c.6e-3)
    status.start_run(n_generations, hours=max_hours)

    showcase = {"log": None}

    def collect_fn(ac_, lg, gen):
        # 3c.7c: one tau drives BOTH collection AND this gen's PPO recompute (3c.7a invariant).
        tau_gen = tau_for_generation(gen, tau_start, tau_end, tau_anneal_gens)
        trainer.cfg.tau = tau_gen
        status.phase("collecting", generation=gen, games_total=gen_cfg.n_games)
        if tau_anneal_gens > 0 and tau_start != tau_end:
            print(f"   collect: tau={tau_gen:.3f} (gen {gen}; anneal {tau_start}->{tau_end} "
                  f"over {tau_anneal_gens} gens)")
        trajs, src, log = asyncio.run(collect_with_league(
            ac_, lg, gen_cfg.n_games, team_pool=team_pool, tau=tau_gen,
            seed=seed + gen * 1000, matchup_seed=gen, team_chooser=team_chooser,
            status=status))
        showcase["log"] = log
        return trajs, src

    def save_fn(ac_, gen):
        p = archive / f"gen{gen}.pt"
        ac_.save(p, generation=gen)
        return str(p)

    def eval_fn(path):
        return gauntlet_eval(path, teams=_eval_pool, team_chooser=team_chooser,
                             battles=eval_battles, manage_server=False)

    def restore_fn(ac_, path):
        ac_.restore_from(path)

    def _save():
        RS.save_snapshot(snap_path, actor_critic=ac, trainer=trainer, league=league,
                         history=history, ppo_cfg=trainer.cfg, train_cfg=trainer.tcfg,
                         gen_cfg=gen_cfg, seed=seed)

    server = R.start_showdown() if manage_server else None
    reports = []
    done = 0
    try:
        while (n_generations is None or done < n_generations) and not stop.should_stop():
            rep = run_generation(ac, trainer, league, history, collect_fn=collect_fn,
                                 eval_fn=eval_fn, save_fn=save_fn, restore_fn=restore_fn,
                                 status=status, cfg=gen_cfg)
            print_generation_report(rep)
            us = rep.get("update_stats", {}) or {}
            print(f"        update: loss={us.get('loss', float('nan')):+.4f} "
                  f"kl_to_bc={us.get('kl_to_bc', float('nan')):.3e} "
                  f"EV={us.get('explained_variance', float('nan')):+.3f} "
                  f"clip_frac={us.get('clip_fraction', float('nan')):.2f} "
                  f"halted={us.get('halted')}")
            _save()                          # per-generation heartbeat snapshot
            from v_dance.selfplay import archive as AR   # manifest (self-play) + Type_D replay (3c.5)
            arts = AR.write_generation_artifacts(   # Type_D -> data/vods/Type_D by default
                archive, history, league, showcase_log=showcase["log"],
                tag=f"g{rep['generation']}")
            if "type_d_html" in arts:
                print(f"        archive: manifest.json + Type_D {Path(arts['type_d_html']).name}")
            status.phase("idle", generation=rep["generation"])
            reports.append(rep)
            done += 1
    finally:
        status.finish_run()
        _save()                              # final flush (also on Ctrl-C / exception)
        if server is not None:
            R.stop_showdown(server)
    print(f"\n  ran {done} generation(s); snapshot -> {snap_path}")
    print(f"  Elo curve: {[(g, round(e) if e else None) for g, e in history.elo_curve()]}")
    print(f"  league   : {[s.snapshot_id for s in league.snapshots]}")
    return {"history": history, "league": league, "reports": reports, "snapshot": str(snap_path)}


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

    # synthetic scripted win-rates per generation (rise, plateau-with-noise, then a
    # collapse) — sized so the gate clearly promotes / holds / reverts at z=1.
    scripted_wr = [0.48, 0.56, 0.63, 0.64, 0.63, 0.55][:n_generations]
    eval_games = 400
    calls = {"refresh": 0, "restore": 0}

    def collect_fn(ac, league, gen):
        return [object()] * 200, {"model": 4000}     # 200 fake trajectories

    def save_fn(ac, gen):
        return f"archive/gen{gen}.pt"

    def eval_fn(path):
        gen = len(history.records)
        wr = scripted_wr[min(gen, len(scripted_wr) - 1)]
        wins = round(wr * eval_games)
        results = {"random": (wins // 3, eval_games // 3),
                   "max_damage": (wins // 3, eval_games // 3),
                   "heuristic": (wins - 2 * (wins // 3), eval_games - 2 * (eval_games // 3))}
        return results, 1000 + (wr - 0.5) * 800

    def refresh_phi_fn(ac): calls["refresh"] += 1
    def restore_fn(ac, path): calls["restore"] += 1

    print("== Generation-loop dry run (no server) =====================")
    print(f"  synthetic scripted win-rate per gen: {scripted_wr}")
    print("  gate: promote if win-rate beats the accepted best beyond noise; "
          "revert on a real regression\n")
    for _ in range(n_generations):
        rep = run_generation(ac, trainer, league, history, collect_fn=collect_fn,
                             eval_fn=eval_fn, save_fn=save_fn, refresh_phi_fn=refresh_phi_fn,
                             restore_fn=restore_fn, cfg=GenConfig(warmup_updates=3))
        print_generation_report(rep)
    print(f"\n  Phi refreshes (on promote): {calls['refresh']}   "
          f"reverts (on regression): {calls['restore']}")
    print(f"  league snapshots admitted : {len(league.snapshots)}  "
          f"({[s.snapshot_id for s in league.snapshots]})")
    print(f"  Elo curve                 : "
          f"{[(g, round(e)) for g, e in history.elo_curve()]}")
    print("============================================================")


if __name__ == "__main__":
    import argparse
    import logging as _logging

    ap = argparse.ArgumentParser(description="Generation loop (3c.3)")
    ap.add_argument("--dry-run", action="store_true",
                    help="simulate the loop with synthetic eval (no server / model)")
    ap.add_argument("--live", action="store_true",
                    help="run REAL generations on the local Showdown server")
    ap.add_argument("--generations", type=int, default=6)
    ap.add_argument("--games", type=int, default=100, help="self-play games per generation")
    ap.add_argument("--eval-battles", type=int, default=None,
                    help="gauntlet battles per scripted opponent (split across the eval "
                         "team-pairs). DEFAULT auto-sizes to 2*N*(N-1) = full both-orientation "
                         "side-balanced coverage of the eval pool (6 teams => 60/opp, ~180 "
                         "games/gen). Pass a small int (e.g. 12) for a fast UI smoke.")
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
    ap.add_argument("--min-ev", type=float, default=None,
                    help="early-halt if critic explained-variance falls below this "
                         "(value-collapse guard; default off)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None,
                    help="resume snapshot to continue from (3c.4); use the SAME --ckpt")
    ap.add_argument("--snapshot", default=None,
                    help="resume-snapshot path to write (default <archive>/resume.pt)")
    ap.add_argument("--hours", type=float, default=None,
                    help="stop cleanly after ~this many wall-clock hours (between gens)")
    ap.add_argument("--no-server", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        _dry_run(args.generations, args.seed)
    elif args.live:
        _logging.basicConfig(level=_logging.DEBUG if args.verbose else _logging.WARNING)
        if not args.verbose:
            _logging.getLogger("poke_env").setLevel(_logging.WARNING)
            _logging.getLogger("websockets").setLevel(_logging.WARNING)
        if not Path(args.ckpt).exists():
            print(f"[gen] checkpoint not found: {args.ckpt}", file=sys.stderr)
            sys.exit(2)
        n_gen = None if args.generations <= 0 else args.generations   # 0 => run until stopped
        ppo_cfg, train_cfg = build_train_configs(
            kl_coef=args.kl_coef, target_kl_bc=args.target_kl_bc,
            tau=args.tau_start, min_ev=args.min_ev)   # gen-0 tau; collect_fn re-sets per gen
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
        run_live_generations(
            Path(args.ckpt), n_generations=n_gen, team_pool=train_pool,
            eval_team_pool=eval_pool,
            team_chooser=args.team_chooser, archive_dir=args.archive,
            gen_cfg=GenConfig(n_games=args.games, warmup_updates=args.warmup),
            ppo_cfg=ppo_cfg, train_cfg=train_cfg,
            tau_start=args.tau_start, tau_end=args.tau, tau_anneal_gens=args.tau_anneal_gens,
            seed=args.seed, eval_battles=args.eval_battles,
            manage_server=not args.no_server, resume_from=args.resume,
            snapshot_path=args.snapshot, max_hours=args.hours)
    else:
        ap.print_help()
