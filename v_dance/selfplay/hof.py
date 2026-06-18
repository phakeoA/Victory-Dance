"""Phase-2 Hall-of-Fame breadth veto — LIVE orchestration (P2.3 plumbing + P2.4 wiring).

The PURE gate logic (``HoFConfig``, ``cluster_hof_suspects``, ``hall_of_fame_gate``) lives in
``gate.py``. This module is the LIVE half:

  * ``hof_eval``      plays the candidate vs each PAST-CHAMPION suspect on a real Showdown server
                      (one ``run_gauntlet`` 'prev_best' mirror call per suspect — zero new battle
                      code), returning the per-suspect ``(id, wins, games)`` the gate consumes.
  * ``apply_hof_gate`` is the ``run_generation`` hook: on a v2 PROMOTE it turns the promote into a
                      HOLD when the candidate is PROVEN losing to one of its past champions
                      (lineage cycling), tracks the standoff streak, and honours ``--hof-override``.

``apply_hof_gate`` takes the games-runner as an INJECTED callable, so the whole wiring (cluster ->
gate -> downgrade -> streak -> override) is unit-tested offline with a fake runner; only
``hof_eval`` touches poke-env.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from v_dance.selfplay.gate import HoFConfig, cluster_hof_suspects, hall_of_fame_gate

log = logging.getLogger(__name__)


async def hof_eval(candidate_path, suspects, *, team_pool, team_chooser,
                   games_per_snapshot: int = 60, matchup_seed: int = 0,
                   battle_timeout: Optional[float] = 90.0, n_workers: int = 1,
                   manage_server: bool = False, gauntlet_fn=None,
                   live_dir=None, save_replays: bool = False):
    """Play the CANDIDATE vs each HoF SUSPECT (a frozen past-champion checkpoint) for
    ``games_per_snapshot`` side-balanced battles, returning ``[(snapshot_id, wins, games), ...]``.

    Reuses the TESTED model-vs-model path — one ``run_gauntlet`` call per suspect with
    ``opponents=["prev_best"]`` + ``prev_best_ckpt=suspect.path`` — so there is ZERO new battle
    code. The server is SHARED across suspects (started here iff ``manage_server``; in the live loop
    it's already up so the caller passes ``manage_server=False``). Each suspect's checkpoint is
    pre-validated (``load_bc_policy``) so a missing/corrupt snapshot SKIPS rather than silently
    falling back to a no-model picker and FALSE-vetoing on garbage. ``gauntlet_fn`` is injectable so
    the aggregation unit-tests offline without a server."""
    import v_dance.play.model_io as model_io
    import v_dance.play.run_local_battle as R
    from v_dance.eval.gauntlet import _ckpt_gen        # candidate/suspect gen for 22d name salts
    if gauntlet_fn is None:                            # lazy (keeps importers torch/poke-env-free)
        import v_dance.eval.gauntlet as GA
        gauntlet_fn = GA.run_gauntlet
    model_io.load_bc_policy(str(candidate_path))       # fail LOUD if the candidate won't load
    cand_gen = _ckpt_gen(candidate_path)               # N in checkpoints/genN.pt (None if unnamed)
    server = R.start_showdown() if manage_server else None
    results: list = []
    try:
        for suspect in suspects:
            try:
                model_io.load_bc_policy(str(suspect.path))
            except Exception as e:                     # missing / corrupt snapshot → skip (never veto on garbage)
                log.warning("HoF suspect %s won't load (%s) — skipping", suspect.snapshot_id, e)
                continue
            # 22d: salt the HoF account names with BOTH the candidate gen AND the suspect, so they
            # never collide with the main prev_best mirror at the SAME gen (which uses salt=str(N)
            # — same opponent KIND, same uids) nor with another suspect's gauntlet. 'h' is a
            # toID-safe separator (Showdown strips '_').
            _sg = _ckpt_gen(suspect.path)
            if _sg is None:
                _sg = "".join(ch for ch in str(suspect.snapshot_id) if ch.isdigit()) or "x"
            _salt = f"{cand_gen if cand_gen is not None else 'c'}h{_sg}"
            res, _sources = await gauntlet_fn(
                opponents=["prev_best"], team_pool=list(team_pool),
                battles_per_opponent=int(games_per_snapshot),
                ckpt=Path(candidate_path), team_chooser=Path(team_chooser),
                prev_best_ckpt=Path(suspect.path), manage_server=False,
                matchup_seed=matchup_seed, battle_timeout=battle_timeout,
                n_workers=int(n_workers), name_salt=_salt,
                # task E: the candidate-vs-past-champion battles also save to eval/league/ named
                # gen<N>_vs_gen<M> (M = the suspect's gen, parsed from its checkpoint path).
                live_dir=live_dir, save_replays=save_replays)
            wins, games = res.get("prev_best", (0, 0))
            results.append((suspect.snapshot_id, int(wins), int(games)))
    finally:
        if server is not None:
            R.stop_showdown(server)
    return results


def apply_hof_gate(verdict, reason, *, candidate_path, league, history,
                   cfg: HoFConfig, hof_eval_fn):
    """Phase-2 HoF wiring (P2.4): on a v2 PROMOTE, require the candidate to ALSO not-lose to its
    past champions. Returns ``(verdict, reason, hof_stats)``:

      * HoF REJECT   → downgrade promote → ``("hold", "hof_reject")`` and bump
                       ``history.hof_reject_streak`` — UNLESS ``cfg.override`` (then the promote
                       STANDS, flagged ``overridden``, streak reset: the operator's manual escape).
      * CONFIRM/SKIP → the promote stands; the standoff streak resets.

    No-op (returns the input verdict, ``hof_stats=None``) when ``verdict != "promote"``, the HoF is
    disabled, or no runner is injected (offline ``--dry-run`` / the existing tests). ``hof_eval_fn(
    candidate_path, suspects) -> [(id, wins, games)]`` is INJECTED, so cluster → gate → downgrade →
    streak → override are all offline-testable with a fake runner."""
    if verdict != "promote" or not cfg.enabled or hof_eval_fn is None:
        return verdict, reason, None
    suspects = cluster_hof_suspects(league.snapshots, n=cfg.n_champions,
                                    current_champion_path=history.best_path)
    if len(suspects) < cfg.min_pool:          # too few past champions to cycle yet → skip, play NO games
        history.hof_reject_streak = 0
        return verdict, reason, {"reason": "thin_pool_skip", "eligible": len(suspects),
                                 "min_pool": cfg.min_pool, "promote_reason": reason,
                                 "n_suspects": len(suspects), "suspects": []}
    snap_results = hof_eval_fn(candidate_path, suspects)
    hof_verdict, hof_stats = hall_of_fame_gate(snap_results, cfg=cfg)
    hof_stats["promote_reason"] = reason       # which promote was checked (beat_champion / plateau_reanchor)
    if hof_verdict == "reject":
        if cfg.override:
            hof_stats["overridden"] = True
            history.hof_reject_streak = 0
            return verdict, reason, hof_stats          # manual override: promote stands (logged loud)
        history.hof_reject_streak = int(getattr(history, "hof_reject_streak", 0) or 0) + 1
        hof_stats["reject_streak"] = history.hof_reject_streak
        return "hold", "hof_reject", hof_stats
    history.hof_reject_streak = 0                       # confirm / skip → no active standoff
    return verdict, reason, hof_stats
