"""P2.4 — apply_hof_gate: the run_generation HoF hook (offline, injected runner).

On a v2 PROMOTE the HoF requires the candidate to also not-lose to its PAST champions; a reject
downgrades the promote to a HOLD (unless --hof-override). The games-runner is injected so the
whole cluster -> gate -> downgrade -> streak -> override path tests without a server.
"""
from __future__ import annotations

from v_dance.selfplay.hof import apply_hof_gate
from v_dance.selfplay.gate import HoFConfig, GenerationHistory
from v_dance.selfplay.league import OpponentLeague


def _league(champion_gens):
    lg = OpponentLeague(latest_path="cur.pt")
    for g in champion_gens:
        lg.admit(f"gen{g}", f"a/gen{g}.pt", g, is_champion=True)
    return lg


def _history(best_path, streak=0):
    h = GenerationHistory()
    h.best_path = best_path
    h.hof_reject_streak = streak
    return h


def _fake_runner(results_by_id):
    def fn(candidate_path, suspects):
        return [(s.snapshot_id, *results_by_id[s.snapshot_id]) for s in suspects]
    return fn


def test_confirm_keeps_promote_and_resets_streak():
    h = _history("a/gen6.pt", streak=1)
    fn = _fake_runner({"gen0": (40, 60), "gen2": (40, 60)})        # beats both past champions
    v, r, st = apply_hof_gate("promote", "beat_champion", candidate_path="c.pt",
                              league=_league([0, 2, 6]), history=h, cfg=HoFConfig(), hof_eval_fn=fn)
    assert v == "promote" and r == "beat_champion"
    assert st["reason"] == "hof_confirm" and h.hof_reject_streak == 0


def test_reject_downgrades_to_hold_and_bumps_streak():
    h = _history("a/gen6.pt")
    fn = _fake_runner({"gen0": (18, 60), "gen2": (40, 60)})        # LOSES to gen0 (lineage cycle)
    v, r, st = apply_hof_gate("promote", "beat_champion", candidate_path="c.pt",
                              league=_league([0, 2, 6]), history=h, cfg=HoFConfig(), hof_eval_fn=fn)
    assert v == "hold" and r == "hof_reject"
    assert st["worst_snapshot_id"] == "gen0" and st["reject_streak"] == 1 and h.hof_reject_streak == 1


def test_override_lets_reject_through_and_resets_streak():
    h = _history("a/gen6.pt", streak=3)
    fn = _fake_runner({"gen0": (18, 60), "gen2": (40, 60)})
    v, r, st = apply_hof_gate("promote", "plateau_reanchor", candidate_path="c.pt",
                              league=_league([0, 2, 6]), history=h,
                              cfg=HoFConfig(override=True), hof_eval_fn=fn)
    assert v == "promote" and r == "plateau_reanchor"
    assert st.get("overridden") is True and h.hof_reject_streak == 0


def test_thin_pool_skips_without_playing_games():
    h = _history("a/gen6.pt")
    calls = {"n": 0}

    def fn(candidate_path, suspects):
        calls["n"] += 1
        return [(s.snapshot_id, 40, 60) for s in suspects]

    # champions [0, 6]: current=gen6, only gen0 is a past champion → 1 < min_pool (2) → SKIP, no games
    v, r, st = apply_hof_gate("promote", "beat_champion", candidate_path="c.pt",
                              league=_league([0, 6]), history=h, cfg=HoFConfig(), hof_eval_fn=fn)
    assert v == "promote" and st["reason"] == "thin_pool_skip" and calls["n"] == 0


def test_noop_when_not_a_promote_or_disabled_or_no_runner():
    h = _history("a/gen6.pt")
    fn = _fake_runner({"gen0": (0, 60), "gen2": (0, 60)})          # would veto if ever run
    # a HOLD is never HoF-checked
    assert apply_hof_gate("hold", "hold", candidate_path="c.pt", league=_league([0, 2, 6]),
                          history=h, cfg=HoFConfig(), hof_eval_fn=fn) == ("hold", "hold", None)
    # disabled HoF
    assert apply_hof_gate("promote", "beat_champion", candidate_path="c.pt",
                          league=_league([0, 2, 6]), history=h,
                          cfg=HoFConfig(enabled=False), hof_eval_fn=fn)[0] == "promote"
    # no runner injected (offline --dry-run without a fake)
    assert apply_hof_gate("promote", "beat_champion", candidate_path="c.pt",
                          league=_league([0, 2, 6]), history=h,
                          cfg=HoFConfig(), hof_eval_fn=None)[2] is None
