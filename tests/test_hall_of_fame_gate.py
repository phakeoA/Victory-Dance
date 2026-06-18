"""P2.2 — the Hall-of-Fame breadth veto (pure gate + suspect builder).

cluster_hof_suspects selects the candidate's PAST CHAMPIONS (accepted promotions, EXCLUDING the
current one the mirror already tests) — the strong, era-spread milestones of its own lineage;
hall_of_fame_gate VETOES a mirror-promote iff ANY eligible suspect is PROVEN losing (wilson_upper
< 0.50) — worst-of-snapshots, NOT band averaging. Requiring the candidate to not-lose to its prior
N champions catches LINEAGE CYCLING (beats the current champion but loses to an older one).
Numbers locked in P2.1.
"""
from __future__ import annotations

from v_dance.selfplay.gate import (
    cluster_hof_suspects, hall_of_fame_gate, HoFConfig, wilson_upper_bound)
from v_dance.selfplay.league import LeagueSnapshot


def _champ(gen):
    return LeagueSnapshot(f"gen{gen}", f"a/gen{gen}.pt", gen, is_champion=True)


def _hold(gen):
    return LeagueSnapshot(f"gen{gen}", f"a/gen{gen}.pt", gen, is_champion=False)


# ── cluster_hof_suspects (past champions, excluding the current) ───────────────
def test_cluster_selects_past_champions_newest_first_excluding_current():
    # champions promoted at gens 0,2,5,9,13,17 ; non-champion HOLD gens interspersed
    snaps = [_champ(0), _hold(1), _champ(2), _hold(3), _hold(4), _champ(5),
             _hold(6), _champ(9), _champ(13), _champ(17)]
    suspects = cluster_hof_suspects(snaps, n=5)
    ids = [s.snapshot_id for s in suspects]
    assert "gen17" not in ids                                 # the CURRENT champion (highest gen) excluded
    assert all(s.is_champion for s in suspects)               # only champions, never a held gen
    assert ids == ["gen13", "gen9", "gen5", "gen2", "gen0"]   # the last 5 past champions, newest first


def test_cluster_excludes_current_by_explicit_path():
    snaps = [_champ(0), _champ(2), _champ(5)]
    suspects = cluster_hof_suspects(snaps, n=5, current_champion_path="a/gen2.pt")
    ids = {s.snapshot_id for s in suspects}
    assert "gen2" not in ids and ids == {"gen0", "gen5"}


def test_cluster_empty_when_no_or_only_current_champion():
    assert cluster_hof_suspects([_hold(0), _hold(1)], n=5) == []   # no champions yet
    assert cluster_hof_suspects([_champ(3)], n=5) == []            # only the current champion → nothing to test


def test_cluster_caps_at_n_and_is_deterministic():
    snaps = [_champ(g) for g in range(0, 20, 2)]                   # 10 champions (gens 0..18)
    a = [s.snapshot_id for s in cluster_hof_suspects(snaps, n=5)]
    b = [s.snapshot_id for s in cluster_hof_suspects(snaps, n=5)]
    assert a == b and len(a) == 5                                  # last 5 past champions, deterministic
    assert "gen18" not in a                                        # current excluded


# ── hall_of_fame_gate (the breadth veto) ──────────────────────────────────────
def test_hof_confirm_when_no_suspect_proven_losing():
    res = [(f"gen{i}", 36, 60) for i in range(5)]           # all ~0.60 → none proven losing
    v, st = hall_of_fame_gate(res)
    assert v == "confirm" and st["reason"] == "hof_confirm"


def test_hof_reject_on_one_proven_losing_suspect():
    res = [(f"gen{i}", 36, 60) for i in range(4)] + [("gen5", 18, 60)]   # one hidden 0.30 counter
    v, st = hall_of_fame_gate(res)
    assert v == "reject" and st["reason"] == "hof_reject"
    assert st["worst_snapshot_id"] == "gen5"
    assert st["worst_upper"] < 0.5


def test_hof_thin_pool_skips_fail_open():
    res = [("gen0", 18, 60)]                                # only 1 eligible < min_pool (2) → SKIP
    v, st = hall_of_fame_gate(res)
    assert v == "skip" and st["reason"] == "thin_pool_skip"
    assert st["eligible"] == 1


def test_hof_below_games_floor_cannot_veto():
    res = [(f"gen{i}", 36, 60) for i in range(3)] + [("low", 0, 20)]    # 'low' 0/20 < floor 40
    v, st = hall_of_fame_gate(res)
    rows = {r["snapshot_id"]: r for r in st["suspects"]}
    assert rows["low"]["eligible"] is False and rows["low"]["vetoed"] is False
    assert v == "confirm"                                   # the 3 eligible all >=0.5 → confirm


def test_hof_decision_matches_wilson_upper_rule():
    """Fidelity: the gate rejects iff some eligible suspect's wilson_upper < 0.50 — the exact rule
    the P2.1 sim calibrated."""
    cfg = HoFConfig()
    for wins in (15, 18, 21, 24, 27, 30):
        res = [(f"g{i}", 36, 60) for i in range(4)] + [("suspect", wins, 60)]
        v, _ = hall_of_fame_gate(res, cfg=cfg)
        assert (v == "reject") == (wilson_upper_bound(wins, 60, cfg.z) < 0.5), f"wins={wins}"
