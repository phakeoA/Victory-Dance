"""Task 3c.2: opponent league — mixture, anchor decay, PFSP weighting, sampling,
admission/promotion, persistence, and the opponent dispatch. Pure (no poke-env).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
from v_dance.selfplay.league import (OpponentLeague, LeagueConfig, LeagueSnapshot,  # noqa: E402
                              make_league_opponent)


def _league(snaps=()):
    lg = OpponentLeague(latest_path="latest.pt")
    for sid, gen, wins, games in snaps:
        s = lg.admit(sid, f"{sid}.pt", gen)
        s.wins_vs_latest, s.games_vs_latest = wins, games
    return lg


# ── mixture + anchor decay ────────────────────────────────────────────────────
def test_mixture_empty_pool_has_no_past():
    m = _league().mixture()
    assert m["past"] == 0.0
    assert m["scripted"] == pytest.approx(0.25)              # scripted_frac_max early
    assert m["latest"] == pytest.approx(0.75)
    assert sum(m.values()) == pytest.approx(1.0)


def test_mixture_with_pool_keeps_latest_past_ratio():
    m = _league([("g1", 1, 50, 100)]).mixture()
    assert sum(m.values()) == pytest.approx(1.0)
    # within the non-scripted remainder, latest:past stays 0.5:0.3
    assert m["latest"] / m["past"] == pytest.approx(0.5 / 0.3)
    assert m["past"] > 0


def test_scripted_fraction_decays_and_floors():
    lg = OpponentLeague(latest_path="x")
    assert lg.scripted_fraction() == pytest.approx(0.25)    # n=0
    prev = 0.25
    for i in range(20):
        lg.admit(f"g{i}", f"{i}.pt", i)
        cur = lg.scripted_fraction()
        assert cur <= prev + 1e-12                          # monotone non-increasing
        prev = cur
    assert lg.scripted_fraction() == pytest.approx(0.10)    # floored at min


def test_no_scripted_redistributes():
    lg = OpponentLeague(latest_path="x", scripted=())
    assert lg.scripted_fraction() == 0.0
    m = lg.mixture()
    assert m["scripted"] == 0.0 and m["latest"] == pytest.approx(1.0)


# ── PFSP weighting ────────────────────────────────────────────────────────────
def test_pfsp_favours_hard_counters():
    lg = _league([("beats", 1, 80, 100), ("even", 2, 50, 100), ("loses", 3, 20, 100)])
    w = dict(zip([s.snapshot_id for s in lg.snapshots], lg.pfsp_weights()))
    assert w["loses"] > w["even"] > w["beats"]              # favour what the latest loses to
    assert w["beats"] >= lg.cfg.pfsp_floor                  # never zero (still recurs)


def test_pfsp_unplayed_is_neutral():
    lg = _league([("new", 1, 0, 0)])
    # winrate prior 0.5 -> weight (1-0.5)^2 = 0.25
    assert lg.pfsp_weights()[0] == pytest.approx(0.25)


# ── sampling distribution (seeded, statistical) ───────────────────────────────
def test_sample_matches_mixture_and_pfsp():
    lg = _league([("beats", 1, 80, 100), ("loses", 3, 20, 100)])
    rng = np.random.default_rng(0)
    mix = lg.mixture()
    buckets, snaps = {}, {}
    N = 30000
    for _ in range(N):
        kind, val = lg.sample(rng)
        buckets[kind] = buckets.get(kind, 0) + 1
        if kind == "snapshot":
            snaps[val.snapshot_id] = snaps.get(val.snapshot_id, 0) + 1
    for k in ("latest", "past", "scripted"):
        got = buckets.get("snapshot" if k == "past" else k, 0) / N
        assert got == pytest.approx(mix[k], abs=0.02)
    assert snaps["loses"] > snaps["beats"]                  # PFSP favours the hard counter


def test_sample_returns_valid_specs():
    lg = _league([("g1", 1, 50, 100)])
    rng = np.random.default_rng(1)
    for _ in range(50):
        kind, val = lg.sample(rng)
        assert kind in ("latest", "snapshot", "scripted")
        if kind == "snapshot":
            assert isinstance(val, LeagueSnapshot)


# ── admission / promotion / results ───────────────────────────────────────────
def test_admit_and_record_result():
    lg = _league()
    lg.admit("g1", "g1.pt", 1)
    lg.record_result("g1", latest_won=True)
    lg.record_result("g1", latest_won=False)
    s = lg.snapshots[0]
    assert s.games_vs_latest == 2 and s.wins_vs_latest == 1
    assert s.latest_winrate() == pytest.approx(0.5)


def test_promote_latest_demotes_and_resets_pfsp():
    lg = _league([("g1", 1, 80, 100)])
    lg.promote_latest("gen5.pt", demote_old_as="g4", generation=4)
    assert lg.latest_path == "gen5.pt"
    assert any(s.snapshot_id == "g4" and s.path == "latest.pt" for s in lg.snapshots)
    # the new latest hasn't faced the pool -> all PFSP records reset to the 0.5 prior
    assert all(s.games_vs_latest == 0 for s in lg.snapshots)


# ── persistence ───────────────────────────────────────────────────────────────
def test_persistence_round_trip():
    lg = _league([("g1", 1, 30, 100), ("g2", 2, 70, 100)])
    lg.cfg = LeagueConfig(latest_frac=0.6, pfsp_power=3.0)
    back = OpponentLeague.from_obj(lg.to_obj())
    assert back.latest_path == lg.latest_path
    assert [s.to_obj() for s in back.snapshots] == [s.to_obj() for s in lg.snapshots]
    assert back.cfg.latest_frac == 0.6 and back.cfg.pfsp_power == 3.0


# ── opponent dispatch (DI, no poke-env) ───────────────────────────────────────
def test_make_league_opponent_dispatch():
    def make_model(username, team, model_path):
        return ("model", model_path)

    def make_scripted(kind, username, team):
        return ("scripted", kind)

    snap = LeagueSnapshot("g1", "g1.pt", 1)
    kw = dict(username="u", team="t", make_model=make_model, make_scripted=make_scripted)
    assert make_league_opponent(("latest", "latest.pt"), **kw) == ("model", "latest.pt")
    assert make_league_opponent(("snapshot", snap), **kw) == ("model", "g1.pt")
    assert make_league_opponent(("scripted", "heuristic"), **kw) == ("scripted", "heuristic")
    with pytest.raises(ValueError, match="unknown opponent spec"):
        make_league_opponent(("bogus", None), **kw)
