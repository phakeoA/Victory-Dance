"""Tests for the pure (no-server) parts of the win-rate gauntlet (#3):
Elo estimation, versioned persistence, the regression gate, and the
side-balanced team rotation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
import v_dance.eval.gauntlet as G  # noqa: E402


# ── gauntlet_eval forwards the v2 mirror-battles override ─────────────────────
def test_gauntlet_eval_forwards_mirror_battles(monkeypatch):
    """The champion MIRROR runs more battles than the scripted ladder for the v2 gate's 70% bar.
    gauntlet_eval must forward ``mirror_battles`` to run_gauntlet while ``battles`` stays the
    scripted count."""
    import v_dance.play.model_io as MIO
    from v_dance.selfplay.generation import gauntlet_eval
    monkeypatch.setattr(MIO, "load_bc_policy", lambda p: None)   # skip the real checkpoint load
    captured = {}

    async def fake_run_gauntlet(**kw):
        captured.update(kw)
        return {"random": (5, 10), "max_damage": (5, 10), "heuristic": (5, 10),
                "prev_best": (7, 10)}, {}
    monkeypatch.setattr(G, "run_gauntlet", fake_run_gauntlet)

    gauntlet_eval("cand.pt", teams=["A", "B"], team_chooser="tc.pt", battles=60,
                  prev_best_path="champ.pt", mirror_battles=240)
    assert captured["mirror_battles"] == 240          # mirror bumped
    assert captured["battles_per_opponent"] == 60     # scripted ladder unchanged


# ── Elo ───────────────────────────────────────────────────────────────────────
def test_expected_score_symmetry_and_monotonicity():
    assert abs(G.expected_score(1500, 1500) - 0.5) < 1e-9
    assert G.expected_score(1700, 1500) > 0.5
    assert G.expected_score(1300, 1500) < 0.5
    # +400 rating ≈ 10:1 expectation
    assert abs(G.expected_score(1900, 1500) - 10 / 11) < 1e-3


def test_implied_rating_inverts_winrate():
    # 50% vs a 1500 anchor → exactly 1500
    assert abs(G.implied_rating(10, 20, 1500) - 1500) < 1e-6
    # higher win-rate → higher implied rating
    assert G.implied_rating(18, 20, 1500) > G.implied_rating(10, 20, 1500)
    # 0% and 100% stay finite (continuity-clamped), not +/-inf
    assert G.implied_rating(0, 20, 1500) is not None
    assert G.implied_rating(20, 20, 1500) > G.implied_rating(0, 20, 1500)
    assert G.implied_rating(5, 0, 1500) is None        # no games


def test_model_elo_weights_anchors_and_ignores_non_anchor():
    # strong vs random, even vs heuristic → elo lands between the two anchors
    res = {"random": (19, 20), "heuristic": (10, 20)}
    elo = G.model_elo(res)
    assert 1000 < elo < 1700
    # a non-anchor opponent (prev_best mirror) does NOT shift the estimate
    res2 = dict(res, prev_best=(15, 20))
    assert abs(G.model_elo(res2) - elo) < 1e-9
    assert G.model_elo({}) is None


# ── Run row + persistence ─────────────────────────────────────────────────────
def test_build_run_row_scripted_winrate_excludes_mirror():
    res = {"random": (18, 20), "max_damage": (12, 20), "heuristic": (6, 20),
           "prev_best": (11, 20)}
    row = G.build_run_row(res, ckpt="bc_best.pt", run_id="r1", timestamp="t")
    # scripted win-rate averages only the 3 anchored opponents: (18+12+6)/60
    assert abs(row["scripted_win_rate"] - 36 / 60) < 1e-9
    assert row["per_opponent"]["random"]["win_rate"] == 0.9
    assert row["per_opponent"]["prev_best"]["anchor"] is None
    assert row["model_elo"] is not None


def test_history_roundtrip_and_append(tmp_path):
    hist_path = tmp_path / "h.json"
    assert G.load_history(hist_path) == []                  # missing → empty
    r1 = G.build_run_row({"random": (15, 20)}, "c", "r1", "t1")
    G.append_run(hist_path, r1)
    r2 = G.build_run_row({"random": (18, 20)}, "c", "r2", "t2")
    hist = G.append_run(hist_path, r2)
    assert len(hist) == 2
    reloaded = json.loads(hist_path.read_text(encoding="utf-8"))
    assert [r["run"] for r in reloaded] == ["r1", "r2"]


def test_history_tolerates_corrupt_file(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("not json{", encoding="utf-8")
    assert G.load_history(bad) == []


# ── Regression gate ───────────────────────────────────────────────────────────
def test_regression_gate_no_baseline():
    cur = G.build_run_row({"random": (15, 20)}, "c", "r1", "t")
    g = G.regression_gate([], cur)
    assert g["have_baseline"] is False and g["regressed"] is False


def test_regression_gate_detects_drop_and_passes_improvement():
    prior = {"model_elo": 1400.0}
    better = {"model_elo": 1480.0}
    worse = {"model_elo": 1320.0}
    assert G.regression_gate([prior], better)["regressed"] is False
    assert G.regression_gate([prior], better)["delta"] == 80.0
    assert G.regression_gate([prior], worse)["regressed"] is True
    # tolerance absorbs a tiny dip
    assert G.regression_gate([prior], {"model_elo": 1395.0},
                             tolerance=10.0)["regressed"] is False


# ── Side-balanced team rotation ───────────────────────────────────────────────
def test_team_matchups_single_team_is_mirror():
    assert G.team_matchups(["team1"], 8) == [("team1", "team1", 8)]
    assert G.team_matchups([], 4) == [("team1", "team1", 4)]


def test_team_matchups_swaps_sides_and_sums():
    chunks = G.team_matchups(["A", "B"], 10)
    assert sum(n for _, _, n in chunks) == 10
    pairs = {(a, b) for a, b, _ in chunks}
    assert ("A", "B") in pairs and ("B", "A") in pairs      # both side assignments
    # each team is the MODEL side and the OPP side equally
    as_model = sum(n for m, _, n in chunks if m == "A")
    as_opp = sum(n for _, o, n in chunks if o == "A")
    assert as_model == as_opp


def test_team_matchups_distributes_remainder():
    chunks = G.team_matchups(["A", "B"], 3)                 # 2 ordered pairs
    assert sum(n for _, _, n in chunks) == 3
    assert sorted(n for _, _, n in chunks) == [1, 2]


def test_team_matchups_large_pool_samples_representatively():
    pool = [f"T{i}" for i in range(10)]                     # 90 ordered pairs
    chunks = G.team_matchups(pool, 20, seed=0)              # N < #pairs
    assert sum(n for _, _, n in chunks) == 20
    assert all(n == 1 for _, _, n in chunks)               # distinct single matchups
    assert all(a != b for a, b, _ in chunks)               # never a self-mirror
    # NOT a biased prefix of team[0]: many distinct teams appear on the model side
    model_side = {a for a, _, _ in chunks}
    assert len(model_side) >= 5


def test_team_matchups_seed_is_reproducible_and_varies():
    pool = [f"T{i}" for i in range(8)]
    a = G.team_matchups(pool, 16, seed=0)
    b = G.team_matchups(pool, 16, seed=0)
    c = G.team_matchups(pool, 16, seed=1)
    assert a == b                                          # same seed → identical (A/B control)
    assert a != c                                          # different seed → different sample


# ── Async orchestration glue (mocked battle_against — no server) ──────────────
def test_run_gauntlet_aggregates_over_team_rotation(monkeypatch):
    import asyncio
    import types as _t
    import pytest
    pytest.importorskip("poke_env")
    import v_dance.play.run_local_battle as R

    class FakePlayer:
        def __init__(self, win_rate=0.0):
            self.n_won_battles = 0
            self.n_finished_battles = 0
            self._wr = win_rate
            self.closed = False
            async def _stop():
                return None
            self.ps_client = _t.SimpleNamespace(stop_listening=_stop)

        async def battle_against(self, opp, n_battles):
            self.n_finished_battles += n_battles
            self.n_won_battles += int(round(self._wr * n_battles))

        def close(self):
            self.closed = True

    made = {"model": [], "opp": []}

    def fake_make_player(username, team, *, model_path=None, team_chooser_path=None):
        p = FakePlayer(win_rate=1.0)               # the model wins every battle
        made["model"].append(p)
        return p

    def fake_make_opponent(kind, username, team, model_path=None, team_chooser_path=None):
        p = FakePlayer(win_rate=0.0)
        made["opp"].append(p)
        return p

    monkeypatch.setattr(R, "start_showdown", lambda: None)
    monkeypatch.setattr(R, "stop_showdown", lambda proc: None)
    monkeypatch.setattr(R, "make_player", fake_make_player)
    monkeypatch.setattr(R, "load_team", lambda p: "TEAMSTR")
    monkeypatch.setattr(R, "resolve_team_path", lambda n: n)
    monkeypatch.setattr(G, "_make_opponent", fake_make_opponent)

    results, sources = asyncio.run(G.run_gauntlet(
        opponents=["random", "max_damage"],
        team_pool=["A", "B"],
        battles_per_opponent=4,                    # 2 ordered pairs × 2 each
        ckpt=Path("bc.pt"), team_chooser=Path("tc.pt"),
        manage_server=True,
    ))
    # model won every battle: 4 per opponent
    assert results == {"random": (4, 4), "max_damage": (4, 4)}
    assert isinstance(sources, dict)               # decision-source tally returned
    # 2 opponents × 2 team-pairings = 4 model players + 4 opponents, all closed
    assert len(made["model"]) == 4 and len(made["opp"]) == 4
    assert all(p.closed for p in made["model"] + made["opp"])


def test_run_gauntlet_parallel_aggregates_same(monkeypatch):
    """3c.8c: with n_workers>1 the gauntlet runs chunks concurrently — the per-kind (wins,
    finished) accumulation + source tally must still aggregate correctly (atomic finally)."""
    import asyncio
    import types as _t
    import pytest
    pytest.importorskip("poke_env")
    import v_dance.play.run_local_battle as R

    class FakePlayer:
        def __init__(self, wr=0.0):
            self.n_won_battles = 0
            self.n_finished_battles = 0
            self._wr = wr
            self.closed = False
            async def _stop():
                return None
            self.ps_client = _t.SimpleNamespace(stop_listening=_stop)

        async def battle_against(self, opp, n_battles):
            self.n_finished_battles += n_battles
            self.n_won_battles += int(round(self._wr * n_battles))

        def close(self):
            self.closed = True

    made = []
    monkeypatch.setattr(R, "start_showdown", lambda: None)
    monkeypatch.setattr(R, "stop_showdown", lambda p: None)
    monkeypatch.setattr(R, "make_player",
                        lambda u, t, **kw: made.append("m") or FakePlayer(wr=1.0))
    monkeypatch.setattr(R, "load_team", lambda p: "T")
    monkeypatch.setattr(R, "resolve_team_path", lambda n: n)
    monkeypatch.setattr(G, "_make_opponent", lambda *a, **kw: FakePlayer(wr=0.0))

    results, _sources = asyncio.run(G.run_gauntlet(
        opponents=["random", "max_damage"], team_pool=["A", "B", "C"],
        battles_per_opponent=6, ckpt=Path("bc.pt"), team_chooser=Path("tc.pt"),
        manage_server=True, n_workers=4))
    assert results == {"random": (6, 6), "max_damage": (6, 6)}   # model wins all; agg correct
    assert len(made) == 12                                       # 2 opps × 6 pairings


def test_play_chunk_watchdog_abandons_hung_battle():
    """A battle that never resolves must NOT hang the gauntlet — the watchdog
    abandons the chunk and returns promptly (the 3-hour-hang guarantee)."""
    import asyncio

    class HangPlayer:
        def __init__(self):
            self.n_won_battles = 0
            self.n_finished_battles = 0

        async def battle_against(self, opp, n_battles):
            await asyncio.sleep(30)          # never finishes within the timeout

    wins, fin = asyncio.run(G._play_chunk(HangPlayer(), object(), 1, timeout=0.05))
    assert (wins, fin) == (0, 0)             # timed out → abandoned, nothing counted


def test_play_chunk_no_timeout_counts_wins():
    import asyncio

    class QuickPlayer:
        def __init__(self):
            self.n_won_battles = 0
            self.n_finished_battles = 0

        async def battle_against(self, opp, n_battles):
            self.n_won_battles += n_battles
            self.n_finished_battles += n_battles

    wins, fin = asyncio.run(G._play_chunk(QuickPlayer(), object(), 3, timeout=10))
    assert (wins, fin) == (3, 3)


# ── 3c.7b: team discovery under teams/Champions/ (M-A now, M-B as added) ───────
def test_discover_teams_spans_regs_and_skips_non_team(tmp_path):
    import pytest
    pytest.importorskip("poke_env")
    import v_dance.play.run_local_battle as R

    (tmp_path / "M-A").mkdir()
    (tmp_path / "M-B").mkdir()
    (tmp_path / "M-A" / "TeamAlpha").write_text("paste", encoding="utf-8")
    (tmp_path / "M-A" / "TeamBeta").write_text("paste", encoding="utf-8")
    (tmp_path / "M-B" / "TeamGamma").write_text("paste", encoding="utf-8")     # future reg
    (tmp_path / "M-A" / "README.md").write_text("doc", encoding="utf-8")        # skipped
    (tmp_path / "M-A" / "known_teams.json").write_text("{}", encoding="utf-8")  # skipped
    (tmp_path / "M-A" / ".gitkeep").write_text("", encoding="utf-8")            # skipped

    names = sorted(Path(f).name for f in R.discover_teams(root=tmp_path))
    assert names == ["TeamAlpha", "TeamBeta", "TeamGamma"]      # both regs, no doc/json/hidden


def test_discover_teams_real_pool_is_archetype_rich():
    import pytest
    pytest.importorskip("poke_env")
    import v_dance.play.run_local_battle as R

    pool = R.discover_teams()
    assert len(pool) >= 50                                       # the ~70-team M-A set
    names = {Path(p).name for p in pool}
    # sec 12 archetypes the exploration step needs present in the training pool
    assert {"UB_Perish_trap", "Trickery", "Justified_Mega_Gallade"} <= names


def test_resolve_team_path_finds_name_and_repo_relative(tmp_path):
    import pytest
    pytest.importorskip("poke_env")
    import v_dance.play.run_local_battle as R

    by_name = R.resolve_team_path("team1")                       # bare name under Champions
    assert by_name.exists() and by_name.name == "team1"
    rel = R.discover_teams()[0]                                  # repo-relative path
    assert R.resolve_team_path(rel).exists()


def test_resolve_team_path_missing_exits(monkeypatch):
    import pytest
    pytest.importorskip("poke_env")
    import v_dance.play.run_local_battle as R
    with pytest.raises(SystemExit):
        R.resolve_team_path("definitely_not_a_team_xyz")


def test_resolve_train_pool_all_expands_to_full_pool():
    import pytest
    pytest.importorskip("poke_env")
    import v_dance.play.run_local_battle as R
    from v_dance.selfplay.generation import resolve_train_pool

    assert resolve_train_pool(["all"]) == R.discover_teams()
    assert len(resolve_train_pool(["all"])) >= 50


def test_make_opponent_threads_max_concurrent_battles(monkeypatch):
    """3c.8c: the worker count must reach the opponent constructors so poke-env actually
    runs battles in parallel (a min over BOTH players' max_concurrent_battles)."""
    import pytest
    pytest.importorskip("poke_env")
    import v_dance.play.run_local_battle as R
    import v_dance.eval.eval_opponents as EO
    import v_dance.eval.gauntlet as Gmod

    rec = {}
    monkeypatch.setattr(R, "make_player",
                        lambda u, t, **kw: rec.update({"random_mcb": kw.get("max_concurrent_battles")}))
    Gmod._make_opponent("random", "u", "team", max_concurrent_battles=6)
    assert rec["random_mcb"] == 6

    class _FakeScripted:
        def __init__(self, **kw):
            rec["scripted_mcb"] = kw.get("max_concurrent_battles")
    monkeypatch.setattr(EO, "MaxDamageVGCPlayer", _FakeScripted)
    Gmod._make_opponent("max_damage", "u", "team", max_concurrent_battles=5)
    assert rec["scripted_mcb"] == 5


def test_default_eval_teams_all_resolve_exact_case():
    """Every curated eval team must resolve to a real file with EXACT casing — guards
    against typos like 'Perish_Rain' (real file is 'Perish_rain') silently relying on
    Windows case-insensitivity and then breaking the gate on a case-sensitive FS."""
    import pytest
    pytest.importorskip("poke_env")
    import v_dance.play.run_local_battle as R
    from v_dance.selfplay.generation import DEFAULT_EVAL_TEAMS

    for name in DEFAULT_EVAL_TEAMS:
        p = R.resolve_team_path(name)
        assert p.is_file() and p.name == name, f"{name!r} resolved to {p.name!r}"
