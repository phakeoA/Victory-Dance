"""W2 / B1 (2026-09-03) — the OWN-TEAM asymmetric pairing and the promoted-arm registration.

The run design (docs/w2_era5b_run_design_2026-09-03.md) specialises the OWN seat on one team while
the opponent seat stays general. These tests pin the contract OFFLINE (no server, no torch):

  * ``own_team_matchups``  — own always on the model seat, exact sum, mirror share, weighted
                             opponents with the usage floor, seed-reproducible.
  * ``observed_team_weights`` — mean ladder usage of a team's species, floored; mega -> base.
  * ``collection_pairings`` / ``eval_pairings`` — the two planners the collect/eval paths call,
                             INCLUDING the "off" path (must equal the legacy ``team_matchups``).
  * the three plumbing switches — ``generation.build_collection_chunks`` (asyncio),
                             ``mp_collect.build_chunk_specs`` (mp), ``mp_eval.build_eval_specs``.
  * ``ladder_update.register_arm`` — what ``--register-arms`` writes for a promoted generation.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from v_dance.eval import gauntlet as GA

_REPO_ROOT = Path(__file__).resolve().parents[1]


POOL = ["teams/Champions/M-B/The_Big_6_v2", "teams/Champions/M-B/pyroar",
        "teams/Champions/M-B/grimsnarl", "teams/Champions/M-B/mal_speed"]
OWN = "The_Big_6_v2"
OWN_ENTRY = POOL[0]


class _FakeLeague:
    """Canned sample() results in order (round-robin) — mirrors tests/test_mp_collect.py."""

    def __init__(self, samples):
        self._samples = list(samples)
        self._i = 0

    def sample(self, rng):
        s = self._samples[self._i % len(self._samples)]
        self._i += 1
        return s


# ── own_team_matchups ─────────────────────────────────────────────────────────
def test_own_matchups_sum_and_own_seat():
    """Every chunk puts the OWN team on the model seat and the counts sum to n exactly."""
    out = GA.own_team_matchups(OWN, POOL, 100, mirror_frac=0.2, seed=0)
    assert sum(n for _a, _b, n in out) == 100
    assert all(a == OWN_ENTRY for a, _b, _n in out)          # canonicalised to the pool entry
    assert all(n > 0 for _a, _b, n in out)


def test_own_matchups_mirror_share():
    """round(mirror_frac * n) games are the own-vs-own mirror; the rest never face the own team."""
    out = GA.own_team_matchups(OWN, POOL, 100, mirror_frac=0.2, seed=0)
    mirror = sum(n for a, b, n in out if a == b)
    assert mirror == 20
    assert all(GA.team_key(b) != GA.team_key(OWN) for _a, b, _n in out if _a != b)


@pytest.mark.parametrize("frac,expect", [(0.0, 0), (0.5, 50), (1.0, 100)])
def test_own_matchups_mirror_fraction_edges(frac, expect):
    out = GA.own_team_matchups(OWN, POOL, 100, mirror_frac=frac, seed=0)
    assert sum(n for a, b, n in out if a == b) == expect
    assert sum(n for _a, _b, n in out) == 100


def test_own_matchups_weights_respected():
    """A heavier opponent gets proportionally more games (weights keyed by pool entry)."""
    w = {POOL[1]: 30.0, POOL[2]: 10.0, POOL[3]: 10.0}
    out = GA.own_team_matchups(OWN, POOL, 100, mirror_frac=0.2, weights=w, seed=0)
    got = {b: n for a, b, n in out if a != b}
    assert got[POOL[1]] == 48                                 # 80 * 30/50
    assert got[POOL[2]] == got[POOL[3]] == 16                 # 80 * 10/50
    assert sum(got.values()) == 80


def test_own_matchups_zero_weights_fall_back_to_uniform():
    w = {t: 0.0 for t in POOL}
    out = GA.own_team_matchups(OWN, POOL, 90, mirror_frac=0.0, weights=w, seed=1)
    counts = sorted(n for _a, _b, n in out)
    assert sum(counts) == 90 and counts == [30, 30, 30]


def test_own_matchups_seed_reproducible_and_seed_sensitive():
    a = GA.own_team_matchups(OWN, POOL, 37, mirror_frac=0.2, seed=7)
    b = GA.own_team_matchups(OWN, POOL, 37, mirror_frac=0.2, seed=7)
    assert a == b                                             # same seed => identical plan
    assert sum(n for _x, _y, n in GA.own_team_matchups(OWN, POOL, 37, seed=8)) == 37


def test_own_matchups_degenerate_pools():
    assert GA.own_team_matchups(OWN, POOL, 0) == []
    assert GA.own_team_matchups(OWN, [OWN_ENTRY], 12) == [(OWN_ENTRY, OWN_ENTRY, 12)]
    assert GA.own_team_matchups(OWN, [], 12) == [(OWN, OWN, 12)]


def test_own_matchups_name_vs_path_canonicalisation():
    """The CLI passes a bare NAME; the pool holds repo-relative paths — they must unify."""
    assert GA.canonical_own_team(OWN, POOL) == OWN_ENTRY
    assert GA.canonical_own_team("not_in_pool", POOL) == "not_in_pool"
    out = GA.own_team_matchups(OWN_ENTRY, POOL, 20, mirror_frac=0.2, seed=0)
    assert all(a == OWN_ENTRY for a, _b, _n in out)
    # the own team is never ALSO an opponent (dedup by file name, not by string)
    assert not [1 for a, b, _n in out if a != b and GA.team_key(b) == GA.team_key(OWN)]


def test_own_matchups_tiny_n_still_sums():
    """Fewer games than opponents: the plan still sums exactly (no silent drop)."""
    for n in (1, 2, 3, 5):
        out = GA.own_team_matchups(OWN, POOL, n, mirror_frac=0.2, seed=3)
        assert sum(c for _a, _b, c in out) == n


# ── observed_team_weights ─────────────────────────────────────────────────────
def _write_team(tmp_path, name, species):
    p = tmp_path / name
    blocks = []
    for sp in species:
        blocks.append(f"{sp} @ Life Orb\nAbility: Levitate\nLevel: 50\n- Protect")
    p.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    return str(p)


def test_observed_weights_mean_usage_and_floor(tmp_path):
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"pokemon": {"Garchomp": {"usage_pct": 22.0},
                                            "Kingambit": {"usage_pct": 18.0},
                                            "Sableye": {"usage_pct": 0.5}}}), encoding="utf-8")
    hot = _write_team(tmp_path, "hot", ["Garchomp", "Kingambit"])      # mean 20.0
    cold = _write_team(tmp_path, "cold", ["Sableye", "Sableye"])       # mean 0.5 -> floor
    w = GA.observed_team_weights([hot, cold], meta)
    assert w[hot] == pytest.approx(20.0)
    assert w[cold] == pytest.approx(GA.OBSERVED_FLOOR_PCT)             # never below the floor


def test_observed_weights_mega_counts_as_base_species(tmp_path):
    """A mega paste (Charizard-Mega-Y) must score under the BASE species the meta file keys."""
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"pokemon": {"Charizard": {"usage_pct": 40.0}}}), encoding="utf-8")
    t = _write_team(tmp_path, "zard", ["Charizard-Mega-Y", "Charizard-Mega-Y"])
    assert GA.observed_team_weights([t], meta)[t] == pytest.approx(40.0)


def test_observed_weights_missing_meta_is_all_floor(tmp_path):
    t = _write_team(tmp_path, "t", ["Garchomp"])
    w = GA.observed_team_weights([t], tmp_path / "does_not_exist.json")
    assert w[t] == pytest.approx(GA.OBSERVED_FLOOR_PCT)


def test_observed_weights_unreadable_team_is_floor(tmp_path):
    """A pool entry with no resolvable paste weighs the floor rather than raising."""
    w = GA.observed_team_weights(["no_such_team_anywhere_xyz"], tmp_path / "nope.json")
    assert w["no_such_team_anywhere_xyz"] == pytest.approx(GA.OBSERVED_FLOOR_PCT)


def test_observed_weights_on_the_real_pool_prefers_the_meta_teams():
    """Smoke against the REAL repo data: every weight >= the floor and the file parses."""
    w = GA.observed_team_weights([OWN_ENTRY, "teams/Champions/M-B/wyrdeer_flo"])
    assert set(w) == {OWN_ENTRY, "teams/Champions/M-B/wyrdeer_flo"}
    assert all(v >= GA.OBSERVED_FLOOR_PCT for v in w.values())


# ── the two planners ──────────────────────────────────────────────────────────
def test_collection_pairings_off_equals_legacy():
    """own_team=None must be BYTE-identical to the symmetric planner (no behaviour change)."""
    assert GA.collection_pairings(POOL, 60, seed=4) == GA.team_matchups(POOL, 60, seed=4)


def test_collection_pairings_on_uses_own_seat():
    out = GA.collection_pairings(POOL, 60, seed=4, own_team=OWN, own_mirror_frac=0.2)
    assert out == GA.own_team_matchups(OWN, POOL, 60, mirror_frac=0.2, weights=None, seed=4)
    assert all(a == OWN_ENTRY for a, _b, _n in out)


def test_eval_pairings_off_equals_legacy():
    for kind in ("random", "prev_best"):
        assert GA.eval_pairings(kind, POOL, 24, seed=2) == GA.team_matchups(POOL, 24, seed=2)


def test_eval_pairings_scripted_is_own_vs_pool_no_mirror():
    """Scripted anchors judge the served team against the eval pool — no own-vs-own games."""
    out = GA.eval_pairings("random", POOL, 60, seed=2, own_team=OWN)
    assert sum(n for _a, _b, n in out) == 60
    assert all(a == OWN_ENTRY for a, _b, _n in out)
    assert not [1 for a, b, _n in out if a == b]              # mirror_frac 0 for scripted kinds


def test_eval_pairings_prev_best_is_own_vs_own():
    """The champion mirror (and the HoF suspects that reuse it) = both seats on the own team."""
    assert GA.eval_pairings("prev_best", POOL, 360, seed=2, own_team=OWN) == \
        [(OWN_ENTRY, OWN_ENTRY, 360)]
    assert GA.eval_pairings("prev_best", POOL, 0, seed=2, own_team=OWN) == []


# ── plumbing switch 1: the asyncio collection planner ─────────────────────────
def test_build_collection_chunks_own_first():
    from v_dance.selfplay.generation import build_collection_chunks
    league = _FakeLeague([("latest", "x.pt")])
    chunks = build_collection_chunks(league, POOL, 40, chunk_size=10, matchup_seed=0, seed=0,
                                     own_team=OWN, own_mirror_frac=0.25)
    assert sum(c["cn"] for c in chunks) == 40
    assert all(c["team_a"] == OWN_ENTRY for c in chunks)
    assert sum(c["cn"] for c in chunks if c["team_b"] == OWN_ENTRY) == 10      # 25% mirror
    assert len({c["uid"] for c in chunks}) == len(chunks)                      # uids stay unique


def test_build_collection_chunks_default_unchanged():
    from v_dance.selfplay.generation import build_collection_chunks
    league = _FakeLeague([("latest", "x.pt")])
    a = build_collection_chunks(league, POOL, 24, chunk_size=6, matchup_seed=1, seed=2)
    league2 = _FakeLeague([("latest", "x.pt")])
    b = build_collection_chunks(league2, POOL, 24, chunk_size=6, matchup_seed=1, seed=2,
                                own_team=None)
    assert a == b and sum(c["cn"] for c in a) == 24


# ── plumbing switch 2: the multiprocess collection planner ────────────────────
def test_build_chunk_specs_own_first_and_weighted():
    from v_dance.selfplay import mp_collect as MP
    league = _FakeLeague([("latest", None)])
    w = {POOL[1]: 30.0, POOL[2]: 10.0, POOL[3]: 10.0}
    specs = MP.build_chunk_specs(league, POOL, 100, chunk_size=100, matchup_seed=0, seed=0,
                                 own_team=OWN, own_mirror_frac=0.2, opp_weights=w)
    assert sum(s.n for s in specs) == 100
    assert all(s.team_a == OWN_ENTRY for s in specs)
    by_opp = {s.team_b: s.n for s in specs}
    assert by_opp[OWN_ENTRY] == 20 and by_opp[POOL[1]] == 48
    assert len({s.uid for s in specs}) == len(specs)


def test_build_chunk_specs_default_unchanged():
    from v_dance.selfplay import mp_collect as MP
    a = MP.build_chunk_specs(_FakeLeague(["latest"]), POOL, 24, chunk_size=6, matchup_seed=1, seed=2)
    b = MP.build_chunk_specs(_FakeLeague(["latest"]), POOL, 24, chunk_size=6, matchup_seed=1, seed=2,
                             own_team=None)
    assert [(s.team_a, s.team_b, s.n) for s in a] == [(s.team_a, s.team_b, s.n) for s in b]
    assert sum(s.n for s in a) == 24


# ── plumbing switch 3: the multiprocess eval planner ──────────────────────────
def test_build_eval_specs_own_seat_and_mirror():
    from v_dance.selfplay import mp_eval as ME
    specs = ME.build_eval_specs(["random", "prev_best"], POOL, 30, matchup_seed=0,
                                mirror_battles=40, own_team=OWN)
    scripted = [s for s in specs if s.kind == "random"]
    mirror = [s for s in specs if s.kind == "prev_best"]
    assert sum(s.n for s in scripted) == 30 and sum(s.n for s in mirror) == 40
    assert all(s.team_a == OWN_ENTRY for s in specs)
    assert not [1 for s in scripted if s.team_a == s.team_b]      # scripted: own vs the pool
    assert all(s.team_a == s.team_b for s in mirror)              # champion mirror: own vs own
    assert len({s.uid for s in specs}) == len(specs)


def test_build_eval_specs_subdivides_the_own_mirror():
    """The own-vs-own mirror is ONE pairing: without a split a 360-game mirror would run in a
    single worker while the rest idle. n_ways fills the slots and the total is preserved."""
    from v_dance.selfplay import mp_eval as ME
    specs = ME.build_eval_specs(["prev_best"], POOL, 12, matchup_seed=0, mirror_battles=360,
                                own_team=OWN, n_ways=8)
    assert len(specs) == 8 and sum(s.n for s in specs) == 360
    assert all(s.team_a == s.team_b == OWN_ENTRY for s in specs)
    assert len({s.uid for s in specs}) == 8               # distinct uids => distinct accounts
    one = ME.build_eval_specs(["prev_best"], POOL, 12, matchup_seed=0, mirror_battles=360,
                              own_team=OWN, n_ways=1)
    assert len(one) == 1 and one[0].n == 360              # n_ways=1 keeps the single chunk


def test_subdivide_pairings_preserves_totals_and_pairs():
    pairs = [(OWN_ENTRY, OWN_ENTRY, 360)]
    out = GA.subdivide_pairings(pairs, 7)
    assert len(out) == 7 and sum(n for _a, _b, n in out) == 360
    assert all(a == b == OWN_ENTRY for a, b, _n in out)
    assert GA.subdivide_pairings(pairs, 1) == pairs        # no-op at 1 way
    assert sum(n for _a, _b, n in GA.subdivide_pairings(pairs, 1000)) == 360   # capped by games


def test_game_runner_subdivide_still_matches_the_shared_helper():
    """game_runner._subdivide_matchups now delegates - task #13 behaviour must be identical."""
    from v_dance.selfplay.game_runner import _subdivide_matchups
    raw = [("team1", "team1", 2000)]
    assert _subdivide_matchups(raw, 10) == [("team1", "team1", 200)] * 10
    assert _subdivide_matchups(raw, 1) == raw
    assert _subdivide_matchups(raw, 10) == GA.subdivide_pairings(raw, 10)


def test_build_eval_specs_default_unchanged():
    from v_dance.selfplay import mp_eval as ME
    a = ME.build_eval_specs(["random"], POOL, 12, matchup_seed=3)
    b = ME.build_eval_specs(["random"], POOL, 12, matchup_seed=3, own_team=None)
    assert [(s.team_a, s.team_b, s.n) for s in a] == [(s.team_a, s.team_b, s.n) for s in b]


# ── the --register-arms post-step ─────────────────────────────────────────────
def _bandit_cfg(tmp_path):
    p = tmp_path / "serve_bandit.json"
    p.write_text(json.dumps({"arms": [{"name": "era2", "battle_ckpt": "a.pt", "incumbent": True}]},
                            indent=2), encoding="utf-8")
    return p


def test_register_arm_writes_an_argmax_arm(tmp_path):
    """What --register-arms writes for a promoted generation: tau 0, adapt-rules OFF, top-p 1."""
    from v_dance.selfplay.ladder_update import register_arm
    cfg = _bandit_cfg(tmp_path)
    ckpt = tmp_path / "gen7.pt"
    ckpt.write_bytes(b"x")
    entry = register_arm(cfg, name="era5b_g7", battle_ckpt=ckpt, tau=0.0, note="W2 promoted gen 7")
    assert entry["name"] == "era5b_g7" and entry["tau"] == 0.0
    assert entry["adapt_rules"] is False and entry["top_p"] == 1.0
    assert entry["tp_ckpt"] == "default"
    saved = json.loads(cfg.read_text(encoding="utf-8"))["arms"]
    assert [a["name"] for a in saved] == ["era2", "era5b_g7"]
    assert saved[0]["incumbent"] is True                       # the incumbent is untouched


def test_register_arm_refuses_a_duplicate(tmp_path):
    from v_dance.selfplay.ladder_update import register_arm
    cfg = _bandit_cfg(tmp_path)
    ckpt = tmp_path / "gen7.pt"
    ckpt.write_bytes(b"x")
    register_arm(cfg, name="era5b_g7", battle_ckpt=ckpt, tau=0.0, note="first")
    with pytest.raises(ValueError):
        register_arm(cfg, name="era5b_g7", battle_ckpt=ckpt, tau=0.0, note="second")
    assert len(json.loads(cfg.read_text(encoding="utf-8"))["arms"]) == 2


# ── the run wiring accepts the knobs (signature contract) ─────────────────────
def test_run_live_generations_accepts_the_w2_knobs():
    import inspect
    from v_dance.selfplay.generation import run_live_generations
    params = inspect.signature(run_live_generations).parameters
    for k in ("own_team", "own_mirror_frac", "opp_weights", "register_arms", "register_prefix",
              "bandit_config"):
        assert k in params, k
    assert params["own_mirror_frac"].default == 0.2
    assert params["register_arms"].default is False
    assert params["register_prefix"].default == "era5b_g"
    # default None => the LIVE config/serve_bandit.json; a smoke passes a throwaway copy
    assert params["bandit_config"].default is None


def _run_cli(*argv):
    """generation.py builds its parser inline in main(), so the CLI contract is checked for REAL
    in a subprocess (~2 s). NEVER pass --live here: a parseable --live would start a run."""
    import subprocess
    import sys
    return subprocess.run([sys.executable, "-m", "v_dance.selfplay.generation", *argv],
                          cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=180)


def test_cli_documents_the_w2_flags():
    r = _run_cli("--help")
    assert r.returncode == 0, r.stderr[-2000:]
    for flag in ("--own-team", "--own-mirror", "--opp-weights", "--register-arms",
                 "--register-prefix", "--bandit-config"):
        assert flag in r.stdout, flag


def test_cli_rejects_an_unknown_opp_weights_mode():
    """A typo must be a LOUD parse error, not a silent fall-back to uniform."""
    r = _run_cli("--own-team", OWN, "--opp-weights", "bogus")
    assert r.returncode == 2
    assert "opp-weights" in (r.stderr + r.stdout)
